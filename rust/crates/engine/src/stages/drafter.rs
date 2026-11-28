//! The DFlash2 drafter's stages: the context and the block views.
//!
//! The base model's views tap rows of its hidden state into the sequence's tap buffer ([`DrafterTap`]). A context
//! quantum turns the committed rows of the taps into keys and values of the drafter's rings ([`DrafterContext`]); a
//! block quantum drafts: the anchor and the mask rows through the drafter's layers ([`DrafterBody`]), the target's head
//! over the mask rows (a plain linear of the plan), and the selector, which chooses the block's tokens from the head's
//! top candidates ([`DrafterSelect`]).
//!
//! The drafter's intermediates live in the workspace's slots, carved by [`Carve`], which refuses a region the slot
//! does not hold. Nothing here appends to a cache of the sequence: the drafter's state is its rings.

use std::sync::Arc;

use euhedral_graph::{Marker, Stage, StageCx, StageError, Submitted};
use euhedral_inference_cuda::launch::dflash::topk_scratch_bytes;
use euhedral_inference_cuda::{DevicePtr, KernelOps};
use euhedral_inference_edrl::{TensorDataType, WeightFormat, WeightLayout};

use super::sampling::KEY_BYTES;
use super::{Launch, Quantum, eager, fatal, launch_u32};
use crate::model::dflash2::{Dflash2Config, SELECTOR_TOP_K};
use crate::model::{DrafterRole, LayerIndex, Residence, WeightRef};
use crate::plan::PlanBuildError;
use crate::state::attention::{DRAFT_TOKEN_SLOTS, DrafterBuffers};
use crate::workspace::BufferSlot;

/// Bytes the regions of a slot start on.
const ALIGNMENT: u64 = 256;

/// A drafter projection: a weight of the model's shape `[out, in]`, stored as NVFP4 or BF16.
#[derive(Clone, Debug)]
pub struct DrafterProjection {
    weight: WeightRef,
    in_features: u32,
    out_features: u32,
}

impl DrafterProjection {
    /// The projection from `in_features` to `out_features` over `weight`. Refuses a weight of another shape or format,
    /// or one not held on the device.
    pub fn new(weight: WeightRef, out_features: usize, in_features: usize) -> Result<Self, PlanBuildError> {
        Self::check(&weight, &[out_features as u64, in_features as u64])?;
        match (weight.format, weight.layout) {
            (WeightFormat::Nvfp4, WeightLayout::RowSplitK128V1)
            | (WeightFormat::Bf16, WeightLayout::ContiguousLeV1) => {}
            (format, layout) => {
                return Err(PlanBuildError::Format {
                    name: weight.name,
                    format,
                    layout,
                });
            }
        }
        let narrow = |value: usize| {
            u32::try_from(value)
                .map_err(|_| PlanBuildError::Geometry(format!("{}: {value} exceeds a launch", weight.name)))
        };
        Ok(Self {
            in_features: narrow(in_features)?,
            out_features: narrow(out_features)?,
            weight,
        })
    }

    /// Refuses `weight` unless it is a BF16-element object of `shape` held on the device.
    fn check(weight: &WeightRef, shape: &[u64]) -> Result<(), PlanBuildError> {
        if weight.shape != shape {
            return Err(PlanBuildError::Geometry(format!(
                "drafter weight {} is {:?}, the drafter needs {shape:?}",
                weight.name, weight.shape
            )));
        }
        if weight.dtype != TensorDataType::Bf16 {
            return Err(PlanBuildError::Geometry(format!(
                "drafter weight {} holds {:?} elements, the drafter needs BF16",
                weight.name, weight.dtype
            )));
        }
        if weight.residence != Residence::Device {
            return Err(PlanBuildError::Geometry(format!(
                "drafter weight {} is held in {:?}, the drafter reads its weights on the device",
                weight.name, weight.residence
            )));
        }
        Ok(())
    }

    /// `output = input x weight^T` over `rows` rows.
    fn apply(
        &self,
        kernels: &mut dyn KernelOps,
        input: DevicePtr,
        output: DevicePtr,
        rows: u32,
    ) -> Result<(), StageError> {
        match self.weight.format {
            WeightFormat::Nvfp4 => kernels.linear_nvfp4_bf16(
                input,
                self.weight.ptr,
                output,
                rows,
                self.in_features,
                self.out_features,
                self.weight.bytes,
            ),
            _ => kernels.dflash_linear_bf16(
                input,
                self.weight.ptr,
                output,
                rows,
                self.in_features,
                self.out_features,
            ),
        }
        .map_err(fatal)
    }
}

/// A BF16 vector or table of a known shape, held on the device.
fn bf16_object(weight: &WeightRef, shape: &[usize]) -> Result<DevicePtr, PlanBuildError> {
    let shape: Vec<u64> = shape.iter().map(|extent| *extent as u64).collect();
    DrafterProjection::check(weight, &shape)?;
    if weight.format != WeightFormat::Bf16 {
        return Err(PlanBuildError::Format {
            name: weight.name.clone(),
            format: weight.format,
            layout: weight.layout,
        });
    }
    Ok(weight.ptr)
}

/// A dynamic convolution's weights: the base filters and the projection of the rows to the filters' dynamic part.
#[derive(Clone, Debug)]
struct DrafterConvolution {
    base: DevicePtr,
    projection: DrafterProjection,
}

/// One drafter layer's weights.
#[derive(Clone, Debug)]
struct DrafterLayer {
    input_norm: DevicePtr,
    post_attention_norm: DevicePtr,
    query: DrafterProjection,
    key_value: DrafterProjection,
    output: DrafterProjection,
    query_norm: DevicePtr,
    key_norm: DevicePtr,
    gate_up: DrafterProjection,
    down: DrafterProjection,
    attention_convolution: DrafterConvolution,
    mlp_convolution: DrafterConvolution,
}

/// The drafter's weights, checked against its configuration.
#[derive(Clone, Debug)]
pub struct DrafterWeights {
    config: Dflash2Config,
    fc: DrafterProjection,
    hidden_norm: DevicePtr,
    final_norm: DevicePtr,
    selector_hidden: DevicePtr,
    predecessor: DevicePtr,
    successor: DevicePtr,
    layers: Vec<DrafterLayer>,
}

impl DrafterWeights {
    /// The drafter's weights, each found through `weight` by layer and role, and checked against `config`.
    pub fn new(
        config: &Dflash2Config,
        mut weight: impl FnMut(LayerIndex, DrafterRole) -> Result<WeightRef, PlanBuildError>,
    ) -> Result<Self, PlanBuildError> {
        let global = LayerIndex::GLOBAL;
        let hidden = config.hidden;
        let mut get = |layer: LayerIndex, role: DrafterRole| weight(layer, role);
        let fc = DrafterProjection::new(get(global, DrafterRole::Fc)?, hidden, config.tap_width())?;
        let hidden_norm = bf16_object(&get(global, DrafterRole::HiddenNorm)?, &[hidden])?;
        let final_norm = bf16_object(&get(global, DrafterRole::FinalNorm)?, &[hidden])?;
        let selector_hidden = bf16_object(
            &get(global, DrafterRole::SelectorHidden)?,
            &[config.selector_rank, hidden],
        )?;
        let codebook = [config.vocab, config.selector_rank];
        let predecessor = bf16_object(&get(global, DrafterRole::SelectorPredecessor)?, &codebook)?;
        let successor = bf16_object(&get(global, DrafterRole::SelectorSuccessor)?, &codebook)?;
        let mut layers = Vec::with_capacity(config.layers);
        for index in 0..config.layers {
            let layer = LayerIndex(
                u16::try_from(index).map_err(|_| PlanBuildError::Geometry("more than 65535 drafter layers".into()))?,
            );
            let mut convolution = |base: DrafterRole, projection: DrafterRole| {
                Ok::<_, PlanBuildError>(DrafterConvolution {
                    base: bf16_object(&get(layer, base)?, &[2, config.conv_kernel, hidden])?,
                    projection: DrafterProjection::new(
                        get(layer, projection)?,
                        config.conv_projection_width(),
                        hidden,
                    )?,
                })
            };
            let attention_convolution =
                convolution(DrafterRole::AttentionConvBase, DrafterRole::AttentionConvProjection)?;
            let mlp_convolution = convolution(DrafterRole::MlpConvBase, DrafterRole::MlpConvProjection)?;
            layers.push(DrafterLayer {
                input_norm: bf16_object(&get(layer, DrafterRole::InputNorm)?, &[hidden])?,
                post_attention_norm: bf16_object(&get(layer, DrafterRole::PostAttentionNorm)?, &[hidden])?,
                query: DrafterProjection::new(get(layer, DrafterRole::Query)?, config.query_width(), hidden)?,
                key_value: DrafterProjection::new(get(layer, DrafterRole::KeyValue)?, 2 * config.kv_width(), hidden)?,
                output: DrafterProjection::new(get(layer, DrafterRole::Output)?, hidden, config.query_width())?,
                query_norm: bf16_object(&get(layer, DrafterRole::QueryNorm)?, &[config.head_dim])?,
                key_norm: bf16_object(&get(layer, DrafterRole::KeyNorm)?, &[config.head_dim])?,
                gate_up: DrafterProjection::new(get(layer, DrafterRole::GateUp)?, 2 * config.intermediate, hidden)?,
                down: DrafterProjection::new(get(layer, DrafterRole::Down)?, hidden, config.intermediate)?,
                attention_convolution,
                mlp_convolution,
            });
        }
        Ok(Self {
            config: config.clone(),
            fc,
            hidden_norm,
            final_norm,
            selector_hidden,
            predecessor,
            successor,
            layers,
        })
    }

    /// The configuration the weights were checked against.
    pub fn config(&self) -> &Dflash2Config {
        &self.config
    }
}

/// Regions carved one after another out of a workspace slot, each refused when it would pass the slot's end.
struct Carve {
    slot: BufferSlot,
    base: DevicePtr,
    capacity: u64,
    used: u64,
}

impl Carve {
    fn of(quantum: &Quantum, slot: BufferSlot) -> Self {
        Self {
            slot,
            base: quantum.slot(slot),
            capacity: quantum.workspace.buffer(slot).bytes() as u64,
            used: 0,
        }
    }

    /// The next region of `bytes` bytes, aligned.
    fn take(&mut self, bytes: u64, what: &str) -> Result<DevicePtr, StageError> {
        let start = self.used.next_multiple_of(ALIGNMENT);
        let end = start + bytes;
        if end > self.capacity {
            return Err(fatal(format!(
                "the drafter's {what} needs {end} bytes of the {:?} slot, which holds {}",
                self.slot, self.capacity
            )));
        }
        self.used = end;
        Ok(self.base.offset(start))
    }
}

/// A region of `rows` rows of `width` BF16 values.
fn rows_bytes(rows: usize, width: usize) -> u64 {
    (rows * width * 2) as u64
}

/// The drafter's configuration, as the quantum's model carries it.
fn drafter_of(quantum: &Quantum) -> Result<&Dflash2Config, StageError> {
    quantum
        .config
        .dflash2
        .as_ref()
        .ok_or_else(|| fatal("the model has no DFlash2 drafter"))
}

/// The sequence's drafter buffers.
fn buffers_of(quantum: &Quantum) -> Result<&DrafterBuffers, StageError> {
    quantum
        .segment()
        .refs
        .device()
        .scratch
        .drafter()
        .ok_or_else(|| fatal("the sequence has no DFlash2 drafter buffers"))
}

/// Copies the rows of the base model's hidden state after tapped layer `tap` into the sequence's tap buffer, at the
/// tap's columns (it copies after the next layer's input norm).
#[derive(Clone, Debug)]
pub struct DrafterTap {
    tap: usize,
    source: BufferSlot,
}

impl DrafterTap {
    /// The copy of `source`'s rows into the columns of tap number `tap`.
    pub fn new(tap: usize, source: BufferSlot) -> Self {
        Self { tap, source }
    }
}

impl Launch for DrafterTap {
    fn launch(&self, quantum: &Quantum, kernels: &mut dyn KernelOps) -> Result<(), StageError> {
        let drafter = drafter_of(quantum)?;
        if self.tap >= drafter.taps.len() {
            return Err(fatal(format!(
                "tap {} is out of the drafter's {} taps",
                self.tap,
                drafter.taps.len()
            )));
        }
        let row_bytes = quantum.config.hidden_size as u64 * 2;
        let pitch = drafter.tap_width() as u64 * 2;
        for segment in quantum.segments() {
            let buffers = segment
                .refs
                .device()
                .scratch
                .drafter()
                .ok_or_else(|| fatal("the sequence has no DFlash2 drafter buffers"))?;
            if segment.rows as u64 > buffers.tap_capacity() {
                return Err(fatal(format!(
                    "{} rows exceed the {} the tap buffer holds",
                    segment.rows,
                    buffers.tap_capacity()
                )));
            }
            kernels
                .copy_device_to_device_2d(
                    buffers.taps().offset(self.tap as u64 * row_bytes),
                    pitch,
                    quantum.slot(self.source).offset(segment.row_offset as u64 * row_bytes),
                    row_bytes,
                    row_bytes,
                    segment.rows as u64,
                )
                .map_err(fatal)?;
        }
        Ok(())
    }
}

/// The context update: the tap rows of the committed rows through the drafter's input projection and norm, and from
/// them the keys and values of every drafter layer into its rings, at the slots of the rows' positions.
#[derive(Clone, Debug)]
pub struct DrafterContext {
    weights: Arc<DrafterWeights>,
}

impl DrafterContext {
    pub fn new(weights: Arc<DrafterWeights>) -> Self {
        Self { weights }
    }
}

impl Launch for DrafterContext {
    /// The upload of the quantum's input record, which holds the position the ring slots follow from.
    fn waits(&self, quantum: &Quantum) -> Vec<Marker> {
        quantum.prepared.into_iter().collect()
    }

    fn launch(&self, quantum: &Quantum, kernels: &mut dyn KernelOps) -> Result<(), StageError> {
        let config = drafter_of(quantum)?;
        if *config != self.weights.config {
            return Err(fatal(
                "the drafter's weights were checked against another configuration",
            ));
        }
        let buffers = buffers_of(quantum)?;
        let rows = quantum.rows;
        if rows > config.sliding_window {
            return Err(fatal(format!(
                "a context update of {rows} rows exceeds the drafter's window of {}",
                config.sliding_window
            )));
        }
        if rows as u64 > buffers.tap_capacity() {
            return Err(fatal(format!(
                "a context update of {rows} rows exceeds the {} tap rows",
                buffers.tap_capacity()
            )));
        }
        let launch_rows = quantum.rows_u32()?;
        let mut fused_region = Carve::of(quantum, BufferSlot::HiddenState);
        let mut context_region = Carve::of(quantum, BufferSlot::InputNormalized);
        let mut kv_region = Carve::of(quantum, BufferSlot::QkProjected);
        let fused = fused_region.take(rows_bytes(rows, config.hidden), "fused context rows")?;
        let context = context_region.take(rows_bytes(rows, config.hidden), "normalized context rows")?;
        let kv = kv_region.take(rows_bytes(rows, 2 * config.kv_width()), "context keys and values")?;
        let hidden = launch_u32(config.hidden, "the drafter's hidden size")?;
        self.weights.fc.apply(kernels, buffers.taps(), fused, launch_rows)?;
        kernels
            .dflash_rms_norm_bf16(
                fused,
                self.weights.hidden_norm,
                context,
                launch_rows,
                hidden,
                config.eps,
            )
            .map_err(fatal)?;
        let position = quantum.position_of(0);
        for (index, layer) in self.weights.layers.iter().enumerate() {
            layer.key_value.apply(kernels, context, kv, launch_rows)?;
            kernels
                .dflash_context_kv_bf16(
                    kv,
                    layer.key_norm,
                    buffers.ring_keys(index),
                    buffers.ring_values(index),
                    launch_rows,
                    position,
                    launch_u32(config.sliding_window, "the drafter's window")?,
                    launch_u32(config.kv_heads, "the drafter's key/value heads")?,
                    launch_u32(config.head_dim, "the drafter's head dimension")?,
                    config.eps,
                    config.theta,
                )
                .map_err(fatal)?;
        }
        Ok(())
    }
}

/// The block's body: the drafter's layers over the anchor and the mask rows, then its final norm, which lands in the
/// rows of the final-normalized slot the target's head reads: row `i` holds the body's row `i + 1`, so the head scores
/// the mask rows only.
#[derive(Clone, Debug)]
pub struct DrafterBody {
    weights: Arc<DrafterWeights>,
}

impl DrafterBody {
    pub fn new(weights: Arc<DrafterWeights>) -> Self {
        Self { weights }
    }
}

impl Launch for DrafterBody {
    fn launch(&self, quantum: &Quantum, kernels: &mut dyn KernelOps) -> Result<(), StageError> {
        let config = drafter_of(quantum)?;
        if *config != self.weights.config {
            return Err(fatal(
                "the drafter's weights were checked against another configuration",
            ));
        }
        let buffers = buffers_of(quantum)?;
        let rows = quantum.rows;
        if rows != config.block_size {
            return Err(fatal(format!(
                "a block of {rows} rows, the drafter's block is {} rows",
                config.block_size
            )));
        }
        let launch_rows = quantum.rows_u32()?;
        let hidden_width = config.hidden;
        let hidden = launch_u32(hidden_width, "the drafter's hidden size")?;
        let intermediate = launch_u32(config.intermediate, "the drafter's intermediate size")?;
        let (heads, kv_heads, head_dim) = (
            launch_u32(config.heads, "the drafter's heads")?,
            launch_u32(config.kv_heads, "the drafter's key/value heads")?,
            launch_u32(config.head_dim, "the drafter's head dimension")?,
        );
        let window = launch_u32(config.sliding_window, "the drafter's window")?;
        let take =
            |slot: BufferSlot, width: usize, what: &str| Carve::of(quantum, slot).take(rows_bytes(rows, width), what);
        let residual = take(BufferSlot::HiddenState, hidden_width, "residual rows")?;
        let normed = take(BufferSlot::InputNormalized, hidden_width, "normalized rows")?;
        let dynamic = take(
            BufferSlot::MixerHidden,
            config.conv_projection_width(),
            "dynamic filters",
        )?;
        let convolved = take(BufferSlot::PostMixerNormalized, hidden_width, "convolved rows")?;
        let query = take(BufferSlot::QkProjected, config.query_width(), "queries")?;
        let kv = take(BufferSlot::ValueZProjected, 2 * config.kv_width(), "keys and values")?;
        let mut rope = Carve::of(quantum, BufferSlot::AttentionQkNormalized);
        let query_rope = rope.take(rows_bytes(rows, config.query_width()), "rotated queries")?;
        let key_rope = rope.take(rows_bytes(rows, config.kv_width()), "rotated keys")?;
        let attended = take(BufferSlot::AttentionContext, config.query_width(), "attended rows")?;
        let mixer = take(BufferSlot::MixerDelta, hidden_width, "mixer rows")?;
        let delta = take(BufferSlot::FfnDelta, hidden_width, "convolved deltas")?;
        let gate_up = take(BufferSlot::GateUp, 2 * config.intermediate, "gate and up rows")?;
        let activated = take(BufferSlot::Swiglu, config.intermediate, "activated rows")?;
        // The attention's partial results: per row, 8 key splits of the heads' 128 outputs, maximum and sum. The
        // slot belongs to the head, which writes it after this stage.
        let partial = Carve::of(quantum, BufferSlot::Logits)
            .take((rows * config.heads * 8 * 130 * 4) as u64, "attention partials")?;
        let final_normed = take(BufferSlot::FinalNormalized, hidden_width, "final rows")?;
        let position = quantum.position_of(0);
        let (epsilon, theta) = (config.eps, config.theta);
        let (group, taps) = (
            launch_u32(config.conv_group, "the drafter's convolution group")?,
            launch_u32(config.conv_kernel, "the drafter's convolution kernel")?,
        );
        for (index, layer) in self.weights.layers.iter().enumerate() {
            kernels
                .dflash_rms_norm_bf16(residual, layer.input_norm, normed, launch_rows, hidden, epsilon)
                .map_err(fatal)?;
            // The attention, between the two halves of its dynamic convolution.
            let convolution = &layer.attention_convolution;
            convolution.projection.apply(kernels, normed, dynamic, launch_rows)?;
            kernels
                .dflash_conv_bf16(
                    normed,
                    dynamic,
                    convolution.base,
                    convolved,
                    launch_rows,
                    hidden,
                    group,
                    taps,
                    0,
                )
                .map_err(fatal)?;
            layer.query.apply(kernels, convolved, query, launch_rows)?;
            layer.key_value.apply(kernels, convolved, kv, launch_rows)?;
            kernels
                .dflash_block_qk_bf16(
                    query,
                    kv,
                    layer.query_norm,
                    layer.key_norm,
                    query_rope,
                    key_rope,
                    launch_rows,
                    position,
                    heads,
                    kv_heads,
                    head_dim,
                    epsilon,
                    theta,
                )
                .map_err(fatal)?;
            kernels
                .dflash_attention_bf16(
                    query_rope,
                    key_rope,
                    kv,
                    buffers.ring_keys(index),
                    buffers.ring_values(index),
                    attended,
                    partial,
                    launch_rows,
                    position,
                    window,
                    heads,
                    kv_heads,
                    head_dim,
                )
                .map_err(fatal)?;
            layer.output.apply(kernels, attended, mixer, launch_rows)?;
            kernels
                .dflash_conv_bf16(
                    mixer,
                    dynamic,
                    convolution.base,
                    delta,
                    launch_rows,
                    hidden,
                    group,
                    taps,
                    1,
                )
                .map_err(fatal)?;
            kernels
                .residual_add_bf16(residual, delta, residual, launch_rows, hidden)
                .map_err(fatal)?;
            kernels
                .dflash_rms_norm_bf16(
                    residual,
                    layer.post_attention_norm,
                    normed,
                    launch_rows,
                    hidden,
                    epsilon,
                )
                .map_err(fatal)?;
            // The feed-forward, between the two halves of its own.
            let convolution = &layer.mlp_convolution;
            convolution.projection.apply(kernels, normed, dynamic, launch_rows)?;
            kernels
                .dflash_conv_bf16(
                    normed,
                    dynamic,
                    convolution.base,
                    convolved,
                    launch_rows,
                    hidden,
                    group,
                    taps,
                    0,
                )
                .map_err(fatal)?;
            layer.gate_up.apply(kernels, convolved, gate_up, launch_rows)?;
            kernels
                .dflash_swiglu_bf16(gate_up, activated, launch_rows, intermediate)
                .map_err(fatal)?;
            layer.down.apply(kernels, activated, mixer, launch_rows)?;
            kernels
                .dflash_conv_bf16(
                    mixer,
                    dynamic,
                    convolution.base,
                    delta,
                    launch_rows,
                    hidden,
                    group,
                    taps,
                    1,
                )
                .map_err(fatal)?;
            kernels
                .residual_add_bf16(residual, delta, residual, launch_rows, hidden)
                .map_err(fatal)?;
        }
        // The final norm of the mask rows, into the head's first rows: the anchor's row is not scored.
        let row_bytes = rows_bytes(1, hidden_width);
        kernels
            .dflash_rms_norm_bf16(
                residual.offset(row_bytes),
                self.weights.final_norm,
                final_normed,
                launch_rows - 1,
                hidden,
                epsilon,
            )
            .map_err(fatal)
    }
}

/// The selector: the block's tokens from the target head's top candidates of each mask row and the selector's
/// projection of the row's hidden state. The tokens go to the step's tokens (the verification reads them there, after
/// the anchor in slot 0) and to the quantum's selection words (the host reads them when the block retires).
#[derive(Clone, Debug)]
pub struct DrafterSelect {
    weights: Arc<DrafterWeights>,
}

impl DrafterSelect {
    pub fn new(weights: Arc<DrafterWeights>) -> Self {
        Self { weights }
    }
}

impl Launch for DrafterSelect {
    fn launch(&self, quantum: &Quantum, kernels: &mut dyn KernelOps) -> Result<(), StageError> {
        let config = drafter_of(quantum)?;
        if *config != self.weights.config {
            return Err(fatal(
                "the drafter's weights were checked against another configuration",
            ));
        }
        let rows = quantum.rows;
        let picks = rows - 1;
        if quantum.logits_rows != picks {
            return Err(fatal(format!(
                "a block of {rows} rows scores {} rows, not {picks}",
                quantum.logits_rows
            )));
        }
        if config.selector_top_k != SELECTOR_TOP_K {
            return Err(fatal(format!(
                "a selector keeps {} candidates, the kernels keep {SELECTOR_TOP_K}",
                config.selector_top_k
            )));
        }
        if picks + 1 > DRAFT_TOKEN_SLOTS as usize {
            return Err(fatal(format!(
                "{} step tokens exceed the {DRAFT_TOKEN_SLOTS} slots",
                picks + 1
            )));
        }
        let segment = quantum.segment();
        let selection = segment
            .selection
            .as_ref()
            .ok_or_else(|| fatal("a block has no selection words for its tokens"))?;
        if picks * 4 > selection.capacity() * KEY_BYTES {
            return Err(fatal(format!(
                "{picks} tokens exceed the selection's {} bytes",
                selection.capacity() * KEY_BYTES
            )));
        }
        let step_tokens = segment
            .refs
            .device()
            .scratch
            .draft()
            .map(|draft| draft.tokens.ptr())
            .ok_or_else(|| fatal("the sequence has no step tokens"))?;
        let launch_picks = launch_u32(picks, "the block's mask rows")?;
        let candidates = SELECTOR_TOP_K;
        let mut region = Carve::of(quantum, BufferSlot::HiddenState);
        let values = region.take((picks * candidates * 2) as u64, "top values")?;
        let indices = region.take((picks * candidates * 4) as u64, "top indices")?;
        let scratch = region.take(topk_scratch_bytes(launch_picks), "top-k scratch")?;
        let selector = region.take(rows_bytes(picks, config.selector_rank), "selector rows")?;
        let tokens = region.take((picks * 4) as u64, "tokens")?;
        let scores = region.take((picks * candidates * 4) as u64, "scores")?;
        let ids = quantum.slot(BufferSlot::TokenIds);
        kernels
            .dflash_topk_bf16(
                quantum.slot(BufferSlot::Logits),
                launch_picks,
                launch_u32(config.vocab, "the vocabulary")?,
                values,
                indices,
                scratch,
            )
            .map_err(fatal)?;
        kernels
            .dflash_linear_bf16(
                quantum.slot(BufferSlot::FinalNormalized),
                self.weights.selector_hidden,
                selector,
                launch_picks,
                launch_u32(config.hidden, "the drafter's hidden size")?,
                launch_u32(config.selector_rank, "the selector's rank")?,
            )
            .map_err(fatal)?;
        kernels
            .dflash_select_bf16(
                selector,
                values,
                indices,
                self.weights.predecessor,
                self.weights.successor,
                ids,
                launch_picks,
                launch_u32(config.selector_rank, "the selector's rank")?,
                tokens,
                scores,
            )
            .map_err(fatal)?;
        kernels.copy_device_to_device(step_tokens, ids, 4).map_err(fatal)?;
        kernels
            .copy_device_to_device(step_tokens.offset(4), tokens, (picks * 4) as u64)
            .map_err(fatal)?;
        // SAFETY: the selection is held by the quantum (an `Arc`), which retires only after the marker that follows
        // this launch is reached; the host reads the words through atomic loads, and only once that marker has been
        // reached. No other copy writes the range: the block has no argmax.
        unsafe {
            kernels
                .copy_device_to_readback_shared(selection.readback(), 0, tokens, picks * 4)
                .map_err(fatal)
        }
    }
}

eager!(DrafterTap, DrafterContext, DrafterBody, DrafterSelect);

#[cfg(test)]
mod tests {
    use euhedral_inference_cuda::DevicePtr;

    use super::*;

    fn weight(name: &str, shape: &[u64], format: WeightFormat, layout: WeightLayout) -> WeightRef {
        WeightRef {
            name: name.to_string(),
            shape: shape.to_vec(),
            dtype: TensorDataType::Bf16,
            format,
            layout,
            ptr: DevicePtr::NULL,
            bytes: 0,
            residence: Residence::Device,
            trellis: None,
        }
    }

    fn bf16(shape: &[u64]) -> WeightRef {
        weight("w", shape, WeightFormat::Bf16, WeightLayout::ContiguousLeV1)
    }

    #[test]
    fn a_projection_takes_nvfp4_and_bf16_of_its_shape() {
        let nvfp4 = weight("w", &[64, 32], WeightFormat::Nvfp4, WeightLayout::RowSplitK128V1);
        assert!(DrafterProjection::new(nvfp4, 64, 32).is_ok());
        assert!(DrafterProjection::new(bf16(&[64, 32]), 64, 32).is_ok());
    }

    #[test]
    fn a_projection_refuses_a_shape_that_is_not_its_own() {
        let error = DrafterProjection::new(bf16(&[32, 64]), 64, 32).unwrap_err().to_string();
        assert!(error.contains("[32, 64]") && error.contains("[64, 32]"), "{error}");
    }

    #[test]
    fn a_projection_refuses_a_format_the_drafter_does_not_run() {
        let trellis = weight("w", &[64, 32], WeightFormat::Trellis, WeightLayout::TrellisTile16V1);
        assert!(matches!(
            DrafterProjection::new(trellis, 64, 32),
            Err(PlanBuildError::Format { .. })
        ));
        let other_layout = weight("w", &[64, 32], WeightFormat::Nvfp4, WeightLayout::ContiguousLeV1);
        assert!(matches!(
            DrafterProjection::new(other_layout, 64, 32),
            Err(PlanBuildError::Format { .. })
        ));
    }

    #[test]
    fn a_projection_refuses_weights_that_are_not_on_the_device() {
        let held = WeightRef {
            residence: Residence::Host { offset: 0 },
            ..bf16(&[64, 32])
        };
        let error = DrafterProjection::new(held, 64, 32).unwrap_err().to_string();
        assert!(error.contains("device"), "{error}");
    }

    #[test]
    fn a_norm_refuses_a_width_that_is_not_its_own() {
        assert!(bf16_object(&bf16(&[128]), &[128]).is_ok());
        assert!(bf16_object(&bf16(&[64]), &[128]).is_err());
        let quantized = weight("w", &[128], WeightFormat::Nvfp4, WeightLayout::RowSplitK128V1);
        assert!(bf16_object(&quantized, &[128]).is_err());
    }
}
