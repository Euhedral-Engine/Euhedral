//! The layer's forward pass.

use std::ffi::c_void;
use std::fmt::{Debug, Formatter, Result as FmtResult};
use std::sync::Arc;

use euhedral_gpu::cuda::CudaKernel;
use euhedral_gpu::{LaneEnqueue, LaneStream};
use euhedral_inference_cuda::{CudaLibrary, DeviceBuffer, DevicePtr, KernelOps, LaneOps, PinnedPool};
use euhedral_inference_engine::state::kv::{KvCache, KvRetained};
use euhedral_inference_engine::state::released::ReleasedBuffers;
use euhedral_inference_settings::KvFormat;

use super::math::LaneContext;
use super::weights::{LayerWeights, Part};
use super::{ForwardError, Shape};
use crate::checkpoint::objects::Config;

const BF16_BYTES: u64 = 2;
const FP32_BYTES: u64 = 4;

/// The input of a quantized object, which the observer sees before the object is applied.
#[derive(Clone, Copy, Debug)]
pub struct Observation<'a> {
    /// The object's name (`text/layers/N/mlp/down`, ...).
    pub object: &'a str,
    /// The input rows: BF16 `[rows, columns]` on the device.
    pub input: DevicePtr,
    /// The rows (the tokens of every sequence).
    pub rows: u32,
    /// The input's width.
    pub columns: u32,
}

/// What the observer is: called with each quantized linear's input, on the lane's context, before the linear.
pub type Observer<'o, Q> = &'o mut dyn FnMut(&Observation<'_>, &mut LaneContext<'_, Q>) -> Result<(), ForwardError>;

struct Slot {
    convolution: DeviceBuffer,
    recurrent: DeviceBuffer,
    cache: KvCache,
    retained: KvRetained,
}

/// One decoder layer over a batch of whole sequences. Sequences start at position 0 with fresh GDN state; attention
/// keeps its keys and values in the engine's NVFP4 cache.
///
/// The workspace and the per-sequence slots are allocated in stream order on a lane. Free them with
/// [`LayerForward::release`] on a lane once the work is retired.
pub struct LayerForward {
    shape: Shape,
    epsilon: f32,
    theta: f64,
    max_rows: u32,
    normalized: DeviceBuffer,
    first: DeviceBuffer,
    second: DeviceBuffer,
    mixed: DeviceBuffer,
    context: DeviceBuffer,
    gate_up: DeviceBuffer,
    swiglu: DeviceBuffer,
    delta: DeviceBuffer,
    alpha: DeviceBuffer,
    beta: DeviceBuffer,
    position: DeviceBuffer,
    slots: Vec<Slot>,
    library: Arc<CudaLibrary>,
    pool: Arc<PinnedPool>,
    released: Arc<ReleasedBuffers>,
}

impl LayerForward {
    /// A layer for up to `max_rows` rows (all sequences together), with its workspace allocated on `lane`. `pool` is
    /// where the KV cache's table uploads stage their bytes.
    ///
    /// The workspace and the slots are ordered only on the lane that queued them: work on another lane must wait for
    /// this lane's work to retire first, and [`release`](Self::release) takes the same lane. On `Err` what was
    /// allocated is freed in stream order.
    pub fn new(
        lane: &mut dyn LaneOps,
        library: &Arc<CudaLibrary>,
        pool: &Arc<PinnedPool>,
        config: &Config,
        max_rows: u32,
    ) -> Result<LayerForward, ForwardError> {
        if max_rows == 0 {
            return Err(ForwardError::Invalid("a layer for 0 rows".to_owned()));
        }
        let shape = Shape::new(config)?;
        let rows = u64::from(max_rows);
        let widest = shape
            .vz_width
            .max(shape.attention_q + shape.attention_kv)
            .max(shape.convolution_width);
        let hidden_bytes = BF16_BYTES * rows * u64::from(shape.hidden);
        let wide_bytes = BF16_BYTES * rows * u64::from(widest);
        let sizes = [
            hidden_bytes,
            wide_bytes,
            wide_bytes,
            BF16_BYTES * rows * u64::from(shape.convolution_width.max(shape.attention_q + shape.attention_kv)),
            BF16_BYTES * rows * u64::from(shape.gdn_value_width.max(shape.attention_q)),
            BF16_BYTES * rows * 2 * u64::from(shape.intermediate),
            BF16_BYTES * rows * u64::from(shape.intermediate),
            hidden_bytes,
            FP32_BYTES * rows * u64::from(shape.value_heads),
            FP32_BYTES * rows * u64::from(shape.value_heads),
            8,
        ];
        let mut made = Vec::with_capacity(sizes.len());
        let outcome = (|| -> Result<(), ForwardError> {
            for bytes in sizes {
                let bytes =
                    usize::try_from(bytes).map_err(|_| ForwardError::Invalid("a workspace too large".to_owned()))?;
                made.push(lane.alloc_async(bytes)?);
            }
            if let Some(position) = made.last() {
                lane.zero_device_memory(position.ptr(), 8)?;
            }
            Ok(())
        })();
        if let Err(error) = outcome {
            free_all(lane, made);
            return Err(error);
        }
        let [
            normalized,
            first,
            second,
            mixed,
            context,
            gate_up,
            swiglu,
            delta,
            alpha,
            beta,
            position,
        ] = <[DeviceBuffer; 11]>::try_from(made)
            .map_err(|_| ForwardError::Invalid("the workspace is 11 buffers".to_owned()))?;
        Ok(LayerForward {
            shape,
            epsilon: config.eps as f32,
            theta: config.theta,
            max_rows,
            normalized,
            first,
            second,
            mixed,
            context,
            gate_up,
            swiglu,
            delta,
            alpha,
            beta,
            position,
            slots: Vec::new(),
            library: Arc::clone(library),
            pool: Arc::clone(pool),
            released: Arc::new(ReleasedBuffers::new()),
        })
    }

    /// The widths this layer runs with.
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Where [`head`](Self::head) leaves the mixer's two projections of the normalized rows: attention's query_key and
    /// gate_value, or GDN's query_key and value_z. BF16 `[rows, width]` on the device, valid once the lane has run the
    /// head and until the next work that uses the workspace.
    pub fn projections(&self) -> (DevicePtr, DevicePtr) {
        (self.first.ptr(), self.second.ptr())
    }

    /// The rows the workspace holds.
    pub fn max_rows(&self) -> u32 {
        self.max_rows
    }

    /// Runs `layer` over `sequences` sequences of `length` tokens whose hidden states are `hidden` (BF16 device
    /// rows), every step on `context`'s lane: [`head`](Self::head), then [`sequence`](Self::sequence) for each
    /// sequence, then [`tail`](Self::tail). Call [`retire`](Self::retire) once the lane has retired the work.
    pub fn forward<Q>(
        &mut self,
        context: &mut LaneContext<'_, Q>,
        layer: &LayerWeights,
        hidden: DevicePtr,
        sequences: u32,
        length: u32,
        mut observer: Option<Observer<'_, Q>>,
    ) -> Result<(), ForwardError>
    where
        Q: LaneEnqueue<Kernel = CudaKernel> + LaneStream<Raw = *mut c_void>,
    {
        let rows = sequences
            .checked_mul(length)
            .ok_or_else(|| ForwardError::Invalid("the rows overflow".to_owned()))?;
        self.prepare_slots(context.ops(), sequences)?;
        self.head(context, layer, hidden, rows, again(&mut observer))?;
        for sequence in 0..sequences {
            self.sequence(context, layer, sequence, length)?;
        }
        self.tail(context, layer, hidden, rows, observer)
    }

    /// Makes the GDN states and attention caches of `sequences` slots.
    pub fn prepare_slots(&mut self, lane: &mut dyn LaneOps, sequences: u32) -> Result<(), ForwardError> {
        let shape = self.shape;
        let channels = u64::from(shape.convolution_width);
        let convolution_bytes = channels * (u64::from(shape.conv_kernel) - 1) * BF16_BYTES;
        let recurrent_bytes =
            u64::from(shape.value_heads) * u64::from(shape.key_dim) * u64::from(shape.value_dim) * FP32_BYTES;
        while self.slots.len() < sequences as usize {
            let convolution = lane.alloc_async(to_usize(convolution_bytes)?)?;
            let recurrent = match lane.alloc_async(to_usize(recurrent_bytes)?) {
                Ok(recurrent) => recurrent,
                Err(error) => {
                    free_all(lane, vec![convolution]);
                    return Err(error.into());
                }
            };
            let made = KvCache::new(
                &self.library,
                &self.released,
                1,
                u64::from(shape.attention_kv),
                KvFormat::Nvfp4,
                u64::from(self.max_rows),
                0,
            )
            .map_err(ForwardError::from)
            .and_then(|cache| {
                // The first reservation would otherwise allocate the pinned memory its table upload stages from.
                cache.prime_staging(&self.pool, 1)?;
                Ok(cache)
            });
            let cache = match made {
                Ok(cache) => cache,
                Err(error) => {
                    free_all(lane, vec![convolution, recurrent]);
                    return Err(error);
                }
            };
            self.slots.push(Slot {
                convolution,
                recurrent,
                cache,
                retained: KvRetained::default(),
            });
        }
        Ok(())
    }

    /// The rows of all sequences at once: the input norm and the mixer's input projections (with GDN's a/b control).
    pub fn head<Q>(
        &mut self,
        context: &mut LaneContext<'_, Q>,
        layer: &LayerWeights,
        hidden: DevicePtr,
        rows: u32,
        mut observer: Option<Observer<'_, Q>>,
    ) -> Result<(), ForwardError>
    where
        Q: LaneEnqueue<Kernel = CudaKernel> + LaneStream<Raw = *mut c_void>,
    {
        self.check_rows(rows)?;
        let shape = self.shape;
        let h = shape.hidden;
        context.ops().rms_norm_unit_offset_bf16(
            hidden,
            required(layer, &Part::InputNorm)?,
            self.normalized.ptr(),
            rows,
            h,
            self.epsilon,
        )?;
        if layer.attention() {
            let width = shape.attention_q + shape.attention_kv;
            let fused = format!("{}attention/query_key_gate_value", layer.prefix());
            if let Some(weights) = layer.object(&fused) {
                // The MTP block's one object holds the query_key rows, then the gate_value rows.
                if let Some(observer) = again(&mut observer) {
                    let observation = Observation {
                        object: &fused,
                        input: self.normalized.ptr(),
                        rows,
                        columns: h,
                    };
                    observer(&observation, context)?;
                }
                context.linear(self.normalized.ptr(), weights, self.first.ptr(), rows, h, width)?;
                let second_half = weights.offset(BF16_BYTES * u64::from(width) * u64::from(h));
                context.linear(self.normalized.ptr(), second_half, self.second.ptr(), rows, h, width)?;
            } else {
                self.apply(
                    context,
                    layer,
                    "attention/query_key",
                    (self.normalized.ptr(), self.first.ptr()),
                    (rows, h, width),
                    again(&mut observer),
                )?;
                self.apply(
                    context,
                    layer,
                    "attention/gate_value",
                    (self.normalized.ptr(), self.second.ptr()),
                    (rows, h, width),
                    observer,
                )?;
            }
            return Ok(());
        }
        // query_key and value_z read the same rows: the observer sees them under both names.
        self.apply(
            context,
            layer,
            "gdn/query_key",
            (self.normalized.ptr(), self.first.ptr()),
            (rows, h, shape.qk_width),
            again(&mut observer),
        )?;
        self.apply(
            context,
            layer,
            "gdn/value_z",
            (self.normalized.ptr(), self.second.ptr()),
            (rows, h, shape.vz_width),
            observer,
        )?;
        context.ops().gdn_project_control_fp32(
            self.normalized.ptr(),
            required(layer, &Part::AProjection)?,
            required(layer, &Part::BProjection)?,
            required(layer, &Part::ALog)?,
            required(layer, &Part::DtBias)?,
            self.alpha.ptr(),
            self.beta.ptr(),
            rows,
            h,
            shape.value_heads,
        )?;
        Ok(())
    }

    /// Sequence `s` of `length` tokens: the GDN convolution and recurrence from fresh state, or the Q/K norm, RoPE and
    /// causal attention over a fresh cache. Sequences are independent of each other. Call
    /// [`prepare_slots`](Self::prepare_slots) first.
    pub fn sequence<Q>(
        &mut self,
        context: &mut LaneContext<'_, Q>,
        layer: &LayerWeights,
        s: u32,
        length: u32,
    ) -> Result<(), ForwardError>
    where
        Q: LaneEnqueue<Kernel = CudaKernel> + LaneStream<Raw = *mut c_void>,
    {
        let shape = self.shape;
        let offset = u64::from(s) * u64::from(length);
        let end = offset + u64::from(length);
        if end > u64::from(self.max_rows) {
            return Err(ForwardError::Invalid(format!(
                "rows up to {end} exceed the workspace's {}",
                self.max_rows
            )));
        }
        let slot = self
            .slots
            .get_mut(s as usize)
            .ok_or_else(|| ForwardError::Invalid(format!("sequence {s} has no slot; prepare slots first")))?;
        let at = |buffer: &DeviceBuffer, width: u32, bytes: u64| buffer.ptr().offset(bytes * offset * u64::from(width));
        if !layer.attention() {
            let ops = context.ops();
            ops.zero_device_memory(slot.convolution.ptr(), slot.convolution.bytes() as u64)?;
            ops.zero_device_memory(slot.recurrent.ptr(), slot.recurrent.bytes() as u64)?;
            ops.gdn_convolution_bf16(
                at(&self.first, shape.qk_width, BF16_BYTES),
                at(&self.second, shape.vz_width, BF16_BYTES),
                required(layer, &Part::Convolution)?,
                slot.convolution.ptr(),
                at(&self.mixed, shape.convolution_width, BF16_BYTES),
                length,
                shape.qk_width,
                shape.gdn_value_width,
                shape.convolution_width,
                shape.conv_kernel,
            )?;
            ops.gdn_recurrence_bf16(
                at(&self.mixed, shape.convolution_width, BF16_BYTES),
                at(&self.alpha, shape.value_heads, FP32_BYTES),
                at(&self.beta, shape.value_heads, FP32_BYTES),
                slot.recurrent.ptr(),
                at(&self.context, shape.gdn_value_width, BF16_BYTES),
                length,
                shape.key_heads,
                shape.value_heads,
                shape.key_dim,
                shape.value_dim,
                (1.0 / f64::from(shape.key_dim).sqrt()) as f32,
            )?;
            return Ok(());
        }
        let width = shape.attention_q + shape.attention_kv;
        let query_key = at(&self.first, width, BF16_BYTES);
        let gate_value = at(&self.second, width, BF16_BYTES);
        let normed = at(&self.mixed, width, BF16_BYTES);
        context.ops().attention_qk_norm_rope_bf16(
            query_key,
            required(layer, &Part::QueryNorm)?,
            required(layer, &Part::KeyNorm)?,
            normed,
            length,
            shape.heads,
            shape.kv_heads,
            shape.head_dim,
            shape.rotary_dim,
            0,
            self.position.ptr(),
            self.epsilon,
            self.theta,
        )?;
        // A previous run's rows are dropped; the cache starts empty.
        slot.cache.discard();
        slot.cache.truncate(0)?;
        slot.cache
            .reserve(context.ops(), &self.pool, 0, u64::from(length), &mut slot.retained)?;
        let (keys, values) = match (slot.cache.key_table(0), slot.cache.value_table(0)) {
            (Some(keys), Some(values)) => (keys, values),
            _ => return Err(ForwardError::Invalid("the cache has no page tables".to_owned())),
        };
        let ops = context.ops();
        ops.attention_kv_append_nvfp4(
            normed,
            gate_value,
            keys,
            values,
            length,
            shape.attention_q,
            shape.attention_kv,
            0,
            self.position.ptr(),
        )?;
        slot.cache.appended(u64::from(length))?;
        ops.attention_causal_nvfp4(
            normed,
            gate_value,
            keys,
            values,
            at(&self.context, shape.attention_q, BF16_BYTES),
            length,
            shape.heads,
            shape.kv_heads,
            shape.head_dim,
            length,
            0,
            self.position.ptr(),
            DevicePtr::NULL,
        )?;
        Ok(())
    }

    /// The rows of all sequences at once: the mixer's output (GDN's gated norm first), the residual and the MLP.
    pub fn tail<Q>(
        &mut self,
        context: &mut LaneContext<'_, Q>,
        layer: &LayerWeights,
        hidden: DevicePtr,
        rows: u32,
        mut observer: Option<Observer<'_, Q>>,
    ) -> Result<(), ForwardError>
    where
        Q: LaneEnqueue<Kernel = CudaKernel> + LaneStream<Raw = *mut c_void>,
    {
        self.check_rows(rows)?;
        let shape = self.shape;
        let (h, intermediate) = (shape.hidden, shape.intermediate);
        if layer.attention() {
            self.apply(
                context,
                layer,
                "attention/output",
                (self.context.ptr(), self.delta.ptr()),
                (rows, shape.attention_q, h),
                again(&mut observer),
            )?;
        } else {
            context.ops().gdn_gated_rms_norm_bf16(
                self.context.ptr(),
                self.second.ptr(),
                required(layer, &Part::GdnNorm)?,
                self.first.ptr(),
                rows,
                shape.value_heads,
                shape.value_dim,
                self.epsilon,
            )?;
            self.apply(
                context,
                layer,
                "gdn/output",
                (self.first.ptr(), self.delta.ptr()),
                (rows, shape.gdn_value_width, h),
                again(&mut observer),
            )?;
        }
        context
            .ops()
            .residual_add_bf16(hidden, self.delta.ptr(), hidden, rows, h)?;
        context.ops().rms_norm_unit_offset_bf16(
            hidden,
            required(layer, &Part::PostNorm)?,
            self.normalized.ptr(),
            rows,
            h,
            self.epsilon,
        )?;
        self.apply(
            context,
            layer,
            "mlp/gate_up",
            (self.normalized.ptr(), self.gate_up.ptr()),
            (rows, h, 2 * intermediate),
            again(&mut observer),
        )?;
        context
            .ops()
            .swiglu_bf16(self.gate_up.ptr(), self.swiglu.ptr(), rows, intermediate)?;
        self.apply(
            context,
            layer,
            "mlp/down",
            (self.swiglu.ptr(), self.delta.ptr()),
            (rows, intermediate, h),
            observer,
        )?;
        context
            .ops()
            .residual_add_bf16(hidden, self.delta.ptr(), hidden, rows, h)?;
        Ok(())
    }

    /// Publishes the attention rows the finished work appended and drops the staging its table uploads read. Call it
    /// once the lane has retired everything queued since the last call.
    pub fn retire(&mut self) {
        for slot in &mut self.slots {
            slot.cache.commit_submitted();
            slot.retained.staging.clear();
        }
    }

    /// Frees the workspace and the slots in stream order on `lane`, every buffer even when one refuses (the first
    /// refusal is the error). The frees are ordered only against `lane`'s work.
    pub fn release(self, lane: &mut dyn LaneOps) -> Result<(), ForwardError> {
        let LayerForward {
            normalized,
            first,
            second,
            mixed,
            context,
            gate_up,
            swiglu,
            delta,
            alpha,
            beta,
            position,
            slots,
            released,
            ..
        } = self;
        let mut first_failure = None;
        let mut free = |buffer: DeviceBuffer| {
            if let Err(error) = lane.free_async(buffer) {
                first_failure.get_or_insert(error);
            }
        };
        for slot in slots {
            let Slot {
                convolution,
                recurrent,
                cache,
                retained,
            } = slot;
            // The cache hands its pages and tables to `released` when it is dropped.
            drop(cache);
            drop(retained);
            free(convolution);
            free(recurrent);
        }
        released.drain(&mut free);
        for buffer in [
            normalized, first, second, mixed, context, gate_up, swiglu, delta, alpha, beta, position,
        ] {
            free(buffer);
        }
        first_failure.map_or(Ok(()), |error| Err(error.into()))
    }

    fn check_rows(&self, rows: u32) -> Result<(), ForwardError> {
        if rows > self.max_rows {
            return Err(ForwardError::Invalid(format!(
                "{rows} rows exceed the workspace's {}",
                self.max_rows
            )));
        }
        Ok(())
    }

    fn apply<Q>(
        &self,
        context: &mut LaneContext<'_, Q>,
        layer: &LayerWeights,
        role: &str,
        (input, output): (DevicePtr, DevicePtr),
        (rows, k, n): (u32, u32, u32),
        observer: Option<Observer<'_, Q>>,
    ) -> Result<(), ForwardError>
    where
        Q: LaneEnqueue<Kernel = CudaKernel> + LaneStream<Raw = *mut c_void>,
    {
        let name = format!("{}{role}", layer.prefix());
        if let Some(observer) = observer {
            let observation = Observation {
                object: &name,
                input,
                rows,
                columns: k,
            };
            observer(&observation, context)?;
        }
        let weights = layer
            .object(&name)
            .ok_or_else(|| ForwardError::Invalid(format!("the layer has no object {name}")))?;
        context.linear(input, weights, output, rows, k, n)
    }
}

/// The observer lent again for a shorter time.
fn again<'s, Q>(observer: &'s mut Option<Observer<'_, Q>>) -> Option<Observer<'s, Q>> {
    match observer {
        Some(observer) => Some(&mut **observer),
        None => None,
    }
}

fn required(layer: &LayerWeights, part: &Part) -> Result<DevicePtr, ForwardError> {
    layer
        .ptr(part)
        .ok_or_else(|| ForwardError::Invalid(format!("the layer has no {part:?}")))
}

fn to_usize(bytes: u64) -> Result<usize, ForwardError> {
    usize::try_from(bytes).map_err(|_| ForwardError::Invalid("a state too large".to_owned()))
}

impl Debug for LayerForward {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        formatter
            .debug_struct("LayerForward")
            .field("shape", &self.shape)
            .field("sequences", &self.slots.len())
            .finish_non_exhaustive()
    }
}

/// Frees `buffers` in stream order on `lane` after a failure; a refusal adds nothing the caller can act on.
fn free_all(lane: &mut dyn LaneOps, buffers: Vec<DeviceBuffer>) {
    for buffer in buffers {
        let _ = lane.free_async(buffer);
    }
}
