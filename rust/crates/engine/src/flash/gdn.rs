//! The Gated DeltaNet block with the Flash-Next convolution (`GdnLayer.run`): the projections, the causal convolution
//! over the qkv channels with its carried history, the gate controls, the delta-rule recurrence, the gated norm and the
//! output projection.
//!
//! The per-sequence state (the convolution history and the recurrent state) is supplied by the caller as device
//! addresses, as Java's `GdnLayer` reads them from the sequence's state.

use euhedral_inference_cuda::DevicePtr;
use euhedral_inference_edrl::GdnConfig;

use super::block::{BlockBody, BlockCall};
use super::buffers::{Carver, bf16_bytes, f32_bytes};
use super::ops;
use super::sequence::{Op, OpList};
use super::{FlashError, Weight};

/// The kernel's key and value head width.
const HEAD_DIM: u32 = 128;
/// The most history rows the convolution keeps.
const MAX_TAPS: u32 = 32;

/// The sizes of one GDN layer, checked against what the kernels serve.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GdnShape {
    hidden: u32,
    key_heads: u32,
    value_heads: u32,
    taps: u32,
    epsilon: f32,
    /// 0 for silu, 1 for sigmoid.
    activation: u32,
}

impl GdnShape {
    /// The layer of `config` over `hidden` values, with the attention's `output_gate` activation and the norm's
    /// `epsilon`.
    pub fn new(hidden: u32, config: &GdnConfig, output_gate: &str, epsilon: f32) -> Result<Self, FlashError> {
        const OP: &str = "gdn layer";
        if config.key_head_dim != HEAD_DIM || config.value_head_dim != HEAD_DIM {
            return Err(FlashError::refused(
                OP,
                format!(
                    "the recurrence serves heads of {HEAD_DIM}, got key {} and value {}",
                    config.key_head_dim, config.value_head_dim
                ),
            ));
        }
        if config.num_key_heads == 0 || config.num_value_heads % config.num_key_heads != 0 {
            return Err(FlashError::refused(
                OP,
                format!(
                    "{} value heads are not a multiple of {} key heads",
                    config.num_value_heads, config.num_key_heads
                ),
            ));
        }
        if !(2..=MAX_TAPS).contains(&config.conv_kernel_dim) {
            return Err(FlashError::refused(
                OP,
                format!(
                    "the convolution has 2 to {MAX_TAPS} taps, got {}",
                    config.conv_kernel_dim
                ),
            ));
        }
        let activation = match output_gate {
            "silu" => 0,
            "sigmoid" => 1,
            other => {
                return Err(FlashError::refused(
                    OP,
                    format!("output gate '{other}' is not silu or sigmoid"),
                ));
            }
        };
        let channels = 2 * u64::from(config.num_key_heads) * u64::from(HEAD_DIM)
            + u64::from(config.num_value_heads) * u64::from(HEAD_DIM);
        if hidden == 0 || hidden % 8 != 0 || channels > u64::from(u32::MAX) / 2 || !epsilon.is_finite() {
            return Err(FlashError::refused(
                OP,
                format!("hidden {hidden} (a positive multiple of 8), {channels} channels, epsilon {epsilon}"),
            ));
        }
        Ok(Self {
            hidden,
            key_heads: config.num_key_heads,
            value_heads: config.num_value_heads,
            taps: config.conv_kernel_dim,
            epsilon,
            activation,
        })
    }

    /// The channels of the convolution: query, key and value.
    pub fn channels(&self) -> u32 {
        (2 * self.key_heads + self.value_heads) * HEAD_DIM
    }

    /// The width of the values the recurrence writes.
    pub fn value_width(&self) -> u32 {
        self.value_heads * HEAD_DIM
    }

    /// The rows of convolution history a sequence keeps.
    pub fn history_rows(&self) -> u32 {
        self.taps - 1
    }

    fn carve(&self, carver: &mut Carver, rows: u32) -> GdnScratch {
        GdnScratch {
            qkv: carver.take(bf16_bytes(rows, self.channels())),
            convolved: carver.take(bf16_bytes(rows, self.channels())),
            z: carver.take(bf16_bytes(rows, self.value_width())),
            core: carver.take(bf16_bytes(rows, self.value_width())),
            normed: carver.take(bf16_bytes(rows, self.value_width())),
            a: carver.take(bf16_bytes(rows, self.value_heads)),
            b: carver.take(bf16_bytes(rows, self.value_heads)),
            alpha: carver.take(f32_bytes(rows, self.value_heads)),
            beta: carver.take(f32_bytes(rows, self.value_heads)),
        }
    }

    /// The bytes of the scratch of `rows` rows.
    pub fn scratch_bytes(&self, rows: u32) -> u64 {
        let mut carver = Carver::new(DevicePtr::NULL);
        self.carve(&mut carver, rows);
        carver.bytes()
    }
}

#[derive(Clone, Copy, Debug)]
struct GdnScratch {
    qkv: DevicePtr,
    convolved: DevicePtr,
    z: DevicePtr,
    core: DevicePtr,
    normed: DevicePtr,
    a: DevicePtr,
    b: DevicePtr,
    alpha: DevicePtr,
    beta: DevicePtr,
}

/// The weights of one GDN layer: the NVFP4 projections and the BF16 gate projections, convolution and norm.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GdnWeights {
    pub qkv: Weight,
    pub z: Weight,
    pub a: Weight,
    pub b: Weight,
    pub conv: Weight,
    pub a_log: Weight,
    pub dt_bias: Weight,
    pub norm: Weight,
    pub out: Weight,
}

/// A sequence's GDN state: the last `taps - 1` qkv rows and the recurrent matrices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GdnState {
    pub conv_history: DevicePtr,
    pub recurrent: DevicePtr,
}

/// The GDN block as the body of a layer.
#[derive(Clone, Debug)]
pub struct GdnBody {
    shape: GdnShape,
    weights: GdnWeights,
    state: GdnState,
}

impl GdnBody {
    pub fn new(shape: GdnShape, weights: GdnWeights, state: GdnState) -> Self {
        Self { shape, weights, state }
    }
}

impl BlockBody for GdnBody {
    fn ops(&self, call: &BlockCall) -> Result<OpList<'static>, FlashError> {
        let shape = &self.shape;
        let weights = &self.weights;
        let (rows, exact) = (call.rows, call.exact);
        let scratch = shape.carve(&mut Carver::new(call.scratch), rows);
        let (channels, value_width, heads) = (shape.channels(), shape.value_width(), shape.value_heads);
        let linear = |input, weight: Weight, output, k, n| Op::Nvfp4Linear {
            input,
            weights: weight.address,
            output,
            rows,
            in_features: k,
            out_features: n,
            weights_bytes: weight.bytes,
        };
        let mut list = OpList::new();
        list.push(linear(call.input, weights.qkv, scratch.qkv, shape.hidden, channels));
        list.push(linear(call.input, weights.z, scratch.z, shape.hidden, value_width));
        list.push(Op::Launch(ops::linear_bf16(
            call.input,
            weights.a.address,
            scratch.a,
            rows,
            shape.hidden,
            heads,
            exact,
        )?));
        list.push(Op::Launch(ops::linear_bf16(
            call.input,
            weights.b.address,
            scratch.b,
            rows,
            shape.hidden,
            heads,
            exact,
        )?));
        list.push(Op::Launch(ops::gdn_conv(
            scratch.qkv,
            self.state.conv_history,
            weights.conv.address,
            scratch.convolved,
            rows,
            channels,
            shape.taps,
        )?));
        list.push(Op::Launch(ops::conv_history(
            scratch.qkv,
            self.state.conv_history,
            rows,
            channels,
            shape.history_rows(),
        )?));
        list.push(Op::Launch(ops::gdn_control(
            scratch.a,
            scratch.b,
            weights.a_log.address,
            weights.dt_bias.address,
            scratch.alpha,
            scratch.beta,
            rows,
            heads,
        )?));
        list.push(Op::GdnRecurrence {
            convolved: scratch.convolved,
            alpha: scratch.alpha,
            beta: scratch.beta,
            recurrent_state: self.state.recurrent,
            output: scratch.core,
            rows,
            key_heads: shape.key_heads,
            value_heads: heads,
            key_head_dim: HEAD_DIM,
            value_head_dim: HEAD_DIM,
            output_scale: (1.0 / f64::from(HEAD_DIM).sqrt()) as f32,
        });
        list.push(Op::Launch(ops::gdn_gated_norm(
            scratch.core,
            scratch.z,
            weights.norm.address,
            scratch.normed,
            rows,
            heads,
            HEAD_DIM,
            shape.epsilon,
            shape.activation,
        )?));
        list.push(linear(
            scratch.normed,
            weights.out,
            call.output,
            value_width,
            shape.hidden,
        ));
        Ok(list)
    }
}

#[cfg(test)]
mod tests;
