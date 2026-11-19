//! One Qwen 3.8 decoder layer (or the MTP block) in BF16 over a batch of whole sequences, on the engine's kernels.
//!
//! - [`LayerWeights`] holds a layer's weights on the device. [`StagedLayer`] is the same layer read into host memory
//!   from the checkpoint (a frame-driven read), ready to upload on a lane.
//! - [`LayerForward`] runs the layer: [`LayerForward::head`] and [`LayerForward::tail`] over all rows,
//!   [`LayerForward::sequence`] once per sequence. Every quantized linear's input goes to the observer first, with a
//!   [`LaneContext`] so the observer can accumulate its Hessian on the same lane.
//! - [`LaneContext`] is the lane's kernel operations, its cuBLAS handles and the kernels this module adds to the
//!   engine's: the Hessian accumulation (upper blocks, then mirrored) and the row gather.
//!
//! Sequences start at position 0 with a fresh GDN state; attention keeps its keys and values in the engine's NVFP4
//! paged cache. Nothing here waits: every call queues work on the lane, and the caller learns of its completion from
//! the lane's marker.

mod layer;
mod math;
mod weights;

use euhedral_gpu::GpuError;
use euhedral_inference_cuda::CudaError;
use euhedral_inference_engine::state::kv::KvError;
use thiserror::Error;

use crate::checkpoint::CheckpointError;
use crate::checkpoint::objects::Config;

pub use layer::{LayerForward, Observation, Observer};
pub use math::{ForwardKernels, LaneContext, plan_gather_rows, plan_mirror_upper};
pub use weights::{LayerWeights, Part, StagedLayer, Target, stage, transpose_convolution, widen_bf16};

/// Why a forward pass or the loading of a layer failed.
#[derive(Debug, Error)]
pub enum ForwardError {
    /// A device operation or the driver failed.
    #[error(transparent)]
    Cuda(#[from] CudaError),
    /// A launch failed or a kernel could not be resolved.
    #[error(transparent)]
    Gpu(#[from] GpuError),
    /// The KV cache refused a reservation or a frontier move.
    #[error(transparent)]
    Kv(#[from] KvError),
    /// The checkpoint could not be read.
    #[error(transparent)]
    Checkpoint(#[from] CheckpointError),
    /// An argument or a model dimension the layer cannot run with.
    #[error("{0}")]
    Invalid(String),
}

/// The widths the layer's kernels work in, derived from the model's configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    pub hidden: u32,
    pub intermediate: u32,
    pub heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub rotary_dim: u32,
    pub key_heads: u32,
    pub value_heads: u32,
    pub key_dim: u32,
    pub value_dim: u32,
    pub conv_kernel: u32,
    /// Query width of attention (`heads * head_dim`).
    pub attention_q: u32,
    /// Key (and value) width of attention (`kv_heads * head_dim`).
    pub attention_kv: u32,
    /// Query and key width of GDN (`2 * key_heads * key_dim`).
    pub qk_width: u32,
    /// Value width of GDN (`value_heads * value_dim`).
    pub gdn_value_width: u32,
    /// Value and gate width of GDN.
    pub vz_width: u32,
    /// Channels of GDN's convolution.
    pub convolution_width: u32,
}

impl Shape {
    /// The widths of `config`, or why a width does not fit the kernels' 32-bit parameters.
    pub fn new(config: &Config) -> Result<Shape, ForwardError> {
        let narrow = |name: &str, value: u64| {
            u32::try_from(value).map_err(|_| ForwardError::Invalid(format!("{name} {value} does not fit 32 bits")))
        };
        let wide = |name: &str, product: Option<u64>| {
            product
                .ok_or_else(|| ForwardError::Invalid(format!("{name} overflows")))
                .and_then(|value| narrow(name, value))
        };
        let qk_width = wide(
            "the GDN query/key width",
            config
                .key_heads
                .checked_mul(config.key_dim)
                .and_then(|width| width.checked_mul(2)),
        )?;
        let gdn_value_width = wide("the GDN value width", config.value_heads.checked_mul(config.value_dim))?;
        Ok(Shape {
            hidden: narrow("hidden", config.hidden)?,
            intermediate: narrow("intermediate", config.intermediate)?,
            heads: narrow("heads", config.heads)?,
            kv_heads: narrow("kv_heads", config.kv_heads)?,
            head_dim: narrow("head_dim", config.head_dim)?,
            rotary_dim: narrow("rotary_dim", config.rotary_dim)?,
            key_heads: narrow("key_heads", config.key_heads)?,
            value_heads: narrow("value_heads", config.value_heads)?,
            key_dim: narrow("key_dim", config.key_dim)?,
            value_dim: narrow("value_dim", config.value_dim)?,
            conv_kernel: narrow("conv_kernel", config.conv_kernel)?,
            attention_q: wide("the attention query width", config.heads.checked_mul(config.head_dim))?,
            attention_kv: wide("the attention key width", config.kv_heads.checked_mul(config.head_dim))?,
            qk_width,
            gdn_value_width,
            vz_width: wide("the GDN value/gate width", 2u64.checked_mul(u64::from(gdn_value_width)))?,
            convolution_width: wide(
                "the GDN convolution width",
                u64::from(qk_width).checked_add(u64::from(gdn_value_width)),
            )?,
        })
    }
}
