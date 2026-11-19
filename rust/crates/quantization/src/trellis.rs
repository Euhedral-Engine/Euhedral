//! The trellis kernels: what each launch of `trellis.cu` is for the library, as plain values.
//!
//! - [`Rate`]: the bits per weight the kernels support, and what each rate needs (words, scratch, threads).
//! - the planners (`plan_*`): from a shape and device addresses to the [`Step`]s that run it. A planner touches no
//!   device and refuses every shape a launch would fault on (including a grid beyond the device's limits), so a
//!   planned step is never refused for its shape and the geometry is tested without a card.
//! - [`TrellisKernels`]: the resolved kernels of the trellis module, and the enqueue of planned steps on a lane.
//!
//! Tiles are 16x16 weights in tensor-core element order ([`TENSOR_CORE_ORDER`]); every address is a device address.

mod launch;
mod plan;
mod quantizer;
mod rate;
mod run;
mod search;

use euhedral_gpu::{Args, GpuError, Launch};
use thiserror::Error;

pub use plan::{
    Reconstruction, plan_clean_row_scales, plan_decode, plan_fill_floats, plan_from_tiles, plan_hadamard_columns,
    plan_hadamard_rows, plan_pack, plan_reconstruct, plan_sum_products, plan_tile_mean_square, plan_to_tiles,
    plan_viterbi,
};
pub use quantizer::{
    LaneParts, QuantizeError, Quantized, Refused, Request, TrellisLane, TrellisSettings, working_bytes,
};
pub use rate::{Rate, TENSOR_CORE_ORDER, TILE_WEIGHTS};
pub use run::TrellisKernels;
pub use search::{CHOLESKY_RETRIES, SIGMA_REG};

pub use crate::chain::{Finished, Scratch};

/// One kernel launch: the kernel's name in the trellis module, its geometry and its arguments. Only the planners
/// build one, so its arguments always match the kernel's parameter list.
#[derive(Clone, Debug)]
pub struct Step {
    kernel: &'static str,
    launch: Launch,
    args: Args,
}

impl Step {
    /// The kernel's symbol name.
    pub fn kernel(&self) -> &'static str {
        self.kernel
    }

    /// The grid, block and dynamic shared memory.
    pub fn launch(&self) -> &Launch {
        &self.launch
    }

    /// The arguments, in the kernel's parameter order.
    pub fn args(&self) -> &Args {
        &self.args
    }
}

/// Why a launch cannot be planned or run.
#[derive(Debug, Error)]
pub enum TrellisError {
    /// The rate is not one the kernels support (integers 2 to 6, and 2.5 and 3.5).
    #[error("unsupported rate {0}")]
    UnsupportedRate(f64),
    /// The scratch is smaller than one tile needs.
    #[error("Viterbi scratch of {bytes} bytes holds no tile at {bits} bits (one needs {needed})")]
    ScratchTooSmall { bytes: u64, bits: f64, needed: u64 },
    /// The scratch tile count given is zero.
    #[error("the scratch holds no tile")]
    NoScratchTiles,
    /// A dimension is not a multiple of the unit the kernel works in.
    #[error("{name} {value} is not a positive multiple of {unit}")]
    NotMultiple { name: &'static str, value: u64, unit: u32 },
    /// A strip of rows lies outside the matrix.
    #[error("a strip of 16 rows from row {first} lies outside {rows} rows")]
    StripOutside { first: u32, rows: u32 },
    /// A launch grid would be taller than the device allows.
    #[error("{name} {value} would make a grid taller than {most}")]
    GridTooTall { name: &'static str, value: u64, most: u32 },
    /// A count or extent does not fit the kernel's 32-bit parameter.
    #[error("{name} {value} does not fit a 32-bit kernel parameter")]
    TooLarge { name: &'static str, value: u64 },
    /// A device address is null.
    #[error("{0} is a null address")]
    NullAddress(&'static str),
    /// The module has no kernel of this name.
    #[error("the trellis module has no kernel {0}")]
    NoKernel(String),
    /// The device or the driver failed.
    #[error("device failure: {0}")]
    Gpu(#[from] GpuError),
}

#[cfg(test)]
mod tests;
