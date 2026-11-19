//! Kernel launches as data: the catalog of every kernel the Rust engine launches, and the families that plan them.

pub mod attention;
pub mod catalog;
pub mod elementwise;
pub mod embedding;
pub mod gdn;
pub mod load;
pub mod nvfp4;
pub mod ops;
pub mod queue;
pub mod table;

use crate::DevicePtr;
use crate::launch::attention::{DEFAULT_FUSED_VERIFY_FROM, FUSED_ROWS};
use crate::launch::elementwise::DEFAULT_RMS_NORM_DECODE_SLICES;
use crate::launch::nvfp4::skinny_target_ctas;

/// What the native wrapper rejects: the status the C function would return instead of launching.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlanError {
    /// `EUHEDRAL_CUDA_INVALID_ARGUMENT`.
    InvalidArgument,
    /// `EUHEDRAL_CUDA_SIZE_OVERFLOW`.
    SizeOverflow,
    /// `EUHEDRAL_CUDA_FORMAT_MISMATCH`.
    FormatMismatch,
    /// `EUHEDRAL_CUDA_ROUTE_UNAVAILABLE`: no kernel serves the shape or the mode; the caller takes another route.
    RouteUnavailable,
}

/// The thread-local modes the C wrappers read (`euhedral_cuda_row_exact`, `euhedral_cuda_exact_numerics`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Modes {
    /// Every row computes as a one-row launch would.
    pub row_exact: bool,
    /// The key-ordered softmax sum and the per-element decode kernel.
    pub exact_numerics: bool,
}

/// `pointer + bytes` as C computes it: wrapping, never a panic.
pub(crate) fn advance(pointer: DevicePtr, bytes: u64) -> DevicePtr {
    // SAFETY: the address is only carried into a launch's arguments, never dereferenced on the host.
    unsafe { DevicePtr::from_raw(pointer.get().wrapping_add(bytes)) }
}

/// The launch choices that depend on the card: sized from the device, or measured on one card and set by the
/// operator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dispatch {
    /// Streaming multiprocessors of the device; the skinny NVFP4 kernel's K split aims for two blocks per one.
    pub sm_count: u32,
    /// The grid-y of a one-row RMS norm (`kernels.rms-norm-decode-slices`).
    pub rms_norm_decode_slices: u32,
    /// The fewest keys at which the fused verifier kernel serves each count of verified rows
    /// (`kernels.fused-verify-from`).
    pub fused_verify_from: [u32; FUSED_ROWS],
}

impl Dispatch {
    /// The defaults of the settings on a device of `sm_count` streaming multiprocessors.
    pub const fn measured(sm_count: u32) -> Self {
        Self {
            sm_count,
            rms_norm_decode_slices: DEFAULT_RMS_NORM_DECODE_SLICES,
            fused_verify_from: DEFAULT_FUSED_VERIFY_FROM,
        }
    }

    /// The blocks a skinny NVFP4 launch splits over K to reach.
    pub const fn skinny_target_ctas(&self) -> u64 {
        skinny_target_ctas(self.sm_count)
    }
}
