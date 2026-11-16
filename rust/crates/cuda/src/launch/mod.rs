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
