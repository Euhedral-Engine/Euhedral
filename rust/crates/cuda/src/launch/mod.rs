//! Kernel launches as data: the catalog of every kernel the Rust engine launches, and the families that plan them.

pub mod attention;
pub mod catalog;
pub mod elementwise;
pub mod embedding;
pub mod gdn;
pub mod nvfp4;
pub mod table;

/// What the native wrapper rejects: the status the C function would return instead of launching.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlanError {
    /// `EUHEDRAL_CUDA_INVALID_ARGUMENT`.
    InvalidArgument,
    /// `EUHEDRAL_CUDA_SIZE_OVERFLOW`.
    SizeOverflow,
    /// `EUHEDRAL_CUDA_FORMAT_MISMATCH`.
    FormatMismatch,
}

/// The thread-local modes the C wrappers read (`euhedral_cuda_row_exact`, `euhedral_cuda_exact_numerics`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Modes {
    /// Every row computes as a one-row launch would.
    pub row_exact: bool,
    /// The key-ordered softmax sum and the per-element decode kernel.
    pub exact_numerics: bool,
}
