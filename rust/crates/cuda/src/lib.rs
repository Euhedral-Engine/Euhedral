//! The Euhedral CUDA library from Rust.
//!
//! - [`CudaLibrary`] loads `libeuhedral_cuda` (preloading its runtime, setting the kernel include directory, binding
//!   the device) and carries the generated entry points ([`sys`]).
//! - [`DeviceBuffer`], [`PinnedBuffer`] and [`HostWeightsArena`] own device, page-locked and host-weight memory.
//! - [`PinnedPool`] recycles pinned buffers.
//! - [`KernelOps`] and [`LaneOps`] are the operations the stages queue; [`launch`] plans and enqueues them on RustEE
//!   lanes ([`RusteeOps`](launch::ops::RusteeOps)).
//!
//! The device waits are the ones documented as blocking: `synchronize` (the library's), the host copies, and every
//! plain drop that frees device or pinned memory (`cudaFree` and `cudaFreeHost` wait for the whole device under the
//! driver's lock, which can deadlock against a lane parked on a host flag). Those are for startup, shutdown and tests;
//! on a running engine pinned buffers come from the pool and stream-ordered device buffers are freed with `free_async`,
//! which does not wait. There is no callback from the driver into Rust.
//!
//! The waiting wrappers and the waiting entry points they call are banned by the workspace clippy rules, and the
//! wrapper that legitimately waits carries a scoped allow. The generated [`sys::EuhedralCuda`] also exposes each entry
//! point as a `pub` function-pointer field; clippy cannot see a call through a field, so `(api.euhedral_cuda_free)(p)`
//! needs the same scoped allow by convention. Call the methods, which are banned, rather than the fields.

mod device;
mod error;
pub mod launch;
mod library;
mod memory;
pub mod ops;
mod pool;

/// The generated declarations of the C ABI (`cargo xtask bindings` rewrites this file).
#[rustfmt::skip]
#[allow(
    unsafe_op_in_unsafe_fn,
    missing_debug_implementations,
    missing_docs,
    dead_code,
    non_camel_case_types,
    clippy::absolute_paths,
    clippy::all
)]
pub mod sys;

pub use device::DeviceInfo;
pub use error::CudaError;
pub use library::CudaLibrary;
pub use memory::{AllocationId, ArenaRange, DeviceBuffer, DevicePtr, HostWeightsArena, PinnedBuffer};
pub use ops::{KernelOps, LaneOps, SubmitMode};
pub use pool::{PinnedPool, PooledPinned};
