//! The Euhedral CUDA library from Rust.
//!
//! - [`CudaLibrary`] loads `libeuhedral_cuda` (preloading its runtime, setting the kernel include directory, binding
//!   the device) and carries the generated entry points ([`sys`]).
//! - [`DeviceBuffer`], [`PinnedBuffer`] and [`HostWeightsArena`] own device, page-locked and host-weight memory.
//! - [`PinnedPool`] recycles pinned buffers; [`RetireList`] frees a buffer once the lane work that last used it has
//!   retired.
//! - [`CudaLane`] is one stream with an event ring and a host-mapped completion word, and implements
//!   `DeviceQueue`: `completed()` is one atomic read.
//! - [`Submission`] takes a lane (one at a time; a second `begin` returns `LaneBusy`) and selects it for the calling
//!   thread; [`ops`] holds the operations, each taking one.
//! - [`GraphRecorder`] and [`GraphExec`] capture a lane's submissions and replay them.
//!
//! The device waits are the ones documented as blocking: `synchronize` (library and lane), the host copies, and every
//! plain drop that frees device or pinned memory (`cudaFree` and `cudaFreeHost` wait for the whole device under the
//! driver's lock, which can deadlock against a lane parked on a host flag). Those are for startup, shutdown and tests;
//! on a running engine pinned buffers come from the pool and stream-ordered device buffers go through the retire list
//! or `free_async`, none of which wait. There is no callback from the driver into Rust.
//!
//! The waiting wrappers and the waiting entry points they call are banned by the workspace clippy rules, and the
//! wrapper that legitimately waits carries a scoped allow. The generated [`sys::EuhedralCuda`] also exposes each entry
//! point as a `pub` function-pointer field; clippy cannot see a call through a field, so `(api.euhedral_cuda_free)(p)`
//! needs the same scoped allow by convention. Call the methods, which are banned, rather than the fields.

mod device;
mod error;
mod graph;
mod lane;
mod library;
mod memory;
pub mod ops;
mod pool;
mod retire;
mod stack;
mod submission;

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
pub use graph::{GraphExec, GraphRecorder};
pub use lane::CudaLane;
pub use library::CudaLibrary;
pub use memory::{AllocationId, DeviceBuffer, DevicePtr, HostWeightsArena, PinnedBuffer};
pub use pool::{PinnedPool, PooledPinned};
pub use retire::{Retire, RetireList};
pub use submission::{Submission, SubmitMode};
