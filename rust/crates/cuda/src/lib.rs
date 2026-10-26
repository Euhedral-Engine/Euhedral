//! The Euhedral CUDA library from Rust.
//!
//! - [`CudaLibrary`] loads `libeuhedral_cuda` (preloading its runtime, setting the kernel include directory, binding
//!   the device) and carries the generated entry points ([`sys`]).
//! - [`DeviceBuffer`], [`PinnedBuffer`] and [`HostWeightsArena`] own device, page-locked and host-weight memory.
//! - [`CudaLane`] is one stream with an event ring and a host-mapped completion word, and implements
//!   `DeviceQueue`: `completed()` is one atomic read.
//! - [`Submission`] selects a lane for the calling thread; [`ops`] holds the operations, each taking one.
//! - [`GraphRecorder`] and [`GraphExec`] capture a lane's submissions and replay them.
//!
//! Nothing here waits for the device except the calls documented as blocking (`synchronize`, the host copies), which
//! are for startup, shutdown and tests; there is no callback from the driver into Rust.
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
pub use submission::{Submission, SubmitMode};
