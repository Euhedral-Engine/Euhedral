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
//! drop that frees device or pinned memory (`cudaFree` and `cudaFreeHost` wait for the device). Those are for startup,
//! shutdown and tests; on the hot path buffers go through the pool and the retire list instead. There is no callback
//! from the driver into Rust.

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
pub use retire::RetireList;
pub use submission::{Submission, SubmitMode};
