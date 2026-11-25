//! Reading and writing Hugging Face checkpoints, and fusing their tensors into the quantized objects.
//!
//! - [`safetensors`]: the tensors of a checkpoint directory, read with positioned reads into memory the caller owns
//!   (a pinned arena, say). Nothing is memory-mapped.
//! - [`writer`]: one safetensors file, written as `<name>.part` and renamed when complete.
//! - [`objects`]: the quantized objects of a Qwen 3.8 checkpoint as row ranges of its BF16 tensors.
//!
//! Every operation that touches a file goes through `AsyncIo` (the writer's durability barrier and cleanup, which it
//! has no call for yet, go through [`durable`](crate::durable) on a host lake) and reports through a [`Done`] callback. The callback is
//! called once, from a frame of the lattice that drives the `AsyncIo`, and must not wait: it hands the result on (to a
//! lake, a hand-off) and returns. A function that returns `Err` started nothing and never calls its callback.

mod io;
pub mod objects;
pub mod safetensors;
pub mod writer;

use std::io::Error as IoError;
use std::sync::Arc;

use thiserror::Error;

pub use io::Done;

/// Why a checkpoint could not be read or written.
#[derive(Debug, Error)]
pub enum CheckpointError {
    /// The file system refused an operation.
    #[error("{context}: {source}")]
    Io {
        context: String,
        #[source]
        source: IoError,
    },
    /// A file does not hold what its format says.
    #[error("{0}")]
    Format(String),
    /// The checkpoint has no tensor of this name.
    #[error("checkpoint has no tensor {0}")]
    MissingTensor(String),
    /// A request that cannot be served: a range outside its tensor, a duplicate name, an unknown dtype.
    #[error("{0}")]
    Invalid(String),
    /// The host lake would not take the call: it is closed, or no unit of it is admitted.
    #[error("the host lake is closed: nothing was started")]
    LakeClosed,
}

impl CheckpointError {
    pub(crate) fn io(context: impl Into<String>, source: IoError) -> Self {
        CheckpointError::Io {
            context: context.into(),
            source,
        }
    }
}

/// Calls `done` with the failure of a step that could not be submitted.
pub(crate) fn report<T>(done: &Arc<dyn Fn(Result<T, CheckpointError>) + Send + Sync>, error: CheckpointError) {
    done(Err(error));
}
