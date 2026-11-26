//! A mixed artifact from calibrated source artifacts and an allocation: every object of the base, except that the
//! projections the allocation names come from the source of their option, their companions with them.
//!
//! - [`SourceSet`]: the source artifacts, opened and checked against each other. It answers the search's questions
//!   about the objects ([`Sources`](crate::mixed::Sources)).
//! - [`view`]: which object comes from which source, in the order the written artifact holds them.
//! - [`write`]: the artifact as a file, each payload copied from its source in chunks, through `AsyncIo`.
//!
//! Every operation that touches a file goes through `AsyncIo` (the durability barrier and the removal of a failed
//! file, which it has no call for yet, go through [`durable`](crate::durable) on a host lake) and reports through a
//! [`Done`] callback, called once from a frame of the lattice that drives the `AsyncIo`. A function that returns `Err`
//! started nothing and never calls its callback.

mod artifact;
mod chunks;
mod command;
mod sources;
mod view;
mod write;

use std::sync::Arc;

use euhedral_inference_edrl::EdrlError;
use thiserror::Error;

use crate::checkpoint::CheckpointError;
use crate::mixed::MixedError;

pub use artifact::ArtifactFile;
// What `SourceSet::descriptor` returns and the companions of a trellis object are read with.
pub use command::{Assembled, Request, run};
pub use euhedral_inference_edrl::{TensorEntry, TrellisCompanions, WeightFormat};
pub use sources::{COMPANION_SUFFIXES, SourceSet};
pub use view::{Piece, View, view};
pub use write::write;

/// Where a result goes: called once, from a frame, and must not wait.
pub type Done<T> = Arc<dyn Fn(Result<T, AssembleError>) + Send + Sync>;

/// Why an artifact could not be opened, checked, selected or written.
#[derive(Debug, Error)]
pub enum AssembleError {
    /// An argument or a source is not what assembly accepts; the message names the object or the option.
    #[error("{0}")]
    Invalid(String),
    /// A file is not an artifact.
    #[error("{path}: {source}")]
    Artifact {
        path: String,
        #[source]
        source: EdrlError,
    },
    /// The file system refused an operation, or a file is cut short.
    #[error(transparent)]
    Io(#[from] CheckpointError),
    /// The search's data refused something.
    #[error(transparent)]
    Mixed(#[from] MixedError),
}

impl AssembleError {
    pub(crate) fn invalid(message: impl Into<String>) -> AssembleError {
        AssembleError::Invalid(message.into())
    }
}
