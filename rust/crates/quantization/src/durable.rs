//! The three file operations RustEE's `AsyncIo` does not have: making written data durable, cutting a file back to a
//! length, and removing a file. Until it has them, each is one blocking call, and each runs as a frame of the host
//! lake, so the worker that submitted it never waits: the call's completion is the next frame, offered to the same
//! lake.
//!
//! This is a departure from the rule that library code does no blocking I/O. It is confined to this module, and ends
//! when `AsyncIo` gains the three operations: only the bodies of [`sync`], [`truncate`] and [`remove`] change.
//!
//! The caller must hold an admitted unit of `host` until the completion has run: a lake takes frames only while a
//! unit is admitted. A completion that cannot be offered (the lake closed meanwhile) is dropped without running, and
//! what it owns is dropped with it.

use std::fs::{self, File};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use euhedral_graph::Lake;
use euhedral_io::{Continuation, OwnedFile};

use crate::resume::ResumeError;
use crate::step::{Kind, step};

/// What a durable call's completion is told.
pub type Done = Continuation<Result<(), ResumeError>>;

/// A file the library shares between the calls in flight on it.
pub type SharedFile = Arc<File>;

/// Takes over a file `AsyncIo` opened, to share it. On Windows the handle was opened for overlapped I/O, as the reads and
/// writes need.
pub fn share(file: OwnedFile) -> SharedFile {
    Arc::new(File::from(file))
}

/// Runs `work` as a frame of `host`, then offers the frame `then` makes of its result to `host`.
fn on_host(
    host: &Arc<Lake>,
    work: impl FnOnce() -> io::Result<()> + Send + 'static,
    then: Done,
) -> Result<(), ResumeError> {
    let next = Arc::clone(host);
    let frame = step(Kind::Durable, move || {
        let completion = then(work().map_err(ResumeError::Io));
        // A refused completion is dropped, and with it what the caller's continuation owned.
        drop(next.offer(completion));
    });
    host.offer(frame).map_err(|_| ResumeError::LakeClosed)
}

/// Makes everything written to `file` so far durable (`fdatasync`, `FlushFileBuffers`).
#[allow(
    clippy::disallowed_methods,
    reason = "no-wait, blocking I/O with no AsyncIo form yet: one call, run as a host lake frame, whose completion continues the chain"
)]
pub fn sync(host: &Arc<Lake>, file: &SharedFile, then: Done) -> Result<(), ResumeError> {
    let file = Arc::clone(file);
    on_host(host, move || file.sync_data(), then)
}

/// Cuts `file` back to `len` bytes, or extends it with zeros to `len`.
#[allow(
    clippy::disallowed_methods,
    reason = "no-wait, blocking I/O with no AsyncIo form yet: one call, run as a host lake frame, whose completion continues the chain"
)]
pub fn truncate(host: &Arc<Lake>, file: &SharedFile, len: u64, then: Done) -> Result<(), ResumeError> {
    let file = Arc::clone(file);
    on_host(host, move || file.set_len(len), then)
}

/// Removes the file at `path`. A file that is already gone is not an error: the removal is as done as it can be. A
/// handle of the file that is still open is the caller's to close first (Windows refuses to remove an open file).
#[allow(
    clippy::disallowed_methods,
    reason = "no-wait, blocking I/O with no AsyncIo form yet: one call, run as a host lake frame, whose completion continues the chain"
)]
pub fn remove(host: &Arc<Lake>, path: &Path, then: Done) -> Result<(), ResumeError> {
    let path: PathBuf = path.to_owned();
    let work = move || match fs::remove_file(&path) {
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        other => other,
    };
    on_host(host, work, then)
}
