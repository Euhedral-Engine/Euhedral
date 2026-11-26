//! The three file operations RustEE's `AsyncIo` does not have: making written data durable, cutting a file back to a
//! length, and removing a file. Until it has them, each is one blocking call, and each runs as a frame of the host
//! lake, so the worker that submitted it never waits: the call's completion is the next frame, offered to the same
//! lake.
//!
//! This is a departure from the rule that library code does no blocking I/O. It is confined to this module, and ends
//! when `AsyncIo` gains the three operations: only the bodies of [`sync`], [`truncate`] and [`remove`] change.
//!
//! [`clash`] is a host-lake step of the same kind (dup, fstat and open calls that compare file identities). It has no
//! `AsyncIo` counterpart to wait for: it needs none, and stays as it is.
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
use same_file::Handle;
use thiserror::Error;

use crate::step::{Kind, step};

/// What a durable call's completion is told: whether the call worked.
pub type Done = Continuation<io::Result<()>>;

/// The host lake would not take the call, because it is closed or has no admitted unit. Nothing was started and the
/// completion was dropped without running.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
#[error("the host lake is closed: the call was not started")]
pub struct LakeClosed;

/// A file the library shares between the calls in flight on it.
pub type SharedFile = Arc<File>;

/// Takes over a file `AsyncIo` opened, to share it. On Windows the handle was opened for overlapped I/O, as the reads and
/// writes need.
pub fn share(file: OwnedFile) -> SharedFile {
    Arc::new(File::from(file))
}

/// Runs `work` as a frame of `host`, then offers the frame `then` makes of its result to `host`.
fn on_host<T: Send + 'static>(
    host: &Arc<Lake>,
    work: impl FnOnce() -> io::Result<T> + Send + 'static,
    then: Continuation<io::Result<T>>,
) -> Result<(), LakeClosed> {
    let next = Arc::clone(host);
    let frame = step(Kind::Durable, move || {
        let completion = then(work());
        // A refused completion is dropped, and with it what the caller's continuation owned.
        drop(next.offer(completion));
    });
    host.offer(frame).map_err(|_| LakeClosed)
}

/// Makes everything written to `file` so far durable (`fdatasync`, `FlushFileBuffers`).
pub fn sync(host: &Arc<Lake>, file: &SharedFile, then: Done) -> Result<(), LakeClosed> {
    sync_and_close(host, Arc::clone(file), then)
}

/// [`sync`] for a caller that gives the call its reference to the file. The call drops that reference before the
/// completion runs, and the caller holds none when the call returns, so if it held the last one the file is closed
/// when `then` runs. The resume file uses it so that a continuation that removes the file finds no handle of the call.
#[expect(
    clippy::disallowed_methods,
    reason = "no-wait, blocking I/O with no AsyncIo form yet: one call, run as a host lake frame, whose completion continues the chain"
)]
pub(crate) fn sync_and_close(host: &Arc<Lake>, file: SharedFile, then: Done) -> Result<(), LakeClosed> {
    on_host(host, move || file.sync_data(), then)
}

/// Cuts `file` back to `len` bytes, or extends it with zeros to `len`.
pub fn truncate(host: &Arc<Lake>, file: &SharedFile, len: u64, then: Done) -> Result<(), LakeClosed> {
    truncate_and_close(host, Arc::clone(file), len, then)
}

/// [`truncate`] for a caller that gives the call its reference to the file: see [`sync_and_close`], which has the same
/// guarantee.
#[expect(
    clippy::disallowed_methods,
    reason = "no-wait, blocking I/O with no AsyncIo form yet: one call, run as a host lake frame, whose completion continues the chain"
)]
pub(crate) fn truncate_and_close(host: &Arc<Lake>, file: SharedFile, len: u64, then: Done) -> Result<(), LakeClosed> {
    on_host(host, move || file.set_len(len), then)
}

/// Removes the file at `path`. A file that is already gone is not an error: the removal is as done as it can be. A
/// handle of the file that is still open does not stop the removal: `AsyncIo` opens files shared for delete, so on
/// Windows the name goes when the last handle closes.
#[expect(
    clippy::disallowed_methods,
    reason = "no-wait, blocking I/O with no AsyncIo form yet: one call, run as a host lake frame, whose completion continues the chain"
)]
pub fn remove(host: &Arc<Lake>, path: &Path, then: Done) -> Result<(), LakeClosed> {
    let path: PathBuf = path.to_owned();
    let work = move || match fs::remove_file(&path) {
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        other => other,
    };
    on_host(host, work, then)
}

/// Which of two names a file is another name of.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Clash {
    /// The file that was open.
    Open,
    /// The path.
    Path,
}

/// Finds whether `open`, or the file at `path` (if there is one), is the same file as any of `sources`, by identity
/// (device and index, volume and index), not by name. `open` is checked first: if it clashes, the answer is
/// [`Clash::Open`].
pub fn clash(
    host: &Arc<Lake>,
    open: SharedFile,
    path: &Path,
    sources: Vec<Arc<OwnedFile>>,
    then: Continuation<io::Result<Option<Clash>>>,
) -> Result<(), LakeClosed> {
    let path: PathBuf = path.to_owned();
    let work = move || {
        let mut known = Vec::with_capacity(sources.len());
        for source in &sources {
            known.push(Handle::from_file(File::from(source.try_clone()?))?);
        }
        if known.contains(&Handle::from_file(open.try_clone()?)?) {
            return Ok(Some(Clash::Open));
        }
        match Handle::from_path(&path) {
            Ok(named) => Ok(known.contains(&named).then_some(Clash::Path)),
            // A name that does not exist, or that cannot be read, is never a source: every source was opened for reading.
            Err(error) if matches!(error.kind(), io::ErrorKind::NotFound | io::ErrorKind::PermissionDenied) => Ok(None),
            Err(error) => Err(error),
        }
    };
    on_host(host, work, then)
}
