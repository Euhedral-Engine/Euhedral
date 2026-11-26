//! The asynchronous file steps the checkpoint readers and the writer chain: open, length, positioned read.
//!
//! A step that the I/O engine or a lake accepted and later discards unrun ends its call with
//! [`CheckpointError::LakeClosed`] ([`Next`], [`Closing`]). RustEE may discard a frame from the tail of a pull, where a
//! worker must not push, so a continuation that is told `LakeClosed` only records or reports the result: it never
//! offers or submits new work (the writer does not offer the removal of its `.part` file then; it says the file was
//! left). A function that returns `Err` started nothing: it disarms its [`Next`] and never calls it.

use std::io::Error as IoError;
use std::path::Path;
use std::sync::Arc;

use euhedral_inference_engine::Handoff;
use euhedral_io::{AsyncIo, IoDoneOwned, OpenSpec, OwnedFile, RawFile};

use super::CheckpointError;
use crate::step::{Kind, step};

/// Where a result goes: called once, from a frame, and must not wait.
pub type Done<T> = Arc<dyn Fn(Result<T, CheckpointError>) + Send + Sync>;

/// The continuation of a call while a chain holds it. A chain that is dropped without ending (a step that the lake or
/// the I/O engine accepted and later discarded unrun, because it is closing) ends the call with
/// [`CheckpointError::LakeClosed`], once. [`Closing::release`] takes the continuation back, for the step that ends the
/// call itself or for the call that returns `Err` and calls nothing.
pub(crate) struct Closing<T>(Option<Done<T>>);

impl<T> Closing<T> {
    pub(crate) fn new(then: Done<T>) -> Self {
        Self(Some(then))
    }

    /// The continuation; the guard is spent, and dropping it says nothing.
    pub(crate) fn release(mut self) -> Option<Done<T>> {
        self.0.take()
    }

    /// Ends the call with `result`.
    pub(crate) fn end(self, result: Result<T, CheckpointError>) {
        if let Some(then) = self.release() {
            then(result);
        }
    }
}

impl<T> Drop for Closing<T> {
    // This runs the caller's continuation inside a drop, on the thread of the lake or engine that discards the step.
    fn drop(&mut self) {
        if let Some(then) = self.0.take() {
            then(Err(CheckpointError::LakeClosed));
        }
    }
}

/// What follows a step: the result of one read or open, taken once. Dropped without being called (the step that
/// carried it was discarded unrun), it is called with [`CheckpointError::LakeClosed`], so the chain it continues ends.
pub(crate) struct Next<T>(Option<Step<T>>);

/// The body of a [`Next`].
type Step<T> = Box<dyn FnOnce(Result<T, CheckpointError>) + Send>;

impl<T> Next<T> {
    pub(crate) fn new(next: impl FnOnce(Result<T, CheckpointError>) + Send + 'static) -> Self {
        Self(Some(Box::new(next)))
    }

    pub(crate) fn call(mut self, result: Result<T, CheckpointError>) {
        if let Some(next) = self.0.take() {
            next(result);
        }
    }

    /// Drops the step without calling it: the call that returns `Err` started nothing.
    fn disarm(mut self) {
        self.0.take();
    }

    /// Hands `start` the cell the submitted step takes `self` from, and starts it. A submission that returns `Err`
    /// dropped the step and says so by the error: `self` is not called then. A step dropped after the submission
    /// returned `Ok` drops the last holder, which calls `self` with `LakeClosed`.
    fn submit(
        self,
        context: &str,
        start: impl FnOnce(Parked<T>) -> Result<(), IoError>,
    ) -> Result<(), CheckpointError> {
        let cell = Handoff::new();
        cell.send(self);
        start(Parked(Arc::clone(&cell))).map_err(|error| {
            if let Some(next) = cell.take() {
                next.disarm();
            }
            submit_error(context, error)
        })
    }
}

impl<T> Drop for Next<T> {
    fn drop(&mut self) {
        if let Some(next) = self.0.take() {
            next(Err(CheckpointError::LakeClosed));
        }
    }
}

/// Where a submitted step finds its [`Next`].
struct Parked<T>(Arc<Handoff<Next<T>>>);

impl<T> Parked<T> {
    fn take(&self) -> Option<Next<T>> {
        self.0.take()
    }
}

#[cfg(unix)]
pub(crate) fn raw_file(file: &OwnedFile) -> RawFile {
    use std::os::fd::AsRawFd;

    file.as_raw_fd()
}

#[cfg(windows)]
pub(crate) fn raw_file(file: &OwnedFile) -> RawFile {
    use std::os::windows::io::AsRawHandle;

    file.as_raw_handle()
}

/// The error a failed submission becomes.
pub(crate) fn submit_error(context: &str, error: IoError) -> CheckpointError {
    CheckpointError::io(context, error)
}

/// Opens `path` for reading. `Err` started nothing, and `next` is never called.
pub(crate) fn open_read(io: &AsyncIo, path: &Path, next: Next<OwnedFile>) -> Result<(), CheckpointError> {
    let context = path.display().to_string();
    let failed = context.clone();
    next.submit(&failed, |parked| {
        io.open_path(
            path,
            OpenSpec::default(),
            Box::new(move |opened| {
                step(Kind::Resolve, move || {
                    let Some(next) = parked.take() else { return };
                    next.call(opened.map_err(|error| CheckpointError::io(format!("cannot open {context}"), error)));
                })
            }),
        )
    })
}

/// Reads `len` bytes at `offset` of `file`; a file that ends first is an error. `file` stays open until `next` runs.
/// `Err` started nothing, and `next` is never called.
pub(crate) fn read_exact(
    io: &AsyncIo,
    file: Arc<OwnedFile>,
    offset: u64,
    len: usize,
    context: String,
    next: Next<Vec<u8>>,
) -> Result<(), CheckpointError> {
    if len == 0 {
        next.disarm();
        return Err(CheckpointError::Invalid(format!("{context}: a read of no bytes")));
    }
    let raw = raw_file(&file);
    let failed = context.clone();
    next.submit(&failed, |parked| {
        io.read_into(
            raw,
            offset,
            vec![0u8; len],
            Box::new(move |done: IoDoneOwned<Vec<u8>>| {
                step(Kind::Resolve, move || {
                    let _open = file;
                    let Some(next) = parked.take() else { return };
                    next.call(finish(done, len, &context));
                })
            }),
        )
    })
}

/// The memory of a finished read of `len` bytes, or why it failed.
pub(crate) fn finish<B>(done: IoDoneOwned<B>, len: usize, context: &str) -> Result<B, CheckpointError> {
    match (done.result, done.buf) {
        (Ok(moved), Some(memory)) if moved == len => Ok(memory),
        (Ok(moved), Some(_)) => Err(CheckpointError::Format(format!(
            "{context}: the file ends after {moved} of {len} bytes"
        ))),
        (Ok(_), None) => Err(CheckpointError::Invalid(format!("{context}: the read was abandoned"))),
        (Err(error), _) => Err(CheckpointError::io(format!("cannot read {context}"), error)),
    }
}

/// Reads a whole file: open, length, read. `then` gets its bytes. `Err` started nothing and never calls `then`.
pub(crate) fn read_file(io: &Arc<AsyncIo>, path: &Path, then: Done<Vec<u8>>) -> Result<(), CheckpointError> {
    let context = path.display().to_string();
    let io_next = Arc::clone(io);
    open_read(
        io,
        path,
        Next::new(move |opened| match opened {
            Err(error) => then(Err(error)),
            Ok(file) => sized(io_next, Arc::new(file), context, then),
        }),
    )
}

/// The length step of [`read_file`].
fn sized(io: Arc<AsyncIo>, file: Arc<OwnedFile>, context: String, then: Done<Vec<u8>>) {
    let report = Arc::clone(&then);
    let io_next = Arc::clone(&io);
    let again = Arc::clone(&file);
    let read_context = context.clone();
    let next = Next::new(move |length| match length {
        Err(error) => then(Err(error)),
        Ok(0) => then(Ok(Vec::new())),
        Ok(length) => match usize::try_from(length) {
            Err(_) => then(Err(CheckpointError::Invalid(format!(
                "{read_context}: {length} bytes do not fit in memory"
            )))),
            Ok(length) => {
                let report = Arc::clone(&then);
                let next = Next::new(move |bytes| then(bytes));
                if let Err(error) = read_exact(&io_next, file, 0, length, read_context, next) {
                    report(Err(error));
                }
            }
        },
    });
    if let Err(error) = file_length(&io, again, context, next) {
        report(Err(error));
    }
}

/// The length of `file`. `file` stays open until `next` runs. `Err` started nothing, and `next` is never called.
pub(crate) fn file_length(
    io: &AsyncIo,
    file: Arc<OwnedFile>,
    context: String,
    next: Next<u64>,
) -> Result<(), CheckpointError> {
    let raw = raw_file(&file);
    let failed = context.clone();
    next.submit(&failed, |parked| {
        io.file_len(
            raw,
            Box::new(move |length| {
                step(Kind::Resolve, move || {
                    let _open = file;
                    let Some(next) = parked.take() else { return };
                    next.call(length.map_err(|error| CheckpointError::io(format!("cannot size {context}"), error)));
                })
            }),
        )
    })
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use euhedral_core::generics::LatticeSource;
    use euhedral_io::AsyncIo;

    use super::*;

    /// A `Next` that counts its calls.
    fn counting<T: 'static>() -> (Arc<AtomicUsize>, Next<T>) {
        let calls = Arc::new(AtomicUsize::new(0));
        let counted = Arc::clone(&calls);
        let next = Next::new(move |_| {
            counted.fetch_add(1, Ordering::SeqCst);
        });
        (calls, next)
    }

    fn closed_io() -> Arc<AsyncIo> {
        #[allow(clippy::unwrap_used, reason = "a test: a failed open stops it")]
        let io = AsyncIo::open().unwrap();
        io.complete();
        io
    }

    #[allow(
        clippy::unwrap_used,
        clippy::disallowed_methods,
        reason = "a test: a failed open stops it, and the file only has to exist"
    )]
    fn some_file() -> Arc<OwnedFile> {
        // An anonymous temporary file: a path under the crate is not there in a sandboxed test run.
        Arc::new(OwnedFile::from(tempfile::tempfile().unwrap()))
    }

    // Mutant caught: a refused submission that leaves its `Next` armed, so the drop calls it with `LakeClosed`.
    #[test]
    fn a_refused_open_length_or_read_returns_err_and_never_calls_next() {
        let io = closed_io();
        let (calls, next) = counting();
        assert!(open_read(&io, Path::new("absent"), next).is_err());
        let (length_calls, next) = counting();
        assert!(file_length(&io, some_file(), "file".to_owned(), next).is_err());
        let (read_calls, next) = counting();
        assert!(read_exact(&io, some_file(), 0, 4, "file".to_owned(), next).is_err());
        let counts = [calls, length_calls, read_calls].map(|calls| calls.load(Ordering::SeqCst));
        assert_eq!(counts, [0, 0, 0]);
    }

    // Mutant caught: an early refusal (a read of no bytes) that drops its `Next` armed.
    #[test]
    fn a_read_of_no_bytes_returns_err_and_never_calls_next() {
        let io = closed_io();
        let (calls, next) = counting();
        assert!(read_exact(&io, some_file(), 0, 0, "file".to_owned(), next).is_err());
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    // Mutant caught: a `Next` or `Closing` dropped uncalled that says nothing (the caller waits for ever), or that
    // runs a second time after it was called.
    #[test]
    fn a_next_dropped_uncalled_is_told_the_closing_error_once() {
        let told = Arc::new(AtomicUsize::new(0));
        let seen = Arc::clone(&told);
        let next = Next::<()>::new(move |result| {
            assert!(matches!(result, Err(CheckpointError::LakeClosed)));
            seen.fetch_add(1, Ordering::SeqCst);
        });
        drop(next);
        assert_eq!(told.load(Ordering::SeqCst), 1);

        let (calls, next) = counting::<()>();
        next.call(Ok(()));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }
}
