//! The asynchronous file steps the checkpoint readers and the writer chain: open, length, positioned read.

use std::fmt;
use std::io::Error as IoError;
use std::path::Path;
use std::sync::Arc;

use euhedral_core::frames::{Frame, FrameError, FrameHeader};
use euhedral_io::{AsyncIo, IoDoneOwned, OpenSpec, OwnedFile, RawFile};

use super::CheckpointError;

/// Where a result goes: called once, from a frame, and must not wait.
pub type Done<T> = Arc<dyn Fn(Result<T, CheckpointError>) + Send + Sync>;

/// What follows a step: the result of one read or open, taken once.
pub(crate) type Next<T> = Box<dyn FnOnce(Result<T, CheckpointError>) + Send>;

/// The id of the frames of this module.
const FRAME_ID: u64 = 0x7175_616e_7463_6b70;

/// A frame that runs a closure once.
struct Work {
    header: FrameHeader,
    work: Option<Box<dyn FnOnce() + Send>>,
}

impl Frame for Work {
    fn header(&self) -> &FrameHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut FrameHeader {
        &mut self.header
    }

    fn execute(&mut self) -> Result<(), FrameError> {
        if let Some(work) = self.work.take() {
            work();
        }
        Ok(())
    }
}

impl fmt::Debug for Work {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("Work").finish_non_exhaustive()
    }
}

/// The frame an `AsyncIo` continuation returns: it runs `work`.
pub(crate) fn work(work: impl FnOnce() + Send + 'static) -> Box<dyn Frame> {
    Box::new(Work {
        header: FrameHeader::new(FRAME_ID),
        work: Some(Box::new(work)),
    })
}

/// A frame that does nothing: the continuation of a step that is not the last of its group.
pub(crate) fn idle() -> Box<dyn Frame> {
    Box::new(Work {
        header: FrameHeader::new(FRAME_ID),
        work: None,
    })
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

/// Opens `path` for reading. `Err` started nothing, and drops `next`.
pub(crate) fn open_read(io: &AsyncIo, path: &Path, next: Next<OwnedFile>) -> Result<(), CheckpointError> {
    let context = path.display().to_string();
    let failed = context.clone();
    io.open_path(
        path,
        OpenSpec::default(),
        Box::new(move |opened| {
            work(move || next(opened.map_err(|error| CheckpointError::io(format!("cannot open {context}"), error))))
        }),
    )
    .map_err(|error| submit_error(&failed, error))
}

/// Reads `len` bytes at `offset` of `file`; a file that ends first is an error. `file` stays open until `next` runs.
/// `Err` started nothing, and drops `next`.
pub(crate) fn read_exact(
    io: &AsyncIo,
    file: Arc<OwnedFile>,
    offset: u64,
    len: usize,
    context: String,
    next: Next<Vec<u8>>,
) -> Result<(), CheckpointError> {
    if len == 0 {
        return Err(CheckpointError::Invalid(format!("{context}: a read of no bytes")));
    }
    let raw = raw_file(&file);
    let failed = context.clone();
    io.read_into(
        raw,
        offset,
        vec![0u8; len],
        Box::new(move |done: IoDoneOwned<Vec<u8>>| {
            work(move || {
                let _open = file;
                next(finish(done, len, &context))
            })
        }),
    )
    .map_err(|error| submit_error(&failed, error))
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
        Box::new(move |opened| match opened {
            Err(error) => then(Err(error)),
            Ok(file) => sized(io_next, Arc::new(file), context, then),
        }),
    )
}

/// The length step of [`read_file`].
fn sized(io: Arc<AsyncIo>, file: Arc<OwnedFile>, context: String, then: Done<Vec<u8>>) {
    let raw = raw_file(&file);
    let report = Arc::clone(&then);
    let failed = context.clone();
    let io_next = Arc::clone(&io);
    let submitted = io.file_len(
        raw,
        Box::new(move |length| {
            work(move || match length {
                Err(error) => then(Err(CheckpointError::io(format!("cannot size {context}"), error))),
                Ok(0) => then(Ok(Vec::new())),
                Ok(length) => match usize::try_from(length) {
                    Err(_) => then(Err(CheckpointError::Invalid(format!(
                        "{context}: {length} bytes do not fit in memory"
                    )))),
                    Ok(length) => {
                        let report = Arc::clone(&then);
                        let next: Next<Vec<u8>> = Box::new(move |bytes| then(bytes));
                        if let Err(error) = read_exact(&io_next, file, 0, length, context, next) {
                            report(Err(error));
                        }
                    }
                },
            })
        }),
    );
    if let Err(error) = submitted {
        report(Err(submit_error(&failed, error)));
    }
}
