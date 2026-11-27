//! `package-tokenizer`: gives an artifact the tokenizer of a checkpoint directory.
//!
//! In place, the artifact's content stays byte for byte and only the section after it is written; with an output
//! path, the content is copied there first and the artifact is left alone. The files are checked first: the
//! tokenizer must load and the chat template must be the one the server renders. Nothing is written before the
//! checks pass.
//!
//! The run is a chain of `AsyncIo` operations and host-lake frames: each completion starts the next step, and the
//! final result goes to the callback. The section's body is written, made durable, and only then the footer, so an
//! interrupted run leaves no section, and running it again repairs the artifact.
//!
//! How a run ends, and what it never does while it ends:
//! - A call that is refused at once returns `Err` and never calls `done`: an I/O engine that is complete, a lake that
//!   will not admit the run or take its first frame.
//! - A step that `AsyncIo` refuses once the run is going ends the run with the refusal (the I/O error), through the
//!   run's own failure path, once. That path runs on a worker, so it may remove the temporary output.
//! - A step that the I/O engine or the host lake accepted and later discards unrun (it is closing) drops the run.
//!   RustEE may discard a frame from the tail of a pull, where a worker must not push, so dropping the run offers
//!   nothing: not the removal of the temporary output either. `done` is called once with [`PackageError::Lost`], and
//!   with [`PackageError::PartLeft`] when a temporary output was begun, which says the file was left, or
//!   [`PackageError::Cut`] when an artifact written in place had been cut back.
//! - A step that the host lake refuses once the run is going (a removal, a cut, a flush) drops the run the same way.

use std::fs::File;
use std::io;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use euhedral_core::generics::LatticeSource;
use euhedral_graph::{Lake, UnitGuard};
use euhedral_inference_edrl::{FOOTER_BYTES, HEADER_BYTES, Header, TensorTable};
use euhedral_inference_engine::Handoff;
use euhedral_io::{AsyncIo, Create, IoBuf, OpenSpec, RawFile};

use crate::durable::{self, SharedFile};
use crate::package_error::PackageError;
use crate::step::{Kind, idle, step};
use crate::tokenizer_files::TokenizerFiles;

/// The most the copy of an artifact's content moves in one read and write.
pub const DEFAULT_COPY_CHUNK_BYTES: usize = 16 << 20;

/// The suffix of the file an output is written to until it is complete.
pub const PART_SUFFIX: &str = ".part";

/// What a run packages, and how.
#[derive(Debug, Clone)]
pub struct PackageRequest {
    /// The artifact that gains the tokenizer.
    pub artifact: PathBuf,
    /// The checkpoint directory holding the tokenizer files.
    pub checkpoint: PathBuf,
    /// Where the packaged artifact goes; `None` packages the artifact in place.
    pub out: Option<PathBuf>,
    /// The most bytes one read or write of the copy moves (with an output path).
    pub copy_chunk_bytes: usize,
}

impl PackageRequest {
    /// A request to package `artifact` in place with the tokenizer of `checkpoint`.
    pub fn in_place(artifact: PathBuf, checkpoint: PathBuf) -> PackageRequest {
        PackageRequest {
            artifact,
            checkpoint,
            out: None,
            copy_chunk_bytes: DEFAULT_COPY_CHUNK_BYTES,
        }
    }

    /// A request to write `out`, the content of `artifact` followed by the tokenizer of `checkpoint`.
    pub fn to_output(artifact: PathBuf, checkpoint: PathBuf, out: PathBuf) -> PackageRequest {
        PackageRequest {
            out: Some(out),
            ..PackageRequest::in_place(artifact, checkpoint)
        }
    }
}

/// A finished run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Packaged {
    /// The artifact that carries the tokenizer: the output, or the artifact itself.
    pub target: PathBuf,
    /// The bytes of content before the section.
    pub content_bytes: u64,
    /// The bytes of the section, footer included.
    pub section_bytes: u64,
}

/// Receives the result of a run, once. It usually runs on a worker, but it may also run inside the discard of a frame
/// or an I/O continuation (a closing engine or lake), or on the caller's thread when the first frame is discarded
/// before [`package_tokenizer`] returns. It must only record the result and offer nothing.
pub type Done = Box<dyn FnOnce(Result<Packaged, PackageError>) + Send>;

/// The header's regions against a file of `file_len` bytes: the header, and where the table is.
pub fn table_range(header_bytes: &[u8], file_len: u64) -> Result<(Header, Range<u64>), PackageError> {
    let bytes: &[u8; HEADER_BYTES] = header_bytes
        .try_into()
        .map_err(|_| PackageError::Format("the header is truncated"))?;
    let header = Header::parse(bytes)?;
    header.check_extent(file_len)?;
    let range = header.table_range();
    Ok((header, range))
}

/// Where the artifact's content ends, from its format alone: the end of its last payload, or the start of the data
/// section when it has no payload. A tokenizer section, whole or damaged, does not move it, so a section that was
/// cut short is replaced. `table` is the bytes of [`table_range`]; `file_len` bounds the payloads.
pub fn content_end(header: &Header, table: &[u8], file_len: u64) -> Result<u64, PackageError> {
    let table = TensorTable::parse(table, header, file_len)?;
    Ok(table
        .entries()
        .iter()
        .map(|entry| entry.range().end)
        .fold(header.data_offset(), u64::max))
}

/// Starts packaging. The result goes to `done`, which may run inside a discard and must only record it (see [`Done`]);
/// an `Err` here means nothing started, and `done` is never called.
///
/// `io` does the reads and writes; `host` runs the checks (loading the tokenizer is CPU work) and the few file
/// operations `AsyncIo` lacks.
pub fn package_tokenizer(
    io: Arc<AsyncIo>,
    host: Arc<Lake>,
    request: PackageRequest,
    done: Done,
) -> Result<(), PackageError> {
    if request.copy_chunk_bytes == 0 {
        return Err(PackageError::Io {
            what: "invalid request",
            source: io::Error::new(io::ErrorKind::InvalidInput, "the copy chunk must be at least 1 byte"),
        });
    }
    if let Some(out) = &request.out
        && part_path(out) == request.artifact
    {
        return Err(PackageError::Io {
            what: "invalid request",
            source: io::Error::new(
                io::ErrorKind::InvalidInput,
                "the artifact is the temporary name of the output: the run would overwrite its own source",
            ),
        });
    }
    if io.is_complete() {
        return Err(PackageError::Io {
            what: "cannot start",
            source: io::Error::other("the I/O engine is closed"),
        });
    }
    let unit = host.admit().map_err(|error| PackageError::Lake(error.to_string()))?;
    let job = Job {
        io,
        host: Arc::clone(&host),
        request,
        end: RunEnd {
            host: Arc::clone(&host),
            part: None,
            completion: Some(Completion { done: Some(done) }),
            cut: false,
        },
        _unit: unit,
    };
    // The frame is the job's only holder once it is accepted. A lake that refuses it drops the frame, so the job waits
    // in a cell: it comes back, spent, and the call returns `Err` without calling `done`.
    let parked = Handoff::new();
    parked.send(job);
    let taken = Arc::clone(&parked);
    let run = step(Kind::Resolve, move || {
        if let Some(job) = taken.take() {
            job.read_checkpoint();
        }
    });
    if host.offer(run).is_err() {
        if let Some(mut job) = parked.take() {
            job.end.abandon();
        }
        return Err(PackageError::Lake(
            "the host lake refused the run's first step".to_owned(),
        ));
    }
    Ok(())
}

/// The callback of a run, called once. A run dropped before it finished (a step discarded unrun) reports itself lost,
/// from inside the discard.
struct Completion {
    done: Option<Done>,
}

impl Completion {
    /// Spends the callback without calling it: the call that returns `Err` started nothing.
    fn disarm(mut self) {
        self.done.take();
    }

    fn finish(mut self, result: Result<Packaged, PackageError>) {
        if let Some(done) = self.done.take() {
            done(result);
        }
    }
}

impl Drop for Completion {
    fn drop(&mut self) {
        if let Some(done) = self.done.take() {
            done(Err(PackageError::Lost));
        }
    }
}

struct Job {
    io: Arc<AsyncIo>,
    host: Arc<Lake>,
    request: PackageRequest,
    /// The run's end: the callback, and the temporary output to remove before it is called.
    end: RunEnd,
    _unit: UnitGuard,
}

/// How a run ends. It owns the callback and, once there is one, the temporary output, so that a failure calls the
/// callback only after the output is removed ([`RunEnd::remove_then_fail`]), and a caller that starts again never has
/// its new `.part` deleted by the old run.
///
/// A run that is dropped without ending (a step the I/O engine or the lake accepted and discarded unrun) does not
/// remove the output: the discard may be inside a pull, where nothing may be offered. It says what was left.
struct RunEnd {
    host: Arc<Lake>,
    part: Option<PathBuf>,
    completion: Option<Completion>,
    /// An artifact written in place was cut back: until its section lands it has none.
    cut: bool,
}

impl RunEnd {
    fn arm(&mut self, part: PathBuf) {
        self.part = Some(part);
    }

    fn disarm(&mut self) {
        self.part = None;
    }

    /// Spends the run without reporting: the call that returns `Err` started nothing.
    fn abandon(&mut self) {
        self.part = None;
        if let Some(completion) = self.completion.take() {
            completion.disarm();
        }
    }

    /// Reports the end of the run; the output, if armed, was moved into place or is not this run's to remove.
    fn finish(&mut self, result: Result<Packaged, PackageError>) {
        self.part = None;
        if let Some(completion) = self.completion.take() {
            completion.finish(result);
        }
    }

    /// Removes the armed output, then reports `error`. A removal the lake refuses, or accepts and discards, drops
    /// [`Failing`], which reports all the same.
    fn remove_then_fail(&mut self, error: PackageError) {
        let part = self.part.take();
        let Some(completion) = self.completion.take() else {
            return;
        };
        let Some(part) = part else {
            // Nothing to remove. An artifact that was already cut has no section: the error says so.
            let error = if self.cut {
                PackageError::Cut(Box::new(error))
            } else {
                error
            };
            completion.finish(Err(error));
            return;
        };
        let failing = Failing {
            part: part.clone(),
            completion: Some(completion),
            error: Some(error),
        };
        let _ = durable::remove(
            &self.host,
            &part,
            Box::new(move |removed| {
                failing.report(removed);
                idle()
            }),
        );
    }
}

impl Drop for RunEnd {
    // This runs inside the discard of a frame or of an I/O continuation: it reports and offers nothing.
    fn drop(&mut self) {
        let Some(completion) = self.completion.take() else {
            return;
        };
        let error = match self.part.take() {
            Some(part) => PackageError::PartLeft {
                part,
                removal: io::Error::other("the run was closed before the file could be removed"),
                cause: Box::new(PackageError::Lost),
            },
            None if self.cut => PackageError::Cut(Box::new(PackageError::Lost)),
            None => PackageError::Lost,
        };
        completion.finish(Err(error));
    }
}

/// A failure on its way out: the error reaches the callback when the removal ends, or, if the removal is refused and its
/// frame dropped, when this is dropped, so the real error is never replaced by `Lost`. A removal that failed is told
/// with the error, naming the file that was left.
struct Failing {
    part: PathBuf,
    completion: Option<Completion>,
    error: Option<PackageError>,
}

impl Failing {
    fn report(mut self, removed: io::Result<()>) {
        let (Some(completion), Some(error)) = (self.completion.take(), self.error.take()) else {
            return;
        };
        completion.finish(Err(match removed {
            Ok(()) => error,
            Err(removal) => PackageError::PartLeft {
                part: self.part.clone(),
                removal,
                cause: Box::new(error),
            },
        }));
    }
}

impl Drop for Failing {
    fn drop(&mut self) {
        if let (Some(completion), Some(error)) = (self.completion.take(), self.error.take()) {
            completion.finish(Err(PackageError::PartLeft {
                part: self.part.clone(),
                removal: io::Error::other("the removal was not run: the host lake is closed"),
                cause: Box::new(error),
            }));
        }
    }
}

/// What the checkpoint holds.
struct Checkpoint {
    tokenizer: Vec<u8>,
    tokenizer_config: Vec<u8>,
    chat_template: Option<Vec<u8>>,
    generation_config: Vec<u8>,
}

/// Runs `work` as a frame of `host`; a lake that refuses it drops `work`, which the run reports as lost (no output
/// is begun where this is used, so nothing is left).
fn on_host(host: &Arc<Lake>, work: impl FnOnce() + Send + 'static) {
    drop(host.offer(step(Kind::Resolve, work)));
}

type Then<T> = Box<dyn FnOnce(io::Result<T>) + Send>;

/// Starts an I/O step that owns `held`. `start` gets the cell the step takes `held` from. A submission that `AsyncIo`
/// refuses dropped the step, so `held` is given back to `refused` with the error, on this thread, which is not a
/// discard: it may fail the run through its own path. A step dropped after it was accepted drops the last holder of the
/// cell and, with it, `held`.
fn submit<H: Send + 'static>(
    held: H,
    refused: impl FnOnce(H, io::Error),
    start: impl FnOnce(Arc<Handoff<H>>) -> io::Result<()>,
) {
    let cell = Handoff::new();
    cell.send(held);
    if let Err(error) = start(Arc::clone(&cell))
        && let Some(held) = cell.take()
    {
        refused(held, error);
    }
}

#[cfg(unix)]
fn raw_file(file: &File) -> RawFile {
    use std::os::fd::AsRawFd;

    file.as_raw_fd()
}

#[cfg(windows)]
fn raw_file(file: &File) -> RawFile {
    use std::os::windows::io::AsRawHandle;

    file.as_raw_handle()
}

/// Opens `path`; a submission the I/O refuses calls `then` with the error.
fn open(io: &AsyncIo, path: &Path, spec: OpenSpec, then: Then<SharedFile>) {
    submit(
        then,
        |then, error| then(Err(error)),
        |parked| {
            io.open_path(
                path,
                spec,
                Box::new(move |result| {
                    if let Some(then) = parked.take() {
                        then(result.map(durable::share));
                    }
                    idle()
                }),
            )
        },
    );
}

/// Reads exactly `len` bytes at `offset` into the end of `got`.
fn read_exact(io: &Arc<AsyncIo>, file: SharedFile, offset: u64, len: usize, mut got: Vec<u8>, then: Then<Vec<u8>>) {
    if got.len() >= len {
        then(Ok(got));
        return;
    }
    let want = len - got.len();
    let at = offset + got.len() as u64;
    let next = (Arc::clone(io), Arc::clone(&file));
    submit(
        then,
        |then, error| then(Err(error)),
        |parked| {
            io.read(
                raw_file(&file),
                at,
                IoBuf {
                    bytes: vec![0; want],
                    filled: 0,
                },
                Box::new(move |done| {
                    let Some(then) = parked.take() else {
                        return idle();
                    };
                    match done.result {
                        Ok(0) => then(Err(io::ErrorKind::UnexpectedEof.into())),
                        Ok(moved) if got.is_empty() && moved == want => {
                            // The first read filled the request: the buffer is the result, no copy.
                            then(Ok(done.buf.bytes));
                        }
                        Ok(moved) => {
                            got.extend_from_slice(&done.buf.bytes[..moved]);
                            read_exact(&next.0, next.1, offset, len, got, then);
                        }
                        Err(error) => then(Err(error)),
                    }
                    idle()
                }),
            )
        },
    );
}

/// Writes all of `bytes` at `offset`.
fn write_all(io: &Arc<AsyncIo>, file: SharedFile, offset: u64, bytes: Vec<u8>, then: Then<()>) {
    if bytes.is_empty() {
        then(Ok(()));
        return;
    }
    let next = (Arc::clone(io), Arc::clone(&file));
    submit(
        then,
        |then, error| then(Err(error)),
        |parked| {
            io.write(
                raw_file(&file),
                offset,
                IoBuf {
                    filled: bytes.len(),
                    bytes,
                },
                Box::new(move |mut done| {
                    let Some(then) = parked.take() else {
                        return idle();
                    };
                    match done.result {
                        Ok(0) => then(Err(io::ErrorKind::WriteZero.into())),
                        Ok(moved) => {
                            let rest = done.buf.bytes.split_off(moved);
                            write_all(&next.0, next.1, offset + moved as u64, rest, then);
                        }
                        Err(error) => then(Err(error)),
                    }
                    idle()
                }),
            )
        },
    );
}

/// The whole file at `path`.
fn read_all(io: &Arc<AsyncIo>, path: &Path, then: Then<Vec<u8>>) {
    let again = Arc::clone(io);
    open(
        io,
        path,
        OpenSpec::default(),
        Box::new(move |opened| match opened {
            Err(error) => then(Err(error)),
            Ok(file) => {
                let reader = Arc::clone(&again);
                let held = Arc::clone(&file);
                submit(
                    then,
                    |then, error| then(Err(error)),
                    |parked| {
                        again.file_len(
                            raw_file(&file),
                            Box::new(move |length| {
                                let Some(then) = parked.take() else {
                                    return idle();
                                };
                                match length.and_then(|length| usize::try_from(length).map_err(io::Error::other)) {
                                    Ok(length) => {
                                        read_exact(&reader, held, 0, length, Vec::with_capacity(length), then);
                                    }
                                    Err(error) => then(Err(error)),
                                }
                                idle()
                            }),
                        )
                    },
                );
            }
        }),
    );
}

/// Copies the bytes `from..end` of `source` to the same offsets of `target`, `chunk` bytes at a time.
fn copy(io: &Arc<AsyncIo>, source: SharedFile, target: SharedFile, from: u64, end: u64, chunk: usize, then: Then<()>) {
    if from >= end {
        then(Ok(()));
        return;
    }
    let len = usize::try_from(end - from).map_or(chunk, |left| left.min(chunk));
    let again = (Arc::clone(io), Arc::clone(&source), Arc::clone(&target));
    read_exact(
        io,
        source,
        from,
        len,
        Vec::with_capacity(len),
        Box::new(move |read| match read {
            Err(error) => then(Err(error)),
            Ok(bytes) => {
                let writer = Arc::clone(&again.0);
                write_all(
                    &writer,
                    Arc::clone(&again.2),
                    from,
                    bytes,
                    Box::new(move |written| match written {
                        Err(error) => then(Err(error)),
                        Ok(()) => copy(&again.0, again.1, again.2, from + len as u64, end, chunk, then),
                    }),
                );
            }
        }),
    );
}

impl Job {
    fn fail(mut self, error: PackageError) {
        self.end.finish(Err(error));
    }

    /// Reads `name` of the checkpoint; a missing file is `None` when `optional`.
    fn read_checkpoint_file(
        self,
        name: &'static str,
        optional: bool,
        next: impl FnOnce(Job, Option<Vec<u8>>) + Send + 'static,
    ) {
        let io = Arc::clone(&self.io);
        let path = self.request.checkpoint.join(name);
        read_all(
            &io,
            &path,
            Box::new(move |read| match read {
                Ok(bytes) => next(self, Some(bytes)),
                Err(error) if optional && error.kind() == io::ErrorKind::NotFound => next(self, None),
                Err(error) => self.fail(PackageError::Io {
                    what: name,
                    source: error,
                }),
            }),
        );
    }

    fn read_checkpoint(self) {
        self.read_checkpoint_file("tokenizer.json", false, |job, tokenizer| {
            job.read_checkpoint_file("tokenizer_config.json", false, |job, tokenizer_config| {
                job.read_checkpoint_file("generation_config.json", false, |job, generation_config| {
                    job.read_checkpoint_file("chat_template.jinja", true, |job, chat_template| {
                        job.check(Checkpoint {
                            tokenizer: tokenizer.unwrap_or_default(),
                            tokenizer_config: tokenizer_config.unwrap_or_default(),
                            chat_template,
                            generation_config: generation_config.unwrap_or_default(),
                        });
                    });
                });
            });
        });
    }

    /// Builds and checks the section on the host lake: loading the tokenizer is CPU work.
    fn check(self, checkpoint: Checkpoint) {
        let host = Arc::clone(&self.host);
        on_host(&host, move || {
            let section = TokenizerFiles::from_checkpoint(
                checkpoint.tokenizer,
                checkpoint.tokenizer_config,
                checkpoint.chat_template,
                &checkpoint.generation_config,
            )
            .and_then(|files| files.encode_checked());
            match section {
                Ok(section) => self.read_artifact(section),
                Err(error) => self.fail(error),
            }
        });
    }

    /// Finds where the artifact's content ends, from its header and table.
    fn read_artifact(self, section: Vec<u8>) {
        let io = Arc::clone(&self.io);
        let path = self.request.artifact.clone();
        open(
            &io,
            &path,
            OpenSpec::default(),
            Box::new(move |opened| match opened {
                Err(error) => self.fail(PackageError::Io {
                    what: "cannot open the artifact",
                    source: error,
                }),
                Ok(file) => self.read_header(file, section),
            }),
        );
    }

    fn read_header(self, file: SharedFile, section: Vec<u8>) {
        let io = Arc::clone(&self.io);
        let held = Arc::clone(&file);
        let size_failure = |error| PackageError::Io {
            what: "cannot read the artifact's size",
            source: error,
        };
        submit(
            self,
            |job, error| job.fail(size_failure(error)),
            |parked| {
                io.file_len(
                    raw_file(&file),
                    Box::new(move |length| {
                        let Some(job) = parked.take() else {
                            return idle();
                        };
                        match length {
                            Err(error) => job.fail(size_failure(error)),
                            Ok(length) if length < HEADER_BYTES as u64 => {
                                job.fail(PackageError::Format("artifact is truncated before the header"));
                            }
                            Ok(length) => {
                                let io = Arc::clone(&job.io);
                                let held_again = Arc::clone(&held);
                                read_exact(
                                    &io,
                                    held,
                                    0,
                                    HEADER_BYTES,
                                    Vec::new(),
                                    Box::new(move |header| match header {
                                        Err(error) => job.fail(PackageError::Io {
                                            what: "cannot read the artifact's header",
                                            source: error,
                                        }),
                                        Ok(header) => job.read_table(held_again, header, length, section),
                                    }),
                                );
                            }
                        }
                        idle()
                    }),
                )
            },
        );
    }

    fn read_table(self, file: SharedFile, header_bytes: Vec<u8>, file_len: u64, section: Vec<u8>) {
        let (header, range) = match table_range(&header_bytes, file_len) {
            Ok(found) => found,
            Err(error) => return self.fail(error),
        };
        let Ok(table_len) = usize::try_from(range.end - range.start) else {
            return self.fail(PackageError::Format("the tensor table is too large"));
        };
        let io = Arc::clone(&self.io);
        let source = Arc::clone(&file);
        read_exact(
            &io,
            file,
            range.start,
            table_len,
            Vec::with_capacity(table_len),
            Box::new(move |table| match table {
                Err(error) => self.fail(PackageError::Io {
                    what: "cannot read the artifact's table",
                    source: error,
                }),
                Ok(table) => match content_end(&header, &table, file_len) {
                    Ok(end) => self.write(source, end, section),
                    Err(error) => self.fail(error),
                },
            }),
        );
    }

    /// The first write: nothing before this point has changed a file.
    fn write(self, source: SharedFile, content_end: u64, section: Vec<u8>) {
        match self.request.out.clone() {
            None => {
                drop(source);
                let artifact = self.request.artifact.clone();
                self.open_artifact_for_writing(artifact, content_end, section);
            }
            Some(out) => {
                let part = part_path(&out);
                let io = Arc::clone(&self.io);
                // Opened without truncating: the name may be another name of the source, which the run must not cut.
                // Read as well, so that the identity of the file can be queried on every platform.
                let spec = OpenSpec {
                    read: true,
                    write: true,
                    create: Create::IfMissing,
                    ..OpenSpec::default()
                };
                open(
                    &io,
                    &part.clone(),
                    spec,
                    Box::new(move |opened| match opened {
                        Err(error) => self.fail(PackageError::Io {
                            what: "cannot create the output",
                            source: error,
                        }),
                        Ok(target) => self.start_output(source, target, part, out, content_end, section),
                    }),
                );
            }
        }
    }

    /// The temporary output is open: it is refused if it is the source itself, else cut to empty and filled.
    fn start_output(
        self,
        source: SharedFile,
        target: SharedFile,
        part: PathBuf,
        out: PathBuf,
        content_end: u64,
        section: Vec<u8>,
    ) {
        let mut job = self;
        // The file is this run's to remove unless it is the source itself.
        job.end.arm(part.clone());
        match same_file(&source, &target) {
            Ok(false) => {}
            Ok(true) => {
                job.end.disarm();
                return job.fail(PackageError::Io {
                    what: "invalid request",
                    source: io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "the artifact is the temporary name of the output (the same file): the run would overwrite its own source",
                    ),
                });
            }
            Err(error) => {
                return job.end.remove_then_fail(PackageError::Io {
                    what: "cannot compare the output with the artifact",
                    source: error,
                });
            }
        }
        let host = Arc::clone(&job.host);
        let file = Arc::clone(&target);
        let written = Written::Part {
            part: part.clone(),
            out: out.clone(),
        };
        // A stale `.part` of an earlier run goes: the copy starts from an empty file.
        let _ = durable::truncate(
            &host,
            &file,
            0,
            Box::new(move |cut| {
                match cut {
                    Ok(()) => job.copy_content(source, target, part, out, content_end, section),
                    Err(error) => job.fail_written(
                        &written,
                        PackageError::Io {
                            what: "cannot empty the output",
                            source: error,
                        },
                    ),
                }
                idle()
            }),
        );
    }

    fn open_artifact_for_writing(self, artifact: PathBuf, content_end: u64, section: Vec<u8>) {
        let io = Arc::clone(&self.io);
        let spec = OpenSpec {
            read: false,
            write: true,
            ..OpenSpec::default()
        };
        open(
            &io,
            &artifact.clone(),
            spec,
            Box::new(move |opened| match opened {
                Err(error) => self.fail(PackageError::Io {
                    what: "cannot open the artifact for writing",
                    source: error,
                }),
                Ok(target) => {
                    // The artifact is cut back to its content, so a stale tail or a section that was cut short goes.
                    let host = Arc::clone(&self.host);
                    let file = Arc::clone(&target);
                    let _ = durable::truncate(
                        &host,
                        &file,
                        content_end,
                        Box::new(move |cut| {
                            match cut {
                                Ok(()) => {
                                    let mut job = self;
                                    job.end.cut = true;
                                    job.write_section(target, Written::InPlace(artifact), content_end, section);
                                }
                                Err(error) => self.fail(PackageError::Io {
                                    what: "cannot cut the artifact to its content",
                                    source: error,
                                }),
                            }
                            idle()
                        }),
                    );
                    // A lake that refuses the cut drops the continuation, and the run reports itself lost.
                }
            }),
        );
    }

    fn copy_content(
        self,
        source: SharedFile,
        target: SharedFile,
        part: PathBuf,
        out: PathBuf,
        content_end: u64,
        section: Vec<u8>,
    ) {
        let io = Arc::clone(&self.io);
        let chunk = self.request.copy_chunk_bytes;
        let held = Arc::clone(&target);
        let written = Written::Part { part, out };
        copy(
            &io,
            source,
            target,
            0,
            content_end,
            chunk,
            Box::new(move |copied| match copied {
                Ok(()) => self.write_section(held, written, content_end, section),
                Err(error) => self.fail_written(
                    &written,
                    PackageError::Io {
                        what: "cannot copy the artifact's content",
                        source: error,
                    },
                ),
            }),
        );
    }

    /// The body, then a barrier, then the footer, then a barrier: the footer is the last byte to land, so a run that
    /// stops earlier leaves no section.
    fn write_section(self, target: SharedFile, written: Written, content_end: u64, mut section: Vec<u8>) {
        let footer = section.split_off(section.len() - FOOTER_BYTES);
        let body = section;
        let body_bytes = body.len() as u64;
        let io = Arc::clone(&self.io);
        let again = Arc::clone(&target);
        let synced = Arc::clone(&target);
        write_all(
            &io,
            target,
            content_end,
            body,
            Box::new(move |wrote| match wrote {
                Err(error) => self.fail_written(&written, write_failure(error)),
                Ok(()) => self.barrier(&synced, &written.clone(), move |job, written| {
                    let io = Arc::clone(&job.io);
                    let synced_footer = Arc::clone(&again);
                    write_all(
                        &io,
                        again,
                        content_end + body_bytes,
                        footer,
                        Box::new(move |wrote| match wrote {
                            Err(error) => job.fail_written(&written, write_failure(error)),
                            Ok(()) => job.barrier(&synced_footer, &written.clone(), move |job, written| {
                                job.finish(written, content_end, body_bytes + FOOTER_BYTES as u64);
                            }),
                        }),
                    );
                }),
            }),
        );
    }

    /// Makes what was written to `file` durable, then continues.
    fn barrier(self, file: &SharedFile, written: &Written, next: impl FnOnce(Job, Written) + Send + 'static) {
        let host = Arc::clone(&self.host);
        let written = written.clone();
        let _ = durable::sync(
            &host,
            file,
            Box::new(move |synced| {
                match synced {
                    Ok(()) => next(self, written),
                    Err(error) => self.fail_written(
                        &written,
                        PackageError::Io {
                            what: "cannot make the artifact durable",
                            source: error,
                        },
                    ),
                }
                idle()
            }),
        );
    }

    /// An output is complete: it takes its name; an artifact written in place is.
    fn finish(mut self, written: Written, content_bytes: u64, section_bytes: u64) {
        match written {
            Written::InPlace(artifact) => self.end.finish(Ok(Packaged {
                target: artifact,
                content_bytes,
                section_bytes,
            })),
            Written::Part { part, out } => {
                let io = Arc::clone(&self.io);
                let target = out.clone();
                let cleanup = Written::Part {
                    part: part.clone(),
                    out: out.clone(),
                };
                let move_failure = |error| PackageError::Io {
                    what: "cannot move the output into place",
                    source: error,
                };
                let refused_cleanup = cleanup.clone();
                submit(
                    self,
                    |job, error| job.fail_written(&refused_cleanup, move_failure(error)),
                    |parked| {
                        io.rename(
                            &part,
                            &out,
                            Box::new(move |renamed| {
                                let Some(mut job) = parked.take() else {
                                    return idle();
                                };
                                match renamed {
                                    Ok(()) => {
                                        job.end.disarm();
                                        job.end.finish(Ok(Packaged {
                                            target,
                                            content_bytes,
                                            section_bytes,
                                        }));
                                    }
                                    Err(error) => job.fail_written(&cleanup, move_failure(error)),
                                }
                                idle()
                            }),
                        )
                    },
                );
            }
        }
    }

    /// Fails the run; an output that was begun is removed first, so a failed run leaves none. An artifact written in
    /// place was cut before its section was written: the error says so.
    fn fail_written(mut self, written: &Written, error: PackageError) {
        match written {
            Written::InPlace(_) => self.fail(PackageError::Cut(Box::new(error))),
            Written::Part { .. } => self.end.remove_then_fail(error),
        }
    }
}

fn write_failure(error: io::Error) -> PackageError {
    PackageError::Io {
        what: "cannot write the tokenizer section",
        source: error,
    }
}

/// What a run is writing.
#[derive(Debug, Clone)]
enum Written {
    /// The artifact itself.
    InPlace(PathBuf),
    /// An output, under its temporary name until it is complete.
    Part { part: PathBuf, out: PathBuf },
}

/// Whether both handles are the same file: the same device and index (the volume and file index on Windows), however
/// the paths that opened them are spelled.
fn same_file(first: &File, second: &File) -> io::Result<bool> {
    let first = same_file::Handle::from_file(first.try_clone()?)?;
    let second = same_file::Handle::from_file(second.try_clone()?)?;
    Ok(first == second)
}

fn part_path(out: &Path) -> PathBuf {
    let mut name = out.as_os_str().to_owned();
    name.push(PART_SUFFIX);
    PathBuf::from(name)
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test: a failed unwrap is the failure"
)]
#[allow(
    clippy::disallowed_methods,
    reason = "no-wait, waiting call: the test thread waits on the hand-off for the run it started, and writes its files"
)]
#[allow(
    clippy::disallowed_types,
    reason = "a channel carries the run's result to the waiting test thread"
)]
mod tests {
    use std::fs;
    use std::num::NonZeroUsize;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::mpsc;
    use std::time::{Duration, Instant};

    use euhedral_core::control_plane::ControlPlaneLattice;
    use euhedral_inference_engine::lattice_config;
    use euhedral_inference_settings::{ModelSettings, Settings};

    use super::*;

    // Mutant caught: a run dropped inside a discard that offers the removal of its output to the lake (a push where a
    // worker must not push), or that calls `done` more than once or without saying the file was left.
    #[test]
    fn a_dropped_run_offers_no_removal_and_says_its_output_was_left() {
        let lattice =
            ControlPlaneLattice::new(lattice_config(&Settings::with_model(ModelSettings::default()))).unwrap();
        let host = Lake::new(
            Arc::clone(&lattice),
            NonZeroUsize::new(2).unwrap(),
            Duration::from_secs(1),
        );
        let _unit = host.admit().unwrap();
        let directory = tempfile::tempdir().unwrap();
        let part = directory.path().join("out.edrl.part");
        fs::write(&part, b"partial").unwrap();

        let (sender, receiver) = mpsc::channel();
        let mut end = RunEnd {
            host: Arc::clone(&host),
            part: None,
            cut: false,
            completion: Some(Completion {
                done: Some(Box::new(move |result| sender.send(result).unwrap())),
            }),
        };
        end.arm(part.clone());
        drop(end);

        // `done` was called inside the drop: the answer is already there, and nothing was offered to the lake that
        // could still remove the file.
        let result = receiver.try_recv().expect("the drop reports at once");
        assert!(
            matches!(&result, Err(PackageError::PartLeft { part: left, cause, .. })
                if *left == part && matches!(**cause, PackageError::Lost)),
            "{result:?}"
        );
        assert!(receiver.try_recv().is_err(), "the run reports once");
        host.complete_gracefully();
        lattice.close();
        assert!(part.exists(), "the drop removed the output");
    }

    /// A job holding the armed `.part` file `part`, and the channel its `done` reports into.
    fn armed_job(
        host: &Arc<Lake>,
        io: Arc<AsyncIo>,
        part: &Path,
    ) -> (Job, mpsc::Receiver<Result<Packaged, PackageError>>) {
        let (sender, receiver) = mpsc::channel();
        let mut end = RunEnd {
            host: Arc::clone(host),
            part: None,
            cut: false,
            completion: Some(Completion {
                done: Some(Box::new(move |result| sender.send(result).unwrap())),
            }),
        };
        end.arm(part.to_owned());
        let job = Job {
            io,
            host: Arc::clone(host),
            request: PackageRequest::in_place(PathBuf::from("artifact"), PathBuf::from("checkpoint")),
            end,
            _unit: host.admit().unwrap(),
        };
        (job, receiver)
    }

    // Mutant caught: a host frame or an accepted I/O step that is discarded unrun and offers the removal of the output
    // (a push inside the discard), or that ends the run twice or not at all.
    #[test]
    fn a_host_frame_or_an_io_step_discarded_unrun_ends_the_run_once_and_offers_nothing() {
        let lattice =
            ControlPlaneLattice::new(lattice_config(&Settings::with_model(ModelSettings::default()))).unwrap();
        let host = Lake::new(
            Arc::clone(&lattice),
            NonZeroUsize::new(2).unwrap(),
            Duration::from_secs(1),
        );
        let io = AsyncIo::open().unwrap();
        let directory = tempfile::tempdir().unwrap();
        let part = directory.path().join("out.edrl.part");
        fs::write(&part, b"partial").unwrap();

        let (job, receiver) = armed_job(&host, Arc::clone(&io), &part);
        drop(step(Kind::Resolve, move || job.read_checkpoint()));
        let (job, other) = armed_job(&host, Arc::clone(&io), &part);
        submit(
            job,
            |_, _| panic!("an accepted step is not refused"),
            |parked| {
                drop(parked);
                Ok(())
            },
        );

        for reports in [receiver, other] {
            let result = reports.try_recv().expect("the discard reports at once");
            assert!(matches!(&result, Err(PackageError::PartLeft { .. })), "{result:?}");
            assert!(reports.try_recv().is_err(), "the run reports once");
        }
        host.complete_gracefully();
        io.complete();
        lattice.close();
        assert!(part.exists(), "a discard removed the output");
    }

    // Mutant caught: a refused length step that drops its continuation instead of calling it with the refusal. The I/O
    // closes after the open was accepted and before it is reaped, so the length is the operation that is refused.
    #[test]
    fn a_length_step_the_io_refuses_calls_its_continuation_once_with_the_error() {
        let io = AsyncIo::open().unwrap();
        let (sender, receiver) = mpsc::channel();
        read_all(
            &io,
            &PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml"),
            Box::new(move |result| sender.send(result).unwrap()),
        );
        io.complete();
        let begun = Instant::now();
        let result = loop {
            match receiver.try_recv() {
                Ok(result) => break result,
                Err(mpsc::TryRecvError::Disconnected) => {
                    panic!("the length step's continuation was dropped without being called")
                }
                Err(mpsc::TryRecvError::Empty) => {}
            }
            assert!(
                begun.elapsed() < Duration::from_secs(30),
                "the refused length step never called its continuation"
            );
            io.reap();
            io.pull(&mut |frame| drop(frame), &mut |_| false, u64::MAX);
        };
        assert!(result.is_err(), "the length step was accepted");
        assert!(receiver.try_recv().is_err(), "the continuation ran twice");
    }

    // Mutant caught: a refused rename that leaves the output, or reports `Lost` instead of the refusal.
    #[test]
    fn a_rename_the_io_refuses_removes_the_output_and_ends_the_run_once_with_the_error() {
        let lattice =
            ControlPlaneLattice::new(lattice_config(&Settings::with_model(ModelSettings::default()))).unwrap();
        let host = Lake::new(
            Arc::clone(&lattice),
            NonZeroUsize::new(2).unwrap(),
            Duration::from_secs(1),
        );
        let io = AsyncIo::open().unwrap();
        io.complete();
        let directory = tempfile::tempdir().unwrap();
        let part = directory.path().join("out.edrl.part");
        fs::write(&part, b"partial").unwrap();
        let (job, receiver) = armed_job(&host, Arc::clone(&io), &part);
        job.finish(
            Written::Part {
                part: part.clone(),
                out: directory.path().join("out.edrl"),
            },
            0,
            0,
        );
        let result = receiver
            .recv_timeout(Duration::from_secs(30))
            .expect("the refused rename never ended the run");
        assert!(matches!(&result, Err(PackageError::Io { .. })), "{result:?}");
        assert!(receiver.try_recv().is_err(), "the run reported twice");
        assert!(!part.exists(), "the output was left");
        host.complete_gracefully();
        lattice.close();
    }

    // Mutant caught: a refused step that drops its continuation (the step is never told, and reports `Lost`) instead
    // of calling it with the refusal; a step that is refused and also called a second time.
    #[test]
    fn a_step_the_io_refuses_calls_its_continuation_once_with_the_error() {
        let io = AsyncIo::open().unwrap();
        io.complete();
        let file = Arc::new(File::open(env!("CARGO_MANIFEST_DIR").to_owned() + "/Cargo.toml").unwrap());
        let calls = Arc::new(AtomicUsize::new(0));
        fn counted<T: 'static>(calls: &Arc<AtomicUsize>) -> Then<T> {
            let calls = Arc::clone(calls);
            Box::new(move |result| {
                assert!(result.is_err(), "a refused step has an error");
                calls.fetch_add(1, Ordering::SeqCst);
            })
        }
        open(&io, Path::new("absent"), OpenSpec::default(), counted(&calls));
        read_exact(&io, Arc::clone(&file), 0, 4, Vec::new(), counted(&calls));
        write_all(&io, Arc::clone(&file), 0, vec![1], counted(&calls));
        read_all(&io, Path::new("absent"), counted(&calls));
        assert_eq!(calls.load(Ordering::SeqCst), 4);
    }

    /// A run dropped after the artifact was cut says the artifact has no section, not just that it was lost.
    #[test]
    fn a_run_dropped_after_the_cut_reports_the_cut() {
        let lattice =
            ControlPlaneLattice::new(lattice_config(&Settings::with_model(ModelSettings::default()))).unwrap();
        let host = Lake::new(
            Arc::clone(&lattice),
            NonZeroUsize::new(2).unwrap(),
            Duration::from_secs(1),
        );
        let (sender, receiver) = mpsc::channel();
        let end = RunEnd {
            host: Arc::clone(&host),
            part: None,
            cut: true,
            completion: Some(Completion {
                done: Some(Box::new(move |result| sender.send(result).unwrap())),
            }),
        };
        drop(end);
        let result = receiver.recv_timeout(Duration::from_secs(30)).expect("the run reports");
        assert!(
            matches!(&result, Err(PackageError::Cut(cause)) if matches!(**cause, PackageError::Lost)),
            "{result:?}"
        );
        host.complete_gracefully();
        lattice.close();
    }
}
