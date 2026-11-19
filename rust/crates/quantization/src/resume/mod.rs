//! The resume file: the one file a long run overwrites in place and resumes from. It is a header of properties (what
//! the run is, how far it got, which sections hold whole data) and, after it, sections at fixed offsets.
//!
//! The order of a save is the caller's, and the file makes each step of it safe:
//! 1. check the header that will be written last ([`Header::check`]): a header that does not fit is found while the old
//!    state is whole;
//! 2. write a header that marks the sections about to change invalid ([`ResumeFile::set_header`]);
//! 3. write the sections ([`ResumeFile::write`]);
//! 4. make them durable ([`ResumeFile::sync`]);
//! 5. write the header that marks them valid; it is itself written between two durability barriers.
//!
//! A stop at any point leaves the old state or the new one, never a mix.
//!
//! Every call returns at once. It starts the work and hands its result to the continuation the caller gave, which
//! returns the frame that carries on. A call whose first step cannot start returns an error and drops its continuation.
//! A later step that is refused (the I/O engine or the host lake is closing) is told to the continuation as an error:
//! each continuation waits in a take-once slot until a step ends or a refusal takes it, so it runs exactly once. A
//! continuation may call [`ResumeFile::remove`] on the handle it was started from: no call holds a handle while its
//! continuation runs.

mod header;

use std::fmt::{Debug, Formatter, Result as FmtResult};
use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[cfg(unix)]
use std::os::fd::AsRawFd;
#[cfg(windows)]
use std::os::windows::io::AsRawHandle;

use euhedral_core::frames::Frame;
use euhedral_graph::{Lake, UnitGuard};
use euhedral_inference_engine::Handoff;
use euhedral_io::{AsyncIo, Continuation, Create, IoBuf, IoDone, OpenSpec, OwnedFile, RawFile};
use thiserror::Error;

pub use header::Header;

use crate::durable::{self, Done, LakeClosed, SharedFile};
use crate::step::{Kind, idle, step};

/// Bytes of the header; the sections start after it.
pub const HEADER: u64 = 4096;

/// What can go wrong with a resume file.
#[derive(Debug, Error)]
pub enum ResumeError {
    #[error("{0}")]
    Io(#[from] io::Error),
    #[error("the host lake is closed: the call was not started")]
    LakeClosed,
    #[error("the resume header is {bytes} bytes, more than the {HEADER} it may take")]
    HeaderTooLarge { bytes: usize },
    #[error("the resume header is malformed: {why}")]
    HeaderText { why: String },
    #[error("the resume file ends at {at}, before the end of what was asked for")]
    ShortRead { at: u64 },
    #[error("the resume file has another handle in use")]
    Busy,
    #[error("the section runs past the largest file offset")]
    Offset,
    #[error("a write to the resume file stopped at {at}, short of the end of the data")]
    ShortWrite { at: u64 },
}

impl From<LakeClosed> for ResumeError {
    fn from(_: LakeClosed) -> ResumeError {
        ResumeError::LakeClosed
    }
}

/// What a call's continuation is told.
pub type Then<T> = Continuation<Result<T, ResumeError>>;

/// A continuation that waits to be taken by whichever of the steps of a call ends the call, so that it runs once.
struct Slot<T>(Arc<Handoff<Then<T>>>);

impl<T> Clone for Slot<T> {
    fn clone(&self) -> Slot<T> {
        Slot(Arc::clone(&self.0))
    }
}

impl<T> Slot<T> {
    fn new(then: Then<T>) -> Slot<T> {
        let handoff = Handoff::new();
        handoff.send(then);
        Slot(handoff)
    }

    /// Ends the call with `result`: runs the continuation if no step has yet.
    fn finish(&self, result: Result<T, ResumeError>) -> Box<dyn Frame> {
        match self.0.take() {
            Some(then) => then(result),
            None => idle(),
        }
    }

    /// Ends the call with the error of a step that could not start; the step's own frame is nothing.
    fn fail<U>(&self, started: Result<U, ResumeError>) -> Box<dyn Frame> {
        match started {
            Ok(_) => idle(),
            Err(error) => self.finish(Err(error)),
        }
    }
}

/// The file, the ways to reach it, and the unit that lets the host lake take frames while a handle exists.
struct Inner {
    file: SharedFile,
    io: Arc<AsyncIo>,
    host: Arc<Lake>,
    path: PathBuf,
    unit: UnitGuard,
}

/// A resume file. A handle is cheap to copy; the file closes with the last one.
#[derive(Clone)]
pub struct ResumeFile {
    inner: Arc<Inner>,
}

impl Debug for ResumeFile {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        formatter
            .debug_struct("ResumeFile")
            .field("path", &self.inner.path)
            .finish_non_exhaustive()
    }
}

#[cfg(unix)]
fn raw(file: &File) -> RawFile {
    file.as_raw_fd()
}

#[cfg(windows)]
fn raw(file: &File) -> RawFile {
    file.as_raw_handle()
}

impl ResumeFile {
    /// Opens the file at `path`, creating it empty if it is missing. `host` runs the durability calls and must take
    /// frames: the file holds an admitted unit of it until the last handle closes.
    pub fn open(io: &Arc<AsyncIo>, host: &Arc<Lake>, path: &Path, then: Then<ResumeFile>) -> Result<(), ResumeError> {
        let unit = host.admit().map_err(|_| ResumeError::LakeClosed)?;
        let (io_held, host_held, path_held) = (Arc::clone(io), Arc::clone(host), path.to_owned());
        let spec = OpenSpec {
            read: true,
            write: true,
            create: Create::IfMissing,
            unix_mode: 0o644,
        };
        let opened = move |opened: io::Result<OwnedFile>| {
            then(opened.map_err(ResumeError::Io).map(|file| ResumeFile {
                inner: Arc::new(Inner {
                    file: durable::share(file),
                    io: io_held,
                    host: host_held,
                    path: path_held,
                    unit,
                }),
            }))
        };
        Ok(io.open_path(path, spec, Box::new(opened))?)
    }

    /// The header on disk: empty for a new file, and for one that is shorter than a header.
    pub fn header(&self, then: Then<Header>) -> Result<(), ResumeError> {
        let slot = Slot::new(then);
        let me = self.clone();
        let measured = {
            let slot = slot.clone();
            move |len: io::Result<u64>| match len {
                Err(error) => {
                    drop(me);
                    slot.finish(Err(error.into()))
                }
                Ok(len) if len < HEADER => {
                    drop(me);
                    slot.finish(Ok(Header::new()))
                }
                Ok(_) => {
                    let header = slot.clone();
                    let bytes = Slot::<Vec<u8>>::new(Box::new(move |bytes| {
                        header.finish(bytes.and_then(|bytes| Header::decode(&bytes)))
                    }));
                    let started = me.read_slot(0, HEADER as usize, &bytes);
                    drop(me);
                    bytes.fail(started)
                }
            }
        };
        Ok(self.inner.io.file_len(raw(&self.inner.file), Box::new(measured))?)
    }

    /// Replaces the header with `header`, making the sections written before it durable first and the header itself
    /// durable after. The header is checked first: one that does not fit returns the error and touches nothing.
    pub fn set_header(&self, header: &Header, then: Then<()>) -> Result<(), ResumeError> {
        let bytes = header.encode()?;
        let slot = Slot::new(then);
        let me = self.clone();
        let barrier = {
            let slot = slot.clone();
            move |synced: Result<(), ResumeError>| {
                if let Err(error) = synced {
                    drop(me);
                    return slot.finish(Err(error));
                }
                let again = me.clone();
                let last = slot.clone();
                let written = Slot::<u64>::new(Box::new(move |written| {
                    let started = match written {
                        Ok(_) => again.sync_slot(&last),
                        Err(error) => {
                            drop(again);
                            return last.finish(Err(error));
                        }
                    };
                    drop(again);
                    last.fail(started)
                }));
                let started = me.write_slot(0, bytes, &written);
                drop(me);
                written.fail(started)
            }
        };
        self.on_host(durable::sync, &Slot::new(Box::new(barrier)))
    }

    /// Writes `bytes` at `offset`; `then` is told the offset after them.
    pub fn write(&self, offset: u64, bytes: Vec<u8>, then: Then<u64>) -> Result<(), ResumeError> {
        self.write_slot(offset, bytes, &Slot::new(then))
    }

    fn write_slot(&self, offset: u64, bytes: Vec<u8>, slot: &Slot<u64>) -> Result<(), ResumeError> {
        let end = offset.checked_add(bytes.len() as u64).ok_or(ResumeError::Offset)?;
        let filled = bytes.len();
        if filled == 0 {
            return self.resolve(Ok(end), slot);
        }
        let held = self.clone();
        let slot = slot.clone();
        let written = move |done: IoDone| {
            drop(held);
            slot.finish(match done.result {
                Ok(moved) if moved == filled => Ok(end),
                Ok(moved) => Err(ResumeError::ShortWrite {
                    at: offset + moved as u64,
                }),
                Err(error) => Err(error.into()),
            })
        };
        Ok(self.inner.io.write(
            raw(&self.inner.file),
            offset,
            IoBuf { bytes, filled },
            Box::new(written),
        )?)
    }

    /// Reads `len` bytes at `offset`. A file that ends first is an error naming where.
    pub fn read(&self, offset: u64, len: usize, then: Then<Vec<u8>>) -> Result<(), ResumeError> {
        self.read_slot(offset, len, &Slot::new(then))
    }

    fn read_slot(&self, offset: u64, len: usize, slot: &Slot<Vec<u8>>) -> Result<(), ResumeError> {
        if len == 0 {
            return self.resolve(Ok(Vec::new()), slot);
        }
        offset.checked_add(len as u64).ok_or(ResumeError::Offset)?;
        let held = self.clone();
        let slot = slot.clone();
        let read = move |done: IoDone| {
            drop(held);
            slot.finish(match done.result {
                Ok(moved) if moved == len => Ok(done.buf.bytes),
                Ok(moved) => Err(ResumeError::ShortRead {
                    at: offset + moved as u64,
                }),
                Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => {
                    Err(ResumeError::ShortRead { at: offset })
                }
                Err(error) => Err(error.into()),
            })
        };
        let buffer = IoBuf {
            bytes: vec![0; len],
            filled: 0,
        };
        Ok(self
            .inner
            .io
            .read(raw(&self.inner.file), offset, buffer, Box::new(read))?)
    }

    /// Makes everything written so far durable.
    pub fn sync(&self, then: Then<()>) -> Result<(), ResumeError> {
        self.sync_slot(&Slot::new(then))
    }

    fn sync_slot(&self, slot: &Slot<()>) -> Result<(), ResumeError> {
        self.on_host(durable::sync, slot)
    }

    /// Cuts the file back to `len` bytes.
    pub fn truncate(&self, len: u64, then: Then<()>) -> Result<(), ResumeError> {
        self.on_host(
            |host, file, done| durable::truncate(host, file, len, done),
            &Slot::new(then),
        )
    }

    /// Closes the file and removes it: the end of a run that finished. It is an error ([`ResumeError::Busy`]) while
    /// another handle is open, because the file would not be closed; the handle this call took is spent either way.
    pub fn remove(self, then: Then<()>) -> Result<(), ResumeError> {
        let Inner {
            file, host, path, unit, ..
        } = Arc::into_inner(self.inner).ok_or(ResumeError::Busy)?;
        drop(file);
        let lake = Arc::clone(&host);
        let removed = move |result: io::Result<()>| {
            let next = then(result.map_err(ResumeError::Io));
            drop(lake.offer(next));
            step(Kind::Release, move || drop(unit))
        };
        Ok(durable::remove(&host, &path, Box::new(removed))?)
    }

    /// Runs a durability call and ends `slot` with its result. A unit forked from the file's keeps the lake admitted
    /// until the completion has been offered, and no handle is held meanwhile.
    fn on_host(
        &self,
        call: impl FnOnce(&Arc<Lake>, &SharedFile, Done) -> Result<(), LakeClosed>,
        slot: &Slot<()>,
    ) -> Result<(), ResumeError> {
        let (unit, host, slot) = (self.inner.unit.fork(), Arc::clone(&self.inner.host), slot.clone());
        let done = move |result: io::Result<()>| {
            let next = slot.finish(result.map_err(ResumeError::Io));
            drop(host.offer(next));
            step(Kind::Release, move || drop(unit))
        };
        Ok(call(&self.inner.host, &self.inner.file, Box::new(done))?)
    }

    /// Ends `slot` with `value` on the host lake: a call with nothing to wait for.
    fn resolve<T: Send + 'static>(&self, value: Result<T, ResumeError>, slot: &Slot<T>) -> Result<(), ResumeError> {
        let (unit, host, slot) = (self.inner.unit.fork(), Arc::clone(&self.inner.host), slot.clone());
        let offered = Arc::clone(&host);
        let frame = step(Kind::Resolve, move || {
            let next = slot.finish(value);
            drop(host.offer(next));
            drop(unit);
        });
        offered.offer(frame).map_err(|_| ResumeError::LakeClosed)
    }
}
