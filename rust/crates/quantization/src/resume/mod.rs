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
//! each continuation waits in a take-once claim cell until a step ends the call, so it runs at most once. A step that
//! the lake or the I/O engine accepted and later discards unrun (they are closing) ends its call with
//! [`ResumeError::LakeClosed`]; the one call whose continuation does not run is a call that returned an error. A
//! continuation may call [`ResumeFile::remove`] on the handle it was started from, once the caller of the call has
//! dropped its own: no step holds a handle while its continuation runs.

mod header;
mod slot;

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
use euhedral_io::{AsyncIo, Continuation, Create, IoBuf, IoDone, OpenSpec, OwnedFile, RawFile};
use thiserror::Error;

pub use header::Header;

use slot::Slot;

use crate::durable::{self, Done, LakeClosed, SharedFile};
use crate::step::{Kind, step};

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
}

impl From<LakeClosed> for ResumeError {
    fn from(_: LakeClosed) -> ResumeError {
        ResumeError::LakeClosed
    }
}

/// What a call's continuation is told.
pub type Then<T> = Continuation<Result<T, ResumeError>>;

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
        let spec = OpenSpec {
            read: true,
            write: true,
            create: Create::IfMissing,
            unix_mode: 0o644,
        };
        ResumeFile::open_with(io, host, path, spec, then)
    }

    /// Opens the file at `path` for reading only. A missing file is an I/O error of kind `NotFound` and nothing is
    /// created; a file the caller may not write opens. The calls that write, sync, truncate or remove then fail.
    pub fn open_read_only(
        io: &Arc<AsyncIo>,
        host: &Arc<Lake>,
        path: &Path,
        then: Then<ResumeFile>,
    ) -> Result<(), ResumeError> {
        let spec = OpenSpec {
            read: true,
            write: false,
            create: Create::Never,
            unix_mode: 0o644,
        };
        ResumeFile::open_with(io, host, path, spec, then)
    }

    fn open_with(
        io: &Arc<AsyncIo>,
        host: &Arc<Lake>,
        path: &Path,
        spec: OpenSpec,
        then: Then<ResumeFile>,
    ) -> Result<(), ResumeError> {
        let unit = host.admit().map_err(|_| ResumeError::LakeClosed)?;
        let (io_held, host_held, path_held) = (Arc::clone(io), Arc::clone(host), path.to_owned());
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

    /// The length of the file in bytes.
    pub fn len(&self, then: Then<u64>) -> Result<(), ResumeError> {
        let (io, file) = (Arc::clone(&self.inner.io), raw(&self.inner.file));
        let slot = Slot::new(then);
        slot.submit(self.clone(), move |watch| {
            let measured = move |len: io::Result<u64>| {
                let (slot, held) = watch.take();
                drop(held);
                slot.finish(len.map_err(ResumeError::Io))
            };
            Ok(io.file_len(file, Box::new(measured))?)
        })
    }

    /// Ends a call that has nothing to wait for: `then` runs on the host lake, and the file is not touched.
    pub fn settled(&self, then: Then<()>) -> Result<(), ResumeError> {
        self.clone().resolve(Ok(()), &Slot::new(then))
    }

    /// The header on disk: empty for a new file, and for one that is shorter than a header.
    pub fn header(&self, then: Then<Header>) -> Result<(), ResumeError> {
        let (io, file) = (Arc::clone(&self.inner.io), raw(&self.inner.file));
        let slot = Slot::new(then);
        slot.submit(self.clone(), move |watch| {
            let measured = move |len: io::Result<u64>| {
                let (slot, me) = watch.take();
                match len {
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
                        bytes.fail(started)
                    }
                }
            };
            Ok(io.file_len(file, Box::new(measured))?)
        })
    }

    /// Replaces the header with `header`, making the sections written before it durable first and the header itself
    /// durable after. The header is checked first: one that does not fit returns the error and touches nothing.
    pub fn set_header(&self, header: &Header, then: Then<()>) -> Result<(), ResumeError> {
        let bytes = header.encode()?;
        let slot = Slot::new(then);
        let keep = self.clone();
        let barrier = {
            let slot = slot.clone();
            move |synced: Result<(), ResumeError>| {
                if let Err(error) = synced {
                    drop(keep);
                    return slot.finish(Err(error));
                }
                let again = keep.clone();
                let last = slot.clone();
                let written = Slot::<u64>::new(Box::new(move |written| match written {
                    Ok(_) => {
                        let started = again.sync_slot(&last);
                        last.fail(started)
                    }
                    Err(error) => {
                        drop(again);
                        last.finish(Err(error))
                    }
                }));
                let started = keep.write_slot(0, bytes, &written);
                written.fail(started)
            }
        };
        self.clone()
            .on_host(durable::sync_and_close, &Slot::new(Box::new(barrier)))
    }

    /// Writes `bytes` at `offset`; `then` is told the offset after them.
    pub fn write(&self, offset: u64, bytes: Vec<u8>, then: Then<u64>) -> Result<(), ResumeError> {
        self.clone().write_slot(offset, bytes, &Slot::new(then))
    }

    fn write_slot(self, offset: u64, bytes: Vec<u8>, slot: &Slot<u64>) -> Result<(), ResumeError> {
        let end = offset.checked_add(bytes.len() as u64).ok_or(ResumeError::Offset)?;
        let filled = bytes.len();
        if filled == 0 {
            return self.resolve(Ok(end), slot);
        }
        let (io, file) = (Arc::clone(&self.inner.io), raw(&self.inner.file));
        slot.submit(self, move |watch| {
            let written = move |done: IoDone| {
                let (slot, held) = watch.take();
                drop(held);
                // The I/O engine resubmits a write that moved fewer bytes than asked, and reports a write that
                // moved none as an error, so a write that completes moved all of them.
                slot.finish(done.result.map(|_| end).map_err(ResumeError::from))
            };
            Ok(io.write(file, offset, IoBuf { bytes, filled }, Box::new(written))?)
        })
    }

    /// Reads `len` bytes at `offset`. A file that ends first is an error naming where.
    pub fn read(&self, offset: u64, len: usize, then: Then<Vec<u8>>) -> Result<(), ResumeError> {
        self.clone().read_slot(offset, len, &Slot::new(then))
    }

    fn read_slot(self, offset: u64, len: usize, slot: &Slot<Vec<u8>>) -> Result<(), ResumeError> {
        if len == 0 {
            return self.resolve(Ok(Vec::new()), slot);
        }
        offset.checked_add(len as u64).ok_or(ResumeError::Offset)?;
        let (io, file) = (Arc::clone(&self.inner.io), raw(&self.inner.file));
        slot.submit(self, move |watch| {
            let read = move |done: IoDone| {
                let (slot, held) = watch.take();
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
            Ok(io.read(file, offset, buffer, Box::new(read))?)
        })
    }

    /// Makes everything written so far durable.
    pub fn sync(&self, then: Then<()>) -> Result<(), ResumeError> {
        self.clone().sync_slot(&Slot::new(then))
    }

    fn sync_slot(self, slot: &Slot<()>) -> Result<(), ResumeError> {
        self.on_host(durable::sync_and_close, slot)
    }

    /// Cuts the file back to `len` bytes.
    pub fn truncate(&self, len: u64, then: Then<()>) -> Result<(), ResumeError> {
        self.clone().on_host(
            |host, file, done| durable::truncate_and_close(host, file, len, done),
            &Slot::new(then),
        )
    }

    /// Closes the file and removes it: the end of a run that finished. It is an error ([`ResumeError::Busy`]) while
    /// another handle is open or a call on the file is in flight (a `sync` or `truncate` still running holds a handle
    /// until its continuation is about to run), because the file would not be closed; the handle this call took is
    /// spent either way. A continuation may call it on the handle its call was started from.
    ///
    /// The file is closed first, so on Windows (where `AsyncIo` opens files shared for delete, and a removed file
    /// stays until its last handle closes) the removal is not left pending.
    pub fn remove(self, then: Then<()>) -> Result<(), ResumeError> {
        let Inner {
            file, host, path, unit, ..
        } = Arc::into_inner(self.inner).ok_or(ResumeError::Busy)?;
        drop(file);
        let slot = Slot::new(then);
        let lake = Arc::clone(&host);
        slot.submit((), move |watch| {
            let removed = move |result: io::Result<()>| {
                let (slot, ()) = watch.take();
                let next = slot.finish(result.map_err(ResumeError::Io));
                drop(lake.offer(next));
                step(Kind::Release, move || drop(unit))
            };
            Ok(durable::remove(&host, &path, Box::new(removed))?)
        })
    }

    /// Ends a run whose file is the output: cuts the file back to `len` bytes, makes it durable, closes it and renames
    /// it to `to`, replacing a file of that name. The same conditions as [`ResumeFile::remove`] apply: it is an error
    /// ([`ResumeError::Busy`]) while another handle is open or a call on the file is in flight, and the handle this
    /// call took is spent either way. The file is closed before it is renamed, and no handle of the call is left when
    /// `then` runs.
    pub fn seal(self, len: u64, to: &Path, then: Then<()>) -> Result<(), ResumeError> {
        let Inner {
            file,
            io,
            host,
            path,
            unit,
        } = Arc::into_inner(self.inner).ok_or(ResumeError::Busy)?;
        let seal = Seal {
            file,
            io,
            host,
            path,
            to: to.to_owned(),
            len,
            unit,
        };
        seal.cut(&Slot::new(then))
    }

    /// Runs a durability call and ends `slot` with its result. A unit forked from the file's keeps the lake admitted
    /// until the completion has been offered. The step owns this handle until its continuation is about to run, and
    /// the caller holds none when the call returns.
    fn on_host(
        self,
        call: impl FnOnce(&Arc<Lake>, SharedFile, Done) -> Result<(), LakeClosed>,
        slot: &Slot<()>,
    ) -> Result<(), ResumeError> {
        let (unit, host, file) = (
            self.inner.unit.fork(),
            Arc::clone(&self.inner.host),
            Arc::clone(&self.inner.file),
        );
        let lake = Arc::clone(&host);
        slot.submit(self, move |watch| {
            let done = move |result: io::Result<()>| {
                let (slot, held) = watch.take();
                drop(held);
                let next = slot.finish(result.map_err(ResumeError::Io));
                drop(lake.offer(next));
                step(Kind::Release, move || drop(unit))
            };
            Ok(call(&host, file, Box::new(done))?)
        })
    }

    /// Ends `slot` with `value` on the host lake: a call with nothing to wait for.
    fn resolve<T: Send + 'static>(self, value: Result<T, ResumeError>, slot: &Slot<T>) -> Result<(), ResumeError> {
        let (unit, host) = (self.inner.unit.fork(), Arc::clone(&self.inner.host));
        let offered = Arc::clone(&host);
        slot.submit(self, move |watch| {
            let frame = step(Kind::Resolve, move || {
                let (slot, held) = watch.take();
                drop(held);
                let next = slot.finish(value);
                drop(host.offer(next));
                drop(unit);
            });
            offered.offer(frame).map_err(|_| ResumeError::LakeClosed)
        })
    }
}

/// The steps of [`ResumeFile::seal`], each started by the continuation of the one before it.
struct Seal {
    file: SharedFile,
    io: Arc<AsyncIo>,
    host: Arc<Lake>,
    path: PathBuf,
    to: PathBuf,
    len: u64,
    unit: UnitGuard,
}

impl Seal {
    /// Ends the call with `result` on the host lake, and lets the lake go once that is offered.
    fn end(slot: &Slot<()>, host: &Arc<Lake>, unit: UnitGuard, result: Result<(), ResumeError>) -> Box<dyn Frame> {
        let next = slot.finish(result);
        drop(host.offer(next));
        step(Kind::Release, move || drop(unit))
    }

    /// Ends the call, from a continuation, with the error of the next step if that one could not start.
    fn continued(
        started: Result<(), ResumeError>,
        slot: &Slot<()>,
        host: &Arc<Lake>,
        unit: UnitGuard,
    ) -> Box<dyn Frame> {
        match started {
            Ok(()) => step(Kind::Release, move || drop(unit)),
            Err(error) => Seal::end(slot, host, unit, Err(error)),
        }
    }

    fn cut(self, slot: &Slot<()>) -> Result<(), ResumeError> {
        let (host, file, len) = (Arc::clone(&self.host), Arc::clone(&self.file), self.len);
        let kept = self.unit.fork();
        slot.submit((), move |watch| {
            let cut = move |result: io::Result<()>| {
                let (slot, ()) = watch.take();
                match result {
                    Ok(()) => {
                        let host = Arc::clone(&self.host);
                        let started = self.sync(&slot);
                        Seal::continued(started, &slot, &host, kept)
                    }
                    Err(error) => Seal::end(&slot, &self.host, kept, Err(error.into())),
                }
            };
            Ok(durable::truncate_and_close(&host, file, len, Box::new(cut))?)
        })
    }

    fn sync(self, slot: &Slot<()>) -> Result<(), ResumeError> {
        let Seal {
            file,
            io,
            host,
            path,
            to,
            unit,
            ..
        } = self;
        let (lake, kept) = (Arc::clone(&host), unit.fork());
        slot.submit((), move |watch| {
            let synced = move |result: io::Result<()>| {
                let (slot, ()) = watch.take();
                match result {
                    Ok(()) => {
                        let started = Seal::rename(&io, &lake, &path, &to, &slot, unit.fork());
                        drop(unit);
                        Seal::continued(started, &slot, &lake, kept)
                    }
                    Err(error) => Seal::end(&slot, &lake, kept, Err(error.into())),
                }
            };
            Ok(durable::sync_and_close(&host, file, Box::new(synced))?)
        })
    }

    /// The last step: the file is closed, so the name can change.
    fn rename(
        io: &Arc<AsyncIo>,
        host: &Arc<Lake>,
        path: &Path,
        to: &Path,
        slot: &Slot<()>,
        unit: UnitGuard,
    ) -> Result<(), ResumeError> {
        let lake = Arc::clone(host);
        slot.submit((), move |watch| {
            let renamed = move |result: io::Result<()>| {
                let (slot, ()) = watch.take();
                Seal::end(&slot, &lake, unit, result.map_err(ResumeError::Io))
            };
            Ok(io.rename(path, to, Box::new(renamed))?)
        })
    }
}
