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
//! returns the frame that carries on. A call that cannot start returns an error and drops its continuation. A
//! continuation that is dropped without running (a later part of a call that the I/O engine or the host lake refused,
//! because it is closing) takes what it owns with it: a continuation that holds the only sender of a hand-off tells
//! its waiter by dropping it.

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

use euhedral_graph::{Lake, UnitGuard};
use euhedral_io::{AsyncIo, Continuation, Create, IoBuf, OpenSpec, RawFile};
use thiserror::Error;

pub use header::Header;

use crate::durable::{self, Done, SharedFile};
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
        let unit = host.admit().map_err(|_| ResumeError::LakeClosed)?;
        let (io_held, host_held, path_held) = (Arc::clone(io), Arc::clone(host), path.to_owned());
        let spec = OpenSpec {
            read: true,
            write: true,
            create: Create::IfMissing,
            unix_mode: 0o644,
        };
        let opened = move |opened: io::Result<euhedral_io::OwnedFile>| {
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
        let me = self.clone();
        let measured = move |len: io::Result<u64>| match len {
            Err(error) => then(Err(error.into())),
            Ok(len) if len < HEADER => then(Ok(Header::new())),
            Ok(_) => {
                let read = me.read(
                    0,
                    HEADER as usize,
                    Box::new(move |bytes| then(bytes.and_then(|bytes| Header::decode(&bytes)))),
                );
                drop(read);
                idle()
            }
        };
        Ok(self.inner.io.file_len(raw(&self.inner.file), Box::new(measured))?)
    }

    /// Replaces the header with `header`, making the sections written before it durable first and the header itself
    /// durable after. The header is checked first: one that does not fit returns the error and touches nothing.
    pub fn set_header(&self, header: &Header, then: Then<()>) -> Result<(), ResumeError> {
        let bytes = header.encode()?;
        let me = self.clone();
        let barrier = move |synced: Result<(), ResumeError>| {
            if let Err(error) = synced {
                return then(Err(error));
            }
            let again = me.clone();
            let written = move |written: Result<u64, ResumeError>| {
                if let Err(error) = written {
                    return then(Err(error));
                }
                drop(again.sync(then));
                idle()
            };
            drop(me.write(0, bytes, Box::new(written)));
            idle()
        };
        self.on_host(durable::sync, Box::new(barrier))
    }

    /// Writes `bytes` at `offset`; `then` is told the offset after them.
    pub fn write(&self, offset: u64, bytes: Vec<u8>, then: Then<u64>) -> Result<(), ResumeError> {
        let end = offset.checked_add(bytes.len() as u64).ok_or(ResumeError::Offset)?;
        let filled = bytes.len();
        if filled == 0 {
            return self.resolve(Ok(end), then);
        }
        let held = self.clone();
        let written = move |done: euhedral_io::IoDone| {
            drop(held);
            then(match done.result {
                Ok(moved) if moved == filled => Ok(end),
                Ok(moved) => Err(ResumeError::ShortRead {
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
        if len == 0 {
            return self.resolve(Ok(Vec::new()), then);
        }
        offset.checked_add(len as u64).ok_or(ResumeError::Offset)?;
        let held = self.clone();
        let read = move |done: euhedral_io::IoDone| {
            drop(held);
            then(match done.result {
                Ok(moved) if moved == len => Ok(done.buf.bytes),
                Ok(moved) => Err(ResumeError::ShortRead {
                    at: offset + moved as u64,
                }),
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
        self.on_host(durable::sync, then)
    }

    /// Cuts the file back to `len` bytes.
    pub fn truncate(&self, len: u64, then: Then<()>) -> Result<(), ResumeError> {
        self.on_host(|host, file, done| durable::truncate(host, file, len, done), then)
    }

    /// Closes the file and removes it: the end of a run that finished. It is an error ([`ResumeError::Busy`]) while
    /// another handle is open, because the file would not be closed; the handle this call took is spent either way.
    pub fn remove(self, then: Then<()>) -> Result<(), ResumeError> {
        let Inner {
            file, host, path, unit, ..
        } = Arc::into_inner(self.inner).ok_or(ResumeError::Busy)?;
        drop(file);
        let lake = Arc::clone(&host);
        let removed = move |result: Result<(), ResumeError>| {
            let next = then(result);
            drop(lake.offer(next));
            step(Kind::Release, move || drop(unit))
        };
        durable::remove(&host, &path, Box::new(removed))
    }

    /// Runs a durability call, holding this handle until its completion has been offered: the lake takes the
    /// completion only while the file's unit is admitted.
    fn on_host(
        &self,
        call: impl FnOnce(&Arc<Lake>, &SharedFile, Done) -> Result<(), ResumeError>,
        then: Then<()>,
    ) -> Result<(), ResumeError> {
        let held = Arc::clone(&self.inner);
        let done = move |result: Result<(), ResumeError>| {
            let next = then(result);
            drop(held.host.offer(next));
            step(Kind::Release, move || drop(held))
        };
        call(&self.inner.host, &self.inner.file, Box::new(done))
    }

    /// Hands `value` to `then` on the host lake: a call with nothing to wait for.
    fn resolve<T: Send + 'static>(&self, value: Result<T, ResumeError>, then: Then<T>) -> Result<(), ResumeError> {
        let held = Arc::clone(&self.inner);
        let frame = step(Kind::Resolve, move || {
            let next = then(value);
            drop(held.host.offer(next));
            drop(held);
        });
        self.inner.host.offer(frame).map_err(|_| ResumeError::LakeClosed)
    }
}
