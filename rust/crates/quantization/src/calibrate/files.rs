//! The calibration's files on the lattice: a file saved whole, and the hidden states in the resume file, a chunk at a
//! time. None of them touches the card.

use std::path::PathBuf;
use std::slice;
use std::sync::Arc;

use euhedral_core::frames::Frame;
use euhedral_graph::Lake;
use euhedral_inference_engine::Handoff;
use euhedral_io::AsyncIo;

use super::CalibrateError;
use super::driver::{Once, Then, end, single};
use super::manifest::Manifest;
use super::plan::{resume_at, resume_header};
use crate::resume::{HEADER, Header, ResumeError, ResumeFile, Then as FileThen};
use crate::step::idle;

/// A range of memory the calls read the resume file into or write it from.
#[derive(Clone, Copy, Debug)]
pub struct Region {
    pointer: *mut u8,
    length: usize,
}

// SAFETY: the region is plain memory; the owner of `Region::new` promises that the calls touch it one after another.
unsafe impl Send for Region {}
// SAFETY: as for `Send`: nothing reads or writes the region while a call of this module does.
unsafe impl Sync for Region {}

impl Region {
    /// The `length` bytes at `pointer`.
    ///
    /// # Safety
    /// The bytes stay valid, and nothing else reads or writes them, while a call given the region is running.
    pub unsafe fn new(pointer: *mut u8, length: usize) -> Region {
        Region { pointer, length }
    }

    /// Bytes in the region.
    pub fn len(&self) -> usize {
        self.length
    }

    /// Whether the region holds no bytes.
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }
}

/// What a save tells its caller; the frame it returns is what the file's call hands the lake.
pub type Saved = Box<dyn FnOnce(Result<(), CalibrateError>) -> Box<dyn Frame> + Send>;

fn finished(finish: &Handoff<Saved>, result: Result<(), CalibrateError>) -> Box<dyn Frame> {
    match finish.take() {
        Some(finish) => finish(result),
        None => idle(),
    }
}

/// Writes `bytes` whole to `path`: to a temporary file next to it, made durable and renamed over `path`, so a stop
/// leaves the file of a finished save or none, and a temporary file left by a stop is never read as the file.
pub fn save(io: &Arc<AsyncIo>, host: &Arc<Lake>, path: PathBuf, bytes: Vec<u8>, then: Saved) {
    let finish = Arc::new(Handoff::<Saved>::empty());
    finish.send(then);
    let mut temporary = path.clone().into_os_string();
    temporary.push(".tmp");
    let len = bytes.len() as u64;
    let open_finish = Arc::clone(&finish);
    let opened = ResumeFile::open(
        io,
        host,
        &PathBuf::from(temporary),
        Box::new(move |opened| {
            let file = match opened {
                Ok(file) => file,
                Err(error) => return finished(&open_finish, Err(CalibrateError::failed("open", error))),
            };
            let write_finish = Arc::clone(&open_finish);
            let sealing = file.clone();
            let wrote = file.write(
                0,
                bytes,
                Box::new(move |wrote| {
                    if let Err(error) = wrote {
                        return finished(&write_finish, Err(CalibrateError::failed("write", error)));
                    }
                    let seal_finish = Arc::clone(&write_finish);
                    let renamed = sealing.seal(
                        len,
                        &path,
                        Box::new(move |sealed| {
                            finished(
                                &seal_finish,
                                sealed.map_err(|error| CalibrateError::failed("seal", error)),
                            )
                        }),
                    );
                    match renamed {
                        Ok(()) => idle(),
                        Err(error) => finished(&write_finish, Err(CalibrateError::failed("seal", error))),
                    }
                }),
            );
            match wrote {
                Ok(()) => idle(),
                Err(error) => finished(&open_finish, Err(CalibrateError::failed("write", error))),
            }
        }),
    );
    if let Err(error) = opened {
        drop(finished(&finish, Err(CalibrateError::failed("open", error))));
    }
}

/// The calls of the resume file that the states use; the file implements them, and a test's fake records them.
pub trait StatesFile: Clone + Send + 'static {
    /// Reads the header.
    fn header(&self, then: FileThen<Header>) -> Result<(), ResumeError>;
    /// Replaces the header.
    fn set_header(&self, header: &Header, then: FileThen<()>) -> Result<(), ResumeError>;
    /// Writes `bytes` at `offset`; `then` gets the offset after them.
    fn write(&self, offset: u64, bytes: Vec<u8>, then: FileThen<u64>) -> Result<(), ResumeError>;
    /// Reads `len` bytes at `offset`.
    fn read(&self, offset: u64, len: usize, then: FileThen<Vec<u8>>) -> Result<(), ResumeError>;
}

impl StatesFile for ResumeFile {
    fn header(&self, then: FileThen<Header>) -> Result<(), ResumeError> {
        ResumeFile::header(self, then)
    }

    fn set_header(&self, header: &Header, then: FileThen<()>) -> Result<(), ResumeError> {
        ResumeFile::set_header(self, header, then)
    }

    fn write(&self, offset: u64, bytes: Vec<u8>, then: FileThen<u64>) -> Result<(), ResumeError> {
        ResumeFile::write(self, offset, bytes, then)
    }

    fn read(&self, offset: u64, len: usize, then: FileThen<Vec<u8>>) -> Result<(), ResumeError> {
        ResumeFile::read(self, offset, len, then)
    }
}

/// Where a run of `layers` layers starts, from the resume file and the directory's `manifest` ([`resume_at`]): at a
/// layer past 0 the file's states are read into `region` first; at 0 the region is left as it is, for the caller to
/// fill from the tokens.
pub fn resume<F: StatesFile>(
    file: &F,
    manifest: Manifest,
    layers: usize,
    region: Region,
    chunk: usize,
    then: Then<usize>,
) {
    let then = single(then);
    let failed = Arc::clone(&then);
    let reading = file.clone();
    let started = file.header(Box::new(move |header| {
        let at = header
            .map_err(|error| CalibrateError::failed("the resume header", error))
            .and_then(|header| resume_at(&header, &manifest, layers));
        match at {
            Ok(0) => end(&then, Ok(0)),
            Ok(done) => {
                let ended = Arc::clone(&then);
                read_states(
                    &reading,
                    region,
                    chunk,
                    Box::new(move |read| end(&ended, read.map(|()| done))),
                );
            }
            Err(error) => end(&then, Err(error)),
        }
        idle()
    }));
    if let Err(error) = started {
        end(&failed, Err(CalibrateError::failed("the resume header", error)));
    }
}

/// Reads the states of the resume file into `region`, `chunk` bytes at a time.
pub fn read_states<F: StatesFile>(file: &F, region: Region, chunk: usize, then: Then<()>) {
    read_from(file.clone(), region, chunk.max(1), 0, single(then));
}

fn read_from<F: StatesFile>(file: F, region: Region, chunk: usize, at: usize, then: Once<()>) {
    if at >= region.length {
        end(&then, Ok(()));
        return;
    }
    let length = chunk.min(region.length - at);
    let failed = Arc::clone(&then);
    let next = file.clone();
    let started = file.read(
        HEADER + at as u64,
        length,
        Box::new(move |bytes| {
            match bytes {
                Err(error) => end(&then, Err(CalibrateError::failed("the resume states", error))),
                Ok(bytes) if bytes.len() != length => end(
                    &then,
                    Err(CalibrateError::Failed(format!(
                        "the resume states hold {} bytes at {at}, not {length}",
                        bytes.len()
                    ))),
                ),
                Ok(bytes) => {
                    // SAFETY: the range is inside the region, which nothing else touches (the contract of `Region`).
                    unsafe { slice::from_raw_parts_mut(region.pointer.add(at), length) }.copy_from_slice(&bytes);
                    read_from(next, region, chunk, at + length, then);
                }
            }
            idle()
        }),
    );
    if let Err(error) = started {
        end(&failed, Err(CalibrateError::failed("the resume states", error)));
    }
}

/// Replaces the states of the resume file with `region`'s, as the point to resume after `done` layers from. The header
/// first names no layers and, only once every chunk is written, names `done`: a stop in between resumes from the start.
pub fn write_checkpoint<F: StatesFile>(file: &F, key: &str, region: Region, chunk: usize, done: usize, then: Then<()>) {
    let then = single(then);
    let failed = Arc::clone(&then);
    let next = file.clone();
    let key = key.to_owned();
    let started = file.set_header(
        &resume_header(&key, 0),
        Box::new(move |set| {
            match set {
                Err(error) => end(&then, Err(CalibrateError::failed("the resume header", error))),
                Ok(()) => write_from(next, key, region, chunk.max(1), 0, done, then),
            }
            idle()
        }),
    );
    if let Err(error) = started {
        end(&failed, Err(CalibrateError::failed("the resume header", error)));
    }
}

fn write_from<F: StatesFile>(
    file: F,
    key: String,
    region: Region,
    chunk: usize,
    at: usize,
    done: usize,
    then: Once<()>,
) {
    let failed = Arc::clone(&then);
    if at >= region.length {
        let started = file.set_header(
            &resume_header(&key, done),
            Box::new(move |set| {
                end(
                    &then,
                    set.map_err(|error| CalibrateError::failed("the resume header", error)),
                );
                idle()
            }),
        );
        if let Err(error) = started {
            end(&failed, Err(CalibrateError::failed("the resume header", error)));
        }
        return;
    }
    let length = chunk.min(region.length - at);
    // SAFETY: the range is inside the region, which nothing else touches (the contract of `Region`).
    let bytes = unsafe { slice::from_raw_parts(region.pointer.add(at), length) }.to_vec();
    let next = file.clone();
    let started = file.write(
        HEADER + at as u64,
        bytes,
        Box::new(move |wrote: Result<u64, ResumeError>| {
            match wrote {
                Err(error) => end(&then, Err(CalibrateError::failed("the resume states", error))),
                Ok(_) => write_from(next, key, region, chunk, at + length, done, then),
            }
            idle()
        }),
    );
    if let Err(error) = started {
        end(&failed, Err(CalibrateError::failed("the resume states", error)));
    }
}
