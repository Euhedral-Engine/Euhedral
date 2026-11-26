//! The partial file of a reference: `OUT.partial`, of its final size plus 8 bytes, which hold the number of chunks that
//! are complete (a little-endian u64 after the last record). Rows go to their final offsets; a chunk is recorded
//! complete only after its rows are durable, so a stop at any point leaves a count whose chunks are whole.
//!
//! Every call returns at once and ends in its continuation, which runs in a frame. A call that cannot start ends its
//! continuation at once, in the caller's frame, with the error.

use std::ffi::OsString;
use std::fmt::{Debug, Formatter, Result as FmtResult};
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use euhedral_graph::Lake;
use euhedral_inference_engine::Handoff;
use euhedral_io::{AsyncIo, OpenSpec};

use super::{MIN_CONTEXT, ReferenceError};
use crate::kld::{Reference, encode_head};
use crate::resume::{ResumeError, ResumeFile, Then};
use crate::step::{Kind, step};

/// What a call's continuation is told.
pub type Done<T> = Box<dyn FnOnce(Result<T, ReferenceError>) + Send>;

/// The bytes of the count of complete chunks.
const COUNT_BYTES: usize = 8;

/// Starts a call on a file with a continuation that hands its result to `then` in a frame. A call that cannot start
/// ends `then` with its error, at once.
fn on_file<T: Send + 'static>(then: Done<T>, call: impl FnOnce(Then<T>) -> Result<(), ResumeError>) {
    let cell = Handoff::new();
    cell.send(then);
    let waiting = Arc::clone(&cell);
    let started = call(Box::new(move |result| {
        step(Kind::Resolve, move || {
            if let Some(then) = waiting.take() {
                then(result.map_err(ReferenceError::from));
            }
        })
    }));
    if let Err(error) = started
        && let Some(then) = cell.take()
    {
        then(Err(error.into()));
    }
}

/// The name of the partial file of `out`.
pub fn partial_path(out: &Path) -> PathBuf {
    let mut name: OsString = out.as_os_str().to_owned();
    name.push(".partial");
    PathBuf::from(name)
}

/// The offset of the count of complete chunks.
fn count_offset(layout: &Reference) -> u64 {
    layout.row_offset(layout.chunks(), 0)
}

/// An open partial file and the reference it holds.
pub struct Partial {
    file: ResumeFile,
    layout: Arc<Reference>,
    out: PathBuf,
    completed: usize,
}

/// The ways to write rows: a handle to the file and the layout. Cheap to copy.
#[derive(Clone)]
pub struct RowFile {
    file: ResumeFile,
    layout: Arc<Reference>,
}

/// The way to record chunks complete.
#[derive(Clone)]
pub struct Mark {
    file: ResumeFile,
    offset: u64,
}

/// What opening keeps while it goes through its steps.
struct Opening {
    io: Arc<AsyncIo>,
    host: Arc<Lake>,
    out: PathBuf,
    layout: Arc<Reference>,
    head: Vec<u8>,
    then: Arc<Handoff<Done<Partial>>>,
}

impl Opening {
    fn end(&self, result: Result<Partial, ReferenceError>) {
        if let Some(then) = self.then.take() {
            then(result);
        }
    }

    fn ready(self: &Arc<Self>, file: ResumeFile, completed: usize) {
        self.end(Ok(Partial {
            file,
            layout: Arc::clone(&self.layout),
            out: self.out.clone(),
            completed,
        }));
    }
}

impl Partial {
    /// Opens the partial file of `out` for `tokens` (a chunk each, `context` tokens long, over `vocabulary`), starting
    /// it, or taking up the one a stopped run left. It is an error if `out` exists, and if the partial file was
    /// started for other tokens, another context or another vocabulary.
    pub fn open(io: &Arc<AsyncIo>, host: &Arc<Lake>, out: &Path, reference: Reference, then: Done<Partial>) {
        let Reference {
            context,
            vocabulary,
            tokens,
        } = reference;
        if let Err(error) = Self::check(context, vocabulary, &tokens) {
            then(Err(error));
            return;
        }
        let head = encode_head(context, vocabulary, &tokens);
        let cell = Handoff::new();
        cell.send(then);
        let opening = Arc::new(Opening {
            io: Arc::clone(io),
            host: Arc::clone(host),
            out: out.to_owned(),
            layout: Arc::new(Reference {
                context,
                vocabulary,
                tokens,
            }),
            head,
            then: cell,
        });
        Self::absent(&opening);
    }

    fn check(context: usize, vocabulary: usize, tokens: &[Vec<u32>]) -> Result<(), ReferenceError> {
        if context < MIN_CONTEXT || vocabulary == 0 || tokens.is_empty() {
            return Err(ReferenceError::Refused(format!(
                "a reference needs a context of at least {MIN_CONTEXT} tokens, a vocabulary and a chunk"
            )));
        }
        for (chunk, ids) in tokens.iter().enumerate() {
            if ids.len() != context {
                return Err(ReferenceError::Refused(format!(
                    "chunk {chunk} is not {context} tokens"
                )));
            }
            if let Some(token) = ids.iter().find(|token| **token as usize >= vocabulary) {
                return Err(ReferenceError::Refused(format!(
                    "chunk {chunk} holds token {token}, outside the vocabulary of {vocabulary}"
                )));
            }
        }
        Ok(())
    }

    /// The first step: the output must not exist.
    fn absent(opening: &Arc<Opening>) {
        let next = Arc::clone(opening);
        let probed = opening.io.open_path(
            &opening.out,
            OpenSpec::default(),
            Box::new(move |found| {
                step(Kind::Resolve, move || match found {
                    Ok(_) => next.end(Err(ReferenceError::Refused(format!(
                        "{} exists; remove it to write it again",
                        next.out.display()
                    )))),
                    Err(error) if error.kind() == ErrorKind::NotFound => Self::open_partial(&next),
                    Err(error) => next.end(Err(ReferenceError::io("cannot look for the output", &error))),
                })
            }),
        );
        if let Err(error) = probed {
            opening.end(Err(ReferenceError::io("cannot look for the output", &error)));
        }
    }

    fn open_partial(opening: &Arc<Opening>) {
        let next = Arc::clone(opening);
        let path = partial_path(&opening.out);
        let (io, host) = (Arc::clone(&opening.io), Arc::clone(&opening.host));
        on_file(
            Box::new(move |opened| match opened {
                Ok(file) => Self::sized(&next, file),
                Err(error) => next.end(Err(error)),
            }),
            move |then| ResumeFile::open(&io, &host, &path, then),
        );
    }

    /// The size says whether the file is new (empty), a stopped run's (the whole size), a finished run's cut before its rename (no count) or another's.
    fn sized(opening: &Arc<Opening>, file: ResumeFile) {
        let (next, held) = (Arc::clone(opening), file.clone());
        on_file(
            Box::new(move |size| match size {
                Ok(size) => Self::inspect(&next, held, size),
                Err(error) => next.end(Err(error)),
            }),
            move |then| file.len(then),
        );
    }

    fn inspect(opening: &Arc<Opening>, file: ResumeFile, size: u64) {
        let whole = count_offset(&opening.layout) + COUNT_BYTES as u64;
        if size == 0 {
            Self::start(opening, file);
        } else if size == whole {
            Self::compare(opening, file, false);
        } else if size == whole - COUNT_BYTES as u64 {
            Self::compare(opening, file, true);
        } else {
            opening.end(Err(ReferenceError::Mismatch(opening.out.display().to_string())));
        }
    }

    /// A new file: the header and tokens, then the count (which gives the file its size), then durable.
    fn start(opening: &Arc<Opening>, file: ResumeFile) {
        let (next, held) = (Arc::clone(opening), file.clone());
        let head = opening.head.clone();
        on_file(
            Box::new(move |written| match written {
                Ok(_) => Self::zero_count(&next, held),
                Err(error) => next.end(Err(error)),
            }),
            move |then| file.write(0, head, then),
        );
    }

    fn zero_count(opening: &Arc<Opening>, file: ResumeFile) {
        let (next, held) = (Arc::clone(opening), file.clone());
        let offset = count_offset(&opening.layout);
        on_file(
            Box::new(move |written| match written {
                Ok(_) => Self::durable(&next, held),
                Err(error) => next.end(Err(error)),
            }),
            move |then| file.write(offset, vec![0; COUNT_BYTES], then),
        );
    }

    fn durable(opening: &Arc<Opening>, file: ResumeFile) {
        let (next, held) = (Arc::clone(opening), file.clone());
        on_file(
            Box::new(move |synced| match synced {
                Ok(()) => next.ready(held, 0),
                Err(error) => next.end(Err(error)),
            }),
            move |then| file.sync(then),
        );
    }

    /// A file of the right size: it must start with this run's header and tokens. A file without its count is one
    /// whose sealing cut the count off and stopped before the rename: every chunk is complete.
    fn compare(opening: &Arc<Opening>, file: ResumeFile, cut: bool) {
        let (next, held) = (Arc::clone(opening), file.clone());
        let length = opening.head.len();
        on_file(
            Box::new(move |found| match found {
                Ok(found) if found == next.head && cut => {
                    let chunks = next.layout.chunks();
                    next.ready(held, chunks);
                }
                Ok(found) if found == next.head => Self::count(&next, held),
                Ok(_) => next.end(Err(ReferenceError::Mismatch(next.out.display().to_string()))),
                Err(error) => next.end(Err(error)),
            }),
            move |then| file.read(0, length, then),
        );
    }

    fn count(opening: &Arc<Opening>, file: ResumeFile) {
        let (next, held) = (Arc::clone(opening), file.clone());
        let offset = count_offset(&opening.layout);
        on_file(
            Box::new(move |found: Result<Vec<u8>, ReferenceError>| match found {
                Ok(bytes) => {
                    let stored = u64::from_le_bytes(bytes.try_into().unwrap_or_default());
                    match usize::try_from(stored) {
                        Ok(stored) if stored <= next.layout.chunks() => next.ready(held, stored),
                        _ => next.end(Err(ReferenceError::Refused(format!(
                            "{} counts {stored} complete chunks of {}",
                            next.out.display(),
                            next.layout.chunks()
                        )))),
                    }
                }
                Err(error) => next.end(Err(error)),
            }),
            move |then| file.read(offset, COUNT_BYTES, then),
        );
    }

    /// The chunks that are complete: the next to write is this one.
    pub fn completed(&self) -> usize {
        self.completed
    }

    /// The reference being written: its dimensions and tokens.
    pub fn layout(&self) -> &Arc<Reference> {
        &self.layout
    }

    /// The handle the rows of a chunk are written through.
    pub fn rows(&self) -> RowFile {
        RowFile {
            file: self.file.clone(),
            layout: Arc::clone(&self.layout),
        }
    }

    /// The handle that records chunks complete.
    pub fn mark(&self) -> Mark {
        Mark {
            file: self.file.clone(),
            offset: count_offset(&self.layout),
        }
    }

    /// Cuts the count off the file and renames it to the reference. Every chunk must be complete. The file must have
    /// no other handle ([`RowFile`], [`Mark`]) left.
    pub fn finish(self, then: Done<()>) {
        let Partial {
            file,
            layout,
            out,
            completed,
        } = self;
        if completed != layout.chunks() {
            then(Err(ReferenceError::Refused(format!(
                "{completed} of {} chunks written",
                layout.chunks()
            ))));
            return;
        }
        let end = count_offset(&layout);
        on_file(then, move |sealed| file.seal(end, &out, sealed));
    }

    /// Notes that a chunk is recorded complete.
    pub fn set_completed(&mut self, completed: usize) {
        self.completed = completed;
    }
}

impl RowFile {
    /// Writes `record` as record `step` of `chunk`.
    pub fn write(&self, chunk: usize, step: usize, record: Vec<u8>, then: Done<()>) {
        let layout = &self.layout;
        if chunk >= layout.chunks() || step >= layout.records() || record.len() != layout.row_bytes() {
            then(Err(ReferenceError::Refused(format!(
                "record {step} of chunk {chunk} does not fit"
            ))));
            return;
        }
        let offset = layout.row_offset(chunk, step);
        let file = self.file.clone();
        on_file(Box::new(move |written| then(written.map(|_| ()))), move |done| {
            file.write(offset, record, done)
        });
    }
}

impl Mark {
    /// Records that chunks `0` to `count - 1` are all written: their rows are made durable, then the count is, so the
    /// count never names a chunk that is not whole.
    pub fn complete(self, count: usize, then: Done<()>) {
        let file = self.file.clone();
        on_file(
            Box::new(move |synced| match synced {
                Ok(()) => self.store(count, then),
                Err(error) => then(Err(error)),
            }),
            move |done| file.sync(done),
        );
    }

    fn store(self, count: usize, then: Done<()>) {
        let (file, offset) = (self.file.clone(), self.offset);
        on_file(
            Box::new(move |written| match written {
                Ok(_) => self.flush(then),
                Err(error) => then(Err(error)),
            }),
            move |done| file.write(offset, (count as u64).to_le_bytes().to_vec(), done),
        );
    }

    fn flush(self, then: Done<()>) {
        let file = self.file.clone();
        on_file(then, move |done| file.sync(done));
    }
}

macro_rules! opaque {
    ($($name:ident),*) => {$(
        impl Debug for $name {
            fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
                formatter.debug_struct(stringify!($name)).finish_non_exhaustive()
            }
        }
    )*};
}

opaque!(Partial, RowFile, Mark);
