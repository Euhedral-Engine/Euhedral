//! A record read from the artifact: in page-aligned parts, each read straight into its place in the destination slot.
//!
//! One load is `parts` reads. Each part's end is a frame (`Kind::ExpertRead`, one chain per part) that checks what
//! arrived, copies the part's range on to the staging slot when the load fills a tier slot, and joins: a counter
//! the load holds counts the parts down, and the frame that brings it to zero finishes the load. The call that starts
//! the load holds one count itself, so the load never finishes inside the call: when every part ended before the call
//! returned, the call hands back the finishing frame.
//!
//! A load ends once, through its `done`, with a [`ReadOutcome`]; its leases are let go before `done` runs.
//!
//! ## Cancellation
//!
//! The artifact's ring cannot cancel a file read, so a read that is out runs to its end and its memory stays leased
//! until it does. A cancelled load that reads into staging alone (a bypass) submits no part it has not submitted and
//! ends as [`ReadOutcome::Cancelled`]. A load that fills a tier slot reads on (the record is worth keeping) and skips
//! only the copy to staging; it then ends as `Cancelled { filled: true }`, so `Complete` always means staging holds the
//! record.

use std::fmt::{self, Formatter};
use std::io;
use std::iter::once;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

use euhedral_core::frames::Frame;
use euhedral_data_structures::sync::ClaimCell;
use euhedral_io::{AsyncIo, Continuation, IoDoneOwned, ReadOptions};

use super::arena::{SlotWrite, WritePart};
use super::layout::{ExpertError, ExpertId, ExpertLayout};
use crate::frames::{Kind, step};
use crate::residency::disk::{ArtifactFile, DiskReadPiece, refuses_alignment};
use crate::residency::offload::HOST_ALIGNMENT;
use crate::runtime::raw_file;

/// Where a part's read goes: the artifact and the ring that reads it. A fake stands in for it in tests.
pub trait PartPort: Send + Sync + 'static {
    /// Starts reading `memory.len()` bytes at `offset` into `memory`, directly (past the page cache) or not; `then`'s
    /// frame is emitted when the read ended (maybe before `start` returns). An `Err` started nothing and drops `memory`
    /// and `then` unrun.
    fn start(
        &self,
        offset: u64,
        direct: bool,
        memory: WritePart,
        then: Continuation<IoDoneOwned<WritePart>>,
    ) -> io::Result<()>;

    /// Whether a direct read can be asked for.
    fn has_direct(&self) -> bool;
}

/// The artifact read through [`AsyncIo`]. Where io_uring is unavailable (or `positional_io` is set) the ring's own
/// positional engine makes the reads, in its reap frame: this port neither knows nor cares.
#[derive(Debug)]
pub struct FilePartPort {
    io: Arc<AsyncIo>,
    file: Arc<ArtifactFile>,
    piece: DiskReadPiece,
}

impl FilePartPort {
    pub fn new(io: Arc<AsyncIo>, file: Arc<ArtifactFile>, piece: DiskReadPiece) -> FilePartPort {
        FilePartPort { io, file, piece }
    }
}

impl PartPort for FilePartPort {
    fn start(
        &self,
        offset: u64,
        direct: bool,
        memory: WritePart,
        then: Continuation<IoDoneOwned<WritePart>>,
    ) -> io::Result<()> {
        let direct = direct && self.file.has_direct();
        let options = if direct {
            self.piece.options()
        } else {
            ReadOptions::default()
        };
        self.io
            .read_into_with(raw_file_of(&self.file, direct), offset, memory, options, then)
    }

    fn has_direct(&self) -> bool {
        self.file.has_direct()
    }
}

fn raw_file_of(file: &ArtifactFile, direct: bool) -> euhedral_io::RawFile {
    raw_file(file.handle(direct))
}

/// How a load ended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReadOutcome {
    /// Every byte of the record is in the destination and, when there is a copy, in staging.
    Complete,
    /// The load was cancelled: staging does not hold the record. `filled` says the destination, a tier slot the load
    /// read on into, holds the whole record, checked; otherwise the destination holds no usable record.
    Cancelled { filled: bool },
    /// A part failed, or the record is damaged; the destination holds no usable record.
    Failed(ExpertError),
    /// A read's memory was leaked by the ring (it broke): the destination slot is out of use for good.
    Lost,
}

/// What a finished load tells.
pub type Done = Box<dyn FnOnce(ReadOutcome) + Send>;

/// One record to read.
pub struct ReadLoad {
    pub id: ExpertId,
    /// Where the record goes (a tier slot, or staging when no tier slot is taken).
    pub destination: SlotWrite,
    /// A staging slot that gets a copy of each part as it arrives, when `destination` is a tier slot.
    pub copy_to: Option<SlotWrite>,
    /// Set by whoever no longer needs the record.
    pub cancel: Arc<AtomicBool>,
    /// Called once, from the frame that ended the load, with the leases already let go.
    pub done: Done,
}

/// What [`FileRecordSource::begin`] left running.
pub enum Started {
    /// The parts are out: `done` is called from the frame of the last to end.
    Reading,
    /// Every part had ended already: the frame to emit finishes the load.
    Frame(Box<dyn Frame>),
}

/// Reads records of the artifact into slots.
pub struct FileRecordSource {
    layout: Arc<ExpertLayout>,
    port: Arc<dyn PartPort>,
    split: usize,
}

impl FileRecordSource {
    /// A source of the records of `layout`, each read as up to `split` parts. The records are trusted as the artifact holds
    /// them: the file is checked when it is opened, not on every read. Refuses a split of 0.
    pub fn new(
        layout: Arc<ExpertLayout>,
        port: Arc<dyn PartPort>,
        split: usize,
    ) -> Result<FileRecordSource, ExpertError> {
        if split == 0 {
            return Err(ExpertError::BadSplit { split });
        }
        Ok(FileRecordSource { layout, port, split })
    }

    pub fn layout(&self) -> &Arc<ExpertLayout> {
        &self.layout
    }

    /// The bytes of a part: the record split evenly, each part's start a whole number of pages.
    fn chunk(&self, bytes: usize) -> usize {
        bytes.div_ceil(self.split).next_multiple_of(HOST_ALIGNMENT as usize)
    }

    /// Starts reading `load.id`. Refuses an unknown id, and a slot too small for the record, before anything starts: the
    /// `done` is then dropped uncalled and the leases are let go with `load`.
    pub fn begin(&self, load: ReadLoad) -> Result<Started, ExpertError> {
        let ReadLoad {
            id,
            mut destination,
            mut copy_to,
            cancel,
            done,
        } = load;
        let record = self.layout.record(id)?;
        let bytes = record.bytes as usize;
        let needed = bytes.next_multiple_of(HOST_ALIGNMENT as usize);
        let too_small = |slot_bytes: usize| ExpertError::SlotTooSmall {
            bank: self.layout.bank_name(id.bank).to_owned(),
            expert: id.expert,
            slot_bytes,
            needed,
        };
        for slot in once(&destination).chain(copy_to.iter()) {
            if slot.capacity() < needed {
                return Err(too_small(slot.capacity()));
            }
        }
        let chunk = self.chunk(bytes);
        let count = bytes.div_ceil(chunk);
        let block = HOST_ALIGNMENT;
        let mut jobs = Vec::with_capacity(count);
        for index in 0..count {
            let start = index * chunk;
            let len = chunk.min(bytes - start);
            let offset = record.offset + start as u64;
            // A direct read moves whole pages: the last part's tail is read too, when the slot has room for it and the
            // file has the bytes (the record that follows, which the next read of that record reads again).
            let rounded = len.next_multiple_of(block as usize);
            let direct = self.port.has_direct()
                && offset.is_multiple_of(block)
                && offset + rounded as u64 <= self.layout.file_bytes();
            let read_len = if direct { rounded } else { len };
            // The slot has room: `needed` covers the rounded tail of the last part and the parts before it are whole.
            let memory = destination
                .cut(read_len)
                .ok_or_else(|| too_small(destination.capacity()))?;
            let copy = match copy_to.as_mut() {
                Some(slot) => Some(slot.cut(len).ok_or_else(|| too_small(slot.capacity()))?),
                None => None,
            };
            jobs.push(PartJob {
                index,
                offset,
                len,
                direct,
                copy,
                memory,
            });
        }
        let shared = Arc::new(Shared {
            pending: AtomicUsize::new(count + 1),
            leaked: AtomicBool::new(false),
            skipped: AtomicBool::new(false),
            uncopied: AtomicBool::new(false),
            failure: OnceLock::new(),
            layout: Arc::clone(&self.layout),
            port: Arc::clone(&self.port),
            id,
            fills: copy_to.is_some(),
            cancel,
            finish: ClaimCell::new(Some(Finish {
                destination,
                copy_to,
                done,
            })),
        });
        for job in jobs {
            if shared.skips_reads() {
                shared.skipped.store(true, Ordering::Release);
                drop(job);
                shared.tick();
                continue;
            }
            submit(&shared, job);
        }
        // This call's own count: if it is the last, nothing is left to call `done` but a frame.
        if shared.pending.fetch_sub(1, Ordering::AcqRel) == 1 {
            let frame = step(Kind::ExpertRead, count, move || shared.finish());
            return Ok(Started::Frame(frame));
        }
        Ok(Started::Reading)
    }
}

/// One part: where it reads from, its memory, and where its bytes are copied on to.
struct PartJob {
    index: usize,
    offset: u64,
    /// The bytes of the record in this part (the read may move more, to a whole page).
    len: usize,
    direct: bool,
    copy: Option<WritePart>,
    memory: WritePart,
}

/// What the load joins on.
struct Shared {
    /// Parts not ended, and the starting call's own count.
    pending: AtomicUsize,
    /// Some part's memory was leaked by the ring.
    leaked: AtomicBool,
    /// Some part was not read because the load was cancelled.
    skipped: AtomicBool,
    /// Some part was read but not copied to staging because the load was cancelled.
    uncopied: AtomicBool,
    /// The first part's failure.
    failure: OnceLock<ExpertError>,
    layout: Arc<ExpertLayout>,
    port: Arc<dyn PartPort>,
    id: ExpertId,
    /// Whether the destination is a tier slot (which a cancel does not stop reading).
    fills: bool,
    cancel: Arc<AtomicBool>,
    /// Taken once, by the one that brings `pending` to zero.
    finish: ClaimCell<Option<Finish>>,
}

/// What ending the load needs, held until it ends.
struct Finish {
    destination: SlotWrite,
    copy_to: Option<SlotWrite>,
    done: Done,
}

impl Shared {
    /// Whether the parts not yet submitted are not to be: a cancelled load that fills nothing but staging.
    fn skips_reads(&self) -> bool {
        !self.fills && self.cancel.load(Ordering::Acquire)
    }

    fn fail(&self, error: ExpertError) {
        // The first failure is the one told; a later one is a consequence or a duplicate.
        let _ = self.failure.set(error);
    }

    fn read_error(&self, error: &io::Error) -> ExpertError {
        ExpertError::Read {
            bank: self.layout.bank_name(self.id.bank).to_owned(),
            expert: self.id.expert,
            kind: error.kind(),
            message: error.to_string(),
        }
    }

    /// The outcome of a load whose destination holds the whole record: complete unless staging missed a copy.
    fn complete(&self) -> ReadOutcome {
        if self.uncopied.load(Ordering::Acquire) {
            ReadOutcome::Cancelled { filled: true }
        } else {
            ReadOutcome::Complete
        }
    }

    /// One part ended (its memory already dropped): the last one finishes the load.
    fn tick(&self) {
        if self.pending.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.finish();
        }
    }

    /// Ends the load: its outcome, the leases let go, then `done`.
    fn finish(&self) {
        let Some(mut claim) = self.finish.try_claim() else {
            return;
        };
        let Some(Finish {
            destination,
            copy_to,
            done,
        }) = claim.take()
        else {
            return;
        };
        drop(claim);
        let outcome = if self.leaked.load(Ordering::Acquire) {
            ReadOutcome::Lost
        } else if self.skipped.load(Ordering::Acquire) {
            ReadOutcome::Cancelled { filled: false }
        } else if let Some(error) = self.failure.get() {
            ReadOutcome::Failed(error.clone())
        } else {
            self.complete()
        };
        drop((destination, copy_to));
        done(outcome);
    }
}

/// Submits `job`'s read; its end is a frame on the part's chain.
fn submit(shared: &Arc<Shared>, job: PartJob) {
    let PartJob {
        index,
        offset,
        len,
        direct,
        copy,
        memory,
    } = job;
    let wanted = memory.len();
    let ended = Arc::clone(shared);
    let then: Continuation<IoDoneOwned<WritePart>> = Box::new(move |done| {
        step(Kind::ExpertRead, index, move || {
            arrived(
                &ended,
                PartEnd {
                    index,
                    offset,
                    len,
                    direct,
                    wanted,
                    copy,
                },
                done,
            );
        })
    });
    if let Err(error) = shared.port.start(offset, direct, memory, then) {
        // Nothing started: the memory and the frame are gone, unrun.
        let error = shared.read_error(&error);
        shared.fail(error);
        shared.tick();
    }
}

/// A part's place in its record, carried to the frame that ends it.
struct PartEnd {
    index: usize,
    offset: u64,
    len: usize,
    direct: bool,
    wanted: usize,
    copy: Option<WritePart>,
}

/// A part's read ended: check it, copy it on, and join.
fn arrived(shared: &Arc<Shared>, end: PartEnd, done: IoDoneOwned<WritePart>) {
    let PartEnd {
        index,
        offset,
        len,
        direct,
        wanted,
        mut copy,
    } = end;
    let IoDoneOwned { buf, result, .. } = done;
    let Some(mut memory) = buf else {
        // The ring broke and kept the memory: the slot is lost.
        shared.leaked.store(true, Ordering::Release);
        drop(copy);
        shared.tick();
        return;
    };
    match result {
        Ok(moved) if moved == wanted => {
            if let Some(copy) = copy.as_mut() {
                if shared.cancel.load(Ordering::Acquire) {
                    shared.uncopied.store(true, Ordering::Release);
                } else {
                    copy.bytes_mut().copy_from_slice(&memory.bytes()[..len]);
                }
            }
        }
        Ok(moved) => shared.fail(ExpertError::Short {
            bank: shared.layout.bank_name(shared.id.bank).to_owned(),
            expert: shared.id.expert,
            offset,
            wanted,
            got: moved,
        }),
        Err(error) if direct && error.raw_os_error().is_some_and(refuses_alignment) => {
            // The file system cannot align this request: once more, through the page cache, for the record's own bytes.
            if shared.skips_reads() {
                shared.skipped.store(true, Ordering::Release);
            } else {
                memory.shorten(len);
                submit(
                    shared,
                    PartJob {
                        index,
                        offset,
                        len,
                        direct: false,
                        copy,
                        memory,
                    },
                );
                return;
            }
        }
        Err(error) => {
            let error = shared.read_error(&error);
            shared.fail(error);
        }
    }
    drop((memory, copy));
    shared.tick();
}

impl fmt::Debug for ReadLoad {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ReadLoad")
            .field("id", &self.id)
            .finish_non_exhaustive()
    }
}

impl fmt::Debug for Started {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Started::Reading => formatter.write_str("Reading"),
            Started::Frame(_) => formatter.write_str("Frame"),
        }
    }
}

impl fmt::Debug for FileRecordSource {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FileRecordSource")
            .field("split", &self.split)
            .finish_non_exhaustive()
    }
}
