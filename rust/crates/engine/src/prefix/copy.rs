//! The copy jobs of the prefix cache: the pieces that move one checkpoint between a sequence's buffers and the host
//! arena, and the guard that tells the cache how the job ended.
//!
//! A job is built by the cache's owner and admitted by the *sequence's* owner, behind the sequence's own last device work
//! (see [`Order::admit`](crate::generation::Order) with `on_workspace` false), so that it reads the state the sequence
//! had when the job was built, or writes the state its next quantum reads. A job always ends in exactly one report to
//! the cache: [`finish`](CopyJob::finish) with the way it ran, or, when it is dropped without having run (refused, or
//! abandoned with its owner), a failure. The cache unpins or aborts on either.
//!
//! **Admission order of a sequence's work.** The sequence's owner admits the job. A capture taken right after a prefill
//! chunk must be admitted with the chunk's `more` set, because sequence-only work admitted after a quantum that ended its
//! chain gets neither an edge nor a gate (the chain's pins were already dropped and the quantum may have retired); and
//! the next chunk, which overwrites the GDN state the capture reads, must be admitted behind the job.

use std::fmt;
use std::sync::Arc;

use euhedral_graph::Marker;
use euhedral_inference_cuda::PinnedBuffer;
use euhedral_inference_settings::KvFormat;

use crate::generation::Address;
use crate::prefix::Buffer;
use crate::prefix::cache::{Message, PrefixCache};
use crate::state::SequenceId;

/// Which way a job copies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    /// From the sequence's buffers into the arena: a capture.
    ToHost,
    /// From the arena into the sequence's buffers: a restore.
    ToDevice,
}

/// One copy of at most the piece size: `bytes` between the arena at `host` and the device at `device`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Piece {
    /// The offset in the arena.
    pub host: u64,
    /// The device address (of a buffer the job holds).
    pub device: u64,
    pub bytes: u64,
}

/// How a job ended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Ended {
    /// Every piece ran.
    Done,
    /// The job did not run to its end: refused, killed, or failed; what it says.
    Failed(String),
}

/// The host memory a job copies to and from.
pub trait HostArena: Send + Sync + 'static {
    /// The pinned buffer the offsets of a job address, when there is one (the device copier needs it).
    fn pinned(&self) -> Option<&PinnedBuffer>;
}

/// The device addresses of a sequence's state, and what a capture needs to know of it.
pub trait Resolve: Send + Sync + 'static {
    /// The device address of byte `offset` of `buffer`, with at least `bytes` bytes from there; `None` when the
    /// sequence has no such buffer or it is not on the device.
    fn address(&self, buffer: Buffer, offset: u64, bytes: u64) -> Option<u64>;

    /// The format the sequence keeps its attention caches in.
    fn kv_format(&self) -> KvFormat;

    /// The rows the sequence's attention caches hold.
    fn rows(&self) -> u64;

    /// The marker of the work that allocated the pages the sequence's cache reserved, while it has not completed:
    /// a restore waits for it on the device.
    fn ready(&self) -> Option<Marker>;
}

/// What a job needs besides its pieces: whose it is, and where it reports.
pub(crate) struct Link {
    pub(crate) sequence: SequenceId,
    pub(crate) host: Arc<dyn HostArena>,
    /// Keeps the sequence's buffers alive until the job is dropped.
    pub(crate) buffers: Arc<dyn Resolve>,
    pub(crate) ticket: u64,
    pub(crate) cache: Arc<Address<PrefixCache>>,
}

/// What runs the job once admitted, and what ends it: held by the job and dropped with it.
pub struct CopyJob {
    serial: u64,
    sequence: SequenceId,
    direction: Direction,
    pieces: Vec<Piece>,
    host: Arc<dyn HostArena>,
    buffers: Arc<dyn Resolve>,
    ticket: u64,
    report: Option<Arc<Address<PrefixCache>>>,
}

impl CopyJob {
    pub(crate) fn new(serial: u64, direction: Direction, pieces: Vec<Piece>, link: Link) -> CopyJob {
        CopyJob {
            serial,
            sequence: link.sequence,
            direction,
            pieces,
            host: link.host,
            buffers: link.buffers,
            ticket: link.ticket,
            report: Some(link.cache),
        }
    }

    /// The job's number, distinct from every quantum's ([`Quantum::id`](crate::stages::Quantum::id)).
    pub fn serial(&self) -> u64 {
        self.serial
    }

    pub fn sequence(&self) -> SequenceId {
        self.sequence
    }

    pub fn direction(&self) -> Direction {
        self.direction
    }

    pub fn pieces(&self) -> &[Piece] {
        &self.pieces
    }

    pub fn host(&self) -> &Arc<dyn HostArena> {
        &self.host
    }

    /// The marker a restore's copies wait for, if any.
    pub fn ready(&self) -> Option<Marker> {
        self.buffers.ready()
    }

    /// Tells the cache how the job ended. Once: a second call, and the drop, say nothing.
    pub fn finish(&mut self, ended: Ended) {
        if let Some(cache) = self.report.take() {
            cache.post(Message::Done {
                ticket: self.ticket,
                ended,
            });
        }
    }
}

impl Drop for CopyJob {
    fn drop(&mut self) {
        self.finish(Ended::Failed("the copy was dropped before it ran".to_string()));
    }
}

impl fmt::Debug for CopyJob {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CopyJob")
            .field("serial", &self.serial)
            .field("sequence", &self.sequence)
            .field("direction", &self.direction)
            .field("pieces", &self.pieces.len())
            .finish_non_exhaustive()
    }
}

/// Cuts `bytes` at `host`/`device` into pieces of at most `limit` bytes, appending them to `pieces`.
pub(crate) fn cut(pieces: &mut Vec<Piece>, host: u64, device: u64, bytes: u64, limit: u64) {
    let limit = limit.max(1);
    let mut done = 0;
    while done < bytes {
        let take = (bytes - done).min(limit);
        pieces.push(Piece {
            host: host + done,
            device: device + done,
            bytes: take,
        });
        done += take;
    }
}
