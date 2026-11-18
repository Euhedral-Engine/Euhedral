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
use std::mem;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

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
    /// The job did not run to its end: refused or failed; what it says.
    Failed(String),
    /// The job's kill flag was set before it ran to its end.
    Cancelled,
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
    /// The most bytes of one frame of the job: the job runs in groups of pieces that hold no more than this (a piece
    /// that is larger alone).
    pub(crate) group_bytes: u64,
    /// Set when the job fails (a restore's: the request must not go on over a half-written state), see [`CopyJob::finish`].
    pub(crate) on_failure: Option<Arc<AtomicBool>>,
}

/// What runs the job once admitted, and what ends it: held by the job and dropped with it.
pub struct CopyJob {
    serial: u64,
    sequence: SequenceId,
    direction: Direction,
    pieces: Vec<Piece>,
    host: Arc<dyn HostArena>,
    /// The sequence's buffers; let go when the job ends, before the cache is told, so that the answer finds the sequence
    /// at rest.
    buffers: Option<Arc<dyn Resolve>>,
    ticket: u64,
    report: Option<Arc<Address<PrefixCache>>>,
    group_bytes: u64,
    on_failure: Option<Arc<AtomicBool>>,
    /// Whether this group is the job's last, which alone reports.
    closing: bool,
    /// Whether this group has been told how it ended.
    ended: bool,
    /// Why a group of the job failed, once one did: the groups after it copy nothing and the last reports it.
    failed: Arc<OnceLock<String>>,
}

impl CopyJob {
    pub(crate) fn new(serial: u64, direction: Direction, pieces: Vec<Piece>, link: Link) -> CopyJob {
        CopyJob {
            serial,
            sequence: link.sequence,
            direction,
            pieces,
            host: link.host,
            buffers: Some(link.buffers),
            ticket: link.ticket,
            report: Some(link.cache),
            group_bytes: link.group_bytes,
            on_failure: link.on_failure,
            closing: true,
            ended: false,
            failed: Arc::new(OnceLock::new()),
        }
    }

    /// Cuts the job into the groups it runs in, one frame each, in order: each holds pieces of at most the group size
    /// in all. Only the last reports to the cache, and it reports a failure of any group.
    pub fn split(mut self) -> Vec<CopyJob> {
        let mut groups: Vec<Vec<Piece>> = vec![Vec::new()];
        let mut held = 0u64;
        for piece in mem::take(&mut self.pieces) {
            let full = groups.last().is_some_and(|group| !group.is_empty()) && held + piece.bytes > self.group_bytes;
            if full {
                groups.push(Vec::new());
                held = 0;
            }
            held += piece.bytes;
            if let Some(group) = groups.last_mut() {
                group.push(piece);
            }
        }
        let last = groups.len() - 1;
        let mut jobs = Vec::with_capacity(groups.len());
        for (index, pieces) in groups.into_iter().enumerate() {
            jobs.push(CopyJob {
                serial: self.serial,
                sequence: self.sequence,
                direction: self.direction,
                pieces,
                host: Arc::clone(&self.host),
                buffers: self.buffers.clone(),
                ticket: self.ticket,
                report: if index == last { self.report.take() } else { None },
                group_bytes: self.group_bytes,
                on_failure: self.on_failure.clone(),
                closing: index == last,
                ended: false,
                failed: Arc::clone(&self.failed),
            });
        }
        // The groups carry the job on: this one has nothing left to tell.
        self.ended = true;
        jobs
    }

    /// Whether this group is the job's last: the one that reports.
    pub fn is_last(&self) -> bool {
        self.closing
    }

    /// Whether an earlier group of the job failed: this one copies nothing.
    pub fn skipped(&self) -> bool {
        self.failed.get().is_some()
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
        self.buffers.as_ref().and_then(|buffers| buffers.ready())
    }

    /// Tells the cache how the job ended. Once: a second call, and the drop, say nothing.
    pub fn finish(&mut self, ended: Ended) {
        if mem::replace(&mut self.ended, true) {
            return;
        }
        let ended = match (ended, self.failed.get()) {
            (Ended::Done, Some(why)) => Ended::Failed(why.clone()),
            (ended, _) => ended,
        };
        match &ended {
            Ended::Failed(why) => {
                // The first failure is the one told.
                let _ = self.failed.set(why.clone());
                // Before the order is told the job ended, so that what runs behind it sees the request ended.
                if let Some(flag) = &self.on_failure {
                    flag.store(true, Ordering::Release);
                }
            }
            Ended::Cancelled => {
                let _ = self.failed.set("the copy was cancelled".to_string());
            }
            Ended::Done => {}
        }
        if !self.closing {
            return;
        }
        self.buffers = None;
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
