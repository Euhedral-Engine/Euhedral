//! The prefix cache's owner: the tree of checkpoints, the host arena it leases from, and the copy jobs in flight.
//!
//! The cache is one owner in a [`Mailbox`]: nothing waits for it and nothing runs inline in its turn. A caller hands it a
//! request (a [`lookup`](PrefixCacheHandle::lookup), a [`restore`](PrefixCacheHandle::restore), a
//! [`capture`](PrefixCacheHandle::capture)) with a closure that answers it; the answer is told once the owner's claim is
//! free, so a listener may start the next request from it.
//!
//! **Copies.** The cache does not copy and does not admit: it plans the copies of a checkpoint, cuts them into pieces
//! of at most the piece size and builds a [`CopyJob`], which it gives to the `submit` closure of the request. That closure
//! is the sequence's owner's, which admits the job behind the sequence's own last device work, so that the cache never
//! orders a sequence's work itself. The job ends in one report to the cache ([`Message::Done`], or a failure when it was
//! dropped unrun), and the cache then publishes the node and counts the capture, or aborts it; a restore unpins its match
//! on either. The request is answered on every path: a copy that fails leaves nothing waiting.
//!
//! **A capture right after a prefill chunk** must be admitted with that chunk's `more` set (see [`copy`](super::copy)):
//! the job reads the GDN state the chunk left, and the next chunk, which overwrites it, is admitted behind the job.
//!
//! **What a match pins.** A [`Hit`] pins the nodes it matched until it is restored or dropped; a dropped one tells the
//! cache, which releases the pins.

use std::collections::HashMap;
use std::fmt::{Debug, Formatter, Result as FmtResult};
use std::mem;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use euhedral_data_structures::queues::{Consumer, MpscQueue, Single};
use euhedral_inference_settings::KvFormat;
use euhedral_memory::SlotArena;

use crate::generation::{Address, Mailbox, Spawn, Turn};
use crate::handoff::Handoff;
use crate::prefix::copy::{CopyJob, Direction, Ended, HostArena, Link, Piece, Resolve, cut};
use crate::prefix::{CheckpointLayout, Match, NodeId, PrefixTree, ReserveError, SlotCopy, Speculation};
use crate::state::SequenceId;

/// Slots in a chunk of the cache's inbox.
const INBOX_CHUNK: usize = 16;

/// Hands a job to the sequence's owner, which admits it. The job is dropped, and so reported failed, if it is not
/// admitted.
pub type Submit = Box<dyn FnOnce(CopyJob) + Send>;

/// What is posted to the cache.
pub(crate) enum Message {
    /// The job of `ticket` ended.
    Done { ticket: u64, ended: Ended },
    /// A hit was dropped without being restored.
    Unpin(Match),
}

/// What the cache tells, once its claim is free.
pub(crate) enum Finished {
    Told(Box<dyn FnOnce() + Send>),
}

/// The counters and sizes of a cache. The counters only grow; the times are in nanoseconds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PrefixCacheStats {
    pub lookups: u64,
    pub hits: u64,
    pub restored_tokens: u64,
    pub captures: u64,
    pub skipped: u64,
    pub failed: u64,
    pub evictions: u64,
    pub restores: u64,
    pub capture_nanos: u64,
    pub restore_nanos: u64,
    pub nodes: u64,
    pub used_bytes: u64,
    pub capacity_bytes: u64,
}

/// Why a restore did not restore.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum RestoreError {
    /// The request was cancelled: not an error to report.
    #[error("the request was cancelled")]
    Cancelled,
    #[error("the prefix cache is closed")]
    Closed,
    #[error("the hit does not hold the speculative state asked for")]
    Speculation,
    #[error("the sequence keeps its attention caches in {sequence:?}, the prefix cache in {cache:?}")]
    Format { sequence: KvFormat, cache: KvFormat },
    #[error("the sequence has no buffer for a copy of the checkpoint")]
    Unresolved,
    #[error("the sequence has {rows} rows of its attention caches reserved and the checkpoint is {position} tokens")]
    Unreserved { rows: u64, position: u32 },
    #[error("the copy failed: {0}")]
    Copy(String),
}

/// Why a cache could not be opened.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum OpenError {
    #[error("the arena's slots are {arena} bytes and a checkpoint's are {layout}")]
    SlotBytes { arena: u64, layout: u64 },
    #[error("the checkpoint interval must be a positive multiple of the chunk")]
    Interval,
    /// The arena cannot hold the checkpoint of one chunk of base state. Every checkpoint ends on a multiple of the
    /// chunk, so the span a capture adds to the tree is at least one chunk: an arena that holds one chunk's checkpoint
    /// holds any node.
    #[error("a chunk's checkpoint takes {needed} slots and the arena has {slots}")]
    BaseCapacity { needed: usize, slots: u32 },
    #[error("the requests' checkpoints hold {kind:?} state, which the layout has no room for")]
    NoState { kind: Speculation },
    /// The arena cannot hold one chunk's checkpoint of the draft state the requests store (`CacheParts::drafting`).
    /// The cache asks for room only for state its requests will store: an engine whose requests never draft opens with
    /// `drafting` unset and is sized for base state alone.
    #[error("a chunk's checkpoint that holds {kind:?} state takes {needed} slots and the arena has {slots}")]
    Capacity {
        kind: Speculation,
        needed: usize,
        slots: u32,
    },
}

/// A match the cache pinned: the position it restores, and the pins until it is restored or dropped.
#[must_use]
pub struct Hit {
    matched: Option<Match>,
    position: u32,
    paired: Option<u32>,
    leaf: NodeId,
    address: Arc<Address<PrefixCache>>,
}

impl Hit {
    /// The tokens the hit restores: the prompt resumes from here.
    pub fn position(&self) -> u32 {
        self.position
    }

    /// The token the deepest node's last draft row was paired with: the stored prompt's token at the hit's position.
    /// A prompt with the same token there restores the row whole.
    pub fn paired(&self) -> Option<u32> {
        self.paired
    }

    /// The deepest matched node: the parent of the next capture.
    pub fn leaf(&self) -> NodeId {
        self.leaf
    }

    fn take(mut self) -> Option<Match> {
        self.matched.take()
    }
}

impl Debug for Hit {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        formatter
            .debug_struct("Hit")
            .field("position", &self.position)
            .finish_non_exhaustive()
    }
}

impl Drop for Hit {
    fn drop(&mut self) {
        if let Some(matched) = self.matched.take() {
            self.address.post(Message::Unpin(matched));
        }
    }
}

/// The counters and limits the handle reads without asking the owner.
#[derive(Debug)]
struct Shared {
    interval: u32,
    chunk: u32,
    limit: AtomicU32,
    lookups: AtomicU64,
    hits: AtomicU64,
    restored_tokens: AtomicU64,
    captures: AtomicU64,
    skipped: AtomicU64,
    failed: AtomicU64,
    evictions: AtomicU64,
    restores: AtomicU64,
    capture_nanos: AtomicU64,
    restore_nanos: AtomicU64,
    nodes: AtomicU64,
    used_bytes: AtomicU64,
    capacity_bytes: AtomicU64,
    serials: AtomicU64,
}

fn count(counter: &AtomicU64, by: u64) {
    counter.fetch_add(by, Ordering::Relaxed);
}

fn nanos(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

enum Pending {
    Capture {
        node: NodeId,
        parent: NodeId,
        started: Instant,
        done: Box<dyn FnOnce(NodeId) + Send>,
    },
    Restore {
        matched: Match,
        position: u32,
        cancel: Arc<AtomicBool>,
        started: Instant,
        done: Box<dyn FnOnce(Result<u32, RestoreError>) + Send>,
    },
}

/// The cache's owner.
pub(crate) struct PrefixCache {
    inbox: Consumer<Message, Single>,
    tree: PrefixTree,
    arena: Arc<SlotArena>,
    layout: CheckpointLayout,
    host: Arc<dyn HostArena>,
    format: KvFormat,
    piece_bytes: u64,
    shared: Arc<Shared>,
    address: Arc<Address<PrefixCache>>,
    pending: HashMap<u64, Pending>,
    ticket: u64,
    told: Vec<Finished>,
    closed: bool,
    /// Told once the cache is shut and no copy job is left in flight.
    idle_signal: Option<Arc<Handoff<()>>>,
}

/// What a cache is opened over.
#[derive(Clone)]
pub struct CacheParts {
    pub layout: CheckpointLayout,
    pub arena: Arc<SlotArena>,
    pub host: Arc<dyn HostArena>,
    pub format: KvFormat,
    /// Tokens between checkpoints.
    pub interval: u32,
    /// Tokens a prefill chunk takes.
    pub chunk: u32,
    /// The draft state the requests' checkpoints hold beside the base state: the kind of the requests that draft, and
    /// `None` when no request does.
    pub drafting: Option<Speculation>,
    /// The most a copy piece moves.
    pub piece_bytes: u64,
    /// The first serial of the jobs it builds, so that no job's number is a quantum's.
    pub first_serial: u64,
}

impl Debug for CacheParts {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        formatter
            .debug_struct("CacheParts")
            .field("interval", &self.interval)
            .finish_non_exhaustive()
    }
}

impl PrefixCache {
    fn open(parts: CacheParts, spawn: Spawn) -> Result<PrefixCacheHandle, OpenError> {
        let CacheParts {
            layout,
            arena,
            host,
            format,
            interval,
            chunk,
            drafting,
            piece_bytes,
            first_serial,
        } = parts;
        if arena.slot_bytes() != layout.slot_bytes() {
            return Err(OpenError::SlotBytes {
                arena: arena.slot_bytes(),
                layout: layout.slot_bytes(),
            });
        }
        if interval == 0 || chunk == 0 || !interval.is_multiple_of(chunk) {
            return Err(OpenError::Interval);
        }
        // Refuses an arena that cannot hold one chunk's checkpoint.
        let base = layout.slots(0, chunk, None);
        if base > arena.capacity() as usize {
            return Err(OpenError::BaseCapacity {
                needed: base,
                slots: arena.capacity(),
            });
        }
        if let Some(kind) = drafting {
            if !layout.has_speculation(kind) {
                return Err(OpenError::NoState { kind });
            }
            let needed = layout.slots(0, chunk, Some(kind));
            if needed > arena.capacity() as usize {
                return Err(OpenError::Capacity {
                    kind,
                    needed,
                    slots: arena.capacity(),
                });
            }
        }
        let tree = PrefixTree::new(Arc::clone(&arena));
        let root = tree.root();
        let shared = Arc::new(Shared {
            interval,
            chunk,
            limit: AtomicU32::new(u32::MAX),
            lookups: AtomicU64::new(0),
            hits: AtomicU64::new(0),
            restored_tokens: AtomicU64::new(0),
            captures: AtomicU64::new(0),
            skipped: AtomicU64::new(0),
            failed: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            restores: AtomicU64::new(0),
            capture_nanos: AtomicU64::new(0),
            restore_nanos: AtomicU64::new(0),
            nodes: AtomicU64::new(0),
            used_bytes: AtomicU64::new(0),
            capacity_bytes: AtomicU64::new(u64::from(arena.capacity()) * arena.slot_bytes()),
            serials: AtomicU64::new(first_serial),
        });
        let queue = MpscQueue::<Message>::new(INBOX_CHUNK);
        let (producer, inbox) = (queue.producer(), queue.consumer());
        let address = Address::new();
        let cache = PrefixCache {
            inbox,
            tree,
            arena,
            layout,
            host,
            format,
            piece_bytes: piece_bytes.max(1),
            shared: Arc::clone(&shared),
            address: Arc::clone(&address),
            pending: HashMap::new(),
            ticket: 0,
            told: Vec::new(),
            closed: false,
            idle_signal: None,
        };
        let mailbox = Arc::new(Mailbox::new(cache, producer, spawn));
        address.set(&mailbox);
        Ok(PrefixCacheHandle { mailbox, shared, root })
    }

    fn told(&mut self, then: impl FnOnce() + Send + 'static) {
        self.told.push(Finished::Told(Box::new(then)));
    }

    fn publish_stats(&self) {
        let shared = &self.shared;
        shared.nodes.store(self.tree.size() as u64, Ordering::Relaxed);
        shared.evictions.store(self.tree.evictions(), Ordering::Relaxed);
        shared.used_bytes.store(
            u64::from(self.arena.leased()) * self.arena.slot_bytes(),
            Ordering::Relaxed,
        );
    }

    fn lookup(&mut self, prompt: &[u32], kind: Option<Speculation>) -> Option<Hit> {
        if self.closed {
            return None;
        }
        count(&self.shared.lookups, 1);
        let matched = self.tree.lookup(prompt, kind)?;
        let leaf = matched.leaf();
        let position = self.tree.get(leaf).map_or(0, |node| node.position());
        let paired = self.tree.get(leaf).and_then(|node| node.paired());
        count(&self.shared.hits, 1);
        count(&self.shared.restored_tokens, u64::from(position));
        Some(Hit {
            matched: Some(matched),
            position,
            paired,
            leaf,
            address: Arc::clone(&self.address),
        })
    }

    /// The pieces that move `copies` between the arena and `refs`' buffers; `None` when one has no buffer.
    fn pieces(&self, copies: &[SlotCopy], refs: &dyn Resolve) -> Option<Vec<Piece>> {
        let mut pieces = Vec::new();
        for copy in copies {
            let device = refs.address(copy.buffer, copy.offset, copy.bytes)?;
            cut(&mut pieces, copy.host, device, copy.bytes, self.piece_bytes);
        }
        Some(pieces)
    }

    fn job(
        &mut self,
        sequence: SequenceId,
        direction: Direction,
        pieces: Vec<Piece>,
        refs: Arc<dyn Resolve>,
        on_failure: Option<Arc<AtomicBool>>,
    ) -> (u64, CopyJob) {
        self.ticket += 1;
        let serial = self.shared.serials.fetch_add(1, Ordering::Relaxed);
        let link = Link {
            sequence,
            host: Arc::clone(&self.host),
            buffers: refs,
            ticket: self.ticket,
            cache: Arc::clone(&self.address),
            group_bytes: self.piece_bytes,
            on_failure,
        };
        (self.ticket, CopyJob::new(serial, direction, pieces, link))
    }

    /// The pieces of a restore of `matched`, or why it does not run.
    fn plan_restore(
        &self,
        matched: &Match,
        refs: &dyn Resolve,
        kind: Option<Speculation>,
        cancel: &AtomicBool,
        position: u32,
    ) -> Result<Vec<Piece>, RestoreError> {
        let chain: Vec<_> = matched.chain().iter().filter_map(|&id| self.tree.get(id)).collect();
        if kind.is_some() && chain.iter().any(|node| node.speculation() != kind) {
            return Err(RestoreError::Speculation);
        }
        if self.closed {
            return Err(RestoreError::Closed);
        }
        if refs.rows() < u64::from(position) {
            return Err(RestoreError::Unreserved {
                rows: refs.rows(),
                position,
            });
        }
        if refs.kv_format() != self.format {
            return Err(RestoreError::Format {
                sequence: refs.kv_format(),
                cache: self.format,
            });
        }
        if cancel.load(Ordering::Acquire) {
            return Err(RestoreError::Cancelled);
        }
        let copies = self.layout.restore_copies(&chain, kind);
        self.pieces(&copies, refs).ok_or(RestoreError::Unresolved)
    }

    #[allow(clippy::too_many_arguments, reason = "one request's parts, as the handle takes them")]
    fn begin_restore(
        &mut self,
        hit: Hit,
        sequence: SequenceId,
        refs: Arc<dyn Resolve>,
        kind: Option<Speculation>,
        cancel: Arc<AtomicBool>,
        submit: Submit,
        done: Box<dyn FnOnce(Result<u32, RestoreError>) + Send>,
    ) {
        let position = hit.position();
        let Some(matched) = hit.take() else {
            self.told(move || done(Err(RestoreError::Closed)));
            return;
        };
        let planned = self.plan_restore(&matched, refs.as_ref(), kind, &cancel, position);
        let pieces = match planned {
            Ok(pieces) => pieces,
            Err(error) => {
                self.tree.release(matched);
                self.told(move || done(Err(error)));
                return;
            }
        };
        let (ticket, job) = self.job(sequence, Direction::ToDevice, pieces, refs, Some(Arc::clone(&cancel)));
        self.pending.insert(
            ticket,
            Pending::Restore {
                matched,
                position,
                cancel,
                started: Instant::now(),
                done,
            },
        );
        self.told(move || submit(job));
    }

    #[allow(clippy::too_many_arguments, reason = "one request's parts, as the handle takes them")]
    fn begin_capture(
        &mut self,
        sequence: SequenceId,
        refs: Arc<dyn Resolve>,
        tokens: &[u32],
        parent: NodeId,
        end: u32,
        kind: Option<Speculation>,
        submit: Submit,
        done: Box<dyn FnOnce(NodeId) + Send>,
    ) {
        let started = Instant::now();
        let Some(start) = self.tree.get(parent).map(|node| node.position()) else {
            count(&self.shared.failed, 1);
            self.told(move || done(parent));
            return;
        };
        if self.closed || refs.kv_format() != self.format || refs.rows() < u64::from(end) {
            self.told(move || done(parent));
            return;
        }
        let kind = kind
            .filter(|&kind| self.layout.has_speculation(kind))
            .and_then(|kind| self.tree.effective_speculation(parent, Some(kind)));
        let existing = self.tree.find(parent, tokens, end, kind).or_else(|| {
            if kind.is_none() {
                self.tree.find_any(parent, tokens, end)
            } else {
                None
            }
        });
        if let Some(existing) = existing {
            self.told(move || done(existing));
            return;
        }
        let slots = self.layout.slots(start, end, kind);
        let node = match self.tree.reserve(parent, tokens, end, kind, slots) {
            Ok(node) => node,
            Err(error) => {
                count(
                    if error == ReserveError::NoRoom {
                        &self.shared.skipped
                    } else {
                        &self.shared.failed
                    },
                    1,
                );
                self.publish_stats();
                self.told(move || done(parent));
                return;
            }
        };
        let copies = self.tree.get(node).map(|reserved| self.layout.copies(reserved));
        let Some(pieces) = copies.and_then(|copies| self.pieces(&copies, refs.as_ref())) else {
            self.tree.abort(node);
            count(&self.shared.failed, 1);
            self.told(move || done(parent));
            return;
        };
        let (ticket, job) = self.job(sequence, Direction::ToHost, pieces, refs, None);
        self.pending.insert(
            ticket,
            Pending::Capture {
                node,
                parent,
                started,
                done,
            },
        );
        self.told(move || submit(job));
    }

    fn complete(&mut self, ticket: u64, ended: Ended) {
        let Some(pending) = self.pending.remove(&ticket) else {
            return;
        };
        match pending {
            Pending::Capture {
                node,
                parent,
                started,
                done,
            } => {
                let next = if ended == Ended::Done {
                    self.tree.publish(node);
                    count(&self.shared.captures, 1);
                    count(&self.shared.capture_nanos, nanos(started));
                    node
                } else {
                    self.tree.abort(node);
                    count(&self.shared.failed, 1);
                    parent
                };
                self.told(move || done(next));
            }
            Pending::Restore {
                matched,
                position,
                cancel,
                started,
                done,
            } => {
                self.tree.release(matched);
                // A failed job sets the request's flag itself, so the flag says nothing of why it ended.
                let result = match ended {
                    Ended::Failed(why) => Err(RestoreError::Copy(why)),
                    Ended::Cancelled => Err(RestoreError::Cancelled),
                    Ended::Done if cancel.load(Ordering::Acquire) => Err(RestoreError::Cancelled),
                    Ended::Done => {
                        count(&self.shared.restores, 1);
                        count(&self.shared.restore_nanos, nanos(started));
                        Ok(position)
                    }
                };
                self.told(move || done(result));
            }
        }
    }

    /// Tells whoever waits for the shut cache that no job is left in flight.
    fn signal_idle(&mut self) {
        if self.pending.is_empty()
            && let Some(idle) = self.idle_signal.take()
        {
            idle.send(());
        }
    }

    fn fail_all(&mut self, why: &str) {
        let tickets: Vec<u64> = self.pending.keys().copied().collect();
        for ticket in tickets {
            self.complete(ticket, Ended::Failed(why.to_string()));
        }
    }

    fn take_told(&mut self) -> Vec<Finished> {
        self.publish_stats();
        mem::take(&mut self.told)
    }
}

impl Turn for PrefixCache {
    type Message = Message;
    type Finished = Finished;

    fn turn(&mut self) {
        while let Some(message) = self.inbox.poll() {
            match message {
                Message::Done { ticket, ended } => self.complete(ticket, ended),
                Message::Unpin(matched) => self.tree.release(matched),
            }
        }
        self.signal_idle();
    }

    fn take_finished(&mut self) -> Vec<Finished> {
        self.take_told()
    }

    fn abandon(&mut self, why: &str) -> Vec<Finished> {
        self.fail_all(why);
        self.signal_idle();
        self.take_told()
    }

    fn deliver(finished: Finished) {
        let Finished::Told(then) = finished;
        then();
    }

    /// Stops new requests; `idle` is told once every copy job in flight has ended and been answered.
    fn shut(&mut self, idle: Arc<Handoff<()>>) {
        self.closed = true;
        self.idle_signal = Some(idle);
        self.signal_idle();
    }

    fn into_barrier(finished: Finished) -> Result<Arc<Handoff<()>>, Finished> {
        Err(finished)
    }

    fn idle(&self) -> bool {
        self.pending.is_empty()
    }
}

/// How callers reach a cache: requests answered by closures, and the numbers that need no owner.
#[derive(Clone)]
pub struct PrefixCacheHandle {
    mailbox: Arc<Mailbox<PrefixCache>>,
    shared: Arc<Shared>,
    root: NodeId,
}

impl Debug for PrefixCacheHandle {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        formatter.debug_struct("PrefixCacheHandle").finish_non_exhaustive()
    }
}

impl PrefixCacheHandle {
    /// Opens a cache over `parts`; its hand-offs to a worker go through `spawn`.
    pub(crate) fn open(parts: CacheParts, spawn: Spawn) -> Result<PrefixCacheHandle, OpenError> {
        PrefixCache::open(parts, spawn)
    }

    /// The tree's root: the parent of the first capture of a prompt.
    pub fn root(&self) -> NodeId {
        self.root
    }

    /// Whether a checkpoint at `end` tokens of a prompt of `len` is wanted: on the interval, or at the last chunk
    /// before the prompt ends; never past the limit.
    pub fn wants_checkpoint(&self, end: u32, len: u32) -> bool {
        let shared = &self.shared;
        end <= shared.limit.load(Ordering::Relaxed)
            && (end.is_multiple_of(shared.interval) || (end < len && len - end <= shared.chunk))
    }

    /// Whether a prompt of `len` tokens can match a checkpoint: the smallest is one chunk long, and a match leaves a
    /// token to prefill.
    pub fn can_match(&self, len: usize) -> bool {
        u32::try_from(len).map_or(true, |len| len > self.shared.chunk)
    }

    /// Stops checkpoints past `tokens`: the pages beyond it are not on the device (a sequence whose cache spills to the
    /// host has nothing a copy can read there).
    pub fn limit_checkpoints(&self, tokens: u32) {
        self.shared.limit.store(tokens, Ordering::Relaxed);
    }

    /// The counters and sizes, as of the last request the owner finished.
    pub fn stats(&self) -> PrefixCacheStats {
        let shared = &self.shared;
        let load = |counter: &AtomicU64| counter.load(Ordering::Relaxed);
        PrefixCacheStats {
            lookups: load(&shared.lookups),
            hits: load(&shared.hits),
            restored_tokens: load(&shared.restored_tokens),
            captures: load(&shared.captures),
            skipped: load(&shared.skipped),
            failed: load(&shared.failed),
            evictions: load(&shared.evictions),
            restores: load(&shared.restores),
            capture_nanos: load(&shared.capture_nanos),
            restore_nanos: load(&shared.restore_nanos),
            nodes: load(&shared.nodes),
            used_bytes: load(&shared.used_bytes),
            capacity_bytes: load(&shared.capacity_bytes),
        }
    }

    /// Asks the cache for the longest match of `prompt` that leaves a token to prefill, pinned in the [`Hit`] `done`
    /// receives; `None` for a miss or a closed cache.
    pub fn lookup(
        &self,
        prompt: Arc<[u32]>,
        kind: Option<Speculation>,
        done: impl FnOnce(Option<Hit>) + Send + 'static,
    ) {
        self.mailbox.defer(move |cache| match cache {
            Some(cache) => {
                let hit = cache.lookup(&prompt, kind);
                cache.told(move || done(hit));
                cache.take_told()
            }
            None => vec![Finished::Told(Box::new(move || done(None)))],
        });
    }

    /// Restores `hit` into `sequence`'s buffers (`refs`) through the job `submit` admits, and answers with the position
    /// restored. A cancel seen before or after the copies answers [`RestoreError::Cancelled`]; every answer unpins.
    ///
    /// **The protocol.** The caller is the sequence's owner, before the request's first chunk:
    ///
    /// 1. Its KV pages for `position` (`hit.position()`) must be reserved already: `refs.rows() >= position`, checked
    ///    here as the precondition of a capture is ([`RestoreError::Unreserved`]). With a speculation `kind` the pages
    ///    of the draft cache below the last row of `position` (and that row's too, when the prompt has the token the checkpoint
    ///    paired it with, see [`Hit::paired`]) must be reserved too, and the draft seed row is written.
    /// 2. `submit` arrives once, from a worker, when the job is built: the owner admits it behind the sequence's last
    ///    device work with `on_workspace` false and **`more` set** (the first chunk follows it), and counts it in
    ///    flight until it ends. A refused restore never calls `submit`; it answers.
    /// 3. The owner holds the request's first chunk until `submit` has admitted the job: a chunk admitted before it
    ///    would not wait for it and would read the state the job is still writing. The chunk is then admitted at
    ///    `position` with the usual `more`.
    /// 4. `done` arrives after the job ended (or was refused or dropped): `Ok(position)` when the copies ran, an error
    ///    otherwise. On any failure of the copy the job has already set `cancel`, the request's kill flag, before
    ///    whatever was admitted behind it can start, so the chunk behind a failed restore does not run and cannot emit a
    ///    token before the failure reaches the owner, which then ends the request with the error.
    ///
    /// The match stays pinned until the job has ended, not until it is built: the arena slots it reads cannot be
    /// evicted or reused by another request meanwhile.
    #[allow(clippy::too_many_arguments, reason = "one request's parts")]
    pub fn restore(
        &self,
        hit: Hit,
        sequence: SequenceId,
        refs: Arc<dyn Resolve>,
        kind: Option<Speculation>,
        cancel: Arc<AtomicBool>,
        submit: Submit,
        done: impl FnOnce(Result<u32, RestoreError>) + Send + 'static,
    ) {
        let done: Box<dyn FnOnce(Result<u32, RestoreError>) + Send> = Box::new(done);
        // A closed owner never runs the closure: the hit then drops with it and unpins into nothing.
        self.mailbox.defer(move |cache| match cache {
            Some(cache) => {
                cache.begin_restore(hit, sequence, refs, kind, cancel, submit, done);
                cache.take_told()
            }
            None => vec![Finished::Told(Box::new(move || done(Err(RestoreError::Closed))))],
        });
    }

    /// Captures `sequence`'s state at `end` tokens as a child of `parent` and answers with the node to capture under
    /// next: the new node, or the one that already held the span, or `parent` when nothing was stored (no room, the
    /// sequence is not ready, or the copy failed). `tokens` are the prompt from its start.
    ///
    /// The job `submit` admits must be admitted with the chunk's `more` set when it follows a prefill chunk (see the
    /// module documentation).
    #[allow(clippy::too_many_arguments, reason = "one request's parts")]
    pub fn capture(
        &self,
        sequence: SequenceId,
        refs: Arc<dyn Resolve>,
        tokens: Arc<[u32]>,
        parent: NodeId,
        end: u32,
        kind: Option<Speculation>,
        submit: Submit,
        done: impl FnOnce(NodeId) + Send + 'static,
    ) {
        let done: Box<dyn FnOnce(NodeId) + Send> = Box::new(done);
        self.mailbox.defer(move |cache| match cache {
            Some(cache) => {
                cache.begin_capture(sequence, refs, &tokens, parent, end, kind, submit, done);
                cache.take_told()
            }
            None => vec![Finished::Told(Box::new(move || done(parent)))],
        });
    }

    /// Shuts the cache: it stores and restores nothing from now on, and this waits (at most `bound`, from outside the
    /// lattice) for the copy jobs in flight to end and be answered, then frees the cache. After `bound` the jobs left
    /// are abandoned (failed and unpinned) and the arena stays allocated until the last of them is dropped. `false`
    /// when it could not wait (called from inside a turn) or the bound passed.
    pub fn close(&self, bound: Duration) -> bool {
        let Some(idle) = self.mailbox.stopping() else {
            log::error!("the prefix cache cannot be waited for from inside a turn at shutdown");
            return false;
        };
        let ended = idle.wait(bound).is_ok();
        if !ended {
            log::error!(
                "the prefix cache's copy jobs did not end within {bound:?} of the shutdown: they are abandoned"
            );
            self.mailbox.defer(|cache| match cache {
                Some(cache) => cache.abandon("the engine is shutting down"),
                None => Vec::new(),
            });
        }
        drop(self.mailbox.close());
        ended
    }
}
