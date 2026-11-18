//! One request's dealings with the prefix cache: the lookup before its first chunk, the restore of a hit, and the
//! captures after the chunks that end at a wanted checkpoint.
//!
//! The type holds no device and no owner: it is the state the sequence's owner keeps for the request, and the answer to
//! one question, "may the next chunk be admitted now?" ([`PromptCheckpoints::holds`]). The owner makes what is the
//! sequence's to make (the reservation of the pages a restore writes, the references a capture reads) and hands the
//! cache's requests through here, so that every request is counted until it is answered: a request's sequence is not
//! given back while one is unanswered ([`PromptCheckpoints::pending`]).
//!
//! **Order.** What keeps the state a capture reads from being overwritten is the order of the jobs on the sequence:
//! the chunk that ends at the checkpoint is admitted with no chunk behind it (`more` false), so it retires before
//! anything is asked; the capture is asked only once that chunk completed (a chunk that failed or was stopped never
//! has a capture, so nothing it half-wrote is published); and the owner admits the capture's job with `more` set when
//! the request goes on, so the next chunk starts behind the job, on the sequence's edge, and cannot touch the
//! state before the job has read it. The next chunk is not admitted until the job is, which is what [`holds`] says.
//! A restore is the same in the other direction: the first chunk is held until the restore's job is admitted, and
//! starts behind it.
//!
//! [`holds`]: PromptCheckpoints::holds

use std::fmt::{Debug, Formatter, Result as FmtResult};
use std::mem;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use super::{Hit, NodeId, PrefixCacheHandle, Resolve, RestoreError, Speculation, Submit};
use crate::state::SequenceId;

/// Where the request is in the lookup and the restore.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Start {
    /// The lookup is not answered.
    Looking,
    /// The restore is asked: `submitted` once the cache's job is handed to the owner.
    Restoring { submitted: bool },
    /// Nothing is left to wait for at the start.
    Settled,
}

/// The chunk that ends at a checkpoint, until the capture of its state is asked.
enum Next {
    None,
    /// The chunk is admitted and has not concluded.
    Expected {
        end: u32,
        refs: Arc<dyn Resolve>,
    },
    /// The chunk completed; the capture is asked once the one before it is answered (its node is the parent).
    Due {
        end: u32,
        refs: Arc<dyn Resolve>,
    },
}

/// How a restore ended, for the owner.
#[derive(Debug, PartialEq, Eq)]
pub enum Restored {
    /// The checkpoint's state is in the sequence: the prompt resumes from this position.
    Resumed(u32),
    /// The request was cancelled.
    Cancelled,
    /// Nothing was copied (no buffer, a format that differs, ...): the prompt prefills from its start.
    Refused(RestoreError),
    /// A copy ran and failed: the sequence holds a half-written state.
    Failed(RestoreError),
}

/// A request's prefix-cache state. See the module documentation.
pub struct PromptCheckpoints {
    cache: PrefixCacheHandle,
    tokens: Arc<[u32]>,
    /// The speculative state the request's checkpoints hold: its lookups match only nodes that hold it, its captures
    /// store it, its restores write it. `None` for a request that does not draft.
    kind: Option<Speculation>,
    /// The node the next capture hangs under.
    parent: NodeId,
    start: Start,
    next: Next,
    /// A capture is asked and unanswered: `submitted` once its job is handed to the owner.
    open: Option<bool>,
    /// Requests asked of the cache and not answered.
    asks: usize,
    /// The kill flags of the jobs asked for: set when the request stops.
    kills: Vec<Arc<AtomicBool>>,
    cached: u32,
}

impl Debug for PromptCheckpoints {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.debug_struct("PromptCheckpoints")
            .field("start", &self.start)
            .field("open", &self.open)
            .field("asks", &self.asks)
            .field("cached", &self.cached)
            .finish_non_exhaustive()
    }
}

impl PromptCheckpoints {
    /// Asks the cache for the longest checkpoint of `tokens`; `looked` receives the answer (and the owner then calls
    /// [`looked`](Self::looked)). A prompt that cannot match one (at most a chunk long) is not looked up: `looked` is
    /// never called, and the request starts at once ([`holds`](Self::holds) is false). A request that drafts passes the
    /// `kind` of the state its drafts read: only checkpoints that hold it match, and its own hold it.
    pub fn begin(
        cache: PrefixCacheHandle,
        tokens: Arc<[u32]>,
        kind: Option<Speculation>,
        looked: impl FnOnce(Option<Hit>) + Send + 'static,
    ) -> PromptCheckpoints {
        let parent = cache.root();
        let asking = Arc::clone(&tokens);
        let lookup = cache.can_match(tokens.len());
        let checkpoints = PromptCheckpoints {
            cache: cache.clone(),
            tokens,
            kind,
            parent,
            start: if lookup { Start::Looking } else { Start::Settled },
            next: Next::None,
            open: None,
            asks: usize::from(lookup),
            kills: Vec::new(),
            cached: 0,
        };
        if lookup {
            cache.lookup(asking, kind, looked);
        }
        checkpoints
    }

    /// The lookup was answered.
    pub fn looked(&mut self) {
        self.asks = self.asks.saturating_sub(1);
        self.start = Start::Settled;
    }

    /// Asks the cache to restore `hit` into the sequence's buffers (`refs`, whose pages up to the hit's position are
    /// reserved). `submit` makes the [`Submit`] that hands the job to the owner, given the flag that kills it; the flag
    /// is returned, and is the kill flag of the first chunk too, so that a failed restore stops the chunk behind it.
    /// `done` receives the answer (and the owner then calls [`restored`](Self::restored)).
    pub fn restore(
        &mut self,
        hit: Hit,
        sequence: SequenceId,
        refs: Arc<dyn Resolve>,
        submit: impl FnOnce(Arc<AtomicBool>) -> Submit,
        done: impl FnOnce(Result<u32, RestoreError>) + Send + 'static,
    ) -> Arc<AtomicBool> {
        let kill = Arc::new(AtomicBool::new(false));
        self.kills.push(Arc::clone(&kill));
        self.parent = hit.leaf();
        self.start = Start::Restoring { submitted: false };
        self.asks += 1;
        self.cache.restore(
            hit,
            sequence,
            refs,
            self.kind,
            Arc::clone(&kill),
            submit(Arc::clone(&kill)),
            done,
        );
        kill
    }

    /// The restore's job was handed to the owner and admitted.
    pub fn restore_submitted(&mut self) {
        if let Start::Restoring { submitted } = &mut self.start {
            *submitted = true;
        }
    }

    /// The restore was answered.
    pub fn restored(&mut self, result: Result<u32, RestoreError>) -> Restored {
        self.asks = self.asks.saturating_sub(1);
        let submitted = matches!(self.start, Start::Restoring { submitted: true });
        self.start = Start::Settled;
        match result {
            Ok(position) => {
                self.cached = position;
                Restored::Resumed(position)
            }
            Err(RestoreError::Cancelled) => Restored::Cancelled,
            Err(error) if !submitted => {
                // Nothing was copied: the prompt starts again from the root.
                self.parent = self.cache.root();
                Restored::Refused(error)
            }
            Err(error) => Restored::Failed(error),
        }
    }

    /// Whether a checkpoint at `end` tokens of the prompt is wanted. One at the prompt's end is, when the end is on the
    /// interval: a later prompt that extends this one (the next turn of a chat) restores it. The identical prompt never
    /// does: a match always leaves a token to prefill.
    pub fn wants(&self, end: usize) -> bool {
        u32::try_from(end)
            .ok()
            .zip(u32::try_from(self.tokens.len()).ok())
            .is_some_and(|(end, len)| end <= len && self.cache.wants_checkpoint(end, len))
    }

    /// The chunk that ends at `end` is admitted; `refs` are the references it was admitted with.
    pub fn expect(&mut self, end: usize, refs: Arc<dyn Resolve>) {
        if let Ok(end) = u32::try_from(end) {
            self.next = Next::Expected { end, refs };
        }
    }

    /// The chunk completed and the request goes on: its state is what a capture reads.
    pub fn concluded(&mut self) {
        if let Next::Expected { end, refs } = mem::replace(&mut self.next, Next::None) {
            self.next = Next::Due { end, refs };
        }
    }

    /// Asks for the capture a completed chunk is due, if the one before it is answered. `submit` makes the [`Submit`]
    /// that hands the job to the owner, given the flag that kills it; `done` receives the answer (and the owner then
    /// calls [`captured`](Self::captured)). Returns whether it asked.
    pub fn ask(
        &mut self,
        sequence: SequenceId,
        submit: impl FnOnce(Arc<AtomicBool>) -> Submit,
        done: impl FnOnce(NodeId) + Send + 'static,
    ) -> bool {
        if self.open.is_some() || !matches!(self.next, Next::Due { .. }) {
            return false;
        }
        let Next::Due { end, refs } = mem::replace(&mut self.next, Next::None) else {
            return false;
        };
        let kill = Arc::new(AtomicBool::new(false));
        self.kills.push(Arc::clone(&kill));
        self.open = Some(false);
        self.asks += 1;
        self.cache.capture(
            sequence,
            refs,
            Arc::clone(&self.tokens),
            self.parent,
            end,
            self.kind,
            submit(kill),
            done,
        );
        true
    }

    /// The capture's job was handed to the owner and admitted.
    pub fn capture_submitted(&mut self) {
        if let Some(submitted) = &mut self.open {
            *submitted = true;
        }
    }

    /// The capture was answered: the node the next one hangs under (the one captured, the one that held the span, or
    /// the parent when nothing was stored).
    pub fn captured(&mut self, node: NodeId) {
        self.asks = self.asks.saturating_sub(1);
        self.open = None;
        self.parent = node;
    }

    /// Whether the next chunk must wait: the lookup is unanswered, a restore's job is not yet admitted, or the state
    /// of a chunk that ends at a checkpoint is not yet read (the chunk is in flight, its capture is not yet asked, or
    /// the capture's job is not yet admitted).
    pub fn holds(&self) -> bool {
        self.start == Start::Looking
            || self.start == (Start::Restoring { submitted: false })
            || !matches!(self.next, Next::None)
            || self.open == Some(false)
    }

    /// Whether a capture is due and asked for nothing yet.
    pub fn due(&self) -> bool {
        matches!(self.next, Next::Due { .. })
    }

    /// Requests asked of the cache and not answered: the sequence is not given back while there are any.
    pub fn pending(&self) -> usize {
        self.asks
    }

    /// The restored position: 0 on a miss.
    pub fn cached(&self) -> u32 {
        self.cached
    }

    /// The request stops: the jobs asked for are told to stop, and no capture is asked from now on (the references of a
    /// chunk not captured are let go).
    pub fn stop(&mut self) {
        for kill in &self.kills {
            kill.store(true, Ordering::Release);
        }
        self.next = Next::None;
    }
}
