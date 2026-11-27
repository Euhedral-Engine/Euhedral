//! The cache's owner as a source of the lattice.
//!
//! Whatever changes the owner's bookkeeping is a [`Message`] posted to its [`Inbox`], from any thread: a lease dropped,
//! a read ended, a copy submitted or retired, a request. Posting offers the message to a lock-free queue and sets a
//! flag, and returns: nothing runs on the posting thread, so a lease dropped while a panic unwinds, or a driver
//! callback, only enqueues.
//!
//! The records are applied by a worker. [`OwnerSource`] is attached to the lattice (`Lattice::add_upstream`) like the
//! device completions are; the source owns one reusable *turn frame*, a [`KeySlot`] key. A worker's `request` of it takes
//! the key from its home and pushes it while the flag is set, and a worker runs the frame like any other; the frame's
//! `recycle` sends it home. While it is out the source has no second one, so no two turns run at once and nothing is
//! allocated per turn. The frame applies the records in the order they were posted, with
//! the claim of the owner's state, then lets the claim go and tells the listeners of the outcomes the records led to.
//! `request` and `pull` themselves apply nothing and tell nobody: applying records and running a listener is a
//! frame's work (a worker can only measure it when the frame comes back through its executor). Once a close has ended
//! with nothing in flight, the source completes: the lattice lets it go, and a fetch finds it complete and is refused at
//! once, on the calling thread.
//!
//! **Why no record is stranded.** A poster offers, then sets the flag; the turn clears the flag (one swap), then drains the queue.
//! A record whose offer finished before the clear is drained by the turn; one whose offer finishes after it sets the
//! flag after it, so the next `request` finds the flag set and pushes the next frame. Workers visit a source on every
//! cycle and park only for microseconds, so a record posted by a thread outside the lattice is applied at the next
//! visit, as a registration on the completions is.
//!
//! **A panic does not escape.** A record's application is caught by the owner (which poisons itself, see
//! [`Owner::turn`]); a listener's panic is caught and logged here, and the listeners after it are still told. The turn
//! frame is replaced by a new one when it is dropped without having run, so a discarded frame cannot stop the owner.

use std::fmt::{self, Formatter};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Weak};

use euhedral_core::frames::{Frame, FrameError, FrameHeader};
use euhedral_core::generics::{DownstreamSlot, KeySlot, LatticeReceiver, LatticeSource, SourceCell};
use euhedral_data_structures::queues::{Multi, Producer};

use super::owner::{Listener, Message, Owner, Reply};
use crate::frames::{Kind, id_hash};

/// Where messages for the owner are posted: the producing end of its record queue and the flag that says one waits.
pub(crate) struct Inbox {
    records: Producer<Message, Multi>,
    /// Set after every offer, cleared by a turn before it drains the queue.
    pending: AtomicBool,
}

impl Inbox {
    pub(super) fn new(records: Producer<Message, Multi>) -> Inbox {
        Inbox {
            records,
            pending: AtomicBool::new(false),
        }
    }

    /// Posts `message` for the owner and returns: it is applied by the next turn.
    pub(crate) fn post(&self, message: Message) {
        if self.records.offer(message).is_err() {
            log::error!("the expert cache owner's inbox refused a record");
            return;
        }
        // The offer is published before the flag is set: see the module documentation.
        self.pending.store(true, Ordering::SeqCst);
    }

    fn waits(&self) -> bool {
        self.pending.load(Ordering::SeqCst)
    }
}

/// The key frame of the owner's source: its `execute` is the owner's turn.
struct TurnFrame {
    header: FrameHeader,
    source: Weak<OwnerSource>,
    /// Counts the pushes, so that each is routed by a hash of its own: the turn may run on any worker.
    seed: u64,
}

impl TurnFrame {
    fn new(source: Weak<OwnerSource>) -> Box<TurnFrame> {
        Box::new(TurnFrame {
            header: FrameHeader::new(id_hash(Kind::ExpertOwner, 0)),
            source,
            seed: 0,
        })
    }

    fn arm(&mut self) {
        self.seed = self.seed.wrapping_add(1);
        self.header.randomize_hash(self.seed);
    }
}

impl Frame for TurnFrame {
    fn header(&self) -> &FrameHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut FrameHeader {
        &mut self.header
    }

    fn execute(&mut self) -> Result<(), FrameError> {
        if let Some(source) = self.source.upgrade() {
            source.run_turn();
        }
        Ok(())
    }

    /// The key goes home: the next `request` that finds a record waiting pushes it again. (A source that is gone has no
    /// home: the frame is dropped.)
    fn recycle(self: Box<Self>) {
        if let Some(source) = self.source.upgrade() {
            source.key.home(self);
        }
    }
}

impl Drop for TurnFrame {
    /// A key dropped without having run (a lattice that discarded it) must not strand the owner: a new one goes home. A
    /// key that is homed does not get here, and there is no other: the slot is empty.
    fn drop(&mut self) {
        if let Some(source) = self.source.upgrade() {
            source
                .key
                .home_replacement(TurnFrame::new(self.source.clone()), |frame| {
                    frame.source = Weak::new();
                });
        }
    }
}

/// The owner of the expert cache's bookkeeping, as a [`LatticeSource`] of one kind of frame: its turn.
pub(crate) struct OwnerSource {
    inbox: Arc<Inbox>,
    /// The owner: only a turn frame enters it.
    state: SourceCell<Owner>,
    downstream: DownstreamSlot,
    /// The home of the turn frame: empty while a turn frame is out (pushed and not yet recycled or dropped).
    key: KeySlot<TurnFrame>,
}

impl fmt::Debug for OwnerSource {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("OwnerSource").finish_non_exhaustive()
    }
}

impl OwnerSource {
    pub(super) fn new(inbox: Arc<Inbox>, owner: Owner) -> Arc<OwnerSource> {
        Arc::new_cyclic(|me| OwnerSource {
            inbox,
            state: SourceCell::new(owner),
            downstream: DownstreamSlot::new(),
            key: KeySlot::with_key(TurnFrame::new(me.clone())),
        })
    }

    /// Posts `message` for the owner.
    pub(crate) fn post(&self, message: Message) {
        self.inbox.post(message);
    }

    /// A reply for `listener` that posts here if it is dropped untold.
    pub(super) fn reply(&self, listener: Listener) -> Reply {
        Reply::new(listener, Arc::downgrade(&self.inbox))
    }

    /// The turn: applies the records posted, then tells the outcomes they led to. Runs in the turn frame.
    fn run_turn(&self) {
        let Some(mut owner) = self.state.try_enter() else {
            // Another turn holds the state (a caller driving by hand): it was posted for, and looks again.
            self.inbox.pending.store(true, Ordering::SeqCst);
            return;
        };
        // Cleared before the drain, so that a record offered after the drain's end sets it again.
        self.inbox.pending.swap(false, Ordering::SeqCst);
        let finished = owner.turn();
        let shut = owner.is_shut();
        // The claim is free before anyone is told: a listener may post, and the next turn may enter.
        owner.release_seq_cst();
        for finished in finished {
            if catch_unwind(AssertUnwindSafe(|| finished.deliver())).is_err() {
                log::error!("a listener of the expert cache panicked: the outcomes after it were still told");
            }
        }
        if shut {
            // Closed with nothing in flight: the source ends, and the lattice lets it go. A fetch finds it complete.
            self.downstream.complete();
        }
    }

    /// Pushes the turn frame to the downstream when a record waits and the frame is home.
    fn push_turn(&self) {
        if !self.inbox.waits() || self.downstream.is_complete() {
            return;
        }
        let Some(receiver) = self.downstream.receiver() else {
            return;
        };
        let Some(mut key) = self.key.take() else {
            return;
        };
        key.arm();
        receiver.push(key);
    }
}

impl LatticeSource for OwnerSource {
    fn add_downstream(self: Arc<Self>, downstream: Arc<dyn LatticeReceiver>) {
        self.downstream
            .attach(Arc::clone(&self) as Arc<dyn LatticeSource>, downstream);
    }

    /// Hands out nothing: the turn leaves as a frame, pushed by [`request`](Self::request).
    fn pull(
        &self,
        _consumer: &mut dyn FnMut(Box<dyn Frame>),
        _stop: &mut dyn FnMut(&dyn Frame) -> bool,
        _demand: u64,
    ) -> u64 {
        0
    }

    /// Pushes the turn frame when a record waits. The frame is a control frame, not part of the demand.
    fn request(&self, demand: u64) {
        if demand == 0 {
            return;
        }
        self.push_turn();
    }

    fn complete(&self) {
        self.downstream.complete();
    }

    fn is_complete(&self) -> bool {
        self.downstream.is_complete()
    }
}
