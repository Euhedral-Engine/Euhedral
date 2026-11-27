//! The cache's owner as a source of the lattice.
//!
//! Whatever changes the owner's bookkeeping is a [`Message`] posted to its [`Inbox`], from any thread: a lease dropped,
//! a read ended, a copy submitted or retired, a request. Posting offers the message to a lock-free queue and sets a
//! flag, and returns: nothing runs on the posting thread, so a lease dropped while a panic unwinds, or a driver
//! callback, only enqueues.
//!
//! The records are applied by a worker. [`OwnerSource`] is attached to the lattice (`Lattice::add_upstream`) like the
//! device completions are; a worker's `request` of it pushes one *turn frame* while the flag is set and no turn frame is
//! out, and a worker runs the frame like any other. The frame applies the records in the order they were posted, with
//! the claim of the owner's state, then lets the claim go and tells the listeners of the outcomes the records led to.
//! `request` and `pull` themselves apply nothing and tell nobody: applying records and running a listener is a
//! frame's work (a worker can only measure it when the frame comes back through its executor).
//!
//! **Why no record is stranded.** A poster offers, then sets the flag; the turn clears the flag, then drains the queue.
//! A record whose offer finished before the clear is drained by the turn; one whose offer finishes after it sets the
//! flag after it, so the next `request` finds the flag set and pushes the next frame. Workers visit a source on every
//! cycle and park only for microseconds, so a record posted by a thread outside the lattice is applied at the next
//! visit, as a registration on the completions is.
//!
//! **A panic does not escape.** A record's application is caught by the owner (which poisons itself, see
//! [`Owner::turn`]); a listener's panic is caught and logged here, and the listeners after it are still told. The turn
//! frame's guard clears its flag when the frame is dropped, run or not, so a killed frame cannot stop the owner.

use std::fmt::{self, Formatter};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Weak};

use euhedral_core::frames::Frame;
use euhedral_core::generics::{DownstreamSlot, LatticeReceiver, LatticeSource, SourceCell};
use euhedral_data_structures::queues::{Multi, Producer};

use super::owner::{Message, Owner};
use crate::frames::{Kind, step};

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

    fn clear(&self) {
        self.pending.store(false, Ordering::SeqCst);
    }

    fn waits(&self) -> bool {
        self.pending.load(Ordering::SeqCst)
    }
}

/// The owner of the expert cache's bookkeeping, as a [`LatticeSource`] of one kind of frame: its turn.
pub(crate) struct OwnerSource {
    inbox: Arc<Inbox>,
    /// The owner: only a turn frame enters it.
    state: SourceCell<Owner>,
    downstream: DownstreamSlot,
    /// A turn frame is out (pushed and not yet run or dropped).
    frame_out: AtomicBool,
    me: Weak<OwnerSource>,
}

impl fmt::Debug for OwnerSource {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("OwnerSource").finish_non_exhaustive()
    }
}

/// Clears `frame_out` when the turn frame is dropped, whether it ran or was discarded.
struct Out(Arc<OwnerSource>);

impl Drop for Out {
    fn drop(&mut self) {
        // A record posted while the frame ran left `pending` set: the next `request` pushes the next frame.
        self.0.frame_out.store(false, Ordering::SeqCst);
    }
}

impl OwnerSource {
    pub(super) fn new(inbox: Arc<Inbox>, owner: Owner) -> Arc<OwnerSource> {
        Arc::new_cyclic(|me| OwnerSource {
            inbox,
            state: SourceCell::new(owner),
            downstream: DownstreamSlot::new(),
            frame_out: AtomicBool::new(false),
            me: me.clone(),
        })
    }

    /// Posts `message` for the owner.
    pub(crate) fn post(&self, message: Message) {
        self.inbox.post(message);
    }

    /// The turn: applies the records posted, then tells the outcomes they led to. Runs in the turn frame.
    fn run_turn(&self) {
        let Some(mut owner) = self.state.try_enter() else {
            // Another turn holds the state (a caller driving by hand): it was posted for, and looks again.
            self.inbox.pending.store(true, Ordering::SeqCst);
            return;
        };
        self.inbox.clear();
        let finished = owner.turn();
        // The claim is free before anyone is told: a listener may post, and the next turn may enter.
        owner.release_seq_cst();
        for finished in finished {
            if catch_unwind(AssertUnwindSafe(|| finished.deliver())).is_err() {
                log::error!("a listener of the expert cache panicked: the outcomes after it were still told");
            }
        }
    }

    /// Pushes the turn frame to the downstream when a record waits and no turn frame is out.
    fn push_turn(&self) {
        if !self.inbox.waits() {
            return;
        }
        let Some(receiver) = self.downstream.receiver() else {
            return;
        };
        let Some(me) = self.me.upgrade() else {
            return;
        };
        if self.downstream.is_complete() || self.frame_out.swap(true, Ordering::SeqCst) {
            return;
        }
        let out = Out(me);
        let turn: Box<dyn Frame> = step(Kind::ExpertOwner, 0, move || {
            out.0.run_turn();
            drop(out);
        });
        receiver.push(turn);
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
