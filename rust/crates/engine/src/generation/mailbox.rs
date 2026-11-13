//! The mailbox of a sequence's owner: the owner is driven by the messages posted to it, on whichever thread posts.
//!
//! The owner's state is a [`ClaimCell`]: one thread at a time may hold the claim and run a turn of the owner. A frame
//! that has something to tell the owner (a quantum retired, a logits row arrived, a parked quantum's lane work ran)
//! [`post`](Mailbox::post)s a message and then [`drive`](Mailbox::drive)s: it takes the claim if it is free and runs
//! turns while messages keep arriving; if the claim is held it goes on, and the holder sees the message. Nothing polls
//! the owner, and nothing waits for it.
//!
//! **Why no message is stranded.** The mailbox keeps a `pending` flag besides the inbox: the inbox's emptiness counts
//! slots a poster has reserved and not yet published, so looking at it would send the holder through empty turns until
//! the poster is scheduled again. A poster writes first and tries second: it offers the message, stores `pending`
//! (`SeqCst`), fences (`SeqCst`), then compares-and-sets the claim. A holder swaps `pending` to false before its turn
//! (so the turn takes in every message offered before the swap), gives the claim back with
//! [`ClaimGuard::release_seq_cst`] (a `SeqCst` store, then a `SeqCst` fence) and only then loads `pending`. Both sides
//! are a write followed by a read of the other's location with a `SeqCst` fence between them, so at least one sees the
//! other: either the poster's compare-and-set finds the claim free and the poster holds it, or the holder's load finds
//! `pending` and the holder runs another turn. A message offered after the swap sets `pending` after it, so the load
//! finds it unless another thread already drove it. The claim goes back before the load, never after: loading while
//! holding it would leave a window in which a poster fails the compare-and-set after the load. A message whose
//! `pending` was swapped away by a turn that began before the message's offer completed is the one false positive: it
//! costs one extra turn, never a lost one.
//!
//! **Bounded work.** A frame runs at most [`TURNS_PER_DRIVE`] turns; if messages still wait it hands the rest to a
//! worker through the mailbox's [`Spawn`] (a frame offered to the `stages` lake), so no frame carries unbounded work and
//! a producer that keeps posting cannot hold one worker. The same hand-off serves a thread that is not a worker: it
//! claims with [`with`](Mailbox::with), and the turn it provokes runs on a worker.
//!
//! Every thread that claims the owner and gives it back (`with`, `with_idle`, `close`) loads `pending` after
//! `release_seq_cst` and hands a waiting message to a worker, also when its work or a listener it told panicked: the
//! panic is caught, the hand-off made, and the panic resumed.

use std::fmt::{self, Formatter};
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::atomic::{AtomicBool, Ordering, fence};
use std::sync::{Arc, OnceLock, Weak};

use euhedral_data_structures::queues::{Multi, Producer};
use euhedral_data_structures::sync::ClaimCell;

/// Turns one drive runs before it hands the rest to a worker. A turn admits quanta and concludes retired ones: host
/// work of tens of microseconds. Four keep a burst of retirements on one worker without letting a producer that never
/// stops posting pin it.
pub(crate) const TURNS_PER_DRIVE: usize = 4;

/// Runs a closure on a worker, later, unordered; `false` when no worker will (the engine is shutting down).
pub(crate) type Spawn = Box<dyn Fn(Box<dyn FnOnce() + Send>) -> bool + Send + Sync>;

/// What a mailbox drives: the state of an owner, and what it does in a turn.
pub(crate) trait Turn: Send + 'static {
    /// What is posted to the owner.
    type Message: Send;
    /// What a turn that ended the owner's work leaves to be told once the claim is released.
    type Finished;

    /// Takes in what arrived and does what follows from it.
    fn turn(&mut self);

    /// The outcome the last turn left, if it ended the owner's work.
    fn take_finished(&mut self) -> Option<Self::Finished>;

    /// Tells `finished`'s audience. The claim is free: a listener may start the next request from here.
    fn deliver(finished: Self::Finished);

    /// Whether no request is in flight and nothing the owner queued on the device is still held: only then may a
    /// thread outside the frames look at the owner's state or end it.
    fn idle(&self) -> bool;
}

/// The inbox of an owner and the claim on its state.
pub(crate) struct Mailbox<O: Turn> {
    /// The owner, until [`close`](Self::close) takes it.
    owner: ClaimCell<Option<O>>,
    /// The producing end of the owner's inbox; the consuming end lives inside the owner.
    inbox: Producer<O::Message, Multi>,
    /// Set by every post after its offer, cleared by the holder before its turn.
    pending: AtomicBool,
    spawn: Spawn,
    /// Test hook: told whether the claim is held at the moment `pending` is loaded after a release.
    #[cfg(test)]
    looked: OnceLock<Box<dyn Fn(bool) + Send + Sync>>,
}

impl<O: Turn> fmt::Debug for Mailbox<O> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("Mailbox").finish_non_exhaustive()
    }
}

impl<O: Turn> Mailbox<O> {
    /// A mailbox over `owner`, whose inbox's producing end is `inbox`; `spawn` runs a frame on a worker.
    pub(crate) fn new(owner: O, inbox: Producer<O::Message, Multi>, spawn: Spawn) -> Mailbox<O> {
        Mailbox {
            owner: ClaimCell::new(Some(owner)),
            inbox,
            pending: AtomicBool::new(false),
            spawn,
            #[cfg(test)]
            looked: OnceLock::new(),
        }
    }

    /// Tells the owner `message` and runs its turns if no one else is running them. Never waits.
    pub(crate) fn post(self: &Arc<Self>, message: O::Message) {
        if self.inbox.offer(message).is_err() {
            log::error!("the owner's inbox refused a message");
            return;
        }
        // The message is written before the claim is tried: see the module documentation.
        self.pending.store(true, Ordering::SeqCst);
        fence(Ordering::SeqCst);
        self.drive();
    }

    /// Hands a frame that drives the mailbox to a worker. `false` when no worker will take it.
    pub(crate) fn nudge(self: &Arc<Self>) -> bool {
        let mailbox = Arc::clone(self);
        (self.spawn)(Box::new(move || mailbox.drive()))
    }

    /// Runs up to [`TURNS_PER_DRIVE`] turns of the owner while the claim is free and messages wait; hands the rest to
    /// a worker. A caller that finds the claim held returns: the holder looks at `pending` after it releases the claim.
    pub(crate) fn drive(self: &Arc<Self>) {
        for _ in 0..TURNS_PER_DRIVE {
            let Some(mut claim) = self.owner.try_claim() else {
                return;
            };
            let Some(owner) = claim.as_mut() else {
                // Closed: nothing will ever take a turn again.
                return;
            };
            self.pending.swap(false, Ordering::SeqCst);
            let turned = catch_unwind(AssertUnwindSafe(|| {
                owner.turn();
                owner.take_finished()
            }));
            // The claim goes back before `pending` is loaded, and before anyone is told: a listener that starts the
            // next request from `finished` finds the owner free.
            claim.release_seq_cst();
            let finished = match turned {
                Ok(finished) => finished,
                Err(payload) => {
                    self.hand_on();
                    resume_unwind(payload);
                }
            };
            if let Some(Err(payload)) = finished.map(|finished| catch_unwind(AssertUnwindSafe(|| O::deliver(finished))))
            {
                self.hand_on();
                resume_unwind(payload);
            }
            if !self.waiting() {
                return;
            }
        }
        // The turns of this frame are used and a message still waits.
        self.hand_on();
    }

    /// Whether a message waits, loaded after a release (see the module documentation).
    fn waiting(&self) -> bool {
        #[cfg(test)]
        if let Some(looked) = self.looked.get() {
            looked(self.owner.is_claimed());
        }
        self.pending.load(Ordering::SeqCst)
    }

    /// Gives a waiting message to a worker, for a thread that must not run the turn itself.
    fn hand_on(self: &Arc<Self>) {
        if self.waiting() && !self.nudge() {
            log::error!("a message waits for the owner and no worker will take it");
        }
    }

    /// Claims the owner, runs `work` and gives it back; a message that arrived meanwhile goes to a worker.
    fn claimed<R>(self: &Arc<Self>, idle_only: bool, work: impl FnOnce(&mut O) -> R) -> Option<R> {
        let mut claim = self.owner.try_claim()?;
        let outcome = match claim.as_mut() {
            None => Ok(None),
            Some(owner) if idle_only && !owner.idle() => Ok(None),
            Some(owner) => catch_unwind(AssertUnwindSafe(|| work(owner))).map(Some),
        };
        claim.release_seq_cst();
        self.hand_on();
        match outcome {
            Ok(out) => out,
            Err(payload) => resume_unwind(payload),
        }
    }

    /// Runs `work` on the owner if the claim is free, for a thread that is not a frame (it starts a request). `None`
    /// when another holds the claim, or the owner is closed. The caller that began a request offers the first turn
    /// itself ([`nudge`](Self::nudge)); a message that arrived while `work` held the claim goes to a worker here.
    pub(crate) fn with<R>(self: &Arc<Self>, work: impl FnOnce(&mut O) -> R) -> Option<R> {
        self.claimed(false, work)
    }

    /// As [`with`](Self::with), and only while the owner is idle.
    pub(crate) fn with_idle<R>(self: &Arc<Self>, work: impl FnOnce(&mut O) -> R) -> Option<R> {
        self.claimed(true, work)
    }

    /// Takes the owner out, for its teardown, while it is idle. `None` when another holds the claim, a request is in
    /// flight or it is taken already; the mailbox is closed from here on.
    pub(crate) fn close(self: &Arc<Self>) -> Option<O> {
        let mut claim = self.owner.try_claim()?;
        let owner = claim.take_if(|owner| owner.idle());
        claim.release_seq_cst();
        self.hand_on();
        owner
    }

    /// Test hook: `look` is told whether the claim is held each time `pending` is loaded after a release.
    #[cfg(test)]
    fn on_look(&self, look: impl Fn(bool) + Send + Sync + 'static) {
        assert!(self.looked.set(Box::new(look)).is_ok(), "one hook");
    }
}

/// Where a mailbox will be, for the parts the owner is built from: the executor's retire callback and the frames the
/// owner queues exist before the mailbox does. Holds the mailbox weakly, so a mailbox that is gone drops the message.
pub(crate) struct Address<O: Turn> {
    mailbox: OnceLock<Weak<Mailbox<O>>>,
}

impl<O: Turn> Address<O> {
    /// An address that leads nowhere until [`set`](Self::set).
    pub(crate) fn new() -> Arc<Address<O>> {
        Arc::new(Address {
            mailbox: OnceLock::new(),
        })
    }

    /// Makes the address lead to `mailbox`.
    pub(crate) fn set(&self, mailbox: &Arc<Mailbox<O>>) {
        if self.mailbox.set(Arc::downgrade(mailbox)).is_err() {
            log::error!("a mailbox's address was set twice");
        }
    }

    /// The mailbox, while it exists.
    pub(crate) fn mailbox(&self) -> Option<Arc<Mailbox<O>>> {
        self.mailbox.get().and_then(Weak::upgrade)
    }

    /// Posts `message` to the mailbox, on the calling thread. A mailbox that is gone drops it.
    pub(crate) fn post(&self, message: O::Message) {
        if let Some(mailbox) = self.mailbox() {
            mailbox.post(message);
        }
    }
}

#[cfg(test)]
#[allow(
    clippy::disallowed_methods,
    clippy::disallowed_types,
    reason = "the tests start threads to race posters and join them, and a round waits at a barrier for its posters"
)]
mod tests {
    use super::*;
    use std::hint::black_box;
    use std::mem;
    use std::panic;
    use std::sync::Barrier;
    use std::sync::atomic::AtomicUsize;
    use std::thread;
    use std::time::Duration;

    use euhedral_data_structures::queues::{Consumer, MpscQueue, Single};

    use crate::handoff::Handoff;

    /// How long a posted message may wait for its turn before the test fails.
    const BOUND: Duration = Duration::from_secs(1);

    type Work = Box<dyn FnOnce() + Send>;
    type Queued = (Arc<Mailbox<Probe>>, Arc<Address<Probe>>, Consumer<Work, Single>);

    /// What a test owner does in a turn.
    struct Probe {
        inbox: Consumer<usize, Single>,
        /// Each message handled, by value.
        handled: Vec<u32>,
        /// Whether a turn is running now.
        inside: AtomicBool,
        /// Turns that found another running.
        overlaps: Arc<AtomicUsize>,
        /// The number of messages handled.
        total: Arc<AtomicUsize>,
        /// The number of turns.
        turns: Arc<AtomicUsize>,
        /// Sent once `total` reaches the number a round waits for.
        goals: Vec<(usize, Arc<Handoff<()>>)>,
        /// Burn this many iterations after the inbox is empty, so that a post can land late in the turn.
        linger: usize,
        /// Finish (ending the owner's work) when this message is handled.
        last: Option<usize>,
        done: Option<Arc<Done>>,
        address: Option<Arc<Address<Probe>>>,
        /// What a claim from inside the turn got.
        claimed_inside: Option<Arc<AtomicBool>>,
        /// A request is in flight.
        busy: bool,
        /// Turns that post a message to the owner from inside the turn, which they are the holder of.
        repost: usize,
    }

    struct Done {
        address: Arc<Address<Probe>>,
        admitted: Arc<Handoff<bool>>,
        /// The delivery panics.
        panics: bool,
    }

    impl Probe {
        fn new(inbox: Consumer<usize, Single>) -> Probe {
            Probe {
                inbox,
                handled: Vec::new(),
                inside: AtomicBool::new(false),
                overlaps: Arc::new(AtomicUsize::new(0)),
                total: Arc::new(AtomicUsize::new(0)),
                turns: Arc::new(AtomicUsize::new(0)),
                goals: Vec::new(),
                linger: 0,
                last: None,
                done: None,
                address: None,
                claimed_inside: None,
                busy: false,
                repost: 0,
            }
        }
    }

    impl Turn for Probe {
        type Message = usize;
        type Finished = Arc<Done>;

        fn turn(&mut self) {
            if self.inside.swap(true, Ordering::SeqCst) {
                self.overlaps.fetch_add(1, Ordering::SeqCst);
            }
            self.turns.fetch_add(1, Ordering::SeqCst);
            let mut ended = false;
            while let Some(message) = self.inbox.poll() {
                self.handled.push(message as u32);
                ended |= self.last == Some(message);
                let total = self.total.fetch_add(1, Ordering::SeqCst) + 1;
                for (goal, handoff) in &self.goals {
                    if *goal == total {
                        handoff.send(());
                    }
                }
            }
            for step in 0..self.linger {
                black_box(step);
            }
            if let Some(address) = &self.address {
                if let Some(seen) = &self.claimed_inside
                    && let Some(mailbox) = address.mailbox()
                {
                    seen.store(mailbox.with(|_| ()).is_some(), Ordering::SeqCst);
                }
                if self.repost > 0 {
                    self.repost -= 1;
                    address.post(1_000_000);
                }
            }
            if ended {
                self.last = None;
            } else {
                self.done = None;
            }
            self.inside.store(false, Ordering::SeqCst);
        }

        fn take_finished(&mut self) -> Option<Arc<Done>> {
            self.done.take()
        }

        fn deliver(finished: Arc<Done>) {
            assert!(!finished.panics, "the listener panicked");
            // A listener that starts the next request from here: it needs the claim.
            let admitted = finished
                .address
                .mailbox()
                .is_some_and(|mailbox| mailbox.with(|_| ()).is_some());
            finished.admitted.send(admitted);
        }

        fn idle(&self) -> bool {
            !self.busy
        }
    }

    /// A mailbox whose hand-offs to a worker run on a thread of their own.
    fn threaded(probe: impl FnOnce(Consumer<usize, Single>) -> Probe) -> (Arc<Mailbox<Probe>>, Arc<Address<Probe>>) {
        mailbox_of(
            probe,
            Box::new(|work: Work| {
                thread::spawn(work);
                true
            }),
        )
    }

    /// A mailbox whose hand-offs wait in a queue for the test to run.
    fn queued(probe: impl FnOnce(Consumer<usize, Single>) -> Probe) -> Queued {
        let handoffs = MpscQueue::<Work>::new(8);
        let producer = handoffs.producer();
        let (mailbox, address) = mailbox_of(probe, Box::new(move |work: Work| producer.offer(work).is_ok()));
        (mailbox, address, handoffs.consumer())
    }

    fn mailbox_of(
        probe: impl FnOnce(Consumer<usize, Single>) -> Probe,
        spawn: Spawn,
    ) -> (Arc<Mailbox<Probe>>, Arc<Address<Probe>>) {
        let queue = MpscQueue::<usize>::new(8);
        let producer = queue.producer();
        let owner = probe(queue.consumer());
        let mailbox = Arc::new(Mailbox::new(owner, producer, spawn));
        let address = Address::new();
        address.set(&mailbox);
        (mailbox, address)
    }

    #[test]
    fn messages_from_8_threads_are_each_handled_once_and_turns_never_overlap() {
        const POSTERS: usize = 8;
        const EACH: usize = 10_000;
        let all = Handoff::new();
        let (mailbox, _address) = threaded(|inbox| {
            let mut probe = Probe::new(inbox);
            probe.goals = vec![(POSTERS * EACH, Arc::clone(&all))];
            probe
        });
        let (overlaps, total) = mailbox
            .with(|probe| (Arc::clone(&probe.overlaps), Arc::clone(&probe.total)))
            .expect("free");
        for poster in 0..POSTERS {
            let mailbox = Arc::clone(&mailbox);
            thread::spawn(move || {
                for message in 0..EACH {
                    mailbox.post(poster * EACH + message);
                }
            });
        }
        // Detached posters: a lost message fails here at the bound instead of hanging a join.
        assert!(
            all.wait(Duration::from_secs(10)).is_ok(),
            "only {} of {} messages were handled",
            total.load(Ordering::SeqCst),
            POSTERS * EACH
        );
        assert_eq!(overlaps.load(Ordering::SeqCst), 0, "two turns overlapped");
        assert_eq!(total.load(Ordering::SeqCst), POSTERS * EACH);
        // The last handled message's turn may still be running on a poster: the claim is taken when it ends.
        let mut handled = loop {
            if let Some(handled) = mailbox.with(|probe| mem::take(&mut probe.handled)) {
                break handled;
            }
        };
        handled.sort_unstable();
        assert!(
            handled.iter().copied().eq(0..(POSTERS * EACH) as u32),
            "a message was lost or handled twice"
        );
    }

    #[test]
    fn a_post_that_races_the_end_of_a_turn_is_handled() {
        const ROUNDS: usize = 2_000;
        const POSTERS: usize = 3;
        let goals: Vec<Arc<Handoff<()>>> = (0..ROUNDS).map(|_| Handoff::new()).collect();
        let (mailbox, _address) = threaded(|inbox| {
            let mut probe = Probe::new(inbox);
            probe.linger = 300;
            probe.goals = goals
                .iter()
                .enumerate()
                .map(|(round, goal)| ((round + 1) * POSTERS, Arc::clone(goal)))
                .collect();
            probe
        });
        let start = Arc::new(Barrier::new(POSTERS + 1));
        for poster in 0..POSTERS {
            let (mailbox, start) = (Arc::clone(&mailbox), Arc::clone(&start));
            // Detached: a round that fails leaves the posters at the barrier, and the test must not wait for them.
            thread::spawn(move || {
                for round in 0..ROUNDS {
                    start.wait();
                    mailbox.post(round * POSTERS + poster);
                }
            });
        }
        for (round, goal) in goals.iter().enumerate() {
            start.wait();
            assert!(
                goal.wait(BOUND).is_ok(),
                "round {round}: a message was posted and no turn ran for it"
            );
        }
    }

    #[test]
    fn the_claim_is_free_when_pending_is_loaded_after_a_release() {
        let (mailbox, _address, _handoffs) = queued(Probe::new);
        let held = Arc::new(AtomicUsize::new(0));
        let looks = Arc::new(AtomicUsize::new(0));
        let (held_by_look, looks_by_look) = (Arc::clone(&held), Arc::clone(&looks));
        mailbox.on_look(move |claimed| {
            looks_by_look.fetch_add(1, Ordering::SeqCst);
            if claimed {
                held_by_look.fetch_add(1, Ordering::SeqCst);
            }
        });
        mailbox.post(1);
        assert!(looks.load(Ordering::SeqCst) > 0, "no look was made");
        assert_eq!(
            held.load(Ordering::SeqCst),
            0,
            "pending was loaded while the claim was held: a post that lost the claim after the load is stranded"
        );
    }

    #[test]
    fn a_claim_from_outside_while_a_turn_runs_is_refused() {
        let (mailbox, address, _handoffs) = queued(Probe::new);
        let seen = Arc::new(AtomicBool::new(true));
        mailbox
            .with(|probe| {
                probe.address = Some(Arc::clone(&address));
                probe.claimed_inside = Some(Arc::clone(&seen));
            })
            .expect("free");
        mailbox.post(1);
        assert!(!seen.load(Ordering::SeqCst), "a claim was granted during a turn");
        assert!(mailbox.with(|_| ()).is_some(), "the claim is free after the turn");
    }

    #[test]
    fn a_finished_outcome_is_delivered_after_the_claim_is_released() {
        let (mailbox, address, _handoffs) = queued(Probe::new);
        let admitted = Handoff::new();
        mailbox
            .with(|probe| {
                probe.last = Some(7);
                probe.done = Some(Arc::new(Done {
                    address: Arc::clone(&address),
                    admitted: Arc::clone(&admitted),
                    panics: false,
                }));
            })
            .expect("free");
        mailbox.post(7);
        assert_eq!(
            admitted.wait(BOUND),
            Ok(true),
            "the listener could not start the next request from the outcome"
        );
    }

    #[test]
    fn a_message_that_arrives_while_a_thread_outside_holds_the_claim_goes_to_a_worker() {
        let (mailbox, _address, handoffs) = queued(Probe::new);
        let poster = Arc::clone(&mailbox);
        mailbox.with(move |_| poster.post(5)).expect("free");
        assert_eq!(
            mailbox.with(|probe| probe.handled.len()),
            Some(0),
            "the claimer ran the turn"
        );
        let mut handoffs = handoffs;
        let work = handoffs.poll().expect("the waiting message was handed to a worker");
        work();
        assert_eq!(mailbox.with(|probe| probe.handled.clone()), Some(vec![5]));
    }

    #[test]
    fn a_drive_runs_a_bounded_number_of_turns_and_hands_the_rest_to_a_worker() {
        let (mailbox, address, mut handoffs) = queued(Probe::new);
        let turns = mailbox
            .with(|probe| {
                probe.address = Some(Arc::clone(&address));
                probe.repost = TURNS_PER_DRIVE + 1;
                Arc::clone(&probe.turns)
            })
            .expect("free");
        mailbox.post(1);
        assert_eq!(
            turns.load(Ordering::SeqCst),
            TURNS_PER_DRIVE,
            "the frame ran more than its turns"
        );
        handoffs.poll().expect("the rest was handed to a worker")();
        assert_eq!(
            turns.load(Ordering::SeqCst),
            TURNS_PER_DRIVE + 2,
            "the hand-off ran the rest"
        );
        assert!(handoffs.poll().is_none(), "nothing is left to hand on");
    }

    #[test]
    fn a_panic_in_a_claimers_work_hands_a_waiting_message_to_a_worker() {
        let (mailbox, _address, mut handoffs) = queued(Probe::new);
        let poster = Arc::clone(&mailbox);
        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            mailbox.with(move |_| {
                poster.post(3);
                panic!("the inspecting closure panicked");
            })
        }));
        assert!(result.is_err(), "the panic was swallowed");
        handoffs.poll().expect("the waiting message was handed to a worker")();
        assert_eq!(mailbox.with(|probe| probe.handled.clone()), Some(vec![3]));
    }

    #[test]
    fn a_panic_in_a_delivery_hands_a_waiting_message_to_a_worker() {
        let (mailbox, address, mut handoffs) = queued(Probe::new);
        mailbox
            .with(|probe| {
                probe.last = Some(7);
                probe.done = Some(Arc::new(Done {
                    address: Arc::clone(&address),
                    admitted: Handoff::new(),
                    panics: true,
                }));
                // A turn that posts while it holds the claim leaves a message waiting after its release.
                probe.address = Some(Arc::clone(&address));
                probe.repost = 1;
            })
            .expect("free");
        let result = panic::catch_unwind(AssertUnwindSafe(|| mailbox.post(7)));
        assert!(result.is_err(), "the panic was swallowed");
        handoffs.poll().expect("the waiting message was handed to a worker")();
        assert_eq!(mailbox.with(|probe| probe.handled.clone()), Some(vec![7, 1_000_000]));
    }

    #[test]
    fn a_thread_outside_is_refused_while_a_request_is_in_flight_and_admitted_when_idle() {
        let (mailbox, _address, _handoffs) = queued(Probe::new);
        mailbox.with(|probe| probe.busy = true).expect("free");
        assert!(mailbox.with_idle(|_| ()).is_none(), "looked at the owner mid-request");
        assert!(mailbox.close().is_none(), "ended the owner mid-request");
        mailbox
            .with(|probe| probe.busy = false)
            .expect("the refusals gave the claim back");
        assert!(mailbox.with_idle(|_| ()).is_some());
        assert!(mailbox.close().is_some());
    }

    #[test]
    fn a_closed_mailbox_takes_no_turn_and_refuses_claims() {
        let (mailbox, _address, _handoffs) = queued(Probe::new);
        assert!(mailbox.close().is_some());
        assert!(mailbox.close().is_none());
        assert!(mailbox.with(|_| ()).is_none());
        // A message to a closed mailbox ends the call rather than looping on a message nobody takes.
        mailbox.post(1);
    }

    #[test]
    fn a_close_that_races_posts_ends_them_all() {
        const POSTERS: usize = 4;
        let ended = Handoff::new();
        let count = Arc::new(AtomicUsize::new(0));
        let (mailbox, _address) = threaded(Probe::new);
        for _ in 0..POSTERS {
            let (mailbox, ended, count) = (Arc::clone(&mailbox), Arc::clone(&ended), Arc::clone(&count));
            thread::spawn(move || {
                for message in 0..5_000 {
                    mailbox.post(message);
                }
                if count.fetch_add(1, Ordering::SeqCst) + 1 == POSTERS {
                    ended.send(());
                }
            });
        }
        // The claim is busy while a post runs a turn: ask again until the owner is taken.
        let owner = loop {
            if let Some(owner) = mailbox.close() {
                break owner;
            }
        };
        assert!(owner.idle());
        assert!(
            ended.wait(Duration::from_secs(10)).is_ok(),
            "a post did not return after the close"
        );
    }
}
