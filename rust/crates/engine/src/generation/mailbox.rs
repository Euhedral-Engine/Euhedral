//! The mailbox of a sequence's owner: the owner is driven by the messages posted to it, on whichever thread posts.
//!
//! The owner's state is a [`ClaimCell`]: one thread at a time may hold the claim and run a turn of the owner. A frame
//! that has something to tell the owner (a quantum retired, a logits row arrived, a parked quantum's lane work ran)
//! [`post`](Mailbox::post)s a message and then [`drive`](Mailbox::drive)s: it takes the claim if it is free and runs
//! turns while messages keep arriving; if the claim is held it goes on, and the holder sees the message. Nothing polls
//! the owner, and nothing waits for it.
//!
//! **Why no message is stranded.** A poster writes first and tries second: it offers the message, then a `SeqCst`
//! fence, then the claim's compare-and-set. A holder gives the claim back with [`ClaimGuard::release_seq_cst`] (a
//! `SeqCst` store, then a `SeqCst` fence) and only then looks at the inbox. Both sides are a write followed by a read
//! of the other's location, with a `SeqCst` fence between them, so at least one of them sees the other's write: either
//! the poster's compare-and-set finds the claim free and the poster holds it, or the holder's look finds the message
//! and the holder runs another turn. A poster that lost the compare-and-set before the holder's store has also written
//! its message before the holder's look, so the look finds it. The claim is given back before the look, never after:
//! looking while still holding it would leave a window in which a poster fails the compare-and-set after the look.
//!
//! The same holds for every thread that claims the owner and gives it back (`with`): it looks at the inbox after
//! `release_seq_cst` and says whether something waits, so that its caller can have a worker drive.

use std::fmt::{self, Formatter};
use std::sync::atomic::{Ordering, fence};
use std::sync::{Arc, OnceLock, Weak};

use euhedral_data_structures::queues::{Multi, Producer};
use euhedral_data_structures::sync::ClaimCell;

/// What a mailbox drives: the state of an owner, and what it does in a turn.
pub(crate) trait Turn: Send {
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
}

/// The inbox of an owner and the claim on its state.
pub(crate) struct Mailbox<O: Turn> {
    /// The owner, until [`close`](Self::close) takes it.
    owner: ClaimCell<Option<O>>,
    /// The producing end of the owner's inbox; the consuming end lives inside the owner.
    inbox: Producer<O::Message, Multi>,
}

impl<O: Turn> fmt::Debug for Mailbox<O> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("Mailbox").finish_non_exhaustive()
    }
}

impl<O: Turn> Mailbox<O> {
    /// A mailbox over `owner`, whose inbox's producing end is `inbox`.
    pub(crate) fn new(owner: O, inbox: Producer<O::Message, Multi>) -> Mailbox<O> {
        Mailbox {
            owner: ClaimCell::new(Some(owner)),
            inbox,
        }
    }

    /// Tells the owner `message` and runs its turns if no one else is running them. Never waits.
    pub(crate) fn post(&self, message: O::Message) {
        if self.inbox.offer(message).is_err() {
            log::error!("the owner's inbox refused a message");
            return;
        }
        // The message is written before the claim is tried: see the module documentation. The queue's own ordering is
        // weaker than `SeqCst`, so the fence makes the write one the holder's look after its release cannot miss.
        fence(Ordering::SeqCst);
        self.drive();
    }

    /// Runs turns of the owner while the claim is free and messages wait. A caller that finds the claim held returns:
    /// the holder looks at the inbox after it releases the claim.
    pub(crate) fn drive(&self) {
        while let Some(mut claim) = self.owner.try_claim() {
            let Some(owner) = claim.as_mut() else {
                // Closed: nothing will ever take a turn again.
                return;
            };
            owner.turn();
            let finished = owner.take_finished();
            // The claim goes back before the inbox is looked at, and before anyone is told: a listener that starts
            // the next request from `finished` finds the owner free.
            claim.release_seq_cst();
            if let Some(finished) = finished {
                O::deliver(finished);
            }
            if self.inbox.is_empty() {
                return;
            }
        }
    }

    /// Runs `work` on the owner if the claim is free, for a thread that is not a frame (it starts a request, or looks
    /// at the state). `None` when another holds the claim, or the owner is closed. Also says whether the inbox holds a
    /// message after the claim was given back: a message that arrived while `work` held the claim was not seen by its
    /// poster, and the caller must have a worker [`drive`](Self::drive).
    pub(crate) fn with<R>(&self, work: impl FnOnce(&mut O) -> R) -> Option<(R, bool)> {
        let mut claim = self.owner.try_claim()?;
        let out = work(claim.as_mut()?);
        claim.release_seq_cst();
        Some((out, !self.inbox.is_empty()))
    }

    /// Takes the owner out, for its teardown. `None` when another holds the claim or it is taken already; the mailbox
    /// is closed from here on.
    pub(crate) fn close(&self) -> Option<O> {
        let mut claim = self.owner.try_claim()?;
        let owner = claim.take();
        claim.release_seq_cst();
        owner
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
    use std::sync::Barrier;
    use std::sync::atomic::{AtomicBool, AtomicUsize};
    use std::thread;
    use std::time::Duration;

    use euhedral_data_structures::queues::{Consumer, MpscQueue, Single};

    use crate::handoff::Handoff;

    /// How long a posted message may wait for its turn before the test fails.
    const BOUND: Duration = Duration::from_secs(1);

    /// What a test owner does at the end of a turn.
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
        /// Sent once `total` reaches the number a round waits for.
        goals: Vec<(usize, Arc<Handoff<()>>)>,
        /// Burn this many iterations after the inbox is empty, so that a post can land late in the turn.
        linger: usize,
        /// Finish (ending the owner's work) when this message is handled.
        last: Option<usize>,
        done: Option<Arc<Done>>,
        address: Option<Arc<Address<Probe>>>,
        /// What a message did to the mailbox from inside the turn.
        claimed_inside: Option<Arc<AtomicBool>>,
    }

    struct Done {
        address: Arc<Address<Probe>>,
        admitted: Arc<Handoff<bool>>,
    }

    impl Probe {
        fn new(inbox: Consumer<usize, Single>) -> Probe {
            Probe {
                inbox,
                handled: Vec::new(),
                inside: AtomicBool::new(false),
                overlaps: Arc::new(AtomicUsize::new(0)),
                total: Arc::new(AtomicUsize::new(0)),
                goals: Vec::new(),
                linger: 0,
                last: None,
                done: None,
                address: None,
                claimed_inside: None,
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
            if let (Some(address), Some(seen)) = (&self.address, &self.claimed_inside)
                && let Some(mailbox) = address.mailbox()
            {
                seen.store(mailbox.with(|_| ()).is_some(), Ordering::SeqCst);
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
            // A listener that starts the next request from here: it needs the claim.
            let admitted = finished
                .address
                .mailbox()
                .is_some_and(|mailbox| mailbox.with(|_| ()).is_some());
            finished.admitted.send(admitted);
        }
    }

    fn mailbox_of(probe: impl FnOnce(Consumer<usize, Single>) -> Probe) -> (Arc<Mailbox<Probe>>, Arc<Address<Probe>>) {
        let queue = MpscQueue::<usize>::new(8);
        let producer = queue.producer();
        let owner = probe(queue.consumer());
        let mailbox = Arc::new(Mailbox::new(owner, producer));
        let address = Address::new();
        address.set(&mailbox);
        (mailbox, address)
    }

    #[test]
    fn messages_from_8_threads_are_each_handled_once_and_turns_never_overlap() {
        const POSTERS: usize = 8;
        const EACH: usize = 10_000;
        let (mailbox, _address) = mailbox_of(Probe::new);
        let (overlaps, total) = mailbox
            .with(|probe| (Arc::clone(&probe.overlaps), Arc::clone(&probe.total)))
            .expect("free")
            .0;
        let threads: Vec<_> = (0..POSTERS)
            .map(|poster| {
                let mailbox = Arc::clone(&mailbox);
                thread::spawn(move || {
                    for message in 0..EACH {
                        mailbox.post(poster * EACH + message);
                    }
                })
            })
            .collect();
        for thread in threads {
            thread.join().expect("a poster ended");
        }
        // Every poster returned from its last post, so the last turn ended: the claim is free and the inbox empty.
        assert_eq!(overlaps.load(Ordering::SeqCst), 0, "two turns overlapped");
        assert_eq!(total.load(Ordering::SeqCst), POSTERS * EACH);
        let (mut handled, _) = mailbox
            .with(|probe| mem::take(&mut probe.handled))
            .expect("free after the posters");
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
        let (mailbox, _address) = mailbox_of(|inbox| {
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
    fn a_claim_from_outside_while_a_turn_runs_is_refused() {
        let (mailbox, address) = mailbox_of(Probe::new);
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
        let (mailbox, address) = mailbox_of(Probe::new);
        let admitted = Handoff::new();
        mailbox
            .with(|probe| {
                probe.last = Some(7);
                probe.done = Some(Arc::new(Done {
                    address: Arc::clone(&address),
                    admitted: Arc::clone(&admitted),
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
    fn a_closed_mailbox_takes_no_turn_and_refuses_claims() {
        let (mailbox, _address) = mailbox_of(Probe::new);
        assert!(mailbox.close().is_some());
        assert!(mailbox.close().is_none());
        assert!(mailbox.with(|_| ()).is_none());
        // A message to a closed mailbox ends the call rather than looping on a non-empty inbox.
        mailbox.post(1);
    }
}
