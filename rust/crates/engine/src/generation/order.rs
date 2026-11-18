//! The workspace's owner: the one place that decides what a quantum starts behind.
//!
//! The engine has one workspace, and its buffers are reused layer after layer inside a quantum, so the quanta of every
//! sequence use it one at a time. A quantum therefore starts behind two quanta (two [`ExternalEdge`]s of the executor):
//!
//! - **the last on the workspace**, whatever sequence it belongs to, so that no two quanta use the buffers at once;
//! - **its own sequence's last**, the quantum before it in its chain, whose selection it takes as its token.
//!
//! With one sequence they are the same quantum and the edge set is the chain's: one edge, to the quantum before it.
//! Work that touches only a sequence's state (`on_workspace` false) takes only the second edge.
//!
//! **Who runs it.** The owner sits in a [`Mailbox`], like a sequence's owner: a sequence's owner posts an admission and
//! drives. With the claim free (always, with one active sequence) the start runs inline in the same turn; otherwise it
//! is a deferred operation, which whoever next takes the claim runs, after the operations posted before it. Nothing
//! waits. Admissions of one sequence are posted by one owner, serially, so they start in post order. No frame of this
//! path is ordered in the lake: the frames that drive the mailbox are unordered (random routing hash), and the order
//! that matters is carried by the mailbox's operations queue and the owners' inboxes, which keep each poster's order.
//! A hash shared by two sequences' frames would let one sequence's stalled head hold the other back.
//!
//! **What holds what.** A [`Launch::Started`] pins its quantum: the quantum retires only once every clone of it is
//! dropped. The owner keeps the clones of the quanta that something may still start behind, and nothing else:
//!
//! - an admission with `more` set says the sequence admits another behind it; the owner keeps it as the sequence's last
//!   (and, on the workspace, as the workspace's last);
//! - an admission without it, or [`release`](WorkspaceOwner::release) of a sequence, says the chain ends: the pins are
//!   dropped so that the quantum can retire. A quantum that is the workspace's last then has nothing to be started
//!   behind, so the owner **gates** the workspace on it: admissions that use the workspace wait in the owner, in order,
//!   until [`Order::retired`] tells it that quantum retired, and then start with no edge to it (a retired quantum has
//!   finished on the device). Admissions that touch only a sequence's state do not wait for it.
//!
//! The gate costs nothing on the path of one sequence: its next admission follows the retirement of the last quantum.

use std::collections::VecDeque;
use std::fmt::{Debug, Formatter, Result as FmtResult};
use std::mem;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use euhedral_data_structures::queues::{Consumer, MpscQueue, Single};

use super::mailbox::{Mailbox, Spawn, Turn};
use crate::handoff::Handoff;
use crate::prefix::CopyJob;
use crate::quantum::{Executor, StartError, Started};
use crate::stages::Quantum;
use crate::state::SequenceId;

/// What stands for "no quantum is gated on".
const NONE: u64 = u64::MAX;

/// Messages of the workspace's owner posted to its inbox.
const INBOX_CHUNK: usize = 4;

/// What the owner starts quanta with.
pub(crate) trait Launch: Send + Sync + 'static {
    /// What is started.
    type Job: Send + 'static;
    /// A started job, which pins it until dropped.
    type Started: Clone + Send + 'static;

    /// The job's own number, distinct for every job.
    fn serial(job: &Self::Job) -> u64;

    /// The number of the job `started` started.
    fn serial_of(started: &Self::Started) -> u64;

    /// Starts `job` behind each of `after` (a job refused, or one that does not start, is dropped).
    fn start(
        &self,
        job: Self::Job,
        kill: Arc<AtomicBool>,
        after: &[&Self::Started],
    ) -> Result<Self::Started, StartError>;
}

/// What an [`Executor`] starts: a quantum, or one of the prefix cache's copy jobs.
pub(crate) enum Job {
    Quantum(Box<Quantum>),
    Copy(CopyJob),
}

impl From<Quantum> for Job {
    fn from(quantum: Quantum) -> Job {
        Job::Quantum(Box::new(quantum))
    }
}

impl From<CopyJob> for Job {
    fn from(job: CopyJob) -> Job {
        Job::Copy(job)
    }
}

impl Launch for Executor {
    type Job = Job;
    type Started = Started;

    fn serial(job: &Job) -> u64 {
        match job {
            Job::Quantum(quantum) => quantum.id(),
            Job::Copy(copy) => copy.serial(),
        }
    }

    fn serial_of(started: &Started) -> u64 {
        started.serial
    }

    fn start(&self, job: Job, kill: Arc<AtomicBool>, after: &[&Started]) -> Result<Started, StartError> {
        match job {
            Job::Quantum(quantum) => Executor::start(self, *quantum, kill, after),
            Job::Copy(copy) => self.start_copy(copy, kill, after),
        }
    }
}

/// Tells a sequence's owner that the quantum numbered `.0` did not start, and why.
pub(crate) type Reply = Box<dyn Fn(u64, StartError) + Send + Sync>;

/// What the workspace's owner is told besides admissions.
pub(crate) enum Message {
    /// The quantum with this number retired.
    Retired(u64),
}

/// An admission that waits for the workspace.
enum Item<L: Launch> {
    Start(Start<L>),
    Release(SequenceId),
}

struct Start<L: Launch> {
    sequence: SequenceId,
    launcher: Arc<L>,
    job: L::Job,
    kill: Arc<AtomicBool>,
    on_workspace: bool,
    more: bool,
}

impl<L: Launch> Item<L> {
    fn sequence(&self) -> SequenceId {
        match self {
            Item::Start(start) => start.sequence,
            Item::Release(sequence) => *sequence,
        }
    }

    fn on_workspace(&self) -> bool {
        matches!(self, Item::Start(start) if start.on_workspace)
    }

    /// The number of the job a start admits.
    fn serial(&self) -> Option<u64> {
        match self {
            Item::Start(start) => Some(L::serial(&start.job)),
            Item::Release(_) => None,
        }
    }
}

/// A pinned quantum and the sequence it belongs to.
struct Held<S> {
    sequence: SequenceId,
    started: S,
}

/// The workspace's owner. See the module documentation.
pub(crate) struct WorkspaceOwner<L: Launch> {
    inbox: Consumer<Message, Single>,
    replies: Arc<[Reply]>,
    /// The last quantum on the workspace, while something may still start behind it.
    last: Option<Held<L::Started>>,
    /// Each sequence's last quantum, while the sequence may admit another behind it.
    last_of: Vec<Option<L::Started>>,
    /// The quantum whose retirement the workspace waits for (see the module documentation).
    draining: Option<u64>,
    /// `draining`, for the owners that tell it a quantum retired.
    gate: Arc<AtomicU64>,
    /// Admissions that wait for the gate, in post order.
    waiting: VecDeque<Item<L>>,
}

impl<L: Launch> WorkspaceOwner<L> {
    fn new(inbox: Consumer<Message, Single>, replies: Arc<[Reply]>, gate: Arc<AtomicU64>) -> WorkspaceOwner<L> {
        WorkspaceOwner {
            inbox,
            last_of: (0..replies.len()).map(|_| None).collect(),
            replies,
            last: None,
            draining: None,
            gate,
            waiting: VecDeque::new(),
        }
    }

    /// Starts `job` of `sequence` behind the sequence's last quantum and, with `on_workspace`, the workspace's. `more`
    /// says the sequence admits another quantum behind this one before it may retire; without it nothing is kept.
    /// A job that cannot start now is the error returned; one that waits and then cannot start is told to the sequence's
    /// owner, with its number. Either way the sequence's pins are let go when `more` is not set.
    pub(crate) fn admit(
        &mut self,
        sequence: SequenceId,
        launcher: Arc<L>,
        job: L::Job,
        kill: Arc<AtomicBool>,
        on_workspace: bool,
        more: bool,
    ) -> Result<(), StartError> {
        self.take_in();
        let item = Item::Start(Start {
            sequence,
            launcher,
            job,
            kill,
            on_workspace,
            more,
        });
        if self.blocked(&item, &self.waiting) {
            self.waiting.push_back(item);
            Ok(())
        } else {
            self.run(item)
        }
    }

    /// Ends `sequence`'s chain: nothing will start behind its last quantum, so its pins are dropped and it may retire.
    pub(crate) fn release(&mut self, sequence: SequenceId) {
        self.take_in();
        let item = Item::Release(sequence);
        if self.blocked(&item, &self.waiting) {
            self.waiting.push_back(item);
        } else {
            self.release_now(sequence);
        }
    }

    /// Leaves `sequence`'s pin allocated for ever: its quanta were abandoned with the owner's accounting lost, and they
    /// must not retire. A quantum that is the workspace's last stays the workspace's last.
    pub(crate) fn forget(&mut self, sequence: SequenceId) {
        if let Some(pin) = self.last_of.get_mut(sequence.0).and_then(Option::take) {
            mem::forget(pin);
        }
    }

    /// Whether `item` has to wait: for the gate, or for an earlier item that waits and that it must start behind.
    fn blocked(&self, item: &Item<L>, earlier: &VecDeque<Item<L>>) -> bool {
        if item.on_workspace() {
            self.draining.is_some() || earlier.iter().any(Item::on_workspace)
        } else {
            earlier.iter().any(|before| before.sequence() == item.sequence())
        }
    }

    fn run(&mut self, item: Item<L>) -> Result<(), StartError> {
        match item {
            Item::Start(start) => self.start(start),
            Item::Release(sequence) => {
                self.release_now(sequence);
                Ok(())
            }
        }
    }

    /// Starts a job. One that does not start (or whose start panics) lets go of its sequence's pins: nothing is kept
    /// for a chain that ends there, whatever `more` said.
    fn start(&mut self, start: Start<L>) -> Result<(), StartError> {
        let Start {
            sequence,
            launcher,
            job,
            kill,
            on_workspace,
            more,
        } = start;
        let own = self.last_of.get(sequence.0).and_then(Option::as_ref);
        let shared = self
            .last
            .as_ref()
            .filter(|_| on_workspace)
            .map(|held| &held.started)
            .filter(|last| Some(L::serial_of(last)) != own.map(L::serial_of));
        // A panic in the launcher is a start that failed: what waits behind this one is still started.
        let launched = catch_unwind(AssertUnwindSafe(|| match (own, shared) {
            (Some(own), Some(shared)) => launcher.start(job, kill, &[own, shared]),
            (Some(only), None) | (None, Some(only)) => launcher.start(job, kill, &[only]),
            (None, None) => launcher.start(job, kill, &[]),
        }));
        match launched.unwrap_or(Err(StartError::Panicked)) {
            Ok(started) => {
                self.keep(sequence, started, on_workspace, more);
                Ok(())
            }
            Err(error) => {
                self.release_now(sequence);
                Err(error)
            }
        }
    }

    /// Records a quantum that started: what pins it, and the gate when nothing will start behind it.
    fn keep(&mut self, sequence: SequenceId, started: L::Started, on_workspace: bool, more: bool) {
        if sequence.0 >= self.last_of.len() {
            return;
        }
        if more {
            if on_workspace {
                self.last = Some(Held {
                    sequence,
                    started: started.clone(),
                });
            }
            // The quantum before it is let go here, after the start took its edge.
            self.last_of[sequence.0] = Some(started);
        } else if on_workspace {
            self.gate_on(L::serial_of(&started));
            self.last_of[sequence.0] = None;
        } else {
            // Nothing is kept of it, and the workspace's last, if it is this sequence's, has nothing behind it either.
            self.release_now(sequence);
        }
    }

    fn release_now(&mut self, sequence: SequenceId) {
        let own = self.last_of.get_mut(sequence.0).and_then(Option::take);
        if let Some(serial) = self
            .last
            .as_ref()
            .filter(|held| held.sequence == sequence)
            .map(|held| L::serial_of(&held.started))
        {
            self.gate_on(serial);
        }
        drop(own);
    }

    /// Gates the workspace on the quantum `serial`, which is the workspace's last and is no longer kept. The gate is set
    /// before the pin is dropped, so that the quantum cannot retire before it.
    fn gate_on(&mut self, serial: u64) {
        debug_assert!(self.draining.is_none(), "a gated workspace admits nothing");
        self.draining = Some(serial);
        self.gate.store(serial, Ordering::Release);
        self.last = None;
    }

    fn take_in(&mut self) {
        while let Some(Message::Retired(serial)) = self.inbox.poll() {
            if self.draining == Some(serial) {
                self.draining = None;
                self.gate.store(NONE, Ordering::Release);
            }
        }
    }

    /// Starts what waited, in order, as far as the gate lets it.
    fn flush(&mut self) {
        if self.waiting.is_empty() {
            return;
        }
        let mut kept = VecDeque::new();
        for item in mem::take(&mut self.waiting) {
            if self.blocked(&item, &kept) {
                kept.push_back(item);
                continue;
            }
            let (sequence, serial) = (item.sequence(), item.serial());
            if let (Err(error), Some(serial)) = (self.run(item), serial) {
                self.refuse(sequence, serial, error);
            }
        }
        self.waiting = kept;
    }

    fn refuse(&self, sequence: SequenceId, serial: u64, error: StartError) {
        if let Some(reply) = self.replies.get(sequence.0) {
            reply(serial, error);
        }
    }
}

impl<L: Launch> Turn for WorkspaceOwner<L> {
    type Message = Message;
    type Finished = Arc<Handoff<()>>;

    fn turn(&mut self) {
        self.take_in();
        self.flush();
    }

    fn take_finished(&mut self) -> Vec<Self::Finished> {
        Vec::new()
    }

    /// Whoever waited for the workspace is told its quantum did not start.
    fn abandon(&mut self, _why: &str) -> Vec<Self::Finished> {
        for item in mem::take(&mut self.waiting) {
            if let Item::Start(start) = item {
                let serial = L::serial(&start.job);
                self.release_now(start.sequence);
                self.refuse(start.sequence, serial, StartError::Closed);
            }
        }
        Vec::new()
    }

    fn deliver(_finished: Self::Finished) {}

    fn shut(&mut self, idle: Arc<Handoff<()>>) {
        // No request runs here: what waits is started or refused by the turns that follow, and the sequences' owners are
        // what a shutdown ends.
        idle.send(());
    }

    fn into_barrier(finished: Self::Finished) -> Result<Arc<Handoff<()>>, Self::Finished> {
        Ok(finished)
    }

    fn idle(&self) -> bool {
        self.waiting.is_empty()
    }
}

/// The workspace's owner, its mailbox, and what the sequences' owners use to reach it.
pub(crate) struct Order<L: Launch> {
    mailbox: Arc<Mailbox<WorkspaceOwner<L>>>,
    gate: Arc<AtomicU64>,
    replies: Arc<[Reply]>,
    /// The input records each sequence has.
    records: usize,
}

impl<L: Launch> Debug for Order<L> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        formatter.debug_struct("Order").finish_non_exhaustive()
    }
}

impl<L: Launch> Order<L> {
    /// The owner for the sequences `replies` lead to (one for each, by sequence number), each with `records` input
    /// records; `spawn` runs a frame on a worker.
    pub(crate) fn new(replies: Vec<Reply>, records: usize, spawn: Spawn) -> Order<L> {
        let queue = MpscQueue::<Message>::new(INBOX_CHUNK);
        let (producer, inbox) = (queue.producer(), queue.consumer());
        let gate = Arc::new(AtomicU64::new(NONE));
        let replies: Arc<[Reply]> = replies.into();
        let owner = WorkspaceOwner::new(inbox, Arc::clone(&replies), Arc::clone(&gate));
        Order {
            mailbox: Arc::new(Mailbox::new(owner, producer, spawn)),
            gate,
            replies,
            records: records.max(1),
        }
    }

    /// The input record of `sequence`'s quantum that takes record `local` of the sequence's own: the sequences' records
    /// follow one another, so that no two sequences' quanta share one.
    pub(crate) fn record(&self, sequence: SequenceId, local: usize) -> usize {
        sequence.0 * self.records + local % self.records
    }

    /// The input records each sequence has.
    pub(crate) fn records(&self) -> usize {
        self.records
    }

    /// Starts `job` (see [`WorkspaceOwner::admit`]): now, if the owner is free, else as an operation the owner runs. Never
    /// waits. A job that cannot start now (also because the owner is closed) is the error returned, so that the caller
    /// admits nothing behind it; one that is deferred and cannot start is told to the sequence's owner.
    pub(crate) fn admit(
        &self,
        sequence: SequenceId,
        launcher: &Arc<L>,
        job: impl Into<L::Job>,
        kill: Arc<AtomicBool>,
        on_workspace: bool,
        more: bool,
    ) -> Result<(), StartError> {
        let job = job.into();
        let serial = L::serial(&job);
        let launcher = Arc::clone(launcher);
        let work = move |owner: &mut WorkspaceOwner<L>| owner.admit(sequence, launcher, job, kill, on_workspace, more);
        match self.mailbox.try_with(work) {
            Ok(Some(started)) => started,
            Ok(None) => Err(StartError::Closed),
            Err(work) => {
                let replies = Arc::clone(&self.replies);
                self.mailbox.defer(move |owner| {
                    let result = match owner {
                        Some(owner) => work(owner),
                        None => Err(StartError::Closed),
                    };
                    if let (Err(error), Some(reply)) = (result, replies.get(sequence.0)) {
                        reply(serial, error);
                    }
                    Vec::new()
                });
                Ok(())
            }
        }
    }

    /// Ends `sequence`'s chain (see [`WorkspaceOwner::release`]).
    pub(crate) fn release(&self, sequence: SequenceId) {
        self.mailbox.tell(move |owner| owner.release(sequence));
    }

    /// Abandons `sequence`'s pin (see [`WorkspaceOwner::forget`]).
    pub(crate) fn forget(&self, sequence: SequenceId) {
        self.mailbox.tell(move |owner| owner.forget(sequence));
    }

    /// Tells the owner the quantum numbered `serial` retired; only the quantum the workspace is gated on matters, and
    /// for any other this is one load.
    pub(crate) fn retired(&self, serial: u64) {
        if self.gate.load(Ordering::Acquire) == serial {
            self.mailbox.post(Message::Retired(serial));
        }
    }

    /// Takes the owner out for the engine's teardown, which drops what it still pins.
    pub(crate) fn close(&self) -> Option<WorkspaceOwner<L>> {
        self.mailbox.close()
    }
}

#[cfg(test)]
#[allow(
    clippy::disallowed_methods,
    clippy::disallowed_types,
    reason = "the fake launcher logs each start from whichever thread runs it"
)]
mod tests {
    use std::collections::HashSet;
    use std::sync::Mutex;
    use std::sync::atomic::AtomicUsize;

    use super::*;
    use crate::plan::View;

    const A: SequenceId = SequenceId(0);
    const B: SequenceId = SequenceId(1);

    /// A started job: its number; a clone is another pin, counted while it lives.
    struct Pin {
        serial: u64,
        pins: Arc<AtomicUsize>,
    }

    impl Pin {
        fn new(serial: u64, pins: &Arc<AtomicUsize>) -> Pin {
            pins.fetch_add(1, Ordering::SeqCst);
            Pin {
                serial,
                pins: Arc::clone(pins),
            }
        }
    }

    impl Clone for Pin {
        fn clone(&self) -> Pin {
            Pin::new(self.serial, &self.pins)
        }
    }

    impl Drop for Pin {
        fn drop(&mut self) {
            self.pins.fetch_sub(1, Ordering::SeqCst);
        }
    }

    #[derive(Default)]
    struct Fake {
        /// For each start: the job, and the jobs it started behind.
        log: Mutex<Vec<(u64, Vec<u64>)>>,
        pins: Arc<AtomicUsize>,
        refuse: Mutex<Vec<u64>>,
        panics: Mutex<Vec<u64>>,
    }

    impl Fake {
        fn starts(&self) -> Vec<(u64, Vec<u64>)> {
            self.log.lock().unwrap().clone()
        }

        fn alive(&self) -> usize {
            self.pins.load(Ordering::SeqCst)
        }
    }

    impl Launch for Fake {
        type Job = u64;
        type Started = Pin;

        fn serial(job: &u64) -> u64 {
            *job
        }

        fn serial_of(started: &Pin) -> u64 {
            started.serial
        }

        fn start(&self, job: u64, _kill: Arc<AtomicBool>, after: &[&Pin]) -> Result<Pin, StartError> {
            assert!(!self.panics.lock().unwrap().contains(&job), "the launcher panics");
            if self.refuse.lock().unwrap().contains(&job) {
                return Err(StartError::NoView(View::Decode));
            }
            let behind = after.iter().map(|started| started.serial).collect();
            self.log.lock().unwrap().push((job, behind));
            Ok(Pin::new(job, &self.pins))
        }
    }

    type Told = Arc<Mutex<Vec<(usize, u64, String)>>>;

    fn owner(sequences: usize) -> (WorkspaceOwner<Fake>, Arc<Fake>, Told, Arc<AtomicU64>) {
        let told: Told = Arc::default();
        let replies = (0..sequences)
            .map(|sequence| {
                let told = Arc::clone(&told);
                Box::new(move |serial, error: StartError| {
                    told.lock().unwrap().push((sequence, serial, error.to_string()))
                }) as Reply
            })
            .collect::<Vec<_>>();
        let queue = MpscQueue::<Message>::new(INBOX_CHUNK);
        let gate = Arc::new(AtomicU64::new(NONE));
        let owner = WorkspaceOwner::new(queue.consumer(), replies.into(), Arc::clone(&gate));
        // The queue's producing end is not needed: the tests tell the owner through `retire`.
        mem::forget(queue);
        (owner, Arc::new(Fake::default()), told, gate)
    }

    fn admit(
        owner: &mut WorkspaceOwner<Fake>,
        fake: &Arc<Fake>,
        sequence: SequenceId,
        job: u64,
        on_workspace: bool,
        more: bool,
    ) -> Result<(), StartError> {
        owner.admit(sequence, Arc::clone(fake), job, Arc::default(), on_workspace, more)
    }

    /// Tells the owner `serial` retired, as the sequence's owner does when the gate names it.
    fn retire(owner: &mut WorkspaceOwner<Fake>, serial: u64) {
        if owner.gate.load(Ordering::Acquire) == serial {
            owner.draining = None;
            owner.gate.store(NONE, Ordering::Release);
            owner.flush();
        }
    }

    #[test]
    fn starts_follow_post_order_and_wait_on_both_edges() {
        let (mut owner, fake, _, _) = owner(2);
        admit(&mut owner, &fake, A, 1, true, true).unwrap();
        admit(&mut owner, &fake, B, 2, true, true).unwrap();
        admit(&mut owner, &fake, A, 3, true, true).unwrap();
        admit(&mut owner, &fake, B, 4, true, true).unwrap();
        // Each waits on the last on the workspace; the sequence's own is a second edge unless it is the same quantum.
        assert_eq!(
            fake.starts(),
            vec![(1, vec![]), (2, vec![1]), (3, vec![1, 2]), (4, vec![2, 3])]
        );
    }

    #[test]
    fn a_job_of_the_sequence_alone_does_not_wait_on_another_sequence() {
        let (mut owner, fake, _, _) = owner(2);
        admit(&mut owner, &fake, A, 1, true, true).unwrap();
        admit(&mut owner, &fake, B, 2, false, true).unwrap();
        admit(&mut owner, &fake, B, 3, false, true).unwrap();
        // The workspace's last stays A's: B's jobs took no edge to it, and B's second waits only on B's first.
        assert_eq!(fake.starts(), vec![(1, vec![]), (2, vec![]), (3, vec![2])]);
        admit(&mut owner, &fake, A, 4, true, true).unwrap();
        assert_eq!(fake.starts()[3], (4, vec![1]));
    }

    /// What a sequence alone started behind before the workspace had an owner: the quantum before it while that was
    /// admitted with `more` and the chain was not ended.
    #[test]
    fn with_one_sequence_the_edges_are_the_chains() {
        // (job, more, release after it)
        let patterns: [&[(u64, bool, bool)]; 5] = [
            // greedy: depth two chains each behind the one before
            &[(1, true, false), (2, true, false), (3, true, false), (4, false, false)],
            // forced and sampled: depth one, nothing chained
            &[(1, false, false), (2, false, false), (3, false, false)],
            // speculative: catch-up and draft chained, the verification ends the chain
            &[
                (1, true, false),
                (2, true, false),
                (3, false, false),
                (4, true, false),
                (5, false, false),
            ],
            // a stop ends the chain after a quantum that expected another
            &[(1, true, false), (2, true, true), (3, true, false), (4, false, false)],
            // the end of the steps
            &[(1, true, false), (2, true, true), (3, false, false)],
        ];
        for pattern in patterns {
            let (mut owner, fake, _, _) = owner(1);
            let mut chain: Option<u64> = None;
            let mut expected = Vec::new();
            for &(job, more, release) in pattern {
                // The sequence's owner admits the next only after the retirement of a quantum with nothing behind it.
                expected.push((job, chain.into_iter().collect::<Vec<_>>()));
                admit(&mut owner, &fake, A, job, true, more).unwrap();
                chain = more.then_some(job);
                if release {
                    owner.release(A);
                    chain = None;
                }
                retire(&mut owner, job);
            }
            assert_eq!(fake.starts(), expected, "{pattern:?}");
            assert_eq!(
                fake.alive(),
                2 * usize::from(chain.is_some()),
                "{pattern:?}: only the chain's last is kept"
            );
        }
    }

    #[test]
    fn the_workspace_waits_for_the_quantum_nothing_is_started_behind() {
        let (mut owner, fake, _, gate) = owner(2);
        admit(&mut owner, &fake, A, 1, true, true).unwrap();
        admit(&mut owner, &fake, B, 2, true, true).unwrap();
        // A's chain ends: its last, the workspace's last, is let go so that it can retire, and the workspace is gated.
        admit(&mut owner, &fake, A, 3, true, false).unwrap();
        assert_eq!(gate.load(Ordering::Acquire), 3);
        // B's next, a job of B that uses only B's state (behind B's own), and a third of A's wait in post order.
        admit(&mut owner, &fake, B, 4, true, true).unwrap();
        admit(&mut owner, &fake, B, 5, false, true).unwrap();
        admit(&mut owner, &fake, A, 6, true, true).unwrap();
        assert_eq!(fake.starts(), vec![(1, vec![]), (2, vec![1]), (3, vec![1, 2])]);
        retire(&mut owner, 3);
        // They start with no edge to the retired quantum, B's behind B's own.
        assert_eq!(
            fake.starts()[3..],
            [(4, vec![2]), (5, vec![4]), (6, vec![4])],
            "in the order they were posted"
        );
        assert_eq!(gate.load(Ordering::Acquire), NONE);
    }

    #[test]
    fn a_release_lets_the_last_quantum_retire_and_gates_the_workspace_on_it() {
        let (mut owner, fake, _, gate) = owner(2);
        admit(&mut owner, &fake, A, 1, true, true).unwrap();
        assert_eq!(fake.alive(), 2);
        owner.release(A);
        assert_eq!(fake.alive(), 0, "nothing is pinned: the quantum can retire");
        assert_eq!(gate.load(Ordering::Acquire), 1);
        admit(&mut owner, &fake, B, 2, true, true).unwrap();
        assert_eq!(
            fake.starts().len(),
            1,
            "B waits for the quantum it cannot be ordered behind"
        );
        retire(&mut owner, 1);
        assert_eq!(fake.starts()[1], (2, vec![]));
        // A release of a sequence that is not the workspace's last gates nothing.
        admit(&mut owner, &fake, A, 3, true, true).unwrap();
        owner.release(B);
        assert_eq!(gate.load(Ordering::Acquire), NONE);
        admit(&mut owner, &fake, B, 4, true, true).unwrap();
        assert_eq!(fake.starts()[3], (4, vec![3]));
    }

    #[test]
    fn a_release_waits_behind_its_sequences_admissions() {
        let (mut owner, fake, _, _) = owner(2);
        admit(&mut owner, &fake, A, 1, true, false).unwrap();
        admit(&mut owner, &fake, B, 2, true, true).unwrap();
        owner.release(B);
        // B's job and its release wait for A's quantum: B's pin is not dropped before its job started.
        assert_eq!(fake.starts().len(), 1);
        retire(&mut owner, 1);
        assert_eq!(fake.starts()[1], (2, vec![]));
        assert_eq!(fake.alive(), 0);
    }

    #[test]
    fn a_start_that_fails_now_is_the_calls_result_and_takes_no_place_on_the_workspace() {
        let (mut owner, fake, told, _) = owner(2);
        fake.refuse.lock().unwrap().push(2);
        admit(&mut owner, &fake, A, 1, true, true).unwrap();
        assert!(admit(&mut owner, &fake, B, 2, true, true).is_err());
        admit(&mut owner, &fake, A, 3, true, true).unwrap();
        // A start that fails now is the call's result: nothing is told, and the caller admits nothing behind it.
        assert!(told.lock().unwrap().is_empty());
        // The refused job took no place on the workspace.
        assert_eq!(fake.starts(), vec![(1, vec![]), (3, vec![1])]);
    }

    #[test]
    fn a_start_that_fails_without_more_lets_go_of_the_chain_before_it() {
        let (mut owner, fake, told, gate) = owner(1);
        fake.refuse.lock().unwrap().push(2);
        // Drafts are admitted with more behind them; the verification that ends the step is not.
        admit(&mut owner, &fake, A, 1, true, true).unwrap();
        assert_eq!(fake.alive(), 2);
        let refused = admit(&mut owner, &fake, A, 2, true, false);
        assert!(matches!(refused, Err(StartError::NoView(_))));
        assert_eq!(fake.alive(), 0, "nothing stays pinned for a chain that ends");
        assert_eq!(
            gate.load(Ordering::Acquire),
            1,
            "the workspace waits for the quantum let go of"
        );
        assert!(
            told.lock().unwrap().is_empty(),
            "a start that fails now is the call's result"
        );
    }

    #[test]
    fn a_quantum_of_the_sequence_alone_that_ends_its_chain_lets_go_of_its_last_on_the_workspace() {
        let (mut owner, fake, _, gate) = owner(2);
        admit(&mut owner, &fake, A, 1, true, true).unwrap();
        admit(&mut owner, &fake, A, 2, false, false).unwrap();
        assert_eq!(fake.alive(), 0);
        assert_eq!(gate.load(Ordering::Acquire), 1);
    }

    #[test]
    fn a_waiting_start_that_fails_is_told_when_the_gate_opens() {
        let (mut owner, fake, told, _) = owner(2);
        fake.refuse.lock().unwrap().push(2);
        admit(&mut owner, &fake, A, 1, true, false).unwrap();
        admit(&mut owner, &fake, B, 2, true, true).unwrap();
        assert!(told.lock().unwrap().is_empty());
        retire(&mut owner, 1);
        let told = told.lock().unwrap();
        assert_eq!(told.len(), 1);
        assert_eq!((told[0].0, told[0].1), (1, 2));
    }

    #[test]
    fn a_launcher_that_panics_is_a_failed_start_and_what_waits_behind_it_still_starts() {
        let (mut owner, fake, told, _) = owner(2);
        fake.panics.lock().unwrap().push(2);
        admit(&mut owner, &fake, A, 1, true, false).unwrap();
        admit(&mut owner, &fake, B, 2, true, true).unwrap();
        admit(&mut owner, &fake, B, 3, true, true).unwrap();
        assert_eq!(fake.starts().len(), 1);
        retire(&mut owner, 1);
        assert_eq!(
            told.lock().unwrap().iter().map(|told| told.1).collect::<Vec<_>>(),
            vec![2]
        );
        assert_eq!(
            fake.starts()[1],
            (3, vec![]),
            "the admission behind the panic was not lost"
        );
        // Started at once, it is the call's result.
        let refused = admit(&mut owner, &fake, B, 2, true, true);
        assert!(matches!(refused, Err(StartError::Panicked)));
    }

    #[test]
    fn the_gate_opens_through_the_owner_when_the_quantum_retires() {
        let order = Order::<Fake>::new(vec![Box::new(|_, _| {}), Box::new(|_, _| {})], 2, Box::new(|_| false));
        let fake = Arc::new(Fake::default());
        order.admit(A, &fake, 1u64, Arc::default(), true, false).unwrap();
        order.admit(B, &fake, 2u64, Arc::default(), true, true).unwrap();
        assert_eq!(fake.starts().len(), 1, "the workspace waits for the quantum 1");
        // Another quantum's retirement opens nothing.
        order.retired(2);
        assert_eq!(fake.starts().len(), 1);
        order.retired(1);
        assert_eq!(fake.starts()[1], (2, vec![]));
    }

    #[test]
    fn a_closed_owner_refuses_every_admission_it_is_given() {
        let told: Told = Arc::default();
        let log = Arc::clone(&told);
        let reply: Reply = Box::new(move |serial, error| log.lock().unwrap().push((0, serial, error.to_string())));
        let order = Order::<Fake>::new(vec![reply], 2, Box::new(|_| false));
        let fake = Arc::new(Fake::default());
        assert!(order.close().is_some());
        // The refusal is the result of the call, so that the caller admits nothing behind it.
        let refused = order.admit(A, &fake, 7u64, Arc::default(), true, true);
        assert!(matches!(refused, Err(StartError::Closed)));
        assert!(told.lock().unwrap().is_empty());
        assert!(fake.starts().is_empty());
    }

    #[test]
    fn admissions_through_the_mailbox_start_in_post_order() {
        let order = Order::<Fake>::new(vec![Box::new(|_, _| {}), Box::new(|_, _| {})], 2, Box::new(|_| false));
        let fake = Arc::new(Fake::default());
        for job in 1u64..=6 {
            order
                .admit(SequenceId(job as usize % 2), &fake, job, Arc::default(), true, true)
                .unwrap();
        }
        let starts = fake.starts();
        assert_eq!(
            starts.iter().map(|start| start.0).collect::<Vec<_>>(),
            vec![1, 2, 3, 4, 5, 6]
        );
        assert_eq!(starts[2], (3, vec![1, 2]));
        // The pins go with the owner at the engine's end.
        drop(order.close());
        assert_eq!(fake.alive(), 0);
    }

    #[test]
    fn the_records_of_the_sequences_never_overlap() {
        let order = Order::<Fake>::new(
            vec![Box::new(|_, _| {}), Box::new(|_, _| {}), Box::new(|_, _| {})],
            3,
            Box::new(|_| false),
        );
        let mut seen = HashSet::new();
        for sequence in 0..3 {
            // However far a sequence counts its quanta, it stays inside its own range.
            for local in 0..20 {
                let record = order.record(SequenceId(sequence), local);
                assert!((sequence * 3..sequence * 3 + 3).contains(&record));
                seen.insert((record, sequence));
            }
        }
        assert_eq!(seen.len(), 9);
        assert_eq!(order.records(), 3);
    }
}
