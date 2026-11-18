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
//!   finished on the device). Admissions that touch only another sequence's state do not wait for it.
//!
//! **A job never starts ahead of its own sequence's last device work.** A sequence's last job whose chain ended is no
//! longer pinned, and may still be running: whatever is admitted for the sequence next (a quantum, or work that touches
//! only its state, such as a prefix-cache copy) waits in the owner until that job retired ([`Order::retired`]), and then
//! starts with no edge to it. This holds whether or not the job was on the workspace, and whichever sequence's work
//! the workspace is gated on.
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
use crate::pass_assembler::{PassAssembler, PassOffer, PassStatistics};
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

    /// The rows `job` takes in a pass.
    fn rows(job: &Self::Job) -> usize;

    /// The class of passes `job` can join: jobs of different classes never share a pass.
    fn class(job: &Self::Job) -> u64;

    /// Whether `members` (a sequence and the job it offered each) can run as one pass: what the pass would refuse,
    /// found without consuming them.
    fn batchable(&self, members: &[(SequenceId, &Self::Job)]) -> bool;

    /// Starts `members` (a sequence, the job it offered and the flag its request is cancelled by, each) as one job
    /// behind each of `after`. `kill` stops the whole job: the pass sets it once every member is cancelled, never for
    /// one.
    fn start_pass(
        &self,
        members: Vec<(SequenceId, Self::Job, Arc<AtomicBool>)>,
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

    fn rows(job: &Job) -> usize {
        match job {
            Job::Quantum(quantum) => quantum.rows(),
            Job::Copy(_) => 0,
        }
    }

    fn class(job: &Job) -> u64 {
        match job {
            Job::Quantum(quantum) => quantum.pass_class(),
            Job::Copy(_) => 0,
        }
    }

    fn batchable(&self, members: &[(SequenceId, &Job)]) -> bool {
        let quanta: Option<Vec<(&Quantum, SequenceId)>> = members
            .iter()
            .map(|(sequence, job)| match job {
                Job::Quantum(quantum) => Some((&**quantum, *sequence)),
                Job::Copy(_) => None,
            })
            .collect();
        quanta.is_some_and(|quanta| Quantum::check_batch(&quanta).is_ok())
    }

    fn start_pass(
        &self,
        members: Vec<(SequenceId, Job, Arc<AtomicBool>)>,
        kill: Arc<AtomicBool>,
        after: &[&Started],
    ) -> Result<Started, StartError> {
        let mut quanta = Vec::with_capacity(members.len());
        for (sequence, job, cancel) in members {
            match job {
                Job::Quantum(quantum) => quanta.push((sequence, *quantum, cancel)),
                Job::Copy(_) => return Err(StartError::Batch("a copy job is not part of a pass".to_string())),
            }
        }
        Executor::start_batch(self, quanta, kill, after)
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
    /// The job is a pass of one, whose retirement the assembler awaits.
    pass: bool,
}

/// A step a sequence's owner offered for a pass: it runs when the assembler forms a pass of it, alone or with others.
struct Offer<L: Launch> {
    sequence: SequenceId,
    launcher: Arc<L>,
    job: L::Job,
    /// The flag the sequence's request is cancelled by.
    kill: Arc<AtomicBool>,
    rows: usize,
    /// The class of passes the step can join.
    class: u64,
}

impl<L: Launch> PassOffer for Offer<L> {
    fn sequence(&self) -> SequenceId {
        self.sequence
    }

    fn rows(&self) -> usize {
        self.rows
    }

    fn class(&self) -> u64 {
        self.class
    }
}

/// How many rows and sequences a pass holds at most.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PassCaps {
    pub(crate) rows: usize,
    pub(crate) parts: usize,
}

impl PassCaps {
    /// The caps of a pass, refused when either is zero (the assembler cannot form a pass of nothing).
    pub(crate) fn new(rows: usize, parts: usize) -> Result<PassCaps, String> {
        if rows == 0 || parts == 0 {
            return Err(format!("a pass of {rows} rows and {parts} sequences holds nothing"));
        }
        Ok(PassCaps { rows, parts })
    }
}

#[derive(Debug, Default)]
struct Counters {
    passes: AtomicU64,
    rows: AtomicU64,
    segments: AtomicU64,
    batched: AtomicU64,
}

impl Counters {
    fn count(&self, rows: usize, segments: usize) {
        self.passes.fetch_add(1, Ordering::Relaxed);
        self.rows.fetch_add(rows as u64, Ordering::Relaxed);
        self.segments.fetch_add(segments as u64, Ordering::Relaxed);
        if segments > 1 {
            self.batched.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn read(&self) -> PassStatistics {
        PassStatistics {
            passes: self.passes.load(Ordering::Relaxed),
            rows: self.rows.load(Ordering::Relaxed),
            segments: self.segments.load(Ordering::Relaxed),
            batched: self.batched.load(Ordering::Relaxed),
        }
    }
}

/// A pass in flight: its members and the flag each is cancelled by, and the flag that stops the whole job when it is
/// one job of several members.
struct Flight {
    members: Vec<(SequenceId, Arc<AtomicBool>)>,
    kill: Option<Arc<AtomicBool>>,
}

impl Flight {
    /// Stops the job once every member is cancelled, never for one.
    fn reevaluate(&self) {
        if let Some(kill) = &self.kill
            && self.members.iter().all(|(_, cancel)| cancel.load(Ordering::Acquire))
        {
            kill.store(true, Ordering::Release);
        }
    }
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

    fn is_pass(&self) -> bool {
        matches!(self, Item::Start(start) if start.pass)
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
    /// Each sequence's last job whose chain ended and whose retirement is not told yet: nothing of the sequence starts
    /// before it.
    ended: Vec<Option<u64>>,
    /// `ended`, for the owners that tell a job retired.
    ended_gate: Arc<[AtomicU64]>,
    /// The verifications offered for a pass, and the rule that forms passes of them.
    assembler: PassAssembler<Offer<L>>,
    /// The passes in flight, by the number of the job each started.
    flights: Vec<(u64, Flight)>,
    counters: Arc<Counters>,
}

impl<L: Launch> WorkspaceOwner<L> {
    fn new(
        inbox: Consumer<Message, Single>,
        replies: Arc<[Reply]>,
        gate: Arc<AtomicU64>,
        ended_gate: Arc<[AtomicU64]>,
        caps: PassCaps,
        counters: Arc<Counters>,
    ) -> WorkspaceOwner<L> {
        WorkspaceOwner {
            assembler: PassAssembler::new(caps.rows, caps.parts),
            flights: Vec::new(),
            counters,
            inbox,
            ended: vec![None; replies.len()],
            ended_gate,
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
            pass: false,
        });
        let started = if self.blocked(&item, &self.waiting) {
            self.waiting.push_back(item);
            Ok(())
        } else {
            self.run(item)
        };
        self.advance();
        started
    }

    /// Offers a verification of `sequence` for a pass. It starts, alone or in a pass with the offers of other sequences,
    /// when the assembler forms the pass and the workspace is free for it; one that cannot start then is told to the
    /// sequence's owner, with its number, as a deferred admission is. An offer the assembler refuses (the sequence has
    /// one pending, or the job has no rows) is the error returned, and nothing of it remains.
    pub(crate) fn offer(
        &mut self,
        sequence: SequenceId,
        launcher: Arc<L>,
        job: L::Job,
        kill: Arc<AtomicBool>,
    ) -> Result<(), StartError> {
        self.settle();
        let rows = L::rows(&job);
        let class = L::class(&job);
        let offer = Offer {
            sequence,
            launcher,
            job,
            kill,
            rows,
            class,
        };
        let offered = self.assembler.offer(offer).map_err(|refusal| {
            // The offer is gone, and so is the chain it ended: nothing starts behind the sequence's last job.
            self.release_now(sequence);
            StartError::Batch(refusal.to_string())
        });
        self.advance();
        offered
    }

    /// `sequence` offers no more steps (its request ended or its next step is not offered): the passes stop waiting for
    /// it. A step it has pending is returned to its owner as not started, and a pass in flight whose members are all
    /// cancelled is stopped.
    pub(crate) fn leave(&mut self, sequence: SequenceId) {
        self.settle();
        if let Some(offer) = self.assembler.leave(sequence) {
            let serial = L::serial(&offer.job);
            self.release_now(sequence);
            self.refuse(sequence, serial, StartError::Withdrawn);
        }
        for (_, flight) in &self.flights {
            flight.reevaluate();
        }
        self.advance();
    }

    /// Takes in what retired, starts what waited for it, and forms the next pass.
    fn settle(&mut self) {
        self.take_in();
        self.flush();
        self.advance();
    }

    /// Whether a pass may start now: the workspace is not gated, nothing that starts behind the workspace waits, and
    /// nothing of an offering sequence waits for its own earlier work.
    fn pass_startable(&self) -> bool {
        self.assembler.pending() > 0
            && self.draining.is_none()
            && !self.waiting.iter().any(Item::on_workspace)
            && !self
                .waiting
                .iter()
                .any(|item| self.assembler.has_pending(item.sequence()))
            && !self
                .ended
                .iter()
                .enumerate()
                .any(|(sequence, ended)| ended.is_some() && self.assembler.has_pending(SequenceId(sequence)))
    }

    /// Forms and starts passes while the rule and the workspace allow one. A pass that did not start owes no retirement,
    /// and the next is asked for at once.
    fn advance(&mut self) {
        while self.pass_startable() {
            let Some(pass) = self.assembler.next_pass() else {
                return;
            };
            let owed = self.start_pass(pass);
            // The count is always told: a pass that failed to start owes none, and `next_pass` would give nothing again.
            let _ = self.assembler.admitted(owed);
            if owed > 0 {
                return;
            }
        }
    }

    /// Starts a pass and returns the retirements it owes: one for a pass that runs as one job, one for each of the
    /// members that run alone.
    fn start_pass(&mut self, pass: Vec<Offer<L>>) -> usize {
        let batched = pass.len() > 1 && {
            let members: Vec<(SequenceId, &L::Job)> = pass.iter().map(|offer| (offer.sequence, &offer.job)).collect();
            // A launcher that panics when asked cannot batch: the members run alone, each as it would without a pass.
            catch_unwind(AssertUnwindSafe(|| pass[0].launcher.batchable(&members))).unwrap_or(false)
        };
        if !batched {
            return self.start_alone(pass);
        }
        let rows = pass.iter().map(|offer| offer.rows).sum();
        let serials: Vec<(SequenceId, u64)> = pass
            .iter()
            .map(|offer| (offer.sequence, L::serial(&offer.job)))
            .collect();
        let launcher = Arc::clone(&pass[0].launcher);
        let members: Vec<(SequenceId, Arc<AtomicBool>)> = pass
            .iter()
            .map(|offer| (offer.sequence, Arc::clone(&offer.kill)))
            .collect();
        let jobs: Vec<(SequenceId, L::Job, Arc<AtomicBool>)> = pass
            .into_iter()
            .map(|offer| (offer.sequence, offer.job, offer.kill))
            .collect();
        let kill = Arc::new(AtomicBool::new(false));
        let mut after: Vec<&L::Started> = Vec::new();
        let mut seen: Vec<u64> = Vec::new();
        let edges = members
            .iter()
            .filter_map(|(sequence, _)| self.last_of.get(sequence.0).and_then(Option::as_ref))
            .chain(self.last.as_ref().map(|held| &held.started));
        for started in edges {
            let serial = L::serial_of(started);
            if !seen.contains(&serial) {
                seen.push(serial);
                after.push(started);
            }
        }
        let launched = catch_unwind(AssertUnwindSafe(|| {
            launcher.start_pass(jobs, Arc::clone(&kill), &after)
        }));
        drop(after);
        match launched.unwrap_or(Err(StartError::Panicked)) {
            Ok(started) => {
                let serial = L::serial_of(&started);
                for (sequence, _) in &members {
                    self.mark_ended(*sequence, serial);
                    if let Some(own) = self.last_of.get_mut(sequence.0) {
                        *own = None;
                    }
                }
                self.gate_on(serial);
                self.counters.count(rows, members.len());
                let flight = Flight {
                    members,
                    kill: Some(kill),
                };
                flight.reevaluate();
                self.flights.push((serial, flight));
                1
            }
            Err(error) => {
                let why = error.to_string();
                for (sequence, serial) in serials {
                    self.release_now(sequence);
                    self.refuse(sequence, serial, StartError::Batch(why.clone()));
                }
                0
            }
        }
    }

    /// Runs the members of a pass one after the other, each as it would run with no pass: the first now, the others
    /// when the one before retired. Returns the retirements owed.
    fn start_alone(&mut self, pass: Vec<Offer<L>>) -> usize {
        let mut owed = 0;
        for offer in pass {
            let (sequence, serial) = (offer.sequence, L::serial(&offer.job));
            let item = Item::Start(Start {
                sequence,
                launcher: offer.launcher,
                job: offer.job,
                kill: offer.kill,
                on_workspace: true,
                more: false,
                pass: true,
            });
            let queued = if self.blocked(&item, &self.waiting) {
                self.waiting.push_back(item);
                Ok(())
            } else {
                self.run(item)
            };
            match queued {
                Ok(()) => owed += 1,
                Err(error) => self.refuse(sequence, serial, error),
            }
        }
        owed
    }

    /// Pays the retirement a pass owed for `sequence`'s job that will never start.
    fn paid(&mut self, sequence: SequenceId) {
        let _ = self.assembler.retired(&[sequence]);
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
        self.advance();
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
        if matches!(item, Item::Start(_)) && self.ended.get(item.sequence().0).is_some_and(Option::is_some) {
            return true;
        }
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
            pass,
        } = start;
        let rows = L::rows(&job);
        let cancel = pass.then(|| Arc::clone(&kill));
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
                if let Some(cancel) = cancel {
                    self.counters.count(rows, 1);
                    self.flights.push((
                        L::serial_of(&started),
                        Flight {
                            members: vec![(sequence, cancel)],
                            kill: None,
                        },
                    ));
                }
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
            self.mark_ended(sequence, L::serial_of(&started));
            self.gate_on(L::serial_of(&started));
            self.last_of[sequence.0] = None;
        } else {
            // Nothing is kept of it, and the workspace's last, if it is this sequence's, has nothing behind it either.
            self.mark_ended(sequence, L::serial_of(&started));
            self.release_now(sequence);
        }
    }

    /// Records that `sequence`'s last job, numbered `serial`, is no longer pinned: nothing of the sequence starts until
    /// it retired. Set before the pin is dropped, so that it cannot retire first.
    fn mark_ended(&mut self, sequence: SequenceId, serial: u64) {
        if let Some(ended) = self.ended.get_mut(sequence.0) {
            *ended = Some(serial);
            self.ended_gate[sequence.0].store(serial, Ordering::Release);
        }
    }

    fn release_now(&mut self, sequence: SequenceId) {
        let own = self.last_of.get_mut(sequence.0).and_then(Option::take);
        // A job that ended the chain after `own` is the later: it has an edge to `own`, so waiting for it covers it.
        if let Some(serial) = own
            .as_ref()
            .map(L::serial_of)
            .filter(|_| self.ended.get(sequence.0) == Some(&None))
        {
            self.mark_ended(sequence, serial);
        }
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
            self.retire(serial);
        }
    }

    /// The job `serial` retired: what waited for it may start.
    fn retire(&mut self, serial: u64) {
        if let Some(at) = self.flights.iter().position(|(started, _)| *started == serial) {
            let (_, flight) = self.flights.swap_remove(at);
            let members: Vec<SequenceId> = flight.members.iter().map(|(sequence, _)| *sequence).collect();
            let _ = self.assembler.retired(&members);
        }
        if self.draining == Some(serial) {
            self.draining = None;
            self.gate.store(NONE, Ordering::Release);
        }
        for (sequence, ended) in self.ended.iter_mut().enumerate() {
            if *ended == Some(serial) {
                *ended = None;
                self.ended_gate[sequence].store(NONE, Ordering::Release);
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
            let (sequence, serial, pass) = (item.sequence(), item.serial(), item.is_pass());
            if let (Err(error), Some(serial)) = (self.run(item), serial) {
                self.refuse(sequence, serial, error);
                if pass {
                    self.paid(sequence);
                }
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
        self.settle();
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
                if start.pass {
                    self.paid(start.sequence);
                }
            }
        }
        for offer in self.assembler.withdraw(|_| true) {
            let serial = L::serial(&offer.job);
            self.release_now(offer.sequence);
            self.refuse(offer.sequence, serial, StartError::Closed);
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
        self.waiting.is_empty() && self.assembler.pending() == 0
    }
}

/// The workspace's owner, its mailbox, and what the sequences' owners use to reach it.
pub(crate) struct Order<L: Launch> {
    mailbox: Arc<Mailbox<WorkspaceOwner<L>>>,
    gate: Arc<AtomicU64>,
    /// Each sequence's last job that ended its chain and has not retired (see [`WorkspaceOwner`]).
    ended: Arc<[AtomicU64]>,
    replies: Arc<[Reply]>,
    /// The input records each sequence has.
    records: usize,
    counters: Arc<Counters>,
}

impl<L: Launch> Debug for Order<L> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        formatter.debug_struct("Order").finish_non_exhaustive()
    }
}

impl<L: Launch> Order<L> {
    /// The owner for the sequences `replies` lead to (one for each, by sequence number), each with `records` input
    /// records; `spawn` runs a frame on a worker.
    pub(crate) fn new(replies: Vec<Reply>, records: usize, caps: PassCaps, spawn: Spawn) -> Order<L> {
        let queue = MpscQueue::<Message>::new(INBOX_CHUNK);
        let (producer, inbox) = (queue.producer(), queue.consumer());
        let gate = Arc::new(AtomicU64::new(NONE));
        let replies: Arc<[Reply]> = replies.into();
        let ended: Arc<[AtomicU64]> = (0..replies.len()).map(|_| AtomicU64::new(NONE)).collect();
        let counters = Arc::new(Counters::default());
        let owner = WorkspaceOwner::new(
            inbox,
            Arc::clone(&replies),
            Arc::clone(&gate),
            Arc::clone(&ended),
            caps,
            Arc::clone(&counters),
        );
        Order {
            counters,
            mailbox: Arc::new(Mailbox::new(owner, producer, spawn)),
            gate,
            ended,
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

    /// Offers a verification of `sequence` for a pass (see [`WorkspaceOwner::offer`]): now, if the owner is free, else as
    /// an operation the owner runs. Never waits. An offer that cannot be taken now is the error returned; one that is
    /// deferred and cannot be taken is told to the sequence's owner. Whatever happens to the offer after this returns
    /// `Ok`, the sequence's owner is told exactly once: its quantum retires, or it did not start.
    pub(crate) fn offer(
        &self,
        sequence: SequenceId,
        launcher: &Arc<L>,
        job: impl Into<L::Job>,
        kill: Arc<AtomicBool>,
    ) -> Result<(), StartError> {
        let job = job.into();
        let serial = L::serial(&job);
        let launcher = Arc::clone(launcher);
        let work = move |owner: &mut WorkspaceOwner<L>| owner.offer(sequence, launcher, job, kill);
        match self.mailbox.try_with(work) {
            Ok(Some(offered)) => offered,
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

    /// `sequence` offers no more steps (see [`WorkspaceOwner::leave`]). A pending offer is told back as not started.
    pub(crate) fn leave(&self, sequence: SequenceId) {
        self.mailbox.tell(move |owner| owner.leave(sequence));
    }

    /// Tells the owner the pass numbered `serial` retired, whichever sequences it holds: the passes' bookkeeping goes
    /// before any member's owner learns of it.
    pub(crate) fn pass_retired(&self, serial: u64) {
        self.mailbox.post(Message::Retired(serial));
    }

    /// What the passes did.
    pub(crate) fn statistics(&self) -> PassStatistics {
        self.counters.read()
    }

    /// Ends `sequence`'s chain (see [`WorkspaceOwner::release`]).
    pub(crate) fn release(&self, sequence: SequenceId) {
        self.mailbox.tell(move |owner| owner.release(sequence));
    }

    /// Abandons `sequence`'s pin (see [`WorkspaceOwner::forget`]).
    pub(crate) fn forget(&self, sequence: SequenceId) {
        self.mailbox.tell(move |owner| owner.forget(sequence));
    }

    /// Tells the owner the job numbered `serial` of `sequence` retired; only the job the workspace is gated on, and the
    /// sequence's own last that ended its chain, matter, and for any other this is two loads.
    pub(crate) fn retired(&self, sequence: SequenceId, serial: u64) {
        let own = self
            .ended
            .get(sequence.0)
            .is_some_and(|ended| ended.load(Ordering::Acquire) == serial);
        if own || self.gate.load(Ordering::Acquire) == serial {
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
        /// Each pass that started as one job: its members, in order, and what it started behind.
        passes: Mutex<Vec<(Vec<u64>, Vec<u64>)>>,
        /// The kill flag of each pass that started as one job.
        kills: Mutex<Vec<Arc<AtomicBool>>>,
        /// Whether the launcher refuses to run passes as one job.
        unbatchable: AtomicBool,
        /// Whether asking the launcher if a pass can be one job panics.
        batchable_panics: AtomicBool,
    }

    /// A job of `rows` rows with the number `serial`.
    fn job(serial: u64, rows: u64) -> u64 {
        serial | (rows << 32)
    }

    /// The number a pass of one job has: far from the jobs'.
    const PASS: u64 = 1_000_000;

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
            *job & 0xFFFF_FFFF
        }

        fn serial_of(started: &Pin) -> u64 {
            started.serial
        }

        fn start(&self, job: u64, _kill: Arc<AtomicBool>, after: &[&Pin]) -> Result<Pin, StartError> {
            let job = Self::serial(&job);
            assert!(!self.panics.lock().unwrap().contains(&job), "the launcher panics");
            if self.refuse.lock().unwrap().contains(&job) {
                return Err(StartError::NoView(View::Decode));
            }
            let behind = after.iter().map(|started| started.serial).collect();
            self.log.lock().unwrap().push((job, behind));
            Ok(Pin::new(job, &self.pins))
        }

        fn rows(job: &u64) -> usize {
            (((*job >> 32) & 0xFFFF) as usize).max(1)
        }

        fn class(job: &u64) -> u64 {
            *job >> 48
        }

        fn batchable(&self, _members: &[(SequenceId, &u64)]) -> bool {
            assert!(!self.batchable_panics.load(Ordering::SeqCst), "the launcher panics");
            !self.unbatchable.load(Ordering::SeqCst)
        }

        fn start_pass(
            &self,
            members: Vec<(SequenceId, u64, Arc<AtomicBool>)>,
            kill: Arc<AtomicBool>,
            after: &[&Pin],
        ) -> Result<Pin, StartError> {
            let serials: Vec<u64> = members.iter().map(|(_, job, _)| Self::serial(job)).collect();
            if serials
                .iter()
                .any(|serial| self.refuse.lock().unwrap().contains(serial))
            {
                return Err(StartError::Batch("refused".to_string()));
            }
            let behind = after.iter().map(|started| started.serial).collect();
            let serial = PASS + serials[0];
            self.passes.lock().unwrap().push((serials, behind));
            self.kills.lock().unwrap().push(kill);
            Ok(Pin::new(serial, &self.pins))
        }
    }

    type Told = Arc<Mutex<Vec<(usize, u64, String)>>>;

    fn owner_of(sequences: usize) -> (WorkspaceOwner<Fake>, Arc<Fake>, Told, Arc<AtomicU64>) {
        owner_with(sequences, PassCaps::new(64, 8).unwrap())
    }

    fn owner_with(sequences: usize, caps: PassCaps) -> (WorkspaceOwner<Fake>, Arc<Fake>, Told, Arc<AtomicU64>) {
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
        let ended: Arc<[AtomicU64]> = (0..sequences).map(|_| AtomicU64::new(NONE)).collect();
        let owner = WorkspaceOwner::new(
            queue.consumer(),
            replies.into(),
            Arc::clone(&gate),
            ended,
            caps,
            Arc::default(),
        );
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
        owner.retire(serial);
        owner.flush();
    }

    #[test]
    fn starts_follow_post_order_and_wait_on_both_edges() {
        let (mut owner, fake, _, _) = owner_of(2);
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
        let (mut owner, fake, _, _) = owner_of(2);
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
            let (mut owner, fake, _, _) = owner_of(1);
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
        let (mut owner, fake, _, gate) = owner_of(2);
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
        let (mut owner, fake, _, gate) = owner_of(2);
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
        // B's own last ended its chain, though, and B starts nothing before it retired.
        admit(&mut owner, &fake, B, 4, true, true).unwrap();
        assert_eq!(fake.starts().len(), 3);
        retire(&mut owner, 2);
        assert_eq!(fake.starts()[3], (4, vec![3]));
    }

    /// The jobs a sequence admits that touch only its state (the prefix cache's copies), in the three places they can
    /// meet the chain.
    #[test]
    fn a_job_of_the_sequence_alone_starts_behind_the_sequences_last_work_in_every_case() {
        // The chain goes on: the job is ordered behind the quantum by an edge.
        {
            let (mut owner, fake, _, _) = owner_of(2);
            admit(&mut owner, &fake, A, 1, true, true).unwrap();
            admit(&mut owner, &fake, A, 2, false, true).unwrap();
            assert_eq!(fake.starts(), vec![(1, vec![]), (2, vec![1])]);
        }

        // The chain ended with the quantum: the quantum is not pinned and may be running, so the job waits for its
        // retirement, and then starts with no edge to it.
        {
            let (mut owner, fake, _, _) = owner_of(2);
            admit(&mut owner, &fake, A, 1, true, false).unwrap();
            admit(&mut owner, &fake, A, 2, false, true).unwrap();
            assert_eq!(fake.starts(), vec![(1, vec![])], "the job waits for the quantum");
            retire(&mut owner, 1);
            assert_eq!(fake.starts(), vec![(1, vec![]), (2, vec![])]);
        }

        // The chain of another sequence ended: the workspace is gated on that quantum, which is no concern of this
        // sequence's job; it starts at once, behind its own sequence's last only.
        let (mut owner, fake, _, _) = owner_of(2);
        admit(&mut owner, &fake, A, 1, true, true).unwrap();
        admit(&mut owner, &fake, B, 2, true, true).unwrap();
        owner.release(B);
        admit(&mut owner, &fake, A, 3, false, true).unwrap();
        assert_eq!(fake.starts()[2], (3, vec![1]));
        admit(&mut owner, &fake, A, 4, false, false).unwrap();
        assert_eq!(fake.starts()[3], (4, vec![3]));
        // The job that ended its chain is waited for in its turn: the sequence's next quantum starts after it retired.
        admit(&mut owner, &fake, A, 5, true, true).unwrap();
        assert_eq!(fake.starts().len(), 4);
        retire(&mut owner, 4);
        assert_eq!(fake.starts().len(), 4, "the workspace is still gated on B's quantum");
        retire(&mut owner, 2);
        assert_eq!(fake.starts()[4], (5, vec![]));
    }

    #[test]
    fn a_job_waits_for_a_chain_ended_by_a_release_too() {
        let (mut owner, fake, _, _) = owner_of(2);
        admit(&mut owner, &fake, A, 1, true, true).unwrap();
        owner.release(A);
        admit(&mut owner, &fake, A, 2, false, true).unwrap();
        assert_eq!(fake.starts().len(), 1);
        retire(&mut owner, 1);
        assert_eq!(fake.starts()[1], (2, vec![]));
    }

    #[test]
    fn a_release_waits_behind_its_sequences_admissions() {
        let (mut owner, fake, _, _) = owner_of(2);
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
        let (mut owner, fake, told, _) = owner_of(2);
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
        let (mut owner, fake, told, gate) = owner_of(1);
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
        let (mut owner, fake, _, gate) = owner_of(2);
        admit(&mut owner, &fake, A, 1, true, true).unwrap();
        admit(&mut owner, &fake, A, 2, false, false).unwrap();
        assert_eq!(fake.alive(), 0);
        assert_eq!(gate.load(Ordering::Acquire), 1);
    }

    #[test]
    fn a_waiting_start_that_fails_is_told_when_the_gate_opens() {
        let (mut owner, fake, told, _) = owner_of(2);
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
        let (mut owner, fake, told, _) = owner_of(2);
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
        let order = Order::<Fake>::new(
            vec![Box::new(|_, _| {}), Box::new(|_, _| {})],
            2,
            PassCaps::new(64, 8).unwrap(),
            Box::new(|_| false),
        );
        let fake = Arc::new(Fake::default());
        order.admit(A, &fake, 1u64, Arc::default(), true, false).unwrap();
        order.admit(B, &fake, 2u64, Arc::default(), true, true).unwrap();
        assert_eq!(fake.starts().len(), 1, "the workspace waits for the quantum 1");
        // Another quantum's retirement opens nothing.
        order.retired(A, 2);
        assert_eq!(fake.starts().len(), 1);
        order.retired(A, 1);
        assert_eq!(fake.starts()[1], (2, vec![]));
    }

    #[test]
    fn a_closed_owner_refuses_every_admission_it_is_given() {
        let told: Told = Arc::default();
        let log = Arc::clone(&told);
        let reply: Reply = Box::new(move |serial, error| log.lock().unwrap().push((0, serial, error.to_string())));
        let order = Order::<Fake>::new(vec![reply], 2, PassCaps::new(64, 8).unwrap(), Box::new(|_| false));
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
        let order = Order::<Fake>::new(
            vec![Box::new(|_, _| {}), Box::new(|_, _| {})],
            2,
            PassCaps::new(64, 8).unwrap(),
            Box::new(|_| false),
        );
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
            PassCaps::new(64, 8).unwrap(),
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

    const C: SequenceId = SequenceId(2);

    fn kill() -> Arc<AtomicBool> {
        Arc::default()
    }

    fn offer(
        owner: &mut WorkspaceOwner<Fake>,
        fake: &Arc<Fake>,
        sequence: SequenceId,
        job: u64,
    ) -> Result<(), StartError> {
        owner.offer(sequence, Arc::clone(fake), job, kill())
    }

    /// Two sequences' offers batched: A's step runs alone first, B's waits, and A's next step joins B's.
    fn batched_pair() -> (WorkspaceOwner<Fake>, Arc<Fake>, Told, Arc<AtomicU64>) {
        let (mut owner, fake, told, gate) = owner_of(3);
        offer(&mut owner, &fake, A, job(1, 4)).unwrap();
        offer(&mut owner, &fake, B, job(2, 4)).unwrap();
        assert_eq!(
            fake.starts(),
            vec![(1, vec![])],
            "the first offer starts at once, alone"
        );
        retire(&mut owner, 1);
        assert_eq!(fake.starts().len(), 1, "B's step waits for A to report");
        offer(&mut owner, &fake, A, job(3, 4)).unwrap();
        (owner, fake, told, gate)
    }

    #[test]
    fn a_pass_of_one_starts_exactly_the_job_an_admission_starts() {
        let (mut owner, fake, _, _) = owner_of(2);
        offer(&mut owner, &fake, A, job(1, 4)).unwrap();
        let (mut plain, plain_fake, _, _) = owner_of(2);
        admit(&mut plain, &plain_fake, A, job(1, 4), true, false).unwrap();
        assert_eq!(fake.starts(), plain_fake.starts());
        assert!(fake.passes.lock().unwrap().is_empty(), "a pass of one is not a batch");
        assert_eq!(
            owner.counters.read(),
            PassStatistics {
                passes: 1,
                rows: 4,
                segments: 1,
                batched: 0
            }
        );
    }

    #[test]
    fn offers_of_several_sequences_run_as_one_job() {
        let (_owner, fake, told, _) = batched_pair();
        let passes = fake.passes.lock().unwrap();
        assert_eq!(passes.len(), 1);
        assert_eq!(passes[0].0, vec![2, 3], "in the order offered");
        assert_eq!(fake.starts().len(), 1, "no member started alone");
        assert!(told.lock().unwrap().is_empty());
    }

    #[test]
    fn the_statistics_count_passes_rows_segments_and_batched_passes() {
        let (owner, _fake, _, _) = batched_pair();
        assert_eq!(
            owner.counters.read(),
            PassStatistics {
                passes: 2,
                rows: 12,
                segments: 3,
                batched: 1
            }
        );
    }

    #[test]
    fn every_started_pass_is_counted_by_its_rows_and_parts() {
        let (mut owner, fake, _, _) = owner_of(2);
        offer(&mut owner, &fake, A, job(1, 1)).unwrap();
        assert_eq!(
            owner.counters.read(),
            PassStatistics {
                passes: 1,
                rows: 1,
                segments: 1,
                batched: 0
            }
        );
        // A pass the launcher refuses to batch is counted as the passes of one its members run as.
        let (mut owner, fake, _, _) = owner_of(2);
        fake.unbatchable.store(true, Ordering::SeqCst);
        offer(&mut owner, &fake, A, job(1, 2)).unwrap();
        offer(&mut owner, &fake, B, job(2, 3)).unwrap();
        retire(&mut owner, 1);
        offer(&mut owner, &fake, A, job(3, 2)).unwrap();
        retire(&mut owner, 2);
        assert_eq!(
            owner.counters.read(),
            PassStatistics {
                passes: 3,
                rows: 7,
                segments: 3,
                batched: 0
            }
        );
    }

    #[test]
    fn a_batch_stops_only_when_every_member_is_cancelled() {
        let (mut owner, fake, _, _) = owner_of(3);
        let (a, b) = (kill(), kill());
        owner.offer(A, Arc::clone(&fake), job(1, 4), kill()).unwrap();
        owner.offer(B, Arc::clone(&fake), job(2, 4), Arc::clone(&b)).unwrap();
        retire(&mut owner, 1);
        owner.offer(A, Arc::clone(&fake), job(3, 4), Arc::clone(&a)).unwrap();
        let batch_kill = Arc::clone(&fake.kills.lock().unwrap()[0]);
        a.store(true, Ordering::Release);
        owner.leave(A);
        assert!(
            !batch_kill.load(Ordering::Acquire),
            "one cancelled member does not stop the others"
        );
        b.store(true, Ordering::Release);
        owner.leave(B);
        assert!(batch_kill.load(Ordering::Acquire), "all cancelled: the job stops");
    }

    #[test]
    fn a_batch_gates_the_workspace_and_its_retirement_opens_it() {
        let (mut owner, _fake, _, gate) = batched_pair();
        assert_eq!(gate.load(Ordering::Acquire), PASS + 2);
        retire(&mut owner, PASS + 2);
        assert_eq!(gate.load(Ordering::Acquire), NONE);
    }

    #[test]
    fn offers_of_different_classes_form_separate_passes() {
        let (mut owner, fake, told, _) = owner_of(3);
        let class = |serial: u64, class: u64| job(serial, 4) | (class << 48);
        offer(&mut owner, &fake, A, class(1, 1)).unwrap();
        offer(&mut owner, &fake, C, class(3, 1)).unwrap();
        offer(&mut owner, &fake, B, class(2, 2)).unwrap();
        retire(&mut owner, 1);
        offer(&mut owner, &fake, A, class(4, 1)).unwrap();
        // C and A share a pass; B, of another class, waits behind it instead of spoiling it.
        assert_eq!(fake.passes.lock().unwrap().len(), 1);
        assert_eq!(fake.passes.lock().unwrap()[0].0, vec![3, 4]);
        assert!(told.lock().unwrap().is_empty());
        assert!(owner.assembler.has_pending(B));
        retire(&mut owner, PASS + 3);
        offer(&mut owner, &fake, C, class(5, 1)).unwrap();
        offer(&mut owner, &fake, A, class(6, 1)).unwrap();
        // B's pass forms next, with its own class: one pass of one job, then the others.
        assert_eq!(fake.starts().last().map(|(job, _)| *job), Some(2));
        assert!(fake.passes.lock().unwrap().len() == 1);
        assert!(told.lock().unwrap().is_empty());
    }

    #[test]
    fn a_launcher_that_panics_when_asked_runs_the_members_alone() {
        let (mut owner, fake, told, _) = owner_of(3);
        fake.batchable_panics.store(true, Ordering::SeqCst);
        offer(&mut owner, &fake, A, job(1, 4)).unwrap();
        offer(&mut owner, &fake, B, job(2, 4)).unwrap();
        retire(&mut owner, 1);
        offer(&mut owner, &fake, A, job(3, 4)).unwrap();
        assert!(fake.passes.lock().unwrap().is_empty());
        assert_eq!(
            fake.starts().iter().map(|(job, _)| *job).collect::<Vec<_>>(),
            vec![1, 2]
        );
        assert!(told.lock().unwrap().is_empty());
        // Each member still pays its retirement, so the next pass forms.
        retire(&mut owner, 2);
        assert_eq!(fake.starts().last().map(|(job, _)| *job), Some(3));
    }

    #[test]
    fn the_retirement_of_a_batched_pass_lets_the_next_pass_form() {
        let (mut owner, fake, _, _) = batched_pair();
        offer(&mut owner, &fake, A, job(4, 4)).unwrap();
        offer(&mut owner, &fake, B, job(5, 4)).unwrap();
        assert_eq!(
            fake.passes.lock().unwrap().len(),
            1,
            "nothing forms while the pass runs"
        );
        retire(&mut owner, PASS + 2);
        owner.settle();
        assert_eq!(
            fake.passes.lock().unwrap().len(),
            2,
            "its retirement releases the next pass"
        );
        assert_eq!(fake.passes.lock().unwrap()[1].0, vec![4, 5]);
    }

    #[test]
    fn a_pass_the_launcher_refuses_to_batch_runs_each_member_alone() {
        let (mut owner, fake, _, _) = owner_of(3);
        fake.unbatchable.store(true, Ordering::SeqCst);
        offer(&mut owner, &fake, A, job(1, 4)).unwrap();
        offer(&mut owner, &fake, B, job(2, 4)).unwrap();
        retire(&mut owner, 1);
        offer(&mut owner, &fake, A, job(3, 4)).unwrap();
        assert!(fake.passes.lock().unwrap().is_empty());
        assert_eq!(
            fake.starts().iter().map(|(job, _)| *job).collect::<Vec<_>>(),
            vec![1, 2]
        );
        // The second member starts behind the first, and the next pass waits for both retirements.
        retire(&mut owner, 2);
        assert_eq!(
            fake.starts().iter().map(|(job, _)| *job).collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
    }

    #[test]
    fn a_batch_that_fails_to_start_tells_every_member_and_the_next_pass_forms() {
        let (mut owner, fake, told, _) = owner_of(3);
        fake.refuse.lock().unwrap().push(3);
        offer(&mut owner, &fake, A, job(1, 4)).unwrap();
        offer(&mut owner, &fake, B, job(2, 4)).unwrap();
        retire(&mut owner, 1);
        offer(&mut owner, &fake, A, job(3, 4)).unwrap();
        assert_eq!(told.lock().unwrap().len(), 2, "both members are told");
        // The failed pass owed nothing: a new pass forms at once.
        offer(&mut owner, &fake, A, job(4, 4)).unwrap();
        assert_eq!(fake.starts().last().map(|(job, _)| *job), Some(4));
    }

    #[test]
    fn a_member_that_fails_alone_still_pays_its_retirement() {
        let (mut owner, fake, told, _) = owner_of(3);
        fake.unbatchable.store(true, Ordering::SeqCst);
        fake.refuse.lock().unwrap().push(2);
        offer(&mut owner, &fake, A, job(1, 4)).unwrap();
        offer(&mut owner, &fake, B, job(2, 4)).unwrap();
        retire(&mut owner, 1);
        offer(&mut owner, &fake, A, job(3, 4)).unwrap();
        // B's job failed to start (told), A's started; once A's retires the next pass may form.
        assert_eq!(told.lock().unwrap().len(), 1);
        retire(&mut owner, 3);
        offer(&mut owner, &fake, A, job(4, 4)).unwrap();
        assert!(
            fake.starts().iter().any(|(job, _)| *job == 4),
            "the pass after the failed one forms"
        );
    }

    #[test]
    fn a_pending_offer_that_leaves_is_returned_and_does_not_block_the_others() {
        let (mut owner, fake, told, _) = owner_of(3);
        offer(&mut owner, &fake, A, job(1, 4)).unwrap();
        offer(&mut owner, &fake, B, job(2, 4)).unwrap();
        offer(&mut owner, &fake, C, job(3, 4)).unwrap();
        owner.leave(B);
        assert_eq!(*told.lock().unwrap(), vec![(1, 2, StartError::Withdrawn.to_string())]);
        retire(&mut owner, 1);
        owner.leave(A);
        assert_eq!(
            fake.starts().iter().map(|(job, _)| *job).collect::<Vec<_>>(),
            vec![1, 3]
        );
    }

    #[test]
    fn a_member_whose_next_step_is_not_offered_stops_the_pass_waiting_for_it() {
        let (mut owner, fake, _, _) = owner_of(3);
        offer(&mut owner, &fake, A, job(1, 4)).unwrap();
        offer(&mut owner, &fake, B, job(2, 4)).unwrap();
        retire(&mut owner, 1);
        assert_eq!(fake.starts().len(), 1, "B waits for A's report");
        owner.leave(A);
        assert_eq!(
            fake.starts().iter().map(|(job, _)| *job).collect::<Vec<_>>(),
            vec![1, 2]
        );
    }

    #[test]
    fn a_second_offer_of_a_sequence_is_refused_and_leaves_nothing() {
        let (mut owner, fake, _, _) = owner_of(2);
        offer(&mut owner, &fake, A, job(1, 4)).unwrap();
        offer(&mut owner, &fake, B, job(2, 4)).unwrap();
        assert!(offer(&mut owner, &fake, B, job(3, 4)).is_err());
        assert_eq!(owner.assembler.pending(), 1, "the refused offer left nothing behind");
    }

    #[test]
    fn closing_the_owner_refuses_the_pending_offers_and_it_goes_idle() {
        let (mut owner, fake, told, _) = owner_of(3);
        offer(&mut owner, &fake, A, job(1, 4)).unwrap();
        offer(&mut owner, &fake, B, job(2, 4)).unwrap();
        assert!(!owner.idle());
        owner.abandon("closed");
        assert_eq!(*told.lock().unwrap(), vec![(1, 2, StartError::Closed.to_string())]);
        assert!(owner.idle());
    }
}
