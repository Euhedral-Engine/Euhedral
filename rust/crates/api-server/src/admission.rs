//! Admission of generations: how many run at once, how many wait, and for how long.
//!
//! The backend holds a number of sequences ([`Backend::max_active_generations`]); a request that finds them all busy
//! waits in order, up to [`Limits::max_queued`], and is refused when the queue is full. A request is refused with the
//! timeout error when it has waited longer than [`Limits::queue_limit`] or has been in the server longer than
//! [`Limits::request_limit`] (a running one is interrupted instead); it leaves the queue when its owner cancels it;
//! and [`Admission::stop`] refuses the waiting ones, interrupts the running ones and reports when every slot is free or
//! [`Limits::shutdown_limit`] has passed.
//!
//! **Who owns what.** The state (the waiting queue, the running entries, the stop) has one owner at a time: a
//! [`ClaimCell`]. Every operation from outside (an offer, a release, a cancel, a stop, a timer that fired) is a command
//! posted to an MPSC inbox, and the poster then publishes a frame through the lattice's sink that drives the owner. The
//! frame takes the claim if it is free and runs turns while commands keep arriving; if the claim is held it returns,
//! and the holder sees the command (the protocol is the engine's mailbox: write, then try; release, then look). Nothing
//! waits for the owner and there is no lock.
//!
//! **What leaves the owner.** A turn only decides. What it decides (start an entry, refuse one, interrupt one, report
//! the stop, arm a timer) is carried out after the claim is given back, and each of those is a frame of its own, so
//! that any worker takes it and an entry's work never runs inside a turn. A frame the lattice refuses (it is closing)
//! still runs its work, on the thread that dropped it: no entry is left without its start or its refusal, and no
//! continuation is dropped.
//!
//! **Time.** The limits are timers of the lattice ([`Completions::at`]), never sleeps. At most the earliest deadline
//! is armed; a timer that fires posts a sweep command, and the turn that handles it expires what is due and arms the
//! next deadline. A timer cannot be cancelled, so one armed for an entry that finished early fires, finds nothing due
//! and arms the next.

use std::collections::VecDeque;
use std::fmt::{self, Debug, Formatter};
use std::mem::take;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering, fence};
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant};

use euhedral_core::frames::Frame;
use euhedral_core::ingest::QueueIngestSink;
use euhedral_data_structures::queues::{Consumer, MpscQueue, Multi, Producer, Single};
use euhedral_data_structures::sync::ClaimCell;
use euhedral_graph::Completions;
use euhedral_inference_api_protocol::ApiError;

use crate::backend::Backend;
use crate::clock::{Clock, Timers};
use crate::frames::once_or_refused;
use crate::limits::Limits;

/// Turns one drive runs before it hands the rest to a new frame, so that a producer that never stops cannot hold a
/// worker.
const TURNS_PER_DRIVE: usize = 4;

/// Slots in a chunk of the inbox.
const COMMANDS_PER_CHUNK: usize = 64;

/// Why a started entry is told to stop. The entry ends its work as soon as it can and then releases its [`Slot`]; it
/// answers its own request with the error that fits ([`ApiError::timeout`], [`ApiError::interrupted_by_shutdown`]; a
/// cancelled entry's owner knows why).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Interruption {
    /// The owner of the entry cancelled it through its [`Handle`] (the client went away).
    Cancelled,
    /// The request limit passed.
    TimedOut,
    /// The server is stopping.
    Shutdown,
    /// The shutdown limit passed and the entry still holds its slot: the last signal. The entry answers with
    /// [`ApiError::interrupted_by_shutdown`] and releases its slot at once, without waiting for its generation to end.
    Abandoned,
}

/// One request's claim on admission. Each entry receives exactly one of [`start`](Entry::start) and
/// [`refuse`](Entry::refuse), each on a worker and none of them waiting. A started entry is [interrupted](Entry::interrupt)
/// at most once for a cause ([`Cancelled`](Interruption::Cancelled), [`TimedOut`](Interruption::TimedOut) or
/// [`Shutdown`](Interruption::Shutdown); the first that happens), and then at most once more with
/// [`Abandoned`](Interruption::Abandoned) if the shutdown limit passes before it released its slot. An interruption is
/// decided while the entry runs but delivered by a frame, so it may arrive before `start` has finished and after the
/// entry released its slot: an entry tolerates one late interruption.
pub trait Entry: Send + Sync + 'static {
    /// A slot is free: run. The entry releases `slot` when its generation has ended on every path (dropping the slot
    /// releases it too). May be preceded by an interruption that arrived while the start was in flight: the entry
    /// remembers it and stops at once.
    fn start(&self, slot: Slot);

    /// The entry never ran and never will: the queue was full, it waited or lived too long, its owner cancelled it, or
    /// the server is stopping.
    fn refuse(&self, error: ApiError);

    /// A started entry must stop: see [`Interruption`]. Called from a worker; must not wait.
    fn interrupt(&self, why: Interruption);

    /// Whether the request limit of a running entry counts from its start instead of its arrival (a stream: the reply
    /// is under way from then on). Waiting always counts from the arrival.
    fn limit_from_start(&self) -> bool {
        false
    }
}

/// Names an offered entry, to cancel it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Handle(u64);

/// A held generation slot. Releasing it (or dropping it) frees the slot and starts the next waiting entry.
pub struct Slot {
    shared: Option<Arc<Shared>>,
    id: u64,
}

impl Debug for Slot {
    fn fmt(&self, out: &mut Formatter<'_>) -> fmt::Result {
        out.debug_struct("Slot").field("id", &self.id).finish_non_exhaustive()
    }
}

impl Slot {
    /// Frees the slot. Returns at once.
    pub fn release(mut self) {
        self.give_back();
    }

    fn give_back(&mut self) {
        if let Some(shared) = self.shared.take() {
            shared.post(Command::Release(self.id));
        }
    }
}

impl Drop for Slot {
    fn drop(&mut self) {
        self.give_back();
    }
}

/// How many entries run and how many wait, as the last turn left them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counts {
    pub active: usize,
    pub waiting: usize,
}

/// What runs once the stop is done.
type Continuation = Box<dyn FnOnce() + Send>;

enum Command {
    Offer {
        id: u64,
        entry: Arc<dyn Entry>,
        arrived: Instant,
    },
    Release(u64),
    Cancel(u64),
    Stop(Continuation),
    /// A timer fired.
    Sweep,
}

/// What a turn decided, carried out once the claim is given back.
enum Effect {
    Start(Arc<dyn Entry>, u64),
    Refuse(Arc<dyn Entry>, ApiError),
    Interrupt(Arc<dyn Entry>, Interruption),
    Arm(Instant),
    Stopped(Vec<Continuation>),
}

struct Queued {
    id: u64,
    entry: Arc<dyn Entry>,
    arrived: Instant,
}

struct Running {
    id: u64,
    entry: Arc<dyn Entry>,
    arrived: Instant,
    /// An interruption for a cause was sent; one is enough.
    interrupted: bool,
    /// The final interruption was sent.
    abandoned: bool,
}

struct Stopping {
    /// When the stop reports itself done without the running entries; `None` when the limit is out of range.
    deadline: Option<Instant>,
    /// The continuations were sent.
    done: bool,
    continuations: Vec<Continuation>,
}

/// The state the claim guards.
struct Owner {
    inbox: Consumer<Command, Single>,
    waiting: VecDeque<Queued>,
    running: Vec<Running>,
    stopping: Option<Stopping>,
    /// The earliest timer armed that has not fired.
    armed: Option<Instant>,
}

/// The limits admission uses, copied out of [`Limits`].
#[derive(Clone, Copy)]
struct Rules {
    max_queued: usize,
    queue_limit: Option<Duration>,
    request_limit: Duration,
    shutdown_limit: Duration,
}

impl Owner {
    fn turn(&mut self, rules: Rules, clock: &dyn Clock, max_active: usize, effects: &mut Vec<Effect>) {
        while let Some(command) = self.inbox.poll() {
            match command {
                Command::Offer { id, entry, arrived } => {
                    self.offer(rules, clock, max_active, id, entry, arrived, effects)
                }
                Command::Release(id) => self.running.retain(|running| running.id != id),
                Command::Cancel(id) => self.cancel(id, effects),
                Command::Stop(continuation) => self.stop(rules, clock, continuation, effects),
                Command::Sweep => self.armed = None,
            }
        }
        let now = clock.now();
        self.expire(rules, now, effects);
        self.promote(clock, max_active, effects);
        self.finish_stop(now, effects);
        self.arm(rules, effects);
    }

    fn offer(
        &mut self,
        rules: Rules,
        clock: &dyn Clock,
        max_active: usize,
        id: u64,
        entry: Arc<dyn Entry>,
        arrived: Instant,
        effects: &mut Vec<Effect>,
    ) {
        if self.stopping.is_some() {
            return effects.push(Effect::Refuse(entry, ApiError::shutting_down()));
        }
        if self.running.len() < max_active && self.waiting.is_empty() {
            return self.run(clock, id, entry, arrived, effects);
        }
        if self.waiting.len() < rules.max_queued {
            self.waiting.push_back(Queued { id, entry, arrived });
        } else {
            effects.push(Effect::Refuse(entry, ApiError::at_capacity()));
        }
    }

    fn run(&mut self, clock: &dyn Clock, id: u64, entry: Arc<dyn Entry>, arrived: Instant, effects: &mut Vec<Effect>) {
        self.running.push(Running {
            id,
            entry: entry.clone(),
            arrived: if entry.limit_from_start() { clock.now() } else { arrived },
            interrupted: false,
            abandoned: false,
        });
        effects.push(Effect::Start(entry, id));
    }

    fn cancel(&mut self, id: u64, effects: &mut Vec<Effect>) {
        if let Some(position) = self.waiting.iter().position(|queued| queued.id == id)
            && let Some(queued) = self.waiting.remove(position)
        {
            return effects.push(Effect::Refuse(queued.entry, ApiError::client_closed()));
        }
        if let Some(running) = self.running.iter_mut().find(|running| running.id == id)
            && !running.interrupted
        {
            running.interrupted = true;
            effects.push(Effect::Interrupt(running.entry.clone(), Interruption::Cancelled));
        }
    }

    fn stop(&mut self, rules: Rules, clock: &dyn Clock, continuation: Continuation, effects: &mut Vec<Effect>) {
        if let Some(stopping) = &mut self.stopping {
            if stopping.done {
                effects.push(Effect::Stopped(vec![continuation]));
            } else {
                stopping.continuations.push(continuation);
            }
            return;
        }
        self.stopping = Some(Stopping {
            deadline: clock.now().checked_add(rules.shutdown_limit),
            done: false,
            continuations: vec![continuation],
        });
        for queued in self.waiting.drain(..) {
            effects.push(Effect::Refuse(queued.entry, ApiError::shutting_down()));
        }
        for running in &mut self.running {
            if !running.interrupted {
                running.interrupted = true;
                effects.push(Effect::Interrupt(running.entry.clone(), Interruption::Shutdown));
            }
        }
    }

    /// Refuses the waiting entries whose limit passed and interrupts the running ones that outlived the request limit.
    fn expire(&mut self, rules: Rules, now: Instant, effects: &mut Vec<Effect>) {
        let due = |since: Instant, limit: Duration| since.checked_add(limit).is_some_and(|deadline| deadline <= now);
        let mut index = 0;
        while index < self.waiting.len() {
            let queued = &self.waiting[index];
            let late =
                due(queued.arrived, rules.request_limit) || rules.queue_limit.is_some_and(|l| due(queued.arrived, l));
            if late && let Some(queued) = self.waiting.remove(index) {
                effects.push(Effect::Refuse(queued.entry, ApiError::timeout()));
            } else {
                index += 1;
            }
        }
        for running in &mut self.running {
            if !running.interrupted && due(running.arrived, rules.request_limit) {
                running.interrupted = true;
                effects.push(Effect::Interrupt(running.entry.clone(), Interruption::TimedOut));
            }
        }
    }

    /// Starts waiting entries, in order, while slots are free.
    fn promote(&mut self, clock: &dyn Clock, max_active: usize, effects: &mut Vec<Effect>) {
        if self.stopping.is_some() {
            return;
        }
        while self.running.len() < max_active {
            let Some(queued) = self.waiting.pop_front() else {
                return;
            };
            self.run(clock, queued.id, queued.entry, queued.arrived, effects);
        }
    }

    /// Reports the stop once every slot is free or the shutdown limit passed.
    fn finish_stop(&mut self, now: Instant, effects: &mut Vec<Effect>) {
        let Some(stopping) = &mut self.stopping else {
            return;
        };
        if stopping.done {
            return;
        }
        let late = stopping.deadline.is_some_and(|deadline| deadline <= now);
        if late {
            for running in &mut self.running {
                if !running.abandoned {
                    running.abandoned = true;
                    effects.push(Effect::Interrupt(running.entry.clone(), Interruption::Abandoned));
                }
            }
        }
        if self.running.is_empty() || late {
            stopping.done = true;
            effects.push(Effect::Stopped(take(&mut stopping.continuations)));
        }
    }

    /// Arms a timer for the earliest deadline, unless one at least as early is armed.
    fn arm(&mut self, rules: Rules, effects: &mut Vec<Effect>) {
        let mut next: Option<Instant> = None;
        let mut consider = |since: Instant, limit: Duration| {
            if let Some(deadline) = since.checked_add(limit) {
                next = Some(next.map_or(deadline, |earliest| earliest.min(deadline)));
            }
        };
        for queued in &self.waiting {
            consider(queued.arrived, rules.request_limit);
            if let Some(limit) = rules.queue_limit {
                consider(queued.arrived, limit);
            }
        }
        for running in self.running.iter().filter(|running| !running.interrupted) {
            consider(running.arrived, rules.request_limit);
        }
        if let Some(stopping) = &self.stopping
            && !stopping.done
            && let Some(deadline) = stopping.deadline
        {
            next = Some(next.map_or(deadline, |earliest| earliest.min(deadline)));
        }
        if let Some(deadline) = next
            && self.armed.is_none_or(|armed| deadline < armed)
        {
            self.armed = Some(deadline);
            effects.push(Effect::Arm(deadline));
        }
    }
}

struct Shared {
    backend: Arc<dyn Backend>,
    rules: Rules,
    /// Weak: a frame queued in the sink holds this state, so a strong reference would be a cycle if the sink is
    /// detached with frames queued.
    sink: Weak<QueueIngestSink>,
    clock: Arc<dyn Clock>,
    inbox: Producer<Command, Multi>,
    owner: ClaimCell<Owner>,
    /// A command was posted that no turn has taken yet.
    pending: AtomicBool,
    next_id: AtomicU64,
    active: AtomicUsize,
    waiting: AtomicUsize,
    me: Weak<Shared>,
}

impl Shared {
    /// Tells the owner `command` and publishes the frame that runs its turns. Never waits.
    fn post(&self, command: Command) {
        if let Err(Command::Offer { entry, .. }) = self.inbox.offer(command) {
            return entry.refuse(ApiError::unavailable("The admission queue refused the request."));
        }
        // The command is written before the claim is tried: see the module documentation.
        self.pending.store(true, Ordering::SeqCst);
        fence(Ordering::SeqCst);
        self.schedule();
    }

    /// Publishes a frame that drives the owner. A frame the lattice refuses drives it on the dropping thread, so no
    /// command waits for a worker that will not come.
    fn schedule(&self) {
        let Some(shared) = self.me.upgrade() else {
            return;
        };
        self.publish(move || shared.drive());
    }

    /// Publishes a frame of work that runs exactly once, on a worker or, if the lattice refuses it, on the thread that
    /// drops it.
    fn publish(&self, work: impl FnOnce() + Send + 'static) {
        self.offer_frame(once_or_refused(move |_| work()));
    }

    /// Offers a frame to the sink; with the sink gone the frame is dropped, which runs its work in place.
    fn offer_frame(&self, frame: Box<dyn Frame>) {
        if let Some(sink) = self.sink.upgrade() {
            drop(sink.offer(frame));
        }
    }

    fn drive(self: &Arc<Self>) {
        for _ in 0..TURNS_PER_DRIVE {
            let Some(mut claim) = self.owner.try_claim() else {
                return;
            };
            self.pending.swap(false, Ordering::SeqCst);
            let mut effects = Vec::new();
            claim.turn(
                self.rules,
                self.clock.as_ref(),
                self.backend.max_active_generations().max(1),
                &mut effects,
            );
            self.active.store(claim.running.len(), Ordering::Release);
            self.waiting.store(claim.waiting.len(), Ordering::Release);
            // The claim goes back before `pending` is loaded and before anyone is told.
            claim.release_seq_cst();
            // An effect that panics (a refused frame runs an entry's work in place) must not lose the others.
            let mut panic = None;
            for effect in effects {
                if let Err(payload) = catch_unwind(AssertUnwindSafe(|| self.carry_out(effect))) {
                    panic.get_or_insert(payload);
                }
            }
            if let Some(payload) = panic {
                self.schedule();
                resume_unwind(payload);
            }
            if !self.pending.load(Ordering::SeqCst) {
                return;
            }
        }
        self.schedule();
    }

    fn carry_out(self: &Arc<Self>, effect: Effect) {
        match effect {
            Effect::Start(entry, id) => {
                let slot = Slot {
                    shared: Some(self.clone()),
                    id,
                };
                self.offer_frame(once_or_refused(move |ran| {
                    if ran {
                        entry.start(slot);
                    } else {
                        entry.refuse(ApiError::unavailable("The server's work queue is closed."));
                        drop(slot);
                    }
                }));
            }
            Effect::Refuse(entry, error) => self.publish(move || entry.refuse(error)),
            Effect::Interrupt(entry, why) => self.publish(move || entry.interrupt(why)),
            Effect::Stopped(continuations) => self.publish(move || {
                for continuation in continuations {
                    continuation();
                }
            }),
            Effect::Arm(deadline) => {
                let me = self.me.clone();
                self.clock.at(
                    deadline,
                    once_or_refused(move |ran| {
                        if ran && let Some(shared) = me.upgrade() {
                            shared.post(Command::Sweep);
                        }
                    }),
                );
            }
        }
    }
}

impl Drop for Shared {
    /// Nothing can drive the owner any more: the entries still waiting are refused, so none is dropped silently.
    fn drop(&mut self) {
        let owner = self.owner.get_mut();
        while let Some(command) = owner.inbox.poll() {
            if let Command::Offer { entry, .. } = command {
                owner.waiting.push_back(Queued {
                    id: 0,
                    entry,
                    arrived: self.clock.now(),
                });
            }
        }
        for queued in owner.waiting.drain(..) {
            queued.entry.refuse(ApiError::shutting_down());
        }
    }
}

/// The admission of generations. Cheap to clone; every clone is the same admission.
#[derive(Clone)]
pub struct Admission {
    shared: Arc<Shared>,
}

impl Debug for Admission {
    fn fmt(&self, out: &mut Formatter<'_>) -> fmt::Result {
        out.debug_struct("Admission")
            .field("counts", &self.counts())
            .finish_non_exhaustive()
    }
}

impl Admission {
    /// An admission over `backend`'s slots with the limits of `limits`. Its frames go through `sink` and its timers
    /// through `timers`; both must be attached to the lattice and outlive the admission's work.
    pub fn new(
        backend: Arc<dyn Backend>,
        limits: &Limits,
        sink: Arc<QueueIngestSink>,
        timers: Arc<Completions>,
    ) -> Admission {
        Admission::with_clock(backend, limits, sink, Arc::new(Timers::new(timers)))
    }

    /// As [`new`](Admission::new), on `clock`: the moments admission reads and the timers it arms are the clock's.
    pub fn with_clock(
        backend: Arc<dyn Backend>,
        limits: &Limits,
        sink: Arc<QueueIngestSink>,
        clock: Arc<dyn Clock>,
    ) -> Admission {
        let queue = MpscQueue::<Command>::new(COMMANDS_PER_CHUNK);
        let inbox = queue.producer();
        let owner = Owner {
            inbox: queue.consumer(),
            waiting: VecDeque::new(),
            running: Vec::new(),
            stopping: None,
            armed: None,
        };
        let shared = Arc::new_cyclic(|me| Shared {
            backend,
            rules: Rules {
                max_queued: limits.max_queued,
                queue_limit: limits.queue_limit,
                request_limit: limits.request_limit,
                shutdown_limit: limits.shutdown_limit,
            },
            sink: Arc::downgrade(&sink),
            clock,
            inbox,
            owner: ClaimCell::new(owner),
            pending: AtomicBool::new(false),
            next_id: AtomicU64::new(0),
            active: AtomicUsize::new(0),
            waiting: AtomicUsize::new(0),
            me: me.clone(),
        });
        Admission { shared }
    }

    /// Offers `entry`: it starts when a slot is free (first come, first served), or is refused with the error that
    /// says why. Returns at once; the request limit counts from this call.
    pub fn offer(&self, entry: Arc<dyn Entry>) -> Handle {
        self.offer_at(entry, self.shared.clock.now())
    }

    /// Offers `entry` as [`offer`](Admission::offer) does, for a request that arrived at `arrived`: the request limit
    /// and the queue limit count from then.
    pub fn offer_at(&self, entry: Arc<dyn Entry>, arrived: Instant) -> Handle {
        let id = self.shared.next_id.fetch_add(1, Ordering::Relaxed);
        self.shared.post(Command::Offer { id, entry, arrived });
        Handle(id)
    }

    /// Cancels an entry: a waiting one leaves the queue and is refused with [`ApiError::client_closed`] (it never
    /// starts and takes no slot); a running one is interrupted with [`Interruption::Cancelled`] and releases its slot
    /// when it ends. An entry that already ended is left alone.
    pub fn cancel(&self, handle: Handle) {
        self.shared.post(Command::Cancel(handle.0));
    }

    /// Stops admission. From now on every offer is refused with [`ApiError::shutting_down`]; the waiting entries are
    /// refused the same way and the running ones are interrupted with [`Interruption::Shutdown`]. `then` runs, on a
    /// worker, once every slot is free or [`Limits::shutdown_limit`] has passed, whichever comes first (at once when
    /// nothing runs, or when a stop already finished).
    pub fn stop(&self, then: impl FnOnce() + Send + 'static) {
        self.shared.post(Command::Stop(Box::new(then)));
    }

    /// How many entries run and wait, as the last turn left them.
    pub fn counts(&self) -> Counts {
        Counts {
            active: self.shared.active.load(Ordering::Acquire),
            waiting: self.shared.waiting.load(Ordering::Acquire),
        }
    }
}
