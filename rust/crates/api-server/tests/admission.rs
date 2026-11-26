//! Admission of generations: slots, the waiting queue, the time limits, cancellation and the stop. Each test has a
//! lattice of its own; entries report to the test thread through a channel it reads with a bound.
#![allow(
    clippy::expect_used,
    clippy::disallowed_types,
    clippy::disallowed_methods,
    reason = "test fixture: the test thread is not a lattice worker; it waits on a channel with a bound"
)]

mod common;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, Sender, TryRecvError, channel};
use std::sync::{Arc, Mutex, PoisonError};
use std::thread::{sleep, spawn, yield_now};
use std::time::{Duration, Instant};

use common::{Fixture, HANG_GUARD, ManualClock, lattice, small_limits};
use euhedral_core::control_plane::ControlPlaneLattice;
use euhedral_core::ingest::{IngestSink, QueueIngestSink};
use euhedral_inference_api_protocol::ApiError;
use euhedral_inference_api_server::scripted::{Script, ScriptedBackend};
use euhedral_inference_api_server::{
    Admission, Backend, BackendError, EncodedPrompt, Entry, Generation, GenerationConfig, GenerationResult, Handle,
    Interruption, Limits, OutputSpec, Slot, TextListener, once,
};

/// The ways a test gives a limit time to pass.
const LIMIT: Duration = Duration::from_millis(150);

#[derive(Debug, PartialEq, Eq)]
enum Event {
    Started(usize),
    Refused(usize, ApiError),
    Interrupted(usize, Interruption),
}

/// What a probe does when it is interrupted.
#[derive(Clone, Copy, PartialEq, Eq)]
enum OnInterrupt {
    /// Releases its slot, as a generation that stopped does.
    Release,
    /// Keeps its slot, as a generation that never stops does.
    Ignore,
    /// Keeps its slot until it is told to give up.
    ReleaseWhenAbandoned,
}

/// An entry that reports and holds its slot until the test says so.
struct Probe {
    name: usize,
    events: Sender<Event>,
    slot: Mutex<Option<Slot>>,
    on_interrupt: OnInterrupt,
    running: Arc<AtomicUsize>,
    most: Arc<AtomicUsize>,
    from_start: bool,
}

impl Probe {
    /// Gives the slot back.
    fn finish(&self) {
        let slot = self.slot.lock().unwrap_or_else(PoisonError::into_inner).take();
        if let Some(slot) = slot {
            self.running.fetch_sub(1, Ordering::AcqRel);
            slot.release();
        }
    }

    /// Fails without releasing: the slot drops with the entry's state, as on an error path.
    fn fail(&self) {
        let slot = self.slot.lock().unwrap_or_else(PoisonError::into_inner).take();
        if slot.is_some() {
            self.running.fetch_sub(1, Ordering::AcqRel);
        }
        drop(slot);
    }
}

impl Entry for Probe {
    fn start(&self, slot: Slot) {
        let now = self.running.fetch_add(1, Ordering::AcqRel) + 1;
        self.most.fetch_max(now, Ordering::AcqRel);
        *self.slot.lock().unwrap_or_else(PoisonError::into_inner) = Some(slot);
        self.events.send(Event::Started(self.name)).ok();
    }

    fn refuse(&self, error: ApiError) {
        self.events.send(Event::Refused(self.name, error)).ok();
    }

    fn limit_from_start(&self) -> bool {
        self.from_start
    }

    fn interrupt(&self, why: Interruption) {
        self.events.send(Event::Interrupted(self.name, why)).ok();
        match self.on_interrupt {
            OnInterrupt::Release => self.finish(),
            OnInterrupt::ReleaseWhenAbandoned if why == Interruption::Abandoned => self.finish(),
            OnInterrupt::ReleaseWhenAbandoned | OnInterrupt::Ignore => {}
        }
    }
}

struct Rig {
    lattice: Arc<ControlPlaneLattice>,
    backend: Arc<ScriptedBackend>,
    sink: Arc<QueueIngestSink>,
    clock: Arc<ManualClock>,
    admission: Admission,
    events: Receiver<Event>,
    sender: Sender<Event>,
    running: Arc<AtomicUsize>,
    most: Arc<AtomicUsize>,
}

fn limits(max_queued: usize) -> Limits {
    Limits {
        max_queued,
        ..small_limits()
    }
}

impl Rig {
    fn new(max_active: usize, limits: Limits) -> Rig {
        let lattice = lattice(Some(4));
        let backend = ScriptedBackend::on(&lattice);
        backend.set_max_active(max_active);
        let sink = Arc::new(QueueIngestSink::new());
        lattice.add_upstream(sink.delegate()).expect("the sink is attached");
        let clock = ManualClock::new();
        clock.attach(&lattice);
        let admission = Admission::with_clock(backend.clone(), &limits, sink.clone(), clock.clone());
        let (sender, events) = channel();
        Rig {
            lattice,
            backend,
            sink,
            clock,
            admission,
            events,
            sender,
            running: Arc::new(AtomicUsize::new(0)),
            most: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn probe_with(&self, name: usize, on_interrupt: OnInterrupt) -> Arc<Probe> {
        Arc::new(Probe {
            name,
            events: self.sender.clone(),
            slot: Mutex::new(None),
            on_interrupt,
            running: self.running.clone(),
            most: self.most.clone(),
            from_start: false,
        })
    }

    fn probe(&self, name: usize) -> Arc<Probe> {
        self.probe_with(name, OnInterrupt::Release)
    }

    fn offer(&self, name: usize) -> (Arc<Probe>, Handle) {
        let probe = self.probe(name);
        let handle = self.admission.offer(probe.clone());
        (probe, handle)
    }

    fn next(&self) -> Event {
        self.events.recv_timeout(HANG_GUARD).expect("an event")
    }

    /// The next `count` events, sorted by entry: events of separate frames arrive in either order.
    fn next_set(&self, count: usize) -> Vec<Event> {
        let mut events: Vec<_> = (0..count).map(|_| self.next()).collect();
        events.sort_by_key(|event| match event {
            Event::Started(name) | Event::Refused(name, _) | Event::Interrupted(name, _) => *name,
        });
        events
    }

    /// Waits until `condition` holds, for as long as the test may take.
    fn until(&self, what: &str, condition: impl Fn(&Rig) -> bool) {
        let end = Instant::now() + HANG_GUARD;
        while !condition(self) {
            assert!(Instant::now() < end, "never: {what}; {:?}", self.admission.counts());
            sleep(Duration::from_millis(1));
        }
    }

    /// Waits until a turn left `active` entries running and `waiting` waiting.
    fn until_counts(&self, active: usize, waiting: usize) {
        self.until(&format!("{active} running and {waiting} waiting"), |rig| {
            rig.admission.counts() == euhedral_inference_api_server::Counts { active, waiting }
        });
    }

    /// Waits until the earliest timer armed is the one for `after` on the clock.
    fn until_armed_for(&self, after: Duration) {
        self.until("the timer to be armed for the deadline", |rig| {
            rig.clock.next_deadline() == Some(rig.clock.at_offset(after))
        });
    }

    /// Closes the lattice, which carries out every frame published before, and asserts that no event was left over:
    /// what the test did not expect to happen, did not.
    fn finish(self) {
        self.clock.complete();
        self.sink.complete_gracefully();
        self.backend.release();
        self.lattice.close();
        let strays: Vec<_> = self.events.try_iter().collect();
        assert!(strays.is_empty(), "events nobody expected: {strays:?}");
    }
}

#[test]
fn entries_start_in_the_order_they_were_offered_and_never_exceed_the_slots() {
    let rig = Rig::new(2, limits(8));
    let probes: Vec<_> = (0..6).map(|name| rig.offer(name).0).collect();
    assert_eq!(rig.next_set(2), vec![Event::Started(0), Event::Started(1)]);
    assert_eq!(rig.admission.counts().active, 2);
    assert_eq!(rig.admission.counts().waiting, 4);
    for (finished, started) in [(0, 2), (1, 3), (2, 4), (3, 5)] {
        probes[finished].finish();
        assert_eq!(rig.next(), Event::Started(started));
    }
    probes[4].finish();
    probes[5].finish();
    assert_eq!(rig.most.load(Ordering::Acquire), 2, "never more than two at once");
    rig.finish();
}

#[test]
fn the_slots_are_the_backends_at_the_time_of_the_offer() {
    let rig = Rig::new(1, limits(8));
    rig.backend.set_max_active(3);
    for name in 0..3 {
        rig.offer(name);
    }
    assert_eq!(
        rig.next_set(3),
        vec![Event::Started(0), Event::Started(1), Event::Started(2)]
    );
    rig.finish();
}

#[test]
fn a_request_that_finds_the_queue_full_is_refused_with_the_capacity_error() {
    let rig = Rig::new(1, limits(1));
    let (first, _) = rig.offer(0);
    rig.offer(1);
    rig.offer(2);
    let events = rig.next_set(2);
    assert_eq!(events[0], Event::Started(0));
    let Event::Refused(2, error) = &events[1] else {
        panic!("the third request is refused");
    };
    assert_eq!(*error, ApiError::at_capacity());
    assert_eq!(
        (error.status, error.kind, error.code, error.message.as_str()),
        (
            503,
            "service_unavailable_error",
            Some("engine_unavailable"),
            "The server is at generation capacity; retry later."
        )
    );
    first.finish();
    assert_eq!(rig.next(), Event::Started(1));
    rig.finish();
}

#[test]
fn with_no_queue_a_request_that_cannot_start_at_once_is_refused() {
    let rig = Rig::new(1, limits(0));
    rig.offer(0);
    rig.offer(1);
    assert_eq!(
        rig.next_set(2),
        vec![Event::Started(0), Event::Refused(1, ApiError::at_capacity())]
    );
    assert_eq!(rig.admission.counts().waiting, 0);
    rig.finish();
}

#[test]
fn a_slot_is_freed_by_a_dropped_slot_and_by_a_finish() {
    let rig = Rig::new(1, limits(4));
    let (first, _) = rig.offer(0);
    let (second, _) = rig.offer(1);
    rig.offer(2);
    assert_eq!(rig.next(), Event::Started(0));
    first.fail();
    assert_eq!(rig.next(), Event::Started(1));
    second.finish();
    assert_eq!(rig.next(), Event::Started(2));
    rig.finish();
}

#[test]
fn a_request_that_waits_longer_than_the_request_limit_is_refused_with_the_timeout_error() {
    let rig = Rig::new(
        1,
        Limits {
            request_limit: LIMIT,
            ..limits(4)
        },
    );
    let (first, _) = rig.offer(0);
    rig.offer(1);
    assert_eq!(rig.next(), Event::Started(0));
    rig.until_counts(1, 1);
    // The timer is armed for the request limit from the arrival, and the limit passes when the clock reaches it.
    rig.until_armed_for(LIMIT);
    rig.clock.advance(LIMIT);
    // The running one is interrupted too, and the waiting one is refused: both limits are the same instant.
    let mut seen = [rig.next(), rig.next()];
    seen.sort_by_key(|event| matches!(event, Event::Refused(..)));
    assert_eq!(seen[0], Event::Interrupted(0, Interruption::TimedOut));
    assert_eq!(seen[1], Event::Refused(1, ApiError::timeout()));
    let error = ApiError::timeout();
    assert_eq!(
        (error.status, error.code, error.message.as_str()),
        (
            503,
            Some("timeout"),
            "The request timed out before generation completed."
        )
    );
    drop(first);
    rig.finish();
}

#[test]
fn a_running_request_over_the_request_limit_is_interrupted_once_and_frees_its_slot() {
    let rig = Rig::new(
        1,
        Limits {
            request_limit: LIMIT,
            ..limits(4)
        },
    );
    rig.offer(0);
    assert_eq!(rig.next(), Event::Started(0));
    rig.until_armed_for(LIMIT);
    rig.clock.advance(LIMIT);
    assert_eq!(rig.next(), Event::Interrupted(0, Interruption::TimedOut));
    rig.offer(1);
    assert_eq!(rig.next(), Event::Started(1));
    rig.finish();
}

#[test]
fn the_queue_limit_refuses_a_waiting_request_and_leaves_a_running_one_alone() {
    let rig = Rig::new(
        1,
        Limits {
            queue_limit: Some(LIMIT),
            ..limits(4)
        },
    );
    rig.offer(0);
    rig.offer(1);
    assert_eq!(rig.next(), Event::Started(0));
    rig.until_counts(1, 1);
    rig.until_armed_for(LIMIT);
    rig.clock.advance(LIMIT);
    assert_eq!(rig.next(), Event::Refused(1, ApiError::timeout()));
    // The running request is left alone: the limit that passed is the queue's, and the request limit is far off.
    rig.until_counts(1, 0);
    rig.finish();
}

#[test]
fn a_cancelled_waiting_request_never_starts_and_takes_no_slot() {
    let rig = Rig::new(1, limits(4));
    let (first, _) = rig.offer(0);
    let (_, middle) = rig.offer(1);
    rig.offer(2);
    assert_eq!(rig.next(), Event::Started(0));
    rig.admission.cancel(middle);
    assert_eq!(rig.next(), Event::Refused(1, ApiError::client_closed()));
    assert_eq!(rig.admission.counts().waiting, 1);
    first.finish();
    assert_eq!(rig.next(), Event::Started(2));
    rig.finish();
}

#[test]
fn a_cancelled_running_request_is_interrupted_and_its_slot_goes_to_the_next() {
    let rig = Rig::new(1, limits(4));
    let (_, first) = rig.offer(0);
    rig.offer(1);
    assert_eq!(rig.next(), Event::Started(0));
    rig.admission.cancel(first);
    assert_eq!(rig.next(), Event::Interrupted(0, Interruption::Cancelled));
    assert_eq!(rig.next(), Event::Started(1));
    // Cancelling what already ended does nothing.
    rig.admission.cancel(first);
    rig.finish();
}

#[test]
fn stop_refuses_the_waiting_interrupts_the_running_and_completes_when_they_end() {
    let rig = Rig::new(1, limits(4));
    rig.offer(0);
    rig.offer(1);
    rig.offer(2);
    assert_eq!(rig.next(), Event::Started(0));
    let (done, stopped) = channel();
    rig.admission.stop(move || done.send(()).unwrap_or_default());
    let mut seen = vec![rig.next(), rig.next(), rig.next()];
    seen.sort_by_key(|event| match event {
        Event::Refused(name, _) | Event::Started(name) | Event::Interrupted(name, _) => *name,
    });
    assert_eq!(
        seen,
        vec![
            Event::Interrupted(0, Interruption::Shutdown),
            Event::Refused(1, ApiError::shutting_down()),
            Event::Refused(2, ApiError::shutting_down()),
        ]
    );
    stopped.recv_timeout(HANG_GUARD).expect("the stop completed");
    // After the stop nothing is admitted.
    rig.offer(3);
    assert_eq!(rig.next(), Event::Refused(3, ApiError::shutting_down()));
    assert_eq!(rig.admission.counts().active, 0);
    rig.finish();
}

#[test]
fn stop_with_nothing_running_completes_at_once_and_a_second_stop_too() {
    let rig = Rig::new(1, limits(4));
    for _ in 0..2 {
        let (done, stopped) = channel();
        rig.admission.stop(move || done.send(()).unwrap_or_default());
        stopped.recv_timeout(HANG_GUARD).expect("the stop completed");
    }
    rig.finish();
}

#[test]
fn stop_with_a_generation_that_never_ends_completes_at_the_shutdown_limit() {
    let rig = Rig::new(
        1,
        Limits {
            shutdown_limit: LIMIT,
            ..limits(4)
        },
    );
    let stubborn = rig.probe_with(0, OnInterrupt::Ignore);
    rig.admission.offer(stubborn.clone());
    assert_eq!(rig.next(), Event::Started(0));
    let (done, stopped) = channel();
    rig.admission.stop(move || done.send(()).unwrap_or_default());
    assert_eq!(rig.next(), Event::Interrupted(0, Interruption::Shutdown));
    // The clock has not moved, so the limit has not passed: the generation still holds its slot and the stop waits.
    rig.until_armed_for(LIMIT);
    assert_eq!(
        stopped.try_recv(),
        Err(TryRecvError::Empty),
        "the stop completed before the limit"
    );
    rig.clock.advance(LIMIT);
    stopped
        .recv_timeout(HANG_GUARD)
        .expect("the stop completed at the limit");
    assert_eq!(rig.next(), Event::Interrupted(0, Interruption::Abandoned));
    stubborn.finish();
    rig.finish();
}

#[test]
fn entries_offered_after_the_lattice_stopped_taking_work_are_refused() {
    let rig = Rig::new(1, limits(4));
    rig.sink.complete();
    rig.offer(0);
    let Event::Refused(0, _) = rig.next() else {
        panic!("the entry is refused");
    };
    rig.finish();
}

/// An entry that runs a real (scripted) generation and releases its slot when the generation ended.
struct Generating {
    backend: Arc<ScriptedBackend>,
    generation: Mutex<Option<Arc<dyn Generation>>>,
    cancelled: Mutex<bool>,
    ended: Sender<Result<GenerationResult, BackendError>>,
    begun: Sender<()>,
}

struct Release {
    slot: Option<Slot>,
    generation: Arc<dyn Generation>,
    ended: Sender<Result<GenerationResult, BackendError>>,
    begun: Sender<()>,
}

impl TextListener for Release {
    fn text(&mut self, _text: &str) {
        self.begun.send(()).ok();
    }

    fn finished(&mut self, result: Result<GenerationResult, BackendError>) {
        self.generation.close();
        drop(self.slot.take());
        self.ended.send(result).ok();
    }
}

impl Entry for Generating {
    fn start(&self, slot: Slot) {
        let generation = self
            .backend
            .open_generation(GenerationConfig::greedy(1), OutputSpec::text())
            .expect("a generation");
        *self.generation.lock().unwrap_or_else(PoisonError::into_inner) = Some(generation.clone());
        if *self.cancelled.lock().unwrap_or_else(PoisonError::into_inner) {
            generation.cancel();
        }
        generation.clone().generate(
            EncodedPrompt {
                text: "hi".to_owned(),
                token_ids: ScriptedBackend::prompt_tokens("hi"),
            },
            100_000,
            Box::new(Release {
                slot: Some(slot),
                generation,
                ended: self.ended.clone(),
                begun: self.begun.clone(),
            }),
        );
    }

    fn refuse(&self, _error: ApiError) {}

    fn interrupt(&self, _why: Interruption) {
        *self.cancelled.lock().unwrap_or_else(PoisonError::into_inner) = true;
        if let Some(generation) = &*self.generation.lock().unwrap_or_else(PoisonError::into_inner) {
            generation.cancel();
        }
    }
}

#[test]
fn stop_interrupts_running_generations_and_completes_when_they_ended() {
    let rig = Rig::new(2, limits(4));
    rig.backend.set_script(Script::Endless);
    let (ended, ends) = channel();
    let (begun, begins) = channel();
    for _ in 0..3 {
        rig.admission.offer(Arc::new(Generating {
            backend: rig.backend.clone(),
            generation: Mutex::new(None),
            cancelled: Mutex::new(false),
            ended: ended.clone(),
            begun: begun.clone(),
        }));
    }
    // Two generations run and the third waits.
    for _ in 0..2 {
        begins.recv_timeout(HANG_GUARD).expect("a generation began");
    }
    assert_eq!(rig.admission.counts().waiting, 1);
    let (done, stopped) = channel();
    rig.admission.stop(move || done.send(()).unwrap_or_default());
    stopped.recv_timeout(HANG_GUARD).expect("the stop completed");
    for _ in 0..2 {
        let result = ends.recv_timeout(HANG_GUARD).expect("a generation ended");
        assert!(!result.expect("an ended generation").stop_token_reached);
    }
    let generations = rig.backend.generations();
    assert_eq!(generations.len(), 2, "the waiting one never opened a generation");
    assert!(generations.iter().all(|generation| generation.is_cancelled()));
    assert!(generations.iter().all(|generation| generation.close_count() == 1));
    rig.finish();
}

fn standalone(name: usize, events: &Sender<Event>, on_interrupt: OnInterrupt) -> Arc<Probe> {
    Arc::new(Probe {
        name,
        events: events.clone(),
        slot: Mutex::new(None),
        on_interrupt,
        running: Arc::new(AtomicUsize::new(0)),
        most: Arc::new(AtomicUsize::new(0)),
        from_start: false,
    })
}

#[test]
fn a_stop_with_nothing_in_flight_stops_admission_at_once() {
    let fixture = Fixture::start();
    let (events, seen) = channel();
    fixture
        .server
        .admission()
        .offer(standalone(0, &events, OnInterrupt::Release));
    fixture
        .server
        .admission()
        .offer(standalone(1, &events, OnInterrupt::Release));
    assert_eq!(seen.recv_timeout(HANG_GUARD).expect("an event"), Event::Started(0));
    fixture.stop();
    let mut after = [
        seen.recv_timeout(HANG_GUARD).expect("an event"),
        seen.recv_timeout(HANG_GUARD).expect("an event"),
    ];
    after.sort_by_key(|event| matches!(event, Event::Refused(..)));
    assert_eq!(
        after,
        [
            Event::Interrupted(0, Interruption::Shutdown),
            Event::Refused(1, ApiError::shutting_down())
        ]
    );
    fixture.finish_after_stop();
}

/// Stops a server whose drain limit is `drain` with two entries in admission (one runs, one waits) and checks that
/// admission stopped once and the stop was reported once.
fn stop_with_drain_limit(drain: Duration) {
    let fixture = Fixture::with(
        Limits {
            drain_limit: drain,
            ..small_limits()
        },
        Some(2),
    );
    let (events, seen) = channel();
    fixture
        .server
        .admission()
        .offer(standalone(0, &events, OnInterrupt::Release));
    fixture
        .server
        .admission()
        .offer(standalone(1, &events, OnInterrupt::Release));
    assert_eq!(seen.recv_timeout(HANG_GUARD).expect("an event"), Event::Started(0));
    let reports = Arc::new(AtomicUsize::new(0));
    let (tx, stopped) = channel();
    let counted = reports.clone();
    fixture.server.stop(once(move || {
        counted.fetch_add(1, Ordering::AcqRel);
        tx.send(()).unwrap_or_default();
    }));
    let mut after = [
        seen.recv_timeout(HANG_GUARD).expect("an event"),
        seen.recv_timeout(HANG_GUARD).expect("an event"),
    ];
    after.sort_by_key(|event| matches!(event, Event::Refused(..)));
    assert_eq!(after[0], Event::Interrupted(0, Interruption::Shutdown));
    assert_eq!(after[1], Event::Refused(1, ApiError::shutting_down()));
    stopped.recv_timeout(HANG_GUARD).expect("the server stopped");
    // The lattice closes after every frame published before ran: what was reported or told twice has been by then.
    fixture.finish_after_stop();
    assert_eq!(
        reports.load(Ordering::Acquire),
        1,
        "the stop was reported more than once"
    );
    let told: Vec<_> = seen.try_iter().collect();
    assert!(told.is_empty(), "an entry was told twice: {told:?}");
}

#[test]
fn a_drain_limit_that_has_passed_when_the_transport_stops_still_stops_admission_once() {
    stop_with_drain_limit(Duration::ZERO);
}

#[test]
fn a_drain_limit_that_is_short_races_the_transport_and_stops_admission_once() {
    stop_with_drain_limit(Duration::from_millis(1));
}

#[test]
fn a_drain_limit_too_long_to_count_leaves_the_stop_to_the_transport() {
    stop_with_drain_limit(Duration::MAX);
}

#[test]
fn a_generation_that_never_ends_does_not_keep_the_server_from_stopping() {
    let limit = Duration::from_millis(200);
    let fixture = Fixture::with(
        Limits {
            shutdown_limit: limit,
            ..small_limits()
        },
        Some(2),
    );
    let (events, seen) = channel();
    let stubborn = standalone(0, &events, OnInterrupt::ReleaseWhenAbandoned);
    fixture.server.admission().offer(stubborn);
    assert_eq!(seen.recv_timeout(HANG_GUARD).expect("an event"), Event::Started(0));
    // `stop` returns once the server reported the stop: it does, whatever the generation does.
    fixture.stop();
    assert_eq!(
        seen.recv_timeout(HANG_GUARD).expect("an event"),
        Event::Interrupted(0, Interruption::Shutdown)
    );
    assert_eq!(
        seen.recv_timeout(HANG_GUARD).expect("an event"),
        Event::Interrupted(0, Interruption::Abandoned)
    );
    fixture.finish_after_stop();
}

#[test]
fn at_the_shutdown_limit_a_running_entry_is_told_to_give_up_once() {
    let rig = Rig::new(
        1,
        Limits {
            shutdown_limit: LIMIT,
            ..limits(4)
        },
    );
    let stubborn = rig.probe_with(0, OnInterrupt::ReleaseWhenAbandoned);
    rig.admission.offer(stubborn);
    assert_eq!(rig.next(), Event::Started(0));
    let (done, stopped) = channel();
    rig.admission.stop(move || done.send(()).unwrap_or_default());
    assert_eq!(rig.next(), Event::Interrupted(0, Interruption::Shutdown));
    rig.until_armed_for(LIMIT);
    rig.clock.advance(LIMIT);
    assert_eq!(rig.next(), Event::Interrupted(0, Interruption::Abandoned));
    stopped.recv_timeout(HANG_GUARD).expect("the stop completed");
    rig.until_counts(0, 0);
    rig.finish();
}

#[test]
fn a_shutdown_limit_too_long_to_count_is_never_reached() {
    let rig = Rig::new(
        1,
        Limits {
            shutdown_limit: Duration::MAX,
            ..limits(4)
        },
    );
    let first = rig.probe_with(0, OnInterrupt::Ignore);
    rig.admission.offer(first.clone());
    assert_eq!(rig.next(), Event::Started(0));
    let (done, stopped) = channel();
    rig.admission.stop(move || done.send(()).unwrap_or_default());
    assert_eq!(rig.next(), Event::Interrupted(0, Interruption::Shutdown));
    assert_eq!(
        stopped.try_recv(),
        Err(TryRecvError::Empty),
        "the stop completed with a slot held"
    );
    first.finish();
    stopped
        .recv_timeout(HANG_GUARD)
        .expect("the stop completed once the slot was free");
    rig.finish();
}

#[test]
fn a_second_stop_while_the_first_waits_completes_with_it() {
    let rig = Rig::new(
        1,
        Limits {
            shutdown_limit: LIMIT,
            ..limits(4)
        },
    );
    rig.admission
        .offer(rig.probe_with(0, OnInterrupt::ReleaseWhenAbandoned));
    assert_eq!(rig.next(), Event::Started(0));
    let (first_done, first) = channel();
    let (second_done, second) = channel();
    rig.admission.stop(move || first_done.send(()).unwrap_or_default());
    rig.admission.stop(move || second_done.send(()).unwrap_or_default());
    assert_eq!(rig.next(), Event::Interrupted(0, Interruption::Shutdown));
    rig.until_armed_for(LIMIT);
    rig.clock.advance(LIMIT);
    first.recv_timeout(HANG_GUARD).expect("the first stop completed");
    second.recv_timeout(HANG_GUARD).expect("the second stop completed");
    assert_eq!(rig.next(), Event::Interrupted(0, Interruption::Abandoned));
    rig.finish();
}

#[test]
fn a_timer_armed_for_a_request_that_ended_fires_without_effect() {
    let rig = Rig::new(
        1,
        Limits {
            request_limit: LIMIT,
            ..limits(4)
        },
    );
    let (first, _) = rig.offer(0);
    assert_eq!(rig.next(), Event::Started(0));
    rig.until_armed_for(LIMIT);
    first.finish();
    rig.until_counts(0, 0);
    // The timer for the first request fires here and finds nothing due: no event follows, which the end of the test
    // checks.
    rig.clock.advance(LIMIT);
    let (second, _) = rig.offer(1);
    assert_eq!(rig.next(), Event::Started(1));
    second.finish();
    rig.finish();
}

/// An entry that reports and gives its slot back at once.
struct Quick {
    name: usize,
    events: Sender<Event>,
}

impl Entry for Quick {
    fn start(&self, slot: Slot) {
        self.events.send(Event::Started(self.name)).ok();
        slot.release();
    }

    fn refuse(&self, error: ApiError) {
        self.events.send(Event::Refused(self.name, error)).ok();
    }

    fn interrupt(&self, _why: Interruption) {}
}

/// Offers `threads * each` quick entries from as many threads, calling `meanwhile` once part of them is in.
fn flood(rig: &Rig, threads: usize, each: usize, meanwhile: impl FnOnce()) -> Vec<Event> {
    let offered = Arc::new(AtomicUsize::new(0));
    let posters: Vec<_> = (0..threads)
        .map(|thread| {
            let admission = rig.admission.clone();
            let events = rig.sender.clone();
            let offered = offered.clone();
            spawn(move || {
                for index in 0..each {
                    let name = thread * each + index;
                    admission.offer(Arc::new(Quick {
                        name,
                        events: events.clone(),
                    }));
                    offered.fetch_add(1, Ordering::AcqRel);
                }
            })
        })
        .collect();
    let began = Instant::now();
    while offered.load(Ordering::Acquire) < threads * each / 4 {
        assert!(
            began.elapsed() < HANG_GUARD,
            "a quarter of the entries were never offered"
        );
        yield_now();
    }
    meanwhile();
    for poster in posters {
        poster.join().expect("a poster");
    }
    let mut events: Vec<_> = (0..threads * each).map(|_| rig.next()).collect();
    events.sort_by_key(|event| match event {
        Event::Started(name) | Event::Refused(name, _) | Event::Interrupted(name, _) => *name,
    });
    events
}

#[test]
fn no_wake_up_is_lost_when_many_threads_offer_and_slots_free_at_once() {
    let rig = Rig::new(2, limits(1000));
    let events = flood(&rig, 4, 200, || {});
    assert!(events.iter().all(|event| matches!(event, Event::Started(_))));
    rig.until_counts(0, 0);
    rig.finish();
}

#[test]
fn a_stop_racing_concurrent_offers_gives_every_entry_a_start_or_a_refusal() {
    let rig = Rig::new(2, limits(1000));
    let (done, stopped) = channel();
    let admission = rig.admission.clone();
    let events = flood(&rig, 4, 200, move || {
        admission.stop(move || done.send(()).unwrap_or_default());
    });
    for (index, event) in events.iter().enumerate() {
        match event {
            Event::Started(name) => assert_eq!(*name, index),
            Event::Refused(name, error) => {
                assert_eq!(*name, index);
                assert_eq!(*error, ApiError::shutting_down());
            }
            Event::Interrupted(..) => panic!("quick entries are never interrupted"),
        }
    }
    stopped.recv_timeout(HANG_GUARD).expect("the stop completed");
    rig.finish();
}

#[test]
fn a_request_that_arrived_earlier_counts_its_limit_from_its_arrival() {
    let rig = Rig::new(
        1,
        Limits {
            request_limit: Duration::from_secs(6),
            ..limits(4)
        },
    );
    rig.offer(0);
    assert_eq!(rig.next(), Event::Started(0));
    rig.until_armed_for(Duration::from_secs(6));
    let arrived = rig
        .clock
        .at_offset(Duration::ZERO)
        .checked_sub(Duration::from_secs(5))
        .expect("an earlier instant");
    rig.admission.offer_at(rig.probe_with(1, OnInterrupt::Release), arrived);
    // The second request's limit ends one second from now, 6 s after its arrival; counted from the offer it would end
    // 6 s from now and the clock moving one second would refuse nobody.
    rig.until_counts(1, 1);
    rig.until_armed_for(Duration::from_secs(1));
    rig.clock.advance(Duration::from_secs(1));
    assert_eq!(rig.next(), Event::Refused(1, ApiError::timeout()));
    rig.finish();
}

#[test]
fn a_running_entry_may_count_its_limit_from_its_start() {
    let rig = Rig::new(
        1,
        Limits {
            request_limit: Duration::from_secs(3),
            ..limits(4)
        },
    );
    let arrived = rig
        .clock
        .at_offset(Duration::ZERO)
        .checked_sub(Duration::from_secs(2))
        .expect("an earlier instant");
    let streaming = Arc::new(Probe {
        name: 0,
        events: rig.sender.clone(),
        slot: Mutex::new(None),
        on_interrupt: OnInterrupt::Release,
        running: rig.running.clone(),
        most: rig.most.clone(),
        from_start: true,
    });
    rig.admission.offer_at(streaming, arrived);
    assert_eq!(rig.next(), Event::Started(0));
    // Counted from the start, not from the arrival (which would be one second from now).
    rig.until_armed_for(Duration::from_secs(3));
    rig.clock.advance(Duration::from_secs(3));
    assert_eq!(rig.next(), Event::Interrupted(0, Interruption::TimedOut));
    rig.finish();
}
