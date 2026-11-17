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
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, channel};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use common::{Fixture, PATIENCE, lattice, small_limits};
use euhedral_core::control_plane::ControlPlaneLattice;
use euhedral_core::generics::LatticeSource;
use euhedral_core::ingest::{IngestSink, QueueIngestSink};
use euhedral_graph::Completions;
use euhedral_inference_api_protocol::ApiError;
use euhedral_inference_api_server::scripted::{Script, ScriptedBackend};
use euhedral_inference_api_server::{
    Admission, Backend, BackendError, EncodedPrompt, Entry, Generation, GenerationConfig, GenerationResult, Handle,
    Interruption, Limits, OutputSpec, Slot, TextListener, once,
};

/// How long a test waits to be sure that something does not happen.
const QUIET: Duration = Duration::from_millis(300);

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
}

/// An entry that reports and holds its slot until the test says so.
struct Probe {
    name: usize,
    events: Sender<Event>,
    slot: Mutex<Option<Slot>>,
    on_interrupt: OnInterrupt,
    running: Arc<AtomicUsize>,
    most: Arc<AtomicUsize>,
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

    fn interrupt(&self, why: Interruption) {
        self.events.send(Event::Interrupted(self.name, why)).ok();
        if self.on_interrupt == OnInterrupt::Release {
            self.finish();
        }
    }
}

struct Rig {
    lattice: Arc<ControlPlaneLattice>,
    backend: Arc<ScriptedBackend>,
    sink: Arc<QueueIngestSink>,
    timers: Arc<Completions>,
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
        let lattice = lattice(Some(2));
        let backend = ScriptedBackend::on(&lattice);
        backend.set_max_active(max_active);
        let sink = Arc::new(QueueIngestSink::new());
        let timers = Completions::without_lanes();
        lattice.add_upstream(sink.delegate()).expect("the sink is attached");
        lattice.add_upstream(timers.clone()).expect("the timers are attached");
        let admission = Admission::new(backend.clone(), &limits, sink.clone(), timers.clone());
        let (sender, events) = channel();
        Rig {
            lattice,
            backend,
            sink,
            timers,
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
        self.events.recv_timeout(PATIENCE).expect("an event")
    }

    fn assert_quiet(&self) {
        match self.events.recv_timeout(QUIET) {
            Err(RecvTimeoutError::Timeout) => {}
            other => panic!("nothing should have happened, got {other:?}"),
        }
    }

    fn finish(self) {
        self.timers.complete();
        self.sink.complete_gracefully();
        self.backend.release();
        self.lattice.close();
    }
}

#[test]
fn entries_start_in_the_order_they_were_offered_and_never_exceed_the_slots() {
    let rig = Rig::new(2, limits(8));
    let probes: Vec<_> = (0..6).map(|name| rig.offer(name).0).collect();
    assert_eq!(rig.next(), Event::Started(0));
    assert_eq!(rig.next(), Event::Started(1));
    rig.assert_quiet();
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
    for name in 0..3 {
        assert_eq!(rig.next(), Event::Started(name));
    }
    rig.finish();
}

#[test]
fn a_request_that_finds_the_queue_full_is_refused_with_the_capacity_error() {
    let rig = Rig::new(1, limits(1));
    let (first, _) = rig.offer(0);
    rig.offer(1);
    rig.offer(2);
    assert_eq!(rig.next(), Event::Started(0));
    let Event::Refused(2, error) = rig.next() else {
        panic!("the third request is refused");
    };
    assert_eq!(error, ApiError::at_capacity());
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
    assert_eq!(rig.next(), Event::Started(0));
    assert_eq!(rig.next(), Event::Refused(1, ApiError::at_capacity()));
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
            request_limit: Duration::from_millis(150),
            ..limits(4)
        },
    );
    let begun = Instant::now();
    let (first, _) = rig.offer(0);
    rig.offer(1);
    assert_eq!(rig.next(), Event::Started(0));
    // The running one is interrupted too, and the waiting one is refused: both limits are the same instant.
    let mut seen = [rig.next(), rig.next()];
    seen.sort_by_key(|event| matches!(event, Event::Refused(..)));
    assert_eq!(seen[0], Event::Interrupted(0, Interruption::TimedOut));
    assert_eq!(seen[1], Event::Refused(1, ApiError::timeout()));
    assert!(begun.elapsed() >= Duration::from_millis(150));
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
            request_limit: Duration::from_millis(100),
            ..limits(4)
        },
    );
    rig.offer(0);
    assert_eq!(rig.next(), Event::Started(0));
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
            queue_limit: Some(Duration::from_millis(100)),
            ..limits(4)
        },
    );
    rig.offer(0);
    rig.offer(1);
    assert_eq!(rig.next(), Event::Started(0));
    assert_eq!(rig.next(), Event::Refused(1, ApiError::timeout()));
    rig.assert_quiet();
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
    rig.assert_quiet();
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
    rig.assert_quiet();
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
    stopped.recv_timeout(PATIENCE).expect("the stop completed");
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
        stopped.recv_timeout(PATIENCE).expect("the stop completed");
    }
    rig.finish();
}

#[test]
fn stop_with_a_generation_that_never_ends_completes_at_the_shutdown_limit() {
    let rig = Rig::new(
        1,
        Limits {
            shutdown_limit: Duration::from_millis(200),
            ..limits(4)
        },
    );
    let stubborn = rig.probe_with(0, OnInterrupt::Ignore);
    rig.admission.offer(stubborn.clone());
    assert_eq!(rig.next(), Event::Started(0));
    let begun = Instant::now();
    let (done, stopped) = channel();
    rig.admission.stop(move || done.send(()).unwrap_or_default());
    assert_eq!(rig.next(), Event::Interrupted(0, Interruption::Shutdown));
    assert!(matches!(
        stopped.recv_timeout(Duration::from_millis(50)),
        Err(RecvTimeoutError::Timeout)
    ));
    stopped.recv_timeout(PATIENCE).expect("the stop completed at the limit");
    assert!(begun.elapsed() >= Duration::from_millis(200));
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
        begins.recv_timeout(PATIENCE).expect("a generation began");
    }
    assert_eq!(rig.admission.counts().waiting, 1);
    let (done, stopped) = channel();
    rig.admission.stop(move || done.send(()).unwrap_or_default());
    stopped.recv_timeout(PATIENCE).expect("the stop completed");
    for _ in 0..2 {
        let result = ends.recv_timeout(PATIENCE).expect("a generation ended");
        assert!(!result.expect("an ended generation").stop_token_reached);
    }
    let generations = rig.backend.generations();
    assert_eq!(generations.len(), 2, "the waiting one never opened a generation");
    assert!(generations.iter().all(|generation| generation.is_cancelled()));
    assert!(generations.iter().all(|generation| generation.close_count() == 1));
    rig.finish();
}

#[test]
fn the_server_stops_admission_with_its_transport() {
    let fixture = Fixture::start();
    let (events, seen) = channel();
    let busy = Arc::new(Probe {
        name: 0,
        events: events.clone(),
        slot: Mutex::new(None),
        on_interrupt: OnInterrupt::Release,
        running: Arc::new(AtomicUsize::new(0)),
        most: Arc::new(AtomicUsize::new(0)),
    });
    let waiting = Arc::new(Probe {
        name: 1,
        events,
        slot: Mutex::new(None),
        on_interrupt: OnInterrupt::Release,
        running: Arc::new(AtomicUsize::new(0)),
        most: Arc::new(AtomicUsize::new(0)),
    });
    fixture.server.admission().offer(busy);
    fixture.server.admission().offer(waiting);
    assert_eq!(seen.recv_timeout(PATIENCE).expect("an event"), Event::Started(0));
    fixture.stop();
    let mut after = vec![
        seen.recv_timeout(PATIENCE).expect("an event"),
        seen.recv_timeout(PATIENCE).expect("an event"),
    ];
    after.sort_by_key(|event| matches!(event, Event::Refused(..)));
    assert_eq!(
        after,
        vec![
            Event::Interrupted(0, Interruption::Shutdown),
            Event::Refused(1, ApiError::shutting_down())
        ]
    );
    fixture.finish_after_stop();
    drop(once(|| {}));
}
