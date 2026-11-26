//! The pipeline's scheduling over fake stages: the order of a pass's calls, the sequences of a batch as separate
//! frames, the end held behind the device's last marker, and a failure that ends behind a fence. No device: the fake
//! stages record what they are called with, and what the device would complete is released by the test.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::disallowed_methods,
    clippy::disallowed_types,
    reason = "a test: it waits for its own hand-off, logs under a mutex and stops on a failed unwrap"
)]

mod checkpoint_support;

use std::fmt::{Debug, Formatter, Result as FmtResult};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use std::time::Instant;

use checkpoint_support::World;
use euhedral_core::frames::{Frame, FrameError, FrameHeader};
use euhedral_hashing::hasher_api;
use euhedral_inference_engine::Handoff;
use euhedral_inference_quantization::forward::ForwardError;
use euhedral_inference_quantization::forward::pipeline::{Attempt, LaneRef, Scheduler, Stages};

const LONGEST: Duration = Duration::from_secs(8);

#[derive(Clone, Debug, PartialEq, Eq)]
enum Event {
    Head {
        first: u32,
        count: u32,
        mark: u64,
    },
    Sequence {
        index: u32,
        after: u64,
        mark: u64,
    },
    Tail {
        first: u32,
        count: u32,
        waited: Vec<u64>,
        mark: Option<u64>,
        /// Whether the gate of the held sequence was open when the tail ran.
        open: bool,
    },
    /// A slot lane's fence mark claimed.
    FenceMark {
        lane: u32,
    },
    Fence,
    Retire {
        slots: usize,
    },
}

/// A device that is a log: every call is recorded and answered with the next mark. What `after` is given waits in a
/// cell for the test to release, as the lanes' completions would emit it when the marker is reached. A call that
/// answers Busy has its retry parked until the test completes it, as a lane's completion would.
struct Fake {
    log: Mutex<Vec<Event>>,
    marks: AtomicU64,
    slots: u32,
    /// The sequence that fails, if any.
    failing: Option<u32>,
    /// Calls of `head` that answer Busy before one is taken, and the calls made.
    busy_heads: AtomicU32,
    head_calls: AtomicU32,
    /// A sequence that answers Busy until `gate` is opened.
    held: Option<u32>,
    gate: AtomicBool,
    /// Sequence 0 answers Busy until sequence 1 has been called: they must be separate frames.
    overlap: bool,
    sequence_one_ran: AtomicBool,
    /// The fence answers Busy once on slot lane 1.
    fence_busy: AtomicBool,
    /// `after` refuses its frame.
    refuse_after: bool,
    parked: Mutex<Vec<(LaneRef, Box<dyn Frame>)>>,
    ended: Handoff<(u64, Box<dyn Frame>)>,
}

impl Default for Fake {
    fn default() -> Self {
        Fake {
            log: Mutex::new(Vec::new()),
            marks: AtomicU64::new(0),
            slots: 2,
            failing: None,
            busy_heads: AtomicU32::new(0),
            head_calls: AtomicU32::new(0),
            held: None,
            gate: AtomicBool::new(false),
            overlap: false,
            sequence_one_ran: AtomicBool::new(false),
            fence_busy: AtomicBool::new(false),
            refuse_after: false,
            parked: Mutex::new(Vec::new()),
            ended: Handoff::empty(),
        }
    }
}

impl Fake {
    fn record(&self, event: Event) {
        self.log.lock().unwrap().push(event);
    }

    fn mark(&self) -> u64 {
        self.marks.fetch_add(1, Ordering::SeqCst) + 1
    }

    fn events(&self) -> Vec<Event> {
        self.log.lock().unwrap().clone()
    }

    fn parked(&self) -> usize {
        self.parked.lock().unwrap().len()
    }

    /// The device completes a marker: every parked retry is published.
    fn complete(&self, world: &World) -> usize {
        let frames: Vec<_> = self.parked.lock().unwrap().drain(..).collect();
        let count = frames.len();
        for (_, frame) in frames {
            assert!(world.host.offer(frame).is_ok(), "the lake refused a retry");
        }
        count
    }
}

impl Stages for Fake {
    type Main = ();
    type Slot = u32;
    type Mark = u64;
    type Pass = ();

    fn head(&self, (): &(), (): &mut (), first: u32, count: u32) -> Result<Attempt<u64>, ForwardError> {
        self.head_calls.fetch_add(1, Ordering::SeqCst);
        if self
            .busy_heads
            .try_update(Ordering::SeqCst, Ordering::SeqCst, |left| left.checked_sub(1))
            .is_ok()
        {
            return Ok(Attempt::Busy);
        }
        let mark = self.mark();
        self.record(Event::Head { first, count, mark });
        Ok(Attempt::Ready(mark))
    }

    fn sequence(&self, (): &(), slot: &mut u32, index: u32, after: u64) -> Result<Attempt<u64>, ForwardError> {
        assert_eq!(*slot, index, "a sequence runs on its own slot");
        if self.overlap {
            if index == 1 {
                self.sequence_one_ran.store(true, Ordering::SeqCst);
            } else if !self.sequence_one_ran.load(Ordering::SeqCst) {
                return Ok(Attempt::Busy);
            }
        }
        if self.held == Some(index) && !self.gate.load(Ordering::SeqCst) {
            return Ok(Attempt::Busy);
        }
        let mark = self.mark();
        self.record(Event::Sequence { index, after, mark });
        if self.failing == Some(index) {
            return Err(ForwardError::Invalid(format!("sequence {index} failed")));
        }
        Ok(Attempt::Ready(mark))
    }

    fn tail(
        &self,
        (): &(),
        (): &mut (),
        first: u32,
        count: u32,
        last: bool,
        marks: &[u64],
    ) -> Result<Attempt<Option<u64>>, ForwardError> {
        let mut waited = marks.to_vec();
        waited.sort_unstable();
        let mark = last.then(|| self.mark());
        self.record(Event::Tail {
            first,
            count,
            waited,
            mark,
            open: self.gate.load(Ordering::SeqCst),
        });
        Ok(Attempt::Ready(mark))
    }

    fn fence(&self, (): &mut (), marks: &mut Vec<u64>) -> Result<Result<u64, LaneRef>, ForwardError> {
        while (marks.len() as u32) < self.slots {
            let lane = marks.len() as u32;
            if lane == 1 && self.fence_busy.swap(false, Ordering::SeqCst) {
                return Ok(Err(LaneRef::Slot(lane)));
            }
            self.record(Event::FenceMark { lane });
            marks.push(self.mark());
        }
        self.record(Event::Fence);
        Ok(Ok(self.mark()))
    }

    fn retry(&self, lane: LaneRef, then: Box<dyn Frame>) -> Result<(), Box<dyn Frame>> {
        self.parked.lock().unwrap().push((lane, then));
        Ok(())
    }

    fn after(&self, mark: u64, then: Box<dyn Frame>) -> Result<(), ForwardError> {
        if self.refuse_after {
            drop(then);
            return Err(ForwardError::Invalid("the completions refused".to_owned()));
        }
        self.ended.send((mark, then));
        Ok(())
    }

    fn retire(&self, slots: &mut [u32]) {
        self.record(Event::Retire { slots: slots.len() });
    }
}

type Outcome = Handoff<Result<(), ForwardError>>;

/// A frame that runs a closure: offered to the lake, it runs after every frame offered before it.
struct Probe {
    header: FrameHeader,
    work: Option<Box<dyn FnOnce() + Send>>,
}

impl Debug for Probe {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        formatter.debug_struct("Probe").finish_non_exhaustive()
    }
}

impl Frame for Probe {
    fn header(&self) -> &FrameHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut FrameHeader {
        &mut self.header
    }

    fn execute(&mut self) -> Result<(), FrameError> {
        if let Some(work) = self.work.take() {
            work();
        }
        Ok(())
    }
}

/// Waits until the lake has run everything offered to it so far, and everything those frames offered. The lake of a
/// test has one sink, which takes frames in the order they were offered, so a probe offered now runs after them.
fn drained(world: &World) {
    let ran = Handoff::new();
    let sender = Arc::clone(&ran);
    let probe = Probe {
        header: FrameHeader::new(hasher_api::mix(0x0fe5_7a11 | 1 << 63)),
        work: Some(Box::new(move || sender.send(()))),
    };
    assert!(world.host.offer(Box::new(probe)).is_ok(), "the lake refused a probe");
    ran.wait(LONGEST).expect("the probe never ran");
}

fn pipeline(world: &World, fake: Fake, slots: u32) -> Scheduler<Fake> {
    let fake = Fake { slots, ..fake };
    Scheduler::new(fake, Arc::clone(&world.host), (), (0..slots).collect())
}

/// Runs the device until the pass has ended: completes every parked retry as it appears and reaches the pass's end
/// marker. Bounded by the hang deadline.
fn settle(world: &World, scheduler: &Scheduler<Fake>, outcome: &Outcome) {
    let begun = Instant::now();
    while !outcome.is_set() {
        assert!(begun.elapsed() < LONGEST, "the pass never ended");
        scheduler.stages().complete(world);
        if let Some((_, frame)) = scheduler.stages().ended.take() {
            assert!(world.host.offer(frame).is_ok());
        }
    }
}

fn start(scheduler: &Scheduler<Fake>, sequences: u32, batch: u32) -> (Arc<Outcome>, Result<(), ForwardError>) {
    let outcome = Handoff::new();
    let sender = Arc::clone(&outcome);
    let started = scheduler.run((), sequences, batch, Box::new(move |result| sender.send(result)));
    (outcome, started)
}

/// What the device does when the last marker is reached: the frame the pass handed to `after` runs on the lake.
fn reach(world: &World, scheduler: &Scheduler<Fake>) -> u64 {
    let (mark, frame) = scheduler
        .stages()
        .ended
        .wait(LONGEST)
        .expect("the pass never asked to be ended behind a marker");
    if world.host.offer(frame).is_err() {
        panic!("the lake refused the ending frame");
    }
    mark
}

fn ended(outcome: &Outcome) -> Result<(), ForwardError> {
    outcome.wait(LONGEST).expect("the pass never ended")
}

#[test]
fn batches_run_in_order_with_each_sequence_between_its_head_and_tail() {
    let world = World::start();
    let scheduler = pipeline(&world, Fake::default(), 2);
    let (outcome, started) = start(&scheduler, 5, 2);
    started.unwrap();
    let ending = reach(&world, &scheduler);
    ended(&outcome).unwrap();

    let events = scheduler.stages().events();
    let mut at = 0;
    for (first, count) in [(0u32, 2u32), (2, 2), (4, 1)] {
        let Event::Head {
            first: head_first,
            count: head_count,
            mark: head_mark,
        } = events[at]
        else {
            panic!("batch {first} starts with its head, saw {:?}", events[at]);
        };
        assert_eq!(
            (head_first, head_count),
            (first, count),
            "a batch's head follows the previous batch's tail"
        );
        at += 1;
        let mut indices = Vec::new();
        let mut marks = Vec::new();
        for event in &events[at..at + count as usize] {
            let Event::Sequence { index, after, mark } = *event else {
                panic!("batch {first}: {count} sequences follow the head, saw {event:?}");
            };
            assert_eq!(after, head_mark, "a sequence waits for its own batch's head");
            indices.push(index);
            marks.push(mark);
        }
        indices.sort_unstable();
        marks.sort_unstable();
        assert_eq!(
            indices,
            (0..count).collect::<Vec<_>>(),
            "each sequence of the batch runs once"
        );
        at += count as usize;
        let Event::Tail {
            first: tail_first,
            count: tail_count,
            ref waited,
            mark,
            ..
        } = events[at]
        else {
            panic!("batch {first}: the tail follows its sequences, saw {:?}", events[at]);
        };
        assert_eq!((tail_first, tail_count), (first, count));
        assert_eq!(*waited, marks, "the tail waits for exactly its batch's sequences");
        assert_eq!(
            mark.is_some(),
            first + count == 5,
            "only the final batch's tail gives the mark the pass ends behind"
        );
        if let Some(mark) = mark {
            assert_eq!(mark, ending, "the pass ends behind the last tail's marker");
        }
        at += 1;
    }
    assert_eq!(
        events[at],
        Event::Retire { slots: 2 },
        "the pass retires its slots once, at the end"
    );
    assert_eq!(events.len(), at + 1);
    drop(world);
}

#[test]
fn the_end_waits_for_the_device_and_the_state_comes_back() {
    let world = World::start();
    let scheduler = pipeline(&world, Fake::default(), 2);
    let (outcome, started) = start(&scheduler, 3, 2);
    started.unwrap();
    let (mark, frame) = scheduler
        .stages()
        .ended
        .wait(LONGEST)
        .expect("the pass never reached its end");
    assert!(
        !outcome.is_set(),
        "the pass is not over before the device reached the marker"
    );
    assert!(
        scheduler.run((), 1, 1, Box::new(|_| {})).is_err(),
        "a pass runs while another is running"
    );
    assert!(world.host.offer(frame).is_ok());
    ended(&outcome).unwrap();
    let events = scheduler.stages().events();
    assert!(
        events
            .iter()
            .any(|event| matches!(event, Event::Tail { mark: Some(tail), .. } if *tail == mark)),
        "the end waits for the marker after the last tail"
    );
    let Ok((fake, (), slots)) = scheduler.into_parts() else {
        panic!("the lanes' state did not come back after the pass ended");
    };
    assert_eq!(slots, vec![0, 1]);
    assert_eq!(fake.events().last(), Some(&Event::Retire { slots: 2 }));
    drop(world);
}

#[test]
fn the_sequences_of_a_batch_are_separate_frames() {
    let world = World::start();
    let fake = Fake {
        overlap: true,
        ..Fake::default()
    };
    let scheduler = pipeline(&world, fake, 2);
    let (outcome, started) = start(&scheduler, 2, 2);
    started.unwrap();
    // A pass that ran its sequences one after another would never get past sequence 0, which answers Busy until
    // sequence 1 has run.
    settle(&world, &scheduler, &outcome);
    ended(&outcome).unwrap();
    let events = scheduler.stages().events();
    let at = |wanted: u32| {
        events
            .iter()
            .position(|event| matches!(event, Event::Sequence { index, .. } if *index == wanted))
            .expect("the sequence ran")
    };
    assert!(at(1) < at(0), "sequence 0 ran only after sequence 1: {events:?}");
    drop(world);
}

#[test]
fn a_busy_lane_is_asked_again_only_after_a_completion_and_nothing_is_lost() {
    let world = World::start();
    let fake = Fake {
        busy_heads: AtomicU32::new(3),
        ..Fake::default()
    };
    let scheduler = pipeline(&world, fake, 1);
    let (outcome, started) = start(&scheduler, 2, 1);
    started.unwrap();
    let begun = Instant::now();
    for completed in 0..3u32 {
        while scheduler.stages().parked() == 0 {
            assert!(
                begun.elapsed() < LONGEST,
                "a Busy head was not parked behind a completion (answers so far: {})",
                scheduler.stages().head_calls.load(Ordering::SeqCst)
            );
        }
        // The retry rests in the lane's completions: the head was asked once more only per completion.
        assert_eq!(
            scheduler.stages().head_calls.load(Ordering::SeqCst),
            completed + 1,
            "the head was asked again before a completion"
        );
        assert_eq!(scheduler.stages().parked.lock().unwrap()[0].0, LaneRef::Main);
        assert_eq!(scheduler.stages().complete(&world), 1);
    }
    settle(&world, &scheduler, &outcome);
    ended(&outcome).unwrap();
    let heads = scheduler
        .stages()
        .events()
        .into_iter()
        .filter(|event| matches!(event, Event::Head { .. }))
        .count();
    assert_eq!(heads, 2, "each batch's head is taken once after the lane's refusals");
    drop(world);
}

/// A sequence held behind a gate holds the batch's tail: it runs after the gate opened, with every sequence's mark.
#[test]
fn the_tail_of_a_batch_waits_for_its_slowest_sequence() {
    let world = World::with_sinks(1);
    let fake = Fake {
        held: Some(0),
        ..Fake::default()
    };
    let scheduler = pipeline(&world, fake, 2);
    let (outcome, started) = start(&scheduler, 2, 2);
    started.unwrap();
    let begun = Instant::now();
    // Sequence 1 runs and arrives; sequence 0 answers Busy and rests behind its lane's completion.
    loop {
        let ran = scheduler
            .stages()
            .events()
            .iter()
            .any(|event| matches!(event, Event::Sequence { index: 1, .. }));
        if ran && scheduler.stages().parked() == 1 {
            break;
        }
        assert!(
            begun.elapsed() < LONGEST,
            "sequence 1 never ran or sequence 0 never parked"
        );
    }
    // Sequence 1 arrived after its event: the lake runs what that offered (a tail, were one wrongly published) before
    // anything offered from here, one probe for the arrival and one for the frame it offered.
    drained(&world);
    drained(&world);
    assert!(!outcome.is_set(), "the batch ended before its held sequence ran");
    assert!(
        !scheduler
            .stages()
            .events()
            .iter()
            .any(|event| matches!(event, Event::Tail { .. } | Event::Fence)),
        "the tail or a fence ran before the held sequence was released"
    );
    scheduler.stages().gate.store(true, Ordering::SeqCst);
    settle(&world, &scheduler, &outcome);
    ended(&outcome).expect("the batch ends once its held sequence ran");
    let tails: Vec<_> = scheduler
        .stages()
        .events()
        .into_iter()
        .filter_map(|event| match event {
            Event::Tail { open, waited, .. } => Some((open, waited.len())),
            _ => None,
        })
        .collect();
    assert_eq!(
        tails,
        vec![(true, 2)],
        "the tail ran before the held sequence was released"
    );
    drop(world);
}

/// A failed pass whose fence answers Busy on a slot lane claims one mark on each lane, not one per attempt.
#[test]
fn a_fence_that_is_refused_keeps_the_marks_it_made() {
    let world = World::start();
    let fake = Fake {
        failing: Some(1),
        fence_busy: AtomicBool::new(true),
        ..Fake::default()
    };
    let scheduler = pipeline(&world, fake, 2);
    let (outcome, started) = start(&scheduler, 2, 2);
    started.unwrap();
    let begun = Instant::now();
    while scheduler.stages().parked() == 0 {
        assert!(begun.elapsed() < LONGEST, "the refused fence was not parked");
    }
    assert_eq!(scheduler.stages().parked.lock().unwrap()[0].0, LaneRef::Slot(1));
    settle(&world, &scheduler, &outcome);
    assert!(ended(&outcome).is_err());
    let marks: Vec<u32> = scheduler
        .stages()
        .events()
        .into_iter()
        .filter_map(|event| match event {
            Event::FenceMark { lane } => Some(lane),
            _ => None,
        })
        .collect();
    assert_eq!(marks, vec![0, 1], "one fence mark per lane");
    drop(world);
}

/// The caller's buffers are not handed back while the device may still use them: a pass whose frames were dropped
/// (the lake closed), and a failed pass that could not be fenced, tell the caller so.
#[test]
fn a_pass_that_cannot_be_fenced_says_the_device_may_still_use_the_buffers() {
    let world = World::start();
    let scheduler = pipeline(&world, Fake::default(), 2);
    let (outcome, started) = start(&scheduler, 2, 2);
    started.unwrap();
    let (_, frame) = scheduler
        .stages()
        .ended
        .wait(LONGEST)
        .expect("the pass never reached its end");
    drop(frame);
    let error = ended(&outcome).expect_err("a pass that lost its end says so");
    assert!(matches!(error, ForwardError::Unfenced(_)), "{error}");

    let refusing = Fake {
        failing: Some(0),
        refuse_after: true,
        ..Fake::default()
    };
    let scheduler = pipeline(&world, refusing, 2);
    let (outcome, started) = start(&scheduler, 2, 2);
    started.unwrap();
    let error = ended(&outcome).expect_err("the failed pass");
    assert!(matches!(error, ForwardError::Unfenced(_)), "{error}");
    drop(world);
}

#[test]
fn a_failed_sequence_ends_the_pass_behind_a_fence_with_its_error() {
    let world = World::start();
    let fake = Fake {
        failing: Some(1),
        ..Fake::default()
    };
    let scheduler = pipeline(&world, fake, 2);
    let (outcome, started) = start(&scheduler, 4, 2);
    started.unwrap();
    // The error reaches the caller only once the device is through with what the pass queued.
    let (mark, frame) = scheduler
        .stages()
        .ended
        .wait(LONGEST)
        .expect("the failed pass never asked for a fence");
    assert!(
        !outcome.is_set(),
        "the error was delivered before the device reached the fence"
    );
    assert!(world.host.offer(frame).is_ok());
    let error = ended(&outcome).expect_err("the pass reports the failure");
    assert!(error.to_string().contains("sequence 1 failed"), "{error}");
    let events = scheduler.stages().events();
    assert!(events.contains(&Event::Fence));
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, Event::Tail { .. } | Event::Retire { .. })),
        "a failed batch runs no tail and publishes nothing: {events:?}"
    );
    assert!(
        !events.iter().any(|event| matches!(event, Event::Head { first: 2, .. })),
        "no batch starts after the failure: {events:?}"
    );
    assert!(
        events.iter().filter(|event| matches!(event, Event::Fence)).count() == 1 && mark > 0,
        "one fence ends the failed pass"
    );
    // The lanes' state came back: another pass runs.
    let (second, started) = start(&scheduler, 1, 1);
    started.unwrap();
    let (_, frame) = scheduler
        .stages()
        .ended
        .wait(LONGEST)
        .expect("the next pass never ended");
    assert!(world.host.offer(frame).is_ok());
    // Sequence 1 does not exist in a pass of one sequence: it succeeds.
    ended(&second).unwrap();
    drop(world);
}
