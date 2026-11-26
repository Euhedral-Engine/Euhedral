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

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use checkpoint_support::World;
use euhedral_core::frames::Frame;
use euhedral_inference_engine::Handoff;
use euhedral_inference_quantization::forward::ForwardError;
use euhedral_inference_quantization::forward::pipeline::{Attempt, Scheduler, Stages};

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
    },
    Fence,
    Retire {
        slots: usize,
    },
}

/// A device that is a log: every call is recorded and answered with the next mark. What `after` is given waits in a
/// cell for the test to release, as the lanes' completions would emit it when the marker is reached.
struct Fake {
    log: Mutex<Vec<Event>>,
    marks: AtomicU64,
    /// The sequence that fails, if any.
    failing: Option<u32>,
    /// Calls of `head` that answer Busy before one is taken.
    busy_heads: AtomicU32,
    /// Sequence 0 answers Busy until sequence 1 has been called: they must be separate frames.
    overlap: bool,
    sequence_one_ran: AtomicBool,
    spins: AtomicU32,
    ended: Handoff<(u64, Box<dyn Frame>)>,
}

impl Default for Fake {
    fn default() -> Self {
        Fake {
            log: Mutex::new(Vec::new()),
            marks: AtomicU64::new(0),
            failing: None,
            busy_heads: AtomicU32::new(0),
            overlap: false,
            sequence_one_ran: AtomicBool::new(false),
            spins: AtomicU32::new(0),
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
}

impl Stages for Fake {
    type Main = ();
    type Slot = u32;
    type Mark = u64;
    type Pass = ();

    fn head(&self, (): &(), (): &mut (), first: u32, count: u32) -> Result<Attempt<u64>, ForwardError> {
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
                if self.spins.fetch_add(1, Ordering::SeqCst) > 200_000 {
                    return Err(ForwardError::Invalid(
                        "sequence 1 never ran while sequence 0 waited for it".to_owned(),
                    ));
                }
                return Ok(Attempt::Busy);
            }
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
        });
        Ok(Attempt::Ready(mark))
    }

    fn fence(&self, (): &mut ()) -> Result<Attempt<u64>, ForwardError> {
        self.record(Event::Fence);
        Ok(Attempt::Ready(self.mark()))
    }

    fn after(&self, mark: u64, then: Box<dyn Frame>) -> Result<(), ForwardError> {
        self.ended.send((mark, then));
        Ok(())
    }

    fn retire(&self, slots: &mut [u32]) {
        self.record(Event::Retire { slots: slots.len() });
    }
}

type Outcome = Handoff<Result<(), ForwardError>>;

fn pipeline(world: &World, fake: Fake, slots: u32) -> Scheduler<Fake> {
    Scheduler::new(fake, Arc::clone(&world.host), (), (0..slots).collect())
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
    // A pass that ran its sequences one after another would fail here, on sequence 0 never seeing sequence 1.
    let (_, frame) = scheduler
        .stages()
        .ended
        .wait(LONGEST)
        .expect("the pass never reached its end");
    assert!(world.host.offer(frame).is_ok());
    ended(&outcome).unwrap();
    drop(world);
}

#[test]
fn a_busy_lane_is_asked_again_and_nothing_is_lost() {
    let world = World::start();
    let fake = Fake {
        busy_heads: AtomicU32::new(3),
        ..Fake::default()
    };
    let scheduler = pipeline(&world, fake, 1);
    let (outcome, started) = start(&scheduler, 2, 1);
    started.unwrap();
    reach(&world, &scheduler);
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
