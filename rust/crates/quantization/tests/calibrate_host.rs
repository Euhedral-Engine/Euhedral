//! The calibration's host core: the plan, the Hessians and the layer loop, over a fake of the device's calls.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::disallowed_methods,
    clippy::disallowed_types,
    reason = "a test: the fake records in a mutex and a failed unwrap is the failure"
)]

use std::fmt::Debug;
use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use euhedral_inference_engine::Handoff;
use euhedral_inference_quantization::calibrate::device::refuse_finish;
use euhedral_inference_quantization::calibrate::manifest::{Ledger, Manifest, differences, inspect};
use euhedral_inference_quantization::calibrate::plan::{resume_at, resume_header, resume_key, tokens_digest};
use euhedral_inference_quantization::calibrate::schedule::{Schedule, Settled};
use euhedral_inference_quantization::calibrate::{
    CalibrateError, HessianSet, Plan, Stages, Then, drive, embedding, round_order,
};

const HANG: Duration = Duration::from_secs(20);

#[derive(Clone, Debug, PartialEq, Eq)]
enum Event {
    Stage(usize),
    Layer(usize),
    Checkpoint(usize),
    Finish,
}

/// A fake device: records the calls, fails one layer on request and keeps the checkpoint a resume reads.
struct Fake {
    events: Mutex<Vec<Event>>,
    fail_layer: Option<usize>,
    saved: AtomicUsize,
}

impl Fake {
    fn new(fail_layer: Option<usize>, saved: usize) -> Arc<Fake> {
        Arc::new(Fake {
            events: Mutex::new(Vec::new()),
            fail_layer,
            saved: AtomicUsize::new(saved),
        })
    }

    fn log(&self, event: Event) {
        self.events.lock().unwrap().push(event);
    }

    fn events(&self) -> Vec<Event> {
        self.events.lock().unwrap().clone()
    }
}

impl Stages for Fake {
    type Staged = usize;

    fn begin(&self, _plan: &Plan, then: Then<usize>) {
        then(Ok(self.saved.load(Ordering::Acquire)));
    }

    fn stage(&self, layer: usize, then: Then<usize>) {
        self.log(Event::Stage(layer));
        then(Ok(layer));
    }

    fn layer(&self, layer: usize, staged: usize, then: Then<()>) {
        assert_eq!(layer, staged, "a layer runs over its own weights");
        self.log(Event::Layer(layer));
        if self.fail_layer == Some(layer) {
            then(Err(CalibrateError::Failed(format!("layer {layer} failed"))));
        } else {
            then(Ok(()));
        }
    }

    fn checkpoint(&self, done: usize, then: Then<()>) {
        self.log(Event::Checkpoint(done));
        self.saved.store(done, Ordering::Release);
        then(Ok(()));
    }

    fn finish(&self, then: Then<()>) {
        self.log(Event::Finish);
        self.saved.store(0, Ordering::Release);
        then(Ok(()));
    }
}

fn result_sink() -> (Then<()>, Receiver<Result<(), CalibrateError>>) {
    let (sender, receiver) = channel();
    let sender: Sender<_> = sender;
    (Box::new(move |result| sender.send(result).unwrap()), receiver)
}

fn run(fake: &Arc<Fake>, plan: Plan) -> Result<(), CalibrateError> {
    let (done, receiver) = result_sink();
    drive(Arc::clone(fake), plan, done);
    receiver.recv_timeout(HANG).expect("the calibration never ended")
}

#[test]
fn a_full_run_reads_ahead_checkpoints_every_eighth_layer_and_finishes() {
    let fake = Fake::new(None, 0);
    run(&fake, Plan::new(17)).unwrap();
    let events = fake.events();
    let layers: Vec<usize> = events
        .iter()
        .filter_map(|event| {
            if let Event::Layer(layer) = event {
                Some(*layer)
            } else {
                None
            }
        })
        .collect();
    assert_eq!(layers, (0..17).collect::<Vec<_>>(), "every layer runs once, in order");
    let checkpoints: Vec<usize> = events
        .iter()
        .filter_map(|event| {
            if let Event::Checkpoint(done) = event {
                Some(*done)
            } else {
                None
            }
        })
        .collect();
    assert_eq!(checkpoints, vec![8, 16]);
    assert_eq!(events.last(), Some(&Event::Finish), "the heads come last");
    let read_before = |layer: usize| {
        let stage = events.iter().position(|event| *event == Event::Stage(layer)).unwrap();
        let run = events.iter().position(|event| *event == Event::Layer(layer)).unwrap();
        stage < run
    };
    assert!((1..17).all(read_before), "a layer is read before it runs");
    let read_ahead = events.iter().position(|event| *event == Event::Stage(1)).unwrap()
        < events.iter().position(|event| *event == Event::Checkpoint(8)).unwrap();
    assert!(read_ahead, "the next layer is read while the previous one runs");
}

#[test]
fn a_run_that_stops_early_skips_the_heads_and_keeps_its_checkpoint() {
    let fake = Fake::new(None, 0);
    let mut plan = Plan::new(20);
    plan.stop_after = 9;
    run(&fake, plan).unwrap();
    let events = fake.events();
    assert!(!events.contains(&Event::Finish));
    assert!(!events.contains(&Event::Stage(9)), "no layer past the stop is read");
    assert_eq!(
        fake.saved.load(Ordering::Acquire),
        8,
        "the checkpoint stays for a resume"
    );
}

#[test]
fn a_failure_mid_layer_stops_the_run_and_a_resume_starts_after_the_last_checkpoint() {
    let fake = Fake::new(Some(10), 0);
    let error = run(&fake, Plan::new(12)).unwrap_err();
    assert!(error.to_string().contains("layer 10 failed"));
    assert!(
        !fake.events().contains(&Event::Layer(11)),
        "nothing runs after the failure"
    );
    assert!(!fake.events().contains(&Event::Finish));
    assert_eq!(fake.saved.load(Ordering::Acquire), 8);

    let again = Fake::new(None, fake.saved.load(Ordering::Acquire));
    run(&again, Plan::new(12)).unwrap();
    let layers: Vec<usize> = again
        .events()
        .iter()
        .filter_map(|event| {
            if let Event::Layer(layer) = event {
                Some(*layer)
            } else {
                None
            }
        })
        .collect();
    assert_eq!(
        layers,
        vec![8, 9, 10, 11],
        "the resume repeats only the layers after the checkpoint"
    );
}

#[test]
fn a_run_resumed_past_its_last_layer_only_finishes() {
    let fake = Fake::new(None, 4);
    run(&fake, Plan::new(4)).unwrap();
    assert_eq!(fake.events(), vec![Event::Finish]);
}

#[test]
fn a_resume_file_of_another_run_or_of_more_layers_is_not_resumed() {
    let key = resume_key(&["tokens", "8", "512", "recipe", "false", "0.01"]);
    let header = resume_header(&key, 16);
    assert_eq!(resume_at(&header, &key, 64), 16);
    assert_eq!(resume_at(&header, &resume_key(&["other"]), 64), 0, "another run's file");
    assert_eq!(resume_at(&header, &key, 8), 0, "more layers done than the run covers");
    let mut broken = resume_header(&key, 0);
    broken.set("layers", "many");
    assert_eq!(resume_at(&broken, &key, 64), 0, "an unreadable count");
}

#[test]
fn the_largest_linears_are_rounded_first_and_equals_keep_their_order() {
    let shapes: Vec<(String, (u64, u64))> = [("a", (4, 4)), ("b", (10, 10)), ("c", (4, 4)), ("d", (20, 20))]
        .into_iter()
        .map(|(name, shape)| (name.to_owned(), shape))
        .collect();
    assert_eq!(round_order(&shapes), vec![3, 1, 0, 2]);
}

/// A fake matrix: the sum of `x^T x` over the rows it was given.
fn add_rows(matrix: &mut [f64], rows: &[Vec<f64>]) {
    let k = rows[0].len();
    for row in rows {
        for i in 0..k {
            for j in 0..k {
                matrix[i * k + j] += row[i] * row[j];
            }
        }
    }
}

fn data(rows: usize, k: usize) -> Vec<Vec<f64>> {
    (0..rows)
        .map(|row| {
            (0..k)
                .map(|column| ((row * 7 + column * 3) % 11) as f64 - 5.0)
                .collect()
        })
        .collect()
}

#[test]
fn hessians_summed_over_batches_equal_one_pass_over_all_rows() {
    let (k, rows) = (4, 12);
    let all = data(rows, k);
    let mut once = vec![0.0; k * k];
    add_rows(&mut once, &all);

    let mut set = HessianSet::<Vec<f64>>::new();
    for (batch, chunk) in all.chunks(4).enumerate() {
        set.observe(
            "mlp/down",
            (batch as u64, 4, k as u32),
            |columns| Ok(vec![0.0; (columns * columns) as usize]),
            |matrix| {
                add_rows(matrix, chunk);
                Ok(())
            },
        )
        .unwrap();
    }
    assert_eq!(set.matrix("mlp/down").unwrap(), &once);
}

#[test]
fn linears_that_read_the_same_rows_share_one_matrix_and_add_once() {
    let k = 3;
    let rows = data(5, k);
    let adds = AtomicUsize::new(0);
    let mut set = HessianSet::<Vec<f64>>::new();
    for _batch in 0..2 {
        for (name, address) in [("gate", 100), ("up", 100), ("down", 200)] {
            set.observe(
                name,
                (address, 5, k as u32),
                |columns| Ok(vec![0.0; (columns * columns) as usize]),
                |matrix| {
                    adds.fetch_add(1, Ordering::Relaxed);
                    add_rows(matrix, &rows);
                    Ok(())
                },
            )
            .unwrap();
        }
    }
    assert!(set.shares("gate", "up"));
    assert!(!set.shares("gate", "down"));
    assert_eq!(set.matrices().len(), 2);
    assert_eq!(
        adds.load(Ordering::Relaxed),
        4,
        "gate and down add in each of the two batches, up never"
    );
    let mut once = vec![0.0; k * k];
    add_rows(&mut once, &rows);
    add_rows(&mut once, &rows);
    assert_eq!(
        set.matrix("up").unwrap(),
        &once,
        "the shared matrix holds the two batches once each"
    );
}

#[test]
fn a_plan_stops_at_the_smaller_of_its_layers_and_its_stop_and_checkpoints_never_at_the_end() {
    let mut plan = Plan::new(16);
    assert_eq!(plan.stop(), 16, "no stop runs to the end");
    assert!(!plan.stops_early());
    plan.stop_after = 100;
    assert_eq!(plan.stop(), 16, "a stop past the end is the end");
    assert!(!plan.stops_early());
    plan.stop_after = 16;
    assert!(!plan.stops_early(), "a stop at the end is a full run");
    plan.stop_after = 9;
    assert!(plan.stops_early());
    assert!(plan.checkpoint_due(8));
    assert!(!plan.checkpoint_due(16), "the end is not checkpointed");
    assert!(!plan.checkpoint_due(7));
    assert!(
        plan.checkpoint_due(0),
        "nothing done is a multiple of every count, as Java's loop never asks it"
    );
    plan.checkpoint_every = 0;
    assert!(!plan.checkpoint_due(8), "no interval checkpoints nothing");
    assert!(!Plan::new(8).checkpoint_due(8));
}

#[test]
fn no_shapes_have_no_round_order() {
    assert!(round_order(&[]).is_empty());
}

/// Where a fake fails.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Point {
    Begin,
    Stage(usize),
    Checkpoint,
    Finish,
}

/// A fake that fails at one point and counts the layers it ran.
struct Failing {
    at: Point,
    layers: AtomicUsize,
}

impl Failing {
    fn run(at: Point, plan: Plan) -> (Result<(), CalibrateError>, usize) {
        let fake = Arc::new(Failing {
            at,
            layers: AtomicUsize::new(0),
        });
        let (done, receiver) = result_sink();
        drive(Arc::clone(&fake), plan, done);
        let result = receiver.recv_timeout(HANG).expect("the calibration never ended");
        (result, fake.layers.load(Ordering::Acquire))
    }

    fn fail(&self, at: Point, then: Then<()>, otherwise: impl FnOnce(Then<()>)) {
        if self.at == at {
            then(Err(CalibrateError::Failed("stopped".to_owned())));
        } else {
            otherwise(then);
        }
    }
}

impl Stages for Failing {
    type Staged = usize;

    fn begin(&self, _plan: &Plan, then: Then<usize>) {
        if self.at == Point::Begin {
            then(Err(CalibrateError::Failed("stopped".to_owned())));
        } else {
            then(Ok(0));
        }
    }

    fn stage(&self, layer: usize, then: Then<usize>) {
        if self.at == Point::Stage(layer) {
            then(Err(CalibrateError::Failed("stopped".to_owned())));
        } else {
            then(Ok(layer));
        }
    }

    fn layer(&self, _layer: usize, _staged: usize, then: Then<()>) {
        self.layers.fetch_add(1, Ordering::AcqRel);
        then(Ok(()));
    }

    fn checkpoint(&self, _done: usize, then: Then<()>) {
        self.fail(Point::Checkpoint, then, |then| then(Ok(())));
    }

    fn finish(&self, then: Then<()>) {
        self.fail(Point::Finish, then, |then| then(Ok(())));
    }
}

#[test]
fn a_failure_to_begin_runs_no_layer() {
    let (result, layers) = Failing::run(Point::Begin, Plan::new(4));
    assert!(result.is_err());
    assert_eq!(layers, 0);
}

#[test]
fn a_failure_to_read_the_next_layer_ends_the_run_after_the_layers_before_it() {
    let (result, layers) = Failing::run(Point::Stage(2), Plan::new(4));
    assert!(result.is_err());
    assert!(layers <= 2, "the layer whose read failed never runs, ran {layers}");
}

#[test]
fn a_failure_to_checkpoint_ends_the_run_before_the_next_layer() {
    let (result, layers) = Failing::run(Point::Checkpoint, Plan::new(12));
    assert!(result.is_err());
    assert_eq!(layers, 8, "the layers up to the checkpoint ran and no more");
}

#[test]
fn a_failure_of_the_heads_is_the_runs_result() {
    let (result, layers) = Failing::run(Point::Finish, Plan::new(3));
    assert!(result.is_err());
    assert_eq!(layers, 3);
}

#[test]
fn a_plan_of_no_layers_only_finishes() {
    let fake = Fake::new(None, 0);
    run(&fake, Plan::new(0)).unwrap();
    assert_eq!(fake.events(), vec![Event::Finish]);
}

// ---- the driver's join, with the layer and the read of the next layer ending in either order ----

/// What a gated fake holds back: the continuation of a call the test ends by hand.
enum Held {
    Layer(usize, Then<()>),
    Stage(usize, Then<usize>),
}

/// A fake whose layers and reads (past the first) end when the test says.
struct Gated {
    log: Mutex<Vec<Event>>,
    held: Mutex<Vec<Held>>,
}

impl Gated {
    fn run(plan: Plan) -> (Arc<Gated>, Receiver<Result<(), CalibrateError>>) {
        let fake = Arc::new(Gated {
            log: Mutex::new(Vec::new()),
            held: Mutex::new(Vec::new()),
        });
        let (done, receiver) = result_sink();
        drive(Arc::clone(&fake), plan, done);
        (fake, receiver)
    }

    fn events(&self) -> Vec<Event> {
        self.log.lock().unwrap().clone()
    }

    /// Ends the held layer `layer`.
    fn end_layer(&self, layer: usize) {
        let then = self.take(|held| matches!(held, Held::Layer(at, _) if *at == layer));
        if let Held::Layer(_, then) = then {
            then(Ok(()));
        }
    }

    /// Ends the held read of `layer`.
    fn end_read(&self, layer: usize) {
        let then = self.take(|held| matches!(held, Held::Stage(at, _) if *at == layer));
        if let Held::Stage(at, then) = then {
            then(Ok(at));
        }
    }

    fn take(&self, wanted: impl Fn(&Held) -> bool) -> Held {
        let mut held = self.held.lock().unwrap();
        let at = held.iter().position(wanted).expect("the call is not held");
        held.remove(at)
    }
}

impl Stages for Gated {
    type Staged = usize;

    fn begin(&self, _plan: &Plan, then: Then<usize>) {
        then(Ok(0));
    }

    fn stage(&self, layer: usize, then: Then<usize>) {
        self.log.lock().unwrap().push(Event::Stage(layer));
        if layer == 0 {
            then(Ok(0));
        } else {
            self.held.lock().unwrap().push(Held::Stage(layer, then));
        }
    }

    fn layer(&self, layer: usize, _staged: usize, then: Then<()>) {
        self.log.lock().unwrap().push(Event::Layer(layer));
        self.held.lock().unwrap().push(Held::Layer(layer, then));
    }

    fn checkpoint(&self, done: usize, then: Then<()>) {
        self.log.lock().unwrap().push(Event::Checkpoint(done));
        then(Ok(()));
    }

    fn finish(&self, then: Then<()>) {
        self.log.lock().unwrap().push(Event::Finish);
        then(Ok(()));
    }
}

#[test]
fn the_next_layer_is_read_while_the_layer_runs() {
    let (fake, _receiver) = Gated::run(Plan::new(3));
    assert_eq!(
        fake.events(),
        vec![Event::Stage(0), Event::Stage(1), Event::Layer(0)],
        "layer 1 is read before layer 0 ends, and layer 0 runs without waiting for it"
    );
}

#[test]
fn the_next_layer_runs_when_the_read_ends_after_the_layer() {
    let (fake, receiver) = Gated::run(Plan::new(2));
    fake.end_layer(0);
    assert!(!fake.events().contains(&Event::Layer(1)), "layer 1 waits for its read");
    fake.end_read(1);
    assert!(fake.events().contains(&Event::Layer(1)));
    fake.end_layer(1);
    receiver.recv_timeout(HANG).expect("the run never ended").unwrap();
    assert_eq!(fake.events().last(), Some(&Event::Finish));
}

#[test]
fn the_next_layer_runs_when_the_layer_ends_after_the_read() {
    let (fake, receiver) = Gated::run(Plan::new(2));
    fake.end_read(1);
    assert!(!fake.events().contains(&Event::Layer(1)), "layer 1 waits for layer 0");
    fake.end_layer(0);
    assert!(fake.events().contains(&Event::Layer(1)));
    fake.end_layer(1);
    receiver.recv_timeout(HANG).expect("the run never ended").unwrap();
}

#[test]
fn a_read_that_fails_after_its_layer_ended_ends_the_run_with_the_failure() {
    let (fake, receiver) = Gated::run(Plan::new(2));
    fake.end_layer(0);
    let held = fake.take(|held| matches!(held, Held::Stage(1, _)));
    if let Held::Stage(_, then) = held {
        then(Err(CalibrateError::Failed("read failed".to_owned())));
    }
    let error = receiver.recv_timeout(HANG).expect("the run never ended").unwrap_err();
    assert!(error.to_string().contains("read failed"));
    assert!(!fake.events().contains(&Event::Layer(1)));
}

#[test]
fn a_layer_that_fails_before_its_read_ended_ends_the_run_once() {
    let (fake, receiver) = Gated::run(Plan::new(2));
    let held = fake.take(|held| matches!(held, Held::Layer(0, _)));
    if let Held::Layer(_, then) = held {
        then(Err(CalibrateError::Failed("layer failed".to_owned())));
    }
    fake.end_read(1);
    let error = receiver.recv_timeout(HANG).expect("the run never ended").unwrap_err();
    assert!(error.to_string().contains("layer failed"));
    assert!(receiver.try_recv().is_err(), "the run's result is told once");
    assert!(!fake.events().contains(&Event::Layer(1)));
}

// ---- the rounding schedule, over fake lanes ----

/// A fake lane: a number the test follows.
#[derive(Debug, PartialEq, Eq)]
struct FakeLane(usize);

/// A fake decoded buffer: the index of the linear it belongs to.
#[derive(Debug, PartialEq, Eq)]
struct FakeBuffer(usize);

type FakeSchedule = Schedule<FakeLane, FakeBuffer>;

/// What a round ends with.
type Ended = Result<Settled<FakeLane, FakeBuffer>, CalibrateError>;

fn schedule(linears: usize, lanes: usize, order: Vec<usize>) -> (Arc<FakeSchedule>, Receiver<Ended>, usize) {
    let decoded = (0..linears)
        .map(|index| {
            let held = Handoff::empty();
            held.send(FakeBuffer(index));
            held
        })
        .collect();
    let (sender, receiver) = channel();
    let (schedule, started) = Schedule::new(
        (0..linears).map(|index| format!("linear{index}")).collect(),
        order,
        decoded,
        lanes,
        Box::new(move |settled| sender.send(settled).unwrap()),
    );
    (Arc::new(schedule), receiver, started)
}

/// A linear a lane took: which lane, which linear, with which buffer.
type Taken = (usize, usize, FakeBuffer, FakeLane);

/// Pulls for lane `number` and returns what it took, if anything.
fn pull(schedule: &FakeSchedule, number: usize) -> Option<Taken> {
    let mut taken = None;
    schedule.pull(number, FakeLane(number), |index, buffer, lane| {
        taken = Some((number, index, buffer, lane));
    });
    taken
}

#[test]
fn lanes_take_the_linears_in_order_and_every_lane_and_buffer_comes_back() {
    let (schedule, receiver, started) = schedule(4, 3, vec![2, 0, 3, 1]);
    assert_eq!(started, 3);
    let first: Vec<Taken> = (0..3).map(|number| pull(&schedule, number).unwrap()).collect();
    assert_eq!(
        first.iter().map(|taken| taken.1).collect::<Vec<_>>(),
        vec![2, 0, 3],
        "largest first"
    );
    let (number, index, buffer, lane) = first.into_iter().next().unwrap();
    assert_eq!(
        (buffer, &lane),
        (FakeBuffer(index), &FakeLane(number)),
        "a linear's own buffer and the lane's own"
    );
    schedule.keep(index, FakeBuffer(index));
    let fourth = pull(&schedule, number).expect("the lane that ended takes the last linear");
    assert_eq!(fourth.1, 1);
    schedule.keep(fourth.1, fourth.2);
    schedule.keep(0, FakeBuffer(0));
    schedule.keep(3, FakeBuffer(3));
    assert!(receiver.try_recv().is_err(), "the round has not ended while lanes work");
    for lane in 0..3 {
        assert!(pull(&schedule, lane).is_none(), "no linear is left");
    }
    let settled = receiver.recv_timeout(HANG).expect("the round never ended").unwrap();
    assert_eq!(settled.lanes, vec![FakeLane(0), FakeLane(1), FakeLane(2)]);
    assert_eq!(
        settled.decoded,
        (0..4).map(|index| Some(FakeBuffer(index))).collect::<Vec<_>>(),
        "every decoded buffer is given back"
    );
}

#[test]
fn more_lanes_than_linears_start_only_as_many_as_there_are_linears_and_park_the_rest() {
    let (schedule, receiver, started) = schedule(2, 3, vec![0, 1]);
    assert_eq!(started, 2);
    schedule.park(2, FakeLane(2));
    for number in 0..2 {
        let taken = pull(&schedule, number).unwrap();
        schedule.keep(taken.1, taken.2);
    }
    for number in 0..2 {
        assert!(pull(&schedule, number).is_none());
    }
    let settled = receiver.recv_timeout(HANG).expect("the round never ended").unwrap();
    assert_eq!(settled.lanes.len(), 3, "the parked lane comes back with the others");
}

#[test]
fn a_failure_while_others_are_in_flight_lets_them_end_takes_no_more_linears_and_tells_the_first_error() {
    let (schedule, receiver, _) = schedule(5, 3, vec![0, 1, 2, 3, 4]);
    let taken: Vec<Taken> = (0..3).map(|number| pull(&schedule, number).unwrap()).collect();
    let mut taken = taken.into_iter();
    let (failing, _, _, lane) = taken.next().unwrap();
    schedule.abort(failing, lane, CalibrateError::Failed("first".to_owned()));
    assert!(receiver.try_recv().is_err(), "the round waits for the lanes in flight");
    let (second, index, buffer, _) = taken.next().unwrap();
    schedule.keep(index, buffer);
    assert!(pull(&schedule, second).is_none(), "a failed round hands out no linear");
    assert!(receiver.try_recv().is_err(), "one lane is still in flight");
    let (third, index, buffer, lane) = taken.next().unwrap();
    schedule.abort(third, lane, CalibrateError::Failed("second".to_owned()));
    schedule.keep(index, buffer);
    let error = receiver.recv_timeout(HANG).expect("the round never ended").unwrap_err();
    assert_eq!(error.to_string(), "first", "the first error is the round's");
}

#[test]
fn a_linear_whose_buffer_was_not_given_fails_the_round_naming_it() {
    let decoded = vec![Handoff::empty()];
    let (sender, receiver) = channel();
    let (schedule, _) = Schedule::<FakeLane, FakeBuffer>::new(
        vec!["query_key".to_owned()],
        vec![0],
        decoded,
        1,
        Box::new(move |settled| sender.send(settled).unwrap()),
    );
    assert!(pull(&schedule, 0).is_none());
    let error = receiver.recv_timeout(HANG).expect("the round never ended").unwrap_err();
    assert!(error.to_string().contains("query_key"), "{error}");
}

// ---- the embedding ----

#[test]
fn the_embedding_table_must_be_bf16_with_one_row_per_token_of_the_hidden_size() {
    assert_eq!(embedding::vocabulary("BF16", &[10, 4], 4).unwrap(), 10);
    assert!(embedding::vocabulary("F32", &[10, 4], 4).is_err(), "another dtype");
    assert!(embedding::vocabulary("BF16", &[10, 5], 4).is_err(), "another width");
    assert!(embedding::vocabulary("BF16", &[40], 4).is_err(), "not a matrix");
    assert!(embedding::vocabulary("BF16", &[0, 4], 4).is_err(), "no rows");
    assert_eq!(embedding::row_bytes(4), 8);
}

#[test]
fn a_token_outside_the_table_is_refused_by_position_before_any_row_is_copied() {
    assert!(embedding::check_tokens(&[0, 9], 10).is_ok());
    assert!(embedding::check_tokens(&[], 10).is_ok());
    let past = embedding::check_tokens(&[3, 10], 10).unwrap_err().to_string();
    assert!(past.contains("token 10 at 1"), "{past}");
    let negative = embedding::check_tokens(&[-1], 10).unwrap_err().to_string();
    assert!(negative.contains("token -1 at 0"), "{negative}");
    let table = [0u8; 8];
    let mut states = [7u8; 8];
    assert!(embedding::copy_rows(&[0, 5], &table, 4, &mut states).is_err());
    assert_eq!(states, [7u8; 8], "nothing was copied");
}

#[test]
fn each_token_gets_its_own_row_in_its_own_position() {
    let table: Vec<u8> = (0..12).collect();
    let mut states = [0u8; 12];
    embedding::copy_rows(&[2, 0, 2], &table, 4, &mut states).unwrap();
    assert_eq!(states, [8, 9, 10, 11, 0, 1, 2, 3, 8, 9, 10, 11]);
    let mut wrong = [0u8; 8];
    assert!(
        embedding::copy_rows(&[0, 1, 2], &table, 4, &mut wrong).is_err(),
        "states of another size"
    );
}

// ---- the manifest and the output directory ----

const FIELDS: [&str; 2] = ["the tokens", "the batch"];

fn touch(directory: &Path, names: &[&str]) {
    for name in names {
        fs::write(directory.join(name), name.as_bytes()).unwrap();
    }
}

fn names(directory: &Path) -> Vec<String> {
    let mut names: Vec<String> = fs::read_dir(directory)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .collect();
    names.sort();
    names
}

/// A directory that holds the manifest of a run of `key` that listed `files` (which exist) and some neighbours.
fn run_directory(key: &str, files: &[&str], neighbours: &[&str]) -> tempfile::TempDir {
    let directory = tempfile::tempdir().unwrap();
    let mut manifest = Manifest::new(key, 64).after(2);
    manifest.files = files.iter().map(|name| (*name).to_owned()).collect();
    fs::write(directory.path().join("calibration.json"), manifest.to_bytes().unwrap()).unwrap();
    touch(directory.path(), files);
    touch(directory.path(), neighbours);
    directory
}

fn refusal(result: Result<impl Debug, CalibrateError>) -> String {
    result.unwrap_err().to_string()
}

#[test]
fn an_empty_or_missing_directory_starts_a_run_that_lists_nothing() {
    let directory = tempfile::tempdir().unwrap();
    let prepared = inspect(directory.path(), "run", 64, &FIELDS, false).unwrap();
    assert_eq!(prepared.manifest(), &Manifest::new("run", 64));
    assert!(names(directory.path()).is_empty(), "inspecting writes nothing");
    let missing = directory.path().join("new");
    let prepared = inspect(&missing, "run", 64, &FIELDS, false).unwrap();
    assert!(!missing.exists(), "inspecting creates nothing");
    prepared.apply().unwrap();
    let written = Manifest::parse(&fs::read(missing.join("calibration.json")).unwrap()).unwrap();
    assert_eq!(written, Manifest::new("run", 64));
    assert_eq!(
        names(&missing),
        vec!["calibration.json"],
        "the manifest is renamed into place, no temporary left"
    );
}

#[test]
fn the_directory_of_the_same_key_resumes_and_keeps_every_file() {
    let directory = run_directory("run", &["a.npz", "b.npz"], &["hidden.bin", "c.npz.tmp", "notes.txt"]);
    let before = names(directory.path());
    let prepared = inspect(directory.path(), "run", 64, &FIELDS, false).unwrap();
    assert_eq!(prepared.manifest().layers_done, 2);
    assert_eq!(prepared.manifest().files, vec!["a.npz", "b.npz"]);
    prepared.apply().unwrap();
    assert_eq!(names(directory.path()), before);
}

#[test]
fn a_directory_with_files_and_no_manifest_is_refused_naming_them_and_nothing_is_removed() {
    let directory = tempfile::tempdir().unwrap();
    touch(directory.path(), &["mine.npz", "text.layers.0.mlp.npz", "hidden.bin"]);
    for fresh in [false, true] {
        let why = refusal(inspect(directory.path(), "run", 64, &FIELDS, fresh));
        assert!(why.contains("mine.npz") && why.contains("calibration.json"), "{why}");
        assert_eq!(names(directory.path()).len(), 3, "nothing was removed");
    }
}

#[test]
fn a_manifest_that_is_not_a_calibrations_is_refused_and_nothing_is_removed() {
    let directory = tempfile::tempdir().unwrap();
    touch(directory.path(), &["a.npz"]);
    for manifest in [
        &b"{ torn"[..],
        br#"{"producer":"fake_quant","objects":["a"],"complete":true}"#,
    ] {
        fs::write(directory.path().join("calibration.json"), manifest).unwrap();
        for fresh in [false, true] {
            let why = refusal(inspect(directory.path(), "run", 64, &FIELDS, fresh));
            assert!(why.contains("not a calibration's manifest"), "{why}");
        }
        assert_eq!(names(directory.path()), vec!["a.npz", "calibration.json"]);
        assert_eq!(fs::read(directory.path().join("calibration.json")).unwrap(), manifest);
    }
}

#[test]
fn listed_files_that_are_missing_refuse_the_run_naming_them() {
    let directory = run_directory("run", &["a.npz", "b.npz"], &[]);
    fs::remove_file(directory.path().join("b.npz")).unwrap();
    let why = refusal(inspect(directory.path(), "run", 64, &FIELDS, false));
    assert!(why.contains("b.npz") && !why.contains("a.npz"), "{why}");
}

#[test]
fn another_key_is_refused_naming_the_fields_that_differ_and_nothing_is_removed() {
    let directory = run_directory("tokens|8", &["a.npz"], &["hidden.bin"]);
    let why = refusal(inspect(directory.path(), "tokens|4", 64, &FIELDS, false));
    assert!(why.contains("the batch") && !why.contains("the tokens,"), "{why}");
    assert!(why.contains("--fresh"), "{why}");
    let why = refusal(inspect(directory.path(), "tokens|8", 40, &FIELDS, false));
    assert!(why.contains("the model's layers"), "{why}");
    assert_eq!(names(directory.path()), vec!["a.npz", "calibration.json", "hidden.bin"]);
}

#[test]
fn fresh_removes_only_the_files_the_old_manifest_lists_and_a_foreign_file_survives() {
    let directory = run_directory(
        "tokens|8",
        &["a.npz", "b.npz"],
        &["notes.txt", "hidden.bin", "stray.npz"],
    );
    let prepared = inspect(directory.path(), "tokens|4", 64, &FIELDS, true).unwrap();
    assert_eq!(
        names(directory.path()).len(),
        6,
        "inspecting removes nothing, even with --fresh"
    );
    let manifest = prepared.apply().unwrap();
    assert_eq!(manifest, Manifest::new("tokens|4", 64));
    assert_eq!(
        names(directory.path()),
        vec!["calibration.json", "hidden.bin", "notes.txt", "stray.npz"],
        "the listed files are gone; the others stay"
    );
    let written = Manifest::parse(&fs::read(directory.path().join("calibration.json")).unwrap()).unwrap();
    assert_eq!(written.key, "tokens|4");
}

#[test]
fn a_prepared_directory_that_is_dropped_because_the_open_failed_changes_nothing() {
    let directory = run_directory("tokens|8", &["a.npz"], &[]);
    let before = names(directory.path());
    let manifest = fs::read(directory.path().join("calibration.json")).unwrap();
    drop(inspect(directory.path(), "tokens|4", 64, &FIELDS, true).unwrap());
    assert_eq!(names(directory.path()), before);
    assert_eq!(fs::read(directory.path().join("calibration.json")).unwrap(), manifest);
}

// ---- the ledger: a layer is counted only when its files and its manifest are saved ----

/// What a fake manifest save did: the bytes it was asked to write, and whether it succeeds.
struct FakeWrite {
    written: Arc<Mutex<Vec<Manifest>>>,
    succeed: bool,
}

impl FakeWrite {
    fn call(&self) -> impl FnOnce(Vec<u8>, Then<()>) + use<> {
        let (written, succeed) = (Arc::clone(&self.written), self.succeed);
        move |bytes, then| {
            written.lock().unwrap().push(Manifest::parse(&bytes).unwrap());
            then(if succeed {
                Ok(())
            } else {
                Err(CalibrateError::Failed("save failed".to_owned()))
            });
        }
    }
}

fn recorded(
    ledger: &Arc<Ledger>,
    done: usize,
    result: Result<(), CalibrateError>,
    write: &FakeWrite,
) -> Result<(), CalibrateError> {
    let (sender, receiver) = channel();
    ledger.record(
        done,
        result,
        write.call(),
        Box::new(move |result| sender.send(result).unwrap()),
    );
    receiver.recv_timeout(HANG).expect("the record never ended")
}

#[test]
fn a_layer_is_counted_with_its_listed_files_after_its_manifest_is_saved() {
    let ledger = Arc::new(Ledger::new(Manifest::new("run", 4)));
    let write = FakeWrite {
        written: Arc::default(),
        succeed: true,
    };
    ledger.list(["a.npz".to_owned(), "b.npz".to_owned()]);
    ledger.list(["b.npz".to_owned()]);
    recorded(&ledger, 1, Ok(()), &write).unwrap();
    let saved = write.written.lock().unwrap().clone();
    assert_eq!(saved.len(), 1);
    assert_eq!(
        (saved[0].layers_done, saved[0].files.clone()),
        (1, vec!["a.npz".to_owned(), "b.npz".to_owned()])
    );
    assert_eq!(ledger.manifest().unwrap(), saved[0]);
}

#[test]
fn a_failed_layer_records_nothing() {
    let ledger = Arc::new(Ledger::new(Manifest::new("run", 4)));
    let write = FakeWrite {
        written: Arc::default(),
        succeed: true,
    };
    let error = recorded(
        &ledger,
        1,
        Err(CalibrateError::Failed("layer failed".to_owned())),
        &write,
    )
    .unwrap_err();
    assert!(error.to_string().contains("layer failed"));
    assert!(write.written.lock().unwrap().is_empty(), "no manifest was written");
    assert_eq!(ledger.manifest().unwrap().layers_done, 0);
}

#[test]
fn a_failed_manifest_save_does_not_count_the_layer_and_a_later_one_can() {
    let ledger = Arc::new(Ledger::new(Manifest::new("run", 4)));
    let failing = FakeWrite {
        written: Arc::default(),
        succeed: false,
    };
    let error = recorded(&ledger, 1, Ok(()), &failing).unwrap_err();
    assert!(error.to_string().contains("save failed"));
    assert_eq!(ledger.manifest().unwrap().layers_done, 0, "the layer is not counted");
    let working = FakeWrite {
        written: Arc::default(),
        succeed: true,
    };
    recorded(&ledger, 1, Ok(()), &working).unwrap();
    assert_eq!(ledger.manifest().unwrap().layers_done, 1);
}

#[test]
fn the_fields_that_differ_are_named_and_a_key_of_another_shape_is_named_whole() {
    let (old, new) = (Manifest::new("t|8", 4), Manifest::new("t|4", 4));
    assert_eq!(differences(&old, &new, &FIELDS), vec!["the batch"]);
    assert_eq!(differences(&old, &Manifest::new("t|8|x", 4), &FIELDS), vec!["the key"]);
    assert!(differences(&old, &old, &FIELDS).is_empty());
}

#[test]
fn a_run_of_the_whole_model_ends_with_an_error_naming_the_heads_and_the_mtp_block() {
    let (sender, receiver) = channel();
    refuse_finish(Box::new(move |result| sender.send(result).unwrap()));
    let why = receiver.recv_timeout(HANG).unwrap().unwrap_err().to_string();
    assert!(
        why.contains("output head") && why.contains("MTP") && why.contains("incomplete"),
        "{why}"
    );
}

#[test]
fn a_manifest_is_complete_only_when_every_layer_and_both_heads_and_the_mtp_block_are_written() {
    let mut manifest = Manifest::new("run", 4);
    assert!(!manifest.is_complete());
    manifest = manifest.after(4);
    assert!(!manifest.is_complete(), "every layer but no heads");
    manifest.heads_done = true;
    assert!(!manifest.is_complete(), "no MTP block");
    manifest.mtp_done = true;
    assert!(manifest.is_complete());
    assert!(!manifest.after(3).is_complete());
    assert_eq!(manifest.after(9).layers_done, 4, "never past the model");
    assert_eq!(Manifest::parse(&manifest.to_bytes().unwrap()).unwrap(), manifest);
    assert!(Manifest::parse(b"{}").is_err());
}

#[test]
fn the_token_digest_follows_the_tokens_contents() {
    assert_eq!(tokens_digest(&[1, 2, 3]), tokens_digest(&[1, 2, 3]));
    assert_ne!(tokens_digest(&[1, 2, 3]), tokens_digest(&[1, 2, 4]));
    assert_ne!(tokens_digest(&[1, 2]), tokens_digest(&[1, 2, 0]));
}
