//! The calibration's host core: the plan, the Hessians and the layer loop, over a fake of the device's calls.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::disallowed_methods,
    clippy::disallowed_types,
    reason = "a test: the fake records in a mutex and a failed unwrap is the failure"
)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{Sender, channel};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use euhedral_inference_quantization::calibrate::plan::{resume_at, resume_header, resume_key};
use euhedral_inference_quantization::calibrate::{CalibrateError, HessianSet, Plan, Stages, Then, drive, round_order};

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

fn result_sink() -> (Then<()>, std::sync::mpsc::Receiver<Result<(), CalibrateError>>) {
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
fn add_rows(matrix: &mut Vec<f64>, rows: &[Vec<f64>]) {
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
