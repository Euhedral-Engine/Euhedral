//! Device test of the score kernel on its lane: the five values of each row against the same arithmetic on the host.
//! `#[ignore]`d like the crate's other device tests:
//! `cargo test -p euhedral-inference-quantization --test score_gpu -- --ignored --nocapture`.
//!
//! The engine library comes from the `euhedral.cuda.library` property (`EUHEDRAL_TEST_CUDA_LIBRARY`); without it the
//! test fails: a device test that skips is no check.
#![cfg(target_os = "linux")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::disallowed_methods,
    clippy::disallowed_types,
    reason = "a test file: the pump is the worker, and a failed unwrap is the failure"
)]

use std::f64::consts::TAU;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use euhedral_core::generics::LatticeSource;
use euhedral_gpu::cuda::{CudaGpu, CudaModule};
use euhedral_gpu::{Gpu, StreamOptions};
use euhedral_graph::Completions;
use euhedral_inference_cuda::CudaLibrary;
use euhedral_inference_quantization::kernels::{ModuleOptions, Modules};
use euhedral_inference_quantization::score::device::{Finished, Geometry, Refused, ScoreKernels, ScoreLane, ScoreRow};
use euhedral_inference_quantization::score::summary::FIELDS;

const VOCABULARY: usize = 1001;
const ROWS: usize = 5;
/// The reference floor below which a row's probabilities do not count towards the KL.
const FLOOR: f64 = -16.0;

/// A small deterministic generator: the test needs repeatable rows, not good ones.
struct Random(u64);

impl Random {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn gaussian(&mut self) -> f64 {
        let (a, b) = (self.unit().max(1e-12), self.unit());
        (-2.0 * a.ln()).sqrt() * (TAU * b).cos()
    }
}

struct Case {
    logits: Vec<u16>,
    record: Vec<u16>,
    target: usize,
}

fn bf16(value: f32) -> u16 {
    (value.to_bits() >> 16) as u16
}

fn cases() -> Vec<Case> {
    let mut random = Random(0x9e37_79b9_7f4a_7c15);
    let width = 2 * VOCABULARY.div_ceil(2) + 4;
    let mut cases: Vec<Case> = (0..ROWS)
        .map(|_| {
            let logits = (0..VOCABULARY)
                .map(|_| bf16((random.gaussian() * 3.0) as f32))
                .collect();
            let scale = 16.0f32 / 65535.0;
            let minimum = (FLOOR - random.unit()) as f32;
            let mut record = vec![0u16; width];
            for (at, bits) in [scale.to_bits(), minimum.to_bits()].into_iter().enumerate() {
                record[2 * at] = bits as u16;
                record[2 * at + 1] = (bits >> 16) as u16;
            }
            for code in &mut record[4..4 + VOCABULARY] {
                *code = random.next() as u16;
            }
            Case {
                logits,
                record,
                target: (random.next() % VOCABULARY as u64) as usize,
            }
        })
        .collect();
    // Ties in row 0: two equal maxima on each side, the first of them the one counted.
    cases[0].logits[17] = bf16(40.0);
    cases[0].logits[900] = bf16(40.0);
    cases[0].record[4 + 900] = 65535;
    cases[0].record[4 + 17] = 65535;
    cases
}

/// The kernel's arithmetic on the host: nll, reference nll, KL over the entries above the floor, top-1 agreement and
/// the difference of the probabilities of the target.
fn expected(case: &Case) -> [f64; FIELDS] {
    let word = |at: usize| u32::from(case.record[at]) | u32::from(case.record[at + 1]) << 16;
    let (scale, minimum) = (f32::from_bits(word(0)), f32::from_bits(word(2)));
    let q: Vec<f64> = case
        .logits
        .iter()
        .map(|&bits| f64::from(f32::from_bits(u32::from(bits) << 16)))
        .collect();
    let p: Vec<f64> = case.record[4..4 + VOCABULARY]
        .iter()
        .map(|&code| f64::from(scale) * f64::from(code) + f64::from(minimum))
        .collect();
    let first_max = |values: &[f64]| {
        values
            .iter()
            .enumerate()
            .fold(0, |best, (at, value)| if *value > values[best] { at } else { best })
    };
    let top = q[first_max(&q)];
    let log_z = top + q.iter().map(|value| (value - top).exp()).sum::<f64>().ln();
    let kl: f64 = q
        .iter()
        .zip(&p)
        .filter(|(_, p)| **p > FLOOR)
        .map(|(q, p)| p.exp() * (p - (q - log_z)))
        .sum();
    let (q_target, p_target) = (q[case.target] - log_z, p[case.target]);
    [
        -q_target,
        -p_target,
        kl,
        f64::from(u8::from(first_max(&q) == first_max(&p))),
        q_target.exp() - p_target.exp(),
    ]
}

fn row(case: &Case) -> ScoreRow {
    let bytes = |words: &[u16]| words.iter().flat_map(|word| word.to_le_bytes()).collect();
    ScoreRow {
        logits: bytes(&case.logits),
        record: bytes(&case.record),
        target: u32::try_from(case.target).unwrap(),
    }
}

/// Runs one group on `lane`, pumping the lane's completions until it is done (a minute at most).
fn run(completions: &Arc<Completions>, lane: ScoreLane, rows: Vec<ScoreRow>) -> (ScoreLane, Vec<f64>) {
    let slot: Arc<Mutex<Option<Finished>>> = Arc::default();
    let sink = Arc::clone(&slot);
    let begun = lane.score(rows, Box::new(move |finished| *sink.lock().unwrap() = Some(finished)));
    if let Err(refused) = begun {
        let Refused { error, .. } = *refused;
        panic!("the group was refused: {error}");
    }
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        completions.reap();
        completions.pull(
            &mut |mut frame| {
                frame.execute().unwrap();
                frame.do_finally();
            },
            &mut |_| false,
            u64::MAX,
        );
        if let Some(finished) = slot.lock().unwrap().take() {
            let values = finished
                .result
                .unwrap_or_else(|error| panic!("the group failed: {error}"));
            return (finished.lane, values);
        }
        assert!(
            Instant::now() < deadline,
            "the group never finished: its retirement did not arrive"
        );
    }
}

fn close(label: &str, actual: f64, wanted: f64) {
    assert!(
        (actual - wanted).abs() <= 1e-9 * wanted.abs().max(1.0),
        "{label}: the kernel gave {actual}, the host arithmetic {wanted}"
    );
}

#[test]
#[ignore = "needs a GPU and the bundled NVRTC"]
fn rows_match_the_host_arithmetic_and_a_reused_lane_leaves_nothing_behind() {
    let path = test_support::test_setting("euhedral.cuda.library")
        .expect("set euhedral.cuda.library (EUHEDRAL_TEST_CUDA_LIBRARY) to the engine library");
    let library = CudaLibrary::load(&PathBuf::from(path)).expect("the CUDA library loads");
    let gpu = CudaGpu::open(0).expect("the device opens");
    let options = ModuleOptions::beside(&library).expect("the product directory");
    let mut modules = Modules::<CudaModule>::open(&gpu, options).expect("the compiler gives machine code");
    let kernels = ScoreKernels::load(&mut modules).expect("score.cu compiles");
    let (pool, mut streams) = gpu
        .open_lanes(&["score"], StreamOptions::default())
        .expect("a lane opens");
    let completions = Completions::new(&pool);
    let geometry = Geometry {
        vocabulary: VOCABULARY,
        record_bytes: 2 * (2 * VOCABULARY.div_ceil(2) + 4),
        group: ROWS,
    };
    let lane =
        ScoreLane::new(streams.remove(0), kernels, &library, geometry, Arc::clone(&completions)).expect("the lane");

    let cases = cases();
    let (lane, values) = run(&completions, lane, cases.iter().map(row).collect());
    assert_eq!(values.len(), ROWS * FIELDS);
    for (index, case) in cases.iter().enumerate() {
        for (field, wanted) in expected(case).into_iter().enumerate() {
            close(
                &format!("row {index} field {field}"),
                values[index * FIELDS + field],
                wanted,
            );
        }
    }
    // The tie in row 0 is a top-1 agreement: both sides count the first maximum.
    assert_eq!(values[3], 1.0, "the argmax of a tie is the first maximum on both sides");

    // The same lane again with a smaller group in another order: what the first group staged must not show.
    let (lane, again) = run(&completions, lane, vec![row(&cases[3]), row(&cases[1])]);
    assert_eq!(again.len(), 2 * FIELDS);
    for (slot, source) in [3, 1].into_iter().enumerate() {
        for field in 0..FIELDS {
            assert_eq!(
                again[slot * FIELDS + field],
                values[source * FIELDS + field],
                "row {source} field {field} on the second group"
            );
        }
    }
    lane.close();
}
