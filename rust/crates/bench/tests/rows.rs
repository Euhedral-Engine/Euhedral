//! The row pipeline on a real lattice, without a GPU: rows are handed to a sink as the engine hands them, the reference
//! records are read with `AsyncIo`, and the steps are scored in `host` lake frames.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]
#![allow(
    clippy::disallowed_methods,
    reason = "no-wait, waiting call: a test thread waits on the latch for the frames it published, and writes its file"
)]

use std::fs;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::Duration;

use euhedral_core::control_plane::ControlPlaneLattice;
use euhedral_core::generics::LatticeSource;
use euhedral_graph::{AsyncIo, Lake};
use euhedral_inference_bench::kld::{self, Reference};
use euhedral_inference_bench::rows::{ReferenceRows, RowInput, RowSink, RowTask};
use euhedral_inference_bench::score::{FIELDS, score_row};
use euhedral_inference_engine::generation::{FinishReason, GenerationOutcome, LogitsSink};
use euhedral_inference_engine::lattice_config;
use euhedral_inference_settings::{ModelSettings, Settings};

const CONTEXT: usize = 40;
const VOCABULARY: usize = 33;

fn bf16(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

/// The logits of step `step`: a deterministic spread of values BF16 holds exactly.
fn logits(step: usize) -> Vec<u8> {
    bf16(
        &(0..VOCABULARY)
            .map(|token| ((token * 7 + step * 3) % 11) as f32 - 5.0)
            .collect::<Vec<_>>(),
    )
}

struct Scoring {
    rows: ReferenceRows,
}

impl RowTask for Scoring {
    fn fields(&self) -> usize {
        FIELDS
    }

    fn reference(&self) -> Option<&ReferenceRows> {
        Some(&self.rows)
    }

    fn compute(&self, input: &RowInput, out: &mut [f64]) {
        let row = kld::Row::new(&input.reference, VOCABULARY);
        out.copy_from_slice(&score_row(&input.logits, &row, VOCABULARY, input.forced as usize));
    }
}

#[test]
fn every_step_is_read_and_scored_in_frames_and_the_values_come_back_in_step_order() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("reference.kld");
    let tokens: Vec<Vec<u32>> = vec![(0..CONTEXT as u32).map(|i| i % VOCABULARY as u32).collect(); 2];
    let bytes = kld::encode(CONTEXT, VOCABULARY, &tokens, |chunk, step| {
        (
            1.0 / 16.0,
            -16.0,
            (0..VOCABULARY)
                .map(|token| (200 + 13 * ((token + step + chunk) % 17)) as u16)
                .collect(),
        )
    });
    fs::write(&path, &bytes).unwrap();
    let (reference, file) = Reference::open(&path).unwrap();
    let file = Arc::new(file);
    let steps = reference.records();
    assert_eq!(steps, 19);

    let settings = Settings::with_model(ModelSettings::default());
    let lattice = ControlPlaneLattice::new(lattice_config(&settings)).expect("the lattice starts");
    let io = AsyncIo::open().expect("the asynchronous I/O opens");
    lattice
        .add_upstream(Arc::clone(&io) as Arc<dyn LatticeSource>)
        .expect("the lattice takes the I/O");
    let host = Lake::new(
        Arc::clone(&lattice),
        NonZeroUsize::new(2).unwrap(),
        Duration::from_secs(1),
    );

    // Chunk 1: the reference's records of that chunk.
    let task = Arc::new(Scoring {
        rows: ReferenceRows {
            file: Arc::clone(&file),
            first_offset: reference.row_offset(1, 0),
            row_bytes: reference.row_bytes(),
        },
    });
    let (mut sink, engine, values) = RowSink::new(Arc::clone(&io), Arc::clone(&host), task, steps).unwrap();
    for step in 0..steps {
        sink.row(step, reference.forced(1)[step], &logits(step));
    }
    sink.finished(GenerationOutcome {
        reason: FinishReason::Length,
        prompt_tokens: reference.prefix_len(),
        generated: steps,
        elapsed: Duration::ZERO,
    });
    drop(sink);
    assert_eq!(engine.wait().generated, steps);
    let scored = values.wait().expect("every step is scored");
    assert_eq!(scored.len(), steps * FIELDS);

    for step in 0..steps {
        let at = reference.row_offset(1, step) as usize;
        let row = kld::Row::new(&bytes[at..at + reference.row_bytes()], VOCABULARY);
        let expected = score_row(&logits(step), &row, VOCABULARY, reference.forced(1)[step] as usize);
        assert_eq!(&scored[step * FIELDS..(step + 1) * FIELDS], &expected, "step {step}");
    }

    host.complete_gracefully();
    io.complete();
    lattice.close();
}

#[test]
fn a_row_past_the_run_or_a_short_reference_record_fails_the_run_with_the_reason() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("short.bin");
    fs::write(&path, vec![0u8; 100]).unwrap();
    let file = Arc::new(euhedral_graph::OwnedFile::from(fs::File::open(&path).unwrap()));

    let settings = Settings::with_model(ModelSettings::default());
    let lattice = ControlPlaneLattice::new(lattice_config(&settings)).expect("the lattice starts");
    let io = AsyncIo::open().expect("the asynchronous I/O opens");
    lattice
        .add_upstream(Arc::clone(&io) as Arc<dyn LatticeSource>)
        .expect("the lattice takes the I/O");
    let host = Lake::new(
        Arc::clone(&lattice),
        NonZeroUsize::new(2).unwrap(),
        Duration::from_secs(1),
    );

    let task = Arc::new(Scoring {
        rows: ReferenceRows {
            file,
            first_offset: 0,
            row_bytes: 160,
        },
    });
    let (mut sink, _engine, values) = RowSink::new(Arc::clone(&io), Arc::clone(&host), task, 2).unwrap();
    sink.row(0, 1, &logits(0));
    sink.row(5, 1, &logits(1));
    drop(sink);
    let why = values.wait().expect_err("the run fails");
    assert!(
        why.contains("row 5 arrived in a run of 2 steps") || why.contains("reference record"),
        "{why}"
    );

    host.complete_gracefully();
    io.complete();
    lattice.close();
}
