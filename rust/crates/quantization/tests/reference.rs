//! The reference writer on a real lattice, without a device: a stand-in for the engine forces each chunk from a thread
//! of its own, and the file the writer leaves is compared with the bytes the rows must make.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]
#![allow(
    clippy::disallowed_methods,
    reason = "no-wait, waiting call: a test thread waits on the hand-off for the run it started, and reads its files"
)]

use std::fs;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::Duration;

use euhedral_core::control_plane::ControlPlaneLattice;
use euhedral_core::generics::LatticeSource;
use euhedral_graph::Lake;
use euhedral_inference_engine::generation::{FinishReason, GenerationOutcome, LogitsSink};
use euhedral_inference_engine::lattice_config;
use euhedral_inference_quantization::kld::{Reference, encode_head};
use euhedral_inference_quantization::reference::{
    Forcer, Options, ReferenceError, Written, encode_record, partial::partial_path, write,
};
use euhedral_inference_settings::{ModelSettings, Settings};
use euhedral_io::AsyncIo;

const CONTEXT: usize = 12;
const VOCABULARY: usize = 21;
const CHUNKS: usize = 3;
const HANG: Duration = Duration::from_secs(20);

struct World {
    lattice: Arc<ControlPlaneLattice>,
    io: Arc<AsyncIo>,
    host: Arc<Lake>,
}

impl World {
    fn start() -> World {
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
        World { lattice, io, host }
    }

    fn run(
        &self,
        forcer: &Arc<Fake>,
        out: &Path,
        tokens: Vec<Vec<u32>>,
        options: Options,
    ) -> Result<Written, ReferenceError> {
        let forcer: Arc<dyn Forcer> = Arc::clone(forcer) as Arc<dyn Forcer>;
        write(&self.io, &self.host, forcer, out, CONTEXT, VOCABULARY, tokens, options)
            .wait(HANG)
            .expect("the run never ended")
    }

    fn finish(self) {
        self.host.complete_gracefully();
        self.io.complete();
        self.lattice.close();
    }
}

fn tokens() -> Vec<Vec<u32>> {
    (0..CHUNKS)
        .map(|chunk| {
            (0..CONTEXT)
                .map(|i| ((chunk * 7 + i * 3) % VOCABULARY) as u32)
                .collect()
        })
        .collect()
}

/// The BF16 logits row that predicts step `step` of a chunk that starts with `first`.
fn logits(first: u32, step: usize) -> Vec<u8> {
    (0..VOCABULARY)
        .flat_map(|token| {
            let value = ((first as usize * 13 + step * 5 + token * 11) % 19) as f32 - 9.0;
            ((value.to_bits() >> 16) as u16).to_le_bytes()
        })
        .collect()
}

/// What the engine does, said by a test: the rows come from `logits`; a call can be made to fail.
struct Fake {
    calls: AtomicUsize,
    /// The call (counting from 0) that ends with a failed outcome after one row, if any.
    fail_call: Option<usize>,
    /// The call that does not start.
    refuse_call: Option<usize>,
    /// Rows shorter than the vocabulary.
    short_rows: bool,
}

impl Fake {
    fn new() -> Arc<Fake> {
        Arc::new(Fake {
            calls: AtomicUsize::new(0),
            fail_call: None,
            refuse_call: None,
            short_rows: false,
        })
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

fn outcome(reason: FinishReason, generated: usize) -> GenerationOutcome {
    GenerationOutcome {
        reason,
        prompt_tokens: 0,
        generated,
        elapsed: Duration::ZERO,
        cached_prompt_tokens: 0,
    }
}

impl Forcer for Fake {
    fn force(&self, prefix: Vec<u32>, forced: Vec<u32>, mut sink: Box<dyn LogitsSink>) -> Result<(), String> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst);
        if self.refuse_call == Some(call) {
            return Err("no sequence is free".to_owned());
        }
        let fail = self.fail_call == Some(call);
        let short = self.short_rows;
        thread::spawn(move || {
            for (step, token) in forced.iter().enumerate() {
                if fail && step == 1 {
                    sink.finished(outcome(FinishReason::Failed("the device failed".to_owned()), step));
                    return;
                }
                let mut row = logits(prefix[0], step);
                if short {
                    row.truncate(row.len() - 2);
                }
                sink.row(step, *token, &row);
            }
            sink.finished(outcome(FinishReason::Length, forced.len()));
        });
        Ok(())
    }
}

/// The finished file the rows must make.
fn expected(tokens: &[Vec<u32>]) -> Vec<u8> {
    let layout = Reference {
        context: CONTEXT,
        vocabulary: VOCABULARY,
        tokens: tokens.to_vec(),
    };
    let mut bytes = encode_head(CONTEXT, VOCABULARY, tokens);
    for chunk in 0..tokens.len() {
        for step in 0..layout.records() {
            bytes.extend(encode_record(&logits(tokens[chunk][0], step)));
        }
    }
    bytes
}

fn out_path(directory: &Path) -> PathBuf {
    directory.join("reference.kld")
}

#[test]
fn a_run_writes_the_file_the_rows_make_and_leaves_no_partial() {
    let directory = tempfile::tempdir().unwrap();
    let out = out_path(directory.path());
    let world = World::start();
    let fake = Fake::new();
    let written = world.run(&fake, &out, tokens(), Options::default()).unwrap();
    assert_eq!(
        written,
        Written {
            completed: CHUNKS,
            finished: true
        }
    );
    assert_eq!(fake.calls(), CHUNKS);
    assert_eq!(fs::read(&out).unwrap(), expected(&tokens()));
    assert!(!partial_path(&out).exists());
    world.finish();
}

/// A run stopped after one chunk and run again forces only the chunks left, and its file is byte for byte the file of
/// a run that was never stopped. Catches a resume that restarts at chunk 0 or leaves the count in the file.
#[test]
fn a_stopped_run_resumes_to_the_bytes_of_an_unstopped_run() {
    let directory = tempfile::tempdir().unwrap();
    let out = out_path(directory.path());
    let world = World::start();
    let fake = Fake::new();
    let stopped = world
        .run(&fake, &out, tokens(), Options { stop_after: Some(1) })
        .unwrap();
    assert_eq!(
        stopped,
        Written {
            completed: 1,
            finished: false
        }
    );
    assert!(!out.exists());
    let partial = partial_path(&out);
    let whole = expected(&tokens()).len() as u64 + 8;
    assert_eq!(fs::metadata(&partial).unwrap().len(), whole);
    assert_eq!(fake.calls(), 1);

    let resumed = world.run(&fake, &out, tokens(), Options::default()).unwrap();
    assert!(resumed.finished);
    assert_eq!(fake.calls(), CHUNKS, "only the two chunks left were forced");
    assert_eq!(fs::read(&out).unwrap(), expected(&tokens()));
    assert!(!partial.exists());
    world.finish();
}

/// A chunk whose run fails is not recorded: the partial file keeps the count of the chunks before it, and a run that
/// starts again redoes that chunk. Catches a count recorded before the chunk's rows are whole.
#[test]
fn a_chunk_that_fails_is_redone_after_the_chunks_before_it() {
    let directory = tempfile::tempdir().unwrap();
    let out = out_path(directory.path());
    let world = World::start();
    let failing = Arc::new(Fake {
        calls: AtomicUsize::new(0),
        fail_call: Some(1),
        refuse_call: None,
        short_rows: false,
    });
    let error = world.run(&failing, &out, tokens(), Options::default()).unwrap_err();
    assert!(error.to_string().contains("the device failed"), "{error}");
    assert!(!out.exists());
    let bytes = fs::read(partial_path(&out)).unwrap();
    assert_eq!(&bytes[bytes.len() - 8..], &1u64.to_le_bytes());

    let fake = Fake::new();
    let resumed = world.run(&fake, &out, tokens(), Options::default()).unwrap();
    assert!(resumed.finished);
    assert_eq!(fake.calls(), CHUNKS - 1);
    assert_eq!(fs::read(&out).unwrap(), expected(&tokens()));
    world.finish();
}

/// Catches a writer that takes up a partial file of other tokens, or touches it before it has checked.
#[test]
fn a_partial_file_of_other_tokens_is_refused_and_left_alone() {
    let directory = tempfile::tempdir().unwrap();
    let out = out_path(directory.path());
    let world = World::start();
    let fake = Fake::new();
    world
        .run(&fake, &out, tokens(), Options { stop_after: Some(1) })
        .unwrap();
    let before = fs::read(partial_path(&out)).unwrap();
    let mut other = tokens();
    other[2][5] = (other[2][5] + 1) % VOCABULARY as u32;
    let calls = fake.calls();
    let error = world.run(&fake, &out, other, Options::default()).unwrap_err();
    assert!(matches!(error, ReferenceError::Mismatch(_)), "{error}");
    assert_eq!(fake.calls(), calls);
    assert_eq!(fs::read(partial_path(&out)).unwrap(), before);
    world.finish();
}

/// Catches a writer that overwrites a finished reference.
#[test]
fn an_existing_output_is_refused_and_untouched() {
    let directory = tempfile::tempdir().unwrap();
    let out = out_path(directory.path());
    fs::write(&out, b"keep me").unwrap();
    let world = World::start();
    let fake = Fake::new();
    let error = world.run(&fake, &out, tokens(), Options::default()).unwrap_err();
    assert!(matches!(error, ReferenceError::Refused(_)), "{error}");
    assert_eq!(fs::read(&out).unwrap(), b"keep me");
    assert_eq!(fake.calls(), 0);
    assert!(!partial_path(&out).exists());
    world.finish();
}

/// Catches a run that waits for a chunk the engine never started, or reports success for it.
#[test]
fn an_engine_that_does_not_start_ends_the_run_with_its_reason() {
    let directory = tempfile::tempdir().unwrap();
    let out = out_path(directory.path());
    let world = World::start();
    let fake = Arc::new(Fake {
        calls: AtomicUsize::new(0),
        fail_call: None,
        refuse_call: Some(1),
        short_rows: false,
    });
    let error = world.run(&fake, &out, tokens(), Options::default()).unwrap_err();
    assert!(error.to_string().contains("no sequence is free"), "{error}");
    assert!(!out.exists());
    world.finish();
}

/// Catches a row of the wrong size being written as a record.
#[test]
fn a_row_of_the_wrong_size_fails_the_chunk() {
    let directory = tempfile::tempdir().unwrap();
    let out = out_path(directory.path());
    let world = World::start();
    let fake = Arc::new(Fake {
        calls: AtomicUsize::new(0),
        fail_call: None,
        refuse_call: None,
        short_rows: true,
    });
    let error = world.run(&fake, &out, tokens(), Options::default()).unwrap_err();
    assert!(error.to_string().contains("bytes"), "{error}");
    world.finish();
}

#[test]
fn stopping_before_the_end_at_the_last_chunk_still_finishes() {
    let directory = tempfile::tempdir().unwrap();
    let out = out_path(directory.path());
    let world = World::start();
    let fake = Fake::new();
    let written = world
        .run(
            &fake,
            &out,
            tokens(),
            Options {
                stop_after: Some(CHUNKS),
            },
        )
        .unwrap();
    assert!(written.finished);
    assert_eq!(fs::read(&out).unwrap(), expected(&tokens()));
    world.finish();
}
