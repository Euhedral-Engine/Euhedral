//! Constrained requests on a GPU: the host selects the best allowed token at each step, the request never speculates,
//! and a constraint that fails ends the request with an error and leaves the sequence free. The tests are `#[ignore]`d
//! and run locally (each loads the 12 GB artifact), through the system arbiter:
//! `system-run --gpu shared:14 --mem 24G --est 600 --who <name> -- cargo test -p euhedral-inference-engine --test constraint_gpu -- --ignored --test-threads=1`.
//!
//! The artifact comes from `euhedral.qwen.rco-artifact` and the library from `euhedral.cuda.library`.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]
#![allow(
    clippy::disallowed_methods,
    reason = "no-wait, blocking I/O and waiting calls: a test waits on hand-offs and the device"
)]

use std::mem;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::thread::sleep;
use std::time::{Duration, Instant};

use euhedral_inference_engine::generation::{
    ConstraintError, FinishReason, GenerationListener, GenerationOutcome, GenerationRequest, QuantumEvent, Speculation,
    TokenConstraint,
};
use euhedral_inference_engine::sampling::{SamplingConfig, TokenFilter};
use euhedral_inference_engine::stages::QuantumKind;
use euhedral_inference_engine::{EngineRuntime, Handoff, RuntimeOwner};
use euhedral_inference_settings::{ModelSettings, Settings};

/// The longest a device test waits for the engine to start; past it the test fails.
const WAIT: Duration = Duration::from_secs(20 * 60);

/// The longest a test waits for a generation of a few tokens: a request that hangs fails the test inside the grant.
const GENERATION_WAIT: Duration = Duration::from_secs(60);

/// The prompt: any ids inside the vocabulary will do.
const PROMPT: [u32; 6] = [3000, 1500, 2200, 800, 4100, 1200];

fn start() -> Option<(Arc<EngineRuntime>, RuntimeOwner)> {
    let library = PathBuf::from(test_support::require_property("euhedral.cuda.library")?);
    let artifact = PathBuf::from(test_support::require_property("euhedral.qwen.rco-artifact")?);
    if !artifact.is_file() {
        return None;
    }
    let mut settings = Settings::with_model(ModelSettings {
        artifact,
        name: "test".to_string(),
    });
    settings.context.max_tokens = 256;
    settings.kernels.graphs.captures_per_view = 8;
    let (owner, ready) = EngineRuntime::start(&settings, &library);
    let runtime = ready
        .wait(WAIT)
        .expect("the engine reports")
        .expect("the engine starts");
    Some((runtime, owner))
}

struct Recording {
    tokens: Vec<u32>,
    quanta: Vec<QuantumKind>,
    /// When each decode quantum entered the engine.
    decodes: Vec<Instant>,
    outcome: GenerationOutcome,
}

struct Recorder {
    tokens: Vec<u32>,
    quanta: Vec<QuantumKind>,
    decodes: Vec<Instant>,
    done: Option<Arc<Handoff<Recording>>>,
}

impl GenerationListener for Recorder {
    fn prefill_done(&mut self, _at: Instant) {}

    fn token(&mut self, id: u32, _at: Instant) {
        self.tokens.push(id);
    }

    fn quantum(&mut self, event: QuantumEvent) {
        self.quanta.push(event.kind);
        if event.kind == QuantumKind::Decode {
            self.decodes.push(event.admitted);
        }
    }

    fn finished(&mut self, outcome: GenerationOutcome) {
        let done = self.done.take().expect("a generation finishes once");
        done.send(Recording {
            tokens: mem::take(&mut self.tokens),
            quanta: mem::take(&mut self.quanta),
            decodes: mem::take(&mut self.decodes),
            outcome,
        });
    }
}

/// How long a slow preparation takes: long enough for the decode that runs beside it to retire first.
const SLOW: Duration = Duration::from_millis(300);

/// What a fake constraint does at its `fail_at`th step.
#[derive(Clone, Copy, PartialEq)]
enum Fault {
    None,
    AcceptErrors,
    PrepareErrors,
    PrepareSleepsThenPanics,
    /// The preparation at `fail_at` takes a while and succeeds; the time it ended is kept.
    PrepareSleeps,
}

/// A constraint that allows one token a step, the next of `cycle`.
struct Cycle {
    cycle: Vec<u32>,
    step: usize,
    fault: Fault,
    fail_at: usize,
    accepted: Arc<AtomicUsize>,
    prepared: Arc<AtomicUsize>,
    /// Nanoseconds from `base` to the end of the slow preparation.
    ended: Arc<AtomicU64>,
    base: Instant,
}

impl Cycle {
    fn new(cycle: &[u32], fault: Fault, fail_at: usize) -> (Cycle, Arc<AtomicUsize>, Arc<AtomicUsize>) {
        let accepted = Arc::new(AtomicUsize::new(0));
        let prepared = Arc::new(AtomicUsize::new(0));
        let constraint = Cycle {
            cycle: cycle.to_vec(),
            step: 0,
            fault,
            fail_at,
            accepted: Arc::clone(&accepted),
            prepared: Arc::clone(&prepared),
            ended: Arc::new(AtomicU64::new(0)),
            base: Instant::now(),
        };
        (constraint, accepted, prepared)
    }
}

impl TokenFilter for Cycle {
    fn allows(&self, id: u32) -> bool {
        id == self.cycle[self.step % self.cycle.len()]
    }
}

impl TokenConstraint for Cycle {
    fn prepare(&mut self) -> Result<(), ConstraintError> {
        self.prepared.fetch_add(1, Ordering::SeqCst);
        if self.step == self.fail_at {
            match self.fault {
                Fault::PrepareErrors => return Err(ConstraintError::new("no mask")),
                Fault::PrepareSleepsThenPanics => {
                    // Long enough for the decode that follows to retire first: its row waits for this mask.
                    sleep(SLOW);
                    panic!("a constraint that cannot prepare");
                }
                Fault::PrepareSleeps => {
                    sleep(SLOW);
                    let nanos = u64::try_from(self.base.elapsed().as_nanos()).unwrap_or(u64::MAX);
                    self.ended.store(nanos, Ordering::SeqCst);
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn accept(&mut self, id: u32) -> Result<(), ConstraintError> {
        if self.step == self.fail_at && self.fault == Fault::AcceptErrors {
            return Err(ConstraintError::new("refused"));
        }
        if !self.allows(id) {
            return Err(ConstraintError::new("not allowed"));
        }
        self.accepted.fetch_add(1, Ordering::SeqCst);
        self.step += 1;
        Ok(())
    }
}

fn run(
    runtime: &EngineRuntime,
    max_tokens: u32,
    sampling: SamplingConfig,
    speculation: Speculation,
    constraint: Option<Box<dyn TokenConstraint>>,
) -> Recording {
    let done = Handoff::new();
    runtime
        .generate(GenerationRequest {
            prompt: PROMPT.to_vec(),
            max_tokens,
            stop_tokens: Vec::new(),
            speculation,
            sampling,
            constraint,
            listener: Box::new(Recorder {
                tokens: Vec::new(),
                quanta: Vec::new(),
                decodes: Vec::new(),
                done: Some(Arc::clone(&done)),
            }),
        })
        .expect("the request is admitted");
    done.wait(GENERATION_WAIT).expect("the generation reports")
}

/// Quanta in flight, selections and row readbacks leased after a generation: all free.
fn sequence_free(runtime: &EngineRuntime) -> bool {
    runtime
        .inspect_sequence(|sequence| {
            sequence.in_flight() == 0 && sequence.selections.leased() == 0 && sequence.row_readbacks.leased() == 0
        })
        .expect("the generation gave the sequence back")
}

#[test]
#[ignore = "loads the artifact on the device"]
fn a_constraint_that_allows_one_token_a_step_decides_every_token_and_the_request_never_speculates() {
    let Some((runtime, owner)) = start() else { return };
    let cycle = [4000, 120, 9000, 77];
    let sampled = SamplingConfig::new(1.0, 20, 0.95, 5, false).unwrap();
    for (what, sampling) in [("greedy", SamplingConfig::greedy(0)), ("sampled", sampled)] {
        let (constraint, accepted, prepared) = Cycle::new(&cycle, Fault::None, usize::MAX);
        let recording = run(
            &runtime,
            24,
            sampling,
            Speculation::Artifact,
            Some(Box::new(constraint)),
        );
        assert_eq!(recording.outcome.reason, FinishReason::Length, "{what}");
        let expected: Vec<u32> = (0..24).map(|index| cycle[index % cycle.len()]).collect();
        assert_eq!(recording.tokens, expected, "{what}: the only allowed token each step");
        assert_eq!(
            accepted.load(Ordering::SeqCst),
            expected.len(),
            "{what}: every token accepted once, in order"
        );
        assert_eq!(
            prepared.load(Ordering::SeqCst),
            24,
            "{what}: a mask before the prompt's token and one before each next"
        );
        assert!(
            !recording.quanta.contains(&QuantumKind::Verify) && !recording.quanta.contains(&QuantumKind::Draft),
            "{what}: a constrained request does not speculate"
        );
        assert!(sequence_free(&runtime), "{what}: every lease is back");
    }
    // A request without a constraint still runs.
    let free = run(&runtime, 24, SamplingConfig::greedy(0), Speculation::Off, None);
    assert_eq!(free.outcome.reason, FinishReason::Length);
    assert_eq!(free.tokens.len(), 24);
    drop(runtime);
    drop(owner);
}

#[test]
#[ignore = "loads the artifact on the device"]
fn a_constraint_that_fails_ends_the_request_once_with_an_error_and_frees_the_sequence() {
    let Some((runtime, owner)) = start() else { return };
    let cycle = [4000, 120];
    for (what, fault) in [
        ("accept", Fault::AcceptErrors),
        ("prepare", Fault::PrepareErrors),
        (
            "a panic in prepare while the next row waits",
            Fault::PrepareSleepsThenPanics,
        ),
    ] {
        let (constraint, _, _) = Cycle::new(&cycle, fault, 3);
        let recording = run(
            &runtime,
            40,
            SamplingConfig::greedy(0),
            Speculation::Off,
            Some(Box::new(constraint)),
        );
        assert!(
            matches!(recording.outcome.reason, FinishReason::Failed(_)),
            "{what}: {:?}",
            recording.outcome.reason
        );
        assert!(
            recording.tokens.len() <= 4,
            "{what}: nothing is reported past the failure"
        );
        assert!(sequence_free(&runtime), "{what}: every lease is back");
        // The sequence takes the next request.
        let next = run(&runtime, 8, SamplingConfig::greedy(0), Speculation::Off, None);
        assert_eq!(
            next.outcome.reason,
            FinishReason::Length,
            "{what}: the next request runs"
        );
    }
    drop(runtime);
    drop(owner);
}

#[test]
#[ignore = "loads the artifact on the device"]
fn the_next_mask_is_prepared_while_the_next_decode_runs_and_a_row_that_retires_first_waits_for_it() {
    let Some((runtime, owner)) = start() else { return };
    let cycle = [4000, 120, 9000, 77];
    let slow_at = 5;
    let (constraint, _, _) = Cycle::new(&cycle, Fault::PrepareSleeps, slow_at);
    let ended = Arc::clone(&constraint.ended);
    let base = constraint.base;
    let recording = run(
        &runtime,
        12,
        SamplingConfig::greedy(0),
        Speculation::Off,
        Some(Box::new(constraint)),
    );
    assert_eq!(recording.outcome.reason, FinishReason::Length);
    let expected: Vec<u32> = (0..12).map(|index| cycle[index % cycle.len()]).collect();
    assert_eq!(
        recording.tokens, expected,
        "a row that waited for its mask selects the same tokens"
    );
    // The mask for step `slow_at` is prepared after token `slow_at - 1`, while the decode that feeds that token runs.
    let ended = Duration::from_nanos(ended.load(Ordering::SeqCst));
    assert!(ended > Duration::ZERO, "the slow preparation ran");
    let admitted = recording.decodes[slow_at - 1].duration_since(base);
    assert!(
        admitted < ended,
        "the decode ({admitted:?}) was admitted before the mask was ready ({ended:?})"
    );
    assert!(sequence_free(&runtime));
    drop(runtime);
    drop(owner);
}
