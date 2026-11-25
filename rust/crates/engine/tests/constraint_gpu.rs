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
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, Instant};

mod common;

use common::{Card, Closer};
use euhedral_inference_engine::generation::{
    ConstraintError, FinishReason, GenerationListener, GenerationOutcome, GenerationRequest, QuantumEvent, Speculation,
    TokenConstraint,
};
use euhedral_inference_engine::sampling::{SamplingConfig, TokenFilter};
use euhedral_inference_engine::stages::QuantumKind;
use euhedral_inference_engine::{EngineRuntime, Handoff};
use euhedral_inference_settings::{ModelSettings, Settings};

/// The longest a device test waits for the engine to start; past it the test fails.
const WAIT: Duration = Duration::from_secs(20 * 60);

/// The longest a test waits for a generation of a few tokens: a request that hangs fails the test inside the grant.
const GENERATION_WAIT: Duration = Duration::from_secs(60);

/// The prompt: any ids inside the vocabulary will do.
const PROMPT: [u32; 6] = [3000, 1500, 2200, 800, 4100, 1200];

/// The engine every test of this binary runs on, one at a time; `None` without the card's settings.
fn start() -> Option<(Card, Arc<EngineRuntime>)> {
    let library = PathBuf::from(test_support::require_property("euhedral.cuda.library")?);
    let artifact = PathBuf::from(test_support::require_property("euhedral.qwen.rco-artifact")?);
    if !artifact.is_file() {
        return None;
    }
    Some(common::shared(|| {
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
        (Closer::new(owner, &runtime), runtime)
    }))
}

struct Recording {
    tokens: Vec<u32>,
    quanta: Vec<QuantumKind>,
    outcome: GenerationOutcome,
}

struct Recorder {
    tokens: Vec<u32>,
    quanta: Vec<QuantumKind>,
    decodes: Vec<Instant>,
    done: Option<Arc<Handoff<Recording>>>,
    /// Opened once this many decode quanta were reported.
    opens: Option<(Arc<Gate>, usize)>,
}

/// A door a slow preparation waits at until the test's listener has seen the decode beside it retire.
struct Gate {
    door: AtomicBool,
}

impl Gate {
    fn new() -> Arc<Gate> {
        Arc::new(Gate {
            door: AtomicBool::new(false),
        })
    }

    fn open(&self) {
        self.door.store(true, Ordering::Release);
    }

    /// Waits for the gate to open; `false` if it did not within the time a hung engine is given. The waiter is a
    /// worker of the engine's lattice, which `Handoff::wait` refuses, so it yields on the flag.
    fn wait(&self) -> bool {
        let start = Instant::now();
        while !self.door.load(Ordering::Acquire) {
            if start.elapsed() > GENERATION_WAIT {
                return false;
            }
            thread::yield_now();
        }
        true
    }
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
            if let Some((gate, count)) = &self.opens
                && self.decodes.len() == *count
            {
                gate.open();
            }
        }
    }

    fn finished(&mut self, outcome: GenerationOutcome) {
        let done = self.done.take().expect("a generation finishes once");
        done.send(Recording {
            tokens: mem::take(&mut self.tokens),
            quanta: mem::take(&mut self.quanta),
            outcome,
        });
    }
}

/// What a fake constraint does at its `fail_at`th step.
#[derive(Clone, Copy, PartialEq)]
enum Fault {
    None,
    AcceptErrors,
    PrepareErrors,
    /// The preparation at `fail_at` waits for the decode beside it to retire, so that the row of the decode after it
    /// waits for the mask, and then panics.
    PrepareWaitsThenPanics,
    /// The preparation at `fail_at` waits for the decode beside it to retire and succeeds: the row of the decode
    /// after it waits for the mask.
    PrepareWaits,
}

/// A constraint that allows one token a step, the next of `cycle`.
struct Cycle {
    cycle: Vec<u32>,
    step: usize,
    fault: Fault,
    fail_at: usize,
    accepted: Arc<AtomicUsize>,
    prepared: Arc<AtomicUsize>,
    /// Opened when `fail_at` decodes were reported, which the preparation at `fail_at` waits for.
    gate: Arc<Gate>,
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
            gate: Gate::new(),
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
                Fault::PrepareWaitsThenPanics => {
                    // The decode that follows retires first: its row waits for this mask.
                    assert!(self.gate.wait(), "the decode never retired while the mask was prepared");
                    panic!("a constraint that cannot prepare");
                }
                Fault::PrepareWaits if !self.gate.wait() => {
                    return Err(ConstraintError::new(
                        "the decode never retired while the mask was prepared",
                    ));
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
    stop_tokens: Vec<u32>,
    sampling: SamplingConfig,
    speculation: Speculation,
    constraint: Option<Box<dyn TokenConstraint>>,
) -> Recording {
    run_opening(
        runtime,
        max_tokens,
        stop_tokens,
        sampling,
        speculation,
        constraint,
        None,
    )
}

/// As [`run`], with the gate `opens` opened once its count of decode quanta were reported.
fn run_opening(
    runtime: &EngineRuntime,
    max_tokens: u32,
    stop_tokens: Vec<u32>,
    sampling: SamplingConfig,
    speculation: Speculation,
    constraint: Option<Box<dyn TokenConstraint>>,
    opens: Option<(Arc<Gate>, usize)>,
) -> Recording {
    let done = Handoff::new();
    runtime
        .generate(GenerationRequest {
            prompt: PROMPT.to_vec(),
            max_tokens,
            stop_tokens,
            speculation,
            sampling,
            constraint,
            listener: Box::new(Recorder {
                tokens: Vec::new(),
                quanta: Vec::new(),
                decodes: Vec::new(),
                done: Some(Arc::clone(&done)),
                opens,
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
    let Some((_card, runtime)) = start() else { return };
    let cycle = [4000, 120, 9000, 77];
    let sampled = SamplingConfig::new(1.0, 20, 0.95, 5, false).unwrap();
    for (what, sampling) in [("greedy", SamplingConfig::greedy(0)), ("sampled", sampled)] {
        let (constraint, accepted, prepared) = Cycle::new(&cycle, Fault::None, usize::MAX);
        let recording = run(
            &runtime,
            24,
            Vec::new(),
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
    let free = run(
        &runtime,
        24,
        Vec::new(),
        SamplingConfig::greedy(0),
        Speculation::Off,
        None,
    );
    assert_eq!(free.outcome.reason, FinishReason::Length);
    assert_eq!(free.tokens.len(), 24);
}

#[test]
#[ignore = "loads the artifact on the device"]
fn a_constraint_that_fails_ends_the_request_once_with_an_error_and_frees_the_sequence() {
    let Some((_card, runtime)) = start() else { return };
    let cycle = [4000, 120];
    for (what, fault) in [
        ("accept", Fault::AcceptErrors),
        ("prepare", Fault::PrepareErrors),
        (
            "a panic in prepare while the next row waits",
            Fault::PrepareWaitsThenPanics,
        ),
    ] {
        let (constraint, _, _) = Cycle::new(&cycle, fault, 3);
        let gate = Arc::clone(&constraint.gate);
        let recording = run_opening(
            &runtime,
            40,
            Vec::new(),
            SamplingConfig::greedy(0),
            Speculation::Off,
            Some(Box::new(constraint)),
            Some((gate, 3)),
        );
        assert!(
            matches!(recording.outcome.reason, FinishReason::Failed(_)),
            "{what}: {:?}",
            recording.outcome.reason
        );
        assert_eq!(
            recording.tokens.len(),
            3,
            "{what}: the tokens before the failure are reported, and nothing past it"
        );
        assert!(sequence_free(&runtime), "{what}: every lease is back");
        // The sequence takes the next request.
        let next = run(
            &runtime,
            8,
            Vec::new(),
            SamplingConfig::greedy(0),
            Speculation::Off,
            None,
        );
        assert_eq!(
            next.outcome.reason,
            FinishReason::Length,
            "{what}: the next request runs"
        );
    }
}

#[test]
#[ignore = "loads the artifact on the device"]
fn the_next_mask_is_prepared_while_the_next_decode_runs_and_a_row_that_retires_first_waits_for_it() {
    let Some((_card, runtime)) = start() else { return };
    let cycle = [4000, 120, 9000, 77];
    let slow_at = 5;
    let (constraint, _, _) = Cycle::new(&cycle, Fault::PrepareWaits, slow_at);
    let gate = Arc::clone(&constraint.gate);
    // The mask for step `slow_at` is prepared after token `slow_at - 1`, while the decode that feeds that token runs:
    // the preparation waits until that decode retired, and a row that retires first waits for the mask.
    let recording = run_opening(
        &runtime,
        12,
        Vec::new(),
        SamplingConfig::greedy(0),
        Speculation::Off,
        Some(Box::new(constraint)),
        Some((gate, slow_at)),
    );
    assert_eq!(recording.outcome.reason, FinishReason::Length);
    let expected: Vec<u32> = (0..12).map(|index| cycle[index % cycle.len()]).collect();
    assert_eq!(
        recording.tokens, expected,
        "a row that waited for its mask selects the same tokens"
    );
    assert!(sequence_free(&runtime));
}

#[test]
#[ignore = "loads the artifact on the device"]
fn a_constrained_request_ends_at_a_stop_token_it_chose_and_at_a_budget_of_one() {
    let Some((_card, runtime)) = start() else { return };
    let cycle = [4000, 120, 9000, 77];
    let (constraint, accepted, _) = Cycle::new(&cycle, Fault::None, usize::MAX);
    let stopped = run(
        &runtime,
        40,
        vec![9000],
        SamplingConfig::greedy(0),
        Speculation::Off,
        Some(Box::new(constraint)),
    );
    assert_eq!(stopped.outcome.reason, FinishReason::Stop);
    assert_eq!(
        stopped.tokens,
        vec![4000, 120, 9000],
        "the stop token is reported, and ends the request"
    );
    assert_eq!(accepted.load(Ordering::SeqCst), 3);
    assert!(sequence_free(&runtime), "stop: every lease is back");

    let (constraint, accepted, _) = Cycle::new(&cycle, Fault::None, usize::MAX);
    let single = run(
        &runtime,
        1,
        Vec::new(),
        SamplingConfig::greedy(0),
        Speculation::Off,
        Some(Box::new(constraint)),
    );
    assert_eq!(single.outcome.reason, FinishReason::Length);
    assert_eq!(single.tokens, vec![4000]);
    assert_eq!(accepted.load(Ordering::SeqCst), 1);
    assert!(sequence_free(&runtime), "one token: every lease is back");
}
