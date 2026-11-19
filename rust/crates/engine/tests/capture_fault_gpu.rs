//! A captured quantum that fails in the middle of its body, on a GPU. The tests are `#[ignore]`d and run locally, one at a
//! time (each loads the 12 GB artifact twice), through the system arbiter:
//! `system-run --gpu shared:16 --mem 24G --est 300 --who <name> -- cargo test -p euhedral-inference-engine --test capture_fault_gpu -- --ignored --test-threads=1`.
//!
//! A test hook ends the first graph body that runs for a new key (after its first launch with work) by a refusal or a
//! panic. The run's hold must be stored in the quantum, released only after the quantum retired (its queued work ended),
//! and dropped with it; the generation fails; the sequence is free and keeps its pages; and the engine serves the next
//! request with the tokens a fresh engine selects. The artifact comes from `euhedral.qwen.rco-artifact` and the library
//! from `euhedral.cuda.library`.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]
#![allow(clippy::print_stdout, reason = "a test says why it skipped")]
#![allow(
    clippy::disallowed_methods,
    reason = "no-wait, blocking I/O and waiting calls: a test waits on hand-offs"
)]

use std::mem;
use std::path::PathBuf;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

mod common;

use common::Closer;
use euhedral_inference_engine::generation::{
    FinishReason, GenerationListener, GenerationOutcome, GenerationRequest, QuantumEvent, Speculation,
};
use euhedral_inference_engine::quantum::body_fault::{BodyFault, HoldEvent};
use euhedral_inference_engine::sampling::SamplingConfig;
use euhedral_inference_engine::{EngineRuntime, Handoff};
use euhedral_inference_settings::{ByteSize, ModelSettings, Settings};

/// The longest a device test waits for one hand-off; past it the test fails.
const WAIT: Duration = Duration::from_secs(20 * 60);
/// How long the quantum's drop may trail the end of its request.
const DROP_WINDOW: Duration = Duration::from_secs(20);
/// Tokens of a generation.
const TOKENS: u32 = 24;

struct Machine {
    artifact: PathBuf,
    library: PathBuf,
}

fn machine() -> Option<Machine> {
    let library = PathBuf::from(test_support::require_property("euhedral.cuda.library")?);
    let artifact = PathBuf::from(test_support::require_property("euhedral.qwen.rco-artifact")?);
    if !artifact.is_file() {
        println!("skipped: euhedral.qwen.rco-artifact names no file");
        return None;
    }
    Some(Machine { artifact, library })
}

fn start(machine: &Machine) -> (Closer, Arc<EngineRuntime>) {
    let mut settings = Settings::with_model(ModelSettings {
        artifact: machine.artifact.clone(),
        name: "test".to_string(),
    });
    settings.context.max_tokens = 4096;
    settings.kernels.graphs.captures_per_view = 8;
    settings.prefix_cache.size = ByteSize(0);
    let (owner, ready) = EngineRuntime::start(&settings, &machine.library);
    let runtime = ready
        .wait(WAIT)
        .expect("the engine reports")
        .expect("the engine starts");
    (Closer::new(owner, &runtime), runtime)
}

fn prompt() -> Vec<u32> {
    (0..64).map(|index| 1000 + ((index * 7919) % 5000) as u32).collect()
}

struct Recording {
    tokens: Vec<u32>,
    outcome: GenerationOutcome,
}

struct Recorder {
    tokens: Vec<u32>,
    done: Option<Arc<Handoff<Recording>>>,
}

impl GenerationListener for Recorder {
    fn prefill_done(&mut self, _at: Instant) {}

    fn quantum(&mut self, _event: QuantumEvent) {}

    fn token(&mut self, id: u32, _at: Instant) {
        self.tokens.push(id);
    }

    fn finished(&mut self, outcome: GenerationOutcome) {
        let done = self.done.take().expect("a generation finishes once");
        done.send(Recording {
            tokens: mem::take(&mut self.tokens),
            outcome,
        });
    }
}

/// A greedy generation of [`TOKENS`] tokens from [`prompt`], waited for.
fn generate(runtime: &EngineRuntime) -> Recording {
    let done = Handoff::new();
    runtime
        .generate(GenerationRequest {
            prompt: prompt(),
            max_tokens: TOKENS,
            stop_tokens: Vec::new(),
            speculation: Speculation::Off,
            sampling: SamplingConfig::greedy(0),
            constraint: None,
            listener: Box::new(Recorder {
                tokens: Vec::new(),
                done: Some(Arc::clone(&done)),
            }),
        })
        .expect("the request is admitted");
    done.wait(WAIT).expect("the generation reports")
}

/// The sequence is free: nothing in flight, no selection or row leased. Returns the pages its cache holds.
fn assert_free(runtime: &EngineRuntime) -> usize {
    let (in_flight, leased, pages) = runtime
        .inspect_sequence(|sequence| {
            (
                sequence.in_flight(),
                sequence.selections.leased() + sequence.row_readbacks.leased(),
                sequence.kv.device_pages(),
            )
        })
        .expect("the generation gave the sequence back");
    assert_eq!((in_flight, leased), (0, 0), "no quantum in flight, nothing leased");
    pages
}

/// The events of the faulted quantum, once its drop has come (the owner drops it after its request ended).
fn hold_events(runtime: &EngineRuntime) -> Vec<HoldEvent> {
    let begun = Instant::now();
    loop {
        let events = runtime.body_events_for_tests().remove(0);
        if events.len() == 3 || begun.elapsed() > DROP_WINDOW {
            return events;
        }
        thread::sleep(Duration::from_millis(5));
    }
}

fn fails_and_recovers(fault: BodyFault) {
    let Some(machine) = machine() else { return };
    let (closer, runtime) = start(&machine);
    runtime.fail_next_body_for_tests(fault);
    let failed = generate(&runtime);
    assert!(
        matches!(failed.outcome.reason, FinishReason::Failed(_)),
        "{fault:?}: the generation fails, ended with {:?}",
        failed.outcome.reason
    );
    assert!(
        failed.tokens.len() < TOKENS as usize,
        "{fault:?}: the failure cut the generation"
    );
    // The hold was stored with the quantum, outlived the quantum's device work, and went with the quantum.
    assert_eq!(
        hold_events(&runtime),
        [HoldEvent::Stored, HoldEvent::Retired, HoldEvent::Released],
        "{fault:?}: the hold of the failed body"
    );
    let pages = assert_free(&runtime);
    assert!(pages > 0, "{fault:?}: the sequence holds its cache pages");

    // Its pages retired: the next request on the engine reuses them, and selects what a fresh engine selects.
    let after = generate(&runtime);
    assert_eq!(
        after.outcome.reason,
        FinishReason::Length,
        "{fault:?}: the next request"
    );
    assert_eq!(
        assert_free(&runtime),
        pages,
        "{fault:?}: the next request takes the pages back"
    );
    closer.shutdown(runtime).unwrap();

    let (closer, runtime) = start(&machine);
    let fresh = generate(&runtime);
    assert_eq!(fresh.outcome.reason, FinishReason::Length);
    assert_eq!(
        after.tokens, fresh.tokens,
        "{fault:?}: the tokens after the failure equal a fresh engine's"
    );
    closer.shutdown(runtime).unwrap();
}

#[test]
#[ignore = "needs a GPU, the native library and the RCO artifact"]
fn a_captured_quantum_refused_in_the_middle_of_its_body_releases_its_hold_after_it_retired() {
    fails_and_recovers(BodyFault::Refuse);
}

#[test]
#[ignore = "needs a GPU, the native library and the RCO artifact"]
fn a_captured_quantum_that_panics_in_the_middle_of_its_body_releases_its_hold_after_it_retired() {
    fails_and_recovers(BodyFault::Panic);
}
