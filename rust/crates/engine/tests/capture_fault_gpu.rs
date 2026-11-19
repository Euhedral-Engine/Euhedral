//! A captured quantum that fails in the middle of its body, on a GPU. The tests are `#[ignore]`d and run locally, one at a
//! time (each loads the 12 GB artifact twice), through the system arbiter:
//! `system-run --gpu shared:16 --mem 24G --est 300 --who <name> -- cargo test -p euhedral-inference-engine --test capture_fault_gpu -- --ignored --test-threads=1`.
//!
//! A test hook ends a captured quantum by a refusal or a panic: at its first sighting (a direct run), at its second (the
//! recording pass) or at a later one (a replay, after the replay queued its work). A body fails after its first launch
//! with work. The run's hold must be stored in the quantum, released only after the quantum retired (its queued work ended),
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
use euhedral_gpu::GraphStats;
use euhedral_inference_engine::generation::{
    FinishReason, GenerationListener, GenerationOutcome, GenerationRequest, QuantumEvent, Speculation,
};
use euhedral_inference_engine::quantum::body_fault::{BodyFault, HoldEvent, Sighting};
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

/// The graph cache's counters and the number of graphs it holds.
fn graphs(runtime: &EngineRuntime) -> (GraphStats, usize) {
    runtime.graph_counts_for_tests().remove(0)
}

/// What a fault is applied to, and what the test checks of the hold.
struct Case {
    fault: BodyFault,
    sighting: Sighting,
    /// Whether the body keeps a test-owned buffer in the hold of its run. A direct run's hold carries it to the quantum's
    /// drop; a recording that failed made no graph, so its buffer goes with the recording; a replay runs no body.
    keeps: bool,
}

/// Arms `case`, runs a generation that fails, and checks the hold, the pages, the cache and the next request.
///
/// Mutants of the release order each of these assertions fails on:
/// - the hold is dropped at the unwind of a refusal or a panic, before it is stored in the quantum: the count of the
///   kept buffer at the first event is 1 (a first sighting), or the first event is not `Stored` (the others);
/// - the quantum is dropped, with its hold, before it retired: `Released` comes before `Retired`, and for a direct run
///   the count at `Retired` is 1;
/// - the hold is released when the quantum's request ends, ahead of the quantum's retirement: the same two;
/// - the quantum is never dropped (its hold never released): fewer than three events, and the buffer's count never
///   returns to 1;
/// - the failed quantum is not retired, so its pages stay: the next request does not get them back, or hangs;
/// - a failed replay drops the recording of its key: the graph count falls to 0 and the next request records again.
fn fails_and_recovers(case: Case) {
    let Case { fault, sighting, keeps } = case;
    let Some(machine) = machine() else { return };
    let (closer, runtime) = start(&machine);
    let buffer = Arc::new(());
    let before = graphs(&runtime);
    assert_eq!(before.1, 0, "{fault:?} at {sighting:?}: a new engine records nothing");
    runtime.fail_next_quantum_for_tests(fault, sighting, keeps.then_some(&buffer));
    let failed = generate(&runtime);
    assert!(
        matches!(failed.outcome.reason, FinishReason::Failed(_)),
        "{fault:?} at {sighting:?}: the generation fails, ended with {:?}",
        failed.outcome.reason
    );
    assert!(
        failed.tokens.len() < TOKENS as usize,
        "{fault:?} at {sighting:?}: the failure cut the generation"
    );
    // The hold was stored with the quantum, outlived the quantum's device work, and went with the quantum.
    assert_eq!(
        hold_events(&runtime),
        [HoldEvent::Stored, HoldEvent::Retired, HoldEvent::Released],
        "{fault:?} at {sighting:?}: the hold of the failed quantum"
    );
    let held = runtime.body_held_counts_for_tests().remove(0);
    if keeps && sighting == Sighting::First {
        // The hold is not empty: the buffer of the test is alive in it at every event, until the quantum dropped.
        assert!(
            held.iter().all(|count| *count > 1),
            "{fault:?} at {sighting:?}: the kept buffer is alive until the release, counts {held:?}"
        );
    }
    if keeps {
        // And nothing keeps it after the quantum: the hold dropped with the quantum.
        let begun = Instant::now();
        while Arc::strong_count(&buffer) > 1 && begun.elapsed() < DROP_WINDOW {
            thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(
            Arc::strong_count(&buffer),
            1,
            "{fault:?} at {sighting:?}: the buffer is freed with the quantum's hold"
        );
    }
    let pages = assert_free(&runtime);
    assert!(
        pages > 0,
        "{fault:?} at {sighting:?}: the sequence holds its cache pages"
    );
    let failed_graphs = graphs(&runtime);
    match sighting {
        Sighting::Recording => {
            // The failed recording made no graph and marked nothing unrecordable: the key stays at its second
            // sighting, so a quantum that follows records it (a graph exists only if one did).
            assert_eq!(
                (failed_graphs.0.unrecordable_keys, failed_graphs.1 as u64),
                (0, failed_graphs.0.records),
                "{fault:?} at {sighting:?}: the failed recording left no graph and no mark, {:?}",
                failed_graphs.0
            );
        }
        Sighting::Replay => {
            // The failure came after the replay queued its work, so the cache gave the instance back: its recording stays.
            assert!(
                failed_graphs.1 >= 1,
                "{fault:?} at {sighting:?}: the graph the failed replay ran is still cached"
            );
            assert!(
                failed_graphs.0.replays >= 1,
                "{fault:?} at {sighting:?}: the quantum was a replay"
            );
        }
        Sighting::First => {}
    }

    // Its pages retired: the next request on the engine reuses them, and selects what a fresh engine selects.
    let after = generate(&runtime);
    assert_eq!(
        after.outcome.reason,
        FinishReason::Length,
        "{fault:?} at {sighting:?}: the next request"
    );
    assert_eq!(
        assert_free(&runtime),
        pages,
        "{fault:?} at {sighting:?}: the next request takes the pages back"
    );
    let served = graphs(&runtime);
    match sighting {
        Sighting::Replay => {
            // The cache serves the next request from the recording it kept: it recorded nothing again and replayed more.
            assert_eq!(
                served.0.records, failed_graphs.0.records,
                "{fault:?} at {sighting:?}: the cached graph served the next request, nothing was recorded again"
            );
            assert!(
                served.0.replays > failed_graphs.0.replays,
                "{fault:?} at {sighting:?}: the next request replayed the cached graph"
            );
        }
        Sighting::Recording | Sighting::First => {
            assert!(
                served.1 >= 1 && served.0.replays > 0,
                "{fault:?} at {sighting:?}: the next request recorded and replayed, {:?}",
                served.0
            );
        }
    }
    closer.shutdown(runtime).unwrap();

    let (closer, runtime) = start(&machine);
    let fresh = generate(&runtime);
    assert_eq!(fresh.outcome.reason, FinishReason::Length);
    assert_eq!(
        after.tokens, fresh.tokens,
        "{fault:?} at {sighting:?}: the tokens after the failure equal a fresh engine's"
    );
    closer.shutdown(runtime).unwrap();
}

#[test]
#[ignore = "needs a GPU, the native library and the RCO artifact"]
fn a_captured_quantum_refused_in_the_middle_of_its_body_releases_its_hold_after_it_retired() {
    fails_and_recovers(Case {
        fault: BodyFault::Refuse,
        sighting: Sighting::First,
        keeps: true,
    });
}

#[test]
#[ignore = "needs a GPU, the native library and the RCO artifact"]
fn a_captured_quantum_that_panics_in_the_middle_of_its_body_releases_its_hold_after_it_retired() {
    fails_and_recovers(Case {
        fault: BodyFault::Panic,
        sighting: Sighting::First,
        keeps: true,
    });
}

#[test]
#[ignore = "needs a GPU, the native library and the RCO artifact"]
fn a_quantum_refused_in_its_recording_pass_releases_its_hold_after_it_retired() {
    fails_and_recovers(Case {
        fault: BodyFault::Refuse,
        sighting: Sighting::Recording,
        keeps: true,
    });
}

#[test]
#[ignore = "needs a GPU, the native library and the RCO artifact"]
fn a_quantum_that_panics_in_its_recording_pass_releases_its_hold_after_it_retired() {
    fails_and_recovers(Case {
        fault: BodyFault::Panic,
        sighting: Sighting::Recording,
        keeps: true,
    });
}

#[test]
#[ignore = "needs a GPU, the native library and the RCO artifact"]
fn a_replay_refused_after_its_work_was_queued_releases_its_hold_after_it_retired() {
    fails_and_recovers(Case {
        fault: BodyFault::Refuse,
        sighting: Sighting::Replay,
        keeps: false,
    });
}

#[test]
#[ignore = "needs a GPU, the native library and the RCO artifact"]
fn a_replay_that_panics_after_its_work_was_queued_releases_its_hold_after_it_retired() {
    fails_and_recovers(Case {
        fault: BodyFault::Panic,
        sighting: Sighting::Replay,
        keeps: false,
    });
}
