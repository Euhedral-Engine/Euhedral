//! The prompt prefix cache in the engine, on a GPU with the RCO artifact: a request restores the checkpoints an earlier
//! one captured and then produces the tokens a cold run produces, bit for bit. `#[ignore]`d; run one at a time in a
//! memory-capped scope (each loads the 12 GB artifact), with the CUDA library named by `euhedral.cuda.library` and the
//! artifact by `euhedral.qwen.rco-artifact`; a test skips without them. Every wait is bounded.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]
#![allow(clippy::print_stdout, reason = "a test reports what it measured")]
#![allow(
    clippy::disallowed_methods,
    clippy::disallowed_types,
    reason = "the test thread waits for hand-offs against a deadline and shares a flag with the hook it gives the engine"
)]

use std::mem;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use euhedral_inference_engine::generation::{
    AdmissionError, CancelHandle, FinishReason, GenerationListener, GenerationOutcome, GenerationRequest, QuantumEvent,
    Speculation,
};
use euhedral_inference_engine::sampling::SamplingConfig;
use euhedral_inference_engine::{EngineRuntime, Handoff, RuntimeOwner};
use euhedral_inference_settings::{ByteSize, ModelSettings, Settings};

/// The longest a device test waits for one hand-off; past it the test fails.
const WAIT: Duration = Duration::from_secs(20 * 60);
/// The tokens each request generates.
const GENERATED: u32 = 24;
/// The prompts' length: checkpoints fall at 2048 (the interval) and 2560 (the last chunk boundary before the end).
const LENGTH: usize = 2600;

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

fn settings(artifact: &Path, sequences: u32, cache: ByteSize) -> Settings {
    let mut settings = Settings::with_model(ModelSettings {
        artifact: artifact.to_path_buf(),
        name: "test".to_string(),
    });
    settings.context.max_tokens = 4096;
    settings.kernels.graphs.captures_per_view = 8;
    settings.batching.max_sequences = sequences;
    settings.prefix_cache.size = cache;
    settings
}

/// Starts the engine with `sequences` sequences and a prefix cache of `cache` bytes (0: off).
fn start(machine: &Machine, sequences: u32, cache: ByteSize) -> (Arc<EngineRuntime>, RuntimeOwner) {
    let (owner, ready) = EngineRuntime::start(&settings(&machine.artifact, sequences, cache), &machine.library);
    (
        ready
            .wait(WAIT)
            .expect("the engine reports")
            .expect("the engine starts"),
        owner,
    )
}

/// A prompt of `LENGTH` tokens whose first `shared` tokens are those of every prompt of `seed` 0 and whose others
/// are `seed`'s own.
fn prompt(seed: u64, shared: usize) -> Vec<u32> {
    (0..LENGTH)
        .map(|index| {
            let owner = if index < shared { 0 } else { seed };
            let mixed =
                (index as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ owner.wrapping_mul(0xBF58_476D_1CE4_E5B9);
            1000 + (mixed >> 33) as u32 % 50_000
        })
        .collect()
}

#[derive(Debug)]
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

    fn token(&mut self, id: u32, _at: Instant) {
        self.tokens.push(id);
    }

    fn quantum(&mut self, _event: QuantumEvent) {}

    fn finished(&mut self, outcome: GenerationOutcome) {
        let done = self.done.take().expect("a generation finishes once");
        done.send(Recording {
            tokens: mem::take(&mut self.tokens),
            outcome,
        });
    }
}

impl Drop for Recorder {
    /// A listener dropped before its generation finished reports it as failed, so the test fails at once.
    fn drop(&mut self) {
        if let Some(done) = self.done.take() {
            done.send(Recording {
                tokens: mem::take(&mut self.tokens),
                outcome: GenerationOutcome {
                    reason: FinishReason::Failed("the listener was dropped before the generation ended".to_string()),
                    prompt_tokens: 0,
                    generated: 0,
                    elapsed: Duration::ZERO,
                    cached_prompt_tokens: 0,
                },
            });
        }
    }
}

/// Starts a request and returns the hand-off its listener reports on.
fn submit(
    runtime: &EngineRuntime,
    prompt: &[u32],
    max_tokens: u32,
    sampling: SamplingConfig,
    cancel: &CancelHandle,
) -> Result<Arc<Handoff<Recording>>, AdmissionError> {
    let done = Handoff::new();
    runtime.generate_cancellable(
        GenerationRequest {
            prompt: prompt.to_vec(),
            max_tokens,
            stop_tokens: Vec::new(),
            speculation: Speculation::Off,
            sampling,
            constraint: None,
            listener: Box::new(Recorder {
                tokens: Vec::new(),
                done: Some(Arc::clone(&done)),
            }),
        },
        cancel.clone(),
    )?;
    Ok(done)
}

fn generate(runtime: &EngineRuntime, prompt: &[u32], sampling: SamplingConfig) -> Recording {
    submit(runtime, prompt, GENERATED, sampling, &CancelHandle::default())
        .expect("the request is admitted")
        .wait(WAIT)
        .expect("the generation reports")
}

fn sampled(seed: u64) -> SamplingConfig {
    SamplingConfig::new(1.0, 40, 1.0, seed, false).unwrap()
}

/// The prompts of the first test, with the sampling each is run with and the position a warm engine restores.
fn cases() -> Vec<(&'static str, Vec<u32>, SamplingConfig, usize)> {
    let a = prompt(0, LENGTH);
    let b = prompt(1, 2100);
    let c = prompt(2, 2100);
    let d = prompt(3, 1000);
    vec![
        ("A greedy, first time", a.clone(), SamplingConfig::greedy(0), 0),
        ("A greedy, again", a.clone(), SamplingConfig::greedy(0), 2560),
        ("A sampled", a, sampled(11), 2560),
        (
            "B greedy, sharing 2100 tokens",
            b.clone(),
            SamplingConfig::greedy(0),
            2048,
        ),
        ("B sampled, after B's own checkpoints", b, sampled(12), 2560),
        ("C sampled, sharing 2100 tokens", c, sampled(13), 2048),
        ("D greedy, sharing 1000 tokens", d, SamplingConfig::greedy(0), 0),
    ]
}

#[test]
#[ignore = "needs a GPU, the native library and the RCO artifact"]
fn a_restored_prefix_gives_the_tokens_of_a_cold_run() {
    let Some(machine) = machine() else { return };
    // The cold references: the cache is off.
    let (runtime, owner) = start(&machine, 1, ByteSize(0));
    let cold: Vec<Recording> = cases()
        .iter()
        .map(|(_, prompt, sampling, _)| generate(&runtime, prompt, *sampling))
        .collect();
    for recording in &cold {
        assert_eq!(recording.outcome.reason, FinishReason::Length);
        assert_eq!(recording.outcome.cached_prompt_tokens, 0, "no cache restores nothing");
        assert_eq!(recording.tokens.len(), GENERATED as usize);
    }
    owner.shutdown(runtime).expect("the engine shuts down clean");

    let (runtime, owner) = start(&machine, 1, ByteSize::gib(2));
    for ((name, prompt, sampling, restored), cold) in cases().iter().zip(&cold) {
        let warm = generate(&runtime, prompt, *sampling);
        assert_eq!(warm.outcome.reason, FinishReason::Length, "{name}");
        assert_eq!(
            warm.outcome.cached_prompt_tokens, *restored,
            "{name}: the restored position"
        );
        assert_eq!(
            warm.tokens, cold.tokens,
            "{name}: tokens after the restore equal the cold run's"
        );
        println!("{name}: restored {restored} tokens, tokens equal the cold run's");
    }
    owner.shutdown(runtime).expect("the engine shuts down clean");
}

#[test]
#[ignore = "needs a GPU, the native library and the RCO artifact"]
fn a_request_on_one_sequence_restores_what_another_captured() {
    let Some(machine) = machine() else { return };
    let (runtime, owner) = start(&machine, 2, ByteSize::gib(2));
    let shared = prompt(0, LENGTH);
    let first = generate(&runtime, &shared, SamplingConfig::greedy(0));
    assert_eq!(first.outcome.cached_prompt_tokens, 0);
    // A long decode takes the lowest idle sequence (the first, which captured the checkpoints); the next request runs
    // on the other one.
    let busy = submit(
        &runtime,
        &prompt(7, 0)[..600],
        1200,
        SamplingConfig::greedy(0),
        &CancelHandle::default(),
    )
    .expect("the long request is admitted");
    let second = submit(
        &runtime,
        &shared,
        GENERATED,
        SamplingConfig::greedy(0),
        &CancelHandle::default(),
    )
    .expect("the second sequence takes the request");
    let second = second.wait(WAIT).expect("the generation reports");
    assert_eq!(second.outcome.reason, FinishReason::Length);
    assert_eq!(
        second.outcome.cached_prompt_tokens, 2560,
        "restored from the other sequence's capture"
    );
    assert_eq!(second.tokens, first.tokens);
    let busy = busy.wait(WAIT).expect("the long request reports");
    assert_eq!(busy.outcome.reason, FinishReason::Length);
    owner.shutdown(runtime).expect("the engine shuts down clean");
}

#[test]
#[ignore = "needs a GPU, the native library and the RCO artifact"]
fn a_request_cancelled_while_its_restore_runs_leaves_the_sequence_clean() {
    let Some(machine) = machine() else { return };
    let (runtime, owner) = start(&machine, 1, ByteSize::gib(2));
    let shared = prompt(0, LENGTH);
    let first = generate(&runtime, &shared, SamplingConfig::greedy(0));
    // The hook cancels the request the moment its restore's job is in flight.
    let cancel = CancelHandle::default();
    let armed = Arc::new(AtomicBool::new(false));
    let target = Arc::new(Mutex::new(None::<CancelHandle>));
    {
        let (armed, target) = (Arc::clone(&armed), Arc::clone(&target));
        runtime.on_copy_admitted_for_tests(move |restore| {
            if restore
                && armed.swap(false, Ordering::AcqRel)
                && let Some(handle) = target.lock().unwrap().as_ref()
            {
                handle.cancel();
            }
        });
    }
    *target.lock().unwrap() = Some(cancel.clone());
    armed.store(true, Ordering::Release);
    let cancelled = submit(&runtime, &shared, GENERATED, SamplingConfig::greedy(0), &cancel)
        .expect("the request is admitted")
        .wait(WAIT)
        .expect("the generation reports");
    assert!(!armed.load(Ordering::Acquire), "the restore's job was admitted");
    assert_eq!(cancelled.outcome.reason, FinishReason::Cancelled);
    assert!(
        cancelled.tokens.is_empty(),
        "nothing was generated: {:?}",
        cancelled.tokens
    );
    // The lease came back with nothing in flight.
    let state = runtime
        .inspect_sequence(|sequence| sequence.in_flight())
        .expect("the sequence is idle once the request ended");
    assert_eq!(state, 0, "no quantum or copy still holds the sequence");
    // The next request on the sequence is unaffected: it restores the same checkpoint and generates the same tokens.
    let next = generate(&runtime, &shared, SamplingConfig::greedy(0));
    assert_eq!(next.outcome.cached_prompt_tokens, 2560);
    assert_eq!(next.tokens, first.tokens);
    owner.shutdown(runtime).expect("the engine shuts down clean");
}

#[test]
#[ignore = "needs a GPU, the native library and the RCO artifact"]
fn closing_the_engine_while_a_capture_runs_ends_within_its_bound() {
    let Some(machine) = machine() else { return };
    let (runtime, owner) = start(&machine, 1, ByteSize::gib(2));
    let started = Handoff::new();
    let once = Arc::new(AtomicBool::new(false));
    {
        let (started, once) = (Arc::clone(&started), Arc::clone(&once));
        runtime.on_copy_admitted_for_tests(move |restore| {
            if !restore && !once.swap(true, Ordering::AcqRel) {
                started.send(());
            }
        });
    }
    let done = submit(
        &runtime,
        &prompt(0, LENGTH),
        GENERATED,
        SamplingConfig::greedy(0),
        &CancelHandle::default(),
    )
    .expect("the request is admitted");
    started.wait(WAIT).expect("a capture was admitted");
    let window = Duration::from_secs(60);
    let began = Instant::now();
    let closed = owner.shutdown_within(runtime, window);
    let took = began.elapsed();
    println!("closed in {took:?}: {closed:?}");
    assert!(
        took < window + Duration::from_secs(5),
        "the close returned within its bound"
    );
    closed.expect("the engine shuts down clean with a capture in flight");
    // The request ended through its listener, once: with the engine's close, not a hang.
    let recording = done.wait(Duration::from_secs(30)).expect("the request was told");
    assert!(
        matches!(
            recording.outcome.reason,
            FinishReason::Cancelled | FinishReason::Length | FinishReason::Failed(_)
        ),
        "{:?}",
        recording.outcome.reason
    );
}
