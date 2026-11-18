//! The prompt prefix cache for requests that draft with the artifact's MTP layer, on a GPU with the RCO artifact: a
//! speculative request restores the checkpoints an earlier one captured (the base state, the MTP cache below the last
//! row, and the seed row of the draft) and then produces the tokens a cold run produces, with the drafts a cold run
//! makes. `#[ignore]`d; run one at a time in a memory-capped scope (each loads the 12 GB artifact), with the CUDA library
//! named by `euhedral.cuda.library` and the artifact by `euhedral.qwen.rco-artifact`; a test skips without them. Every
//! wait is bounded.
//!
//! The verifier decides every token, so tokens alone cannot tell a restored MTP cache from a wrong one: the drafts
//! accepted can. A wrong draft state accepts few drafts.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]
#![allow(clippy::print_stdout, reason = "a test reports what it measured")]

use std::mem;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use euhedral_inference_engine::generation::{
    CancelHandle, FinishReason, GenerationListener, GenerationOutcome, GenerationRequest, QuantumEvent, Speculation,
    Verification,
};
use euhedral_inference_engine::sampling::SamplingConfig;
use euhedral_inference_engine::{EngineRuntime, Handoff, RuntimeOwner};
use euhedral_inference_settings::{ByteSize, ModelSettings, Settings};

/// The longest a device test waits for one hand-off; past it the test fails.
const WAIT: Duration = Duration::from_secs(20 * 60);
/// The tokens each request generates.
const GENERATED: u32 = 96;
/// The prompts' length: checkpoints fall at 1024 and 2048 (the interval) and 2560 (the last chunk boundary before the
/// end).
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

fn settings(artifact: &Path, cache: ByteSize) -> Settings {
    let mut settings = Settings::with_model(ModelSettings {
        artifact: artifact.to_path_buf(),
        name: "test".to_string(),
    });
    settings.context.max_tokens = 4096;
    settings.kernels.graphs.captures_per_view = 8;
    settings.prefix_cache.size = cache;
    settings
}

/// Starts the engine with a prefix cache of `cache` bytes (0: off).
fn start(machine: &Machine, cache: ByteSize) -> (Arc<EngineRuntime>, RuntimeOwner) {
    let (owner, ready) = EngineRuntime::start(&settings(&machine.artifact, cache), &machine.library);
    (
        ready
            .wait(WAIT)
            .expect("the engine reports")
            .expect("the engine starts"),
        owner,
    )
}

/// A prompt of `LENGTH` tokens whose first `shared` tokens are those of every prompt of `seed` 0 and whose others
/// are `seed`'s own. Each part repeats a short pattern, so the model continues it and the draft has something to
/// predict (random tokens are never predicted, and then no draft is accepted to tell a restored draft state by).
fn prompt(seed: u64, shared: usize) -> Vec<u32> {
    (0..LENGTH)
        .map(|index| {
            let owner = if index < shared { 0 } else { seed };
            let mixed = ((index % 37) as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                ^ owner.wrapping_mul(0xBF58_476D_1CE4_E5B9);
            1000 + (mixed >> 33) as u32 % 50_000
        })
        .collect()
}

#[derive(Debug)]
struct Recording {
    tokens: Vec<u32>,
    verifications: Vec<Verification>,
    outcome: GenerationOutcome,
}

impl Recording {
    /// The drafts the verifier confirmed, over the drafts it checked.
    fn accepted(&self) -> (usize, usize) {
        (
            self.verifications.iter().map(|v| v.accepted).sum(),
            self.verifications.iter().map(|v| v.drafted).sum(),
        )
    }
}

struct Recorder {
    tokens: Vec<u32>,
    verifications: Vec<Verification>,
    done: Option<Arc<Handoff<Recording>>>,
}

impl GenerationListener for Recorder {
    fn prefill_done(&mut self, _at: Instant) {}

    fn token(&mut self, id: u32, _at: Instant) {
        self.tokens.push(id);
    }

    fn quantum(&mut self, _event: QuantumEvent) {}

    fn verification(&mut self, event: Verification) {
        self.verifications.push(event);
    }

    fn finished(&mut self, outcome: GenerationOutcome) {
        let done = self.done.take().expect("a generation finishes once");
        done.send(Recording {
            tokens: mem::take(&mut self.tokens),
            verifications: mem::take(&mut self.verifications),
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
                verifications: Vec::new(),
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

fn generate(runtime: &EngineRuntime, prompt: &[u32], speculation: Speculation, sampling: SamplingConfig) -> Recording {
    let done = Handoff::new();
    runtime
        .generate_cancellable(
            GenerationRequest {
                prompt: prompt.to_vec(),
                max_tokens: GENERATED,
                stop_tokens: Vec::new(),
                speculation,
                sampling,
                constraint: None,
                listener: Box::new(Recorder {
                    tokens: Vec::new(),
                    verifications: Vec::new(),
                    done: Some(Arc::clone(&done)),
                }),
            },
            CancelHandle::default(),
        )
        .expect("the request is admitted");
    let recording = done.wait(WAIT).expect("the generation reports");
    assert_eq!(recording.outcome.reason, FinishReason::Length);
    assert_eq!(recording.tokens.len(), GENERATED as usize);
    recording
}

fn sampled(seed: u64) -> SamplingConfig {
    SamplingConfig::new(1.0, 40, 1.0, seed, false).unwrap()
}

/// A restored draft state drafts as a cold one does, up to the one row it re-pairs (which differs in its low bits): the
/// drafts accepted stay within a tenth of the cold run's. A draft cache that was not restored accepts far fewer.
fn drafts_as_a_cold_run(name: &str, warm: &Recording, cold: &Recording) {
    let (warm_accepted, warm_drafted) = warm.accepted();
    let (cold_accepted, cold_drafted) = cold.accepted();
    println!("{name}: accepted {warm_accepted}/{warm_drafted} warm, {cold_accepted}/{cold_drafted} cold");
    assert!(cold_accepted > 0, "{name}: the cold run accepts drafts");
    assert!(
        warm_accepted * 10 >= cold_accepted * 9,
        "{name}: the restored draft state accepts {warm_accepted} drafts, a cold run {cold_accepted}"
    );
}

/// The prompts of the first test, how each is run, and the position a warm engine restores.
fn cases() -> Vec<(&'static str, Vec<u32>, SamplingConfig, usize)> {
    let a = prompt(0, LENGTH);
    let b = prompt(1, 2100);
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
    ]
}

#[test]
#[ignore = "needs a GPU, the native library and the RCO artifact"]
fn a_speculative_restore_gives_the_tokens_and_the_drafts_of_a_cold_run() {
    let Some(machine) = machine() else { return };
    // The cold references: the cache is off.
    let (runtime, owner) = start(&machine, ByteSize(0));
    let cold: Vec<Recording> = cases()
        .iter()
        .map(|(_, prompt, sampling, _)| generate(&runtime, prompt, Speculation::Artifact, *sampling))
        .collect();
    for recording in &cold {
        assert_eq!(recording.outcome.cached_prompt_tokens, 0, "no cache restores nothing");
    }
    owner.shutdown(runtime).expect("the engine shuts down clean");

    let (runtime, owner) = start(&machine, ByteSize::gib(2));
    for ((name, prompt, sampling, restored), cold) in cases().iter().zip(&cold) {
        let warm = generate(&runtime, prompt, Speculation::Artifact, *sampling);
        assert_eq!(
            warm.outcome.cached_prompt_tokens, *restored,
            "{name}: the restored position"
        );
        assert_eq!(
            warm.tokens, cold.tokens,
            "{name}: tokens after the restore equal the cold run's"
        );
        if *restored > 0 {
            drafts_as_a_cold_run(name, &warm, cold);
        }
    }
    owner.shutdown(runtime).expect("the engine shuts down clean");
}

#[test]
#[ignore = "needs a GPU, the native library and the RCO artifact"]
fn a_plain_request_restores_the_checkpoints_of_a_speculative_one_but_not_the_reverse() {
    let Some(machine) = machine() else { return };
    let shared = prompt(0, LENGTH);
    let (runtime, owner) = start(&machine, ByteSize(0));
    let cold_plain = generate(&runtime, &shared, Speculation::Off, SamplingConfig::greedy(0));
    let cold_speculative = generate(&runtime, &shared, Speculation::Artifact, SamplingConfig::greedy(0));
    owner.shutdown(runtime).expect("the engine shuts down clean");

    // A speculative request captures checkpoints that hold the draft's state: a plain one restores them and needs none.
    let (runtime, owner) = start(&machine, ByteSize::gib(2));
    let first = generate(&runtime, &shared, Speculation::Artifact, SamplingConfig::greedy(0));
    assert_eq!(first.outcome.cached_prompt_tokens, 0);
    assert_eq!(first.tokens, cold_speculative.tokens);
    let plain = generate(&runtime, &shared, Speculation::Off, SamplingConfig::greedy(0));
    assert_eq!(
        plain.outcome.cached_prompt_tokens, 2560,
        "a plain request restores a speculative checkpoint"
    );
    assert_eq!(plain.tokens, cold_plain.tokens);
    owner.shutdown(runtime).expect("the engine shuts down clean");

    // A plain request captures checkpoints without it: a speculative one cannot use them, starts cold and stores its own.
    let (runtime, owner) = start(&machine, ByteSize::gib(2));
    let first = generate(&runtime, &shared, Speculation::Off, SamplingConfig::greedy(0));
    assert_eq!(first.outcome.cached_prompt_tokens, 0);
    assert_eq!(first.tokens, cold_plain.tokens);
    let speculative = generate(&runtime, &shared, Speculation::Artifact, SamplingConfig::greedy(0));
    assert_eq!(
        speculative.outcome.cached_prompt_tokens, 0,
        "a speculative request misses a checkpoint without the draft's state"
    );
    assert_eq!(speculative.tokens, cold_speculative.tokens);
    let again = generate(&runtime, &shared, Speculation::Artifact, SamplingConfig::greedy(0));
    assert_eq!(again.outcome.cached_prompt_tokens, 2560, "it restores its own");
    assert_eq!(again.tokens, cold_speculative.tokens);
    drafts_as_a_cold_run("speculative after plain", &again, &cold_speculative);
    owner.shutdown(runtime).expect("the engine shuts down clean");
}
