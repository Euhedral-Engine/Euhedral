//! The prompt prefix cache for requests that draft with the DFlash2 drafter, on a GPU: a request that hits a prefix
//! restores the drafter's rings with the target's state, so the drafter's context after the hit is the one a cold
//! prefill leaves, and the request produces the tokens and the verifications of a cold run. `#[ignore]`d; it loads the
//! 16 GB artifact, so run it alone in a memory-capped scope, with the CUDA library named by `euhedral.cuda.library` and
//! the artifact by `euhedral.qwen.dflash2-artifact` (`EUHEDRAL_TEST_QWEN_DFLASH2_ARTIFACT`); it skips without them.
//! Both comparisons are self-comparisons, with no Java golden.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]
#![allow(clippy::print_stdout, reason = "a test reports what it measured")]
#![allow(
    clippy::disallowed_methods,
    reason = "no-wait, blocking I/O and waiting calls: a test waits on hand-offs and the device"
)]

use crate::typed;

use std::mem;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::common;

use euhedral_inference_engine::generation::{
    FinishReason, GenerationListener, GenerationOutcome, GenerationRequest, QuantumEvent, Speculation, Verification,
};
use euhedral_inference_engine::sampling::SamplingConfig;
use euhedral_inference_engine::{EngineRuntime, Handoff};
use euhedral_inference_settings::{ByteSize, ModelSettings, Settings};
use typed::Typed;

/// The longest a device test waits for one hand-off; past it the test fails.
const WAIT: Duration = Duration::from_secs(20 * 60);
/// The tokens each request generates.
const GENERATED: u32 = 96;
/// The prompt's length: a checkpoint falls at 1024, and the prompt with its generated tokens stays inside the
/// drafter's window, so no row of the rings is overwritten by a later position.
const LENGTH: usize = 1300;
/// The position the second run of the prompt is expected to restore.
const RESTORED: usize = 1024;

struct Machine {
    artifact: PathBuf,
    library: PathBuf,
}

fn machine() -> Option<Machine> {
    let library = PathBuf::from(test_support::require_property("euhedral.cuda.library")?);
    let artifact = PathBuf::from(test_support::require_property("euhedral.qwen.dflash2-artifact")?);
    if !artifact.is_file() {
        println!("skipped: euhedral.qwen.dflash2-artifact names no file");
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
    // One sequence: the rings read back are the only sequence's.
    settings.batching.max_sequences = 1;
    settings.prefix_cache.size = cache;
    settings
}

/// A prompt of `length` tokens that repeats a short pattern, so the model continues it and the drafter has something
/// to predict; `seed` makes the pattern its own.
fn prompt(seed: u64, length: usize) -> Vec<u32> {
    (0..length)
        .map(|index| {
            let mixed = ((index % 37) as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                ^ seed.wrapping_mul(0xBF58_476D_1CE4_E5B9);
            1000 + (mixed >> 33) as u32 % 50_000
        })
        .collect()
}

struct Recording {
    tokens: Vec<u32>,
    verifications: Vec<Verification>,
    outcome: GenerationOutcome,
}

impl Recording {
    /// Per verification, the drafts it checked and the drafts it confirmed.
    fn counts(&self) -> Vec<(usize, usize)> {
        self.verifications.iter().map(|v| (v.accepted, v.drafted)).collect()
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
                verifications: mem::take(&mut self.verifications),
                outcome: GenerationOutcome {
                    reason: FinishReason::Failed("the listener was dropped before the generation finished".to_string()),
                    prompt_tokens: 0,
                    generated: 0,
                    elapsed: Duration::ZERO,
                    cached_prompt_tokens: 0,
                },
            });
        }
    }
}

fn generate(runtime: &EngineRuntime, prompt: &[u32]) -> Recording {
    let done = Handoff::new();
    runtime
        .generate(GenerationRequest {
            prompt: prompt.to_vec(),
            max_tokens: GENERATED,
            stop_tokens: Vec::new(),
            speculation: Speculation::Artifact,
            sampling: SamplingConfig::greedy(0),
            constraint: None,
            listener: Box::new(Recorder {
                tokens: Vec::new(),
                verifications: Vec::new(),
                done: Some(Arc::clone(&done)),
            }),
        })
        .expect("the request is admitted");
    let recording = done.wait(WAIT).expect("the generation reports");
    assert_eq!(recording.outcome.reason, FinishReason::Length);
    assert_eq!(recording.tokens.len(), GENERATED as usize);
    recording
}

/// The drafter's rings of the sequence, rows `0..rows` of each layer's keys then values.
fn ring_rows(runtime: &EngineRuntime, lane: &Arc<Typed>, rows: usize) -> Vec<Vec<u8>> {
    let lane = Arc::clone(lane);
    let window = runtime
        .config()
        .dflash2
        .as_ref()
        .expect("the artifact has a drafter")
        .sliding_window;
    runtime
        .inspect_sequence(move |sequence| {
            let rings = sequence.device.scratch.drafter().expect("the engine drafts");
            let row_bytes = rings.ring_size() as usize / window;
            (0..rings.layers())
                .flat_map(|layer| [rings.ring_keys(layer), rings.ring_values(layer)])
                .map(|ring| lane.download(ring, rows * row_bytes))
                .collect()
        })
        .expect("the generation gave the sequence back")
}

#[test]
#[ignore = "needs a GPU, the native library and the DFlash2 artifact"]
fn a_prefix_hit_restores_the_drafters_rings_and_drafts_as_a_cold_run() {
    let Some(machine) = machine() else { return };
    let shared = prompt(0, LENGTH);

    // The cold reference: the cache is off.
    let (card, runtime) = common::engine(&settings(&machine.artifact, ByteSize(0)), &machine.library);
    let window = runtime
        .config()
        .dflash2
        .as_ref()
        .expect("the artifact has a drafter")
        .sliding_window;
    assert!(
        LENGTH + GENERATED as usize <= window,
        "the run stays inside the drafter's window of {window}"
    );
    let lane = Arc::new(Typed::plain(runtime.library(), 1, 4));
    let cold = generate(&runtime, &shared);
    assert_eq!(cold.outcome.cached_prompt_tokens, 0, "no cache restores nothing");
    let cold_rings = ring_rows(&runtime, &lane, RESTORED);
    drop(lane);
    drop(runtime);
    drop(card);
    let _card = common::card();

    let (owner, runtime) = common::start_engine(&settings(&machine.artifact, ByteSize::gib(2)), &machine.library);
    let lane = Arc::new(Typed::plain(runtime.library(), 1, 4));
    let first = generate(&runtime, &shared);
    assert_eq!(first.outcome.cached_prompt_tokens, 0);
    assert_eq!(first.tokens, cold.tokens);
    // Another prompt leaves its own rings in the sequence's buffers, so a restore that did not copy the drafter's
    // state would find the wrong rows there (the same prompt's would pass for the restored ones).
    generate(&runtime, &prompt(99, 900));
    let warm = generate(&runtime, &shared);
    assert_eq!(
        warm.outcome.cached_prompt_tokens, RESTORED,
        "the prompt hits the checkpoint"
    );
    assert_eq!(warm.tokens, cold.tokens, "tokens after the hit equal the cold run's");
    assert_eq!(
        warm.counts(),
        cold.counts(),
        "every verification after the hit drafts and accepts as the cold run's"
    );
    let warm_rings = ring_rows(&runtime, &lane, RESTORED);
    assert_eq!(warm_rings.len(), cold_rings.len());
    for (ring, (warm, cold)) in warm_rings.iter().zip(&cold_rings).enumerate() {
        assert!(
            warm == cold,
            "ring {ring}: the restored rows differ from the cold prefill's"
        );
    }
    drop(lane);
    owner.shutdown(runtime).expect("the engine shuts down clean");
}
