//! DFlash2 speculative decoding on a GPU. The tests are `#[ignore]`d and run locally (each loads the 16 GB artifact).
//! The artifact comes from `euhedral.qwen.dflash2-artifact` (`EUHEDRAL_TEST_QWEN_DFLASH2_ARTIFACT`), the library from
//! `euhedral.cuda.library` and the golden directory from `euhedral.rust.goldens`.
//!
//! Two comparisons prove the path. The engine's plain greedy decode (`Speculation::Off`) equals the tokens the Java
//! engine generated on the card (`greedy-dflash2.txt`), and the drafted decode (`Speculation::Artifact`) equals the
//! plain one token for token: no Java golden exists with the drafter on, so that part is a self-comparison.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]
#![allow(
    clippy::print_stdout,
    reason = "a test reports the histogram and the counts it measured"
)]
#![allow(
    clippy::disallowed_methods,
    reason = "no-wait, blocking I/O and waiting calls: a test reads files and waits on hand-offs and the device"
)]

use std::fs;
use std::mem;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::common;
use euhedral_inference_settings::{ByteSize, ModelSettings, Settings};

use common::Card;
use euhedral_inference_engine::generation::{
    FinishReason, GenerationListener, GenerationOutcome, GenerationRequest, QuantumEvent, Speculation, Verification,
};
use euhedral_inference_engine::sampling::SamplingConfig;
use euhedral_inference_engine::{EngineRuntime, Handoff};

/// The longest a device test waits for one hand-off; past it the test fails.
const WAIT: Duration = Duration::from_secs(20 * 60);

/// Tokens the golden holds.
const GENERATED: u32 = 128;

/// Copies of the golden prompt that make a prompt of several prefill chunks.
const LONG_COPIES: usize = 40;

struct Machine {
    artifact: PathBuf,
    library: PathBuf,
    goldens: PathBuf,
}

fn machine() -> Option<Machine> {
    let library = PathBuf::from(test_support::require_property("euhedral.cuda.library")?);
    let artifact = PathBuf::from(test_support::require_property("euhedral.qwen.dflash2-artifact")?);
    let goldens = PathBuf::from(test_support::require_property("euhedral.rust.goldens")?);
    if !artifact.is_file() {
        println!("skipped: euhedral.qwen.dflash2-artifact names no file");
        return None;
    }
    if !goldens.is_dir() {
        println!("skipped: euhedral.rust.goldens names no directory");
        return None;
    }
    Some(Machine {
        artifact,
        library,
        goldens,
    })
}

fn start(machine: &Machine) -> (Card, Arc<EngineRuntime>) {
    let mut settings = Settings::with_model(ModelSettings {
        artifact: machine.artifact.clone(),
        name: "test".to_string(),
    });
    settings.context.max_tokens = 4096;
    // A prefix cache hit would restore the prompt of the first run into the second.
    settings.prefix_cache.size = ByteSize(0);
    common::engine(&settings, &machine.library)
}

/// The prompt and the generated tokens of the golden.
fn golden(machine: &Machine) -> (Vec<u32>, Vec<u32>) {
    let text = fs::read_to_string(machine.goldens.join("greedy-dflash2.txt")).unwrap();
    let line = |key: &str| -> Vec<u32> {
        let line = text
            .lines()
            .find_map(|line| line.strip_prefix(key))
            .unwrap_or_else(|| panic!("the golden has a `{key}` line"));
        line.split_whitespace().map(|id| id.parse().unwrap()).collect()
    };
    (line("prompt "), line("generated "))
}

struct Recording {
    tokens: Vec<u32>,
    verifications: Vec<Verification>,
    outcome: GenerationOutcome,
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

fn generate(runtime: &EngineRuntime, prompt: &[u32], speculation: Speculation) -> Recording {
    let done = Handoff::new();
    let waiting = Arc::clone(&done);
    runtime
        .generate(GenerationRequest {
            prompt: prompt.to_vec(),
            max_tokens: GENERATED,
            stop_tokens: Vec::new(),
            speculation,
            sampling: SamplingConfig::greedy(0),
            constraint: None,
            listener: Box::new(Recorder {
                tokens: Vec::new(),
                verifications: Vec::new(),
                done: Some(done),
            }),
        })
        .expect("the request is admitted");
    waiting.wait(WAIT).expect("the generation reports")
}

/// The length of the common prefix of two token lists.
fn match_length(left: &[u32], right: &[u32]) -> usize {
    left.iter().zip(right).take_while(|(a, b)| a == b).count()
}

/// `accepted[a]` counts verifications that confirmed `a` drafts.
fn histogram(recording: &Recording, depth: usize) -> Vec<u64> {
    let mut counts = vec![0u64; depth + 1];
    for verification in &recording.verifications {
        counts[verification.accepted] += 1;
    }
    counts
}

fn mean_accepted(counts: &[u64]) -> f64 {
    let total: u64 = counts.iter().sum();
    let sum: u64 = counts.iter().enumerate().map(|(accepted, n)| accepted as u64 * n).sum();
    if total == 0 { 0.0 } else { sum as f64 / total as f64 }
}

/// Plain greedy decode equals the Java golden, and the drafted decode equals plain greedy, on the golden prompt (one
/// prefill chunk) and on a prompt of several chunks (the drafter's context follows the intermediate chunks).
#[test]
#[ignore = "needs a GPU, the native library and the DFlash2 artifact"]
fn dflash2_output_equals_greedy() {
    let Some(machine) = machine() else { return };
    let (prompt, expected) = golden(&machine);
    assert_eq!(
        expected.len(),
        GENERATED as usize,
        "the golden holds {GENERATED} tokens"
    );
    let (_card, runtime) = start(&machine);
    let depth = runtime.plan().draft_depth;
    assert!(depth > 0, "the artifact drafts");

    let plain = generate(&runtime, &prompt, Speculation::Off);
    println!(
        "plain greedy against the golden: {} of {} tokens match",
        match_length(&plain.tokens, &expected),
        expected.len()
    );
    assert_eq!(plain.tokens, expected, "plain greedy decode is the Java golden");

    let long_prompt: Vec<u32> = prompt
        .iter()
        .copied()
        .cycle()
        .take(prompt.len() * LONG_COPIES)
        .collect();
    for (name, prompt) in [("golden prompt", &prompt), ("long prompt", &long_prompt)] {
        let reference = if name == "golden prompt" {
            plain.tokens.clone()
        } else {
            generate(&runtime, prompt, Speculation::Off).tokens
        };
        let drafted = generate(&runtime, prompt, Speculation::Artifact);
        let counts = histogram(&drafted, depth);
        println!(
            "{name} ({} prompt tokens): drafted against plain greedy {} of {} tokens match, {} verifications, accepted \
             {counts:?} (mean {:.2})",
            prompt.len(),
            match_length(&drafted.tokens, &reference),
            reference.len(),
            drafted.verifications.len(),
            mean_accepted(&counts),
        );
        assert_eq!(
            drafted.tokens, reference,
            "{name}: the drafted tokens are greedy decode's (the generation ended with {:?})",
            drafted.outcome.reason
        );
        assert_eq!(drafted.outcome.generated, GENERATED as usize);
        assert!(
            drafted.verifications.len() < drafted.tokens.len(),
            "{name}: the drafter accepts drafts: {} verifications for {} tokens",
            drafted.verifications.len(),
            drafted.tokens.len()
        );
    }
}
