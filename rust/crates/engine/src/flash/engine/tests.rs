//! The engine on the real artifact: a proof run, `#[ignore]`d, that a short prompt generates the same tokens twice.
//! Run it under a shared grant: `cargo test -p euhedral-inference-engine --lib -- --ignored flash::engine::`. The
//! artifact comes from `euhedral.qwen4.artifact` and the library from `euhedral.cuda.library`.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]
#![allow(clippy::print_stdout, reason = "a proof run reports what it generated")]
#![allow(
    clippy::disallowed_methods,
    reason = "a test waits on the hand-off of the start and of the outcome"
)]

use std::mem;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use euhedral_inference_settings::{ModelSettings, Settings};

use super::FlashEngine;
use crate::flash::startup::start;
use crate::generation::{
    CancelHandle, FinishReason, GenerationListener, GenerationOutcome, GenerationRequest, QuantumEvent, Speculation,
};
use crate::handoff::Handoff;
use crate::sampling::SamplingConfig;

/// The longest the proof run waits for the load.
const WAIT: Duration = Duration::from_secs(20 * 60);

/// The longest it waits for one generation: a hang guard only, far above what a debug build needs.
const GENERATION: Duration = Duration::from_secs(10 * 60);

/// The prompt: a one-turn chat with an empty think block (`<|im_start|>user\nWhat is 17 times 23?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`),
/// as the Flash-Next tokenizer encodes it (the `prompt_ids` of the greedy reference's last entry).
const PROMPT: [u32; 22] = [
    248_045, 846, 198, 3710, 369, 220, 16, 22, 2942, 220, 17, 18, 30, 248_046, 198, 248_045, 74_455, 198, 248_068, 271,
    248_069, 271,
];

/// The model's generation end tokens: `<|endoftext|>` and `<|im_end|>`.
const STOP: [u32; 2] = [248_044, 248_046];

struct Recorder {
    tokens: Vec<u32>,
    done: Arc<Handoff<(Vec<u32>, GenerationOutcome)>>,
}

impl GenerationListener for Recorder {
    fn prefill_done(&mut self, _at: Instant) {}

    fn token(&mut self, id: u32, _at: Instant) {
        self.tokens.push(id);
    }

    fn quantum(&mut self, _event: QuantumEvent) {}

    fn finished(&mut self, outcome: GenerationOutcome) {
        self.done.send((mem::take(&mut self.tokens), outcome));
    }
}

fn generate(engine: &FlashEngine, wanted: u32, stops: &[u32]) -> (Vec<u32>, GenerationOutcome) {
    let done = Handoff::new();
    let request = GenerationRequest {
        prompt: PROMPT.to_vec(),
        max_tokens: wanted,
        stop_tokens: stops.to_vec(),
        speculation: Speculation::Off,
        sampling: SamplingConfig::greedy(0),
        constraint: None,
        listener: Box::new(Recorder {
            tokens: Vec::new(),
            done: Arc::clone(&done),
        }),
    };
    engine.generate(request, CancelHandle::default()).expect("admitted");
    done.wait(GENERATION).expect("the generation never finished")
}

#[test]
#[ignore = "needs a GPU, the native library and the Flash-Next artifact"]
fn a_short_prompt_generates_the_same_tokens_twice() {
    let Some(library) = test_support::require_property("euhedral.cuda.library") else {
        return;
    };
    let Some(artifact) = test_support::require_property("euhedral.qwen4.artifact") else {
        return;
    };
    let mut settings = Settings::with_model(ModelSettings {
        artifact: PathBuf::from(artifact),
        name: "test".to_string(),
    });
    settings.context.max_tokens = 4096;
    let (owner, ready) = start(&settings, &PathBuf::from(library), None);
    let model = ready
        .wait(WAIT)
        .expect("the start never reported")
        .expect("the model starts");
    println!("device plan: {:?}", model.plan.device);
    let engine = FlashEngine::new(&model).expect("the sequence is allocated");
    let (first, outcome) = generate(&engine, 16, &STOP);
    println!("first: {first:?} {:?} in {:?}", outcome.reason, outcome.elapsed);
    let (second, _) = generate(&engine, 16, &STOP);
    println!("second: {second:?}");
    // A stop token ends the run at its first sighting, and is the last token reported.
    let stop = first[first.len() / 2];
    let at = first
        .iter()
        .position(|token| *token == stop)
        .expect("the token is in the run");
    let (stopped, outcome) = generate(&engine, 16, &[stop]);
    assert_eq!(outcome.reason, FinishReason::Stop, "{stopped:?}");
    assert_eq!(stopped, first[..=at], "the run did not end at the stop token");
    assert!(!first.is_empty(), "no token was generated");
    assert_eq!(first, second, "the same prompt generated different tokens");
    assert!(engine.stop(Duration::from_secs(60)), "the engine did not go idle");
    assert!(engine.close(), "the engine kept its sequence");
    model.shutdown(owner, true);
}
