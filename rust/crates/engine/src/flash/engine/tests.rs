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
    CancelHandle, GenerationListener, GenerationOutcome, GenerationRequest, QuantumEvent, Speculation,
};
use crate::handoff::Handoff;
use crate::sampling::SamplingConfig;

/// The longest the proof run waits for the load.
const WAIT: Duration = Duration::from_secs(20 * 60);

/// The longest it waits for one generation.
const GENERATION: Duration = Duration::from_secs(90);

/// The prompt: a one-turn chat opening, as token ids.
const PROMPT: [u32; 9] = [151_644, 872, 198, 9_707, 151_645, 198, 151_644, 77_091, 198];

struct Stderr;
impl log::Log for Stderr {
    fn enabled(&self, _m: &log::Metadata<'_>) -> bool { true }
    fn log(&self, r: &log::Record<'_>) { eprintln!("LOG {} {}: {}", r.level(), r.target(), r.args()); }
    fn flush(&self) {}
}
static STDERR: Stderr = Stderr;

struct Recorder {
    tokens: Vec<u32>,
    done: Arc<Handoff<(Vec<u32>, GenerationOutcome)>>,
}

impl GenerationListener for Recorder {
    fn prefill_done(&mut self, _at: Instant) {
        println!("prefill done");
    }

    fn token(&mut self, id: u32, _at: Instant) {
        println!("token {id}");
        self.tokens.push(id);
    }

    fn quantum(&mut self, event: QuantumEvent) {
        println!("quantum {event:?}");
    }

    fn finished(&mut self, outcome: GenerationOutcome) {
        self.done.send((mem::take(&mut self.tokens), outcome));
    }
}

fn generate(engine: &FlashEngine, wanted: u32) -> (Vec<u32>, GenerationOutcome) {
    let done = Handoff::new();
    let request = GenerationRequest {
        prompt: PROMPT.to_vec(),
        max_tokens: wanted,
        stop_tokens: Vec::new(),
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
    let _ = log::set_logger(&STDERR);
    log::set_max_level(log::LevelFilter::Debug);
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
    println!("model started");
    let engine = FlashEngine::new(&model).expect("the sequence is allocated");
    println!("engine built");
    let (first, outcome) = generate(&engine, 16);
    println!("first: {first:?} {:?} in {:?}", outcome.reason, outcome.elapsed);
    let (second, _) = generate(&engine, 16);
    println!("second: {second:?}");
    assert!(!first.is_empty(), "no token was generated");
    assert_eq!(first, second, "the same prompt generated different tokens");
    assert!(engine.stop(Duration::from_secs(60)), "the engine did not go idle");
    engine.close();
    model.shutdown(owner);
}
