//! Device test of the corpus writer: two prompts are answered into a corpus by the engine. `#[ignore]`d like the
//! crate's other device tests: `cargo test -p euhedral-inference-quantization --test generate_gpu -- --ignored --nocapture`.
//!
//! The engine library is the `euhedral.cuda.library` property and the model the `euhedral.qwen.rco-artifact` one
//! (environment `EUHEDRAL_TEST_<KEY>` first); a missing key fails the test.
#![cfg(target_os = "linux")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]
#![allow(
    clippy::disallowed_types,
    reason = "a test file: the collected tokens are read by the test thread after the end"
)]
#![allow(
    clippy::disallowed_methods,
    reason = "no-wait, waiting call: a test thread waits on the hand-off for the run it started, and reads its files"
)]

use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use euhedral_inference_engine::generation::{
    FinishReason, GenerationListener, GenerationOutcome, GenerationRequest, QuantumEvent, Speculation,
};
use euhedral_inference_engine::sampling::SamplingConfig;
use euhedral_inference_engine::{EngineRuntime, Handoff};
use euhedral_inference_quantization::corpus::{self, Writer};
use euhedral_inference_quantization::generate::run::{Answerer, Options, start};
use euhedral_inference_quantization::generate::{Prompt, seed};
use euhedral_inference_quantization::reference::read_tokenizer;
use euhedral_inference_settings::{ByteSize, ModelSettings, SamplingSettings, Settings};

const WAIT: Duration = Duration::from_secs(20 * 60);
const MAX_TOKENS: u32 = 12;

fn property(key: &str) -> String {
    test_support::gradle_property(key).unwrap_or_else(|| panic!("set the property {key}: a device test needs it"))
}

/// Collects the tokens of one generation.
struct Collect {
    tokens: Arc<Mutex<Vec<u32>>>,
    done: Arc<Handoff<FinishReason>>,
}

impl GenerationListener for Collect {
    fn prefill_done(&mut self, _at: Instant) {}

    fn token(&mut self, id: u32, _at: Instant) {
        self.tokens.lock().unwrap().push(id);
    }

    fn quantum(&mut self, _event: QuantumEvent) {}

    fn finished(&mut self, outcome: GenerationOutcome) {
        self.done.send(outcome.reason);
    }
}

/// Two prompts, one answered greedily and one sampled (by the hash of their ids at the greedy share of a quarter), go
/// through the engine into a corpus. The greedy answer is the one the artifact decodes on its own, token for token,
/// and every sequence has the model's hidden width for each row it ran.
#[test]
#[ignore = "needs a GPU, the engine library and the RCO artifact"]
fn the_greedy_answer_is_the_artifacts_own_decode_and_every_row_has_the_models_hidden_width() {
    let library = PathBuf::from(property("euhedral.cuda.library"));
    let artifact = PathBuf::from(property("euhedral.qwen.rco-artifact"));
    let tokenizer = read_tokenizer(&artifact).unwrap();
    let stops = tokenizer.eos_ids().to_vec();
    let prompts: Vec<Prompt> = [
        ("p0", "Write one sentence about a river that runs past an old mill."),
        ("p10", "Name three colours and say which of them you like best."),
    ]
    .iter()
    .map(|(id, text)| Prompt {
        id: (*id).to_owned(),
        tokens: tokenizer.encode(text).unwrap(),
    })
    .collect();
    drop(tokenizer);

    let mut settings = Settings::with_model(ModelSettings {
        artifact: artifact.clone(),
        name: "generate".to_owned(),
    });
    settings.context.max_tokens = 512;
    settings.prefix_cache.size = ByteSize::mib(0);
    settings.batching.max_sequences = 2;
    let (owner, ready) = EngineRuntime::start(&settings, &library);
    let runtime = ready
        .wait(WAIT)
        .expect("the engine reports")
        .expect("the engine starts");
    let width = runtime.config().hidden_size;
    let vocabulary = runtime.config().vocab_size;

    // The artifact's own greedy decode of the first prompt.
    let tokens = Arc::new(Mutex::new(Vec::new()));
    let done = Handoff::new();
    runtime
        .generate(GenerationRequest {
            prompt: prompts[0].tokens.clone(),
            max_tokens: MAX_TOKENS + 1,
            stop_tokens: stops.clone(),
            speculation: Speculation::Off,
            sampling: SamplingConfig::greedy(seed("p0")),
            constraint: None,
            listener: Box::new(Collect {
                tokens: Arc::clone(&tokens),
                done: Arc::clone(&done),
            }),
        })
        .expect("the engine takes the request");
    done.wait(WAIT).expect("the decode ended");
    let own = tokens.lock().unwrap().clone();
    assert!(!own.is_empty());

    let directory = tempfile::tempdir().unwrap();
    let writer = Writer::open(
        directory.path(),
        width,
        Arc::clone(runtime.io()),
        Arc::clone(runtime.host()),
    )
    .unwrap();
    let ended = start(
        runtime.host(),
        Arc::clone(&runtime) as Arc<dyn Answerer>,
        writer,
        prompts.clone(),
        Options {
            max_tokens: MAX_TOKENS,
            greedy_fraction: 0.25,
            batch: 2,
            sampling: SamplingSettings::default(),
            stops,
            vocabulary,
        },
    );
    let summary = ended.wait(WAIT).expect("the run ended").expect("the run succeeded");
    assert_eq!(summary.answered, 2);

    let entries = corpus::entries(directory.path()).unwrap();
    assert_eq!(entries.len(), 2);
    let read = |name: &str| fs::read(directory.path().join(name)).unwrap();
    let (tokens_file, labels_file, hidden_file) = (read("tokens.bin"), read("labels.bin"), read("hidden.bin"));
    let word = |bytes: &[u8], at: usize| u32::from_le_bytes(bytes[4 * at..4 * at + 4].try_into().unwrap());
    for entry in &entries {
        let prompt = prompts.iter().find(|prompt| prompt.id == entry.id).unwrap();
        let (start, length) = (entry.token as usize, entry.length as usize);
        let sequence: Vec<u32> = (start..start + length).map(|at| word(&tokens_file, at)).collect();
        let labels: Vec<u32> = (start..start + length).map(|at| word(&labels_file, at)).collect();
        assert_eq!(sequence[..prompt.tokens.len()], prompt.tokens[..]);
        assert_eq!(labels[..prompt.tokens.len()], prompt.tokens[..]);
        assert_eq!(entry.sampled, entry.id == "p10");
        assert!(
            entry.length as usize > prompt.tokens.len(),
            "{} has no answer",
            entry.id
        );
        if !entry.sampled {
            assert_eq!(
                sequence[prompt.tokens.len()..],
                own[..],
                "the greedy answer is not the artifact's decode"
            );
            assert_eq!(labels, sequence);
        }
    }
    assert_eq!(
        hidden_file.len(),
        2 * width * entries.iter().map(|entry| entry.length as usize - 1).sum::<usize>()
    );
    let last = entries.iter().map(|entry| entry.row + entry.length - 1).max().unwrap() as usize;
    assert_eq!(hidden_file.len(), 2 * width * last);
    owner.shutdown(runtime).expect("the engine shuts down");
}
