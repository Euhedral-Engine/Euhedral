//! Greedy generation and teacher-forced decoding on a GPU, against the Java engine's output for the same artifact. The
//! tests are `#[ignore]`d and run locally in a memory-capped scope, one at a time (each loads the 12 GB artifact):
//! `systemd-run --user --scope -p MemoryMax=24G cargo test -p euhedral-inference-engine --test generation_gpu -- --ignored --test-threads=1`.
//!
//! The references are made once by the Java engine and kept outside the repository, in the directory named by
//! `euhedral.rust.goldens` (`EUHEDRAL_TEST_RUST_GOLDENS`): a test skips with a message when it is absent. Regenerate them
//! from the repository root, with `$RCO` the artifact of `euhedral.qwen.rco-artifact`, `$GOLDENS` that directory, and
//! the serving container stopped:
//!
//! ```text
//! systemd-run --user --scope -p MemoryMax=24G ./gradlew :core:cudaOwnJvmTest \
//!     --tests '*GreedyReferenceCudaIntegrationTest' --tests '*TeacherForcedQualityCudaIntegrationTest' --rerun \
//!     -Peuhedral.greedy.report=$GOLDENS/greedy-rco.txt -Peuhedral.greedy.artifact=$RCO \
//!     -Peuhedral.quality.report=$GOLDENS/teacher-forced-rco.bin -Peuhedral.quality.artifact=$RCO \
//!     -Peuhedral.quality.prefix=1100 -Peuhedral.quality.steps=192 -Peuhedral.quality.rows=1
//! ```
//!
//! `greedy-rco.txt` holds the token ids of a chat prompt and of the 128 tokens the Java engine selects after it;
//! `teacher-forced-rco.bin` holds the logits row of each of 192 teacher-forced steps after a 1100-token prefix of the
//! benchmark corpus document, one-row decode (`tools/compare_teacher_forced.py` reads it).
//!
//! The artifact comes from `euhedral.qwen.rco-artifact` and the library from `euhedral.cuda.library`.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]
#![allow(
    clippy::print_stdout,
    reason = "a test reports the speeds and the counts it measured"
)]
#![allow(
    clippy::disallowed_methods,
    reason = "no-wait, blocking I/O and waiting calls: a test reads files, runs the comparison script and waits on latches"
)]

use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::mem;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::time::{Duration, Instant};

use euhedral_graph::{Latch, LatchSetter};
use euhedral_inference_edrl::{ReadPlan, Region, TokenizerFooter, TokenizerSection};
use euhedral_inference_engine::generation::{
    AdmissionError, FinishReason, GenerationListener, GenerationOutcome, GenerationRequest, LogitsSink, QuantumEvent,
    Speculation,
};
use euhedral_inference_engine::stages::QuantumKind;
use euhedral_inference_engine::{EngineRuntime, RuntimeOwner};
use euhedral_inference_settings::{ModelSettings, Settings};
use euhedral_inference_tokenizer::Tokenizer;

struct Machine {
    artifact: PathBuf,
    library: PathBuf,
    goldens: PathBuf,
}

fn machine() -> Option<Machine> {
    let library = PathBuf::from(test_support::require_property("euhedral.cuda.library")?);
    let artifact = PathBuf::from(test_support::require_property("euhedral.qwen.rco-artifact")?);
    let goldens = PathBuf::from(test_support::require_property("euhedral.rust.goldens")?);
    if !artifact.is_file() {
        println!("skipped: euhedral.qwen.rco-artifact names no file");
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

fn settings(artifact: &Path, max_tokens: u32) -> Settings {
    let mut settings = Settings::with_model(ModelSettings {
        artifact: artifact.to_path_buf(),
        name: "test".to_string(),
    });
    settings.context.max_tokens = max_tokens;
    settings
}

/// Starts the engine and waits for the latch, as `main` does.
fn start(machine: &Machine, max_tokens: u32) -> (Arc<EngineRuntime>, RuntimeOwner) {
    let (ready, waiting) = Latch::new();
    let owner = EngineRuntime::start(&settings(&machine.artifact, max_tokens), &machine.library, ready);
    (waiting.wait().expect("the engine starts"), owner)
}

fn read_range(file: &mut File, range: Range<u64>) -> Vec<u8> {
    file.seek(SeekFrom::Start(range.start)).unwrap();
    let mut bytes = vec![0; (range.end - range.start) as usize];
    file.read_exact(&mut bytes).unwrap();
    bytes
}

/// The tokenizer the artifact packages.
fn tokenizer(artifact: &Path) -> Tokenizer {
    let mut file = File::open(artifact).unwrap();
    let length = file.metadata().unwrap().len();
    let probe = ReadPlan::probe(length).unwrap();
    let footer = TokenizerFooter::parse(
        read_range(&mut file, probe.range(Region::TokenizerFooter).unwrap())
            .as_slice()
            .try_into()
            .unwrap(),
    )
    .unwrap()
    .expect("the artifact carries its tokenizer");
    let body = read_range(&mut file, footer.body_range(length).unwrap());
    Tokenizer::from_section(&TokenizerSection::parse(body, &footer).unwrap()).unwrap()
}

/// The Java engine's greedy generation: the prompt, and the tokens it selected.
struct Greedy {
    prompt: Vec<u32>,
    generated: Vec<u32>,
    stopped_by_eos: bool,
}

fn ids(line: &str, tag: &str) -> Vec<u32> {
    line.strip_prefix(tag)
        .unwrap_or_else(|| panic!("a `{tag}` line, found `{line}`"))
        .split_whitespace()
        .map(|id| id.parse().unwrap())
        .collect()
}

fn greedy_reference(machine: &Machine) -> Greedy {
    let text = fs::read_to_string(machine.goldens.join("greedy-rco.txt")).unwrap();
    let mut lines = text.lines();
    let prompt = ids(lines.next().unwrap(), "prompt ");
    let generated = ids(lines.next().unwrap(), "generated ");
    let stopped_by_eos = lines.next().unwrap().trim() == "stop eos";
    Greedy {
        prompt,
        generated,
        stopped_by_eos,
    }
}

/// What a generation told its listener.
#[derive(Debug)]
struct Recording {
    tokens: Vec<u32>,
    token_times: Vec<Instant>,
    events: Vec<QuantumEvent>,
    prefill_done: Option<Instant>,
    outcome: GenerationOutcome,
    began: Instant,
}

struct Recorder {
    tokens: Vec<u32>,
    token_times: Vec<Instant>,
    events: Vec<QuantumEvent>,
    prefill_done: Option<Instant>,
    began: Instant,
    done: Option<LatchSetter<Recording>>,
}

impl GenerationListener for Recorder {
    fn prefill_done(&mut self, at: Instant) {
        assert!(self.prefill_done.replace(at).is_none(), "the prefill is done once");
    }

    fn token(&mut self, id: u32, at: Instant) {
        self.tokens.push(id);
        self.token_times.push(at);
    }

    fn quantum(&mut self, event: QuantumEvent) {
        self.events.push(event);
    }

    fn finished(&mut self, outcome: GenerationOutcome) {
        let done = self.done.take().expect("a generation finishes once");
        done.set(Recording {
            tokens: mem::take(&mut self.tokens),
            token_times: mem::take(&mut self.token_times),
            events: mem::take(&mut self.events),
            prefill_done: self.prefill_done,
            outcome,
            began: self.began,
        });
    }
}

/// Runs a greedy generation and waits for its listener's latch.
fn generate(runtime: &EngineRuntime, prompt: &[u32], max_tokens: u32, stop_tokens: &[u32]) -> Recording {
    let (done, waiting) = Latch::new();
    let began = Instant::now();
    runtime
        .generate(GenerationRequest {
            prompt: prompt.to_vec(),
            max_tokens,
            stop_tokens: stop_tokens.to_vec(),
            speculation: Speculation::Off,
            listener: Box::new(Recorder {
                tokens: Vec::new(),
                token_times: Vec::new(),
                events: Vec::new(),
                prefill_done: None,
                began,
                done: Some(done),
            }),
        })
        .expect("the request is admitted");
    waiting.wait()
}

fn micros(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1e6
}

/// The rows of the sequence's caches after a generation: committed rows, quanta in flight, selections leased.
fn sequence_state(runtime: &EngineRuntime) -> (u64, usize, usize) {
    runtime
        .inspect_sequence(|sequence| {
            (
                sequence.kv.committed(),
                sequence.in_flight(),
                sequence.selections.leased(),
            )
        })
        .expect("the generation gave the sequence back")
}

#[test]
#[ignore = "needs a GPU, the native library, the RCO artifact and the Java references"]
fn greedy_tokens_match_java_for_short_prompt() {
    let Some(machine) = machine() else { return };
    let reference = greedy_reference(&machine);
    assert_eq!(reference.generated.len(), 128);
    assert!(!reference.stopped_by_eos);
    let eos = tokenizer(&machine.artifact).eos_ids().to_vec();
    let (runtime, owner) = start(&machine, 4096);

    // Twice on one engine: the second request resets the sequence and replays the graphs the first recorded.
    for round in 0..2 {
        let recording = generate(&runtime, &reference.prompt, 128, &eos);
        assert_eq!(recording.outcome.reason, FinishReason::Length, "round {round}");
        assert_eq!(recording.outcome.prompt_tokens, reference.prompt.len());
        assert_eq!(recording.outcome.generated, 128);
        assert_eq!(
            recording.tokens, reference.generated,
            "round {round}: the greedy tokens are the Java engine's"
        );
        let decodes: Vec<&QuantumEvent> = recording
            .events
            .iter()
            .filter(|event| event.kind == QuantumKind::Decode)
            .collect();
        assert_eq!(decodes.len(), 127, "one decode quantum per token after the first");
        assert_eq!(
            recording
                .events
                .iter()
                .filter(|event| event.kind == QuantumKind::Prefill)
                .count(),
            1
        );
        assert!(
            decodes.iter().filter(|event| event.replayed).count() >= 100,
            "the decode quanta replay graphs"
        );
        // The sequence holds the prompt and the 127 rows decoded; the last token was selected, not fed.
        assert_eq!(
            sequence_state(&runtime),
            (reference.prompt.len() as u64 + 127, 0, 0),
            "committed rows, quanta in flight, selections leased"
        );
        let first = recording.token_times[0];
        let last = *recording.token_times.last().unwrap();
        println!(
            "round {round}: {} prompt tokens, prefill {:.1} ms, first token {:.1} ms after the request, then {:.1} tok/s \
             over {} tokens ({} decode quanta replayed, {} recorded)",
            reference.prompt.len(),
            micros(recording.prefill_done.unwrap() - recording.began) / 1e3,
            micros(first - recording.began) / 1e3,
            (recording.tokens.len() - 1) as f64 / (last - first).as_secs_f64(),
            recording.tokens.len(),
            decodes.iter().filter(|event| event.replayed).count(),
            decodes.iter().filter(|event| event.recorded).count(),
        );
    }
    owner.shutdown(runtime).unwrap();
}

#[test]
#[ignore = "needs a GPU, the native library, the RCO artifact and the Java references"]
fn a_stop_token_ends_the_generation_and_leaves_no_overshoot_in_the_sequence() {
    let Some(machine) = machine() else { return };
    let reference = greedy_reference(&machine);
    let (runtime, owner) = start(&machine, 4096);
    let prompt_rows = reference.prompt.len() as u64;

    // The token the Java engine selected sixth ends the generation when it is a stop token: the quantum that already
    // follows it (it feeds the stop token) is stopped, and its row never reaches the cache.
    let stop = reference.generated[5];
    assert!(
        !reference.generated[..5].contains(&stop),
        "the sixth token is the first of its kind"
    );
    let recording = generate(&runtime, &reference.prompt, 128, &[stop]);
    assert_eq!(recording.outcome.reason, FinishReason::Stop);
    assert_eq!(recording.tokens, reference.generated[..6]);
    assert_eq!(sequence_state(&runtime), (prompt_rows + 5, 0, 0));

    // A request for no tokens prefills the prompt and ends.
    let recording = generate(&runtime, &reference.prompt, 0, &[]);
    assert_eq!(recording.outcome.reason, FinishReason::Length);
    assert!(recording.tokens.is_empty());
    assert!(recording.prefill_done.is_some());
    assert_eq!(sequence_state(&runtime), (prompt_rows, 0, 0));

    // One token comes from the prompt alone.
    let recording = generate(&runtime, &reference.prompt, 1, &[]);
    assert_eq!(recording.tokens, reference.generated[..1]);
    assert_eq!(sequence_state(&runtime), (prompt_rows, 0, 0));

    // The engine's sequence takes one request at a time, and refuses what cannot run.
    let nothing = |prompt: Vec<u32>| {
        runtime.generate(GenerationRequest {
            prompt,
            max_tokens: 1,
            stop_tokens: Vec::new(),
            speculation: Speculation::Off,
            listener: Box::new(Recorder {
                tokens: Vec::new(),
                token_times: Vec::new(),
                events: Vec::new(),
                prefill_done: None,
                began: Instant::now(),
                done: None,
            }),
        })
    };
    assert_eq!(nothing(Vec::new()), Err(AdmissionError::EmptyPrompt));
    assert!(matches!(
        nothing(vec![u32::MAX]),
        Err(AdmissionError::TokenOutsideVocabulary { .. })
    ));
    assert!(matches!(
        nothing(vec![1; 5000]),
        Err(AdmissionError::PromptTooLong { capacity: 4096, .. })
    ));
    owner.shutdown(runtime).unwrap();
}

#[test]
#[ignore = "needs a GPU, the native library, the RCO artifact and the Java references"]
fn context_end_stops_cleanly() {
    let Some(machine) = machine() else { return };
    let reference = greedy_reference(&machine);
    let context = 600u32;
    let (runtime, owner) = start(&machine, context);
    let capacity = u64::from(context);
    assert_eq!(runtime.plan().max_context_tokens as u64, capacity);
    let prompt_rows = reference.prompt.len() as u64;

    // The request wants far more than the context holds: the cache fills, the last row is decoded, and the generation
    // ends with the reason that says so. One token comes from the prompt and one from each row decoded after it.
    let recording = generate(&runtime, &reference.prompt, 100_000, &[]);
    assert_eq!(recording.outcome.reason, FinishReason::ContextFull);
    assert_eq!(recording.tokens.len() as u64, capacity - prompt_rows + 1);
    assert_eq!(
        recording.tokens[..128],
        reference.generated[..],
        "the tokens before the end are the Java engine's"
    );
    assert_eq!(sequence_state(&runtime), (capacity, 0, 0));

    // A prompt that fills the context gives its one token and stops.
    let full: Vec<u32> = reference
        .prompt
        .iter()
        .copied()
        .cycle()
        .take(context as usize)
        .collect();
    let recording = generate(&runtime, &full, 10, &[]);
    assert_eq!(recording.outcome.reason, FinishReason::ContextFull);
    assert_eq!(recording.tokens.len(), 1);
    assert_eq!(sequence_state(&runtime), (capacity, 0, 0));

    // And the engine still serves: the same prompt as the first request, from a reset sequence.
    let recording = generate(&runtime, &reference.prompt, 128, &[]);
    assert_eq!(recording.tokens, reference.generated);
    owner.shutdown(runtime).unwrap();
}

/// Collects the rows of a teacher-forced run in the Java report's format.
struct Report {
    bytes: Vec<u8>,
    steps: usize,
    done: Option<LatchSetter<(Vec<u8>, GenerationOutcome)>>,
}

impl LogitsSink for Report {
    fn row(&mut self, step: usize, forced: u32, logits: &[u8]) {
        assert_eq!(step, self.steps, "the rows arrive in order");
        self.steps += 1;
        self.bytes.extend_from_slice(&(forced as i32).to_le_bytes());
        self.bytes.extend_from_slice(logits);
    }

    fn finished(&mut self, outcome: GenerationOutcome) {
        let done = self.done.take().expect("a run finishes once");
        done.set((mem::take(&mut self.bytes), outcome));
    }
}

fn repository_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

#[test]
#[ignore = "needs a GPU, the native library, the RCO artifact, the Java references and python3 with numpy"]
fn teacher_forced_logits_bitwise_equal_java_report() {
    let Some(machine) = machine() else { return };
    let java = machine.goldens.join("teacher-forced-rco.bin");
    let java_bytes = fs::read(&java).unwrap();
    let steps = i32::from_le_bytes(java_bytes[0..4].try_into().unwrap()) as usize;
    let vocabulary = i32::from_le_bytes(java_bytes[4..8].try_into().unwrap()) as usize;
    let record = 4 + 2 * vocabulary;
    assert_eq!(java_bytes.len(), 8 + steps * record);
    let prefix = 1100;

    // The corpus document, tokenized as the Java tool does (`encodeText`): the first `prefix` tokens are the prefix, the
    // next `steps` are the forced tokens the report names.
    let tokenizer = tokenizer(&machine.artifact);
    let document = fs::read_to_string(repository_root().join(
        "benchmark/src/main/resources/io/euhedral_execution/inference/benchmark/prompt/chat-corpus-v1-document.md",
    ))
    .unwrap();
    let document = tokenizer.encode(&document).unwrap();
    assert!(document.len() > prefix + steps);
    let forced: Vec<u32> = document[prefix..prefix + steps].to_vec();
    for (step, token) in forced.iter().enumerate() {
        let at = 8 + step * record;
        let named = i32::from_le_bytes(java_bytes[at..at + 4].try_into().unwrap());
        assert_eq!(named as u32, *token, "step {step}: the forced token of the Java report");
    }

    let (runtime, owner) = start(&machine, 4096);
    let (done, waiting) = Latch::new();
    let mut bytes = Vec::with_capacity(java_bytes.len());
    bytes.extend_from_slice(&(steps as i32).to_le_bytes());
    bytes.extend_from_slice(&(vocabulary as i32).to_le_bytes());
    let began = Instant::now();
    runtime
        .teacher_forced(
            document[..prefix].to_vec(),
            forced,
            Box::new(Report {
                bytes,
                steps: 0,
                done: Some(done),
            }),
        )
        .expect("the run is admitted");
    let (rust_bytes, outcome) = waiting.wait();
    println!(
        "{steps} teacher-forced steps after a {prefix}-token prefix in {:.2} s",
        began.elapsed().as_secs_f64()
    );
    assert_eq!(outcome.reason, FinishReason::Length);
    assert_eq!(outcome.generated, steps);
    owner.shutdown(runtime).unwrap();

    let out = Path::new(env!("CARGO_TARGET_TMPDIR")).join("teacher-forced-rust.bin");
    fs::write(&out, &rust_bytes).unwrap();
    if rust_bytes != java_bytes {
        let first = (0..steps).find(|step| {
            let at = 8 + step * record;
            rust_bytes[at..at + record] != java_bytes[at..at + record]
        });
        panic!(
            "the report differs from the Java engine's, first at step {first:?} (see {})",
            out.display()
        );
    }

    // The script that compares the artifacts says so too: no difference in the negative log-likelihood, no divergence,
    // the same top-1 token at every step.
    let output = Command::new("python3")
        .arg(repository_root().join("tools/compare_teacher_forced.py"))
        .arg(&java)
        .arg(&out)
        .output()
        .expect("python3 runs");
    let text = String::from_utf8_lossy(&output.stdout);
    println!("{text}{}", String::from_utf8_lossy(&output.stderr));
    assert!(
        output.status.success(),
        "the comparison script failed (python3 with numpy?)"
    );
    let candidate = text
        .lines()
        .find(|line| line.contains("teacher-forced-rust.bin"))
        .expect("the script reports the candidate");
    let columns: Vec<&str> = candidate.split_whitespace().collect();
    // path, mean NLL, ppl, dNLL, +-SE, KL, top-1
    assert_eq!(columns[3], "+0.00000", "no paired NLL difference: {candidate}");
    assert_eq!(columns[5], "0.000000", "no KL divergence: {candidate}");
    assert_eq!(columns[6], "1.0000", "the same top-1 token at every step: {candidate}");
}
