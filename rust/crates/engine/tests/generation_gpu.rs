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
    reason = "no-wait, blocking I/O and waiting calls: a test reads files, runs the comparison script and waits on hand-offs"
)]

use std::env;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::mem;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant};

use euhedral_inference_cuda::launch::load::tensor_map_counts;
use euhedral_inference_edrl::{ReadPlan, Region, TokenizerFooter, TokenizerSection};
mod common;

use common::Closer;
use euhedral_inference_engine::generation::{
    AdmissionError, CancelHandle, FinishReason, GenerationListener, GenerationOutcome, GenerationRequest, LogitsSink,
    QuantumEvent, Speculation, Verification,
};
use euhedral_inference_engine::sampling::{SamplingConfig, TokenSampler};
use euhedral_inference_engine::stages::QuantumKind;
use euhedral_inference_engine::{EngineRuntime, Handoff};

/// The longest a device test waits for one hand-off; past it the test fails.
const WAIT: Duration = Duration::from_secs(20 * 60);
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
    settings.kernels.graphs.captures_per_view = 8;
    settings
}

/// Starts the engine and waits for the latch, as `main` does.
fn start(machine: &Machine, max_tokens: u32) -> (Closer, Arc<EngineRuntime>) {
    let (owner, ready) = EngineRuntime::start(&settings(&machine.artifact, max_tokens), &machine.library);
    let runtime = ready
        .wait(WAIT)
        .expect("the engine reports")
        .expect("the engine starts");
    (Closer::new(owner, &runtime), runtime)
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
    verifications: Vec<Verification>,
    prefill_done: Option<Instant>,
    outcome: GenerationOutcome,
    began: Instant,
}

struct Recorder {
    tokens: Vec<u32>,
    token_times: Vec<Instant>,
    events: Vec<QuantumEvent>,
    verifications: Vec<Verification>,
    prefill_done: Option<Instant>,
    began: Instant,
    done: Option<Arc<Handoff<Recording>>>,
    /// Cancels the request once it has reported this many tokens.
    cancel_after: Option<(CancelHandle, usize)>,
    /// Cancels the request when one of its verifications retires after the engine has run this many batched passes:
    /// the request's next step is then offered, with the others', to a pass it is a member of.
    cancel_in_pass: Option<(CancelHandle, Weak<EngineRuntime>, u64)>,
}

impl GenerationListener for Recorder {
    fn prefill_done(&mut self, at: Instant) {
        assert!(self.prefill_done.replace(at).is_none(), "the prefill is done once");
    }

    fn token(&mut self, id: u32, at: Instant) {
        self.tokens.push(id);
        self.token_times.push(at);
        if let Some((cancel, _)) = self
            .cancel_after
            .as_ref()
            .filter(|(_, after)| self.tokens.len() >= *after)
        {
            cancel.cancel();
        }
    }

    fn quantum(&mut self, event: QuantumEvent) {
        self.events.push(event);
    }

    fn verification(&mut self, event: Verification) {
        self.verifications.push(event);
        if let Some((cancel, runtime, passes)) = &self.cancel_in_pass
            && runtime
                .upgrade()
                .is_some_and(|runtime| runtime.pass_statistics().batched >= *passes)
        {
            cancel.cancel();
        }
    }

    fn finished(&mut self, outcome: GenerationOutcome) {
        let done = self.done.take().expect("a generation finishes once");
        done.send(Recording {
            tokens: mem::take(&mut self.tokens),
            token_times: mem::take(&mut self.token_times),
            events: mem::take(&mut self.events),
            verifications: mem::take(&mut self.verifications),
            prefill_done: self.prefill_done,
            outcome,
            began: self.began,
        });
    }
}

impl Drop for Recorder {
    /// A listener dropped before its generation finished reports it as failed, so the test fails at once.
    fn drop(&mut self) {
        if let Some(done) = self.done.take() {
            done.send(Recording {
                tokens: mem::take(&mut self.tokens),
                token_times: mem::take(&mut self.token_times),
                events: mem::take(&mut self.events),
                verifications: mem::take(&mut self.verifications),
                prefill_done: self.prefill_done,
                outcome: dropped_outcome(),
                began: self.began,
            });
        }
    }
}

fn dropped_outcome() -> GenerationOutcome {
    GenerationOutcome {
        reason: FinishReason::Failed("the listener was dropped before the generation finished".to_string()),
        prompt_tokens: 0,
        generated: 0,
        elapsed: Duration::ZERO,
        cached_prompt_tokens: 0,
    }
}

/// Runs a greedy generation and waits for its listener's hand-off.
fn generate(runtime: &EngineRuntime, prompt: &[u32], max_tokens: u32, stop_tokens: &[u32]) -> Recording {
    generate_sampled(runtime, prompt, max_tokens, stop_tokens, SamplingConfig::greedy(0))
}

/// Runs a generation with `sampling` and waits for its listener's hand-off.
fn generate_sampled(
    runtime: &EngineRuntime,
    prompt: &[u32],
    max_tokens: u32,
    stop_tokens: &[u32],
    sampling: SamplingConfig,
) -> Recording {
    submit(
        runtime,
        prompt,
        max_tokens,
        stop_tokens,
        Speculation::Off,
        sampling,
        None,
    )
    .expect("the request is admitted")
    .wait(WAIT)
    .expect("the generation reports")
}

/// Starts a generation and returns the hand-off its listener reports on, or why it was refused. With `cancel`, the
/// request is cancelled by its own listener after that many tokens.
fn submit(
    runtime: &EngineRuntime,
    prompt: &[u32],
    max_tokens: u32,
    stop_tokens: &[u32],
    speculation: Speculation,
    sampling: SamplingConfig,
    cancel: Option<usize>,
) -> Result<Arc<Handoff<Recording>>, AdmissionError> {
    submit_with(
        runtime,
        prompt,
        max_tokens,
        stop_tokens,
        speculation,
        sampling,
        cancel,
        None,
    )
}

/// [`submit`], and with `in_pass` the engine and the number of batched passes after which a verification of the
/// request cancels it.
#[allow(
    clippy::too_many_arguments,
    reason = "the arguments of a request, as the test helpers pass them"
)]
fn submit_with(
    runtime: &EngineRuntime,
    prompt: &[u32],
    max_tokens: u32,
    stop_tokens: &[u32],
    speculation: Speculation,
    sampling: SamplingConfig,
    cancel: Option<usize>,
    in_pass: Option<(Weak<EngineRuntime>, u64)>,
) -> Result<Arc<Handoff<Recording>>, AdmissionError> {
    let done = Handoff::new();
    let handle = CancelHandle::default();
    let began = Instant::now();
    runtime.generate_cancellable(
        GenerationRequest {
            prompt: prompt.to_vec(),
            max_tokens,
            stop_tokens: stop_tokens.to_vec(),
            speculation,
            sampling,
            constraint: None,
            listener: Box::new(Recorder {
                tokens: Vec::new(),
                token_times: Vec::new(),
                events: Vec::new(),
                verifications: Vec::new(),
                prefill_done: None,
                began,
                done: Some(Arc::clone(&done)),
                cancel_after: cancel.map(|after| (handle.clone(), after)),
                cancel_in_pass: in_pass.map(|(runtime, passes)| (handle.clone(), runtime, passes)),
            }),
        },
        handle,
    )?;
    Ok(done)
}

/// Starts the engine with `sequences` sequences, as `batching.max-sequences` states.
fn start_with_sequences(machine: &Machine, sequences: u32) -> (Closer, Arc<EngineRuntime>) {
    let mut settings = settings(&machine.artifact, 4096);
    settings.batching.max_sequences = sequences;
    let (owner, ready) = EngineRuntime::start(&settings, &machine.library);
    let runtime = ready
        .wait(WAIT)
        .expect("the engine reports")
        .expect("the engine starts");
    (Closer::new(owner, &runtime), runtime)
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
    let (owner, runtime) = start(&machine, 4096);

    // Twice on one engine: the second request resets the sequence and replays the graphs the first recorded.
    for round in 0..2 {
        let (asked_before, encoded_before) = tensor_map_counts();
        let recording = generate(&runtime, &reference.prompt, 128, &eos);
        let (asked, encoded) = tensor_map_counts();
        println!(
            "round {round}: {} tensor maps asked for, {} encoded",
            asked - asked_before,
            encoded - encoded_before
        );
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
        let replayed = decodes.iter().filter(|event| event.replayed).count();
        assert!(replayed >= 100, "the decode quanta replay graphs, saw {replayed}");
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
    let (owner, runtime) = start(&machine, 4096);
    let prompt_rows = reference.prompt.len() as u64;

    // The token the Java engine selected sixth ends the generation when it is a stop token: the quantum that already
    // follows it (it feeds the stop token) runs to its end, and its row never reaches the cache.
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
            sampling: SamplingConfig::greedy(0),
            constraint: None,
            listener: Box::new(Recorder {
                tokens: Vec::new(),
                token_times: Vec::new(),
                events: Vec::new(),
                verifications: Vec::new(),
                prefill_done: None,
                began: Instant::now(),
                done: None,
                cancel_after: None,
                cancel_in_pass: None,
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
    let (owner, runtime) = start(&machine, context);
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

/// The rows of a run and how it ended.
type Delivered = (Vec<u8>, GenerationOutcome);

/// Collects the rows of a teacher-forced run in the Java report's format.
struct Report {
    bytes: Vec<u8>,
    steps: usize,
    done: Option<Arc<Handoff<Delivered>>>,
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
        done.send((mem::take(&mut self.bytes), outcome));
    }
}

fn repository_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

#[test]
#[ignore = "needs a GPU, the native library, the RCO artifact, the Java references and python3 with numpy"]
fn teacher_forced_logits_bitwise_equal_java_report() {
    let Some(machine) = machine() else { return };
    teacher_forced_against_java(&machine, None);
}

/// The same run with the device 256 MiB short of the resident need, so the plan reads the embedding in place from pinned
/// host memory: the logits are the report's, bit for bit.
#[test]
#[ignore = "needs a GPU, the native library, the RCO artifact, the Java references and python3 with numpy"]
fn teacher_forced_logits_bitwise_equal_java_report_with_a_mapped_embedding() {
    let Some(machine) = machine() else { return };
    let limit = common::limit_below_need(&machine.library, &settings(&machine.artifact, 4096), 256 << 20);
    teacher_forced_against_java(&machine, Some(limit));
}

/// Runs the teacher-forced comparison against the Java report, under `gpu_limit` when it is given. A forced run that
/// does not get the plan it means (the card has less free than the limit) stops with a message.
fn teacher_forced_against_java(machine: &Machine, gpu_limit: Option<u64>) {
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

    let mut engine = settings(&machine.artifact, 4096);
    if let Some(limit) = gpu_limit {
        common::limit_gpu(&mut engine, limit);
    }
    let (owner, ready) = EngineRuntime::start(&engine, &machine.library);
    let runtime = ready
        .wait(WAIT)
        .expect("the engine reports")
        .expect("the engine starts");
    let (owner, runtime) = (Closer::new(owner, &runtime), runtime);
    if let Some(limit) = gpu_limit {
        common::embedding_alone_is_on_the_host(&runtime, limit);
        assert!(runtime.report().host_backed_bytes > 0);
    }
    let done = Handoff::new();
    let waiting = Arc::clone(&done);
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
    let (rust_bytes, outcome) = waiting.wait(WAIT).expect("the run reports");
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

#[test]
#[ignore = "needs a GPU, the native library, the RCO artifact and the Java references"]
fn a_panic_in_the_owner_ends_the_request_with_a_failure_and_the_engine_serves_the_next() {
    let Some(machine) = machine() else { return };
    let reference = greedy_reference(&machine);
    let (owner, runtime) = start(&machine, 4096);

    // The owner panics in the turn that has admitted a third quantum, with two in flight: the request fails, the quanta
    // in flight run to their end and are discarded, and the listener is told as the last call.
    runtime.panic_after_admissions_for_tests(3);
    let recording = generate(&runtime, &reference.prompt, 128, &[]);
    match &recording.outcome.reason {
        FinishReason::Failed(why) => assert!(why.contains("panicked"), "{why}"),
        other => panic!("the request should have failed, ended with {other:?}"),
    }
    assert!(recording.tokens.len() < 128);
    assert_eq!(recording.tokens, reference.generated[..recording.tokens.len()]);
    assert_eq!(sequence_state(&runtime).1, 0, "no quantum is left in flight");
    assert_eq!(sequence_state(&runtime).2, 0, "no selection is left leased");

    // The engine serves the next request, from a reset sequence, with the Java engine's tokens.
    let recording = generate(&runtime, &reference.prompt, 128, &[]);
    assert_eq!(recording.outcome.reason, FinishReason::Length);
    assert_eq!(recording.tokens, reference.generated);
    owner.shutdown(runtime).unwrap();
}

/// A sampled configuration: temperature 1, the given `top_k`, no nucleus cut, and the seed.
fn sampled(top_k: u32, seed: u64) -> SamplingConfig {
    SamplingConfig::new(1.0, top_k, 1.0, seed, false).unwrap()
}

#[test]
#[ignore = "needs a GPU, the native library, the RCO artifact and the Java references"]
fn sampled_requests_select_on_the_host_and_agree_with_greedy_at_top_k_one() {
    let Some(machine) = machine() else { return };
    let reference = greedy_reference(&machine);
    let tokens = 32;
    let (owner, runtime) = start(&machine, 4096);
    let expected = &reference.generated[..tokens as usize];

    // Temperature 0 is greedy: the device selects, no row is copied.
    let zero = SamplingConfig::new(0.0, 0, 1.0, 5, false).unwrap();
    let greedy_path = generate_sampled(&runtime, &reference.prompt, tokens, &[], zero);
    assert_eq!(greedy_path.outcome.reason, FinishReason::Length);
    assert_eq!(greedy_path.tokens, expected, "temperature 0 equals greedy");

    // Top-k 1 samples on the host and must choose the greedy token every time.
    let top_one = generate_sampled(&runtime, &reference.prompt, tokens, &[], sampled(1, 11));
    assert_eq!(top_one.outcome.reason, FinishReason::Length);
    assert_eq!(top_one.tokens, expected, "top-k 1 equals greedy token for token");
    assert_eq!(top_one.outcome.generated, tokens as usize);
    let (_, in_flight, leased) = sequence_state(&runtime);
    assert_eq!((in_flight, leased), (0, 0), "nothing left in flight or leased");

    // The same seed gives the same tokens; a different seed is allowed to differ.
    let first = generate_sampled(&runtime, &reference.prompt, tokens, &[], sampled(40, 21));
    let second = generate_sampled(&runtime, &reference.prompt, tokens, &[], sampled(40, 21));
    assert_eq!(first.outcome.reason, FinishReason::Length);
    assert_eq!(first.tokens.len(), tokens as usize);
    assert_eq!(first.tokens, second.tokens, "the same seed gives the same output");
    let other = generate_sampled(&runtime, &reference.prompt, tokens, &[], sampled(40, 22));
    assert_eq!(other.tokens.len(), tokens as usize);
    println!("seeds 21 and 22 differ: {}", other.tokens != first.tokens);

    let leased = runtime
        .inspect_sequence(|sequence| sequence.row_readbacks.leased())
        .expect("the generation gave the sequence back");
    assert_eq!(leased, 0, "no row left leased");
    owner.shutdown(runtime).unwrap();
}

/// Collects the logits rows of a teacher-forced run.
struct Rows {
    rows: Vec<Vec<u8>>,
    done: Option<Arc<Handoff<Vec<Vec<u8>>>>>,
}

impl LogitsSink for Rows {
    fn row(&mut self, step: usize, _forced: u32, logits: &[u8]) {
        assert_eq!(step, self.rows.len(), "the rows arrive in order");
        self.rows.push(logits.to_vec());
    }

    fn finished(&mut self, _outcome: GenerationOutcome) {
        let done = self.done.take().expect("a run finishes once");
        done.send(mem::take(&mut self.rows));
    }
}

/// The logits row that predicts each of `forced` after `prompt`.
fn forced_rows(runtime: &EngineRuntime, prompt: &[u32], forced: &[u32]) -> Vec<Vec<u8>> {
    let done = Handoff::new();
    runtime
        .teacher_forced(
            prompt.to_vec(),
            forced.to_vec(),
            Box::new(Rows {
                rows: Vec::new(),
                done: Some(Arc::clone(&done)),
            }),
        )
        .expect("the run is admitted");
    done.wait(WAIT).expect("the run reports")
}

/// The position of each sampled token is the length of what precedes it: the prompt's for the first, one more for
/// each after it. Checked with a configuration in which the position changes the token (a flattened distribution, not top-k 1):
/// each token the engine sampled is the one a fresh sampler draws from the same row at that position, and not the one
/// it draws at the next, over a prompt of one chunk and of several.
#[test]
#[ignore = "needs a GPU, the native library, the RCO artifact and the Java references"]
fn sampled_tokens_are_drawn_at_the_position_they_follow() {
    let Some(machine) = machine() else { return };
    let reference = greedy_reference(&machine);
    let config = SamplingConfig::new(50.0, 40, 1.0, 77, false).unwrap();
    let tokens = 24;
    let mut settings = settings(&machine.artifact, 4096);
    settings.batching.prefill_chunk_tokens = 256;
    let (raw, ready) = EngineRuntime::start(&settings, &machine.library);
    let runtime = ready
        .wait(WAIT)
        .expect("the engine reports")
        .expect("the engine starts");
    // The closer is bound first, so that on unwind the runtime's handle drops before it and its shutdown finds the last one.
    let (owner, runtime) = (Closer::new(raw, &runtime), runtime);
    let prompts = [
        reference.prompt.clone(),
        reference.prompt.iter().copied().cycle().take(700).collect::<Vec<u32>>(),
    ];
    for prompt in prompts {
        let recording = generate_sampled(&runtime, &prompt, tokens, &[], config);
        assert_eq!(recording.tokens.len(), tokens as usize);
        let rows = forced_rows(&runtime, &prompt, &recording.tokens);
        assert_eq!(rows.len(), tokens as usize);
        let vocabulary = rows[0].len() / 2;
        let mut sampler = TokenSampler::new(config, vocabulary).unwrap();
        let mut shifted_differs = false;
        for (index, row) in rows.iter().enumerate() {
            let position = (prompt.len() + index) as u64;
            let chosen = sampler.select_bf16(row, position, None).unwrap();
            assert_eq!(
                chosen,
                recording.tokens[index],
                "token {index} of a {}-token prompt is drawn at position {position}",
                prompt.len()
            );
            shifted_differs |= sampler.select_bf16(row, position + 1, None).unwrap() != chosen;
        }
        assert!(shifted_differs, "the position matters in this configuration");
    }
    owner.shutdown(runtime).unwrap();
}

/// A stop token as the first sampled token ends the request after reporting it, as for a greedy request; a budget of
/// one ends it after one token; neither leaves a quantum in flight or a row leased.
#[test]
#[ignore = "needs a GPU, the native library, the RCO artifact and the Java references"]
fn a_sampled_first_token_can_be_the_stop_token_or_the_whole_budget() {
    let Some(machine) = machine() else { return };
    let reference = greedy_reference(&machine);
    let config = SamplingConfig::new(50.0, 40, 1.0, 5, false).unwrap();
    let (owner, runtime) = start(&machine, 4096);
    let full = generate_sampled(&runtime, &reference.prompt, 8, &[], config);
    assert_eq!(full.tokens.len(), 8);
    let first = full.tokens[0];

    let one = generate_sampled(&runtime, &reference.prompt, 1, &[], config);
    assert_eq!(one.outcome.reason, FinishReason::Length);
    assert_eq!(one.tokens, [first], "a budget of one is the first token");
    assert_eq!(sequence_state(&runtime), (reference.prompt.len() as u64, 0, 0));

    let stopped = generate_sampled(&runtime, &reference.prompt, 8, &[first], config);
    assert_eq!(stopped.outcome.reason, FinishReason::Stop);
    assert_eq!(stopped.tokens, [first], "the stop token is reported, and is the last");
    assert_eq!(stopped.outcome.generated, 1);
    let leased = runtime
        .inspect_sequence(|sequence| sequence.row_readbacks.leased())
        .expect("the generation gave the sequence back");
    assert_eq!(leased, 0);

    // The same stop token, when it is a later token, ends the run there.
    let later = full.tokens[3];
    if !full.tokens[..3].contains(&later) {
        let stopped = generate_sampled(&runtime, &reference.prompt, 8, &[later], config);
        assert_eq!(stopped.outcome.reason, FinishReason::Stop);
        assert_eq!(stopped.tokens, full.tokens[..4]);
    }
    owner.shutdown(runtime).unwrap();
}

#[test]
#[ignore = "needs a GPU, the native library, the RCO artifact and the Java references"]
fn two_sequences_start_hold_a_spare_generate_the_same_tokens_and_shut_down_clean() {
    let Some(machine) = machine() else { return };
    let prompt = greedy_reference(&machine).prompt;
    let eos = tokenizer(&machine.artifact).eos_ids().to_vec();

    let (owner, runtime) = start_with_sequences(&machine, 1);
    assert_eq!(runtime.plan().sequences, 1);
    assert_eq!(runtime.sequences(), 1);
    let one = generate(&runtime, &prompt, 16, &eos);
    owner.shutdown(runtime).expect("one sequence shuts down clean");

    let (owner, runtime) = start_with_sequences(&machine, 2);
    assert_eq!(runtime.plan().sequences, 2);
    assert_eq!(runtime.sequences(), 2, "every sequence has an owner");
    let two = generate(&runtime, &prompt, 16, &eos);
    assert_eq!(two.outcome.reason, FinishReason::Length);
    assert_eq!(two.tokens, one.tokens, "two sequences change no token of the first");
    owner.shutdown(runtime).expect("two sequences shut down clean");
}

/// Four requests at once on four sequences: each produces the tokens it produces alone.
const SEQUENCES: usize = 4;
const CONCURRENT_TOKENS: u32 = 48;
const CONCURRENT_PROMPTS: [&str; SEQUENCES] = [
    "The capital of France is",
    "Write a short poem about the sea and the lighthouse keeper.",
    "List the planets of the solar system, from the sun outward, with one fact about each of them.",
    "Explain in a few sentences how a heat pump moves thermal energy from a cold source to a warm sink.",
];

/// The tokens of a recording per second, from its first token to its last.
fn decode_rate(recording: &Recording) -> f64 {
    let (Some(first), Some(last)) = (recording.token_times.first(), recording.token_times.last()) else {
        return 0.0;
    };
    (recording.tokens.len() - 1) as f64 / last.duration_since(*first).as_secs_f64().max(1e-9)
}

/// Starts every prompt at once and waits for all of them, with `cancel` and the sampling of each.
fn together(
    runtime: &EngineRuntime,
    prompts: &[Vec<u32>],
    speculation: Speculation,
    sampling: &[SamplingConfig],
    cancel: &[Option<usize>],
) -> (Vec<Recording>, Duration) {
    together_with(runtime, prompts, speculation, sampling, cancel, None)
}

/// [`together`], and with `in_pass` the request (by index) a verification of which cancels it once the engine has run
/// that many batched passes more than it had.
fn together_with(
    runtime: &EngineRuntime,
    prompts: &[Vec<u32>],
    speculation: Speculation,
    sampling: &[SamplingConfig],
    cancel: &[Option<usize>],
    in_pass: Option<(usize, &Arc<EngineRuntime>, u64)>,
) -> (Vec<Recording>, Duration) {
    let began = Instant::now();
    let running: Vec<_> = prompts
        .iter()
        .enumerate()
        .map(|(index, prompt)| {
            submit_with(
                runtime,
                prompt,
                CONCURRENT_TOKENS,
                &[],
                speculation,
                sampling[index],
                cancel[index],
                in_pass
                    .filter(|(victim, _, _)| *victim == index)
                    .map(|(_, engine, passes)| (Arc::downgrade(engine), passes)),
            )
            .expect("a sequence is idle for the request")
        })
        .collect();
    // Every sequence runs a generation: one more request is refused, and only that one.
    let refused = submit(
        runtime,
        &prompts[0],
        CONCURRENT_TOKENS,
        &[],
        Speculation::Off,
        SamplingConfig::greedy(0),
        None,
    );
    assert_eq!(
        refused.err(),
        Some(AdmissionError::Busy),
        "the fifth request finds every sequence running"
    );
    let recordings = running
        .into_iter()
        .map(|done| done.wait(WAIT).expect("the generation reports"))
        .collect();
    (recordings, began.elapsed())
}

/// Sampled generation with the published settings gives the same tokens with speculation off and on (with drafts
/// partly rejected among the verifications), and the same tokens when it shares passes with other sessions as when it
/// runs alone; the seed is used.
#[test]
#[ignore = "needs a GPU, the native library and the RCO artifact"]
fn sampled_tokens_do_not_depend_on_speculation_or_on_the_sequences_sharing_a_pass() {
    const NEW_TOKENS: u32 = 64;
    let Some(machine) = machine() else { return };
    let tokenizer = tokenizer(&machine.artifact);
    let prompts: Vec<Vec<u32>> = CONCURRENT_PROMPTS
        .iter()
        .map(|text| tokenizer.encode_with_model_special_tokens(text).unwrap())
        .collect();
    let (owner, runtime) = start_with_sequences(&machine, SEQUENCES as u32);
    let depth = runtime.plan().draft_depth;
    assert!(depth > 0, "the artifact drafts");
    let published = |seed: u64| SamplingConfig::new(1.0, 20, 0.95, seed, false).unwrap();
    // Three sampled sessions and one greedy.
    let settings: Vec<SamplingConfig> = (0..SEQUENCES)
        .map(|index| {
            if index < 3 {
                published(11 + index as u64)
            } else {
                SamplingConfig::greedy(7)
            }
        })
        .collect();
    let run = |index: usize, speculation: Speculation, sampling: SamplingConfig| {
        submit(&runtime, &prompts[index], NEW_TOKENS, &[], speculation, sampling, None)
            .expect("the request is admitted")
    };

    let mut alone = Vec::new();
    let (mut verified, mut accepted, mut partial) = (0, 0, 0);
    for (index, sampling) in settings.iter().enumerate() {
        let (sampling, index) = (*sampling, index);
        let plain = run(index, Speculation::Off, sampling).wait(WAIT).unwrap();
        let speculative = run(index, Speculation::Artifact, sampling).wait(WAIT).unwrap();
        assert_eq!(
            plain.tokens, speculative.tokens,
            "request {index}: speculation changed the tokens"
        );
        if index < 3 {
            verified += speculative.verifications.len();
            accepted += speculative.verifications.iter().map(|v| v.accepted).sum::<usize>();
            partial += speculative
                .verifications
                .iter()
                .filter(|v| v.accepted > 0 && v.accepted < v.drafted)
                .count();
        }
        alone.push(speculative);
    }
    println!(
        "SAMPLED_PASSES depth {depth}: {verified} sampled verifications accepted {accepted} drafts, {partial} partly"
    );
    assert!(accepted > 0, "no sampled verification accepted a draft");
    assert!(partial > 0, "no sampled verification was partially rejected");

    // The seed is used.
    let other = run(0, Speculation::Artifact, published(99)).wait(WAIT).unwrap();
    assert_ne!(other.tokens, alone[0].tokens, "another seed gave the same tokens");

    // Sharing passes changes no session's tokens.
    let before = runtime.pass_statistics();
    let together: Vec<_> = (0..SEQUENCES)
        .map(|index| run(index, Speculation::Artifact, settings[index]))
        .collect();
    for (index, done) in together.into_iter().enumerate() {
        let together = done.wait(WAIT).expect("the generation reports");
        assert_eq!(
            together.tokens, alone[index].tokens,
            "session {index} changed when it shared passes"
        );
    }
    let after = runtime.pass_statistics();
    println!(
        "SAMPLED_PASSES together: {} passes, {} verifications, {} batched passes, {} rows",
        after.passes - before.passes,
        after.segments - before.segments,
        after.batched - before.batched,
        after.rows - before.rows
    );
    assert!(after.batched > before.batched, "no verification was batched");
    assert!(after.segments - before.segments > after.passes - before.passes);
    owner.shutdown(runtime).expect("the engine shuts down clean");
}

/// The decode rates of speculative requests of 512 prompt tokens and 512 new tokens: one request on an engine of one
/// sequence, one request alone on an engine of four, and four at once. Run it in a release build; it prints the rates.
#[test]
#[ignore = "a measurement: needs a GPU, the native library and the RCO artifact"]
fn speculative_decode_rates_of_one_and_of_four_requests() {
    const PROMPT_TOKENS: usize = 512;
    const NEW_TOKENS: u32 = 512;
    let Some(machine) = machine() else { return };
    let tokenizer = tokenizer(&machine.artifact);
    // Four different prompts of 512 tokens, from the texts the concurrent test uses.
    let prompts: Vec<Vec<u32>> = CONCURRENT_PROMPTS
        .iter()
        .enumerate()
        .map(|(index, text)| {
            let text = format!("{text} {}", CONCURRENT_PROMPTS[(index + 1) % SEQUENCES]);
            let seed = tokenizer.encode_with_model_special_tokens(&text).unwrap();
            seed.iter().copied().cycle().take(PROMPT_TOKENS).collect()
        })
        .collect();
    let sampling = SamplingConfig::greedy(0);
    let rate_of = |recordings: &[Recording]| -> f64 {
        let first = recordings
            .iter()
            .map(|r| *r.token_times.first().unwrap())
            .min()
            .unwrap();
        let last = recordings.iter().map(|r| *r.token_times.last().unwrap()).max().unwrap();
        let tokens: usize = recordings.iter().map(|r| r.tokens.len() - 1).sum();
        tokens as f64 / last.duration_since(first).as_secs_f64().max(1e-9)
    };
    let run = |runtime: &EngineRuntime, count: usize| -> Vec<Recording> {
        let running: Vec<_> = prompts[..count]
            .iter()
            .map(|prompt| {
                submit(runtime, prompt, NEW_TOKENS, &[], Speculation::Artifact, sampling, None)
                    .expect("a sequence is idle for the request")
            })
            .collect();
        running.into_iter().map(|done| done.wait(WAIT).unwrap()).collect()
    };
    // `EUHEDRAL_RATES_SEQUENCES` picks the engine (1 or 4 sequences) so that each measurement is a run of its own.
    let only: Option<usize> = env::var("EUHEDRAL_RATES_SEQUENCES")
        .ok()
        .and_then(|value| value.parse().ok());
    for sequences in [1, SEQUENCES]
        .into_iter()
        .filter(|count| only.is_none_or(|only| only == *count))
    {
        let (owner, runtime) = start_with_sequences(&machine, sequences as u32);
        assert!(runtime.plan().draft_depth > 0, "the artifact drafts");
        let _ = run(&runtime, 1);
        let alone = run(&runtime, 1);
        println!(
            "RATES engine of {sequences} sequence(s), one request alone: {:.1} tok/s",
            decode_rate(&alone[0])
        );
        if sequences > 1 {
            let _ = run(&runtime, SEQUENCES);
            let before = runtime.pass_statistics();
            let together = run(&runtime, SEQUENCES);
            let after = runtime.pass_statistics();
            let each: Vec<String> = together.iter().map(|r| format!("{:.1}", decode_rate(r))).collect();
            println!(
                "RATES {SEQUENCES} requests at once: {} tok/s each, {:.1} tok/s in all; {} passes held {} verifications, \
                 {} batched passes",
                each.join(" / "),
                rate_of(&together),
                after.passes - before.passes,
                after.segments - before.segments,
                after.batched - before.batched
            );
        }
        owner.shutdown(runtime).expect("the engine shuts down clean");
    }
}

#[test]
#[ignore = "needs a GPU, the native library and the RCO artifact"]
fn concurrent_sequences_match_alone() {
    let Some(machine) = machine() else { return };
    let tokenizer = tokenizer(&machine.artifact);
    let prompts: Vec<Vec<u32>> = CONCURRENT_PROMPTS
        .iter()
        .map(|text| tokenizer.encode_with_model_special_tokens(text).unwrap())
        .collect();
    let (owner, runtime) = start_with_sequences(&machine, SEQUENCES as u32);
    assert_eq!(runtime.sequences(), SEQUENCES);
    let greedy = [SamplingConfig::greedy(0); SEQUENCES];
    let seeded: Vec<SamplingConfig> = (0..SEQUENCES).map(|index| sampled(40, 7 + index as u64)).collect();

    let modes: [(&str, Speculation, &[SamplingConfig]); 4] = [
        ("greedy", Speculation::Off, &greedy),
        ("sampled", Speculation::Off, &seeded),
        ("greedy+speculation", Speculation::Artifact, &greedy),
        ("sampled+speculation", Speculation::Artifact, &seeded),
    ];
    let mut alone_greedy = Vec::new();
    let mut alone_speculative = Vec::new();
    for (name, speculation, sampling) in modes {
        let speculation = if speculation == Speculation::Artifact && runtime.plan().draft_depth == 0 {
            println!("{name}: skipped, the artifact has no draft layer");
            continue;
        } else {
            speculation
        };
        // Each request alone, in turn.
        let alone: Vec<Recording> = prompts
            .iter()
            .zip(sampling)
            .map(|(prompt, sampling)| {
                submit(&runtime, prompt, CONCURRENT_TOKENS, &[], speculation, *sampling, None)
                    .expect("the request is admitted")
                    .wait(WAIT)
                    .expect("the generation reports")
            })
            .collect();
        // A round that is not measured: it records the graphs of the sequences that have not run yet.
        let _ = together(&runtime, &prompts, speculation, sampling, &[None; SEQUENCES]);
        let before = runtime.pass_statistics();
        let (both, wall) = together(&runtime, &prompts, speculation, sampling, &[None; SEQUENCES]);
        let passes = runtime.pass_statistics();
        if speculation == Speculation::Artifact {
            // The verifications of concurrent requests ran as shared passes.
            let (ran, held, batched) = (
                passes.passes - before.passes,
                passes.segments - before.segments,
                passes.batched - before.batched,
            );
            println!("PASSES {name}: {ran} passes held {held} verifications, {batched} batched passes");
            assert!(batched > 0, "{name}: no verification was batched");
            assert!(held > ran, "{name}: every pass held one verification");
        }
        for (index, (alone, both)) in alone.iter().zip(&both).enumerate() {
            assert_eq!(both.outcome.reason, FinishReason::Length, "{name}: request {index}");
            assert_eq!(
                alone.tokens.len(),
                CONCURRENT_TOKENS as usize,
                "{name}: request {index} alone"
            );
            assert_eq!(
                both.tokens, alone.tokens,
                "{name}: request {index} changed when run together"
            );
        }
        // The requests really ran side by side: every one had its first token before any had its last.
        let first_of_all = both.iter().map(|r| *r.token_times.first().unwrap()).max().unwrap();
        let first_to_end = both.iter().map(|r| *r.token_times.last().unwrap()).min().unwrap();
        assert!(first_of_all < first_to_end, "{name}: the requests did not overlap");
        let alone_rate = alone.iter().map(decode_rate).sum::<f64>() / SEQUENCES as f64;
        let each_rate = both.iter().map(decode_rate).sum::<f64>() / SEQUENCES as f64;
        let all_tokens = both.iter().map(|r| r.tokens.len()).sum::<usize>() as f64;
        println!(
            "CONCURRENT {name}: alone {alone_rate:.1} tok/s each; together {each_rate:.1} tok/s each, {:.1} tok/s in all \
             ({:.0} ms for {all_tokens} tokens)",
            all_tokens / wall.as_secs_f64(),
            wall.as_secs_f64() * 1e3
        );
        if name == "greedy" {
            alone_greedy = alone;
        } else if name == "greedy+speculation" {
            alone_speculative = alone;
        }
    }

    // One of the four is cancelled mid-decode: the others' tokens stay what they are alone.
    let cancelled_at = 12;
    let (cancelled, _) = together(
        &runtime,
        &prompts,
        Speculation::Off,
        &greedy,
        &[None, Some(cancelled_at), None, None],
    );
    assert_eq!(cancelled[1].outcome.reason, FinishReason::Cancelled);
    assert!(
        cancelled[1].tokens.len() >= cancelled_at && cancelled[1].tokens.len() < CONCURRENT_TOKENS as usize,
        "the cancelled request stopped mid-decode, after {} tokens",
        cancelled[1].tokens.len()
    );
    assert_eq!(cancelled[1].tokens, alone_greedy[1].tokens[..cancelled[1].tokens.len()]);
    for index in [0, 2, 3] {
        assert_eq!(cancelled[index].outcome.reason, FinishReason::Length, "request {index}");
        assert_eq!(
            cancelled[index].tokens, alone_greedy[index].tokens,
            "request {index} next to a cancel"
        );
    }

    // The same with speculation on: the cancelled member leaves its pass, and the others' verifications go on batched.
    if !alone_speculative.is_empty() {
        let before = runtime.pass_statistics();
        // The request is cancelled by the retirement of a verification of one of its batched passes, once three more
        // batched passes have run: its next step is offered to a pass with the others' and stops there.
        let (cancelled, _) = together_with(
            &runtime,
            &prompts,
            Speculation::Artifact,
            &greedy,
            &[None; SEQUENCES],
            Some((1, &runtime, before.batched + 3)),
        );
        assert_eq!(cancelled[1].outcome.reason, FinishReason::Cancelled);
        assert!(
            cancelled[1].tokens.len() < CONCURRENT_TOKENS as usize,
            "the cancelled request stopped mid-decode"
        );
        assert_eq!(
            cancelled[1].tokens,
            alone_speculative[1].tokens[..cancelled[1].tokens.len()]
        );
        for index in [0, 2, 3] {
            assert_eq!(cancelled[index].outcome.reason, FinishReason::Length, "request {index}");
            assert_eq!(
                cancelled[index].tokens, alone_speculative[index].tokens,
                "speculative request {index} next to a cancel"
            );
        }
        assert!(runtime.pass_statistics().batched > before.batched);
    }

    // One of the four fails (its owner panics at the third admission, the hook that takes the first request): the
    // others' tokens stay what they are alone.
    runtime.panic_after_admissions_for_tests(3);
    let (failed, _) = together(&runtime, &prompts, Speculation::Off, &greedy, &[None; SEQUENCES]);
    match &failed[0].outcome.reason {
        FinishReason::Failed(why) => assert!(why.contains("panicked"), "{why}"),
        other => panic!("the first request should have failed, ended with {other:?}"),
    }
    assert_eq!(failed[0].tokens, alone_greedy[0].tokens[..failed[0].tokens.len()]);
    for index in [1, 2, 3] {
        assert_eq!(failed[index].outcome.reason, FinishReason::Length, "request {index}");
        assert_eq!(
            failed[index].tokens, alone_greedy[index].tokens,
            "request {index} next to a failure"
        );
    }

    // Every sequence went back to the pool: all four are idle, and each is clean.
    for index in 0..SEQUENCES {
        let state = runtime
            .inspect_sequence_at(index, |sequence| sequence.row_readbacks.leased())
            .expect("the sequence is idle");
        assert_eq!(state, 0, "sequence {index} holds no row readback");
    }
    let (after, _) = together(&runtime, &prompts, Speculation::Off, &greedy, &[None; SEQUENCES]);
    for (index, recording) in after.iter().enumerate() {
        assert_eq!(
            recording.tokens, alone_greedy[index].tokens,
            "request {index} after the failure"
        );
    }
    owner.shutdown(runtime).expect("four sequences shut down clean");
}
