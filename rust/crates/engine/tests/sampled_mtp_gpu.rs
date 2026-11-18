//! Sampled requests that speculate on a GPU: with the artifact's draft layer, a request's tokens for a seed are the
//! tokens it has without speculation. The tests are `#[ignore]`d and run locally, one at a time (each loads the 12 GB
//! artifact), through the system arbiter:
//! `system-run --gpu shared:16 --mem 24G --est 600 --who <name> -- cargo test -p euhedral-inference-engine --test sampled_mtp_gpu -- --ignored --test-threads=1`.
//!
//! The drafts are greedy; each verified row is sampled at the position of the token it chooses, so a draft is kept only
//! when the sampler would have drawn it. The prompts are the chat corpus of the benchmark crate. The artifact comes from
//! `euhedral.qwen.rco-artifact` and the library from `euhedral.cuda.library`.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]
#![allow(clippy::print_stdout, reason = "a test reports the counts it measured")]
#![allow(
    clippy::disallowed_methods,
    reason = "no-wait, blocking I/O and waiting calls: a test reads files and waits on hand-offs and the device"
)]

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::mem;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use euhedral_inference_bench::prompt::PromptCorpus;
use euhedral_inference_bench::prompt::words::DEFAULT_SEED;
use euhedral_inference_edrl::{ReadPlan, Region, TokenizerFooter, TokenizerSection};
use euhedral_inference_engine::generation::{
    CancelHandle, FinishReason, GenerationListener, GenerationOutcome, GenerationRequest, QuantumEvent, Speculation,
    Verification,
};
use euhedral_inference_engine::sampling::SamplingConfig;
use euhedral_inference_engine::stages::QuantumKind;
use euhedral_inference_engine::{EngineRuntime, Handoff, RuntimeOwner};

/// The longest a device test waits for one hand-off; past it the test fails.
const WAIT: Duration = Duration::from_secs(20 * 60);
use euhedral_inference_settings::{ModelSettings, Settings};
use euhedral_inference_tokenizer::Tokenizer;

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

/// Rows of a prompt chunk: a long prompt takes several.
const CHUNK: u32 = 256;

fn start(machine: &Machine, max_tokens: u32) -> (Arc<EngineRuntime>, RuntimeOwner) {
    let mut settings = Settings::with_model(ModelSettings {
        artifact: machine.artifact.clone(),
        name: "test".to_string(),
    });
    settings.context.max_tokens = max_tokens;
    settings.kernels.graphs.captures_per_view = 8;
    settings.batching.prefill_chunk_tokens = CHUNK;
    let (owner, ready) = EngineRuntime::start(&settings, &machine.library);
    (
        ready
            .wait(WAIT)
            .expect("the engine reports")
            .expect("the engine starts"),
        owner,
    )
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

/// The ids of the four chat prompts of a target, built once.
fn chat_prompts(tokenizer: &Tokenizer, target: usize) -> Vec<Vec<u32>> {
    PromptCorpus::Chat
        .prompts(target, DEFAULT_SEED, tokenizer)
        .unwrap()
        .into_iter()
        .map(|prompt| prompt.ids)
        .collect()
}

/// What a generation told its listener.
#[derive(Debug)]
struct Recording {
    tokens: Vec<u32>,
    verifications: Vec<Verification>,
    outcome: GenerationOutcome,
}

/// Where a test's cancel is asked for, from inside the listener.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Trigger {
    Never,
    /// At the report of the `n`th verification quantum, whose rows the host has yet to select.
    Verification(usize),
}

struct Recorder {
    tokens: Vec<u32>,
    verifications: Vec<Verification>,
    quanta: Vec<QuantumKind>,
    trigger: Trigger,
    cancel: CancelHandle,
    ended: bool,
    done: Option<Arc<Handoff<Recording>>>,
}

impl GenerationListener for Recorder {
    fn prefill_done(&mut self, _at: Instant) {}

    fn token(&mut self, id: u32, _at: Instant) {
        assert!(!self.ended, "no call follows `finished`");
        self.tokens.push(id);
    }

    fn quantum(&mut self, event: QuantumEvent) {
        assert!(!self.ended, "no call follows `finished`");
        self.quanta.push(event.kind);
        if let Trigger::Verification(n) = self.trigger
            && event.kind == QuantumKind::Verify
            && self.quanta.iter().filter(|kind| **kind == QuantumKind::Verify).count() == n
        {
            self.cancel.cancel();
        }
    }

    fn verification(&mut self, event: Verification) {
        assert!(!self.ended, "no call follows `finished`");
        self.verifications.push(event);
    }

    fn finished(&mut self, outcome: GenerationOutcome) {
        assert!(!self.ended, "`finished` is called once");
        self.ended = true;
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
                },
            });
        }
    }
}

/// A generation with the sampling configuration `sampling` and the cancel `trigger`.
fn run(
    runtime: &EngineRuntime,
    prompt: &[u32],
    max_tokens: u32,
    stop_tokens: &[u32],
    speculation: Speculation,
    sampling: SamplingConfig,
    trigger: Trigger,
) -> Recording {
    let done = Handoff::new();
    let cancel = CancelHandle::default();
    runtime
        .generate_cancellable(
            GenerationRequest {
                prompt: prompt.to_vec(),
                max_tokens,
                stop_tokens: stop_tokens.to_vec(),
                speculation,
                sampling,
                constraint: None,
                listener: Box::new(Recorder {
                    tokens: Vec::new(),
                    verifications: Vec::new(),
                    quanta: Vec::new(),
                    trigger,
                    cancel: cancel.clone(),
                    ended: false,
                    done: Some(Arc::clone(&done)),
                }),
            },
            cancel,
        )
        .expect("the request is admitted");
    done.wait(WAIT).expect("the generation reports")
}

fn generate(
    runtime: &EngineRuntime,
    prompt: &[u32],
    max_tokens: u32,
    stop_tokens: &[u32],
    speculation: Speculation,
    sampling: SamplingConfig,
) -> Recording {
    run(
        runtime,
        prompt,
        max_tokens,
        stop_tokens,
        speculation,
        sampling,
        Trigger::Never,
    )
}

/// Committed rows of the base cache, quanta in flight, selections and row readbacks leased after a generation.
fn sequence_state(runtime: &EngineRuntime) -> (u64, usize, usize, usize) {
    runtime
        .inspect_sequence(|sequence| {
            (
                sequence.kv.committed(),
                sequence.in_flight(),
                sequence.selections.leased(),
                sequence.row_readbacks.leased(),
            )
        })
        .expect("the generation gave the sequence back")
}

/// The default sampling of a chat request.
fn chat(seed: u64) -> SamplingConfig {
    SamplingConfig::new(1.0, 20, 0.95, seed, false).unwrap()
}

/// A flattened distribution, in which the position of a draw changes the token it picks.
fn flat(seed: u64) -> SamplingConfig {
    SamplingConfig::new(50.0, 40, 1.0, seed, false).unwrap()
}

/// Runs `sampling` with and without speculation and requires the same tokens, the same end and the same state of the
/// sequence; returns the speculative recording.
fn same_with_and_without_speculation(
    runtime: &EngineRuntime,
    prompt: &[u32],
    max_tokens: u32,
    stop_tokens: &[u32],
    sampling: SamplingConfig,
    what: &str,
) -> Recording {
    let off = generate(runtime, prompt, max_tokens, stop_tokens, Speculation::Off, sampling);
    let off_state = sequence_state(runtime);
    let on = generate(
        runtime,
        prompt,
        max_tokens,
        stop_tokens,
        Speculation::Artifact,
        sampling,
    );
    let on_state = sequence_state(runtime);
    assert_eq!(off.outcome.reason, on.outcome.reason, "{what}: how it ended");
    assert_eq!(off.tokens, on.tokens, "{what}: the tokens");
    assert_eq!(off.outcome.generated, on.outcome.generated, "{what}: the count");
    assert!(
        !on.verifications.is_empty() || on.tokens.len() <= 1,
        "{what}: it speculated"
    );
    assert_eq!(
        (on_state.1, on_state.2, on_state.3),
        (0, 0, 0),
        "{what}: nothing left held"
    );
    assert_eq!(
        (off_state.1, off_state.2, off_state.3),
        (0, 0, 0),
        "{what}: nothing left held"
    );
    on
}

#[test]
#[ignore = "needs a GPU, the native library and the RCO artifact"]
fn sampled_speculation_gives_the_tokens_of_decode_without_it() {
    let Some(machine) = machine() else { return };
    let tokenizer = tokenizer(&machine.artifact);
    let eos = tokenizer.eos_ids().to_vec();
    let (runtime, owner) = start(&machine, 4096);
    assert!(runtime.plan().draft_depth > 0, "the artifact drafts");
    let mut accepted_total = 0;
    let mut verified_total = 0;
    // A short prompt, and one of several chunks.
    for target in [128, 700] {
        let prompts = chat_prompts(&tokenizer, target);
        for (index, prompt) in prompts.iter().take(2).enumerate() {
            for (name, sampling) in [("chat", chat(7 + index as u64)), ("flat", flat(9 + index as u64))] {
                let what = format!("{name} sampling, prompt {index} of about {target} tokens");
                let on = same_with_and_without_speculation(&runtime, prompt, 192, &eos, sampling, &what);
                verified_total += on.verifications.len();
                accepted_total += on.verifications.iter().map(|v| v.accepted).sum::<usize>();
                assert!(on.verifications.len() > 1, "{what}: several verifications");
            }
        }
    }
    println!("{verified_total} verifications confirmed {accepted_total} drafts");
    assert!(accepted_total > 0, "some draft was a token the sampler drew");

    // Another seed is another output, equal with and without speculation too.
    let prompt = &chat_prompts(&tokenizer, 128)[0];
    let first = same_with_and_without_speculation(&runtime, prompt, 96, &eos, chat(1), "seed 1");
    let second = same_with_and_without_speculation(&runtime, prompt, 96, &eos, chat(2), "seed 2");
    println!("seeds 1 and 2 differ: {}", first.tokens != second.tokens);

    // A stop token inside what a verification accepts ends the output there, as it does without speculation, whichever
    // verification and row it falls on.
    let base = generate(&runtime, prompt, 96, &[], Speculation::Off, chat(1)).tokens;
    for index in [1usize, 2, 3, 5, 8, 13, 21, 34] {
        let stop = base[index];
        if base[..index].contains(&stop) {
            continue;
        }
        let on = same_with_and_without_speculation(&runtime, prompt, 96, &[stop], chat(1), &format!("stop at {index}"));
        assert_eq!(on.outcome.reason, FinishReason::Stop);
        assert_eq!(on.tokens, base[..=index], "the stop token is the last, and reported");
    }

    // A budget that ends inside a verification cuts the accepted tokens there.
    for budget in [1u32, 2, 3, 4, 5, 6, 7, 9, 11] {
        let on = same_with_and_without_speculation(&runtime, prompt, budget, &[], chat(1), &format!("budget {budget}"));
        assert_eq!(on.outcome.reason, FinishReason::Length);
        assert_eq!(on.tokens, base[..budget as usize]);
    }
    owner.shutdown(runtime).unwrap();
}

#[test]
#[ignore = "needs a GPU, the native library and the RCO artifact"]
fn a_cancel_during_a_sampled_verification_frees_the_sequence_and_its_rows() {
    let Some(machine) = machine() else { return };
    let tokenizer = tokenizer(&machine.artifact);
    let (runtime, owner) = start(&machine, 4096);
    let prompt = &chat_prompts(&tokenizer, 128)[0];
    let sampling = chat(3);
    let uncut = generate(&runtime, prompt, 96, &[], Speculation::Artifact, sampling);
    assert_eq!(uncut.outcome.reason, FinishReason::Length);
    for nth in [1, 2, 4] {
        // The cancel is asked for as the verification's quantum is reported: the host's selections of its rows are
        // published after it, and answer into a request that ended.
        let cut = run(
            &runtime,
            prompt,
            96,
            &[],
            Speculation::Artifact,
            sampling,
            Trigger::Verification(nth),
        );
        assert_eq!(
            cut.outcome.reason,
            FinishReason::Cancelled,
            "cancel at verification {nth}"
        );
        assert!(
            uncut.tokens.starts_with(&cut.tokens),
            "the tokens reported are the uncut run's"
        );
        assert!(cut.tokens.len() < uncut.tokens.len());
        let (_, in_flight, selections, rows) = sequence_state(&runtime);
        assert_eq!(
            (in_flight, selections, rows),
            (0, 0, 0),
            "the sequence and its rows are free"
        );
        let again = generate(&runtime, prompt, 96, &[], Speculation::Artifact, sampling);
        assert_eq!(again.tokens, uncut.tokens, "the engine is whole after the cancel");
    }
    owner.shutdown(runtime).unwrap();
}
