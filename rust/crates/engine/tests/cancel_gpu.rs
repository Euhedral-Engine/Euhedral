//! Cancelling a generation on a GPU. The tests are `#[ignore]`d and run locally, one at a time (each loads the 12 GB
//! artifact), through the system arbiter:
//! `system-run --gpu shared:16 --mem 24G --est 300 --who <name> -- cargo test -p euhedral-inference-engine --test cancel_gpu -- --ignored --test-threads=1`.
//!
//! A generation is cancelled at a point its listener sees (a token, a prompt chunk's quantum) or from the test's thread.
//! Each time the generation must end through its listener once, as `Cancelled`, with the tokens it had reported (a prefix
//! of what the generation selects uncancelled), and leave the sequence free: the next generation on the same engine
//! selects the tokens of a generation that no cancel preceded. The artifact comes from `euhedral.qwen.rco-artifact` and
//! the library from `euhedral.cuda.library`.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]
#![allow(clippy::print_stdout, reason = "a test says why it skipped")]
#![allow(
    clippy::disallowed_methods,
    reason = "no-wait, blocking I/O and waiting calls: a test waits on hand-offs"
)]

use std::mem;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use euhedral_inference_engine::generation::{
    CancelHandle, FinishReason, GenerationListener, GenerationOutcome, GenerationRequest, QuantumEvent, Speculation,
    Verification,
};
use euhedral_inference_engine::sampling::SamplingConfig;
use euhedral_inference_engine::stages::QuantumKind;
use euhedral_inference_engine::{EngineRuntime, Handoff, RuntimeOwner};
use euhedral_inference_settings::{ModelSettings, Settings};

/// The longest a device test waits for one hand-off; past it the test fails.
const WAIT: Duration = Duration::from_secs(20 * 60);
/// Tokens of the reference generation.
const TOKENS: u32 = 64;
/// Rows of a prompt chunk.
const PREFILL_CHUNK: usize = 256;
/// Tokens of the long prompt: twelve prompt chunks.
const LONG_PROMPT: usize = 12 * PREFILL_CHUNK;

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

fn start(machine: &Machine) -> (Arc<EngineRuntime>, RuntimeOwner) {
    let mut settings = Settings::with_model(ModelSettings {
        artifact: machine.artifact.clone(),
        name: "test".to_string(),
    });
    settings.context.max_tokens = 4096;
    settings.kernels.graphs.captures_per_view = 8;
    // The long prompt takes many chunks of this size, whatever the default is.
    settings.batching.prefill_chunk_tokens = PREFILL_CHUNK as u32;
    let (owner, ready) = EngineRuntime::start(&settings, &machine.library);
    (
        ready
            .wait(WAIT)
            .expect("the engine reports")
            .expect("the engine starts"),
        owner,
    )
}

/// A prompt of `length` token ids inside any vocabulary.
fn prompt(length: usize) -> Vec<u32> {
    (0..length).map(|index| 1000 + ((index * 7919) % 5000) as u32).collect()
}

/// Where a test's cancel is asked for, from inside the listener.
#[derive(Clone, Copy, Debug)]
enum Trigger {
    /// Never.
    Never,
    /// At the report of the `n`th token.
    Token(usize),
    /// At the report of the `n`th prompt chunk's quantum.
    PromptChunk(usize),
    /// At the first draft quantum (the catch-up or a recursive draft of a speculative step) that retires after a
    /// verification was reported: the step's rows of the MTP cache are committed and its verification is still to come,
    /// and is discarded.
    StepDraft,
}

/// What a generation told its listener.
#[derive(Debug)]
struct Recording {
    tokens: Vec<u32>,
    prompt_chunks: usize,
    quanta: usize,
    outcome: GenerationOutcome,
}

struct Recorder {
    tokens: Vec<u32>,
    prompt_chunks: usize,
    quanta: usize,
    verified: bool,
    trigger: Trigger,
    cancel: CancelHandle,
    ended: bool,
    done: Option<Arc<Handoff<Recording>>>,
}

impl Recorder {
    fn cancel_here(&self) {
        self.cancel.cancel();
        // Twice, and the flag reads back: idempotent, never waits.
        self.cancel.cancel();
        assert!(self.cancel.is_cancelled());
    }
}

impl GenerationListener for Recorder {
    fn prefill_done(&mut self, _at: Instant) {
        assert!(!self.ended, "no call follows `finished`");
    }

    fn token(&mut self, id: u32, _at: Instant) {
        assert!(!self.ended, "no call follows `finished`");
        self.tokens.push(id);
        if matches!(self.trigger, Trigger::Token(n) if n == self.tokens.len()) {
            self.cancel_here();
        }
    }

    fn quantum(&mut self, event: QuantumEvent) {
        assert!(!self.ended, "no call follows `finished`");
        self.quanta += 1;
        match event.kind {
            QuantumKind::Prefill => {
                self.prompt_chunks += 1;
                if matches!(self.trigger, Trigger::PromptChunk(n) if n == self.prompt_chunks) {
                    self.cancel_here();
                }
            }
            QuantumKind::Draft if self.verified && matches!(self.trigger, Trigger::StepDraft) => self.cancel_here(),
            _ => {}
        }
    }

    fn verification(&mut self, _event: Verification) {
        assert!(!self.ended, "no call follows `finished`");
        self.verified = true;
    }

    fn finished(&mut self, outcome: GenerationOutcome) {
        assert!(!self.ended, "`finished` is called once");
        self.ended = true;
        let done = self.done.take().expect("a generation finishes once");
        done.send(Recording {
            tokens: mem::take(&mut self.tokens),
            prompt_chunks: self.prompt_chunks,
            quanta: self.quanta,
            outcome,
        });
    }
}

/// Runs a generation whose listener cancels `cancel` at `trigger`, and waits for its recording.
fn run(
    runtime: &EngineRuntime,
    prompt: &[u32],
    speculation: Speculation,
    trigger: Trigger,
    cancel: &CancelHandle,
) -> Recording {
    run_sampling(runtime, prompt, speculation, trigger, cancel, SamplingConfig::greedy(0))
}

/// [`run`] with the sampling configuration `sampling`.
fn run_sampling(
    runtime: &EngineRuntime,
    prompt: &[u32],
    speculation: Speculation,
    trigger: Trigger,
    cancel: &CancelHandle,
    sampling: SamplingConfig,
) -> Recording {
    let done = Handoff::new();
    runtime
        .generate_cancellable(
            GenerationRequest {
                prompt: prompt.to_vec(),
                max_tokens: TOKENS,
                stop_tokens: Vec::new(),
                speculation,
                sampling,
                constraint: None,
                listener: Box::new(Recorder {
                    tokens: Vec::new(),
                    prompt_chunks: 0,
                    quanta: 0,
                    verified: false,
                    trigger,
                    cancel: cancel.clone(),
                    ended: false,
                    done: Some(Arc::clone(&done)),
                }),
            },
            cancel.clone(),
        )
        .expect("the request is admitted");
    done.wait(WAIT).expect("the generation reports")
}

/// A generation with a handle of its own that nobody cancels but `trigger`.
fn run_with(runtime: &EngineRuntime, prompt: &[u32], speculation: Speculation, trigger: Trigger) -> Recording {
    run(runtime, prompt, speculation, trigger, &CancelHandle::default())
}

/// The sequence is free: nothing in flight and no selection leased.
fn assert_free(runtime: &EngineRuntime) {
    let (in_flight, leased) = runtime
        .inspect_sequence(|sequence| {
            (
                sequence.in_flight(),
                sequence.selections.leased() + sequence.row_readbacks.leased(),
            )
        })
        .expect("the generation gave the sequence back");
    assert_eq!(
        (in_flight, leased),
        (0, 0),
        "no quantum in flight, no selection or row leased"
    );
}

/// The generation ended cancelled with a prefix of `reference`, strictly shorter than it.
fn assert_cancelled(recording: &Recording, reference: &[u32], prompt_tokens: usize) {
    assert_eq!(recording.outcome.reason, FinishReason::Cancelled);
    assert_eq!(recording.outcome.prompt_tokens, prompt_tokens);
    assert_eq!(recording.outcome.generated, recording.tokens.len());
    assert!(
        recording.tokens.len() < TOKENS as usize,
        "the cancel cut the generation"
    );
    assert_eq!(
        recording.tokens,
        reference[..recording.tokens.len()],
        "the tokens reported are the reference's"
    );
}

/// The next generations on the engine, greedy and speculative, select the reference's tokens.
fn assert_recovered(runtime: &EngineRuntime, prompt: &[u32], reference: &[u32], after: &str) {
    assert_free(runtime);
    for speculation in [Speculation::Off, Speculation::Artifact] {
        let again = run_with(runtime, prompt, speculation, Trigger::Never);
        assert_eq!(
            again.outcome.reason,
            FinishReason::Length,
            "{speculation:?} after {after}"
        );
        assert_eq!(again.tokens, reference, "{speculation:?} after {after}");
    }
}

/// The cancels `speculation` allows, each followed by full generations that equal the reference.
fn cancel_all_ways(
    runtime: &EngineRuntime,
    speculation: Speculation,
    short: &[u32],
    long: &[u32],
    references: &[Vec<u32>; 2],
) {
    let [short_reference, long_reference] = references;

    // Mid-decode, from inside the listener.
    let cut = run_with(runtime, short, speculation, Trigger::Token(8));
    assert_cancelled(&cut, short_reference, short.len());
    assert!(cut.tokens.len() >= 8);
    assert_recovered(runtime, short, short_reference, "a cancel mid-decode");

    // Mid-prefill: the long prompt takes twelve chunks and the cancel comes at the second one's report.
    let cut = run_with(runtime, long, speculation, Trigger::PromptChunk(2));
    assert_eq!(cut.outcome.reason, FinishReason::Cancelled);
    assert_eq!(cut.outcome.generated, 0, "the prompt did not end: nothing was selected");
    assert!(cut.tokens.is_empty());
    assert!(
        cut.prompt_chunks >= 2 && cut.prompt_chunks < LONG_PROMPT / PREFILL_CHUNK,
        "the cancel cut the prefill"
    );
    assert_recovered(runtime, long, long_reference, "a cancel mid-prefill");

    // Part-way through a speculative step.
    if matches!(speculation, Speculation::Artifact) {
        let cut = run_with(runtime, short, speculation, Trigger::StepDraft);
        assert_cancelled(&cut, short_reference, short.len());
        assert!(!cut.tokens.is_empty(), "a verification was reported before the cancel");
        assert_recovered(runtime, short, short_reference, "a cancel inside a speculative step");
    }
}

/// Cancels in every way `speculation` allows: before the first quantum, mid-decode, mid-prefill and (speculating)
/// inside a step, with captured graphs and without; after each, full generations equal one no cancel preceded.
fn cancels_and_recovers(speculation: Speculation) {
    let Some(machine) = machine() else { return };
    let (runtime, owner) = start(&machine);
    let short = prompt(40);
    let long = prompt(LONG_PROMPT);

    // The references come first, before any cancel.
    let short_reference = run_with(&runtime, &short, Speculation::Off, Trigger::Never);
    assert_eq!(short_reference.outcome.reason, FinishReason::Length);
    assert_eq!(short_reference.tokens.len(), TOKENS as usize);
    let long_reference = run_with(&runtime, &long, Speculation::Off, Trigger::Never);
    assert_eq!(long_reference.tokens.len(), TOKENS as usize);
    let references = [short_reference.tokens, long_reference.tokens];

    // A handle cancelled before the request: the generation ends without running a quantum.
    let cancelled = CancelHandle::default();
    cancelled.cancel();
    let cut = run(&runtime, &short, speculation, Trigger::Never, &cancelled);
    assert_eq!(cut.outcome.reason, FinishReason::Cancelled);
    assert_eq!(cut.outcome.generated, 0);
    assert_eq!(cut.quanta, 0, "no quantum ran");
    assert!(cut.tokens.is_empty());
    assert_recovered(&runtime, &short, &references[0], "a cancel before the first quantum");

    cancel_all_ways(&runtime, speculation, &short, &long, &references);

    // A cancel after the end is a no-op, and a second cancel changes nothing.
    let spent = CancelHandle::default();
    let finished = run(&runtime, &short, speculation, Trigger::Never, &spent);
    assert_eq!(finished.outcome.reason, FinishReason::Length);
    spent.cancel();
    spent.cancel();
    assert_recovered(&runtime, &short, &references[0], "a cancel after the end");

    // The same without captured graphs.
    assert!(runtime.disable_graphs_for_tests(), "no request runs");
    cancel_all_ways(&runtime, speculation, &short, &long, &references);
    owner.shutdown(runtime).unwrap();
}

#[test]
#[ignore = "needs a GPU, the native library and the RCO artifact"]
fn a_cancel_ends_a_greedy_generation_and_frees_the_sequence() {
    cancels_and_recovers(Speculation::Off);
}

#[test]
#[ignore = "needs a GPU, the native library and the RCO artifact"]
fn a_cancel_ends_a_speculative_generation_and_frees_the_sequence() {
    cancels_and_recovers(Speculation::Artifact);
}

#[test]
#[ignore = "needs a GPU, the native library and the RCO artifact"]
fn a_cancel_ends_a_sampled_generation_and_frees_the_sequence() {
    let Some(machine) = machine() else { return };
    let (runtime, owner) = start(&machine);
    let short = prompt(40);
    let sampling = SamplingConfig::new(1.0, 40, 1.0, 3, false).unwrap();
    let uncut = run_sampling(
        &runtime,
        &short,
        Speculation::Off,
        Trigger::Never,
        &CancelHandle::default(),
        sampling,
    );
    assert_eq!(uncut.outcome.reason, FinishReason::Length);
    assert_eq!(uncut.tokens.len(), TOKENS as usize);

    // Mid-decode, from the listener, while a selection may be waiting: the tokens reported are the uncut run's.
    for _ in 0..2 {
        let cut = run_sampling(
            &runtime,
            &short,
            Speculation::Off,
            Trigger::Token(8),
            &CancelHandle::default(),
            sampling,
        );
        assert_cancelled(&cut, &uncut.tokens, short.len());
        assert!(cut.tokens.len() >= 8);
        assert_free(&runtime);
    }

    // Before the first quantum.
    let cancelled = CancelHandle::default();
    cancelled.cancel();
    let cut = run_sampling(&runtime, &short, Speculation::Off, Trigger::Never, &cancelled, sampling);
    assert_eq!(cut.outcome.reason, FinishReason::Cancelled);
    assert_eq!(cut.quanta, 0);
    assert_free(&runtime);

    // The engine is whole: the same request gives the same tokens as before.
    let again = run_sampling(
        &runtime,
        &short,
        Speculation::Off,
        Trigger::Never,
        &CancelHandle::default(),
        sampling,
    );
    assert_eq!(again.tokens, uncut.tokens);
    owner.shutdown(runtime).unwrap();
}
