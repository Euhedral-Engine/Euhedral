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
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use euhedral_inference_engine::generation::{
    CancelHandle, FinishReason, GenerationListener, GenerationOutcome, GenerationRequest, QuantumEvent, Speculation,
};
use euhedral_inference_engine::stages::QuantumKind;
use euhedral_inference_engine::{EngineRuntime, Handoff, RuntimeOwner};
use euhedral_inference_settings::{ModelSettings, Settings};

/// The longest a device test waits for one hand-off; past it the test fails.
const WAIT: Duration = Duration::from_secs(20 * 60);
/// Tokens of the reference generation.
const TOKENS: u32 = 64;
/// Tokens of the long prompt, which takes several prompt chunks.
const LONG_PROMPT: usize = 3000;

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
    /// Never: the test's thread cancels, or nobody does.
    Never,
    /// At the report of the `n`th token.
    Token(usize),
    /// At the report of the `n`th prompt chunk's quantum.
    PromptChunk(usize),
}

/// What a generation told its listener.
#[derive(Debug)]
struct Recording {
    tokens: Vec<u32>,
    prompt_chunks: usize,
    outcome: GenerationOutcome,
}

type Slot = Arc<OnceLock<CancelHandle>>;

struct Recorder {
    tokens: Vec<u32>,
    prompt_chunks: usize,
    trigger: Trigger,
    slot: Slot,
    ended: bool,
    done: Option<Arc<Handoff<Recording>>>,
}

impl Recorder {
    fn cancel_here(&self) {
        let handle = self
            .slot
            .get()
            .expect("the handle is set long before the second quantum retires");
        handle.cancel();
        // Twice, and the flag reads back: idempotent, never waits.
        handle.cancel();
        assert!(handle.is_cancelled());
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
        if event.kind == QuantumKind::Prefill {
            self.prompt_chunks += 1;
            if matches!(self.trigger, Trigger::PromptChunk(n) if n == self.prompt_chunks) {
                self.cancel_here();
            }
        }
    }

    fn finished(&mut self, outcome: GenerationOutcome) {
        assert!(!self.ended, "`finished` is called once");
        self.ended = true;
        let done = self.done.take().expect("a generation finishes once");
        done.send(Recording {
            tokens: mem::take(&mut self.tokens),
            prompt_chunks: self.prompt_chunks,
            outcome,
        });
    }
}

/// Starts a generation and returns its handle and the hand-off of its recording.
fn begin(
    runtime: &EngineRuntime,
    prompt: &[u32],
    speculation: Speculation,
    trigger: Trigger,
) -> (CancelHandle, Arc<Handoff<Recording>>) {
    let done = Handoff::new();
    let slot = Slot::default();
    let handle = runtime
        .generate_cancellable(GenerationRequest {
            prompt: prompt.to_vec(),
            max_tokens: TOKENS,
            stop_tokens: Vec::new(),
            speculation,
            listener: Box::new(Recorder {
                tokens: Vec::new(),
                prompt_chunks: 0,
                trigger,
                slot: Arc::clone(&slot),
                ended: false,
                done: Some(Arc::clone(&done)),
            }),
        })
        .expect("the request is admitted");
    slot.set(handle.clone()).expect("the handle is set once");
    (handle, done)
}

fn run(runtime: &EngineRuntime, prompt: &[u32], speculation: Speculation, trigger: Trigger) -> Recording {
    let (_, done) = begin(runtime, prompt, speculation, trigger);
    done.wait(WAIT).expect("the generation reports")
}

/// The sequence is free: nothing in flight and no selection leased.
fn assert_free(runtime: &EngineRuntime) {
    let (in_flight, leased) = runtime
        .inspect_sequence(|sequence| (sequence.in_flight(), sequence.selections.leased()))
        .expect("the generation gave the sequence back");
    assert_eq!((in_flight, leased), (0, 0), "no quantum in flight, no selection leased");
}

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

/// Cancels at a token and at a prompt chunk, with `speculation`; each time the next generation on the engine selects the
/// reference's tokens.
fn cancels_and_recovers(speculation: Speculation) {
    let Some(machine) = machine() else { return };
    let (runtime, owner) = start(&machine);
    let short = prompt(40);
    let long = prompt(LONG_PROMPT);

    // The references come first, before any cancel.
    let short_reference = run(&runtime, &short, Speculation::Off, Trigger::Never);
    assert_eq!(short_reference.outcome.reason, FinishReason::Length);
    assert_eq!(short_reference.tokens.len(), TOKENS as usize);
    let long_reference = run(&runtime, &long, Speculation::Off, Trigger::Never);
    assert_eq!(long_reference.tokens.len(), TOKENS as usize);

    // Mid-decode, from inside the listener.
    let cut = run(&runtime, &short, speculation, Trigger::Token(8));
    assert_cancelled(&cut, &short_reference.tokens, short.len());
    assert!(cut.tokens.len() >= 8);
    assert_free(&runtime);
    let again = run(&runtime, &short, speculation, Trigger::Never);
    assert_eq!(again.outcome.reason, FinishReason::Length);
    assert_eq!(again.tokens, short_reference.tokens, "after a cancel mid-decode");

    // Mid-prefill: the long prompt takes several chunks and the cancel comes at the second one's report.
    let cut = run(&runtime, &long, speculation, Trigger::PromptChunk(2));
    assert_eq!(cut.outcome.reason, FinishReason::Cancelled);
    assert_eq!(cut.outcome.generated, 0, "the prompt did not end: nothing was selected");
    assert!(cut.tokens.is_empty());
    assert!(cut.prompt_chunks >= 2);
    assert_free(&runtime);
    let again = run(&runtime, &long, speculation, Trigger::Never);
    assert_eq!(again.outcome.reason, FinishReason::Length);
    assert_eq!(again.tokens, long_reference.tokens, "after a cancel mid-prefill");

    // From the test's thread, at once (the generation takes tens of milliseconds): a second cancel and a cancel after
    // the end change nothing.
    let (handle, done) = begin(&runtime, &short, speculation, Trigger::Never);
    handle.cancel();
    handle.cancel();
    let cut = done.wait(WAIT).expect("the generation reports");
    assert_eq!(cut.outcome.reason, FinishReason::Cancelled);
    assert_eq!(cut.tokens, short_reference.tokens[..cut.tokens.len()]);
    handle.cancel();
    assert!(handle.is_cancelled());
    assert_free(&runtime);
    let again = run(&runtime, &short, speculation, Trigger::Never);
    assert_eq!(
        again.tokens, short_reference.tokens,
        "after a cancel from the test's thread"
    );

    // A cancel after the end is a no-op: the generation finished at its length, and the next one is not affected.
    let (handle, done) = begin(&runtime, &short, speculation, Trigger::Never);
    let finished = done.wait(WAIT).expect("the generation reports");
    assert_eq!(finished.outcome.reason, FinishReason::Length);
    handle.cancel();
    let next = run(&runtime, &short, speculation, Trigger::Never);
    assert_eq!(next.outcome.reason, FinishReason::Length);
    assert_eq!(next.tokens, short_reference.tokens);
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
