//! Hidden rows on a GPU: a generation hands the post-final-norm hidden state of every row it runs to a sink, with the
//! logits of the rows that select. The tests are `#[ignore]`d and run locally, one at a time (each loads the 12 GB
//! artifact), through the system arbiter:
//! `system-run --gpu shared:14 --mem 24G --est 300 --who <name> -- cargo test -p euhedral-inference-engine --test hidden_gpu -- --ignored --test-threads=1`.
//!
//! The artifact comes from `euhedral.qwen.rco-artifact` and the library from `euhedral.cuda.library`.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]
#![allow(clippy::print_stdout, reason = "a test reports what it measured")]
#![allow(
    clippy::disallowed_methods,
    reason = "no-wait, blocking I/O and waiting calls: a test reads the device and waits on hand-offs"
)]

use std::mem;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

mod common;

use common::Closer;
use euhedral_inference_engine::generation::{
    CancelHandle, FinishReason, GenerationListener, GenerationOutcome, GenerationRequest, HiddenSink, QuantumEvent,
    Speculation,
};
use euhedral_inference_engine::sampling::SamplingConfig;
use euhedral_inference_engine::{EngineRuntime, Handoff};
use euhedral_inference_settings::{ModelSettings, Settings};

/// The longest a device test waits for one hand-off; past it the test fails.
const WAIT: Duration = Duration::from_secs(20 * 60);

/// Rows of a prompt chunk: a longer prompt takes several.
const CHUNK: u32 = 64;

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

fn start(machine: &Machine) -> (Closer, Arc<EngineRuntime>) {
    let mut settings = Settings::with_model(ModelSettings {
        artifact: machine.artifact.clone(),
        name: "test".to_string(),
    });
    settings.context.max_tokens = 512;
    settings.kernels.graphs.captures_per_view = 8;
    settings.batching.prefill_chunk_tokens = CHUNK;
    let (owner, ready) = EngineRuntime::start(&settings, &machine.library);
    let runtime = ready
        .wait(WAIT)
        .expect("the engine reports")
        .expect("the engine starts");
    (Closer::new(owner, &runtime), runtime)
}

/// A prompt of `length` ordinary token ids.
fn prompt(length: usize) -> Vec<u32> {
    (0..length).map(|index| 1000 + (index as u32 * 37) % 5000).collect()
}

/// What a generation told its listener.
struct Recording {
    tokens: Vec<u32>,
    outcome: GenerationOutcome,
}

struct Recorder {
    tokens: Vec<u32>,
    done: Option<Arc<Handoff<Recording>>>,
}

impl GenerationListener for Recorder {
    fn prefill_done(&mut self, _at: Instant) {}

    fn token(&mut self, id: u32, _at: Instant) {
        self.tokens.push(id);
    }

    fn quantum(&mut self, _event: QuantumEvent) {}

    fn finished(&mut self, outcome: GenerationOutcome) {
        let done = self.done.take().expect("a generation finishes once");
        done.send(Recording {
            tokens: mem::take(&mut self.tokens),
            outcome,
        });
    }
}

/// The rows one call of the sink heard.
struct Delivered {
    first_position: u64,
    rows: usize,
    hidden: Vec<u8>,
    logits: Option<Vec<u8>>,
}

/// Keeps what it hears, and hands it over when the engine lets go of it at the generation's end.
struct Collector {
    heard: Vec<Delivered>,
    done: Option<Arc<Handoff<Vec<Delivered>>>>,
}

impl HiddenSink for Collector {
    fn rows(&mut self, first_position: u64, rows: usize, hidden: &[u8], logits: Option<&[u8]>) {
        self.heard.push(Delivered {
            first_position,
            rows,
            hidden: hidden.to_vec(),
            logits: logits.map(<[u8]>::to_vec),
        });
    }
}

impl Drop for Collector {
    fn drop(&mut self) {
        if let Some(done) = self.done.take() {
            done.send(mem::take(&mut self.heard));
        }
    }
}

fn request(
    prompt: &[u32],
    max_tokens: u32,
    speculation: Speculation,
    sampling: SamplingConfig,
    done: &Arc<Handoff<Recording>>,
) -> GenerationRequest {
    GenerationRequest {
        prompt: prompt.to_vec(),
        max_tokens,
        stop_tokens: Vec::new(),
        speculation,
        sampling,
        constraint: None,
        listener: Box::new(Recorder {
            tokens: Vec::new(),
            done: Some(Arc::clone(done)),
        }),
    }
}

fn greedy() -> SamplingConfig {
    SamplingConfig::new(0.0, 0, 1.0, 0, true).unwrap()
}

fn sampled() -> SamplingConfig {
    SamplingConfig::new(0.8, 40, 0.95, 1234, false).unwrap()
}

/// A generation without a sink.
fn plain(
    runtime: &EngineRuntime,
    prompt: &[u32],
    max_tokens: u32,
    speculation: Speculation,
    sampling: SamplingConfig,
) -> Recording {
    let done = Handoff::new();
    runtime
        .generate(request(prompt, max_tokens, speculation, sampling, &done))
        .expect("the request is admitted");
    done.wait(WAIT).expect("the generation reports")
}

/// A generation that hands its hidden rows to a collector; the rows heard by the time its end was reported.
fn hidden(
    runtime: &EngineRuntime,
    prompt: &[u32],
    max_tokens: u32,
    sampling: SamplingConfig,
) -> (Recording, Vec<Delivered>) {
    let done = Handoff::new();
    let heard = Handoff::new();
    runtime
        .generate_hidden(
            request(prompt, max_tokens, Speculation::Off, sampling, &done),
            Collector {
                heard: Vec::new(),
                done: Some(Arc::clone(&heard)),
            },
            CancelHandle::default(),
        )
        .expect("the request is admitted");
    let recording = done.wait(WAIT).expect("the generation reports");
    let rows = heard.wait(WAIT).expect("the sink is let go of");
    (recording, rows)
}

/// The sequence's draft seed rows: the rows the draft layer's catch-up reads, `rows` of them.
fn seed_rows(runtime: &EngineRuntime, rows: usize, hidden_bytes: usize) -> Vec<u8> {
    runtime
        .inspect_sequence(move |sequence| {
            let mut bytes = vec![0u8; rows * hidden_bytes];
            sequence
                .device
                .scratch
                .draft()
                .expect("the artifact drafts")
                .seed_rows
                .copy_to_host(0, &mut bytes)
                .expect("the seed rows are read");
            bytes
        })
        .expect("the generation gave the sequence back")
}

fn bf16(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| f32::from_bits(u32::from(u16::from_le_bytes(*pair)) << 16))
        .collect()
}

/// The hidden width the rows have: the size of one row of a chunk's rows, from a delivery that has one row.
fn hidden_bytes_of(delivered: &[Delivered]) -> usize {
    let one = delivered.iter().find(|call| call.rows == 1).expect("a decode row");
    one.hidden.len()
}

#[test]
#[ignore = "needs a GPU, the native library and the RCO artifact"]
fn hidden_rows_are_the_seed_rows_the_catch_up_reads() {
    let Some(machine) = machine() else { return };
    let (owner, runtime) = start(&machine);

    // A request that drafts runs its verification on the row of the token it has, and the seed rows then hold that
    // row's post-final-norm hidden state, which the next catch-up reads. The hidden request decodes the same row.
    let ids = prompt(40);
    let speculative = plain(&runtime, &ids, 2, Speculation::Artifact, greedy());
    assert_eq!(speculative.outcome.reason, FinishReason::Length);
    let hidden_bytes = runtime.config().hidden_size * 2;
    let catch_up_reads = seed_rows(&runtime, 1, hidden_bytes);
    let (hidden_run, heard) = hidden(&runtime, &ids, 2, greedy());
    assert_eq!(hidden_run.outcome.reason, FinishReason::Length);
    assert_eq!(speculative.tokens, hidden_run.tokens, "the tokens are the same");
    let positions: Vec<(u64, usize)> = heard.iter().map(|call| (call.first_position, call.rows)).collect();
    assert_eq!(positions, [(0, 40), (40, 1)], "the prompt chunk, then the decode row");
    assert_eq!(
        heard[1].hidden, catch_up_reads,
        "the decode row handed out is the row the catch-up reads"
    );
    assert!(heard[1].logits.is_some() && heard[0].logits.is_some());
    println!("the row of {hidden_bytes} bytes equals the seed row");
    owner.shutdown(runtime).unwrap();
}

#[test]
#[ignore = "needs a GPU, the native library and the RCO artifact"]
fn hidden_rows_cover_every_row_with_logits_where_a_token_is_selected() {
    let Some(machine) = machine() else { return };
    let (owner, runtime) = start(&machine);

    // Three chunks (64, 64, 22) and five decode rows.
    let ids = prompt(150);
    let (recording, heard) = hidden(&runtime, &ids, 6, greedy());
    assert_eq!(recording.outcome.reason, FinishReason::Length);
    assert_eq!(recording.tokens.len(), 6);
    let positions: Vec<(u64, usize)> = heard.iter().map(|call| (call.first_position, call.rows)).collect();
    assert_eq!(
        positions,
        [
            (0, 64),
            (64, 64),
            (128, 22),
            (150, 1),
            (151, 1),
            (152, 1),
            (153, 1),
            (154, 1)
        ],
        "every row, once, in order"
    );
    let selecting: Vec<bool> = heard.iter().map(|call| call.logits.is_some()).collect();
    assert_eq!(
        selecting,
        [false, false, true, true, true, true, true, true],
        "the rows that select have logits"
    );
    // The token a row selects is the best of its logits.
    let tokens = &recording.tokens;
    for (call, token) in heard.iter().filter(|call| call.logits.is_some()).zip(tokens) {
        let logits = bf16(call.logits.as_ref().unwrap());
        let best = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert_eq!(logits[*token as usize], best, "token {token} is a maximum of its row");
    }
    // The last row's hidden state is what the seed buffer holds at the end.
    let row_bytes = hidden_bytes_of(&heard);
    assert_eq!(
        heard.last().unwrap().hidden,
        seed_rows(&runtime, 1, row_bytes),
        "the last decode row is the seed row"
    );
    owner.shutdown(runtime).unwrap();
}

#[test]
#[ignore = "needs a GPU, the native library and the RCO artifact"]
fn sampled_tokens_are_the_same_with_the_sink_on_and_off() {
    let Some(machine) = machine() else { return };
    let (owner, runtime) = start(&machine);

    let ids = prompt(100);
    for (name, sampling) in [("sampled", sampled()), ("greedy", greedy())] {
        let off = plain(&runtime, &ids, 24, Speculation::Off, sampling);
        let (on, heard) = hidden(&runtime, &ids, 24, sampling);
        assert_eq!(on.tokens, off.tokens, "{name}: the tokens");
        assert_eq!(on.outcome.generated, off.outcome.generated, "{name}: the count");
        let rows: usize = heard.iter().map(|call| call.rows).sum();
        assert_eq!(rows, 100 + 23, "{name}: a row for every prompt token and every decode");
        println!("{name}: {} tokens equal", on.tokens.len());
    }
    owner.shutdown(runtime).unwrap();
}
