//! MTP speculative decoding on a GPU: the tokens it commits are the ones greedy decode selects. The tests are
//! `#[ignore]`d and run locally, one at a time (each loads the 12 GB artifact), through the GPU runner:
//! `/mnt/Samsung_980_1TB/euhedral-dev/bin/gpu-run -m 24G -t 30m -- cargo test -p euhedral-inference-engine --test mtp_gpu -- --ignored --test-threads=1`.
//!
//! The reference is the engine's own greedy decode (`Speculation::Off`), whose tokens equal the Java engine's
//! (`generation_gpu`). The prompts are the chat corpus of the benchmark crate. The artifact comes from
//! `euhedral.qwen.rco-artifact` and the library from `euhedral.cuda.library`.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]
#![allow(
    clippy::print_stdout,
    reason = "a test reports the histograms and the counts it measured"
)]
#![allow(
    clippy::disallowed_methods,
    reason = "no-wait, blocking I/O and waiting calls: a test reads files and waits on latches and the device"
)]

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::mem;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use euhedral_graph::{LaneId, Latch, LatchSetter};
use euhedral_inference_bench::prompt::PromptCorpus;
use euhedral_inference_bench::prompt::words::DEFAULT_SEED;
use euhedral_inference_cuda::{CudaLane, Submission, SubmitMode, ops};
use euhedral_inference_edrl::{ReadPlan, Region, TokenizerFooter, TokenizerSection};
use euhedral_inference_engine::generation::{
    FinishReason, GenerationListener, GenerationOutcome, GenerationRequest, QuantumEvent, Speculation, Verification,
};
use euhedral_inference_engine::stages::QuantumKind;
use euhedral_inference_engine::{EngineRuntime, RuntimeOwner};
use euhedral_inference_settings::{ModelSettings, Settings};
use euhedral_inference_tokenizer::Tokenizer;

/// Rows of a page of the caches.
const PAGE: u64 = 256;

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

fn start(machine: &Machine, max_tokens: u32) -> (Arc<EngineRuntime>, RuntimeOwner) {
    let mut settings = Settings::with_model(ModelSettings {
        artifact: machine.artifact.clone(),
        name: "test".to_string(),
    });
    settings.context.max_tokens = max_tokens;
    let (ready, waiting) = Latch::new();
    let owner = EngineRuntime::start(&settings, &machine.library, ready);
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
    events: Vec<QuantumEvent>,
    verifications: Vec<Verification>,
    outcome: GenerationOutcome,
    last_token: Instant,
    first_token: Instant,
}

struct Recorder {
    tokens: Vec<u32>,
    events: Vec<QuantumEvent>,
    verifications: Vec<Verification>,
    first: Option<Instant>,
    last: Option<Instant>,
    done: Option<LatchSetter<Recording>>,
}

impl GenerationListener for Recorder {
    fn prefill_done(&mut self, _at: Instant) {}

    fn token(&mut self, id: u32, at: Instant) {
        self.tokens.push(id);
        self.first.get_or_insert(at);
        self.last = Some(at);
    }

    fn quantum(&mut self, event: QuantumEvent) {
        self.events.push(event);
    }

    fn verification(&mut self, event: Verification) {
        self.verifications.push(event);
    }

    fn finished(&mut self, outcome: GenerationOutcome) {
        let done = self.done.take().expect("a generation finishes once");
        let now = Instant::now();
        done.set(Recording {
            tokens: mem::take(&mut self.tokens),
            events: mem::take(&mut self.events),
            verifications: mem::take(&mut self.verifications),
            outcome,
            first_token: self.first.unwrap_or(now),
            last_token: self.last.unwrap_or(now),
        });
    }
}

fn generate(
    runtime: &EngineRuntime,
    prompt: &[u32],
    max_tokens: u32,
    stop_tokens: &[u32],
    speculation: Speculation,
) -> Recording {
    let (done, waiting) = Latch::new();
    runtime
        .generate(GenerationRequest {
            prompt: prompt.to_vec(),
            max_tokens,
            stop_tokens: stop_tokens.to_vec(),
            speculation,
            listener: Box::new(Recorder {
                tokens: Vec::new(),
                events: Vec::new(),
                verifications: Vec::new(),
                first: None,
                last: None,
                done: Some(done),
            }),
        })
        .expect("the request is admitted");
    waiting.wait()
}

/// Committed rows of the base and MTP caches, quanta in flight and selections leased after a generation.
fn sequence_state(runtime: &EngineRuntime) -> (u64, u64, usize, usize) {
    runtime
        .inspect_sequence(|sequence| {
            (
                sequence.kv.committed(),
                sequence.mtp_kv.as_ref().map_or(0, |cache| cache.committed()),
                sequence.in_flight(),
                sequence.selections.leased(),
            )
        })
        .expect("the generation gave the sequence back")
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

/// The device memory of a sequence after a generation: whole pages of both caches below the committed rows, and every
/// GDN state.
#[derive(Debug, PartialEq, Eq)]
struct Snapshot {
    base_pages: Vec<Vec<u8>>,
    mtp_pages: Vec<u8>,
    gdn: Vec<Vec<u8>>,
}

/// A lane of the test's own, for the copies of a snapshot (a lane id exists once per library).
fn snapshot_lane(runtime: &EngineRuntime) -> Arc<CudaLane> {
    let library = runtime.library();
    CudaLane::new(library, LaneId((library.max_lanes() - 1) as u16), 4).unwrap()
}

fn snapshot(runtime: &EngineRuntime, lane: &CudaLane) -> Snapshot {
    let download = |ptr, bytes: usize| {
        let mut host = vec![0u8; bytes];
        let submission = Submission::begin(lane, SubmitMode::default()).unwrap();
        ops::copy_device_to_host(&submission, &mut host, ptr).unwrap();
        drop(submission);
        lane.synchronize().unwrap();
        host
    };
    let layers = runtime.config().layer_kinds.len();
    runtime
        .inspect_sequence(|sequence| {
            let full = (sequence.kv.committed() / PAGE) as usize;
            let page_bytes = 2 * sequence.kv.plane_page_bytes() as usize;
            let base_pages = (0..sequence.kv.layers())
                .map(|layer| {
                    sequence
                        .kv
                        .pages(layer)
                        .take(full)
                        .flat_map(|page| download(page, page_bytes))
                        .collect()
                })
                .collect();
            let mtp = sequence.mtp_kv.as_ref().unwrap();
            let mtp_full = (mtp.committed() / PAGE) as usize;
            let mtp_pages = mtp
                .pages(0)
                .take(mtp_full)
                .flat_map(|page| download(page, page_bytes))
                .collect();
            let gdn = (0..layers)
                .filter_map(|layer| sequence.device.gdn.layer(layer))
                .flat_map(|state| [state.convolution(), state.recurrent()])
                .map(|buffer| download(buffer.ptr(), buffer.bytes()))
                .collect();
            Snapshot {
                base_pages,
                mtp_pages,
                gdn,
            }
        })
        .expect("the generation gave the sequence back")
}

#[test]
#[ignore = "needs a GPU, the native library and the RCO artifact"]
fn mtp_output_equals_greedy() {
    let Some(machine) = machine() else { return };
    let tokenizer = tokenizer(&machine.artifact);
    let eos = tokenizer.eos_ids().to_vec();
    let (runtime, owner) = start(&machine, 4096);
    let depth = runtime.plan().draft_depth;
    assert!(depth > 0, "the artifact drafts");
    let mut total = vec![0u64; depth + 1];
    for target in [128, 2000] {
        for (index, prompt) in chat_prompts(&tokenizer, target).iter().enumerate() {
            let greedy = generate(&runtime, prompt, 256, &eos, Speculation::Off);
            let greedy_state = sequence_state(&runtime);
            let speculative = generate(&runtime, prompt, 256, &eos, Speculation::Artifact);
            let state = sequence_state(&runtime);
            assert_eq!(
                speculative.tokens, greedy.tokens,
                "target {target} prompt {index}: the speculative tokens are greedy decode's"
            );
            assert_eq!(speculative.outcome.reason, greedy.outcome.reason);
            assert_eq!(speculative.outcome.generated, greedy.outcome.generated);
            assert_eq!(state.0, greedy_state.0, "committed base rows");
            assert_eq!((state.2, state.3), (0, 0), "no quantum in flight, no selection leased");
            // The first token comes from the prompt; every other one is committed by a verification.
            let committed: usize = speculative.verifications.iter().map(|v| v.committed).sum();
            assert_eq!(committed + 1, speculative.tokens.len());
            let counts = histogram(&speculative, depth);
            for (accepted, n) in counts.iter().enumerate() {
                total[accepted] += n;
            }
            let rate = |recording: &Recording| {
                (recording.tokens.len() - 1) as f64 / (recording.last_token - recording.first_token).as_secs_f64()
            };
            println!(
                "target {target} prompt {index}: {} prompt tokens, {} tokens, accepted {counts:?} (mean {:.2}), {:.1} tok/s \
                 against {:.1} tok/s greedy (debug build, not a gate)",
                prompt.len(),
                speculative.tokens.len(),
                mean_accepted(&counts),
                rate(&speculative),
                rate(&greedy),
            );
        }
    }
    println!(
        "accepted drafts per verification over all prompts: {total:?}, mean {:.3}",
        mean_accepted(&total)
    );
    owner.shutdown(runtime).unwrap();
}

#[test]
#[ignore = "needs a GPU, the native library and the RCO artifact"]
fn mtp_kv_and_gdn_state_equal_greedy_after_run() {
    let Some(machine) = machine() else { return };
    let tokenizer = tokenizer(&machine.artifact);
    let prompt = chat_prompts(&tokenizer, 2000).swap_remove(0);
    let (runtime, owner) = start(&machine, 4096);
    let depth = runtime.plan().draft_depth as u64;
    let wanted = 100u32;

    // Greedy decode of one token more feeds the `wanted` tokens the speculative request commits; the speculative one
    // ends with the decode that feeds its last token (the replay of its last verification runs in it).
    let greedy = generate(&runtime, &prompt, wanted + 1, &[], Speculation::Off);
    let greedy_state = sequence_state(&runtime);
    let lane = snapshot_lane(&runtime);
    let wanted_snapshot = snapshot(&runtime, &lane);
    runtime.settle_after_speculation_for_tests(true);
    let speculative = generate(&runtime, &prompt, wanted, &[], Speculation::Artifact);
    let state = sequence_state(&runtime);
    let got = snapshot(&runtime, &lane);

    assert_eq!(speculative.outcome.reason, FinishReason::Length);
    assert_eq!(speculative.tokens, greedy.tokens[..wanted as usize]);
    assert_eq!(state.0, prompt.len() as u64 + u64::from(wanted));
    assert_eq!(state.0, greedy_state.0, "the base cache holds the same rows");
    assert_eq!((state.2, state.3), (0, 0));
    // The MTP cache holds the rows up to the last verification; its speculative rows were dropped.
    assert!(
        state.1 <= state.0 && state.1 + depth + 1 >= state.0,
        "MTP rows {} of {}",
        state.1,
        state.0
    );
    assert!(!got.base_pages.is_empty() && !got.base_pages[0].is_empty());
    for (layer, (a, b)) in got.base_pages.iter().zip(&wanted_snapshot.base_pages).enumerate() {
        assert!(
            a == b,
            "attention layer {layer}: the whole pages of the base cache are greedy decode's"
        );
    }
    assert!(!got.gdn.is_empty());
    for (index, (a, b)) in got.gdn.iter().zip(&wanted_snapshot.gdn).enumerate() {
        assert!(a == b, "GDN state buffer {index} equals greedy decode's");
    }
    assert!(!got.mtp_pages.is_empty(), "the MTP cache has whole pages to compare");

    // A second speculative run gives the same MTP rows: they are a function of the committed tokens.
    let again = generate(&runtime, &prompt, wanted, &[], Speculation::Artifact);
    assert_eq!(again.tokens, speculative.tokens);
    let second = snapshot(&runtime, &lane);
    assert!(second.mtp_pages == got.mtp_pages, "the MTP cache is the same again");
    drop(lane);
    owner.shutdown(runtime).unwrap();
}

#[test]
#[ignore = "needs a GPU, the native library and the RCO artifact"]
fn mtp_eos_inside_accepted_drafts() {
    let Some(machine) = machine() else { return };
    let tokenizer = tokenizer(&machine.artifact);
    let prompt = chat_prompts(&tokenizer, 128).swap_remove(0);
    let (runtime, owner) = start(&machine, 4096);
    let depth = runtime.plan().draft_depth;
    let reference = generate(&runtime, &prompt, 128, &[], Speculation::Artifact);
    assert_eq!(reference.tokens.len(), 128);

    // The tokens each verification committed, from the first token on: find a verification that accepted several
    // drafts, and make the second of its tokens (a draft the verifier confirmed, not the verifier's own) the end token.
    let mut start_index = 1;
    let mut chosen = None;
    for verification in &reference.verifications {
        if verification.accepted >= 2 && verification.committed >= 3 {
            let inside = start_index + 1;
            if !reference.tokens[..inside].contains(&reference.tokens[inside]) {
                chosen = Some((inside, *verification));
                break;
            }
        }
        start_index += verification.committed;
    }
    let (inside, step) = chosen.expect("a verification accepted several drafts in 128 tokens");
    let stop = reference.tokens[inside];
    let cut = generate(&runtime, &prompt, 128, &[stop], Speculation::Artifact);
    let greedy = generate(&runtime, &prompt, 128, &[stop], Speculation::Off);
    assert_eq!(cut.outcome.reason, FinishReason::Stop);
    assert_eq!(cut.tokens, reference.tokens[..=inside]);
    assert_eq!(cut.tokens, greedy.tokens, "the cut is greedy decode's");
    let last = cut.verifications.last().unwrap();
    assert!(
        last.accepted >= 2 && last.committed < last.accepted + 1,
        "the cut fell inside the accepted drafts"
    );
    assert_eq!(last.drafted, step.drafted);
    // The end token is selected, not fed: the cache holds the prompt and the tokens before it.
    let state = sequence_state(&runtime);
    assert_eq!(state.0, prompt.len() as u64 + inside as u64);
    assert_eq!((state.2, state.3), (0, 0));
    assert!(depth >= 3);

    // The engine serves the next request from a reset sequence.
    let again = generate(&runtime, &prompt, 128, &[], Speculation::Artifact);
    assert_eq!(again.tokens, reference.tokens);
    owner.shutdown(runtime).unwrap();
}

#[test]
#[ignore = "needs a GPU, the native library and the RCO artifact"]
fn mtp_stops_at_context_capacity() {
    let Some(machine) = machine() else { return };
    let tokenizer = tokenizer(&machine.artifact);
    let base = chat_prompts(&tokenizer, 128).swap_remove(0);
    let context = 600u32;
    let (runtime, owner) = start(&machine, context);
    let capacity = u64::from(context);
    assert_eq!(runtime.plan().max_context_tokens as u64, capacity);

    // Prompts that leave the context the whole of a step, part of one, one row and no row for generation.
    for room in [capacity - base.len() as u64, 9, 6, 3, 2, 1, 0] {
        let prompt: Vec<u32> = base.iter().copied().cycle().take((capacity - room) as usize).collect();
        let greedy = generate(&runtime, &prompt, 100_000, &[], Speculation::Off);
        let speculative = generate(&runtime, &prompt, 100_000, &[], Speculation::Artifact);
        assert_eq!(speculative.outcome.reason, FinishReason::ContextFull, "room {room}");
        assert_eq!(
            speculative.tokens.len() as u64,
            room + 1,
            "room {room}: one token from the prompt, one per row"
        );
        assert_eq!(speculative.tokens, greedy.tokens, "room {room}");
        assert_eq!(sequence_state(&runtime).0, capacity);
        assert_eq!(sequence_state(&runtime).2, 0);
    }

    // A small budget well inside the context ends with the budget.
    for wanted in [1, 2, 3, 5, 8] {
        let speculative = generate(&runtime, &base, wanted, &[], Speculation::Artifact);
        let greedy = generate(&runtime, &base, wanted, &[], Speculation::Off);
        assert_eq!(speculative.tokens.len(), wanted as usize);
        assert_eq!(speculative.tokens, greedy.tokens, "budget {wanted}");
        assert_eq!(speculative.outcome.reason, FinishReason::Length);
    }
    owner.shutdown(runtime).unwrap();
}

/// A run of one engine on the long prompt.
struct LongRun {
    recording: Recording,
    snapshot: Snapshot,
}

fn long_run(machine: &Machine, prompt: &[u32], graphs: bool) -> LongRun {
    let (runtime, owner) = start(machine, 4096);
    if !graphs {
        assert!(runtime.disable_graphs_for_tests());
    }
    // A first round records the graphs of the keys it sees twice; the second is the one measured.
    let warm = generate(&runtime, prompt, 256, &[], Speculation::Artifact);
    let recording = generate(&runtime, prompt, 256, &[], Speculation::Artifact);
    assert_eq!(recording.tokens, warm.tokens);
    let lane = snapshot_lane(&runtime);
    let snapshot = snapshot(&runtime, &lane);
    drop(lane);
    owner.shutdown(runtime).unwrap();
    LongRun { recording, snapshot }
}

#[test]
#[ignore = "needs a GPU, the native library and the RCO artifact"]
fn captured_mtp_equals_uncaptured() {
    let Some(machine) = machine() else { return };
    let tokenizer = tokenizer(&machine.artifact);
    // About 1990 tokens: the generation crosses the page boundary at 2048 rows and the switch of the attention
    // geometry's key there.
    let prompt = chat_prompts(&tokenizer, 2000).swap_remove(2);
    assert!(prompt.len() > 1900 && prompt.len() < 2048, "{} tokens", prompt.len());

    let captured = long_run(&machine, &prompt, true);
    let plain = long_run(&machine, &prompt, false);
    let replayed = captured.recording.events.iter().filter(|event| event.replayed).count();
    let kinds = |kind: QuantumKind| {
        captured
            .recording
            .events
            .iter()
            .filter(|event| event.replayed && event.kind == kind)
            .count()
    };
    println!(
        "{} quanta, {replayed} replayed (verification {}, draft {}), {} recorded",
        captured.recording.events.len(),
        kinds(QuantumKind::Verify),
        kinds(QuantumKind::Draft),
        captured.recording.events.iter().filter(|event| event.recorded).count()
    );
    assert!(replayed >= 100, "at least 100 replayed quanta, saw {replayed}");
    assert_eq!(
        plain.recording.events.iter().filter(|event| event.replayed).count(),
        0,
        "the reference replays nothing"
    );
    assert!(
        captured.recording.tokens.len() as u64 + prompt.len() as u64 > 2048,
        "the generation crosses the key switch at 2048 rows"
    );
    assert_eq!(captured.recording.tokens, plain.recording.tokens);
    let depth = 4;
    assert_eq!(
        histogram(&captured.recording, depth),
        histogram(&plain.recording, depth),
        "the accepted drafts per verification"
    );
    let accepted = |run: &LongRun| {
        run.recording
            .verifications
            .iter()
            .map(|v| v.accepted)
            .collect::<Vec<_>>()
    };
    assert_eq!(
        accepted(&captured),
        accepted(&plain),
        "every verification accepted the same drafts"
    );
    println!("accepted per verification: {:?}", histogram(&captured.recording, depth));
    assert!(
        captured.snapshot == plain.snapshot,
        "base pages, MTP pages and GDN state are the same"
    );
}
