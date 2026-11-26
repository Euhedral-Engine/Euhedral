//! The prompt prefix cache for requests that draft with the artifact's MTP layer, on a GPU with the RCO artifact: a
//! speculative request restores the checkpoints an earlier one captured (the base state, the MTP cache below the last
//! row, the token that row was paired with, and the seed row of the draft) and then produces the tokens a cold run
//! produces, with the drafts a cold run makes: exactly, when the prompt has the token the checkpoint paired its last
//! row with, and up to that one row's low bits when it has another. `#[ignore]`d; run one at a time in a
//! memory-capped scope (each loads the 12 GB artifact), with the CUDA library named by `euhedral.cuda.library` and the artifact by `euhedral.qwen.rco-artifact`; a test skips without them. Every
//! wait is bounded.
//!
//! The verifier decides every token, so tokens alone cannot tell a restored MTP cache from a wrong one: the drafts
//! accepted can. A wrong draft state accepts few drafts.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]
#![allow(clippy::print_stdout, reason = "a test reports what it measured")]
#![allow(
    clippy::disallowed_methods,
    reason = "no-wait, blocking I/O and waiting calls: a test waits on hand-offs and the device"
)]

use crate::typed;

use std::mem;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::common;

use common::Closer;
use euhedral_inference_engine::generation::{
    CancelHandle, FinishReason, GenerationListener, GenerationOutcome, GenerationRequest, QuantumEvent, Speculation,
    Verification,
};
use euhedral_inference_engine::sampling::SamplingConfig;
use euhedral_inference_engine::{EngineRuntime, Handoff};
use euhedral_inference_settings::{ByteSize, ModelSettings, Settings};
use typed::Typed;

/// The longest a device test waits for one hand-off; past it the test fails.
const WAIT: Duration = Duration::from_secs(20 * 60);
/// The tokens each request generates.
const GENERATED: u32 = 96;
/// The prompts' length: checkpoints fall at 1024 and 2048 (the interval) and 2560 (the last chunk boundary before the
/// end).
const LENGTH: usize = 2600;
/// The position of the last checkpoint of the long prompt.
const LAST_OF_D: usize = 3584;
/// The length of the prompt that goes on past the next checkpoints.
const LONG: usize = 3700;

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

fn settings(artifact: &Path, cache: ByteSize) -> Settings {
    let mut settings = Settings::with_model(ModelSettings {
        artifact: artifact.to_path_buf(),
        name: "test".to_string(),
    });
    settings.context.max_tokens = 4096;
    settings.kernels.graphs.captures_per_view = 8;
    settings.prefix_cache.size = cache;
    settings
}

/// The engine without a prefix cache, from the cache of engines: it holds nothing a test could find, so tests share it.
fn cold(machine: &Machine) -> (common::Card, Arc<EngineRuntime>) {
    common::engine(&settings(&machine.artifact, ByteSize(0)), &machine.library)
}

/// Starts the engine with a prefix cache of `cache` bytes (0: off).
fn start(machine: &Machine, cache: ByteSize) -> (Closer, Arc<EngineRuntime>) {
    common::start_engine(&settings(&machine.artifact, cache), &machine.library)
}

/// A prompt of `LENGTH` tokens whose first `shared` tokens are those of every prompt of `seed` 0 and whose others
/// are `seed`'s own. Each part repeats a short pattern, so the model continues it and the draft has something to
/// predict (random tokens are never predicted, and then no draft is accepted to tell a restored draft state by).
fn prompt(seed: u64, shared: usize) -> Vec<u32> {
    prompt_of(seed, shared, LENGTH)
}

/// [`prompt`] of `length` tokens.
fn prompt_of(seed: u64, shared: usize, length: usize) -> Vec<u32> {
    (0..length)
        .map(|index| {
            let owner = if index < shared { 0 } else { seed };
            let mixed = ((index % 37) as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                ^ owner.wrapping_mul(0xBF58_476D_1CE4_E5B9);
            1000 + (mixed >> 33) as u32 % 50_000
        })
        .collect()
}

fn prompt_unrelated() -> Vec<u32> {
    prompt(99, 0)
}

/// Rows of a page of the MTP cache.
const PAGE: usize = 256;
/// The share of the bytes of the re-paired row that may differ from the cold run's (low bits only); a row computed
/// from the wrong hidden row differs in nearly every byte.
const REPAIRED_ROW_TOLERANCE: f64 = 0.5;

/// The whole pages of the sequence's MTP cache, K plane then V plane of each.
struct MtpPages {
    pages: Vec<Vec<u8>>,
    row_bytes: usize,
}

fn mtp_pages(runtime: &EngineRuntime, lane: &Arc<Typed>) -> MtpPages {
    let lane = Arc::clone(lane);
    runtime
        .inspect_sequence(move |sequence| {
            let mtp = sequence.mtp_kv.as_ref().expect("the engine drafts");
            let plane = mtp.plane_page_bytes() as usize;
            let full = (mtp.committed() / PAGE as u64) as usize;
            MtpPages {
                pages: mtp
                    .pages(0)
                    .take(full)
                    .map(|page| lane.download(page, 2 * plane))
                    .collect(),
                row_bytes: plane / PAGE,
            }
        })
        .expect("the generation gave the sequence back")
}

/// A restored run leaves the MTP cache a cold run leaves: every row bit for bit, but the rows `repaired` (the last of
/// the restored prefix when the prompt had another token after it than the checkpoint's, and the same row of an
/// earlier such restore, which a later checkpoint stored), which the restore computes again from the stored hidden row
/// and which may differ in low bits. None, when every restore had the checkpoint's token.
fn same_mtp_rows(name: &str, warm: &MtpPages, cold: &MtpPages, repaired: &[usize]) {
    assert_eq!(
        warm.pages.len(),
        cold.pages.len(),
        "{name}: whole pages of the MTP cache"
    );
    assert!(
        repaired.iter().all(|row| warm.pages.len() * PAGE > *row),
        "{name}: the repaired row is in a whole page"
    );
    let bytes = warm.row_bytes;
    for row in 0..warm.pages.len() * PAGE {
        let (page, inside) = (row / PAGE, row % PAGE);
        let plane = warm.pages[page].len() / 2;
        let mut differing = 0;
        for plane_start in [0, plane] {
            let at = plane_start + inside * bytes..plane_start + (inside + 1) * bytes;
            differing += warm.pages[page][at.clone()]
                .iter()
                .zip(&cold.pages[page][at])
                .filter(|(a, b)| a != b)
                .count();
        }
        if repaired.contains(&row) {
            let share = differing as f64 / (2 * bytes) as f64;
            println!(
                "{name}: the re-paired row {row} differs in {differing} of {} bytes",
                2 * bytes
            );
            assert!(
                share <= REPAIRED_ROW_TOLERANCE,
                "{name}: row {row} differs in {share} of its bytes"
            );
        } else {
            assert_eq!(differing, 0, "{name}: MTP row {row} differs from the cold run's");
        }
    }
}

/// The bytes of row `row` of the MTP cache, both planes.
fn row_of(pages: &MtpPages, row: usize) -> Vec<u8> {
    let (page, inside) = (row / PAGE, row % PAGE);
    let plane = pages.pages[page].len() / 2;
    let bytes = pages.row_bytes;
    [0, plane]
        .iter()
        .flat_map(|start| pages.pages[page][start + inside * bytes..start + (inside + 1) * bytes].to_vec())
        .collect()
}

#[derive(Debug)]
struct Recording {
    tokens: Vec<u32>,
    verifications: Vec<Verification>,
    outcome: GenerationOutcome,
}

impl Recording {
    /// The drafts the verifier confirmed, over the drafts it checked.
    fn accepted(&self) -> (usize, usize) {
        (
            self.verifications.iter().map(|v| v.accepted).sum(),
            self.verifications.iter().map(|v| v.drafted).sum(),
        )
    }
}

struct Recorder {
    tokens: Vec<u32>,
    verifications: Vec<Verification>,
    done: Option<Arc<Handoff<Recording>>>,
}

impl GenerationListener for Recorder {
    fn prefill_done(&mut self, _at: Instant) {}

    fn token(&mut self, id: u32, _at: Instant) {
        self.tokens.push(id);
    }

    fn quantum(&mut self, _event: QuantumEvent) {}

    fn verification(&mut self, event: Verification) {
        self.verifications.push(event);
    }

    fn finished(&mut self, outcome: GenerationOutcome) {
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
                verifications: Vec::new(),
                outcome: GenerationOutcome {
                    reason: FinishReason::Failed("the listener was dropped before the generation ended".to_string()),
                    prompt_tokens: 0,
                    generated: 0,
                    elapsed: Duration::ZERO,
                    cached_prompt_tokens: 0,
                },
            });
        }
    }
}

fn generate(runtime: &EngineRuntime, prompt: &[u32], speculation: Speculation, sampling: SamplingConfig) -> Recording {
    let done = Handoff::new();
    runtime
        .generate_cancellable(
            GenerationRequest {
                prompt: prompt.to_vec(),
                max_tokens: GENERATED,
                stop_tokens: Vec::new(),
                speculation,
                sampling,
                constraint: None,
                listener: Box::new(Recorder {
                    tokens: Vec::new(),
                    verifications: Vec::new(),
                    done: Some(Arc::clone(&done)),
                }),
            },
            CancelHandle::default(),
        )
        .expect("the request is admitted");
    let recording = done.wait(WAIT).expect("the generation reports");
    assert_eq!(recording.outcome.reason, FinishReason::Length);
    assert_eq!(recording.tokens.len(), GENERATED as usize);
    recording
}

fn sampled(seed: u64) -> SamplingConfig {
    SamplingConfig::new(1.0, 40, 1.0, seed, false).unwrap()
}

/// A restored draft state drafts as a cold one does: every verification accepts what the cold run's accepted when the
/// restore brought every row whole (`exact`), and, when the last row was computed again (its low bits may differ), the
/// drafts accepted stay within a tenth of the cold run's. A draft cache that was not restored accepts far fewer.
fn drafts_as_a_cold_run(name: &str, warm: &Recording, cold: &Recording, exact: bool) {
    let (warm_accepted, warm_drafted) = warm.accepted();
    let (cold_accepted, cold_drafted) = cold.accepted();
    println!("{name}: accepted {warm_accepted}/{warm_drafted} warm, {cold_accepted}/{cold_drafted} cold");
    let first =
        |recording: &Recording| -> Vec<usize> { recording.verifications.iter().take(8).map(|v| v.accepted).collect() };
    println!(
        "{name}: first verifications {:?} warm, {:?} cold",
        first(warm),
        first(cold)
    );
    assert!(cold_accepted > 0, "{name}: the cold run accepts drafts");
    if exact {
        let counts = |recording: &Recording| -> Vec<(usize, usize)> {
            recording
                .verifications
                .iter()
                .map(|v| (v.accepted, v.drafted))
                .collect()
        };
        assert_eq!(
            counts(warm),
            counts(cold),
            "{name}: every verification of the restored run drafts and accepts as the cold run's"
        );
        assert_eq!(
            (warm_accepted, warm_drafted),
            (cold_accepted, cold_drafted),
            "{name}: accepted counts"
        );
        return;
    }
    let total = |recording: &Recording| first(recording).iter().sum::<usize>();
    assert!(
        total(warm).abs_diff(total(cold)) <= 2,
        "{name}: the first drafts are accepted as in a cold run (the row the restore re-pairs may differ in its low bits)"
    );
    assert!(
        warm_accepted * 10 >= cold_accepted * 9,
        "{name}: the restored draft state accepts {warm_accepted} drafts, a cold run {cold_accepted}"
    );
}

/// The prompts of the first test, how each is run, the position a warm engine restores, and whether the prompt has the
/// token after that position that the checkpoint paired its last row with.
fn cases() -> Vec<(&'static str, Vec<u32>, SamplingConfig, usize, bool)> {
    let a = prompt(0, LENGTH);
    let b = prompt(1, 2100);
    let c = prompt(2, 2560);
    // C's own tokens, extended past the checkpoints C's prompt is too short to reach.
    let d = prompt_of(2, 2560, LONG);
    assert_ne!(
        c[2560], a[2560],
        "the prompt that differs from the checkpoint's at its position"
    );
    vec![
        ("A greedy, first time", a.clone(), SamplingConfig::greedy(0), 0, true),
        ("A greedy, again", a.clone(), SamplingConfig::greedy(0), 2560, true),
        ("A sampled", a, sampled(11), 2560, true),
        (
            "B greedy, sharing 2100 tokens",
            b.clone(),
            SamplingConfig::greedy(0),
            2048,
            true,
        ),
        ("B sampled, after B's own checkpoints", b, sampled(12), 2560, true),
        (
            "C greedy, another token after A's checkpoint",
            c,
            SamplingConfig::greedy(0),
            2560,
            false,
        ),
        (
            "D greedy, C's tokens on past the next checkpoints, another token after A's checkpoint",
            d.clone(),
            SamplingConfig::greedy(0),
            2560,
            false,
        ),
        (
            "D sampled, through the checkpoints D stored",
            d,
            sampled(13),
            LAST_OF_D,
            true,
        ),
    ]
}

#[test]
#[ignore = "needs a GPU, the native library and the RCO artifact"]
fn a_speculative_restore_gives_the_tokens_and_the_drafts_of_a_cold_run() {
    let Some(machine) = machine() else { return };
    // The cold references: the cache is off.
    let (card, runtime) = cold(&machine);
    let lane = Arc::new(Typed::plain(runtime.library(), 1, 4));
    let cold: Vec<(Recording, MtpPages)> = cases()
        .iter()
        .map(|(_, prompt, sampling, _, _)| {
            let recording = generate(&runtime, prompt, Speculation::Artifact, *sampling);
            (recording, mtp_pages(&runtime, &lane))
        })
        .collect();
    for (recording, _) in &cold {
        assert_eq!(recording.outcome.cached_prompt_tokens, 0, "no cache restores nothing");
    }
    drop(lane);
    drop(runtime);
    drop(card);
    let _card = common::card();

    let (owner, runtime) = start(&machine, ByteSize::gib(2));
    let lane = Arc::new(Typed::plain(runtime.library(), 1, 4));
    // The rows each restore computes again; a checkpoint a later restore uses may have stored one of an earlier restore's.
    let mut repaired = Vec::new();
    // The cold run of the first prompt, whose row 2559 was paired with that prompt's own token at 2560.
    let a_pages = &cold[0].1;
    for ((name, prompt, sampling, restored, whole), (cold, cold_pages)) in cases().iter().zip(&cold) {
        if *restored > 0 {
            // An unrelated prompt leaves its own draft cache in the sequence's buffers, so a restore that did not copy
            // the draft state would find the wrong rows there (the same prompt's would pass for the restored ones).
            generate(
                &runtime,
                &prompt_unrelated(),
                Speculation::Artifact,
                SamplingConfig::greedy(0),
            );
        }
        let warm = generate(&runtime, prompt, Speculation::Artifact, *sampling);
        assert_eq!(
            warm.outcome.cached_prompt_tokens, *restored,
            "{name}: the restored position"
        );
        assert_eq!(
            warm.tokens, cold.tokens,
            "{name}: tokens after the restore equal the cold run's"
        );
        if *restored > 0 {
            drafts_as_a_cold_run(name, &warm, cold, *whole);
            if !whole {
                // A row computed again from the stored hidden row is not the row a cold run of another prompt made with
                // the other token: a restore that took the checkpoint's row whole would leave exactly those bytes.
                let row = restored - 1;
                assert_ne!(
                    row_of(&mtp_pages(&runtime, &lane), row),
                    row_of(a_pages, row),
                    "{name}: row {row} is the row paired with the checkpoint's token"
                );
                repaired.push(row);
            }
            same_mtp_rows(name, &mtp_pages(&runtime, &lane), cold_pages, &repaired);
        }
    }
    drop(lane);
    owner.shutdown(runtime).expect("the engine shuts down clean");
}

#[test]
#[ignore = "needs a GPU, the native library and the RCO artifact"]
fn a_plain_request_restores_the_checkpoints_of_a_speculative_one_but_not_the_reverse() {
    let Some(machine) = machine() else { return };
    let shared = prompt(0, LENGTH);
    let (card, runtime) = cold(&machine);
    let cold_plain = generate(&runtime, &shared, Speculation::Off, SamplingConfig::greedy(0));
    let cold_speculative = generate(&runtime, &shared, Speculation::Artifact, SamplingConfig::greedy(0));
    drop(runtime);
    drop(card);
    let _card = common::card();

    // A speculative request captures checkpoints that hold the draft's state: a plain one restores them and needs none.
    let (owner, runtime) = start(&machine, ByteSize::gib(2));
    let first = generate(&runtime, &shared, Speculation::Artifact, SamplingConfig::greedy(0));
    assert_eq!(first.outcome.cached_prompt_tokens, 0);
    assert_eq!(first.tokens, cold_speculative.tokens);
    let plain = generate(&runtime, &shared, Speculation::Off, SamplingConfig::greedy(0));
    assert_eq!(
        plain.outcome.cached_prompt_tokens, 2560,
        "a plain request restores a speculative checkpoint"
    );
    assert_eq!(plain.tokens, cold_plain.tokens);
    owner.shutdown(runtime).expect("the engine shuts down clean");

    // A plain request captures checkpoints without it: a speculative one cannot use them, starts cold and stores its own.
    let (owner, runtime) = start(&machine, ByteSize::gib(2));
    let first = generate(&runtime, &shared, Speculation::Off, SamplingConfig::greedy(0));
    assert_eq!(first.outcome.cached_prompt_tokens, 0);
    assert_eq!(first.tokens, cold_plain.tokens);
    let speculative = generate(&runtime, &shared, Speculation::Artifact, SamplingConfig::greedy(0));
    assert_eq!(
        speculative.outcome.cached_prompt_tokens, 0,
        "a speculative request misses a checkpoint without the draft's state"
    );
    assert_eq!(speculative.tokens, cold_speculative.tokens);
    let again = generate(&runtime, &shared, Speculation::Artifact, SamplingConfig::greedy(0));
    assert_eq!(again.outcome.cached_prompt_tokens, 2560, "it restores its own");
    assert_eq!(again.tokens, cold_speculative.tokens);
    drafts_as_a_cold_run("speculative after plain", &again, &cold_speculative, true);
    owner.shutdown(runtime).expect("the engine shuts down clean");
}
