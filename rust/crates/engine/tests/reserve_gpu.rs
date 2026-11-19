//! What the device loses after the plan is admitted, against the reserve the startup's claims leave. The tests are
//! `#[ignore]`d and run locally in a memory-capped scope. The artifact comes from `euhedral.qwen.rco-artifact` and the
//! library from `euhedral.cuda.library`.

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

mod common;

use std::mem;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use common::{Closer, limit_below_need, limit_gpu};
use euhedral_inference_cuda::CudaLibrary;
use euhedral_inference_engine::generation::{
    FinishReason, GenerationListener, GenerationOutcome, GenerationRequest, QuantumEvent, Speculation, Verification,
};
use euhedral_inference_engine::sampling::SamplingConfig;
use euhedral_inference_engine::{EngineRuntime, Handoff};
use euhedral_inference_settings::{Auto, ByteSize, KvFormat, ModelSettings, Settings};

/// The free bytes at the quanta of a generation, or `None` when it failed.
type Steps = Option<Vec<(String, u64)>>;

const WAIT: Duration = Duration::from_secs(20 * 60);

/// The free bytes the device showed when the quanta of a generation retired: the first of each kind, and each one lower
/// than all before.
struct Lowest {
    library: Arc<CudaLibrary>,
    steps: Vec<(String, u64)>,
    lowest: u64,
    done: Option<Arc<Handoff<Steps>>>,
}

impl GenerationListener for Lowest {
    fn prefill_done(&mut self, _at: Instant) {}
    fn token(&mut self, _id: u32, _at: Instant) {}
    fn quantum(&mut self, event: QuantumEvent) {
        let free = self.library.free_bytes().unwrap();
        // Every kind's first quantum, and every one that found the device lower than any before.
        let kind = format!("{:?}", event.kind);
        if free < self.lowest || !self.steps.iter().any(|(seen, _)| *seen == kind) {
            self.steps.push((kind, free));
        }
        self.lowest = self.lowest.min(free);
    }
    fn verification(&mut self, _event: Verification) {}
    fn finished(&mut self, outcome: GenerationOutcome) {
        let done = self.done.take().expect("a generation finishes once");
        let failed = matches!(outcome.reason, FinishReason::Failed(_));
        done.send((!failed).then(|| mem::take(&mut self.steps)));
    }
}

fn generate(runtime: &EngineRuntime, prompt: Vec<u32>, max_tokens: u32) -> Vec<(String, u64)> {
    let done = Handoff::new();
    runtime
        .generate(GenerationRequest {
            prompt,
            max_tokens,
            stop_tokens: Vec::new(),
            speculation: Speculation::Artifact,
            sampling: SamplingConfig::greedy(0),
            constraint: None,
            listener: Box::new(Lowest {
                library: Arc::clone(runtime.library()),
                steps: Vec::new(),
                lowest: u64::MAX,
                done: Some(Arc::clone(&done)),
            }),
        })
        .expect("the request is admitted");
    done.wait(WAIT)
        .expect("the generation reports")
        .expect("the generation did not fail")
}

#[test]
#[ignore = "needs a GPU, the native library and the RCO artifact"]
fn the_device_loses_no_more_after_admission_than_the_reserve_the_claims_leave() {
    let Some(library) = test_support::require_property("euhedral.cuda.library").map(PathBuf::from) else {
        return;
    };
    let Some(artifact) = test_support::require_property("euhedral.qwen.rco-artifact").map(PathBuf::from) else {
        return;
    };
    let mut settings = Settings::with_model(ModelSettings {
        artifact,
        name: "test".to_string(),
    });
    settings.context.max_tokens = 8192;
    settings.context.kv_cache_format = Auto::Value(KvFormat::Bf16);
    settings.prefix_cache.size = ByteSize(0);
    // The reserve is only what the settings need: the tightest the plan can be, drafting on.
    settings.memory.runtime_reserve = Auto::Value(ByteSize(0));
    let limit = limit_below_need(&library, &settings, 0);
    limit_gpu(&mut settings, limit);

    let (owner, ready) = EngineRuntime::start(&settings, &library);
    let runtime = ready
        .wait(WAIT)
        .expect("the engine reports")
        .expect("the engine starts");
    let closer = Closer::new(owner, &runtime);
    let (plan, report) = (runtime.plan(), runtime.report());
    assert_eq!(plan.budget_bytes, limit);
    assert!(plan.draft_depth > 0, "the plan drafts");

    let total = runtime.library().device().total_bytes;
    let at_ready = total - report.device_bytes_in_use;
    let left = runtime.reserve_left_bytes();
    // What the device lost after the plan was made, beyond what the engine allocated: the contexts's own use, the
    // modules and the driver's local memory of the warm-up's launches.
    let warm_loss = (report.free_at_plan as i64) - (at_ready as i64) - (report.allocated_bytes as i64);
    println!(
        "reserve {} MiB, left after the claims {} MiB; free at the plan {} MiB, at ready {} MiB, engine allocations {} MiB; \
         lost beyond the allocations by ready: {warm_loss} bytes",
        plan.runtime_reserve_bytes >> 20,
        left >> 20,
        report.free_at_plan >> 20,
        at_ready >> 20,
        report.allocated_bytes >> 20,
    );

    // A prompt of several prefill chunks, then enough tokens for drafts and verifications.
    let prompt: Vec<u32> = (0..1300u32).map(|i| 1000 + (i * 7919) % 50_000).collect();
    let steps = generate(&runtime, prompt, 64);
    let mut lowest = at_ready;
    for (index, (kind, free)) in steps.iter().enumerate() {
        println!(
            "quantum {index} ({kind}): free {free} bytes, {} below ready",
            at_ready as i64 - *free as i64
        );
        lowest = lowest.min(*free);
    }
    let after_ready = at_ready - lowest;
    let lost = warm_loss.max(0) as u64 + after_ready;
    println!(
        "lost beyond the allocations: {warm_loss} by ready, {after_ready} by the generation, {lost} in all; the reserve left {left}"
    );
    assert!(
        steps.iter().any(|(kind, _)| kind == "Verify"),
        "the run verified drafts"
    );
    // The measurement sees what the driver took: a device that lost nothing was not measured.
    assert!(lost > 0, "the device lost nothing after admission");
    assert!(
        lost <= left,
        "the device lost {lost} bytes after admission, the claims left {left} of the reserve"
    );
    closer.shutdown(runtime).unwrap();
}
