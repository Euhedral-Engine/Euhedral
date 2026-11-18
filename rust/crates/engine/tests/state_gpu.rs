//! The sequence state and the workspace on a GPU. The tests are `#[ignore]`d and run locally in a memory-capped scope:
//! `systemd-run --user --scope -p MemoryMax=16G cargo test -p euhedral-inference-engine --test state_gpu -- --ignored`.
//!
//! The library comes from the `euhedral.cuda.library` property (`EUHEDRAL_TEST_CUDA_LIBRARY` in the environment, or
//! `gradle.properties`); a test skips with a message when it is absent.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]
#![allow(
    clippy::disallowed_methods,
    reason = "no-wait, waiting calls: a test synchronizes a lane and copies synchronously to check results"
)]

#[path = "common/typed.rs"]
mod typed;

use std::fmt::Display;
use std::path::PathBuf;
use std::sync::Arc;

use euhedral_graph::Marker;
use euhedral_inference_cuda::{CudaError, CudaLibrary, DeviceBuffer, LaneOps, PinnedPool};
use euhedral_inference_edrl::{LayerType, ModelMetadata};
use euhedral_inference_engine::model::ModelConfig;
use euhedral_inference_engine::residency::footprint;
use euhedral_inference_engine::residency::planner::plan;
use euhedral_inference_engine::state::SequenceState;
use euhedral_inference_engine::state::attention::AttentionScratch;
use euhedral_inference_engine::state::gdn::GdnStates;
use euhedral_inference_engine::state::kv::{KvCache, KvError, KvRetained};
use euhedral_inference_engine::state::released::ReleasedBuffers;
use euhedral_inference_engine::workspace::{BufferSlot, Workspace, WorkspaceError, WorkspaceRows, workspace_bytes};
use euhedral_inference_settings::KvFormat;
use typed::Typed;

mod common;

fn library() -> Option<Arc<CudaLibrary>> {
    let path = test_support::require_property("euhedral.cuda.library")?;
    Some(CudaLibrary::load(&PathBuf::from(path)).expect("the CUDA library loads"))
}

/// One RustEE lane of 16 markers.
fn lane(library: &Arc<CudaLibrary>) -> Typed {
    Typed::plain(library, 1, 16)
}

/// Runs `work` as one submission of the lane and waits for the device; returns its result and its marker.
fn run<R, E: Display>(lane: &Typed, work: impl FnOnce(&mut dyn LaneOps) -> Result<R, E>) -> (R, Marker) {
    let made = lane.table.submit_ops(lane.lane(0), false, work).unwrap();
    lane.sync();
    made
}

fn config() -> ModelConfig {
    ModelConfig::from(&ModelMetadata {
        vocab_size: 248_320,
        hidden_size: 5120,
        num_hidden_layers: 64,
        num_attention_heads: 24,
        num_key_value_heads: 4,
        attention_head_dim: 256,
        intermediate_size: 17_408,
        linear_num_key_heads: 16,
        linear_num_value_heads: 48,
        linear_key_head_dim: 128,
        linear_value_head_dim: 128,
        linear_conv_kernel_dim: 4,
        rms_norm_epsilon: 1e-6,
        rope_theta: 1e7,
        partial_rotary_factor: 0.25,
        max_position_embeddings: 262_144,
        hidden_activation: "silu".to_string(),
        layer_types: (0..64)
            .map(|layer| {
                if layer % 4 == 3 {
                    LayerType::FullAttention
                } else {
                    LayerType::GatedDeltaNet
                }
            })
            .collect(),
        num_experts: 0,
        num_experts_per_token: 0,
        moe_intermediate_size: 0,
        shared_expert_intermediate_size: 0,
        tie_word_embeddings: false,
        attention_output_gate: true,
        mtp_layer_count: 1,
        draft_depth: Some(4),
    })
}

fn words(bytes: &[u8]) -> Vec<u64> {
    bytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|word| u64::from_ne_bytes(*word))
        .collect()
}

#[test]
#[ignore = "needs a GPU and the native library"]
fn a_reservation_allocates_pages_and_a_table_that_names_them() {
    let Some(library) = library() else { return };
    let lane = lane(&library);
    let pool = PinnedPool::new(&library, 64 << 20);
    let released = Arc::new(ReleasedBuffers::new());
    let mut cache = KvCache::new(&library, &released, 2, 1024, KvFormat::Nvfp4, 4096, 0).unwrap();
    let plane = cache.plane_page_bytes();
    assert_eq!(plane, 256 * 144 * 4);

    // 300 rows need two pages and a table of two slots per plane.
    let mut retained = KvRetained::default();
    run(&lane, |ops| cache.reserve(ops, &pool, 0, 300, &mut retained));
    assert_eq!((cache.reserved(), cache.submitted(), cache.committed()), (512, 0, 0));
    assert!(cache.reserves(0, 512) && !cache.reserves(0, 513));
    for layer in 0..2 {
        let key = cache.key_table(layer).unwrap();
        let mut table = vec![0u8; 4 * 8];
        let probe = DeviceBuffer::alloc(&library, 64).unwrap();
        // The table lives on the device: copy it into a buffer the test can read.
        run(&lane, |ops| ops.copy_device_to_device(probe.ptr(), key, 32));
        probe.copy_to_host(0, &mut table).unwrap();
        let entries = words(&table);
        let pages: Vec<u64> = cache.pages(layer).map(|page| page.get()).collect();
        assert_eq!(pages.len(), 2);
        assert_eq!(entries[..2], pages[..], "the K table names the K planes");
        assert_eq!(
            entries[2],
            pages[0] + plane,
            "the V table follows the K table, and names the V planes"
        );
        assert_eq!(entries[3], pages[1] + plane);
        assert_eq!(cache.value_table(layer).unwrap().get(), key.get() + 2 * 8);
    }
    assert!(released.is_empty(), "the first table replaces nothing");
    assert_eq!(retained.staging.len(), 2);
    // The lane has run the upload (the run waited for it), so the staging may go back to the pool.
    retained.staging.clear();

    // Frontiers: the writes were queued, then the quantum retired.
    cache.appended(300).unwrap();
    assert_eq!(cache.submitted(), 300);
    assert!(cache.appended(213).is_err(), "beyond the reserved rows");
    cache.commit(200).unwrap();
    assert_eq!(
        (cache.reserved(), cache.submitted(), cache.committed()),
        (512, 300, 200)
    );
    assert!(cache.commit(101).is_err(), "beyond the submitted rows");
    cache.commit_prefix(50).unwrap();
    assert_eq!((cache.submitted(), cache.committed()), (250, 250));
    cache.truncate(240).unwrap();
    assert_eq!((cache.submitted(), cache.committed()), (240, 240));
    assert!(cache.truncate(241).is_err());

    // The next append must start at the submitted frontier.
    let mut retained = KvRetained::default();
    run(&lane, |ops| {
        assert!(matches!(
            cache.reserve(ops, &pool, 100, 10, &mut retained),
            Err(KvError::NotContiguous {
                start: 100,
                submitted: 240
            })
        ));
        Ok::<(), KvError>(())
    });
    // Growing past two pages doubles the table. No view holds the old tables, so the cache's let-go is their last:
    // they are handed over for the stream-ordered free.
    run(&lane, |ops| cache.reserve(ops, &pool, 240, 300, &mut retained));
    assert_eq!(cache.reserved(), 768);
    assert_eq!(released.waiting(), 2, "the two old tables");
    assert_eq!(retained.staging.len(), 2);
    assert_eq!(
        cache.value_table(0).unwrap().get(),
        cache.key_table(0).unwrap().get() + 4 * 8
    );
    retained.staging.clear();
    // The outgrown tables are freed in stream order on a lane.
    let (freed, _) = run(&lane, |ops| {
        let mut freed = 0;
        released.drain(|buffer| {
            ops.free_async(buffer).unwrap();
            freed += 1;
        });
        Ok::<usize, CudaError>(freed)
    });
    assert_eq!(freed, 2);
    assert!(released.is_empty());

    // Past the planned rows is refused.
    let mut retained = KvRetained::default();
    run(&lane, |ops| {
        assert!(matches!(
            cache.reserve(ops, &pool, 240, 4000, &mut retained),
            Err(KvError::TooLong {
                required: 4240,
                planned: 4096
            })
        ));
        Ok::<(), KvError>(())
    });
}

#[test]
#[ignore = "needs a GPU and the native library"]
fn a_buffer_that_is_not_stream_ordered_is_refused_by_the_stream_ordered_free() {
    let Some(library) = library() else { return };
    let lane = lane(&library);
    let plain = DeviceBuffer::alloc(&library, 4096).unwrap();
    let (refusal, _) = run(&lane, |ops| {
        Ok::<_, String>(ops.free_async(plain).unwrap_err().to_string())
    });
    assert!(refusal.contains("alloc_async"), "{refusal}");
}

#[test]
#[ignore = "needs a GPU and the native library"]
fn host_pages_are_read_and_written_through_the_device_region() {
    let Some(library) = library() else { return };
    let lane = lane(&library);
    let pool = PinnedPool::new(&library, 64 << 20);
    // One page on the device, two in host memory.
    let mut cache = KvCache::new(
        &library,
        &Arc::new(ReleasedBuffers::new()),
        1,
        1024,
        KvFormat::Nvfp4,
        256,
        512,
    )
    .unwrap();
    let page = 2 * cache.plane_page_bytes();
    assert_eq!(cache.planned_rows(), 768);
    let mut retained = KvRetained::default();
    run(&lane, |ops| cache.reserve(ops, &pool, 0, 600, &mut retained));
    assert_eq!(cache.device_pages(), 1);
    assert_eq!(cache.reserved_host_pages(), 2);
    let region = cache.host_region().unwrap().ptr().get();
    let probe = DeviceBuffer::alloc(&library, 64).unwrap();
    let key = cache.key_table(0).unwrap();
    run(&lane, |ops| ops.copy_device_to_device(probe.ptr(), key, 32));
    let mut table = vec![0u8; 32];
    probe.copy_to_host(0, &mut table).unwrap();
    let entries = words(&table);
    let device_page = cache.pages(0).next().unwrap().get();
    assert_eq!(entries[0], device_page);
    assert_eq!(entries[1], region, "the first host page is the region's first");
    assert_eq!(entries[2], region + page);

    // Stage in: the host pages appear in the region.
    let arena = cache.host_arena_mut().unwrap();
    for (index, byte) in arena.as_mut_slice().iter_mut().take(2 * page as usize).enumerate() {
        *byte = (index % 251) as u8;
    }
    run(&lane, |ops| cache.stage_in(ops, 0, 520, 80));
    let mut seen = vec![0u8; 2 * page as usize];
    cache.host_region().unwrap().copy_to_host(0, &mut seen).unwrap();
    assert!(
        seen.iter()
            .enumerate()
            .all(|(index, byte)| *byte == (index % 251) as u8)
    );

    // Stage out: what a quantum wrote to the region (the last two reserved pages) returns to the host.
    let written: Vec<u8> = (0..2 * page as usize).map(|index| (index % 13) as u8).collect();
    cache.host_region().unwrap().copy_from_host(0, &written).unwrap();
    run(&lane, |ops| cache.stage_out(ops, 0, 596, 4));
    let arena = cache.host_arena_mut().unwrap();
    assert_eq!(&arena.as_slice()[..2 * page as usize], &written[..]);
}

#[test]
#[ignore = "needs a GPU and the native library"]
fn gdn_state_is_zeroed_and_as_large_as_the_planner_says() {
    let Some(library) = library() else { return };
    let lane = lane(&library);
    let config = config();
    let before = library.free_bytes().unwrap();
    let (states, _) = run(&lane, |ops| GdnStates::allocate(&library, &config, 5, ops));
    assert_eq!(states.device_bytes(), footprint::gdn_bytes(&config, 5));
    assert!(states.layer(3).is_none(), "layer 3 is full attention");
    let state = states.layer(0).unwrap();
    let mut recurrent = vec![0xFFu8; state.recurrent().bytes()];
    state.recurrent().copy_to_host(0, &mut recurrent).unwrap();
    assert!(recurrent.iter().all(|byte| *byte == 0));
    let mut convolution = vec![0xFFu8; state.convolution().bytes()];
    state.convolution().copy_to_host(0, &mut convolution).unwrap();
    assert!(convolution.iter().all(|byte| *byte == 0));
    assert_eq!(
        state.speculative().unwrap().capacity,
        8,
        "at least 8 rows, as the Java state allocates"
    );
    let used = before - library.free_bytes().unwrap();
    assert!(used >= states.device_bytes(), "{used} < {}", states.device_bytes());

    let (plain, _) = run(&lane, |ops| GdnStates::allocate(&library, &config, 0, ops));
    let state = plain.layer(0).unwrap();
    assert!(state.speculative().is_none());
    assert_eq!(
        plain.device_bytes(),
        48 * (state.convolution().bytes() + state.recurrent().bytes()) as u64
    );
}

#[test]
#[ignore = "needs a GPU and the native library"]
fn the_extra_state_of_a_sequence_is_what_its_buffers_hold() {
    let Some(library) = library() else { return };
    let lane = lane(&library);
    let config = common::config();
    let plan = plan(&config, &common::rco_table(), &common::settings(4096), 64 << 30).unwrap();
    let released = Arc::new(ReleasedBuffers::new());
    // A real sequence of the plan's shape, allocated as startup does.
    let (state, _) = run(&lane, |ops| {
        SequenceState::allocate(&library, &released, &config, &plan, ops)
    });
    let drafts = plan.draft_depth > 0;
    let verified = if drafts { 1 + plan.draft_depth as u64 } else { 0 };
    let selections = if drafts { (verified + 1).max(2) } else { 2 } * verified.max(1) * 8;
    // The MTP layer's cache holds its pages once they are reserved: none yet.
    let mtp_pages = if drafts {
        footprint::pages_for(plan.max_context_tokens as u64) * footprint::page_bytes(&config, plan.kv_format)
    } else {
        0
    };
    assert_eq!(
        footprint::extra_sequence_bytes(
            &config,
            plan.kv_format,
            plan.max_context_tokens as u64,
            verified,
            drafts,
            (plan.rows.rows as u64).max(footprint::MINIMUM_VERIFY_ROWS),
        ),
        state.device_bytes() - state.kv.device_bytes() + selections + mtp_pages
    );
}

#[test]
#[ignore = "needs a GPU and the native library"]
fn the_workspace_holds_one_buffer_per_slot_and_refuses_a_larger_quantum() {
    let Some(library) = library() else { return };
    let config = config();
    let rows = WorkspaceRows {
        rows: 64,
        logits_rows: 8,
    };
    let workspace = Workspace::allocate(&library, &config, rows, &[1 << 20, 0, 2 << 20]).unwrap();
    assert_eq!(
        workspace.retained_bytes(),
        workspace_bytes(&config, rows) + (3 << 20),
        "regions of 0 bytes are not allocated"
    );
    let view = workspace.view(64).unwrap();
    assert_eq!(view.rows(), 64);
    assert_eq!(view.bytes(&config, BufferSlot::GateUp), 64 * 2 * 17_408 * 2);
    assert_eq!(
        view.ptr(BufferSlot::HiddenState),
        workspace.buffer(BufferSlot::HiddenState).ptr()
    );
    assert_eq!(
        workspace.view(65).unwrap_err(),
        WorkspaceError::TooManyRows { rows: 65, capacity: 64 }
    );
    assert!(view.logits(8).is_ok());
    assert!(view.logits(9).is_err());
    // Every slot is its own allocation.
    let mut addresses: Vec<u64> = BufferSlot::ALL
        .iter()
        .map(|slot| workspace.buffer(*slot).ptr().get())
        .collect();
    addresses.sort_unstable();
    addresses.dedup();
    assert_eq!(addresses.len(), BufferSlot::ALL.len());
}

#[test]
#[ignore = "needs a GPU and the native library"]
fn primed_staging_means_a_reservation_allocates_nothing() {
    let Some(library) = library() else { return };
    let lane = lane(&library);
    let pool = PinnedPool::new(&library, 64 << 20);
    let mut cache = KvCache::new(
        &library,
        &Arc::new(ReleasedBuffers::new()),
        2,
        1024,
        KvFormat::Nvfp4,
        8192,
        0,
    )
    .unwrap();
    cache.prime_staging(&pool, 2).unwrap();
    let primed = pool.allocated_bytes();
    assert!(primed > 0);
    // Reserve page by page up to the planned rows, so the table doubles through every size it can have.
    let mut retained = KvRetained::default();
    run(&lane, |ops| {
        for page in 0..32u64 {
            cache.reserve(ops, &pool, page * 256, 256, &mut retained)?;
            cache.appended(256)?;
            // The caller retires the staging behind the lane's marker; here the sets in flight are the two primed.
            if retained.staging.len() >= 4 {
                retained.staging.clear();
            }
        }
        Ok::<(), KvError>(())
    });
    assert_eq!(pool.allocated_bytes(), primed, "no reservation allocated pinned memory");
    assert_eq!(cache.reserved(), 8192);
}
