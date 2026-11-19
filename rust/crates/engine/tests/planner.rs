//! The residency planner on the tensor table of the RCO artifact (`tests/fixtures/rco_table.tsv`: names, shapes and
//! formats, no payload), with fake device sizes.
//!
//! The expected numbers are the Java engine's, worked from `ResidencyPlanner` for the same inputs:
//! - the resident weights are `deviceNeed` (the sum of the uploaded objects' byte sizes);
//! - the cache is `kvBytes` (`ResidencyPlanner.java:84`): 16 attention layers x pages x (K plane + V plane), the
//!   NVFP4 plane of one page being 256 rows x 144 bytes x 4 KV heads;
//! - the reserve is `RESERVE_BYTES` (`ResidencyPlanner.java:36`), 1280 MiB;
//! - the trellis scratch is `trellisScratchBytes` (`ResidencyPlanner.java:303`) at 512 rows with
//!   `TrellisSettings.defaults()` (`TrellisFormat.java:167`);
//! - the spill is `plan` (`ResidencyPlanner.java:161`): device pages = min(pages, ceil(kvDeviceMinimumTokens / 256)),
//!   host pages = the rest, one device region of the host pages beside the device pages.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]

use euhedral_inference_cuda::launch::nvfp4::nvfp4_native_scratch_bytes;
use euhedral_inference_edrl::WeightFormat;
use euhedral_inference_engine::error::PlanError;
use euhedral_inference_engine::model::{ArtifactProfile, Quantization, Speculation};
use euhedral_inference_engine::residency::footprint;
use euhedral_inference_engine::residency::planner::{
    ARENA_ALIGNMENT, DEFAULT_RUNTIME_RESERVE_BYTES, PlacementSource, PlanInputs, ResidencyPlan, plan, plan_closest,
    plan_with, sequences_for,
};
use euhedral_inference_engine::runtime::resolve_sequences;
use euhedral_inference_engine::workspace::{
    BufferSlot, WorkspaceError, WorkspaceRows, input_records, slot_bytes, workspace_bytes, workspace_bytes_with_records,
};
use euhedral_inference_settings::{Auto, ByteSize, KvFormat, Settings};

mod common;

use common::{config, rco_table, rco_table_without, settings};

const GIB: u64 = 1 << 30;
const MIB: u64 = 1 << 20;

/// The Java numbers for the RCO artifact (see the module comment).
const JAVA_WEIGHTS_BYTES: u64 = 12_012_811_100;
const JAVA_KV_131072: u64 = 2_415_919_104;
const JAVA_KV_32768: u64 = 603_979_776;
const JAVA_PAGE_BYTES: u64 = 294_912;
const JAVA_TRELLIS_SCRATCH_512: u64 = 160_669_696;
/// The Java workspace at 512 rows, which holds one input record; this engine holds two (`INPUT_RECORDS`).
const WORKSPACE_512: u64 = 172_140_552 + (512 * 4 + 8);
/// The same for a sequence that drafts 4 tokens a step: each quantum of a step reads a record, 5 in all.
const WORKSPACE_512_MTP4: u64 = WORKSPACE_512 + 3 * (512 * 4 + 8);

fn inputs(device_free: u64) -> PlanInputs {
    PlanInputs {
        device_free,
        multiprocessors: Some(70),
        host_pinned_bytes: Some(u64::MAX),
        sequences: 1,
    }
}

#[test]
fn the_fixture_is_the_rco_table() {
    let table = rco_table();
    assert_eq!(table.len(), 1745);
    let sum: u64 = table.entries().iter().map(|entry| entry.bytes()).sum();
    assert_eq!(sum, JAVA_WEIGHTS_BYTES);
    let config = config();
    let profile = ArtifactProfile::of(&table, &config).unwrap();
    assert_eq!(profile.quantization, Quantization::Trellis);
    assert_eq!(profile.speculation, Speculation::Mtp);
    assert_eq!(profile.speculative_depth().unwrap(), 4);
}

#[test]
fn rco_fits_resident_at_131072_on_16_gib() {
    let plan = plan_with(&config(), &rco_table(), &settings(131_072), inputs(16 * GIB)).unwrap();
    // Every object the load places is resident; the packed MTP projection became its two halves.
    assert_eq!(
        plan.weights_bytes,
        JAVA_WEIGHTS_BYTES + 4,
        "the halves carry one more trailer than the pack"
    );
    assert_eq!(plan.kv_device_tokens, 131_072);
    assert_eq!(plan.kv_host_tokens, 0);
    assert_eq!(plan.kv_device_bytes, JAVA_KV_131072);
    assert_eq!(plan.kv_host_bytes, 0);
    assert_eq!(plan.scratch_bytes, JAVA_TRELLIS_SCRATCH_512);
    assert_eq!(plan.workspace_bytes, WORKSPACE_512_MTP4);
    assert_eq!(plan.kv_format, KvFormat::Nvfp4);
    assert_eq!(plan.draft_depth, 4);
    // Java's device need is the weights, the cache, the reserve and the scratch; the arena aligns each object.
    assert_eq!(
        plan.device_bytes - plan.arena_bytes,
        JAVA_KV_131072 + 1280 * MIB + JAVA_TRELLIS_SCRATCH_512
    );
    assert_eq!(DEFAULT_RUNTIME_RESERVE_BYTES, 1280 * MIB);
    assert_eq!(
        plan.budget_bytes,
        16 * GIB,
        "the dense plan's budget is what is free (Java InferenceEngine.deviceBudget); memory.gpu-reserve is not used"
    );
    assert!(plan.device_bytes <= plan.budget_bytes);
}

#[test]
fn the_arena_holds_every_object_aligned_and_apart() {
    let table = rco_table();
    let plan = plan(&config(), &table, &settings(32_768), 16 * GIB).unwrap();
    let mut ends = 0;
    let mut last_end = 0;
    for placement in &plan.device {
        assert_eq!(placement.offset % ARENA_ALIGNMENT, 0, "{}", placement.name);
        assert!(
            placement.offset >= last_end,
            "{} overlaps its predecessor",
            placement.name
        );
        last_end = placement.offset + placement.bytes;
        ends = ends.max(last_end);
    }
    assert!(ends <= plan.arena_bytes && plan.arena_bytes - ends < ARENA_ALIGNMENT);
    // 1745 objects, with the pack replaced by two halves.
    assert_eq!(plan.device.len(), 1746);
    let pack = "mtp/layer/attention/query_key_gate_value";
    assert!(plan.placement(pack).is_none());
    let (first, second) = (
        plan.placement("mtp/layer/attention/query_key").unwrap(),
        plan.placement("mtp/layer/attention/gate_value").unwrap(),
    );
    assert_eq!(first.shape, [7168, 5120]);
    assert_eq!(first.bytes, second.bytes);
    // Java `Nvfp4Layout.byteSize(7168, 5120)`: codes 18350080, scales 2293760, trailer at the next 256, 4 bytes.
    assert_eq!(first.bytes, 20_643_844);
    let source = table.get(pack).unwrap();
    assert_eq!(first.file_offset, source.offset());
    assert_eq!(
        first.source,
        PlacementSource::Rows {
            object: pack.to_string(),
            first_row: 0,
            object_rows: 14_336
        }
    );
    assert_eq!(
        second.source,
        PlacementSource::Rows {
            object: pack.to_string(),
            first_row: 7168,
            object_rows: 14_336
        }
    );
}

#[test]
fn planner_refuses_when_weights_do_not_fit() {
    // 8 GiB now offloads what leaves it room; a device that cannot hold what stays is refused.
    let error = plan_with(&config(), &rco_table(), &settings(32_768), inputs(GIB)).unwrap_err();
    let PlanError::DeviceTooSmallWithOffload {
        weights_needed,
        state_needed,
        available,
        free,
        sequences,
        context_tokens,
        host,
    } = error
    else {
        panic!("expected DeviceTooSmallWithOffload, got {error:?}");
    };
    assert_eq!(free, GIB);
    assert_eq!(available, GIB);
    assert_eq!(sequences, 1);
    assert_eq!(context_tokens, 32_768);
    let arena = plan_with(&config(), &rco_table(), &settings(32_768), inputs(16 * GIB))
        .unwrap()
        .arena_bytes;
    // What stays on the device is the arena the closest plan keeps and its staging ring, exactly.
    let (closest, _) = plan_closest(&config(), &rco_table(), &settings(32_768), inputs(GIB)).unwrap();
    assert!(closest.arena_bytes < arena);
    assert_eq!(weights_needed, closest.arena_bytes + closest.offload.staging_bytes());
    assert_eq!(state_needed, JAVA_KV_32768 + 1280 * MIB + JAVA_TRELLIS_SCRATCH_512);
    let message = PlanError::DeviceTooSmallWithOffload {
        weights_needed,
        state_needed,
        available,
        free,
        sequences,
        context_tokens,
        host,
    }
    .to_string();
    assert!(
        !message.contains("sequences"),
        "a plan of one sequence blames none: {message}"
    );
    assert!(message.contains(&weights_needed.to_string()), "{message}");
    assert!(message.contains(&available.to_string()), "{message}");
    assert!(message.contains("1024 MiB free on this GPU"), "{message}");
}

#[test]
fn a_device_that_is_not_known_gets_the_widest_trellis_split() {
    // `plan` does not know the device's multiprocessors, so it sizes the K split's partials for the widest split the
    // shapes allow: never less than a device's own.
    let config = config();
    let table = rco_table();
    let known = plan_with(&config, &table, &settings(32_768), inputs(16 * GIB)).unwrap();
    let unknown = plan(&config, &table, &settings(32_768), 16 * GIB).unwrap();
    assert!(unknown.scratch_bytes >= known.scratch_bytes);
}

#[test]
fn kv_spills_beyond_kv_cache_on_gpu_tokens_when_short() {
    let table = rco_table();
    let config = config();
    let mut settings = settings(131_072);
    settings.context.kv_cache_on_gpu_tokens = 32_768;
    let whole = plan_with(&config, &table, &settings, inputs(16 * GIB)).unwrap();
    let arena = whole.arena_bytes;
    // Java: 512 pages, 128 on the device, 384 in host memory; the host pages are copied through one device region.
    let region = 384 * JAVA_PAGE_BYTES;
    let spilled = region * 16;
    assert_eq!(spilled, 1_811_939_328);
    let kept = JAVA_KV_32768 + region;
    let need_spilled = arena + kept + 1280 * MIB + JAVA_TRELLIS_SCRATCH_512;
    let need_whole = arena + JAVA_KV_131072 + 1280 * MIB + JAVA_TRELLIS_SCRATCH_512;
    assert!(need_spilled < need_whole);

    // Room for the whole cache: nothing spills.
    let roomy = plan_with(&config, &table, &settings, inputs(need_whole)).unwrap();
    assert!(roomy.offload.is_empty());
    assert_eq!((roomy.kv_device_tokens, roomy.kv_host_tokens), (131_072, 0));

    // One byte short of that: the cache past 32768 tokens moves to the host.
    let short = plan_with(&config, &table, &settings, inputs(need_whole - 1)).unwrap();
    assert_eq!((short.kv_device_tokens, short.kv_host_tokens), (32_768, 98_304));
    assert_eq!(short.kv_device_bytes, kept);
    assert_eq!(short.kv_host_bytes, spilled);
    assert_eq!(short.device_bytes, need_spilled);

    // The boundary of the spilled plan.
    assert!(
        plan_with(&config, &table, &settings, inputs(need_spilled))
            .unwrap()
            .offload
            .is_empty()
    );
    // One byte short of that: the weights leave the device too, the embedding first, and the pages stay on the host.
    let deeper = plan_with(&config, &table, &settings, inputs(need_spilled - 1)).unwrap();
    assert_eq!(deeper.offload.host, ["text/token_embedding"]);
    assert_eq!(deeper.kv_host_bytes, spilled);

    // Host memory that cannot hold the pages leaves the weights to make room instead: the pages stay on the device.
    let mut host_short = inputs(need_whole - 1);
    host_short.host_pinned_bytes = Some(spilled - 1);
    let moved = plan_with(&config, &table, &settings, host_short).unwrap();
    assert_eq!((moved.kv_device_tokens, moved.kv_host_tokens), (131_072, 0));
    assert!(!moved.offload.is_empty());
    host_short.host_pinned_bytes = Some(spilled);
    let spilling = plan_with(&config, &table, &settings, host_short).unwrap();
    assert_eq!(spilling.kv_host_bytes, spilled);
    assert!(spilling.offload.is_empty());

    // A context that already fits in kv-cache-on-gpu-tokens has nothing to spill.
    settings.context.max_tokens = 32_768;
    settings.context.kv_cache_on_gpu_tokens = 65_536;
    let small_short = arena + JAVA_KV_32768 + 1280 * MIB + JAVA_TRELLIS_SCRATCH_512 - 1;
    let moved = plan_with(&config, &table, &settings, inputs(small_short)).unwrap();
    assert_eq!(moved.kv_host_tokens, 0);
    assert_eq!(moved.offload.host, ["text/token_embedding"]);
}

#[test]
fn workspace_sized_for_prefill_chunk() {
    let config = config();
    // Java `ExecutionPlan.fullBufferSpecs` at 512 rows, the logits at a verification's rows (8), and the input record
    // of `Workspace.inputByteSize` (the ids padded to 8 bytes, then one position).
    let rows = WorkspaceRows {
        rows: 512,
        logits_rows: 8,
    };
    assert_eq!(workspace_bytes(&config, rows), WORKSPACE_512);
    assert_eq!(slot_bytes(&config, BufferSlot::GateUp, rows), 512 * 2 * 17_408 * 2);
    assert_eq!(slot_bytes(&config, BufferSlot::GdnAlpha, rows), 512 * 48 * 4);
    assert_eq!(slot_bytes(&config, BufferSlot::Logits, rows), 8 * 248_320 * 2);
    assert_eq!(slot_bytes(&config, BufferSlot::TokenIds, rows), 512 * 4 + 8);
    // The plan sizes the workspace from batching.prefill-chunk-tokens.
    let mut settings = settings(32_768);
    settings.batching.prefill_chunk_tokens = 256;
    let plan = plan_with(&config, &rco_table(), &settings, inputs(16 * GIB)).unwrap();
    assert_eq!(plan.rows.rows, 256);
    assert_eq!(
        plan.workspace_bytes,
        workspace_bytes_with_records(
            &config,
            WorkspaceRows {
                rows: 256,
                logits_rows: plan.rows.logits_rows
            },
            input_records(4)
        )
    );
    assert!(plan.workspace_bytes < WORKSPACE_512_MTP4);
    assert_eq!(input_records(0), 2);
    assert_eq!(input_records(1), 2);
    assert_eq!(input_records(4), 5);
    // A quantum larger than the workspace is refused, and so is one that scores more rows than the logits hold.
    assert_eq!(rows.check(512), Ok(()));
    assert_eq!(
        rows.check(513),
        Err(WorkspaceError::TooManyRows {
            rows: 513,
            capacity: 512
        })
    );
    assert_eq!(rows.check_logits(8), Ok(()));
    assert!(rows.check_logits(9).is_err());
}

#[test]
fn gpu_limit_respected_and_gpu_reserve_not_used_by_the_dense_plan() {
    let table = rco_table();
    let config = config();
    let base = settings(32_768);
    let need = plan_with(&config, &table, &base, inputs(16 * GIB))
        .unwrap()
        .device_bytes;

    // The plan fits exactly what is free, with no gpu-reserve taken off (Java InferenceEngine.deviceBudget).
    assert!(
        plan_with(&config, &table, &base, inputs(need))
            .unwrap()
            .offload
            .is_empty()
    );
    assert_eq!(
        plan_with(&config, &table, &base, inputs(need - 1))
            .unwrap()
            .offload
            .host,
        ["text/token_embedding"]
    );
    let mut reserved = base.clone();
    reserved.memory.gpu_reserve = ByteSize::gib(8);
    assert!(
        plan_with(&config, &table, &reserved, inputs(need)).is_ok(),
        "gpu-reserve is a Flash-Next setting"
    );

    // The limit caps what is free, however much is.
    let mut limited = base.clone();
    limited.memory.gpu_limit = Auto::Value(ByteSize(need));
    assert!(plan_with(&config, &table, &limited, inputs(64 * GIB)).is_ok());
    limited.memory.gpu_limit = Auto::Value(ByteSize(need - 1));
    let offloaded = plan_with(&config, &table, &limited, inputs(64 * GIB)).unwrap();
    assert_eq!(offloaded.offload.host, ["text/token_embedding"]);
    limited.memory.gpu_limit = Auto::Value(ByteSize(GIB));
    let error = plan_with(&config, &table, &limited, inputs(64 * GIB)).unwrap_err();
    let PlanError::DeviceTooSmallWithOffload { available, free, .. } = error else {
        panic!("expected DeviceTooSmallWithOffload, got {error:?}");
    };
    assert_eq!(available, GIB);
    assert_eq!(free, 64 * GIB, "the free memory is reported as it was");
}

#[test]
fn the_runtime_reserve_is_a_setting_whose_auto_is_javas() {
    let table = rco_table();
    let config = config();
    let base = settings(32_768);
    let auto = plan_with(&config, &table, &base, inputs(16 * GIB)).unwrap();
    let mut stated = base.clone();
    stated.memory.runtime_reserve = Auto::Value(ByteSize::mib(1280));
    assert_eq!(
        plan_with(&config, &table, &stated, inputs(16 * GIB))
            .unwrap()
            .device_bytes,
        auto.device_bytes
    );

    stated.memory.runtime_reserve = Auto::Value(ByteSize::mib(2000));
    let larger = plan_with(&config, &table, &stated, inputs(16 * GIB)).unwrap();
    assert_eq!(larger.device_bytes, auto.device_bytes + 720 * MIB);
    assert!(
        !plan_with(&config, &table, &stated, inputs(larger.device_bytes - 1))
            .unwrap()
            .offload
            .is_empty()
    );
    // A reserve below what the settings make the runtime need is raised to it: the workspace, the decode scratch, the
    // first sequence's state and the MTP layer's cache are paid from the reserve and counted.
    stated.memory.runtime_reserve = Auto::Value(ByteSize(0));
    let none = plan_with(&config, &table, &stated, inputs(16 * GIB)).unwrap();
    assert_eq!(none.runtime_reserve_bytes, reserve_demand(&none));
    assert_eq!(
        none.device_bytes,
        auto.device_bytes - 1280 * MIB + none.runtime_reserve_bytes
    );
}

/// What the runtime reserve pays for in a plan, from the pieces that allocate it.
fn reserve_demand(plan: &ResidencyPlan) -> u64 {
    let config = config();
    plan.workspace_bytes
        + plan.decode_scratch_bytes
        + footprint::first_sequence_bytes(plan, &config)
        + footprint::mtp_cache_bytes_of(plan, &config)
}

/// The device bytes that hold `big` with the reserve at its default: `big`'s bytes less what the reserve was raised by.
fn counted_before(big: &ResidencyPlan) -> u64 {
    big.device_bytes - (big.runtime_reserve_bytes - DEFAULT_RUNTIME_RESERVE_BYTES)
}

/// Whether `settings` are planned against `device_free` and a host that holds nothing: the plan is admitted whole on
/// the device or refused, it cannot offload.
fn admitted_on(settings: &Settings, device_free: u64) -> bool {
    let mut inputs = inputs(device_free);
    inputs.host_pinned_bytes = Some(0);
    plan_with(&config(), &rco_table(), settings, inputs).is_ok()
}

#[test]
fn the_defaults_keep_the_default_reserve() {
    for tokens in [4096, 32_768, 131_072] {
        let plan = plan_with(&config(), &rco_table(), &settings(tokens), inputs(64 * GIB)).unwrap();
        assert_eq!(
            plan.runtime_reserve_bytes, DEFAULT_RUNTIME_RESERVE_BYTES,
            "{tokens} tokens"
        );
        assert!(
            reserve_demand(&plan) <= DEFAULT_RUNTIME_RESERVE_BYTES,
            "{tokens} tokens"
        );
    }
}

#[test]
fn the_mtp_cache_of_a_long_bf16_context_is_counted_and_a_device_it_overflows_is_refused_when_the_host_cannot_take_the_weights()
 {
    let mut long = settings(262_144);
    long.context.kv_cache_format = Auto::Value(KvFormat::Bf16);
    long.context.kv_cache_on_gpu_tokens = 262_144;
    let big = plan_with(&config(), &rco_table(), &long, inputs(64 * GIB)).unwrap();
    let mtp = footprint::mtp_cache_bytes_of(&big, &config());
    assert!(mtp > 1000 * MIB, "the layer's cache at 262144 tokens is {mtp}");
    assert!(big.runtime_reserve_bytes > DEFAULT_RUNTIME_RESERVE_BYTES);
    assert_eq!(big.runtime_reserve_bytes, reserve_demand(&big));
    assert!(admitted_on(&long, big.device_bytes));
    // Enough for everything but the reserve's growth: refused resident now.
    assert!(!admitted_on(&long, counted_before(&big)));
}

#[test]
fn a_plan_without_the_mtp_layer_counts_no_mtp_cache() {
    let table = rco_table_without(|name| name.starts_with("mtp/"));
    let mut config = config();
    config.mtp_layer_count = 0;
    let mut long = settings(262_144);
    long.context.kv_cache_format = Auto::Value(KvFormat::Bf16);
    long.context.kv_cache_on_gpu_tokens = 262_144;
    let plan = plan_with(&config, &table, &long, inputs(64 * GIB)).unwrap();
    assert_eq!(plan.draft_depth, 0);
    assert_eq!(footprint::mtp_cache_bytes_of(&plan, &config), 0);
    assert_eq!(plan.runtime_reserve_bytes, DEFAULT_RUNTIME_RESERVE_BYTES);
    assert!(reserve_demand(&plan) < DEFAULT_RUNTIME_RESERVE_BYTES);
}

#[test]
fn the_first_scratch_region_is_the_larger_of_the_trellis_and_the_native_scratch() {
    let table = rco_table();
    let plan = plan_with(&config(), &table, &settings(4096), inputs(64 * GIB)).unwrap();
    let native = plan
        .device
        .iter()
        .filter(|placement| placement.format == WeightFormat::Nvfp4)
        .map(|placement| nvfp4_native_scratch_bytes(plan.rows.rows, placement.shape[1], placement.shape[0]))
        .max()
        .unwrap();
    assert!(native > 0);
    assert!(plan.scratch_bytes >= native);
    // On this artifact the trellis routes' scratch is the larger: the figures of the plan are the ones it always had.
    assert_eq!(plan.scratch_bytes, JAVA_TRELLIS_SCRATCH_512);
}

#[test]
fn a_larger_prefill_chunk_counts_its_workspace_and_a_device_that_cannot_hold_it_is_refused_when_the_host_cannot_take_the_weights()
 {
    let mut chunked = settings(32_768);
    chunked.batching.prefill_chunk_tokens = 2048;
    let big = plan_with(&config(), &rco_table(), &chunked, inputs(64 * GIB)).unwrap();
    assert!(big.runtime_reserve_bytes > DEFAULT_RUNTIME_RESERVE_BYTES);
    assert_eq!(big.runtime_reserve_bytes, reserve_demand(&big));
    assert!(admitted_on(&chunked, big.device_bytes));
    assert!(!admitted_on(&chunked, counted_before(&big)));
}

#[test]
fn a_table_that_is_not_the_model_is_refused_before_anything_else() {
    let table = rco_table_without(|name| name == "text/layers/10/mlp/down");
    let error = plan(&config(), &table, &settings(32_768), 16 * GIB).unwrap_err();
    assert!(matches!(error, PlanError::Inventory(_)), "{error:?}");
    assert!(error.to_string().contains("text/layers/10/mlp/down"), "{error}");

    let table = rco_table_without(|name| name.ends_with("layers/3/mlp/down.suh"));
    let error = plan(&config(), &table, &settings(32_768), 16 * GIB).unwrap_err();
    assert!(error.to_string().contains("missing its companion"), "{error}");

    let mut too_long = settings(262_145);
    too_long.context.kv_cache_on_gpu_tokens = 1;
    let error = plan(&config(), &rco_table(), &too_long, 16 * GIB).unwrap_err();
    assert!(matches!(error, PlanError::Setting(_)), "{error:?}");
}

#[test]
fn the_cache_format_follows_the_artifact_unless_the_settings_choose() {
    let table = rco_table();
    let config = config();
    let mut settings = settings(4096);
    let auto = plan(&config, &table, &settings, 16 * GIB).unwrap();
    assert_eq!(auto.kv_format, KvFormat::Nvfp4);
    settings.context.kv_cache_format = Auto::Value(KvFormat::Bf16);
    let bf16 = plan(&config, &table, &settings, 16 * GIB).unwrap();
    assert_eq!(bf16.kv_format, KvFormat::Bf16);
    // 512-byte head rows: 16 layers x 16 pages x 2 planes x 256 x 512 x 4 heads.
    assert_eq!(bf16.kv_device_bytes, 16 * 16 * 2 * 256 * 512 * 4);
    assert_eq!(auto.kv_device_bytes, 16 * 16 * 2 * 256 * 144 * 4);
}

fn with_sequences(device_free: u64, sequences: usize) -> PlanInputs {
    PlanInputs {
        sequences,
        ..inputs(device_free)
    }
}

/// The device bytes each sequence beyond the first takes in the RCO plan (it drafts 4 tokens a step), besides its cache.
fn extra_of(plan: &ResidencyPlan) -> u64 {
    footprint::extra_sequence_bytes(
        &config(),
        plan.kv_format,
        plan.max_context_tokens as u64,
        1 + plan.draft_depth as u64,
        true,
        plan.rows.rows,
    )
}

#[test]
fn a_plan_of_one_sequence_is_the_plan_that_knew_none() {
    let (config, table) = (config(), rco_table());
    let one = plan_with(&config, &table, &settings(131_072), with_sequences(16 * GIB, 1)).unwrap();
    let unaware = PlanInputs {
        multiprocessors: None,
        ..with_sequences(16 * GIB, 1)
    };
    assert_eq!(
        plan(&config, &table, &settings(131_072), 16 * GIB).unwrap(),
        plan_with(&config, &table, &settings(131_072), unaware).unwrap()
    );
    assert_eq!(one.sequences, 1);
    assert_eq!(one.extra_sequence_bytes, 0);
    assert_eq!(
        one.device_bytes,
        one.arena_bytes + JAVA_KV_131072 + 1280 * MIB + JAVA_TRELLIS_SCRATCH_512
    );
    assert_eq!(one.kv_device_bytes, JAVA_KV_131072);
}

#[test]
fn a_plan_of_no_sequence_is_refused() {
    let error = plan_with(&config(), &rco_table(), &settings(32_768), with_sequences(16 * GIB, 0)).unwrap_err();
    assert!(matches!(error, PlanError::Setting(_)), "{error:?}");
}

#[test]
fn each_sequence_adds_its_cache_and_its_state() {
    let (config, table) = (config(), rco_table());
    let settings = settings(32_768);
    let one = plan_with(&config, &table, &settings, with_sequences(16 * GIB, 1)).unwrap();
    let four = plan_with(&config, &table, &settings, with_sequences(16 * GIB, 4)).unwrap();
    let extra = extra_of(&four);
    assert!(extra > 0);
    assert_eq!(four.sequences, 4);
    assert_eq!(four.extra_sequence_bytes, extra);
    // What is shared does not grow, but the workspace's input records (each sequence has records of its own), which the runtime reserve holds.
    assert_eq!(four.arena_bytes, one.arena_bytes);
    let records = input_records(four.draft_depth);
    let records_of = |count: u64| workspace_bytes_with_records(&config, four.rows, count * records);
    assert_eq!(
        one.workspace_bytes,
        workspace_bytes_with_records(&config, one.rows, records)
    );
    assert_eq!(four.workspace_bytes, records_of(4));
    assert!(four.workspace_bytes > one.workspace_bytes);
    assert_eq!(four.scratch_bytes, one.scratch_bytes);
    // Each sequence's cache is the one sequence's.
    assert_eq!(four.kv_device_bytes, one.kv_device_bytes);
    assert_eq!(four.device_bytes, one.device_bytes + 3 * (JAVA_KV_32768 + extra));
    assert_eq!(four.location_bytes().device, four.device_bytes);
}

#[test]
fn the_boundary_of_a_plan_of_several_sequences_and_what_the_refusal_says() {
    let (config, table) = (config(), rco_table());
    let settings = settings(32_768);
    let need = plan_with(&config, &table, &settings, with_sequences(16 * GIB, 3))
        .unwrap()
        .device_bytes;
    assert!(plan_with(&config, &table, &settings, with_sequences(need, 3)).is_ok());
    // One byte short: the embedding leaves the device.
    let offloaded = plan_with(&config, &table, &settings, with_sequences(need - 1, 3)).unwrap();
    assert_eq!(offloaded.offload.host, ["text/token_embedding"]);
    let error = plan_with(&config, &table, &settings, with_sequences(GIB, 3)).unwrap_err();
    assert!(
        matches!(error, PlanError::DeviceTooSmallWithOffload { sequences: 3, .. }),
        "{error:?}"
    );
    let message = error.to_string();
    assert!(message.contains("or fewer sequences"), "{message}");
    assert!(message.contains("batching.max-sequences"), "{message}");
    assert!(message.contains("context.max-tokens"), "{message}");
    // The same device holds one sequence.
    assert!(plan_with(&config, &table, &settings, with_sequences(need - 1, 1)).is_ok());
}

#[test]
fn the_host_pages_of_every_sequence_must_fit_the_host() {
    let (config, table) = (config(), rco_table());
    let mut settings = settings(131_072);
    settings.context.kv_cache_on_gpu_tokens = 32_768;
    let spilled = 1_811_939_328;
    let two_need = {
        let whole = plan_with(&config, &table, &settings, with_sequences(64 * GIB, 2)).unwrap();
        // Room for the whole caches of both sequences: nothing spills.
        assert_eq!(whole.kv_host_bytes, 0);
        whole.device_bytes - 1
    };
    let mut host = with_sequences(two_need, 2);
    host.host_pinned_bytes = Some(2 * spilled - 1);
    // The pages of both sequences do not fit the host: they stay on the device and the weights make the room.
    let moved = plan_with(&config, &table, &settings, host).unwrap();
    assert_eq!(moved.kv_host_bytes, 0);
    assert!(!moved.offload.is_empty());
    host.host_pinned_bytes = Some(2 * spilled);
    let plan = plan_with(&config, &table, &settings, host).unwrap();
    assert_eq!(plan.kv_host_bytes, 2 * spilled);
    assert_eq!(plan.kv_host_tokens, 98_304);
}

#[test]
fn the_most_sequences_that_fit_is_found_and_one_more_does_not_fit() {
    let (config, table) = (config(), rco_table());
    let settings = settings(32_768);
    let one = plan_with(&config, &table, &settings, with_sequences(16 * GIB, 1)).unwrap();
    let extra = extra_of(&plan_with(&config, &table, &settings, with_sequences(16 * GIB, 2)).unwrap());
    let each = JAVA_KV_32768 + extra;
    // Room for 5 sequences and a byte short of a sixth.
    let free = one.device_bytes + 4 * each + each - 1;
    assert_eq!(sequences_for(&config, &table, &settings, inputs(free), 1000), Ok(5));
    assert!(plan_with(&config, &table, &settings, with_sequences(free, 5)).is_ok());
    assert!(
        !plan_with(&config, &table, &settings, with_sequences(free, 6))
            .unwrap()
            .offload
            .is_empty(),
        "a sixth sequence is made room for by offloading weights"
    );
    // The cap wins when more would fit, and a cap below 1 is 1.
    assert_eq!(sequences_for(&config, &table, &settings, inputs(free), 3), Ok(3));
    assert_eq!(sequences_for(&config, &table, &settings, inputs(free), 0), Ok(1));
    assert_eq!(sequences_for(&config, &table, &settings, inputs(64 * GIB), 7), Ok(7));
    // Exactly one fits.
    assert_eq!(
        sequences_for(&config, &table, &settings, inputs(one.device_bytes), 7),
        Ok(1)
    );
    // The `sequences` of the inputs is not read.
    assert_eq!(
        sequences_for(&config, &table, &settings, with_sequences(free, 9), 1000),
        Ok(5)
    );
}

#[test]
fn a_device_too_small_for_one_sequence_is_refused_not_given_one() {
    let (config, table) = (config(), rco_table());
    let error = sequences_for(&config, &table, &settings(32_768), inputs(GIB), 7).unwrap_err();
    assert!(
        matches!(error, PlanError::DeviceTooSmallWithOffload { sequences: 1, .. }),
        "{error:?}"
    );
}

#[test]
fn more_sequences_fit_when_each_keeps_part_of_its_cache_on_the_host() {
    let (config, table) = (config(), rco_table());
    let mut whole = settings(131_072);
    let one = plan_with(&config, &table, &whole, with_sequences(16 * GIB, 1)).unwrap();
    let free = one.device_bytes + 2 * (JAVA_KV_131072 + extra_of(&one));
    let on_device = sequences_for(&config, &table, &whole, inputs(free), 16).unwrap();
    whole.context.kv_cache_on_gpu_tokens = 32_768;
    let spilled = sequences_for(&config, &table, &whole, inputs(free), 16).unwrap();
    assert!(
        spilled > on_device,
        "{spilled} sequences with host pages, {on_device} without"
    );
}

#[test]
fn an_explicit_count_is_the_count_and_zero_is_the_most_that_fit() {
    let (config, table) = (config(), rco_table());
    let mut settings = settings(32_768);
    let roomy = inputs(16 * GIB);
    settings.batching.max_sequences = 3;
    assert_eq!(resolve_sequences(&config, &table, &settings, roomy), Ok(3));
    // An explicit count is not judged here: the plan refuses it with the bytes it needed.
    settings.batching.max_sequences = 100_000;
    assert_eq!(resolve_sequences(&config, &table, &settings, inputs(GIB)), Ok(100_000));

    settings.batching.max_sequences = 0;
    // The cap is the rows a pass holds over a verification's rows: 512 / (1 + 4 drafts).
    assert_eq!(
        resolve_sequences(&config, &table, &settings, inputs(1024 * GIB)),
        Ok(102)
    );
    settings.batching.max_tokens_per_pass = Auto::Value(20);
    assert_eq!(resolve_sequences(&config, &table, &settings, inputs(1024 * GIB)), Ok(4));
    settings.batching.max_tokens_per_pass = Auto::Value(1);
    assert_eq!(resolve_sequences(&config, &table, &settings, inputs(1024 * GIB)), Ok(1));
    // A pass cannot hold more rows than a prefill chunk.
    settings.batching.max_tokens_per_pass = Auto::Value(5000);
    assert_eq!(
        resolve_sequences(&config, &table, &settings, inputs(1024 * GIB)),
        Ok(102)
    );
    // Limited by the device.
    settings.batching.max_tokens_per_pass = Auto::Auto;
    let one = plan_with(&config, &table, &settings, with_sequences(16 * GIB, 1)).unwrap();
    let each = JAVA_KV_32768 + extra_of(&one);
    assert_eq!(
        resolve_sequences(&config, &table, &settings, inputs(one.device_bytes + 2 * each)),
        Ok(3)
    );
}

#[test]
fn the_extra_state_of_a_sequence_without_drafts_is_its_gdn_state_scratch_and_selections() {
    let config = config();
    let plain = footprint::extra_sequence_bytes(&config, KvFormat::Nvfp4, 32_768, 0, false, 512);
    let selections = 2 * 8;
    // Without verification the state is the recurrent and convolution state of each GDN layer, nothing more.
    let convolution = (config.gdn_query_key_width() + config.gdn_value_width()) as u64
        * (config.linear_conv_kernel_dim as u64 - 1)
        * 2;
    let recurrent = config.linear_num_value_heads as u64
        * config.linear_key_head_dim as u64
        * config.linear_value_head_dim as u64
        * 4;
    assert_eq!(
        footprint::gdn_bytes(&config, 0),
        config.gdn_layers() as u64 * (recurrent + convolution)
    );
    assert_eq!(
        plain,
        footprint::gdn_bytes(&config, 0)
            + footprint::decode_scratch_bytes(config.num_attention_heads as u64, 1)
            + selections
    );
    let verified = 5;
    let drafting = footprint::extra_sequence_bytes(&config, KvFormat::Nvfp4, 32_768, verified, true, 512);
    let hidden = config.hidden_size as u64;
    // 6 selections of 5 rows' keys: the drafts and two more.
    assert_eq!(
        drafting,
        footprint::gdn_bytes(&config, verified)
            + footprint::decode_scratch_bytes(config.num_attention_heads as u64, 8)
            + 6 * verified * 8
            + 512 * hidden * 2
            + hidden * 2
            + 4 * 17
            + 128 * footprint::page_bytes(&config, KvFormat::Nvfp4)
    );
    // The seed rows are a prefill chunk's, and the MTP cache follows the context.
    let longer = footprint::extra_sequence_bytes(&config, KvFormat::Nvfp4, 65_536, verified, true, 1024);
    assert_eq!(
        longer - drafting,
        512 * hidden * 2 + 128 * footprint::page_bytes(&config, KvFormat::Nvfp4)
    );
}

#[test]
fn the_prefix_cache_arena_is_granted_only_the_host_room_left_over() {
    let (config, table) = (config(), rco_table());
    let mut settings = settings(131_072);
    settings.prefix_cache.size = ByteSize::gib(1);
    let granted = |settings: &Settings, host: u64| {
        let mut inputs = inputs(64 * GIB);
        inputs.host_pinned_bytes = Some(host);
        plan_with(&config, &table, settings, inputs).unwrap().prefix_cache_bytes
    };
    assert_eq!(granted(&settings, u64::MAX), GIB);
    // A host too small for the whole arena grants what is left of it, and the plan is still made.
    let shrunk = granted(&settings, GIB - 1);
    assert!(shrunk > 0 && shrunk < GIB, "{shrunk}");
    assert_eq!(granted(&settings, 0), 0);
    assert_eq!(granted(&settings, GIB), GIB);
    settings.prefix_cache.size = ByteSize(0);
    assert_eq!(granted(&settings, u64::MAX), 0);
}

#[test]
fn a_plan_of_several_sequences_sizes_the_logits_for_a_pass_and_counts_them() {
    let (config, table) = (config(), rco_table());
    let settings = settings(32_768);
    let one = plan_with(&config, &table, &settings, with_sequences(16 * GIB, 1)).unwrap();
    let four = plan_with(&config, &table, &settings, with_sequences(16 * GIB, 4)).unwrap();
    let verified = 1 + one.draft_depth as u64;
    assert!(one.draft_depth > 0);
    assert!(one.rows.logits_rows >= verified && one.rows.logits_rows < four.rows.logits_rows);
    assert_eq!(one.rows.logits_rows, 8, "one sequence is sized as it always was");
    assert_eq!(one.pass_parts, 1);
    assert_eq!(four.pass_parts, 4);
    assert_eq!(four.pass_rows as u64, 4 * verified);
    assert_eq!(four.rows.logits_rows, four.pass_rows as u64);
    assert_eq!(one.rows.rows, four.rows.rows);
    // The workspace the plan counts is the one that holds those rows.
    assert!(
        four.workspace_bytes > workspace_bytes_with_records(&config, one.rows, 4 * input_records(four.draft_depth))
    );
}
