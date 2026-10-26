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

use euhedral_inference_edrl::{HEADER_BYTES, Header, LayerType, ModelMetadata, TensorTable};
use euhedral_inference_engine::error::PlanError;
use euhedral_inference_engine::model::{ArtifactProfile, ModelConfig, Quantization, Speculation};
use euhedral_inference_engine::residency::planner::{
    ARENA_ALIGNMENT, PlacementSource, PlanInputs, RUNTIME_RESERVE_BYTES, plan, plan_with,
};
use euhedral_inference_engine::workspace::{BufferSlot, WorkspaceError, WorkspaceRows, slot_bytes, workspace_bytes};
use euhedral_inference_settings::{Auto, ByteSize, KvFormat, ModelSettings, Settings};

const GIB: u64 = 1 << 30;
const MIB: u64 = 1 << 20;

/// The Java numbers for the RCO artifact (see the module comment).
const JAVA_WEIGHTS_BYTES: u64 = 12_012_811_100;
const JAVA_KV_131072: u64 = 2_415_919_104;
const JAVA_KV_32768: u64 = 603_979_776;
const JAVA_PAGE_BYTES: u64 = 294_912;
const JAVA_TRELLIS_SCRATCH_512: u64 = 160_669_696;
const JAVA_WORKSPACE_512: u64 = 172_140_552;

fn fixture() -> &'static str {
    include_str!("fixtures/rco_table.tsv")
}

fn put_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_be_bytes());
}

fn put_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_be_bytes());
}

/// The RCO artifact's table, rebuilt in the container's binary form with synthetic offsets, and parsed.
fn rco_table_without(skip: impl Fn(&str) -> bool) -> TensorTable {
    let rows: Vec<Vec<&str>> = fixture()
        .lines()
        .filter(|line| !line.starts_with('#'))
        .map(|line| line.split('\t').collect::<Vec<_>>())
        .filter(|columns| !skip(columns[0]))
        .collect();
    let mut table = Vec::new();
    let mut entries: Vec<(Vec<u8>, u64)> = Vec::new();
    for columns in &rows {
        let shape: Vec<u64> = columns[1].split('x').map(|d| d.parse().unwrap()).collect();
        let mut entry = Vec::new();
        put_u32(&mut entry, columns[0].len() as u32);
        entry.extend_from_slice(columns[0].as_bytes());
        put_u32(&mut entry, shape.len() as u32);
        for dimension in &shape {
            put_u64(&mut entry, *dimension);
        }
        for ordinal in &columns[2..5] {
            put_u32(&mut entry, ordinal.parse().unwrap());
        }
        entries.push((entry, columns[5].parse().unwrap()));
    }
    let table_bytes: usize = entries.iter().map(|(entry, _)| entry.len() + 16).sum();
    let table_offset = HEADER_BYTES as u64;
    let data_offset = table_offset + table_bytes as u64;
    let mut offset = data_offset.div_ceil(4096) * 4096;
    for (entry, bytes) in &entries {
        table.extend_from_slice(entry);
        put_u64(&mut table, offset);
        put_u64(&mut table, *bytes);
        offset = (offset + bytes).div_ceil(4096) * 4096;
    }
    let mut header = Vec::new();
    put_u32(&mut header, 0x5157_454E);
    put_u32(&mut header, 2);
    put_u64(&mut header, HEADER_BYTES as u64);
    put_u64(&mut header, 0);
    put_u64(&mut header, table_offset);
    put_u32(&mut header, entries.len() as u32);
    put_u32(&mut header, 0);
    put_u64(&mut header, data_offset);
    let header = Header::parse(header.as_slice().try_into().unwrap()).unwrap();
    TensorTable::parse(&table, &header, offset).unwrap()
}

fn rco_table() -> TensorTable {
    rco_table_without(|_| false)
}

fn metadata() -> ModelMetadata {
    ModelMetadata {
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
    }
}

fn config() -> ModelConfig {
    ModelConfig::from(&metadata())
}

fn settings(max_tokens: u32) -> Settings {
    let mut settings = Settings::with_model(ModelSettings::default());
    settings.context.max_tokens = max_tokens;
    settings
}

fn inputs(device_free: u64) -> PlanInputs {
    PlanInputs {
        device_free,
        multiprocessors: Some(70),
        host_pinned_bytes: u64::MAX,
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
    assert_eq!(plan.workspace_bytes, JAVA_WORKSPACE_512);
    assert_eq!(plan.kv_format, KvFormat::Nvfp4);
    assert_eq!(plan.draft_depth, 4);
    // Java's device need is the weights, the cache, the reserve and the scratch; the arena aligns each object.
    assert_eq!(
        plan.device_bytes - plan.arena_bytes,
        JAVA_KV_131072 + 1280 * MIB + JAVA_TRELLIS_SCRATCH_512
    );
    assert_eq!(RUNTIME_RESERVE_BYTES, 1280 * MIB);
    assert_eq!(
        plan.budget_bytes,
        16 * GIB - 700 * MIB,
        "the default reserve is 700 MiB"
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
    let error = plan_with(&config(), &rco_table(), &settings(32_768), inputs(8 * GIB)).unwrap_err();
    let PlanError::WeightsDoNotFit {
        weights_needed,
        state_needed,
        available,
        free,
    } = error
    else {
        panic!("expected WeightsDoNotFit, got {error:?}");
    };
    assert_eq!(free, 8 * GIB);
    assert_eq!(available, 8 * GIB - 700 * MIB);
    let arena = plan_with(&config(), &rco_table(), &settings(32_768), inputs(16 * GIB))
        .unwrap()
        .arena_bytes;
    assert_eq!(weights_needed, arena);
    assert_eq!(state_needed, JAVA_KV_32768 + 1280 * MIB + JAVA_TRELLIS_SCRATCH_512);
    let message = PlanError::WeightsDoNotFit {
        weights_needed,
        state_needed,
        available,
        free,
    }
    .to_string();
    assert!(message.contains(&weights_needed.to_string()), "{message}");
    assert!(message.contains(&available.to_string()), "{message}");
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
    let roomy = plan_with(&config, &table, &settings, inputs(need_whole + 700 * MIB)).unwrap();
    assert_eq!((roomy.kv_device_tokens, roomy.kv_host_tokens), (131_072, 0));

    // One byte short of that: the cache past 32768 tokens moves to the host.
    let short = plan_with(&config, &table, &settings, inputs(need_whole + 700 * MIB - 1)).unwrap();
    assert_eq!((short.kv_device_tokens, short.kv_host_tokens), (32_768, 98_304));
    assert_eq!(short.kv_device_bytes, kept);
    assert_eq!(short.kv_host_bytes, spilled);
    assert_eq!(short.device_bytes, need_spilled);

    // The boundary of the spilled plan.
    assert!(plan_with(&config, &table, &settings, inputs(need_spilled + 700 * MIB)).is_ok());
    assert!(matches!(
        plan_with(&config, &table, &settings, inputs(need_spilled + 700 * MIB - 1)),
        Err(PlanError::WeightsDoNotFit { .. })
    ));

    // Host memory that cannot hold the pages leaves the weights to make room, which they cannot.
    let mut host_short = inputs(need_whole + 700 * MIB - 1);
    host_short.host_pinned_bytes = spilled - 1;
    assert!(matches!(
        plan_with(&config, &table, &settings, host_short),
        Err(PlanError::WeightsDoNotFit { .. })
    ));
    host_short.host_pinned_bytes = spilled;
    assert!(plan_with(&config, &table, &settings, host_short).is_ok());

    // A context that already fits in kv-cache-on-gpu-tokens has nothing to spill.
    settings.context.max_tokens = 32_768;
    settings.context.kv_cache_on_gpu_tokens = 65_536;
    let small_short = arena + JAVA_KV_32768 + 1280 * MIB + JAVA_TRELLIS_SCRATCH_512 + 700 * MIB - 1;
    assert!(matches!(
        plan_with(&config, &table, &settings, inputs(small_short)),
        Err(PlanError::WeightsDoNotFit { .. })
    ));
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
    assert_eq!(workspace_bytes(&config, rows), JAVA_WORKSPACE_512);
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
        workspace_bytes(
            &config,
            WorkspaceRows {
                rows: 256,
                logits_rows: plan.rows.logits_rows
            }
        )
    );
    assert!(plan.workspace_bytes < JAVA_WORKSPACE_512);
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
fn gpu_reserve_and_limit_respected() {
    let table = rco_table();
    let config = config();
    let base = settings(32_768);
    let need = {
        let plan = plan_with(&config, &table, &base, inputs(16 * GIB)).unwrap();
        plan.device_bytes
    };

    // The reserve comes off what is free.
    let mut reserved = base.clone();
    reserved.memory.gpu_reserve = ByteSize::gib(1);
    assert!(plan_with(&config, &table, &reserved, inputs(need + GIB)).is_ok());
    assert!(matches!(
        plan_with(&config, &table, &reserved, inputs(need + GIB - 1)),
        Err(PlanError::WeightsDoNotFit { .. })
    ));

    // The limit caps what is free, however much is.
    let mut limited = base.clone();
    limited.memory.gpu_reserve = ByteSize(0);
    limited.memory.gpu_limit = Auto::Value(ByteSize(need));
    assert!(plan_with(&config, &table, &limited, inputs(64 * GIB)).is_ok());
    limited.memory.gpu_limit = Auto::Value(ByteSize(need - 1));
    let error = plan_with(&config, &table, &limited, inputs(64 * GIB)).unwrap_err();
    let PlanError::WeightsDoNotFit { available, free, .. } = error else {
        panic!("expected WeightsDoNotFit, got {error:?}");
    };
    assert_eq!(available, need - 1);
    assert_eq!(free, 64 * GIB, "the free memory is reported as it was");

    // A reserve that takes everything leaves no budget.
    let mut empty = base;
    empty.memory.gpu_reserve = ByteSize::gib(16);
    assert!(matches!(
        plan_with(&config, &table, &empty, inputs(16 * GIB)),
        Err(PlanError::NoBudget { .. })
    ));
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
