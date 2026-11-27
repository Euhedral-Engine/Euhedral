//! The plan's contract on synthetic objects: sizes the documented rules give, and the refusals at their edges.

use euhedral_inference_edrl::{
    ComponentGroup, ExpertBank, ExpertProjection, ExpertRecord, FlashTensor, TensorDataType, WeightFormat, WeightLayout,
};
use euhedral_inference_settings::{ByteSize, MemorySettings, WeightStreamingSettings};

use super::*;
use crate::model::flash_config::fixtures::flash_config;

const MIB: u64 = 1 << 20;
const GIB: u64 = 1 << 30;

fn tensor(name: &str, group: ComponentGroup, mib: u64) -> FlashTensor {
    FlashTensor {
        name: name.to_owned(),
        shape: vec![1],
        dtype: TensorDataType::Bf16,
        format: WeightFormat::Bf16,
        layout: WeightLayout::ContiguousLeV1,
        group,
        offset: 0,
        bytes: mib * MIB,
        crc32: 0,
    }
}

fn bank(name: &str, group: ComponentGroup, experts: usize, record_bytes: u64) -> ExpertBank {
    let projection = ExpertProjection {
        name: "gate".to_owned(),
        shape: vec![1],
        dtype: TensorDataType::Bf16,
        format: WeightFormat::Nvfp4,
        layout: WeightLayout::RowSplitK128V1,
        record_offset: 0,
        bytes: 1,
    };
    let record = ExpertRecord {
        offset: 0,
        bytes: record_bytes,
        crc32: 0,
    };
    ExpertBank::new(name, group, 0, vec![projection], vec![record; experts]).unwrap()
}

/// 64 routed experts per layer, 2 serving a token (so 4 minimum slots), two text layers.
fn config() -> FlashModelConfig {
    let mut config = FlashModelConfig::from_flash(&flash_config()).unwrap();
    config.num_experts = 64;
    config
}

/// Two banks of 100 one-MiB experts (200 slots at most), an MTP bank of 10 experts.
fn banks() -> Vec<ExpertBank> {
    vec![
        bank("text/layers/0/experts", ComponentGroup::RoutedExpert, 100, MIB),
        bank("text/layers/1/experts", ComponentGroup::RoutedExpert, 100, MIB),
        bank("mtp/layers/0/experts", ComponentGroup::Mtp, 10, MIB),
    ]
}

/// Objects none of which may leave the device: everything of at most 4 MiB, plus the groups no rank names.
fn immovable() -> Vec<FlashTensor> {
    vec![
        tensor("text/token_embedding", ComponentGroup::TokenEmbedding, 10),
        tensor("text/layers/0/router", ComponentGroup::Router, 1),
        tensor("text/layers/0/norm", ComponentGroup::NormSmallState, 1),
        tensor("ngram/shard", ComponentGroup::Ngram, 50),
        tensor("mtp/proj", ComponentGroup::Mtp, 30),
        tensor("vision/proj", ComponentGroup::Vision, 20),
    ]
}

/// The immovable objects and the ones that may leave: the output head, two shared experts, two GDN projections.
fn movable_world() -> Vec<FlashTensor> {
    let mut tensors = immovable();
    tensors.push(tensor("text/output_head", ComponentGroup::OutputHead, 100));
    for layer in 0..2 {
        tensors.push(tensor(
            &format!("text/layers/{layer}/shared"),
            ComponentGroup::SharedExpert,
            20,
        ));
        tensors.push(tensor(
            &format!("text/layers/{layer}/gdn/proj"),
            ComponentGroup::Gdn,
            40,
        ));
    }
    tensors
}

/// Settings with the smallest staging (4 reads in flight, the two tokens' experts) unless a test says otherwise.
fn memory(gpu_reserve_mib: u64) -> MemorySettings {
    MemorySettings {
        gpu_reserve: ByteSize::mib(gpu_reserve_mib),
        weight_streaming: WeightStreamingSettings {
            concurrent_reads: 4,
            ..WeightStreamingSettings::default()
        },
        ..MemorySettings::default()
    }
}

fn inputs(device_free: u64, context: usize) -> FlashPlanInputs {
    FlashPlanInputs {
        device_free,
        host: HostBudget::stated(64 * GIB),
        max_context_tokens: context,
        mode: FlashMode::TEXT,
    }
}

/// The device bytes the plan keeps beside the model and the cache for `context` tokens at the 512-token chunk.
fn reserved(config: &FlashModelConfig, gpu_reserve_mib: u64, context: usize) -> u64 {
    let sizes = SequenceState::new(config);
    KERNEL_RESERVE_BYTES
        + gpu_reserve_mib * MIB
        + sizes.context_bytes(context).unwrap()
        + sizes.workspace_bytes(512).unwrap()
        + sizes.shared_down_padding_bytes().unwrap()
}

fn plan(tensors: &[FlashTensor], memory: &MemorySettings, inputs: FlashPlanInputs) -> Result<FlashPlan, PlanError> {
    plan_flash(&config(), tensors, &banks(), memory, inputs)
}

#[test]
fn everything_stays_on_the_device_and_the_cache_takes_the_rest_in_whole_slots() {
    let tensors = immovable();
    // Fixed resident: the embedding is mapped, so 2 MiB; the cache budget is 50.5 slots.
    let free = reserved(&config(), 700, 256) + 2 * MIB + 50 * MIB + MIB / 2;
    let plan = plan(&tensors, &memory(700), inputs(free, 256)).unwrap();
    assert_eq!(plan.expert_cache.slot_count, 50);
    assert_eq!(plan.expert_cache.slot_bytes, MIB);
    assert_eq!(
        (plan.expert_cache.total_experts, plan.expert_cache.minimum_slots),
        (200, 4)
    );
    assert_eq!(plan.device.slack, MIB / 2);
    assert_eq!(plan.device.fixed_resident, 2 * MIB);
    assert_eq!(plan.device.staging_ring, 0);
    assert_eq!(plan.staging, None);
    assert_eq!(plan.device.planned() + plan.device.slack, free);
    assert_eq!(plan.prefill_chunk_tokens, 512);
    for name in ["text/layers/0/router", "text/layers/0/norm"] {
        assert_eq!(plan.storage_of(name), Some(StorageClass::DeviceResident));
    }
    assert_eq!(plan.storage_of("text/token_embedding"), Some(StorageClass::HostMapped));
    assert_eq!(plan.storage_of("ngram/shard"), Some(StorageClass::HostStaged));
    assert_eq!(plan.storage_of("mtp/proj"), Some(StorageClass::Deferred));
    assert_eq!(plan.storage_of("vision/proj"), Some(StorageClass::Deferred));
    assert_eq!(
        plan.storage_of("text/layers/1/experts"),
        Some(StorageClass::DeviceCached)
    );
    assert_eq!(plan.storage_of("mtp/layers/0/experts"), Some(StorageClass::Deferred));
    assert_eq!(plan.storage_of("no/such/object"), None);
}

#[test]
fn the_cache_never_holds_more_than_one_slot_per_expert() {
    let free = reserved(&config(), 700, 256) + GIB;
    let plan = plan(&immovable(), &memory(700), inputs(free, 256)).unwrap();
    assert_eq!(plan.expert_cache.slot_count, 200);
    assert_eq!(plan.device.slack, GIB - 2 * MIB - 200 * MIB);
}

#[test]
fn gpu_reserve_is_kept_free_beside_the_runtime_reserve() {
    let free = reserved(&config(), 0, 256) + 2 * MIB + 90 * MIB;
    let small = plan(&immovable(), &memory(0), inputs(free, 256)).unwrap();
    let large = plan(&immovable(), &memory(64), inputs(free, 256)).unwrap();
    assert_eq!(small.device.runtime_reserve, KERNEL_RESERVE_BYTES);
    assert_eq!(large.device.runtime_reserve, KERNEL_RESERVE_BYTES + 64 * MIB);
    assert_eq!(small.expert_cache.slot_count, 90);
    assert_eq!(large.expert_cache.slot_count, 26);
}

#[test]
fn the_expert_staging_slots_follow_the_reads_in_flight_within_a_layer_and_two_tokens() {
    let config = config();
    // 64 experts per layer, 2 per token: never fewer than 4, never more than 64.
    assert_eq!(file_staging_slots(&config, 1), 4);
    assert_eq!(file_staging_slots(&config, 4), 4);
    assert_eq!(file_staging_slots(&config, 6), 6);
    assert_eq!(file_staging_slots(&config, 1000), 64);

    let free = reserved(&config, 700, 256) + GIB;
    let mut settings = memory(700);
    settings.weight_streaming.concurrent_reads = 6;
    let plan = plan(&immovable(), &settings, inputs(free, 256)).unwrap();
    assert_eq!(plan.expert_staging.slots, 6);
    assert_eq!(plan.expert_staging.slot_bytes, MIB);
    assert_eq!(plan.host.expert_staging_pinned, 6 * MIB);
    // The embedding, the staging slots and the n-gram tables, which the roomy host pins.
    assert_eq!(plan.host.pinned(), 10 * MIB + 6 * MIB + 50 * MIB);
}

#[test]
fn the_lane_count_is_the_gpu_streams_setting() {
    let free = reserved(&config(), 700, 256) + GIB;
    let mut settings = memory(700);
    settings.weight_streaming.gpu_streams = 5;
    assert_eq!(plan(&immovable(), &settings, inputs(free, 256)).unwrap().lanes, 5);
}

#[test]
fn zero_reads_zero_streams_and_a_bad_context_are_refused_naming_the_setting() {
    let free = reserved(&config(), 700, 256) + GIB;
    let refusal = |settings: MemorySettings, context: usize| match plan(&immovable(), &settings, inputs(free, context))
    {
        Err(PlanError::Setting(why)) => why,
        other => panic!("expected a refused setting, got {other:?}"),
    };
    let mut reads = memory(700);
    reads.weight_streaming.concurrent_reads = 0;
    assert!(refusal(reads, 256).contains("memory.weight-streaming.concurrent-reads is 0"));
    let mut streams = memory(700);
    streams.weight_streaming.gpu_streams = 0;
    assert!(refusal(streams, 256).contains("memory.weight-streaming.gpu-streams is 0"));
    assert!(refusal(memory(700), 0).contains("context.max-tokens is 0"));
    let why = refusal(memory(700), 4097);
    assert!(why.contains("4097") && why.contains("4096"), "{why}");
}

#[test]
fn a_reserve_that_takes_the_whole_budget_is_refused_with_both_figures() {
    let free = KERNEL_RESERVE_BYTES + 700 * MIB;
    let Err(PlanError::FlashReserveTooLarge {
        gpu_reserve,
        runtime_reserve,
        free: reported,
    }) = plan(&immovable(), &memory(700), inputs(free, 256))
    else {
        panic!("a reserve as large as the free memory leaves nothing");
    };
    assert_eq!(
        (gpu_reserve, runtime_reserve, reported),
        (700 * MIB, KERNEL_RESERVE_BYTES, free)
    );
    assert!(
        plan(&immovable(), &memory(700), inputs(free + 1, 256)).is_err(),
        "a byte over still leaves no cache"
    );
}

#[test]
fn the_smallest_cache_fits_exactly_and_one_byte_less_is_refused_with_the_bytes_needed() {
    let exact = reserved(&config(), 700, 256) + 2 * MIB + 4 * MIB;
    let fits = plan(&immovable(), &memory(700), inputs(exact, 256)).unwrap();
    assert_eq!(fits.expert_cache.slot_count, 4);
    assert_eq!(fits.device.slack, 0);

    let Err(PlanError::FlashDeviceTooSmall {
        context_tokens,
        free,
        minimum_slots,
        minimum_cache_bytes,
        needed_bytes,
        fixed_after_move_bytes,
        ring_bytes,
        runtime_reserve,
        ..
    }) = plan(&immovable(), &memory(700), inputs(exact - 1, 256))
    else {
        panic!("a cache one byte short of its minimum is refused");
    };
    assert_eq!((context_tokens, free), (256, exact - 1));
    assert_eq!((minimum_slots, minimum_cache_bytes), (4, 4 * MIB));
    assert_eq!((fixed_after_move_bytes, ring_bytes), (2 * MIB, 0));
    assert_eq!(runtime_reserve, KERNEL_RESERVE_BYTES + 700 * MIB);
    assert_eq!(needed_bytes, exact);
}

#[test]
fn the_refusal_message_names_the_needed_and_the_free_bytes() {
    let exact = reserved(&config(), 700, 256) + 2 * MIB + 4 * MIB;
    let message = plan(&immovable(), &memory(700), inputs(exact - 1, 256))
        .unwrap_err()
        .to_string();
    assert!(message.contains(&(exact - 1).to_string()), "{message}");
    assert!(message.contains(&exact.to_string()), "{message}");
    assert!(message.contains("4 slots"), "{message}");
}

#[test]
fn the_output_head_leaves_first_and_is_read_in_place() {
    let tensors = movable_world();
    // Everything resident needs 2 + 100 + 2 x 20 + 2 x 40 = 222 MiB and the smallest cache 4 more; the head's move
    // frees 100 MiB, which is enough.
    let free = reserved(&config(), 700, 256) + 200 * MIB;
    let plan = plan(&tensors, &memory(700), inputs(free, 256)).unwrap();
    assert_eq!(plan.storage_of("text/output_head"), Some(StorageClass::HostMapped));
    assert_eq!(
        plan.storage_of("text/layers/0/shared"),
        Some(StorageClass::DeviceResident)
    );
    assert_eq!(plan.staging, None);
    assert_eq!(plan.host.mapped, 10 * MIB + 100 * MIB);
    assert_eq!(plan.device.fixed_resident, 122 * MIB);
    assert_eq!(plan.expert_cache.slot_count, 78);
}

#[test]
fn staged_objects_pass_through_a_ring_of_the_largest_one_and_spread_across_the_layers() {
    let mut tensors = immovable();
    for layer in 0..16 {
        tensors.push(tensor(
            &format!("text/layers/{layer}/gdn/proj"),
            ComponentGroup::Gdn,
            40,
        ));
    }
    // 2 + 640 MiB resident. Staging n layers costs a ring of 4 x 40 MiB and saves n x 40, and the device may hold 481
    // MiB of them. The target grows by 64 MiB: 320 MiB (8 layers) leaves 482, 384 MiB takes 10 layers and leaves 402.
    let free = reserved(&config(), 700, 256) + 4 * MIB + 2 * MIB + 479 * MIB;
    let plan = plan(&tensors, &memory(700), inputs(free, 256)).unwrap();
    let staged: Vec<usize> = (0..16)
        .filter(|layer| plan.storage_of(&format!("text/layers/{layer}/gdn/proj")) == Some(StorageClass::HostStaged))
        .collect();
    assert_eq!(staged, [0, 1, 3, 4, 6, 8, 9, 11, 12, 14]);
    assert_eq!(
        plan.staging,
        Some(StagingGeometry {
            slots: 4,
            slot_bytes: 40 * MIB
        })
    );
    assert_eq!(plan.device.staging_ring, 160 * MIB);
    assert_eq!(plan.host.staged, 400 * MIB);
    assert_eq!(plan.device.fixed_resident, 2 * MIB + 240 * MIB);
    assert_eq!(plan.expert_cache.slot_count, 83);
}

#[test]
fn a_context_that_cannot_fit_even_with_everything_moved_is_refused_with_the_figures_after_the_move() {
    let tensors = movable_world();
    let none_left = reserved(&config(), 700, 256) + 2 * MIB + 4 * MIB;
    let Err(PlanError::FlashDeviceTooSmall {
        fixed_stay_bytes,
        fixed_after_move_bytes,
        ring_bytes,
        ..
    }) = plan(&tensors, &memory(700), inputs(none_left, 256))
    else {
        panic!("the staged ring of the projections is more than they save here");
    };
    assert_eq!(fixed_stay_bytes, 2 * MIB);
    assert_eq!(ring_bytes, 4 * 40 * MIB);
    assert_eq!(fixed_after_move_bytes, 2 * MIB + 4 * 40 * MIB);
}

#[test]
fn a_longer_prefill_chunk_takes_at_most_a_tenth_of_the_cache_budget() {
    let config = config();
    let sizes = SequenceState::new(&config);
    let extra = |chunk: u64| sizes.workspace_bytes(chunk).unwrap() - sizes.workspace_bytes(512).unwrap();
    let base = reserved(&config, 700, 4096) + 2 * MIB;
    let chunk_for = |budget: u64| {
        plan(&immovable(), &memory(700), inputs(base + budget, 4096))
            .unwrap()
            .prefill_chunk_tokens
    };
    // A budget whose tenth is below the first step's extra keeps 512; one that covers 1024 but not 2048 stops there.
    assert_eq!(chunk_for(extra(1024) * 10 - 1), 512);
    assert_eq!(chunk_for(extra(1024) * 10), 1024);
    assert_eq!(chunk_for(extra(2048) * 10 - 1), 1024);
    assert_eq!(chunk_for(extra(4096) * 10), 4096);
    // A short context never gets a chunk longer than twice its length.
    let short = plan(&immovable(), &memory(700), inputs(base + GIB, 512)).unwrap();
    assert_eq!(short.prefill_chunk_tokens, 512);
    let plan = plan(&immovable(), &memory(700), inputs(base + GIB, 4096)).unwrap();
    assert_eq!(
        plan.device.workspace,
        sizes.workspace_bytes(4096).unwrap() + sizes.shared_down_padding_bytes().unwrap()
    );
}

#[test]
fn selecting_the_mtp_layer_loads_it_and_its_bank_and_leaves_vision_deferred() {
    let free = reserved(&config(), 700, 256) + GIB;
    let text = plan(&immovable(), &memory(700), inputs(free, 256)).unwrap();
    assert_eq!(
        (text.host.deferred_mtp, text.host.deferred_vision),
        (40 * MIB, 20 * MIB)
    );
    let mut mtp = inputs(free, 256);
    mtp.mode = FlashMode {
        mtp: true,
        vision: false,
    };
    let plan = plan(&immovable(), &memory(700), mtp).unwrap();
    assert_eq!(plan.storage_of("mtp/proj"), Some(StorageClass::DeviceResident));
    assert_eq!(
        plan.storage_of("mtp/layers/0/experts"),
        Some(StorageClass::DeviceCached)
    );
    assert_eq!((plan.host.deferred_mtp, plan.host.deferred_vision), (0, 20 * MIB));
    assert_eq!(plan.expert_cache.total_experts, 210);
    assert_eq!(plan.device.fixed_resident, 2 * MIB + 30 * MIB);
}

#[test]
fn the_host_must_pin_what_the_plan_holds_there_and_a_byte_short_is_refused() {
    let free = reserved(&config(), 700, 256) + GIB;
    // The embedding (10 MiB) and 4 expert staging slots (4 MiB).
    let pinned = 14 * MIB;
    let at = |pinnable: u64| {
        let mut inputs = inputs(free, 256);
        inputs.host = HostBudget::stated(pinnable);
        plan(&immovable(), &memory(700), inputs)
    };
    assert!(at(pinned).is_ok());
    let Err(PlanError::FlashHostTooSmall {
        needed_bytes,
        expert_staging_bytes,
        pinnable_bytes,
    }) = at(pinned - 1)
    else {
        panic!("the host cannot pin the staging slots and the embedding");
    };
    assert_eq!(
        (needed_bytes, expert_staging_bytes, pinnable_bytes),
        (pinned, 4 * MIB, pinned - 1)
    );
}

#[test]
fn an_embedding_the_host_cannot_pin_stays_on_the_device() {
    let free = reserved(&config(), 700, 256) + GIB;
    let mut inputs = inputs(free, 256);
    inputs.host = HostBudget::stated(8 * MIB);
    let plan = plan(&immovable(), &memory(700), inputs).unwrap();
    assert_eq!(
        plan.storage_of("text/token_embedding"),
        Some(StorageClass::DeviceResident)
    );
    assert_eq!(plan.device.fixed_resident, 12 * MIB);
    assert_eq!(plan.host.mapped, 0);
}

#[test]
fn a_host_that_reports_no_figure_refuses_what_it_would_have_to_pin() {
    let free = reserved(&config(), 700, 256) + GIB;
    let mut inputs = inputs(free, 256);
    inputs.host = HostBudget::unknown();
    assert_eq!(
        plan(&immovable(), &memory(700), inputs).unwrap_err(),
        PlanError::HostBudgetUnknown { needed: 4 * MIB }
    );
}

#[test]
fn the_expert_records_are_held_in_ram_cached_or_read_from_the_file_by_what_the_host_has_left() {
    let free = reserved(&config(), 700, 256) + 50 * MIB + 2 * MIB;
    let with_resident = |resident: u64| {
        let mut inputs = inputs(free, 256);
        inputs.host = HostBudget {
            pinnable_bytes: GIB,
            resident_bytes: resident,
            source: BudgetSource::Explicit,
        };
        plan(&immovable(), &memory(700), inputs).unwrap()
    };
    // Pinned: the embedding and 4 staging slots, 14 MiB. The device cache holds 50 slots; records are 1 MiB.
    let all = with_resident(14 * MIB + 200 * MIB + 50 * MIB);
    assert_eq!(all.expert_store, ExpertStoreMode::RamResident);
    assert_eq!((all.host.expert_ram_slots, all.host.expert_ram), (200, 200 * MIB));
    assert_eq!(
        (all.ngram, all.host.ngram_pinned, all.host.ngram_file),
        (NgramMode::PinnedArena, 50 * MIB, 0)
    );
    let tight = with_resident(14 * MIB + 200 * MIB + 50 * MIB - 1);
    assert_eq!(tight.expert_store, ExpertStoreMode::RamResident);
    assert_eq!((tight.ngram, tight.host.ngram_file), (NgramMode::MappedFile, 50 * MIB));
    let cached = with_resident(14 * MIB + 120 * MIB);
    assert_eq!(
        (cached.expert_store, cached.host.expert_ram_slots),
        (ExpertStoreMode::RamCached, 120)
    );
    let file = with_resident(14 * MIB + 50 * MIB);
    assert_eq!(
        (file.expert_store, file.host.expert_ram_slots, file.host.expert_ram),
        (ExpertStoreMode::FileBacked, 0, 0)
    );
}

#[test]
fn the_consumers_are_what_the_plan_holds_on_each_side() {
    let free = reserved(&config(), 700, 256) + 150 * MIB;
    let plan = plan(&movable_world(), &memory(700), inputs(free, 256)).unwrap();
    let (mut device, mut host) = (0, 0);
    for asked in plan.consumers() {
        if asked.host {
            host += asked.bytes;
        } else {
            device += asked.bytes;
        }
    }
    assert_eq!(device, plan.device.planned());
    assert_eq!(host, plan.host.pinned());
}

#[test]
fn an_artifact_with_fewer_experts_than_the_smallest_cache_is_refused() {
    let few = vec![bank("text/layers/0/experts", ComponentGroup::RoutedExpert, 3, MIB)];
    let error = plan_flash(&config(), &immovable(), &few, &memory(700), inputs(8 * GIB, 256)).unwrap_err();
    assert!(
        matches!(error, PlanError::Unsupported(why) if why.contains("3 routed experts") && why.contains("4 slots"))
    );
    let none = plan_flash(&config(), &immovable(), &[], &memory(700), inputs(8 * GIB, 256)).unwrap_err();
    assert!(matches!(none, PlanError::Unsupported(why) if why.contains("0 routed experts")));
}
