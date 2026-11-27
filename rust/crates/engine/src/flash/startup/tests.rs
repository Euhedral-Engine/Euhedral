//! The contracts of the pieces of the startup that need no device: the refusals of a plan the start cannot serve, the
//! widened-weight lookup, the admission of a plan and the pool of the expert cache, and the owner of a start that never
//! got a lattice.

use super::*;

fn job(layer: u32, offset: u64) -> PadJob {
    let source = Weight {
        // SAFETY: the address is only compared, never dereferenced.
        address: unsafe { DevicePtr::from_raw(0x1000) },
        bytes: 64,
    };
    PadJob {
        layer,
        source,
        target_offset: offset,
        target_bytes: 96,
    }
}

#[test]
fn a_widened_weight_is_found_by_layer_at_its_offset() {
    // SAFETY: the address is only compared, never dereferenced.
    let base = unsafe { DevicePtr::from_raw(0x10_0000) };
    let padded = PaddedDown {
        jobs: vec![job(3, 0), job(7, 256)],
        base: Some(base),
    };
    let weight = padded.weight(7).expect("layer 7 has a widened weight");
    assert_eq!(weight.address.get(), base.get() + 256);
    assert_eq!(weight.bytes, 96);
    assert!(padded.weight(4).is_none(), "layer 4 has none");
}

#[test]
fn no_widened_region_means_no_widened_weight() {
    let padded = PaddedDown {
        jobs: Vec::new(),
        base: None,
    };
    assert!(padded.weight(0).is_none());
}

#[test]
fn an_idle_owner_runs_the_free_and_returns_its_result() {
    assert_eq!(RuntimeOwner::idle().shutdown_with(|| 41 + 1), 42);
}

use euhedral_inference_edrl::{
    ComponentGroup, ExpertBank, ExpertProjection, ExpertRecord, TensorDataType, WeightFormat, WeightLayout,
};

use crate::residency::accounting::MemoryAccount;
use crate::residency::flash::{DeviceBytes, ExpertCacheGeometry, ExpertStagingGeometry, HostBytes};

const RECORD: u64 = 4096;

fn plan_with(store: ExpertStoreMode, ngram: NgramMode, slots: usize, slot_bytes: u64) -> FlashPlan {
    FlashPlan {
        max_context_tokens: 1,
        mode: FlashMode::TEXT,
        objects: Vec::new(),
        banks: Vec::new(),
        device: DeviceBytes {
            free: 1 << 30,
            kv: 0,
            indexer: 0,
            gdn_state: 0,
            workspace: 0,
            runtime_reserve: 0,
            fixed_resident: 0,
            staging_ring: 0,
            expert_cache: slots as u64 * slot_bytes,
            slack: 0,
        },
        host: HostBytes {
            mapped: 0,
            staged: 0,
            expert_staging_pinned: 0,
            expert_ram: 0,
            expert_ram_slots: 0,
            expert_file: 0,
            ngram_pinned: 0,
            ngram_file: 0,
            deferred_mtp: 0,
            deferred_vision: 0,
        },
        expert_cache: ExpertCacheGeometry {
            slot_bytes,
            slot_count: slots,
            total_experts: 8,
            minimum_slots: 1,
        },
        staging: None,
        expert_staging: ExpertStagingGeometry {
            slots: 2,
            slot_bytes: RECORD,
        },
        expert_store: store,
        ngram,
        prefill_chunk_tokens: 1,
        lanes: 1,
    }
}

fn experts() -> ExpertLayout {
    let projection = ExpertProjection {
        name: "gate".to_owned(),
        shape: vec![1],
        dtype: TensorDataType::Bf16,
        format: WeightFormat::Nvfp4,
        layout: WeightLayout::RowSplitK128V1,
        record_offset: 0,
        bytes: 1,
    };
    let records = (0..4)
        .map(|index| ExpertRecord {
            offset: RECORD * (index + 1),
            bytes: RECORD,
            crc32: 0,
        })
        .collect();
    let bank = ExpertBank::new("experts", ComponentGroup::RoutedExpert, 0, vec![projection], records).expect("a bank");
    ExpertLayout::new(&[bank], RECORD * 5).expect("a layout")
}

#[test]
fn a_plan_that_holds_every_expert_record_in_memory_is_refused() {
    let held = plan_with(ExpertStoreMode::RamResident, NgramMode::MappedFile, 2, RECORD);
    let error = refuse_unserved(&held).expect_err("the records are not preloaded");
    assert!(error.to_string().contains("every expert record"), "{error}");
}

#[test]
fn a_plan_that_pins_the_ngram_tables_is_refused() {
    let pinned = plan_with(ExpertStoreMode::FileBacked, NgramMode::PinnedArena, 2, RECORD);
    let error = refuse_unserved(&pinned).expect_err("the tables are read from the file");
    assert!(error.to_string().contains("n-gram"), "{error}");
}

#[test]
fn a_plan_that_caches_or_reads_records_and_maps_the_tables_is_served() {
    for store in [ExpertStoreMode::RamCached, ExpertStoreMode::FileBacked] {
        refuse_unserved(&plan_with(store, NgramMode::MappedFile, 2, RECORD)).expect("served");
    }
}

#[test]
fn a_cache_sized_for_other_records_than_the_layouts_is_refused() {
    let layout = experts();
    let good = plan_with(ExpertStoreMode::FileBacked, NgramMode::MappedFile, 2, RECORD);
    check_cache(&good, &layout, 1, 1).expect("the slots hold the largest record");
    let small = plan_with(ExpertStoreMode::FileBacked, NgramMode::MappedFile, 2, RECORD / 2);
    assert!(check_cache(&small, &layout, 1, 1).is_err(), "device slots too small");
    let mut staged = good;
    staged.expert_staging.slot_bytes = RECORD * 2;
    assert!(
        check_cache(&staged, &layout, 1, 1).is_err(),
        "staging slots of another size"
    );
}

#[test]
fn a_read_setting_of_zero_is_refused_by_name() {
    let layout = experts();
    let good = plan_with(ExpertStoreMode::FileBacked, NgramMode::MappedFile, 2, RECORD);
    let split = check_cache(&good, &layout, 0, 1).expect_err("no parts");
    assert!(split.to_string().contains("read-split"), "{split}");
    let reads = check_cache(&good, &layout, 1, 0).expect_err("no reads");
    assert!(reads.to_string().contains("concurrent-reads"), "{reads}");
}

#[test]
fn the_expert_pool_is_leased_whole_and_the_account_closes_clean() {
    let account = MemoryAccount::open(1 << 30);
    let admitted = plan_with(ExpertStoreMode::FileBacked, NgramMode::MappedFile, 4, RECORD);
    let mut reservation = account.admit_flash(&admitted).expect("admitted");
    let pool = account
        .expert_pool(&mut reservation, 0x1000_0000, RECORD, 4)
        .expect("the plan admitted the pool");
    let slots = DeviceSlots::lease(&pool, account.holders(), 4).expect("every slot leased");
    drop((slots, pool, reservation));
    assert!(account.close().is_clean(), "no claim is left");
}

#[test]
fn a_pool_larger_than_the_plan_admitted_is_refused() {
    let account = MemoryAccount::open(1 << 30);
    let admitted = plan_with(ExpertStoreMode::FileBacked, NgramMode::MappedFile, 4, RECORD);
    let mut reservation = account.admit_flash(&admitted).expect("admitted");
    let error = account
        .expert_pool(&mut reservation, 0x1000_0000, RECORD, 5)
        .expect_err("a fifth slot was not admitted");
    assert!(error.to_string().contains("expert cache"), "{error}");
}

#[test]
fn a_pool_whose_size_overflows_is_refused() {
    let account = MemoryAccount::open(1 << 30);
    let admitted = plan_with(ExpertStoreMode::FileBacked, NgramMode::MappedFile, 4, RECORD);
    let mut reservation = account.admit_flash(&admitted).expect("admitted");
    let error = account
        .expert_pool(&mut reservation, 0, u64::MAX, 2)
        .expect_err("the size overflows");
    assert!(error.to_string().contains("overflow"), "{error}");
}
