use euhedral_inference_edrl::TensorDataType;
use std::slice::from_ref;

use euhedral_inference_edrl::WeightFormat;

use super::*;
use crate::residency::flash::{
    BankPlacement, DeviceBytes, ExpertCacheGeometry, ExpertStagingGeometry, ExpertStoreMode, FlashMode, HostBytes,
    NgramMode, ObjectPlacement,
};
use crate::residency::loader::{Check, Target};
use crate::residency::planner::nvfp4;

fn tensor(name: &str, group: ComponentGroup, offset: u64, bytes: u64) -> FlashTensor {
    FlashTensor {
        name: name.to_owned(),
        shape: vec![bytes],
        dtype: TensorDataType::Bf16,
        format: WeightFormat::Bf16,
        layout: WeightLayout::ContiguousLeV1,
        group,
        offset,
        bytes,
        crc32: 0,
    }
}

fn nvfp4_tensor(name: &str, offset: u64, rows: u64, k: u64) -> FlashTensor {
    FlashTensor {
        shape: vec![rows, k],
        format: WeightFormat::Nvfp4,
        layout: WeightLayout::RowSplitK128V1,
        bytes: nvfp4::byte_size(rows, k, WeightLayout::RowSplitK128V1).unwrap(),
        ..tensor(name, ComponentGroup::FixedText, offset, 0)
    }
}

fn plan_of(tensors: &[FlashTensor], storage: &[StorageClass]) -> FlashPlan {
    let zero_device = DeviceBytes {
        free: 0,
        kv: 0,
        indexer: 0,
        gdn_state: 0,
        workspace: 0,
        runtime_reserve: 0,
        fixed_resident: 0,
        staging_ring: 0,
        expert_cache: 0,
        slack: 0,
    };
    let zero_host = HostBytes {
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
    };
    FlashPlan {
        max_context_tokens: 1,
        mode: FlashMode::TEXT,
        objects: tensors
            .iter()
            .zip(storage)
            .map(|(tensor, storage)| ObjectPlacement {
                name: tensor.name.clone(),
                group: tensor.group,
                bytes: tensor.bytes,
                storage: *storage,
            })
            .collect(),
        banks: Vec::<BankPlacement>::new(),
        device: zero_device,
        host: zero_host,
        expert_cache: ExpertCacheGeometry {
            slot_bytes: 1,
            slot_count: 1,
            total_experts: 1,
            minimum_slots: 1,
        },
        staging: None,
        expert_staging: ExpertStagingGeometry {
            slots: 1,
            slot_bytes: 1,
        },
        expert_store: ExpertStoreMode::FileBacked,
        ngram: NgramMode::MappedFile,
        prefill_chunk_tokens: 1,
        lanes: 1,
    }
}

fn three() -> (Vec<FlashTensor>, FlashPlan) {
    let tensors = vec![
        tensor("a", ComponentGroup::FixedText, 1000, 300),
        tensor("embedding", ComponentGroup::TokenEmbedding, 2000, 5000),
        tensor("table", ComponentGroup::Ngram, 9000, 700),
        tensor("deferred", ComponentGroup::Mtp, 20_000, 64),
        tensor("b", ComponentGroup::FixedText, 30_000, 10),
    ];
    let plan = plan_of(
        &tensors,
        &[
            StorageClass::DeviceResident,
            StorageClass::HostMapped,
            StorageClass::FileMapped,
            StorageClass::Deferred,
            StorageClass::DeviceResident,
        ],
    );
    (tensors, plan)
}

#[test]
fn objects_get_aligned_offsets_in_table_order_in_the_arena_of_their_home() {
    let (tensors, plan) = three();
    let layout = FixedLayout::of(&plan, &tensors).unwrap();
    let homes: Vec<_> = layout
        .placements()
        .iter()
        .map(|placement| placement.name.as_str())
        .zip(["a", "embedding", "table", "b"])
        .map(|(name, _)| (name, layout.home(name).unwrap()))
        .collect();
    assert_eq!(
        homes,
        [
            ("a", Home::Device { offset: 0 }),
            ("embedding", Home::Host { offset: 0 }),
            (
                "table",
                Home::File {
                    offset: 9000,
                    bytes: 700
                }
            ),
            ("b", Home::Device { offset: 512 }),
        ]
    );
    assert_eq!((layout.device_bytes(), layout.host_bytes()), (768, 8192));
}

#[test]
fn a_plan_that_stages_caches_or_pins_an_object_is_refused_naming_it() {
    for storage in [StorageClass::HostStaged, StorageClass::DeviceCached] {
        let (tensors, mut plan) = three();
        plan.objects[0].storage = storage;
        let error = FixedLayout::of(&plan, &tensors).unwrap_err().to_string();
        assert!(error.contains("a is placed as"), "{error}");
    }
    let (tensors, mut plan) = three();
    plan.objects[1].storage = StorageClass::FileMapped;
    assert!(
        FixedLayout::of(&plan, &tensors)
            .unwrap_err()
            .to_string()
            .contains("embedding")
    );
}

#[test]
fn a_plan_over_other_tensors_is_refused() {
    let (tensors, mut plan) = three();
    plan.objects[4].name = "c".to_owned();
    assert!(
        FixedLayout::of(&plan, &tensors)
            .unwrap_err()
            .to_string()
            .contains("object 4")
    );
    plan.objects.pop();
    assert!(
        FixedLayout::of(&plan, &tensors)
            .unwrap_err()
            .to_string()
            .contains("4 objects")
    );
}

#[test]
fn pieces_cover_each_loaded_object_once_cut_to_the_piece_size_and_skip_the_file_and_deferred_ones() {
    let (tensors, plan) = three();
    let layout = FixedLayout::of(&plan, &tensors).unwrap();
    let pieces = layout.pieces(200);
    let covered = |placement: usize| -> Vec<(u64, usize, Target)> {
        pieces
            .iter()
            .filter(|piece| piece.placement == placement)
            .map(|piece| (piece.file_offset, piece.len, piece.target))
            .collect()
    };
    assert_eq!(
        covered(0),
        [
            (1000, 200, Target::Device { arena_offset: 0 }),
            (1200, 100, Target::Device { arena_offset: 200 })
        ]
    );
    assert_eq!(covered(1).iter().map(|piece| piece.1).sum::<usize>(), 5000);
    assert_eq!(covered(1)[0].2, Target::Host { arena_offset: 0 });
    assert!(covered(2).is_empty(), "the n-gram table is read where it lies");
    assert_eq!(covered(3), [(30_000, 10, Target::Device { arena_offset: 512 })]);
    assert!(pieces.windows(2).all(|pair| pair[0].file_offset <= pair[1].file_offset));
}

#[test]
fn an_nvfp4_tensor_is_checked_where_its_format_has_something_to_check() {
    let weight = nvfp4_tensor("w", 4096, 128, 256);
    let plan = plan_of(from_ref(&weight), &[StorageClass::DeviceResident]);
    let layout = FixedLayout::of(&plan, from_ref(&weight)).unwrap();
    let pieces = layout.pieces(1 << 20);
    let scales = nvfp4::scale_offset(128, 256);
    let checks: Vec<_> = pieces
        .iter()
        .map(|piece| (piece.file_offset - 4096, piece.check))
        .collect();
    assert_eq!(checks[0], (0, Check::None));
    assert_eq!(checks[1], (scales, Check::Nvfp4Scales { from: 0 }));
    assert_eq!(checks.last().unwrap().1, Check::Nvfp4Trailer { sd4: false });
    assert_eq!(pieces.iter().map(|piece| piece.len as u64).sum::<u64>(), weight.bytes);
}

#[test]
fn a_cut_scale_plane_checks_each_piece_from_where_it_starts_in_the_plane() {
    let weight = nvfp4_tensor("w", 0, 128, 512);
    let plan = plan_of(from_ref(&weight), &[StorageClass::DeviceResident]);
    let layout = FixedLayout::of(&plan, from_ref(&weight)).unwrap();
    let pieces = layout.pieces(1024);
    let froms: Vec<u64> = pieces
        .iter()
        .filter_map(|piece| match piece.check {
            Check::Nvfp4Scales { from } => Some(from),
            _ => None,
        })
        .collect();
    assert_eq!(froms, [0, 1024, 2048, 3072]);
}

#[test]
fn objects_resolve_to_their_arena_addresses_and_a_host_object_needs_the_host_arena() {
    let (tensors, plan) = three();
    let layout = FixedLayout::of(&plan, &tensors).unwrap();
    // SAFETY: the address is only compared and offset, never dereferenced.
    let base = |address| unsafe { DevicePtr::from_raw(address) };
    let weights = layout.resolve(base(0x1000), Some(base(0x9000))).unwrap();
    assert_eq!(
        weights.weight("b").unwrap(),
        Weight {
            address: base(0x1200),
            bytes: 10
        }
    );
    assert_eq!(weights.weight("embedding").unwrap().address, base(0x9000));
    assert_eq!(weights.file_range("table").unwrap(), (9000, 700));
    assert!(weights.weight("table").unwrap_err().to_string().contains("table"));
    assert!(weights.weight("deferred").is_err());
    assert!(
        layout
            .resolve(base(0x1000), None)
            .unwrap_err()
            .to_string()
            .contains("embedding")
    );
}

fn padded_rig(width: u32, extra: Option<u64>) -> (FixedWeights, u32) {
    let hidden = 128u32;
    let names = [
        "text/layers/0/moe/shared_expert/down_proj",
        "text/layers/1/moe/shared_expert/down_proj",
    ];
    let tensors: Vec<FlashTensor> = names
        .iter()
        .enumerate()
        .map(|(index, name)| FlashTensor {
            bytes: if index == 1 {
                extra.unwrap_or(nvfp4_bytes(hidden, width))
            } else {
                nvfp4_bytes(hidden, width)
            },
            ..tensor(name, ComponentGroup::FixedText, 0, 0)
        })
        .collect();
    let plan = plan_of(&tensors, &[StorageClass::DeviceResident, StorageClass::DeviceResident]);
    let weights = FixedLayout::of(&plan, &tensors)
        .unwrap()
        // SAFETY: the address is only compared and offset, never dereferenced.
        .resolve(unsafe { DevicePtr::from_raw(0x10_0000) }, None)
        .unwrap();
    (weights, hidden)
}

#[test]
fn a_shared_width_that_is_not_a_multiple_of_256_widens_each_plain_tensor_into_its_own_slot() {
    let (weights, hidden) = padded_rig(128, None);
    let (jobs, bytes) = padding(&weights, 2, hidden, 128, 256);
    let target = nvfp4_bytes(hidden, 256);
    assert_eq!(jobs.len(), 2);
    assert_eq!(jobs[1].target_offset, target.next_multiple_of(256));
    assert_eq!(bytes, 2 * target.next_multiple_of(256));
    // SAFETY: the address is only compared and offset, never dereferenced.
    let base = unsafe { DevicePtr::from_raw(0x90_0000) };
    let launches = pad_launches(&jobs, base, hidden, 128, 256).unwrap();
    assert_eq!(launches.len(), 2);
}

#[test]
fn a_tensor_of_another_size_than_the_plain_layout_is_left_alone_and_a_wide_enough_width_needs_none() {
    let (weights, hidden) = padded_rig(128, Some(12_345));
    let (jobs, _) = padding(&weights, 2, hidden, 128, 256);
    assert_eq!(jobs.iter().map(|job| job.layer).collect::<Vec<_>>(), [0]);
    let (none, bytes) = padding(&weights, 2, hidden, 128, 128);
    assert!((none.is_empty(), bytes) == (true, 0));
}

#[test]
fn a_width_the_kernel_cannot_take_is_refused_when_the_launches_are_built() {
    let (weights, hidden) = padded_rig(128, None);
    let (jobs, _) = padding(&weights, 2, hidden, 128, 256);
    // SAFETY: the address is only compared and offset, never dereferenced.
    let base = unsafe { DevicePtr::from_raw(0x90_0000) };
    assert!(pad_launches(&jobs, base, hidden, 100, 256).is_err());
}
