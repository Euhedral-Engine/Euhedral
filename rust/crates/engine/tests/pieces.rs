//! The pieces the weight upload is cut into, the checks it makes on the way, and how its staging ring is sized.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]

mod common;

use common::{config, q3_table, rco_table, settings};
use euhedral_inference_edrl::WeightLayout;
use euhedral_inference_engine::residency::loader::{Check, Piece, Ring, Target, check_piece, pieces};
use euhedral_inference_engine::residency::offload::{EMBEDDING, HOST_ALIGNMENT, OUTPUT_HEAD};
use euhedral_inference_engine::residency::planner::{PlanInputs, ResidencyPlan, Tier, nvfp4, plan, plan_with};

const GIB: u64 = 1 << 30;
const MIB: u64 = 1 << 20;

fn rco_plan() -> ResidencyPlan {
    plan(&config(), &rco_table(), &settings(8192), 16 * GIB).unwrap()
}

fn of<'a>(pieces: &'a [Piece], plan: &ResidencyPlan, name: &str) -> Vec<&'a Piece> {
    let index = plan.device.iter().position(|placement| placement.name == name).unwrap();
    pieces.iter().filter(|piece| piece.placement == index).collect()
}

#[test]
fn pieces_fill_every_object_exactly_once_in_file_order() {
    let plan = rco_plan();
    let all = pieces(&plan, 8 * MIB);
    assert!(
        all.windows(2).all(|pair| pair[0].file_offset <= pair[1].file_offset),
        "file order"
    );
    assert!(all.iter().all(|piece| piece.len > 0 && piece.len as u64 <= 8 * MIB));
    // No two pieces write the same arena byte, and every piece lies inside its placement.
    let mut ranges: Vec<(u64, u64)> = all
        .iter()
        .map(|piece| (piece.arena_offset(), piece.arena_offset() + piece.len as u64))
        .collect();
    ranges.sort_unstable();
    assert!(
        ranges.windows(2).all(|pair| pair[0].1 <= pair[1].0),
        "arena ranges overlap"
    );
    for piece in &all {
        let placement = &plan.device[piece.placement];
        assert!(piece.arena_offset() >= placement.offset);
        assert!(piece.arena_offset() + piece.len as u64 <= placement.offset + placement.bytes);
    }
    // Every object but the packed projection's halves is copied whole, byte for byte: the same bytes at the same
    // distances from the object's start in the file and in the arena.
    for (index, placement) in plan.device.iter().enumerate() {
        if placement.name.starts_with("mtp/layer/attention/query_key") || placement.name.ends_with("gate_value") {
            continue;
        }
        let mine: Vec<&Piece> = all.iter().filter(|piece| piece.placement == index).collect();
        let covered: u64 = mine.iter().map(|piece| piece.len as u64).sum();
        assert_eq!(covered, placement.bytes, "{}", placement.name);
        for piece in mine {
            assert_eq!(
                piece.file_offset - placement.file_offset,
                piece.arena_offset() - placement.offset,
                "{}",
                placement.name
            );
        }
    }
}

#[test]
fn big_objects_are_cut_at_the_piece_size() {
    let plan = rco_plan();
    let all = pieces(&plan, 4 * MIB);
    let embedding = of(&all, &plan, "text/token_embedding");
    assert_eq!(embedding.iter().map(|piece| piece.len as u64).sum::<u64>(), 516_505_600);
    assert_eq!(embedding.len(), 124, "516505600 bytes in pieces of 4 MiB");
    assert!(
        embedding.iter().all(|piece| piece.check == Check::None),
        "a Q3 embedding has no check"
    );
    let small = of(&all, &plan, "text/layers/0/input_norm");
    assert_eq!(small.len(), 1);
    assert_eq!(small[0].len, 10_240);
}

#[test]
fn an_nvfp4_object_checks_its_scale_plane_and_its_trailer() {
    let plan = rco_plan();
    let all = pieces(&plan, MIB);
    let head = of(&all, &plan, "text/draft_head");
    let (rows, k) = (131_072, 5120);
    let row_scale = nvfp4::row_scale_bytes(k, plan.device[head[0].placement].layout).unwrap();
    let scales: Vec<&&Piece> = head
        .iter()
        .filter(|piece| matches!(piece.check, Check::Nvfp4Scales { .. }))
        .collect();
    assert_eq!(
        scales.iter().map(|piece| piece.len as u64).sum::<u64>(),
        rows * row_scale
    );
    // The checks know where in the plane they are, so a NaN is reported at its byte.
    let mut at = 0;
    for piece in &scales {
        assert_eq!(piece.check, Check::Nvfp4Scales { from: at });
        at += piece.len as u64;
    }
    let trailers: Vec<&&Piece> = head
        .iter()
        .filter(|piece| matches!(piece.check, Check::Nvfp4Trailer { .. }))
        .collect();
    assert_eq!(trailers.len(), 1);
    assert_eq!(trailers[0].len, 4);
    assert_eq!(trailers[0].check, Check::Nvfp4Trailer { sd4: false });
}

#[test]
fn the_packed_mtp_projection_is_read_as_its_two_halves() {
    let plan = rco_plan();
    let table = rco_table();
    let pack = "mtp/layer/attention/query_key_gate_value";
    let source = table.get(pack).unwrap();
    let all = pieces(&plan, 64 * MIB);
    let layout = source.layout();
    let (rows, k) = (14_336u64, 5120u64);
    let half = rows / 2;
    let code = nvfp4::row_code_bytes(k);
    let row_scale = nvfp4::row_scale_bytes(k, layout).unwrap();
    for (name, first) in [
        ("mtp/layer/attention/query_key", 0),
        ("mtp/layer/attention/gate_value", half),
    ] {
        let placement = plan.placement(name).unwrap();
        let mine = of(&all, &plan, name);
        assert_eq!(mine.len(), 3, "codes, scales and trailer");
        let by_arena = |offset: u64| {
            mine.iter()
                .find(|piece| piece.arena_offset() == placement.offset + offset)
                .unwrap()
        };
        // The codes of the half's rows: a straight cut of the pack's code plane.
        let codes = by_arena(0);
        assert_eq!(codes.file_offset, source.offset() + first * code);
        assert_eq!(codes.len as u64, half * code);
        // The scales of the half's rows land at the half's own scale plane.
        let scales = by_arena(nvfp4::scale_offset(half, k));
        assert_eq!(
            scales.file_offset,
            source.offset() + nvfp4::scale_offset(rows, k) + first * row_scale
        );
        assert_eq!(scales.len as u64, half * row_scale);
        assert_eq!(scales.check, Check::Nvfp4Scales { from: 0 });
        // Both halves share the pack's global scale.
        let trailer = by_arena(nvfp4::trailer_offset(half, k, layout).unwrap());
        assert_eq!(
            trailer.file_offset,
            source.offset() + nvfp4::trailer_offset(rows, k, layout).unwrap()
        );
        assert_eq!(trailer.len, 4);
    }
}

#[test]
fn a_nan_scale_or_a_bad_global_scale_is_refused_and_named() {
    // E4M3 NaN is 0x7F (and 0xFF).
    let mut scales = vec![0x38u8; 64];
    assert_eq!(check_piece(Check::Nvfp4Scales { from: 1000 }, &scales), Ok(()));
    scales[9] = 0x7F;
    assert_eq!(
        check_piece(Check::Nvfp4Scales { from: 1000 }, &scales).unwrap_err(),
        "NVFP4 block scale at byte 1009 is NaN"
    );
    scales[9] = 0xFF;
    assert!(check_piece(Check::Nvfp4Scales { from: 0 }, &scales).is_err());

    let plain = |value: f32| value.to_le_bytes().to_vec();
    assert_eq!(check_piece(Check::Nvfp4Trailer { sd4: false }, &plain(0.0125)), Ok(()));
    assert_eq!(check_piece(Check::Nvfp4Trailer { sd4: false }, &plain(0.0)), Ok(()));
    for bad in [f32::NAN, f32::INFINITY, -1.0] {
        let error = check_piece(Check::Nvfp4Trailer { sd4: false }, &plain(bad)).unwrap_err();
        assert!(error.contains("global scale"), "{error}");
    }
    // SD4: the table of sixteen scales comes first, and every entry may be selected.
    let mut sd4 = vec![0x30u8; 16];
    sd4.extend_from_slice(&plain(1.0));
    assert_eq!(check_piece(Check::Nvfp4Trailer { sd4: true }, &sd4), Ok(()));
    sd4[5] = 0x7F;
    assert_eq!(
        check_piece(Check::Nvfp4Trailer { sd4: true }, &sd4).unwrap_err(),
        "NVFP4 scale table entry 5 is NaN"
    );
    assert!(
        check_piece(Check::Nvfp4Trailer { sd4: true }, &plain(1.0)).is_err(),
        "a table is missing"
    );
    assert_eq!(check_piece(Check::None, &[0xFF; 8]), Ok(()));
}

#[test]
fn the_staging_ring_follows_the_pool_budget_and_the_lanes() {
    // Two loans per chain fill the budget: a piece is budget / (2 x depth), a power of two.
    let ring = Ring::sized(8, 256 * MIB, 64 * GIB, 32);
    assert_eq!(ring.depth, 8, "host.task-parallelism");
    assert_eq!(ring.piece_bytes, 16 * MIB);
    assert_eq!(ring.pinned_bytes(), 256 * MIB);
    assert_eq!(
        Ring::sized(8, 64 * MIB, 64 * GIB, 32).piece_bytes,
        4 * MIB,
        "the 64 MiB floor of the default pool"
    );
    assert_eq!(
        Ring::sized(8, 100 * MIB, 64 * GIB, 32).piece_bytes,
        4 * MIB,
        "rounded down to a power of two"
    );
    // The copy granularity bounds a piece: one AsyncIo read piece at least, 32 MiB at most.
    assert_eq!(Ring::sized(8, 16 * GIB, 64 * GIB, 32).piece_bytes, 32 * MIB);
    assert_eq!(Ring::sized(8, MIB, 64 * GIB, 32).piece_bytes, 256 * 1024);
    // The host budget bounds the ring too, and the lanes bound its depth.
    let tight = Ring::sized(8, 256 * MIB, 40 * MIB, 32);
    assert!(tight.pinned_bytes() <= 40 * MIB);
    assert_eq!(Ring::sized(64, 256 * MIB, 64 * GIB, 16).depth, 16);
    assert_eq!(Ring::sized(1, 256 * MIB, 64 * GIB, 16).depth, 2, "at least two chains");
}

/// The resident plan of the RCO table at context 4096, and the plan 256 MiB under its device need: the embedding alone
/// is read from host memory.
fn embedding_offloaded() -> ResidencyPlan {
    let inputs = |device_free| PlanInputs {
        device_free,
        multiprocessors: Some(70),
        host_pinned_bytes: Some(u64::MAX),
        sequences: 1,
    };
    let (config, table, settings) = (config(), rco_table(), settings(4096));
    let need = plan_with(&config, &table, &settings, inputs(64 * GIB))
        .unwrap()
        .device_bytes;
    let plan = plan_with(&config, &table, &settings, inputs(need - 256 * MIB)).unwrap();
    assert_eq!(
        plan.offload.host,
        [EMBEDDING],
        "the plan holds the embedding alone on the host"
    );
    assert!(plan.offload.disk.is_empty());
    plan
}

#[test]
fn host_pieces_land_at_the_plans_aligned_host_offsets_and_fill_the_arena() {
    let plan = embedding_offloaded();
    let all = pieces(&plan, 4 * MIB);
    let embedding = plan.placement(EMBEDDING).unwrap();
    let Tier::Host { offset } = embedding.tier else {
        panic!("the embedding is on the host");
    };
    assert_eq!(offset % HOST_ALIGNMENT, 0);
    let mine = of(&all, &plan, EMBEDDING);
    assert_eq!(mine.iter().map(|piece| piece.len as u64).sum::<u64>(), embedding.bytes);
    assert!(mine.iter().all(|piece| matches!(piece.target, Target::Host { .. })));
    // Each piece sits where its bytes are in the file, from the object's place in the host arena.
    for piece in &mine {
        assert_eq!(
            piece.arena_offset() - offset,
            piece.file_offset - embedding.file_offset,
            "the same distance from the object's start"
        );
    }
    let end = mine
        .iter()
        .map(|piece| piece.arena_offset() + piece.len as u64)
        .max()
        .unwrap();
    assert_eq!(end, offset + embedding.bytes);
    // The arena is what the plan allocates: the host objects, each rounded up to the alignment.
    assert_eq!(
        plan.offload.host_arena_bytes,
        embedding.bytes.div_ceil(HOST_ALIGNMENT) * HOST_ALIGNMENT
    );
    assert!(end <= plan.offload.host_arena_bytes);
    // Nothing else is read into the host arena, and the device objects are untouched.
    for piece in &all {
        let placement = &plan.device[piece.placement];
        match (placement.tier, piece.target) {
            (Tier::Host { .. }, Target::Host { .. }) | (Tier::Device, Target::Device { .. }) => {}
            other => panic!("{}: {other:?}", placement.name),
        }
    }
}

#[test]
fn a_host_nvfp4_piece_carries_the_checks_of_a_device_one() {
    let resident = rco_plan();
    // The head, moved to the host as the plan moves it when nothing else is left to offload.
    let mut moved = resident.clone();
    let index = moved
        .device
        .iter()
        .position(|placement| placement.name == OUTPUT_HEAD)
        .unwrap();
    moved.device[index].tier = Tier::Host { offset: 0 };
    moved.device[index].offset = 0;
    let kinds = |plan: &ResidencyPlan| -> Vec<(u64, usize, Check)> {
        pieces(plan, MIB)
            .iter()
            .filter(|piece| piece.placement == index)
            .map(|piece| (piece.file_offset, piece.len, piece.check))
            .collect()
    };
    let (on_device, on_host) = (kinds(&resident), kinds(&moved));
    assert_eq!(on_device, on_host, "the same pieces, with the same checks");
    assert!(
        on_host
            .iter()
            .any(|(_, _, check)| matches!(check, Check::Nvfp4Scales { .. }))
    );
    assert!(
        on_host
            .iter()
            .any(|(_, _, check)| matches!(check, Check::Nvfp4Trailer { .. }))
    );
    assert!(
        pieces(&moved, MIB)
            .iter()
            .filter(|piece| piece.placement == index)
            .all(|piece| matches!(piece.target, Target::Host { arena_offset } if arena_offset == piece.arena_offset()))
    );
}

#[test]
fn an_all_device_plans_pieces_are_all_device_pieces_at_the_arena_offsets() {
    let plan = rco_plan();
    assert!(plan.offload.is_empty());
    let all = pieces(&plan, 8 * MIB);
    assert!(all.iter().all(|piece| matches!(piece.target, Target::Device { .. })));
    for piece in &all {
        let placement = &plan.device[piece.placement];
        assert!(piece.arena_offset() >= placement.offset);
    }
}

#[test]
fn the_pieces_of_a_p2e2_tensor_carry_its_check_from_their_offset_in_it() {
    let plan = plan(&config(), &q3_table(true), &settings(8192), 16 * GIB).unwrap();
    let all = pieces(&plan, 4 * MIB);
    let mut checked = 0;
    for (index, placement) in plan.device.iter().enumerate() {
        let mine: Vec<&Piece> = all.iter().filter(|piece| piece.placement == index).collect();
        for piece in mine {
            match piece.check {
                Check::P2e2 { from, rows, k, bytes } => {
                    checked += 1;
                    assert_eq!(from, piece.file_offset - placement.file_offset, "{}", placement.name);
                    assert_eq!(bytes, placement.bytes, "{}", placement.name);
                    assert_eq!([rows, k].as_slice(), placement.shape.as_slice(), "{}", placement.name);
                }
                other => assert!(
                    !matches!(placement.layout, WeightLayout::RowSplitP2e2V1),
                    "{}: {other:?}",
                    placement.name
                ),
            }
        }
    }
    assert!(checked > 100, "{checked} pieces are checked");
}
