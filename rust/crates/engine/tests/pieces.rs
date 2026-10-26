//! The pieces the weight upload is cut into, the checks it makes on the way, and how its staging ring is sized.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]

mod common;

use common::{config, rco_table, settings};
use euhedral_inference_engine::residency::loader::{Check, Piece, Ring, check_piece, pieces};
use euhedral_inference_engine::residency::planner::{ResidencyPlan, nvfp4, plan};
use euhedral_inference_settings::{Auto, ByteSize};

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
        .map(|piece| (piece.arena_offset, piece.arena_offset + piece.len as u64))
        .collect();
    ranges.sort_unstable();
    assert!(
        ranges.windows(2).all(|pair| pair[0].1 <= pair[1].0),
        "arena ranges overlap"
    );
    for piece in &all {
        let placement = &plan.device[piece.placement];
        assert!(piece.arena_offset >= placement.offset);
        assert!(piece.arena_offset + piece.len as u64 <= placement.offset + placement.bytes);
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
                piece.arena_offset - placement.offset,
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
                .find(|piece| piece.arena_offset == placement.offset + offset)
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
fn the_staging_ring_follows_the_settings_and_the_model() {
    let mut settings = settings(8192);
    // Auto: a piece is the largest object cut 64 ways, a power of two within 1 MiB to 32 MiB.
    let ring = Ring::sized(&settings, 715_161_604, 64 * GIB, 32);
    assert_eq!(ring.depth, 8, "host.task-parallelism");
    assert_eq!(ring.piece_bytes, 8 * MIB);
    assert_eq!(
        ring.pinned_bytes(),
        2 * 8 * 8 * MIB,
        "a read and a copy in flight per chain"
    );
    assert_eq!(Ring::sized(&settings, 100 * MIB, 64 * GIB, 32).piece_bytes, MIB);
    assert_eq!(Ring::sized(&settings, 100 * GIB, 64 * GIB, 32).piece_bytes, 32 * MIB);
    // A stated pool is the ring's: depth pieces fill it.
    settings.memory.pinned_buffer_pool = Auto::Value(ByteSize::mib(256));
    assert_eq!(Ring::sized(&settings, 715_161_604, 64 * GIB, 32).piece_bytes, 32 * MIB);
    settings.memory.pinned_buffer_pool = Auto::Value(ByteSize::mib(24));
    assert_eq!(
        Ring::sized(&settings, 715_161_604, 64 * GIB, 32).piece_bytes,
        2 * MIB,
        "24 MiB / 8, rounded down"
    );
    // The host budget bounds the ring, and the lanes bound its depth.
    settings.memory.pinned_buffer_pool = Auto::Auto;
    let tight = Ring::sized(&settings, 715_161_604, 40 * MIB, 32);
    assert!(tight.depth as u64 * tight.piece_bytes <= 40 * MIB);
    settings.host.task_parallelism = 64;
    assert_eq!(Ring::sized(&settings, 715_161_604, 64 * GIB, 16).depth, 16);
    settings.host.task_parallelism = 1;
    assert_eq!(
        Ring::sized(&settings, 715_161_604, 64 * GIB, 16).depth,
        2,
        "at least two chains"
    );
}
