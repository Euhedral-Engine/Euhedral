//! Host tests of the planners, and a fake device that checks what reaches a lane.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]
#![allow(
    clippy::disallowed_methods,
    clippy::disallowed_types,
    reason = "a test: the fake kernels record their launches behind a lock"
)]

use std::mem::replace;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use euhedral_gpu::fake::{FakeGpu, FakeModule};
use euhedral_gpu::{Args, Dim3, Lane, StreamOptions, open_shared_lanes};
use euhedral_graph::LaneId;

use super::*;
use crate::kernels::{Compiled, Module, ModuleOptions, compile};
use crate::training::plan_zero;

fn rate(bits: f64) -> Rate {
    Rate::new(bits).unwrap()
}

#[test]
fn unsupported_rates_are_rejected() {
    for bits in [1.0, 1.5, 4.5, 7.0, 8.0, 3.25, 2.75, f64::NAN] {
        assert!(
            matches!(Rate::new(bits), Err(TrellisError::UnsupportedRate(_))),
            "{bits}"
        );
    }
    for bits in [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0] {
        assert_eq!(Rate::new(bits).unwrap().bits(), bits);
    }
}

#[test]
fn a_packed_tile_has_sixteen_words_per_bit() {
    for (bits, words) in [
        (2.0, 32),
        (2.5, 40),
        (3.0, 48),
        (3.5, 56),
        (4.0, 64),
        (5.0, 80),
        (6.0, 96),
    ] {
        assert_eq!(rate(bits).words_per_tile(), words, "{bits}");
    }
}

#[test]
fn scratch_below_one_tile_is_rejected_and_above_it_counts_whole_tiles() {
    for bits in [2.0, 2.5, 3.0, 3.5, 4.0] {
        let rate = rate(bits);
        let tile = rate.scratch_bytes(1);
        assert!(matches!(
            rate.scratch_tiles(tile - 1),
            Err(TrellisError::ScratchTooSmall { .. })
        ));
        assert!(rate.scratch_tiles(0).is_err());
        assert_eq!(rate.scratch_tiles(tile).unwrap(), 1);
        assert_eq!(rate.scratch_tiles(3 * tile + 5).unwrap(), 3);
    }
}

#[test]
fn scratch_per_tile_follows_the_history_layout() {
    // One history byte per position and state, then two half-float cost arrays and 16-bit history.
    assert_eq!(rate(2.0).scratch_bytes(1), 256 * 16384);
    assert_eq!(rate(6.0).scratch_bytes(2), 2 * 256 * 1024);
    assert_eq!(rate(2.5).scratch_bytes(1), 4 * 16384 + 512 * 16384);
    assert_eq!(rate(3.5).scratch_bytes(3), 3 * (4 * 8192 + 512 * 8192));
}

#[test]
fn the_tensor_core_order_visits_every_element_of_a_tile_once() {
    let mut seen = [false; TILE_WEIGHTS];
    for element in TENSOR_CORE_ORDER {
        assert!(!replace(&mut seen[usize::from(element)], true));
    }
    assert_eq!(&TENSOR_CORE_ORDER[..8], &[0, 16, 128, 144, 8, 24, 136, 152][..]);
    assert_eq!(TENSOR_CORE_ORDER[8], 32);
}

fn dims(steps: &[Step]) -> Vec<(u32, u32, u32)> {
    steps
        .iter()
        .map(|step| (step.launch().grid.x, step.launch().grid.y, step.launch().block.x))
        .collect()
}

#[test]
fn viterbi_chunks_by_the_scratch_and_advances_each_pointer_by_the_chunk() {
    let steps = plan_viterbi(rate(4.0), 0x1000_0000, 100, 0x2000_0000, 0x3000_0000, 0x4000_0000, 16).unwrap();
    let sizes: Vec<u32> = steps.iter().map(|step| step.launch().grid.x).collect();
    assert_eq!(sizes, [16, 16, 16, 16, 16, 16, 4]);
    for (index, step) in steps.iter().enumerate() {
        let first = 16 * index as u64 * 256;
        assert_eq!(step.kernel(), "viterbi_k4");
        assert_eq!(step.args().read::<u64>(0), Some(0x1000_0000 + 4 * first));
        assert_eq!(step.args().read::<i32>(1), Some(step.launch().grid.x as i32));
        assert_eq!(step.args().read::<u64>(2), Some(0x2000_0000 + 2 * first));
        assert_eq!(step.args().read::<u64>(3), Some(0x3000_0000 + 4 * first));
        assert_eq!(step.args().read::<u64>(4), Some(0x4000_0000));
    }
}

#[test]
fn viterbi_takes_the_block_and_shared_memory_of_its_rate() {
    let at = |bits: f64| {
        let steps = plan_viterbi(rate(bits), 8, 1, 8, 8, 8, 1).unwrap();
        (
            steps[0].kernel(),
            steps[0].launch().block.x,
            steps[0].launch().shared_bytes,
        )
    };
    assert_eq!(at(2.0), ("viterbi_k2", 1024, 2 * 16384 * 2 + 512 + 64 + 128));
    assert_eq!(at(4.0), ("viterbi_k4", 1024, 2 * 4096 * 2 + 512 + 64 + 128));
    assert_eq!(at(5.0), ("viterbi_k5", 512, 2 * 2048 * 2 + 512 + 64 + 128));
    assert_eq!(at(2.5), ("viterbi_frac_k2", 512, 512 + 128 + 128));
    assert_eq!(at(3.5), ("viterbi_frac_k3", 512, 512 + 128 + 128));
}

#[test]
fn viterbi_refuses_what_it_cannot_run() {
    let plan = |count, scratch_tiles, scratch| plan_viterbi(rate(3.0), 8, count, 8, 8, scratch, scratch_tiles);
    assert!(matches!(plan(4, 0, 8), Err(TrellisError::NoScratchTiles)));
    assert!(matches!(plan(4, 2, 0), Err(TrellisError::NullAddress("scratch"))));
    assert!(matches!(plan(u32::MAX, 2, 8), Err(TrellisError::TooLarge { .. })));
    assert!(plan(0, 0, 8).unwrap().is_empty());
}

#[test]
fn the_half_step_rates_pack_a_block_of_tiles_per_launch() {
    assert_eq!(dims(&plan_pack(rate(3.0), 8, 300, 8).unwrap()), [(300, 1, 128)]);
    assert_eq!(dims(&plan_pack(rate(3.5), 8, 300, 8).unwrap()), [(3, 1, 128)]);
    assert_eq!(plan_pack(rate(3.5), 8, 300, 8).unwrap()[0].kernel(), "pack_frac_k3");
    assert_eq!(dims(&plan_decode(rate(2.5), 8, 300, 8).unwrap()), [(300, 1, 256)]);
    assert_eq!(plan_decode(rate(2.5), 8, 300, 8).unwrap()[0].kernel(), "decode_frac_k2");
}

#[test]
fn hadamard_launches_cover_every_block_and_refuse_a_partial_one() {
    let rows = plan_hadamard_rows(8, 256, 384).unwrap();
    assert_eq!(dims(&rows), [(12, 2, 256)]);
    let columns = plan_hadamard_columns(8, 256, 384).unwrap();
    assert_eq!(dims(&columns), [(3, 256, 128)]);
    assert!(matches!(
        plan_hadamard_rows(8, 100, 128),
        Err(TrellisError::NotMultiple { name: "rows", .. })
    ));
    assert!(matches!(
        plan_hadamard_columns(8, 128, 100),
        Err(TrellisError::NotMultiple { name: "cols", .. })
    ));
    assert!(plan_hadamard_rows(8, 0, 128).unwrap().is_empty());
}

#[test]
fn a_strip_is_sixteen_rows_inside_the_matrix_in_either_direction() {
    let forward = plan_to_tiles(8, 48, 64, 16, 16).unwrap();
    let back = plan_from_tiles(16, 48, 64, 16, 8).unwrap();
    assert_eq!(dims(&forward), [(4, 1, 256)]);
    assert_eq!(forward[0].args().read::<i32>(4), Some(0));
    assert_eq!(back[0].args().read::<i32>(4), Some(1));
    assert_eq!(forward[0].args().read::<u64>(0), Some(8));
    assert_eq!(back[0].args().read::<u64>(0), Some(8));
    assert_eq!(back[0].args().read::<u64>(3), Some(16));
    assert!(plan_to_tiles(8, 48, 64, 32, 16).is_ok());
    assert!(matches!(
        plan_to_tiles(8, 48, 64, 33, 16),
        Err(TrellisError::StripOutside { .. })
    ));
    assert!(matches!(
        plan_to_tiles(8, 48, 60, 0, 16),
        Err(TrellisError::NotMultiple { name: "n", .. })
    ));
    assert!(plan_to_tiles(8, 48, 64, u32::MAX, 16).is_err());
}

#[test]
fn a_reconstruction_decodes_rotates_scales_and_transposes_in_that_order() {
    let operands = Reconstruction {
        words: 0x10,
        suh: 0x20,
        svh: 0x30,
        k: 256,
        n: 512,
        out: 0x40,
        scratch: 0x50,
    };
    let steps = plan_reconstruct(rate(3.0), &operands).unwrap();
    let names: Vec<&str> = steps.iter().map(|step| step.kernel()).collect();
    assert_eq!(
        names,
        [
            "reconstruct_k3",
            "hadamard128_rows",
            "scale_rows_fp16",
            "hadamard128_columns",
            "scale_columns_fp16",
            "transpose_to_bf16"
        ]
    );
    assert_eq!(steps[0].launch().grid, Dim3::x(16 * 32));
    assert_eq!(steps[2].args().read::<u64>(1), Some(0x20));
    assert_eq!(steps[4].args().read::<u64>(1), Some(0x30));
    assert_eq!(steps[5].args().read::<u64>(3), Some(0x40));
    assert!(
        steps
            .iter()
            .skip(1)
            .all(|step| step.args().read::<u64>(0) == Some(0x50))
    );
    assert_eq!(dims(&steps[5..]), [(16, 8, 256)]);
    let half = plan_reconstruct(rate(3.5), &operands).unwrap();
    assert_eq!(half[0].kernel(), "reconstruct_frac_k3");
    let ragged = Reconstruction {
        k: 128,
        n: 100,
        ..operands
    };
    assert!(matches!(
        plan_reconstruct(rate(3.0), &ragged),
        Err(TrellisError::NotMultiple { name: "n", .. })
    ));
}

#[test]
fn the_reduction_helpers_size_their_launches_by_the_count() {
    let sum = |count| plan_sum_products(8, 16, count, 24).unwrap()[0].launch().grid.x;
    assert_eq!(sum(0), 1);
    assert_eq!(sum(256), 1);
    assert_eq!(sum(257), 2);
    assert_eq!(sum(1 << 40), 1024);
    let fill = |count| plan_fill_floats(8, count, 0.0).unwrap()[0].launch().grid.x;
    assert_eq!(fill(1 << 40), 1 << 20);
    assert_eq!(dims(&plan_tile_mean_square(8, 64, 128, 16).unwrap()), [(8, 4, 256)]);
    assert!(plan_tile_mean_square(8, 60, 128, 16).is_err());
    assert_eq!(dims(&plan_clean_row_scales(8, 16, 513, 24).unwrap()), [(3, 1, 256)]);
    assert!(plan_clean_row_scales(8, 16, 0, 24).unwrap().is_empty());
    assert!(plan_sum_products(8, 16, u64::MAX, 24).is_err());
}

#[test]
fn a_null_address_is_refused_before_anything_is_planned() {
    assert!(matches!(
        plan_decode(rate(3.0), 0, 4, 8),
        Err(TrellisError::NullAddress("words"))
    ));
    assert!(matches!(
        plan_clean_row_scales(8, 0, 4, 8),
        Err(TrellisError::NullAddress("info"))
    ));
}

/// A fake device whose trellis kernels record the order of their launches.
struct Rig {
    gpu: Arc<FakeGpu>,
    launched: Arc<Mutex<Vec<(String, Dim3, Args)>>>,
}

fn rig() -> Rig {
    let gpu = FakeGpu::new();
    let launched = Arc::new(Mutex::new(Vec::new()));
    for name in Module::Trellis.kernels() {
        let launched = Arc::clone(&launched);
        let name = (*name).to_owned();
        gpu.define(&name.clone(), move |_, launch, args| {
            launched.lock().unwrap().push((name.clone(), launch.grid, args.clone()));
            Ok(())
        });
    }
    Rig { gpu, launched }
}

fn run(rig: &Rig, kernels: &TrellisKernels<FakeModule>, steps: &[Step]) -> Result<(), TrellisError> {
    let lanes = open_shared_lanes(&rig.gpu, &[("test", StreamOptions::default())]).unwrap();
    let lane = Arc::clone(lanes.pool.lane(LaneId(0)));
    let mut refused = None;
    lane.submit(|queue| {
        // SAFETY: the fake kernels read no memory.
        refused = unsafe { kernels.enqueue(queue, steps) }.err();
        Ok(())
    })?;
    while rig.gpu.step() {}
    refused.map_or(Ok(()), Err)
}

fn trellis_kernels(rig: &Rig) -> (Arc<Compiled<FakeModule>>, TrellisKernels<FakeModule>) {
    let options = ModuleOptions::new(PathBuf::from("/cuda/include"));
    let compiled = Arc::new(compile::<FakeModule>(&rig.gpu, &options, Module::Trellis).unwrap());
    let kernels = TrellisKernels::new(&compiled).unwrap();
    (compiled, kernels)
}

#[test]
fn the_viterbi_kernels_that_need_it_are_opted_in_to_their_shared_memory() {
    let rig = rig();
    let (compiled, _kernels) = trellis_kernels(&rig);
    let limit = |name: &str| compiled.kernel(name).unwrap().max_dynamic_shared_bytes();
    assert!(limit("viterbi_k2") >= rate(2.0).viterbi_shared_bytes());
    assert!(limit("viterbi_k2") > 48 * 1024);
    for bits in [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0] {
        let rate = rate(bits);
        let frac = if bits.fract() == 0.0 { "" } else { "_frac" };
        let name = format!("viterbi{frac}_k{}", rate.lower());
        assert!(limit(&name) >= rate.viterbi_shared_bytes(), "{name}");
    }
}

#[test]
fn planned_steps_reach_the_lane_in_order_and_with_their_arguments() {
    let rig = rig();
    let (_compiled, kernels) = trellis_kernels(&rig);
    let mut steps = plan_viterbi(rate(2.0), 8, 5, 8, 8, 8, 2).unwrap();
    steps.extend(plan_pack(rate(2.0), 8, 5, 16).unwrap());
    run(&rig, &kernels, &steps).unwrap();
    let launched = rig.launched.lock().unwrap();
    let names: Vec<&str> = launched.iter().map(|(name, _, _)| name.as_str()).collect();
    assert_eq!(names, ["viterbi_k2", "viterbi_k2", "viterbi_k2", "pack_k2"]);
    assert_eq!(launched[2].1, Dim3::x(1));
    assert_eq!(launched[3].2.read::<u64>(2), Some(16));
}

#[test]
fn a_step_of_another_module_is_refused() {
    let rig = rig();
    let (_compiled, kernels) = trellis_kernels(&rig);
    // A good step first: the check of every step comes before the first launch, so nothing reaches the lane.
    let mut steps = plan_pack(rate(2.0), 8, 5, 16).unwrap();
    steps.extend(plan_zero(8, 4).unwrap());
    assert!(matches!(
        run(&rig, &kernels, &steps),
        Err(TrellisError::Step(StepError::WrongModule { .. }))
    ));
    assert!(rig.launched.lock().unwrap().is_empty());
}

#[test]
fn a_grid_taller_than_the_device_allows_is_refused_when_planned() {
    let too_tall = |result: Result<Vec<Step>, TrellisError>| matches!(result, Err(TrellisError::GridTooTall { .. }));
    assert!(too_tall(plan_hadamard_columns(8, 65536, 128)));
    assert!(plan_hadamard_columns(8, 65535, 128).is_ok());
    let reconstruction = |k| Reconstruction {
        words: 8,
        suh: 16,
        svh: 24,
        k,
        n: 128,
        out: 32,
        scratch: 40,
    };
    assert!(too_tall(plan_reconstruct(rate(3.0), &reconstruction(65536))));
    assert!(plan_reconstruct(rate(3.0), &reconstruction(65408)).is_ok());
    assert!(too_tall(plan_tile_mean_square(8, 16 * 65536, 16, 16)));
    assert!(plan_tile_mean_square(8, 16 * 65535, 16, 16).is_ok());
}
