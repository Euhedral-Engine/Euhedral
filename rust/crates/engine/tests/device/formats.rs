//! The linear formats on a GPU. The tests are `#[ignore]`d and run locally in a memory-capped scope:
//! `systemd-run --user --scope -p MemoryMax=16G cargo test -p euhedral-inference-engine --test device -- --ignored formats::`.
//!
//! The library comes from the `euhedral.cuda.library` property (`EUHEDRAL_TEST_CUDA_LIBRARY` in the environment, in `~/.config/euhedral/test.properties`, or
//! `gradle.properties`) and the real objects from `euhedral.qwen.rco-artifact`; a test skips with a message when one
//! is absent (a skipped test passes in a hundredth of a second: read the output with `--nocapture`).
//!
//! Trellis objects are synthetic here (random tile words, signs and output scales), as in Java's
//! `TrellisLinearCudaIntegrationTest`, and the NVFP4 and embedding objects are the RCO artifact's own. The scalar
//! reference is reached without the library's process-wide exact-numerics switch (which another test of this binary
//! would see): the trellis format runs it when built for it, and NVFP4 takes it above 32 rows of a device without the
//! native route. The switch itself is checked in `formats_oracle`, a binary of its own.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]
#![allow(
    clippy::print_stdout,
    reason = "a test reports the error it measured against the reference"
)]
#![allow(
    clippy::disallowed_methods,
    reason = "no-wait, waiting calls: a test copies device memory and synchronizes"
)]

use crate::common;
use crate::linear_gpu;

use euhedral_inference_cuda::launch::nvfp4::skinny_target_ctas;
use euhedral_inference_cuda::launch::q3::P2E2_EXPAND_LIMIT;
use euhedral_inference_cuda::{DeviceBuffer, HostWeightsArena};
use euhedral_inference_edrl::{TensorDataType, WeightFormat, WeightLayout};
use euhedral_inference_engine::formats::{
    LinearFormat, LinearKernel, LinearMode, Nvfp4Linear, Q3Linear, ROW_EXACT_KERNELS, ShapeRecorder, TokenEmbedding,
    TrellisLinear, nvfp4_native_scratch_bytes,
};
use euhedral_inference_engine::model::{Residence, WeightRef};
use euhedral_inference_engine::residency::offload::{MAPPED_HEAD_MAX_ROWS, OUTPUT_HEAD};
use euhedral_inference_engine::residency::planner::nvfp4;
use euhedral_inference_settings::TrellisKernelSettings;
use linear_gpu::{
    Artifact, Random, Rig, Trellis, activations, assert_close, bf16, bf16_value, compressed_head, relative_rms, rig,
    synthetic_trellis, upload, values,
};

const VERIFY: LinearMode = LinearMode {
    row_exact: true,
    prefill: false,
};
const PLAIN: LinearMode = LinearMode {
    row_exact: false,
    prefill: false,
};

fn run(
    rig: &Rig,
    format: &dyn LinearFormat,
    object: &Trellis,
    x: &DeviceBuffer,
    rows: usize,
    mode: LinearMode,
) -> Vec<f32> {
    values(&rig.linear(format, &object.weight, x, 0, object.k, object.n, rows, mode))
}

#[test]
#[ignore = "needs a GPU, the native library and the RCO artifact"]
fn linear_bitwise_matches_reference_oracle() {
    let _card = common::card();
    let Some(rig) = rig() else { return };
    let mut random = Random::new(7);
    // Mirrors Java's `everyRouteAgreesWithTheScalarReference`: the routes sum in other orders than the scalar
    // reference, so they agree with it within a tolerance, and agree with each other bit for bit where the kernels
    // are the same twins. Tensor-core rows up to 32, 140 target blocks, whole-K from 128 rows, wide blocks from 1024.
    let format = TrellisLinear::new(140, 32, 128, 1024);
    let oracle = TrellisLinear::new(140, 32, 128, 1024).with_exact_numerics();
    // n / 128 = 4 blocks: ceil(140 / 4) = 35 splits, capped at k / 256 = 4.
    let split = synthetic_trellis(&rig.library, &mut random, 512, 1024, 56, &[128, 512]);
    let x_bytes = activations(&mut random, 40, 1024);
    let x = upload(&rig.library, &x_bytes);
    let reference = run(&rig, &oracle, &split, &x, 40, PLAIN);
    let tensor_core = run(&rig, &format, &split, &x, 8, PLAIN);
    assert_close(&reference[..8 * 512], &tensor_core, 512, 0.02, "tensor-core, K split");
    // A verification of 40 rows, over the limit of 32, stays on the same route and equals one-row launches.
    let verify = run(&rig, &format, &split, &x, 40, VERIFY);
    assert_close(&reference, &verify, 512, 0.02, "verification");
    // From the whole-K rows each block runs every split itself: the same bits.
    let whole_k = TrellisLinear::new(140, 32, 16, 1024);
    assert_eq!(verify, run(&rig, &whole_k, &split, &x, 40, VERIFY), "whole-K rows");
    let wide = TrellisLinear::new(140, 32, 16, 16);
    assert_eq!(
        verify,
        run(&rig, &wide, &split, &x, 40, VERIFY),
        "64-row whole-K blocks"
    );
    // Whole-K launches that are not verifications run on block-scaled FP8 operands where the device has them: within
    // FP8's rounding of the reference, and each row independent of the others.
    assert!(
        rig.library
            .linear_kernel_available(LinearKernel::TrellisMmaRowsFp8.index())
            .unwrap()
    );
    for wide_rows in [1024usize, 16] {
        let fp8 = TrellisLinear::new(140, 64, 16, wide_rows);
        let blocks = run(&rig, &fp8, &split, &x, 40, PLAIN);
        let error = relative_rms(&reference, &blocks);
        println!("FP8 rows, wide rows {wide_rows}: relative rms {error:.4}");
        assert!(error < 0.05, "FP8 relative rms {error}");
        assert_close(&reference, &blocks, 512, 0.25, "FP8");
        assert_eq!(
            blocks[..20 * 512],
            run(&rig, &fp8, &split, &x, 20, PLAIN)[..],
            "FP8 rows are independent"
        );
    }
    for row in [0usize, 17, 39] {
        let one = values(&rig.linear(&format, &split.weight, &x, row, split.k, split.n, 1, PLAIN));
        assert_eq!(one[..], verify[row * 512..(row + 1) * 512], "row {row}");
    }
    // 40 rows, not a verification, above the limit: decode to BF16 and the row-exact GEMM.
    let prefill = run(&rig, &format, &split, &x, 40, PLAIN);
    println!("prefill route: relative rms {:.4}", relative_rms(&reference, &prefill));
    assert_close(&reference, &prefill, 512, 0.04, "prefill");

    // 140 blocks of 128: one split, the epilogue fused into the GEMM.
    let single = synthetic_trellis(&rig.library, &mut random, 17920, 1024, 56, &[128, 17920]);
    let x3 = upload(&rig.library, &activations(&mut random, 3, 1024));
    let reference = run(&rig, &oracle, &single, &x3, 3, PLAIN);
    assert_close(
        &reference,
        &run(&rig, &format, &single, &x3, 3, PLAIN),
        17920,
        0.02,
        "tensor-core",
    );

    // NVFP4, on the RCO artifact's own MTP projection (5120 outputs of 6144 inputs).
    let Some(mut artifact) = Artifact::open() else { return };
    let (_weights, weight) = artifact.plain(&rig.library, "mtp/layer/attention/output");
    let (n, k) = (5120usize, 6144usize);
    let x = upload(&rig.library, &activations(&mut random, 64, k));
    let nvfp4 = |format: &Nvfp4Linear, rows: usize, mode: LinearMode| {
        values(&rig.linear(format, &weight, &x, 0, k, n, rows, mode))
    };
    // Above 32 rows a device without the native route runs the scalar reference: the oracle.
    let portable = Nvfp4Linear::with_native_route(false, 70);
    let production = Nvfp4Linear::new(&rig.library);
    let reference = nvfp4(&portable, 64, PLAIN);
    for rows in 1..=8usize {
        let decode = nvfp4(&production, rows, PLAIN);
        let error = relative_rms(&reference[..rows * n], &decode);
        let differing = reference[..rows * n]
            .iter()
            .zip(&decode)
            .filter(|(a, b)| a != b)
            .count();
        println!(
            "NVFP4 decode rows {rows}: relative rms {error:.5}, {differing} of {} elements differ",
            rows * n
        );
        assert!(error < 0.01, "rows {rows}: {error}");
        // The decode kernels sum in another order than the scalar reference: a BF16 element now and then rounds the
        // other way.
        assert!(differing * 1000 < rows * n, "rows {rows}: {differing} elements differ");
        assert_close(&reference[..rows * n], &decode, n, 0.02, "NVFP4 decode");
    }
    for rows in [9usize, 20, 32] {
        let groups = nvfp4(&production, rows, VERIFY);
        let error = relative_rms(&reference[..rows * n], &groups);
        println!("NVFP4 grouped rows {rows}: relative rms {error:.5}");
        assert!(error < 0.01, "rows {rows}: {error}");
    }
    if rig.library.nvfp4_native_available() {
        // Two activation terms bring the native route's error per linear to about 1%.
        for rows in [9usize, 40, 64] {
            let native = nvfp4(&production, rows, PLAIN);
            let error = relative_rms(&reference[..rows * n], &native);
            println!("NVFP4 native rows {rows}: relative rms {error:.5}");
            assert!(error < 0.03, "rows {rows}: {error}");
        }
    }
}

/// A prefill above the skinny route's 64 rows launches the tile kernel with three tensor maps; run again, its maps come
/// from the thread's cache: the same bytes out, none encoded.
#[cfg(debug_assertions)]
#[test]
#[ignore = "needs a GPU, the native library and the RCO artifact"]
fn a_repeated_tile_prefill_encodes_no_tensor_map() {
    let _card = common::card();
    use euhedral_inference_cuda::launch::load::tensor_map_counts;
    let Some(rig) = rig() else { return };
    if !rig.library.nvfp4_native_available() {
        println!("skipped: no native NVFP4 route");
        return;
    }
    let Some(mut artifact) = Artifact::open() else { return };
    let (_weights, weight) = artifact.plain(&rig.library, "mtp/layer/attention/output");
    let (n, k, rows) = (5120usize, 6144usize, 128usize);
    let x = upload(&rig.library, &activations(&mut Random::new(11), rows, k));
    let production = Nvfp4Linear::new(&rig.library);
    let run = || values(&rig.linear(&production, &weight, &x, 0, k, n, rows, PLAIN));
    let first = run();
    let (asked, encoded) = tensor_map_counts();
    let second = run();
    let (asked_again, encoded_again) = tensor_map_counts();
    assert!(
        asked_again > asked,
        "the prefill launched a kernel that takes tensor maps"
    );
    assert_eq!(encoded_again, encoded, "the second run encoded no map");
    assert_eq!(first, second, "the same bytes out");
    assert!(encoded > 0, "the first run encoded its maps");
}

#[test]
#[ignore = "needs a GPU, the native library and the RCO artifact"]
fn row_twins_equal_single_rows() {
    let _card = common::card();
    let Some(rig) = rig() else { return };
    let mut random = Random::new(11);
    // A trellis object of a decode layer's size: 18 K splits at the default 8 blocks per multiprocessor of 70.
    let object = synthetic_trellis(&rig.library, &mut random, 4096, 5120, 40, &[4096]);
    let x = upload(&rig.library, &activations(&mut random, 8, 5120));
    let format = TrellisLinear::from_settings(&TrellisKernelSettings::default(), rig.library.device().sm_count);
    for mode in [PLAIN, VERIFY] {
        let ones: Vec<Vec<f32>> = (0..8)
            .map(|row| values(&rig.linear(&format, &object.weight, &x, row, 5120, 4096, 1, mode)))
            .collect();
        for rows in 2..=8usize {
            let together = run(&rig, &format, &object, &x, rows, mode);
            for (row, one) in ones.iter().take(rows).enumerate() {
                assert_eq!(
                    together[row * 4096..(row + 1) * 4096],
                    one[..],
                    "trellis, {rows} rows, row {row}, {mode:?}"
                );
            }
        }
    }

    let Some(mut artifact) = Artifact::open() else { return };
    let (_weights, weight) = artifact.plain(&rig.library, "mtp/layer/mlp/down");
    let (n, k) = (5120usize, 17408usize);
    let x = upload(&rig.library, &activations(&mut random, 32, k));
    let format = Nvfp4Linear::new(&rig.library);
    let ones: Vec<Vec<f32>> = (0..32)
        .map(|row| values(&rig.linear(&format, &weight, &x, row, k, n, 1, PLAIN)))
        .collect();
    // 2 to 8 rows run the decode twins (a verification and a plain launch alike); 9 to 32 rows of a verification the
    // grouped twins.
    for rows in (2..=8usize).chain([9, 16, 17, 24, 32]) {
        let mode = if rows > 8 { VERIFY } else { PLAIN };
        let together = values(&rig.linear(&format, &weight, &x, 0, k, n, rows, mode));
        for (row, one) in ones.iter().take(rows).enumerate() {
            assert_eq!(
                together[row * n..(row + 1) * n],
                one[..],
                "NVFP4, {rows} rows, row {row}"
            );
        }
    }
}

#[test]
#[ignore = "needs a GPU, the native library and the RCO artifact"]
fn nvfp4_launch_shapes_follow_the_route_of_the_shape() {
    let _card = common::card();
    let Some(rig) = rig() else { return };
    let Some(mut artifact) = Artifact::open() else { return };
    let mut random = Random::new(3);
    // The library queues these launches itself, so the geometry is what the route computes for the shape. The
    // expectations are read from native/src/host/nvfp4_linear.c.
    let (_weights, down) = artifact.plain(&rig.library, "mtp/layer/mlp/down");
    let nvfp4 = Nvfp4Linear::new(&rig.library).with_shape_recording();
    let x = upload(&rig.library, &activations(&mut random, 64, 17408));
    let record = |rows: usize, mode: LinearMode| {
        rig.linear(&nvfp4, &down, &x, 0, 17408, 5120, rows, mode);
        ShapeRecorder::take()
    };
    assert_eq!(
        record(1, PLAIN),
        ["euhedral_nvfp4_decode_rows1_w4 grid=160,1,1 block=128,1,1 smem=0"]
    );
    assert_eq!(
        record(5, PLAIN),
        ["euhedral_nvfp4_decode_rows5_w4 grid=160,1,1 block=128,1,1 smem=0"]
    );
    assert_eq!(
        record(20, VERIFY),
        ["euhedral_nvfp4_decode_groups3_w4 grid=160,1,1 block=128,1,1 smem=0"]
    );
    if rig.library.nvfp4_native_available() {
        // The K split aims at two blocks per multiprocessor of the device over the 80 column tiles of the 5120 rows
        // (`skinny_splits`), at most half the 68 K tiles of 256.
        let splits = skinny_target_ctas(rig.library.device().sm_count)
            .div_ceil(80)
            .clamp(1, 34);
        assert_eq!(
            record(12, PLAIN),
            [
                "euhedral_nvfp4n_quantize_rows grid=12,1,1 block=128,1,1 smem=0".to_string(),
                format!("euhedral_nvfp4n_skinny_16 grid=80,{splits},1 block=128,1,1 smem=61440"),
                "euhedral_nvfp4n_skinny_finish grid=240,1,1 block=256,1,1 smem=0".to_string(),
            ]
        );
    }
}

#[test]
#[ignore = "needs a GPU and the native library"]
fn kernel_tables_and_scratch_match_the_native_library() {
    let _card = common::card();
    let Some(rig) = rig() else { return };
    assert_eq!(rig.library.linear_kernel_count(), LinearKernel::ALL.len());
    for kernel in LinearKernel::ALL {
        assert_eq!(
            rig.library.linear_kernel_name(kernel.index()).as_deref(),
            Some(kernel.symbol())
        );
    }
    for (index, name) in ROW_EXACT_KERNELS {
        assert_eq!(rig.library.qwen4_kernel_name(index).as_deref(), Some(name));
    }
    // The native scratch the format sizes is the library's, for the shapes of the model.
    for (rows, k, n) in [
        (9u64, 5120u64, 5120u64),
        (16, 17408, 5120),
        (64, 6144, 5120),
        (65, 5120, 34816),
        (512, 17408, 5120),
        (512, 5120, 248_320),
    ] {
        assert_eq!(
            nvfp4_native_scratch_bytes(rows, k, n, skinny_target_ctas(rig.library.device().sm_count)),
            rig.library.nvfp4_native_scratch_bytes(rows as u32, k as u32, n as u32),
            "rows {rows} k {k} n {n}"
        );
    }
}

#[test]
#[ignore = "needs a GPU, the native library and the RCO artifact"]
fn q3_embedding_looks_up_token_rows() {
    let _card = common::card();
    let Some(rig) = rig() else { return };
    let Some(mut artifact) = Artifact::open() else { return };
    let (_table, weight) = artifact.plain(&rig.library, "text/token_embedding");
    assert_eq!(
        (weight.format, weight.layout),
        (WeightFormat::Q3G64Fp16, WeightLayout::RowSplitK128V1)
    );
    let (vocabulary, hidden) = (weight.shape[0] as usize, weight.shape[1] as usize);
    // Two real ids, a repeat, and one outside the vocabulary: its row is left as it was.
    let ids: [i32; 4] = [0, 151_643, 151_643, vocabulary as i32 + 5];
    let id_bytes: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
    let tokens = upload(&rig.library, &id_bytes);
    let sentinel = bf16(123.0).to_le_bytes();
    let output = upload(&rig.library, &sentinel.repeat(ids.len() * hidden));
    rig.run(|ops| TokenEmbedding.submit(ops, &weight, tokens.ptr(), output.ptr(), ids.len()));
    let mut host = vec![0u8; ids.len() * hidden * 2];
    output.copy_to_host(0, &mut host).unwrap();
    let rows: Vec<Vec<f32>> = host
        .chunks_exact(hidden * 2)
        .map(|row| {
            row.as_chunks::<2>()
                .0
                .iter()
                .map(|pair| bf16_value(u16::from_le_bytes(*pair)))
                .collect()
        })
        .collect();
    for row in &rows[..3] {
        assert!(
            row.iter()
                .all(|value| value.is_finite() && *value != 123.0 || *value == 0.0)
        );
        assert!(row.iter().any(|value| *value != 0.0));
    }
    assert_eq!(rows[1], rows[2], "the same id looks up the same row");
    assert_ne!(rows[0], rows[1], "different ids look up different rows");
    assert!(
        rows[3].iter().all(|value| *value == 123.0),
        "an id outside the vocabulary leaves its row"
    );
    // A weight that is not a Q3 table is refused before any launch.
    let mut refused = weight.clone();
    refused.format = WeightFormat::Nvfp4;
    let made = rig.typed.table.submit_ops(rig.typed.lane(0), false, |ops| {
        TokenEmbedding.submit(ops, &refused, tokens.ptr(), output.ptr(), 1)
    });
    rig.typed.sync();
    assert!(made.is_err());
}

/// The value of an E4M3 scale byte (finite codes only).
fn e4m3(code: u8) -> f32 {
    let (exponent, mantissa) = (u32::from(code >> 3) & 15, u32::from(code) & 7);
    let magnitude = if exponent == 0 {
        mantissa as f32 * 0.001_953_125
    } else {
        f32::from_bits(((exponent + 120) << 23) | (mantissa << 20))
    };
    if code & 0x80 != 0 { -magnitude } else { magnitude }
}

/// A synthetic plain NVFP4 token table gathers the rows of its ids as the host dequantization of them, bit for bit,
/// and an id outside the vocabulary leaves its row as it was.
#[test]
#[ignore = "needs a GPU and the native library"]
fn nvfp4_embedding_gathers_the_dequantized_rows() {
    let _light = common::light();
    let Some(rig) = rig() else { return };
    const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let (vocabulary, hidden) = (7usize, 256usize);
    let layout = WeightLayout::RowSplitK128V1;
    let (rows64, hidden64) = (vocabulary as u64, hidden as u64);
    let bytes = nvfp4::byte_size(rows64, hidden64, layout).unwrap() as usize;
    let (scale_at, global_at) = (
        nvfp4::scale_offset(rows64, hidden64) as usize,
        nvfp4::trailer_offset(rows64, hidden64, layout).unwrap() as usize,
    );
    let mut random = Random::new(0x004e_5646_5034);
    let mut table = vec![0u8; bytes];
    for byte in &mut table[..vocabulary * hidden / 2] {
        *byte = random.next_u64() as u8;
    }
    // Scales of the E4M3 values from 2^-6 to 2^3 in both signs: finite, and none of them the same row to row.
    for byte in &mut table[scale_at..scale_at + vocabulary * hidden / 16] {
        *byte = (0x20 + random.next_u64() % 0x30) as u8 | if random.next_u64() & 1 == 0 { 0 } else { 0x80 };
    }
    let global = 0.0375f32;
    table[global_at..global_at + 4].copy_from_slice(&global.to_le_bytes());
    let expected = |row: usize| -> Vec<u16> {
        (0..hidden)
            .map(|column| {
                let code = table[row * hidden / 2 + column / 2] >> (4 * (column % 2)) & 15;
                let magnitude = E2M1[usize::from(code & 7)];
                let value = if code & 8 != 0 { -magnitude } else { magnitude };
                let scale = e4m3(table[scale_at + row * hidden / 16 + column / 16]) * global;
                bf16(value * scale)
            })
            .collect()
    };
    let buffer = upload(&rig.library, &table);
    let weight = WeightRef {
        name: "text/token_embedding".to_string(),
        shape: vec![rows64, hidden64],
        dtype: TensorDataType::Bf16,
        format: WeightFormat::Nvfp4,
        layout,
        ptr: buffer.ptr(),
        bytes: bytes as u64,
        residence: Residence::Device,
        trellis: None,
    };
    // The first and the last row, a repeat, and one id outside the vocabulary and one below zero: those rows stay.
    let ids: [i32; 6] = [0, vocabulary as i32 - 1, 3, 3, vocabulary as i32 + 5, -1];
    let id_bytes: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
    let tokens = upload(&rig.library, &id_bytes);
    let sentinel = bf16(123.0);
    let output = upload(&rig.library, &sentinel.to_le_bytes().repeat(ids.len() * hidden));
    rig.run(|ops| TokenEmbedding.submit(ops, &weight, tokens.ptr(), output.ptr(), ids.len()));
    let mut host = vec![0u8; ids.len() * hidden * 2];
    output.copy_to_host(0, &mut host).unwrap();
    let got: Vec<Vec<u16>> = host
        .chunks_exact(hidden * 2)
        .map(|row| {
            row.as_chunks::<2>()
                .0
                .iter()
                .map(|pair| u16::from_le_bytes(*pair))
                .collect()
        })
        .collect();
    for (index, id) in ids.iter().enumerate() {
        if (0..vocabulary as i32).contains(id) {
            assert_eq!(got[index], expected(*id as usize), "output row {index}, id {id}");
        } else {
            assert!(
                got[index].iter().all(|bits| *bits == sentinel),
                "output row {index}, id {id}"
            );
        }
    }
    assert_ne!(got[2], got[1], "different ids gather different rows");
}

#[test]
#[ignore = "needs a GPU, the native library and the RCO artifact"]
fn real_trellis_objects_agree_with_the_scalar_reference() {
    let _card = common::card();
    let Some(rig) = rig() else { return };
    let Some(mut artifact) = Artifact::open() else { return };
    let sm_count = rig.library.device().sm_count;
    let settings = TrellisKernelSettings::default();
    let format = TrellisLinear::from_settings(&settings, sm_count);
    let oracle = TrellisLinear::from_settings(&settings, sm_count).with_exact_numerics();
    let mut random = Random::new(9);
    // The first layer's down projection (one segment, 17408 inputs) and gate/up projection (two segments).
    for name in [
        "text/layers/0/mlp/down",
        "text/layers/0/mlp/gate_up",
        "text/layers/3/attention/query_key",
    ] {
        let object = artifact.trellis(&rig.library, name);
        let x = upload(&rig.library, &activations(&mut random, 64, object.k));
        let reference = run(&rig, &oracle, &object, &x, 64, PLAIN);
        // Decode, two rows, a verification, and 64 rows (whole-K blocks, on FP8 operands where the device has them).
        for (rows, mode, allowed) in [
            (1usize, PLAIN, 0.02),
            (2, PLAIN, 0.02),
            (8, VERIFY, 0.02),
            (64, VERIFY, 0.02),
        ] {
            let actual = run(&rig, &format, &object, &x, rows, mode);
            assert_close(
                &reference[..rows * object.n],
                &actual,
                object.n,
                allowed,
                &format!("{name} rows {rows}"),
            );
        }
        let blocks = run(&rig, &format, &object, &x, 64, PLAIN);
        let error = relative_rms(&reference, &blocks);
        println!("{name}: 64 rows relative rms {error:.4}");
        assert!(error < 0.06, "{name}: {error}");
    }
}

/// The NVFP4 output head computes the same logits from a copy the kernels read in place over the bus as from the copy
/// on the device, at every row count the plan lets a mapped head run: one to eight rows (the decode kernels), the rows
/// of a verification in groups, and the batched-pass bound ([`MAPPED_HEAD_MAX_ROWS`]). The comparison is bit for bit.
#[test]
#[ignore = "needs a GPU, the native library and the RCO artifact"]
fn a_mapped_head_computes_the_device_heads_logits() {
    let _card = common::card();
    let Some(rig) = rig() else { return };
    let Some(mut artifact) = Artifact::open() else { return };
    let entry = artifact.table().get(OUTPUT_HEAD).unwrap();
    assert_eq!(entry.format(), WeightFormat::Nvfp4, "the head is NVFP4");
    let (n, k) = (entry.shape()[0] as usize, entry.shape()[1] as usize);
    let payload = artifact.bytes(OUTPUT_HEAD);
    let (_device_copy, device) = artifact.plain(&rig.library, OUTPUT_HEAD);

    // The mapped copy: the same bytes in a pinned host arena, addressed through the device.
    let mut arena = HostWeightsArena::alloc(&rig.library, payload.len()).unwrap();
    arena.as_mut_slice().copy_from_slice(&payload);
    drop(payload);
    // SAFETY: the arena outlives every launch below, and nothing writes it after the copy above.
    let ptr = unsafe { arena.device_address() }.unwrap();
    let mapped = WeightRef {
        ptr,
        residence: Residence::Mapped,
        ..device.clone()
    };
    assert_ne!(mapped.ptr, device.ptr);

    let mut random = Random::new(0x4EAD);
    let x = upload(
        &rig.library,
        &activations(&mut random, MAPPED_HEAD_MAX_ROWS as usize, k),
    );
    let format = Nvfp4Linear::new(&rig.library);
    let mut compared = 0;
    for (rows, mode) in (1..=8usize).flat_map(|rows| [(rows, PLAIN), (rows, VERIFY)]).chain(
        [9usize, 12, 16, 20, 32]
            .into_iter()
            .flat_map(|rows| [(rows, PLAIN), (rows, VERIFY)]),
    ) {
        let on_device = rig.linear(&format, &device, &x, 0, k, n, rows, mode);
        let in_place = rig.linear(&format, &mapped, &x, 0, k, n, rows, mode);
        assert_eq!(on_device.len(), rows * n);
        assert!(
            on_device == in_place,
            "rows {rows}, row-exact {}: the logits differ",
            mode.row_exact
        );
        compared += 1;
    }
    println!("{compared} launches of the {n} x {k} head, mapped and on the device, bit-identical");
    drop(arena);
}

/// A compressed Q3 tensor too large to expand whole is run in chunks of output rows, and every output equals the one
/// of the same tensor expanded whole and run as a plain Q3 tensor (the FP8 route's split of K follows the whole
/// tensor, the last chunk is short, and each chunk lands at its own columns).
#[test]
#[ignore = "needs a GPU and the native library"]
fn a_compressed_tensor_run_in_chunks_equals_its_whole_expansion() {
    let _card = common::card();
    let Some(rig) = rig() else { return };
    let (out, k) = (70_016usize, 5120usize);
    let mut random = Random::new(29);
    let head = compressed_head(&rig, &mut random, out, k);
    let expanded = head.expanded;
    assert!(expanded > P2E2_EXPAND_LIMIT, "{expanded} bytes must need chunks");
    let (chunked, plain) = (&head.compressed, &head.expanded_weight);
    let format = Q3Linear::new(&rig.library);
    for rows in [9usize, 40] {
        let x = upload(&rig.library, &activations(&mut random, rows, k));
        let wanted = rig.linear(&format, plain, &x, 0, k, out, rows, PLAIN);
        let actual = rig.linear(&format, chunked, &x, 0, k, out, rows, PLAIN);
        let scratch = format.scratch_bytes(chunked, rows, PLAIN);
        assert!(
            scratch < expanded as usize && scratch as u64 <= P2E2_EXPAND_LIMIT,
            "{rows} rows: the chunks are bounded: {scratch} bytes of scratch, {expanded} expanded, limit {P2E2_EXPAND_LIMIT}"
        );
        assert!(
            wanted.iter().any(|bits| *bits != wanted[0]),
            "{rows} rows: the outputs vary"
        );
        let different = wanted.iter().zip(&actual).filter(|(a, b)| a != b).count();
        assert_eq!(
            different,
            0,
            "{rows} rows: {different} of {} outputs differ",
            wanted.len()
        );
    }
}
