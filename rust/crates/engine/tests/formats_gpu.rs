//! The linear formats on a GPU. The tests are `#[ignore]`d and run locally in a memory-capped scope:
//! `systemd-run --user --scope -p MemoryMax=16G cargo test -p euhedral-inference-engine --test formats_gpu -- --ignored`.
//!
//! The library comes from the `euhedral.cuda.library` property (`EUHEDRAL_TEST_CUDA_LIBRARY` in the environment, or
//! `gradle.properties`) and the real objects from `euhedral.qwen.rco-artifact`; a test skips with a message when one
//! is absent (a skipped test passes in a hundredth of a second: read the output with `--nocapture`).
//!
//! Trellis objects are synthetic here (random tile words, signs and output scales), as in Java's
//! `TrellisLinearCudaIntegrationTest`, and the NVFP4 and embedding objects are the RCO artifact's own. The scalar
//! reference is reached without the library's process-wide exact-numerics switch (which another test of this binary
//! would see): the trellis format runs it when built for it, and NVFP4 takes it above 32 rows of a device without the
//! native route. The switch itself is checked in `formats_oracle_gpu`, a binary of its own.

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

#[path = "common/linear_gpu.rs"]
mod linear_gpu;

use std::collections::HashMap;

use euhedral_inference_cuda::{DeviceBuffer, Submission, SubmitMode};
use euhedral_inference_edrl::{WeightFormat, WeightLayout};
use euhedral_inference_engine::formats::{
    LinearFormat, LinearKernel, LinearMode, Nvfp4Linear, Q3Embedding, ROW_EXACT_KERNELS, ShapeRecorder, TrellisLinear,
    nvfp4_native_scratch_bytes,
};
use euhedral_inference_settings::TrellisKernelSettings;
use linear_gpu::{
    Artifact, Random, Rig, Trellis, activations, assert_close, bf16, bf16_value, relative_rms, rig, synthetic_trellis,
    upload, values,
};

/// The target blocks the fixture was recorded with.
const FIXTURE_TARGET_BLOCKS: usize = 560;

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
    let portable = Nvfp4Linear::with_native_route(false);
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

#[test]
#[ignore = "needs a GPU, the native library and the RCO artifact"]
fn row_twins_equal_single_rows() {
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

/// One case of the fixture Java wrote: its key, and the launches it made without their pointers.
fn java_launches() -> HashMap<String, Vec<String>> {
    let mut cases: HashMap<String, Vec<String>> = HashMap::new();
    let mut key = String::new();
    for line in include_str!("fixtures/linear_launches.txt").lines() {
        if let Some(header) = line.strip_prefix("case ") {
            // The name only labels the case; the shape keys it.
            key = header.split_once(' ').unwrap().1.to_string();
            cases.insert(key.clone(), Vec::new());
        } else if !line.starts_with("scratch=") {
            cases.get_mut(&key).unwrap().push(without_pointers(line));
        }
    }
    cases
}

/// A launch line with its pointer arguments blanked: the addresses differ between the fixture and a real device.
fn without_pointers(line: &str) -> String {
    let (head, args) = line.split_once(" args=").unwrap();
    let args: Vec<&str> = args
        .split(',')
        .map(|argument| if argument.starts_with('p') { "p" } else { argument })
        .collect();
    format!("{head} args={}", args.join(","))
}

fn java_key(weight: &Trellis, rows: usize, row_exact: bool, fp8: bool, target: usize) -> String {
    let ends = weight.weight.trellis.as_ref().unwrap().segment_ends.clone();
    format!(
        "n={} k={} w={} ends={:?} rows={rows} rowExact={row_exact} fp8={fp8} exact=false target={target} mma=512 \
         wholeK=64 wide=128",
        weight.n, weight.k, weight.weight.shape[2], ends,
    )
    .replace(", ", ",")
}

#[test]
#[ignore = "needs a GPU, the native library and the RCO artifact"]
fn launch_shapes_match_java_goldens() {
    let Some(rig) = rig() else { return };
    let Some(mut artifact) = Artifact::open() else { return };
    let java = java_launches();
    let fp8 = rig
        .library
        .linear_kernel_available(LinearKernel::TrellisMmaRowsFp8.index())
        .unwrap();
    // The fixture's thresholds, whatever device runs this: 560 target blocks (8 per multiprocessor of 70) and the
    // defaults of the others.
    let format = TrellisLinear::new(FIXTURE_TARGET_BLOCKS, 512, 64, 128).with_shape_recording();
    let mut random = Random::new(3);
    let x = upload(&rig.library, &activations(&mut random, 512, 17408));
    ShapeRecorder::take();
    let mut compared = 0;
    // One decode layer's linears (the first GDN layer) and a full-attention layer's (the fourth): decode, a two-row
    // twin, a four-draft verification (5 rows) and a seven-draft one (8), a batch of 6 and of 20, a batch over the
    // whole-K threshold, and a prefill chunk.
    let layers = [
        (
            0,
            ["gdn/query_key", "gdn/value_z", "gdn/output", "mlp/gate_up", "mlp/down"].as_slice(),
        ),
        (
            3,
            ["attention/query_key", "attention/gate_value", "attention/output"].as_slice(),
        ),
    ];
    let launches = [
        (1usize, false),
        (2, false),
        (5, true),
        (6, false),
        (8, true),
        (20, false),
        (64, false),
        (512, false),
    ];
    for (tensor, layer) in layers
        .iter()
        .flat_map(|(layer, tensors)| tensors.iter().map(move |t| (*t, *layer)))
    {
        let object = artifact.trellis(&rig.library, &format!("text/layers/{layer}/{tensor}"));
        for (rows, row_exact) in launches {
            let mode = LinearMode {
                row_exact,
                prefill: false,
            };
            rig.linear(&format, &object.weight, &x, 0, object.k, object.n, rows, mode);
            let recorded: Vec<String> = ShapeRecorder::take()
                .iter()
                .map(|line| without_pointers(line))
                .collect();
            assert!(!recorded.is_empty(), "{tensor} recorded nothing");
            // A verification is recorded once, whatever the device: it never takes the FP8 kernels.
            let key = java_key(&object, rows, row_exact, row_exact || fp8, FIXTURE_TARGET_BLOCKS);
            let expected = java
                .get(&key)
                .unwrap_or_else(|| panic!("the fixture has no case {key}"));
            assert_eq!(&recorded, expected, "{tensor} rows {rows} rowExact {row_exact}");
            compared += 1;
        }
    }
    assert!(compared >= 60, "only {compared} launches were compared with Java's");

    // NVFP4: the library queues these launches itself, so the geometry is what the route computes for the shape.
    // The expectations are read from native/src/host/nvfp4_linear.c.
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
        assert_eq!(
            record(12, PLAIN),
            [
                "euhedral_nvfp4n_quantize_rows grid=12,1,1 block=128,1,1 smem=0",
                "euhedral_nvfp4n_skinny_16 grid=80,2,1 block=128,1,1 smem=61440",
                "euhedral_nvfp4n_skinny_finish grid=240,1,1 block=256,1,1 smem=0",
            ]
        );
    }
}

#[test]
#[ignore = "needs a GPU and the native library"]
fn kernel_tables_and_scratch_match_the_native_library() {
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
            nvfp4_native_scratch_bytes(rows, k, n),
            rig.library.nvfp4_native_scratch_bytes(rows as u32, k as u32, n as u32),
            "rows {rows} k {k} n {n}"
        );
    }
}

#[test]
#[ignore = "needs a GPU, the native library and the RCO artifact"]
fn q3_embedding_looks_up_token_rows() {
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
    let submission = Submission::begin(&rig.lane, SubmitMode::default()).unwrap();
    Q3Embedding
        .submit(&submission, &weight, tokens.ptr(), output.ptr(), ids.len())
        .unwrap();
    submission.finish().unwrap();
    rig.lane.synchronize().unwrap();
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
    let submission = Submission::begin(&rig.lane, SubmitMode::default()).unwrap();
    assert!(
        Q3Embedding
            .submit(&submission, &refused, tokens.ptr(), output.ptr(), 1)
            .is_err()
    );
}

#[test]
#[ignore = "needs a GPU, the native library and the RCO artifact"]
fn real_trellis_objects_agree_with_the_scalar_reference() {
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
