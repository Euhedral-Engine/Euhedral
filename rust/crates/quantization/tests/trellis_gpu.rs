//! Device tests of the trellis kernels. `#[ignore]`d like the crate's other device tests:
//! `cargo test -p euhedral-inference-quantization --test trellis_gpu -- --ignored --nocapture`.
//!
//! The engine library comes from the `euhedral.cuda.library` property (`EUHEDRAL_TEST_CUDA_LIBRARY`); the bundled
//! NVRTC and the CUDA headers are beside it, and the builtins library of NVRTC must be on `LD_LIBRARY_PATH`.
#![cfg(target_os = "linux")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]

mod support;

use euhedral_inference_quantization::trellis::{
    Rate, Reconstruction, TENSOR_CORE_ORDER, TILE_WEIGHTS, plan_clean_row_scales, plan_decode, plan_fill_floats,
    plan_from_tiles, plan_hadamard_columns, plan_hadamard_rows, plan_pack, plan_reconstruct, plan_sum_products,
    plan_tile_mean_square, plan_to_tiles, plan_viterbi,
};
use support::{Rig, gaussians, rig};

const RATES: [f64; 7] = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0];

/// What one Viterbi pass over `tiles` gives: states, decoded values and, after packing, the decoded packed words.
struct Encoded {
    indices: Vec<u8>,
    decoded: Vec<f32>,
    packed: Vec<u8>,
    redecoded: Vec<f32>,
}

fn encode(rig: &Rig, rate: Rate, tiles: &[f32], scratch_tiles: u32) -> Encoded {
    let count = tiles.len() / TILE_WEIGHTS;
    let words = rate.words_per_tile();
    let tiles_buffer = rig.upload_f32(tiles);
    let indices = rig.buffer(2 * tiles.len());
    let decoded = rig.buffer(4 * tiles.len());
    let packed = rig.buffer(2 * count * words);
    let redecoded = rig.buffer(4 * tiles.len());
    let scratch = rig.buffer(usize::try_from(rate.scratch_bytes(u64::from(scratch_tiles))).unwrap());
    let mut steps = plan_viterbi(
        rate,
        tiles_buffer.addr(),
        count as u32,
        indices.addr(),
        decoded.addr(),
        scratch.addr(),
        scratch_tiles,
    )
    .unwrap();
    steps.extend(plan_pack(rate, indices.addr(), count as u32, packed.addr()).unwrap());
    steps.extend(plan_decode(rate, packed.addr(), count as u32, redecoded.addr()).unwrap());
    rig.run(&steps);
    Encoded {
        indices: rig.download(&indices, 2 * tiles.len()),
        decoded: rig.download_f32(&decoded, tiles.len()),
        packed: rig.download(&packed, 2 * count * words),
        redecoded: rig.download_f32(&redecoded, tiles.len()),
    }
}

#[test]
#[ignore = "needs a GPU and the bundled NVRTC"]
fn chunking_the_scratch_changes_no_state_and_packed_words_decode_to_the_encoded_values() {
    let rig = rig();
    let tiles = gaussians(40 * TILE_WEIGHTS, 11);
    for bits in RATES {
        let rate = Rate::new(bits).unwrap();
        let whole = encode(&rig, rate, &tiles, 40);
        let chunked = encode(&rig, rate, &tiles, 7);
        assert_eq!(whole.indices, chunked.indices, "states, K {bits}");
        assert_eq!(whole.packed, chunked.packed, "words, K {bits}");
        assert_eq!(whole.decoded, chunked.decoded, "values, K {bits}");
        assert_eq!(whole.decoded, whole.redecoded, "decoded words, K {bits}");
        assert_ne!(
            whole.decoded,
            vec![0.0; tiles.len()],
            "the encode wrote nothing, K {bits}"
        );
    }
}

#[test]
#[ignore = "needs a GPU and the bundled NVRTC"]
fn the_hadamard_rotations_are_their_own_inverses() {
    let rig = rig();
    let (rows, cols) = (256usize, 384usize);
    let x = gaussians(rows * cols, 3);
    let buffer = rig.upload_f32(&x);
    let address = buffer.addr();
    let mut steps = plan_hadamard_rows(address, rows as u32, cols as u32).unwrap();
    steps.extend(plan_hadamard_columns(address, rows as u32, cols as u32).unwrap());
    steps.extend(plan_hadamard_columns(address, rows as u32, cols as u32).unwrap());
    steps.extend(plan_hadamard_rows(address, rows as u32, cols as u32).unwrap());
    rig.run(&steps);
    let back = rig.download_f32(&buffer, x.len());
    for (index, (a, b)) in x.iter().zip(&back).enumerate() {
        assert!((a - b).abs() <= 1e-5, "element {index}: {a} against {b}");
    }
}

/// A single block transform against the explicit Sylvester matrix, by rows and by columns.
#[test]
#[ignore = "needs a GPU and the bundled NVRTC"]
fn a_block_rotation_is_the_sylvester_matrix_over_the_root_of_the_block() {
    let rig = rig();
    let x = gaussians(128 * 128, 5);
    let by_rows = rig.upload_f32(&x);
    let by_columns = rig.upload_f32(&x);
    let mut steps = plan_hadamard_rows(by_rows.addr(), 128, 128).unwrap();
    steps.extend(plan_hadamard_columns(by_columns.addr(), 128, 128).unwrap());
    rig.run(&steps);
    let (rows_out, columns_out) = (
        rig.download_f32(&by_rows, x.len()),
        rig.download_f32(&by_columns, x.len()),
    );
    let scale = 1.0 / 128f64.sqrt();
    for i in 0..128usize {
        for c in 0..128usize {
            let (mut row_sum, mut column_sum) = (0.0f64, 0.0f64);
            for j in 0..128usize {
                let h = if (i & j).count_ones().is_multiple_of(2) {
                    scale
                } else {
                    -scale
                };
                row_sum += h * f64::from(x[j * 128 + c]);
                column_sum += h * f64::from(x[c * 128 + j]);
            }
            assert!(
                (row_sum - f64::from(rows_out[i * 128 + c])).abs() <= 1e-4,
                "rows {i},{c}"
            );
            assert!(
                (column_sum - f64::from(columns_out[c * 128 + i])).abs() <= 1e-4,
                "columns {c},{i}"
            );
        }
    }
}

/// `to_tiles` and `from_tiles` are inverse permutations of a 16-row strip, in the tensor-core order.
#[test]
#[ignore = "needs a GPU and the bundled NVRTC"]
fn a_strip_goes_to_tiles_in_tensor_core_order_and_back() {
    let rig = rig();
    let (k, n, first) = (48usize, 64usize, 16usize);
    let w: Vec<f32> = (0..k * n).map(|i| i as f32).collect();
    let w_buffer = rig.upload_f32(&w);
    let tiles = rig.buffer(4 * (n / 16) * TILE_WEIGHTS);
    rig.run(&plan_to_tiles(w_buffer.addr(), k as u32, n as u32, first as u32, tiles.addr()).unwrap());
    let gathered = rig.download_f32(&tiles, (n / 16) * TILE_WEIGHTS);
    for tile in 0..n / 16 {
        for (position, element) in TENSOR_CORE_ORDER.iter().enumerate() {
            let element = usize::from(*element);
            let expected = ((first + element / 16) * n + tile * 16 + element % 16) as f32;
            assert_eq!(
                gathered[tile * TILE_WEIGHTS + position],
                expected,
                "tile {tile} position {position}"
            );
        }
    }
    let target = rig.buffer(4 * w.len());
    rig.run(&plan_from_tiles(tiles.addr(), k as u32, n as u32, first as u32, target.addr()).unwrap());
    let back = rig.download_f32(&target, w.len());
    for (i, value) in back.iter().enumerate() {
        let inside = (first..first + 16).contains(&(i / n));
        assert_eq!(*value, if inside { w[i] } else { 0.0 }, "element {i}");
    }
}

#[test]
#[ignore = "needs a GPU and the bundled NVRTC"]
fn tile_energies_match_a_host_sum() {
    let rig = rig();
    let (k, n) = (64usize, 96usize);
    let a = gaussians(k * n, 21);
    let a_buffer = rig.upload_f32(&a);
    let energy = rig.buffer(4 * (k / 16) * (n / 16));
    rig.run(&plan_tile_mean_square(a_buffer.addr(), k as u32, n as u32, energy.addr()).unwrap());
    let energies = rig.download_f32(&energy, (k / 16) * (n / 16));
    for (tile, got) in energies.iter().enumerate() {
        let (tile_row, tile_column) = (tile / (n / 16), tile % (n / 16));
        let mut sum = 0.0f64;
        for row in 0..16 {
            for column in 0..16 {
                let v = f64::from(a[(tile_row * 16 + row) * n + tile_column * 16 + column]);
                sum += v * v;
            }
        }
        assert!((sum / 256.0 - f64::from(*got)).abs() <= 1e-6, "tile {tile}");
    }
}

/// Fixed-point inputs `i / 2^16` with `|i| < 2^16` make every product a multiple of `2^-32` below 1 and every
/// partial sum of the 524325 of them an exact double, so the sum is the same in any order of additions and is
/// compared exactly. More values than the 1024 blocks of 256 threads make each thread loop more than once.
#[test]
#[ignore = "needs a GPU and the bundled NVRTC"]
fn sums_of_products_are_exact_over_more_values_than_the_threads_of_the_launch() {
    let rig = rig();
    let count = 1024 * 256 * 2 + 37;
    let mut state = 12345u64;
    let mut fixed = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        i64::try_from((state >> 33) % 131071).unwrap() - 65535
    };
    let a: Vec<i64> = (0..count).map(|_| fixed()).collect();
    let b: Vec<i64> = (0..count).map(|_| fixed()).collect();
    let as_floats = |values: &[i64]| -> Vec<f32> { values.iter().map(|v| *v as f32 / 65536.0).collect() };
    let (a_buffer, b_buffer) = (rig.upload_f32(&as_floats(&a)), rig.upload_f32(&as_floats(&b)));
    // The two doubles start as bytes that are not zero, so the zeroing step is what makes them start at zero.
    let sums = rig.upload(&[0xABu8; 16]);
    let mut steps = plan_fill_floats(sums.addr(), 4, 0.0).unwrap();
    steps.extend(plan_sum_products(a_buffer.addr(), b_buffer.addr(), count as u64, sums.addr()).unwrap());
    steps.extend(plan_sum_products(a_buffer.addr(), a_buffer.addr(), count as u64, sums.addr() + 8).unwrap());
    rig.run(&steps);
    let raw = rig.download(&sums, 16);
    let double = |at: usize| f64::from_le_bytes(raw[at..at + 8].try_into().unwrap());
    let exact = |x: &[i64], y: &[i64]| x.iter().zip(y).map(|(p, q)| p * q).sum::<i64>() as f64 / 4294967296.0;
    assert_eq!(double(0), exact(&a, &b));
    assert_eq!(double(8), exact(&a, &a));
}

#[test]
#[ignore = "needs a GPU and the bundled NVRTC"]
fn cleaned_row_scales_keep_finite_positive_values_and_count_a_failed_factorization() {
    let rig = rig();
    let values = [2.0f32, f32::NAN, -1.0, 0.0, f32::INFINITY, 0.5];
    for (status, expect, failures) in [(0i32, [2.0f32, 1.0, 1.0, 1.0, 1.0, 0.5], 0i32), (3, [1.0; 6], 1)] {
        let r = rig.upload_f32(&values);
        let info = rig.upload(&status.to_le_bytes());
        let count = rig.upload(&0i32.to_le_bytes());
        rig.run(&plan_clean_row_scales(r.addr(), info.addr(), values.len() as u32, count.addr()).unwrap());
        assert_eq!(rig.download_f32(&r, values.len()), expect, "status {status}");
        assert_eq!(rig.download(&count, 4), failures.to_le_bytes(), "status {status}");
    }
}

/// Random packed words, decoded by the library's own decode, then the documented transform on the host:
/// `W = had_r(had_l(Q) * suh[:, None]) * svh[None, :]` with `Q` the decoded `[k, n]` matrix, written `[n, k]` in
/// bf16. `k` is not `n`, so a swapped pair of sign vectors, a dropped rotation, a transpose over the wrong grid or
/// the kernel of another rate all differ by far more than one bf16 step.
#[test]
#[ignore = "needs a GPU and the bundled NVRTC"]
fn a_reconstruction_is_the_documented_transform_of_the_decoded_matrix_within_one_bf16_step() {
    let rig = rig();
    let (k, n) = (128usize, 256usize);
    let (tiles_k, tiles_n) = (k / 16, n / 16);
    let mut state = 99u64;
    let mut random = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };
    let signs = |count: usize, random: &mut dyn FnMut() -> u32| -> Vec<f32> {
        (0..count)
            .map(|_| {
                let magnitude = (4 + random() % 9) as f32 / 8.0;
                if random().is_multiple_of(2) {
                    magnitude
                } else {
                    -magnitude
                }
            })
            .collect()
    };
    let suh = signs(k, &mut random);
    let svh = signs(n, &mut random);
    let half_bytes =
        |values: &[f32]| -> Vec<u8> { values.iter().flat_map(|v| f32_to_half(*v).to_le_bytes()).collect() };
    let (suh_buffer, svh_buffer) = (rig.upload(&half_bytes(&suh)), rig.upload(&half_bytes(&svh)));
    for bits in RATES {
        let rate = Rate::new(bits).unwrap();
        let tiles = tiles_k * tiles_n;
        let word_count = tiles * rate.words_per_tile();
        let words: Vec<u8> = (0..word_count).flat_map(|_| (random() as u16).to_le_bytes()).collect();
        let words_buffer = rig.upload(&words);
        let decoded = rig.buffer(4 * tiles * TILE_WEIGHTS);
        rig.run(&plan_decode(rate, words_buffer.addr(), tiles as u32, decoded.addr()).unwrap());
        let values = rig.download_f32(&decoded, tiles * TILE_WEIGHTS);
        let mut q = vec![0.0f64; k * n];
        for tile in 0..tiles {
            let (tile_row, tile_column) = (tile / tiles_n, tile % tiles_n);
            for (position, element) in TENSOR_CORE_ORDER.iter().enumerate() {
                let element = usize::from(*element);
                let at = (tile_row * 16 + element / 16) * n + tile_column * 16 + element % 16;
                q[at] = f64::from(values[tile * TILE_WEIGHTS + position]);
            }
        }
        let expected = reconstruct_on_the_host(&q, &suh, &svh, k, n);

        let out = rig.buffer(2 * k * n);
        let scratch = rig.buffer(4 * k * n);
        let operands = Reconstruction {
            words: words_buffer.addr(),
            suh: suh_buffer.addr(),
            svh: svh_buffer.addr(),
            k: k as u32,
            n: n as u32,
            out: out.addr(),
            scratch: scratch.addr(),
        };
        rig.run(&plan_reconstruct(rate, &operands).unwrap());
        let raw = rig.download(&out, 2 * k * n);
        for (i, want) in expected.iter().enumerate() {
            let got = f64::from(f32::from_bits(
                u32::from(u16::from_le_bytes([raw[2 * i], raw[2 * i + 1]])) << 16,
            ));
            let step = 2f64.powi(want.abs().max(1e-30).log2().floor() as i32 - 7);
            assert!(
                (got - want).abs() <= step.max(1e-5),
                "K {bits}, output {i}: {got} against {want}"
            );
        }
        assert!(
            expected.iter().any(|v| v.abs() > 1e-3),
            "the transform is not all zero, K {bits}"
        );
    }
}

/// The weight `[n, k]` from the decoded `q` `[k, n]`: a Hadamard rotation of each 128-row block of every column, the
/// row scales, a rotation of each 128-column block of every row, the column scales, and the transpose.
fn reconstruct_on_the_host(q: &[f64], suh: &[f32], svh: &[f32], k: usize, n: usize) -> Vec<f64> {
    let scale = 1.0 / 128f64.sqrt();
    let sign = |i: usize, j: usize| {
        if (i & j).count_ones().is_multiple_of(2) {
            scale
        } else {
            -scale
        }
    };
    let mut w = vec![0.0f64; k * n];
    for block in 0..k / 128 {
        for c in 0..n {
            for i in 0..128 {
                let sum: f64 = (0..128).map(|j| sign(i, j) * q[(block * 128 + j) * n + c]).sum();
                let r = block * 128 + i;
                w[r * n + c] = sum * f64::from(suh[r]);
            }
        }
    }
    let rotated_rows = w.clone();
    for r in 0..k {
        for block in 0..n / 128 {
            for i in 0..128 {
                let sum: f64 = (0..128)
                    .map(|j| sign(i, j) * rotated_rows[r * n + block * 128 + j])
                    .sum();
                let c = block * 128 + i;
                w[r * n + c] = sum * f64::from(svh[c]);
            }
        }
    }
    let mut out = vec![0.0f64; k * n];
    for r in 0..k {
        for c in 0..n {
            out[c * k + r] = w[r * n + c];
        }
    }
    out
}

/// The half-precision bits of an `f32` that is exactly representable.
fn f32_to_half(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mantissa = ((bits >> 13) & 0x3ff) as u16;
    sign | ((exponent as u16) << 10) | mantissa
}
