//! Device tests of GPTQ rounding. `#[ignore]`d like the crate's other device tests:
//! `cargo test -p euhedral-inference-quantization --test gptq_gpu -- --ignored --nocapture`.
//!
//! The engine library comes from `euhedral.cuda.library` and the math libraries from
//! `euhedral.quantization.cuda-libraries`; a missing setting fails the test. The kernels' NVRTC and its builtins
//! library must be reachable (`EUHEDRAL_NVRTC`, `LD_LIBRARY_PATH`).
//!
//! The tests round the Hessian and the weight of the frozen calibration fixture and check what must hold of any GPTQ
//! rounding: dead columns, act-order equivariance, the error under H against nearest-code rounding, the code range,
//! the decoded output, the identity table and the SD4 table over chunks.
#![cfg(target_os = "linux")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]
#![allow(clippy::print_stdout, reason = "the error ratios are the test's output")]
#![allow(clippy::disallowed_methods, reason = "a test: reads its fixture files")]

mod gptq_rig;

use std::array::from_fn;
use std::fs::read;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use euhedral_inference_cuda::DeviceBuffer;
use euhedral_inference_quantization::gptq::format::Format;
use euhedral_inference_quantization::gptq::state::QuantState;
use euhedral_inference_quantization::gptq::{GptqError, GptqSettings, RoundRequest, TensorScales};
use euhedral_inference_quantization::memory::Claims;
use euhedral_inference_quantization::npy::Npy;
use gptq_rig::{Rig, range_of, rig};
use serde_json::Value;

const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

fn fixture(name: &str) -> Vec<u8> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../resources/fixtures/gptq-fixture");
    read(path.join(name)).unwrap_or_else(|error| panic!("{name}: {error}"))
}

fn half(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let (exponent, fraction) = (i32::from((bits >> 10) & 0x1f), f32::from(bits & 0x3ff));
    match exponent {
        0 => sign * fraction * f32::powi(2.0, -24),
        31 => sign * f32::INFINITY,
        _ => sign * (1.0 + fraction / 1024.0) * f32::powi(2.0, exponent - 15),
    }
}

/// What a stored scale multiplies a code's value by.
fn scale_of(format: &Format, scales: &[u8], global: f32, table: Option<[u8; 16]>, at: usize) -> f32 {
    if format.nvfp4 {
        let stored = scales[at];
        let e = u32::from(table.map_or(stored, |table| table[usize::from(stored)]));
        let magnitude = if e < 8 {
            (e & 7) as f32 / 8.0 * 0.015625
        } else {
            (1.0 + (e & 7) as f32 / 8.0) * f32::powi(2.0, (e >> 3) as i32 - 7)
        };
        magnitude * global
    } else {
        half(u16::from_le_bytes([scales[2 * at], scales[2 * at + 1]]))
    }
}

/// The value of a code before its scale.
fn value_of(format: &Format, code: u8) -> f32 {
    if format.nvfp4 {
        let sign = if code & 8 != 0 { -1.0 } else { 1.0 };
        E2M1[usize::from(code & 7)] * sign
    } else {
        f32::from(code as i8)
    }
}

/// The decoded weights of a stored rounding, as the kernels compute them.
fn decode(
    format: &Format,
    codes: &[u8],
    scales: &[u8],
    global: f32,
    table: Option<[u8; 16]>,
    n: usize,
    k: usize,
) -> Vec<f32> {
    let groups = k / format.group;
    let mut out = vec![0.0; n * k];
    for r in 0..n {
        for j in 0..k {
            let at = r * groups + j / format.group;
            out[r * k + j] = value_of(format, codes[r * k + j]) * scale_of(format, scales, global, table, at);
        }
    }
    out
}

/// The codes that round every weight to the nearest value under the scales a rounding chose.
fn nearest_codes(format: &Format, w: &[f32], state: &QuantState, n: usize, k: usize) -> Vec<u8> {
    let groups = k / format.group;
    let candidates: Vec<u8> = if format.nvfp4 {
        (0..16).collect()
    } else {
        (format.qmin..=format.qmax).map(|code| code as i8 as u8).collect()
    };
    let mut codes = vec![0u8; n * k];
    for r in 0..n {
        for j in 0..k {
            let at = r * groups + j / format.group;
            let scale = scale_of(format, &state.scales, state.global_scale, state.table, at);
            let target = w[r * k + j];
            let distance = |code: &u8| (value_of(format, *code) * scale - target).abs();
            codes[r * k + j] = *candidates
                .iter()
                .min_by(|a, b| distance(a).total_cmp(&distance(b)))
                .unwrap();
        }
    }
    codes
}

fn bf16_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| f32::from_bits(u32::from(u16::from_le_bytes(*pair)) << 16))
        .collect()
}

/// The BF16 bits of values that are BF16 already.
fn f32_to_bf16(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

/// The sum over rows of (w - q) H (w - q)^T: the output error over the inputs H summarizes.
fn output_error(w: &[f32], q: &[f32], h: &[f32], n: usize, k: usize) -> f64 {
    let mut total = 0.0;
    let mut d = vec![0.0f64; k];
    for r in 0..n {
        for j in 0..k {
            d[j] = f64::from(w[r * k + j]) - f64::from(q[r * k + j]);
        }
        for a in 0..k {
            let row: f64 = (0..k).map(|b| f64::from(h[a * k + b]) * d[b]).sum();
            total += d[a] * row;
        }
    }
    total
}

const GROUPED: (&str, &str) = ("Q3G64_F16S", "row-split-k128-v1");
const NVFP4: (&str, &str) = ("NVFP4", "row-split-k128-v1");
const SD4: (&str, &str) = ("NVFP4", "row-split-k128-sd4-v1");

fn format((name, layout): (&str, &str)) -> Format {
    Format::of(name, layout, None).unwrap()
}

/// The fixture's weight and Hessian.
struct Problem {
    n: usize,
    k: usize,
    damp: f32,
    w: Vec<f32>,
    h: Vec<f32>,
}

fn problem() -> Problem {
    let cases: Value = serde_json::from_slice(&fixture("cases.json")).unwrap();
    let weight_file = fixture("weight_bf16.npy");
    let hessian_file = fixture("hessian.npy");
    Problem {
        n: cases["rows"].as_u64().unwrap() as usize,
        k: cases["columns"].as_u64().unwrap() as usize,
        damp: cases["damp"].as_f64().unwrap() as f32,
        w: bf16_to_f32(Npy::parse(&weight_file).unwrap().data),
        h: Npy::parse(&hessian_file).unwrap().to_vec::<f32>().unwrap(),
    }
}

/// A rounding with the decoded weights the kernel wrote.
struct Rounded {
    state: QuantState,
    decoded: Vec<f32>,
}

fn floats(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|value| value.to_le_bytes()).collect()
}

fn round(rig: &mut Rig, p: &Problem, w: &[f32], h: &[f32], format: &Format, act_order: bool) -> Rounded {
    let weight = rig.upload(&f32_to_bf16(w));
    let hessian = rig.upload(&floats(h));
    let factor = rig.factor(&hessian, p.k, p.damp, act_order).unwrap();
    let tensor = if format.nvfp4 {
        rig.tensor_scales(&weight, p.n, &factor, format).unwrap()
    } else {
        TensorScales {
            global_scale: 0.0,
            table: None,
        }
    };
    let out = DeviceBuffer::alloc(&rig.library, 2 * p.n * p.k).unwrap();
    let state = rig
        .round(RoundRequest {
            weight: range_of(&weight),
            n: p.n,
            factor: &factor,
            format,
            tensor: &tensor,
            decoded_out: Some(range_of(&out)),
        })
        .unwrap();
    let mut bytes = vec![0u8; 2 * p.n * p.k];
    out.copy_to_host(0, &mut bytes).unwrap();
    Rounded {
        state,
        decoded: bf16_to_f32(&bytes),
    }
}

fn error_under(h: &[f32], w: &[f32], q: &[f32], p: &Problem) -> f64 {
    output_error(w, q, h, p.n, p.k)
}

#[test]
#[ignore = "needs a GPU, the native library, the math libraries and the bundled NVRTC"]
fn dead_columns_round_to_zero_and_the_others_do_not() {
    let p = problem();
    let dead = [5usize, 70, 200];
    let mut h = p.h.clone();
    for &c in &dead {
        for i in 0..p.k {
            h[c * p.k + i] = 0.0;
            h[i * p.k + c] = 0.0;
        }
    }
    let mut rig = rig(GptqSettings::default());
    for (case, act_order) in [(GROUPED, false), (NVFP4, true), (SD4, false)] {
        let format = format(case);
        let rounded = round(&mut rig, &p, &p.w, &h, &format, act_order);
        for r in 0..p.n {
            for j in 0..p.k {
                let at = r * p.k + j;
                if dead.contains(&j) {
                    assert_eq!(
                        rounded.state.codes[at], 0,
                        "{} row {r} column {j}: dead code",
                        format.name
                    );
                    assert_eq!(
                        rounded.decoded[at], 0.0,
                        "{} row {r} column {j}: dead output",
                        format.name
                    );
                }
            }
        }
        let live = (0..p.n * p.k).filter(|at| !dead.contains(&(at % p.k)));
        let nonzero = live.filter(|&at| rounded.state.codes[at] != 0).count();
        assert!(
            nonzero > p.n * (p.k - dead.len()) / 2,
            "{}: {nonzero} live codes are nonzero",
            format.name
        );
    }
}

#[test]
#[ignore = "needs a GPU, the native library, the math libraries and the bundled NVRTC"]
fn act_order_permutes_the_codes_with_the_columns() {
    let p = problem();
    // Columns move only inside their blocks of 16, so every group keeps its set of columns and its scales.
    let permutation: Vec<usize> = (0..p.k).map(|j| (j / 16) * 16 + (j % 16 + 5) % 16).collect();
    let w: Vec<f32> = (0..p.n)
        .flat_map(|r| permutation.iter().map(move |&c| (r, c)))
        .map(|(r, c)| p.w[r * p.k + c])
        .collect();
    let h: Vec<f32> = (0..p.k)
        .flat_map(|a| permutation.iter().map(move |&b| (a, b)))
        .map(|(a, b)| p.h[permutation[a] * p.k + b])
        .collect();
    let diagonal: Vec<f32> = (0..p.k).map(|j| p.h[j * p.k + j]).collect();
    let mut sorted = diagonal.clone();
    sorted.sort_by(f32::total_cmp);
    assert!(
        sorted.windows(2).all(|pair| pair[0] != pair[1]),
        "the importances must be distinct for a unique order"
    );
    let mut rig = rig(GptqSettings::default());
    for case in [GROUPED, NVFP4] {
        let format = format(case);
        let plain = round(&mut rig, &p, &p.w, &p.h, &format, true);
        let moved = round(&mut rig, &p, &w, &h, &format, true);
        for r in 0..p.n {
            for (j, &source) in permutation.iter().enumerate() {
                assert_eq!(
                    moved.state.codes[r * p.k + j],
                    plain.state.codes[r * p.k + source],
                    "{} row {r} column {j}",
                    format.name
                );
            }
        }
        assert_eq!(moved.state.scales, plain.state.scales, "{}: scales", format.name);
        assert_eq!(
            moved.state.global_scale, plain.state.global_scale,
            "{}: global",
            format.name
        );
    }
}

#[test]
#[ignore = "needs a GPU, the native library, the math libraries and the bundled NVRTC"]
fn the_error_under_h_beats_nearest_rounding_and_the_codes_stay_in_range() {
    let p = problem();
    let mut rig = rig(GptqSettings::default());
    // The bound of each case sits just above the ratio it was seen to give.
    for (case, act_order, bound) in [
        (GROUPED, false, 0.585),
        (GROUPED, true, 0.63),
        (NVFP4, false, 0.48),
        (SD4, true, 0.70),
    ] {
        let format = format(case);
        let rounded = round(&mut rig, &p, &p.w, &p.h, &format, act_order);
        let state = &rounded.state;
        for &code in &state.codes {
            if format.nvfp4 {
                assert!(code < 16, "{}: code {code}", format.name);
            } else {
                let signed = i32::from(code as i8);
                assert!(
                    (format.qmin..=format.qmax).contains(&signed),
                    "{}: code {signed}",
                    format.name
                );
            }
        }
        let table = state.table;
        let chosen = decode(
            &format,
            &state.codes,
            &state.scales,
            state.global_scale,
            table,
            p.n,
            p.k,
        );
        let nearest = nearest_codes(&format, &p.w, state, p.n, p.k);
        let rtn = decode(&format, &nearest, &state.scales, state.global_scale, table, p.n, p.k);
        let (gptq, plain) = (error_under(&p.h, &p.w, &chosen, &p), error_under(&p.h, &p.w, &rtn, &p));
        println!(
            "{} act-order {act_order}: error {gptq:.6e} against nearest {plain:.6e}, ratio {:.4}",
            format.name,
            gptq / plain
        );
        assert!(
            gptq < bound * plain,
            "{}: error {gptq} is not below {bound} of nearest rounding's {plain}",
            format.name
        );
    }
}

#[test]
#[ignore = "needs a GPU, the native library, the math libraries and the bundled NVRTC"]
fn the_decoded_output_is_the_decode_of_the_codes_and_scales() {
    let p = problem();
    let mut rig = rig(GptqSettings::default());
    for (case, act_order) in [(GROUPED, true), (NVFP4, false), (SD4, false)] {
        let format = format(case);
        let rounded = round(&mut rig, &p, &p.w, &p.h, &format, act_order);
        let state = &rounded.state;
        let expected = decode(
            &format,
            &state.codes,
            &state.scales,
            state.global_scale,
            state.table,
            p.n,
            p.k,
        );
        let expected = bf16_to_f32(&f32_to_bf16(&expected));
        let wrong = (0..p.n * p.k)
            .filter(|&at| {
                // BF16 holds 8 bits of mantissa: the kernel may round the last one differently.
                (rounded.decoded[at] - expected[at]).abs() > expected[at].abs() / 128.0
            })
            .count();
        assert_eq!(
            wrong, 0,
            "{}: {wrong} decoded values differ from the decode of the codes",
            format.name
        );
        assert!(
            rounded.decoded.iter().any(|value| *value != 0.0),
            "{}: nothing was decoded",
            format.name
        );
    }
}

#[test]
#[ignore = "needs a GPU, the native library, the math libraries and the bundled NVRTC"]
fn a_weight_of_zeros_has_a_zero_global_scale_and_the_identity_table() {
    let p = problem();
    let mut rig = rig(GptqSettings::default());
    let zeros = vec![0.0f32; p.n * p.k];
    let weight = rig.upload(&f32_to_bf16(&zeros));
    let hessian = rig.upload(&floats(&p.h));
    let factor = rig.factor(&hessian, p.k, p.damp, false).unwrap();
    let tensor = rig.tensor_scales(&weight, p.n, &factor, &format(SD4)).unwrap();
    assert_eq!(tensor.global_scale, 0.0);
    let identity: Vec<u8> = (0..16).collect();
    assert_eq!(tensor.table.map(|table| table.to_vec()), Some(identity));
}

#[test]
#[ignore = "needs a GPU, the native library, the math libraries and the bundled NVRTC"]
fn the_sd4_table_is_the_same_over_several_chunks_as_over_one() {
    let p = problem();
    let tables: Vec<_> = [GptqSettings::default().scale_chunk_bytes, 4 * p.k as u64 * 5]
        .into_iter()
        .map(|scale_chunk_bytes| {
            // Five rows to a chunk: thirteen passes over the sixty-four rows.
            let mut rig = rig(GptqSettings {
                scale_chunk_bytes,
                ..GptqSettings::default()
            });
            let weight = rig.upload(&f32_to_bf16(&p.w));
            let hessian = rig.upload(&floats(&p.h));
            let factor = rig.factor(&hessian, p.k, p.damp, true).unwrap();
            rig.tensor_scales(&weight, p.n, &factor, &format(SD4)).unwrap()
        })
        .collect();
    assert_eq!(tables[0].table, tables[1].table);
    assert_eq!(tables[0].global_scale, tables[1].global_scale);
    assert!(
        tables[0]
            .table
            .is_some_and(|table| table != from_fn(|entry| entry as u8))
    );
}

#[test]
#[ignore = "needs a GPU, the native library, the math libraries and the bundled NVRTC"]
fn a_matrix_that_is_not_positive_definite_is_refused_by_the_step_that_found_it() {
    let k = 128usize;
    let mut rig = rig(GptqSettings::default());
    // A diagonal of -1 (nonzero, so no column is dead) is not positive definite even after damping.
    let mut h = vec![0.0f32; k * k];
    for i in 0..k {
        h[i * k + i] = -1.0;
    }
    let hessian = rig.upload(&floats(&h));
    let Err(GptqError::Cuda(error)) = rig.factor(&hessian, k, 0.01, false) else {
        panic!("the factor of a negative matrix was accepted");
    };
    let text = error.to_string();
    assert!(
        text.contains("potrf (H)") && !text.contains("H^-1"),
        "the first failing step is named: {text}"
    );
    // The lane survives the refusal: a good matrix factors on it.
    for i in 0..k {
        h[i * k + i] = 1.0;
    }
    let good = rig.upload(&floats(&h));
    let factor = rig.factor(&good, k, 0.01, true).unwrap();
    assert_eq!(factor.order().len(), k);
}

#[test]
#[ignore = "needs a GPU, the native library, the math libraries and the bundled NVRTC"]
fn a_step_takes_exactly_the_bytes_it_charges_and_gives_them_all_back() {
    let p = problem();
    let mut rig = rig(GptqSettings::default());
    let capacity = rig.capacity();
    let weight = rig.upload(&f32_to_bf16(&p.w));
    let hessian = rig.upload(&floats(&p.h));
    let format = format(SD4);
    let factor = rig.factor(&hessian, p.k, p.damp, true).unwrap();
    // The factor's upper matrix is the only memory still held: 4 * k * k device bytes.
    assert_eq!(rig.free().0, capacity - 4 * (p.k * p.k) as u64);
    let tensor = rig.tensor_scales(&weight, p.n, &factor, &format).unwrap();
    let request = RoundRequest {
        weight: range_of(&weight),
        n: p.n,
        factor: &factor,
        format: &format,
        tensor: &tensor,
        decoded_out: None,
    };
    rig.round(request).unwrap();
    assert_eq!(rig.free(), (capacity - 4 * (p.k * p.k) as u64, capacity));
    // A claim one byte short, of either kind, is refused by each step, and nothing stays allocated or charged.
    for short in [(1, 0), (0, 1)] {
        rig.short = short;
        let refused = |step: &str, result: Result<(), GptqError>| match result {
            Err(GptqError::Request(text)) => assert!(text.contains("claim"), "{step}: {text}"),
            other => panic!("{step}: a claim short by {short:?} was accepted: {other:?}"),
        };
        refused("factor", rig.factor(&hessian, p.k, p.damp, true).map(|_| ()));
        refused("scales", rig.tensor_scales(&weight, p.n, &factor, &format).map(|_| ()));
        refused("round", rig.round(request).map(|_| ()));
        rig.short = (0, 0);
    }
    rig.settle();
    assert_eq!(rig.free(), (capacity - 4 * (p.k * p.k) as u64, capacity));
    drop(factor);
    rig.settle();
    assert_eq!(rig.free(), (capacity, capacity));
}

#[test]
#[ignore = "needs a GPU, the native library, the math libraries and the bundled NVRTC"]
fn staging_is_charged_at_its_pool_size_class() {
    // Forty rows of 256 columns: the codes take 20480 bytes, which the pool rounds up to 32768.
    let p = problem();
    let n = 40;
    let mut rig = rig(GptqSettings::default());
    let capacity = rig.capacity();
    let weight = rig.upload(&f32_to_bf16(&p.w[..n * p.k]));
    let hessian = rig.upload(&floats(&p.h));
    let format = format(SD4);
    let factor = rig.factor(&hessian, p.k, p.damp, true).unwrap();
    let tensor = rig.tensor_scales(&weight, n, &factor, &format).unwrap();
    let request = RoundRequest {
        weight: range_of(&weight),
        n,
        factor: &factor,
        format: &format,
        tensor: &tensor,
        decoded_out: None,
    };
    rig.round(request).unwrap();
    rig.settle();
    assert_eq!(rig.free(), (capacity - 4 * (p.k * p.k) as u64, capacity));
    // A pinned claim one byte under the size class is refused.
    rig.short = (0, 1);
    assert!(matches!(rig.round(request), Err(GptqError::Request(text)) if text.contains("claim")));
    rig.short = (0, 0);
    drop(factor);
    rig.settle();
    assert_eq!(rig.free(), (capacity, capacity));
}

#[test]
#[ignore = "needs a GPU, the native library, the math libraries and the bundled NVRTC"]
fn a_step_given_no_claims_is_refused() {
    let p = problem();
    let mut rig = rig(GptqSettings::default());
    let hessian = rig.upload(&floats(&p.h));
    let range = range_of(&hessian);
    let refused = rig.run(|lane, done| lane.factor(range, p.k, p.damp, false, Claims::default(), done));
    assert!(matches!(refused, Err(GptqError::Request(text)) if text.contains("no ") && text.contains("claim")));
}

#[test]
#[ignore = "needs a GPU, the native library, the math libraries and the bundled NVRTC"]
fn the_sd4_size_of_a_weight_with_no_columns_is_an_error() {
    let mut rig = rig(GptqSettings::default());
    assert!(matches!(
        rig.scales_bytes(64, 0, &format(SD4)),
        Err(GptqError::Request(_))
    ));
}

#[test]
#[ignore = "needs a GPU, the native library, the math libraries and the bundled NVRTC"]
fn a_factor_from_another_lane_is_refused() {
    let p = problem();
    let mut first = rig(GptqSettings::default());
    let mut second = rig(GptqSettings::default());
    let hessian = first.upload(&floats(&p.h));
    let factor = first.factor(&hessian, p.k, p.damp, false).unwrap();
    let weight = second.upload(&f32_to_bf16(&p.w));
    let format = format(NVFP4);
    assert!(matches!(
        second.tensor_scales(&weight, p.n, &factor, &format),
        Err(GptqError::ForeignFactor)
    ));
    let tensor = TensorScales {
        global_scale: 1.0,
        table: None,
    };
    let request = RoundRequest {
        weight: range_of(&weight),
        n: p.n,
        factor: &factor,
        format: &format,
        tensor: &tensor,
        decoded_out: None,
    };
    assert!(matches!(second.round(request), Err(GptqError::ForeignFactor)));
    // On its own lane it works.
    let own = first.upload(&f32_to_bf16(&p.w));
    first.tensor_scales(&own, p.n, &factor, &format).unwrap();
}

#[test]
#[ignore = "needs a GPU, the native library, the math libraries and the bundled NVRTC"]
fn a_chain_that_cannot_arm_keeps_its_staging_until_a_retirement_is_confirmed() {
    let p = problem();
    let mut rig = rig(GptqSettings::default());
    let hessian = rig.upload(&floats(&p.h));
    assert_eq!(rig.pinned.allocated_bytes(), 0);
    // A retirement nobody collected is outstanding: the chain's own cannot be armed, after it enqueued its copies.
    rig.block_arming();
    let failed = rig.factor(&hessian, p.k, p.damp, false);
    assert!(matches!(failed, Err(GptqError::Gpu(_))), "{failed:?}");
    assert_eq!(rig.unsettled(), 1);
    assert!(rig.pinned.allocated_bytes() > 0, "the chain took staging");
    assert_eq!(
        rig.pinned.held_bytes(),
        0,
        "the staging went back to the pool while its download was still queued"
    );
    rig.unblock_arming();
    // The next chain's confirmed retirement is behind the queued work: the staging is released then.
    rig.factor(&hessian, p.k, p.damp, false).unwrap();
    assert_eq!(rig.unsettled(), 0);
    assert!(rig.pinned.held_bytes() > 0, "the staging came back to the pool");
}

#[test]
#[ignore = "needs a GPU, the native library, the math libraries and the bundled NVRTC"]
fn a_step_refused_for_a_short_pinned_claim_allocates_no_pinned_memory() {
    let p = problem();
    let mut rig = rig(GptqSettings::default());
    let hessian = rig.upload(&floats(&p.h));
    // No pinned bytes at all: the first staging buffer is refused, so the pool must not have allocated for it.
    rig.short = (0, u64::MAX);
    let refused = rig.factor(&hessian, p.k, p.damp, false);
    assert!(matches!(refused, Err(GptqError::Request(text)) if text.contains("claim")));
    assert_eq!(
        rig.pinned.allocated_bytes(),
        0,
        "the pool allocated before the claim was charged"
    );
}

#[test]
#[ignore = "needs a GPU, the native library, the math libraries and the bundled NVRTC"]
fn shutdown_waits_for_the_stream_before_it_releases_a_failed_chains_staging() {
    let p = problem();
    let mut rig = rig(GptqSettings::default());
    let hessian = rig.upload(&floats(&p.h));
    rig.block_arming();
    let failed = rig.factor(&hessian, p.k, p.damp, false);
    assert!(matches!(failed, Err(GptqError::Gpu(_))), "{failed:?}");
    assert_eq!(rig.unsettled(), 1);
    // The probe runs just before the lane waits for its stream: the staging must still be parked then.
    let pool = Arc::clone(&rig.pinned);
    let calls = Arc::new(AtomicUsize::new(0));
    let seen = Arc::clone(&calls);
    let probe = Box::new(move || {
        assert_eq!(
            pool.held_bytes(),
            0,
            "the staging went back to the pool before the stream was quiesced"
        );
        seen.fetch_add(1, Ordering::SeqCst);
    });
    rig.shutdown(probe).expect("reclaim");
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "shutdown did not wait for the stream before releasing"
    );
    assert!(
        rig.pinned.held_bytes() > 0,
        "the staging came back to the pool after the wait"
    );
}
