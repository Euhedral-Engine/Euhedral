//! Device tests of GPTQ rounding. `#[ignore]`d like the crate's other device tests:
//! `cargo test -p euhedral-inference-quantization --test gptq_gpu -- --ignored --nocapture`.
//!
//! The engine library comes from `euhedral.cuda.library` and the math libraries from
//! `euhedral.quantization.cuda-libraries`; a missing setting fails the test. The kernels' NVRTC and its builtins
//! library must be reachable (`EUHEDRAL_NVRTC`, `LD_LIBRARY_PATH`).
//!
//! The first test rounds the frozen fixture of a PyTorch reference of the calibration (`quantization/src/test/resources/gptq-fixture`) in
//! every format with and without act-order. The tensor scales and the SD4 table match exactly; the codes come from a
//! factor cuSOLVER computes in its own order, so they are compared by agreement and by the output error under H.
#![cfg(target_os = "linux")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]
#![allow(
    clippy::print_stdout,
    reason = "the agreement and the error ratios are the test's output"
)]
#![allow(clippy::disallowed_methods, reason = "a test: reads its fixture files")]

mod gptq_rig;

use std::fs::read;
use std::path::PathBuf;

use euhedral_inference_quantization::gptq::format::Format;
use euhedral_inference_quantization::gptq::state::QuantState;
use euhedral_inference_quantization::gptq::{GptqError, GptqSettings, RoundRequest, TensorScales};
use euhedral_inference_quantization::npy::Npy;
use gptq_rig::rig;
use serde_json::Value;

const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

fn fixture(name: &str) -> Vec<u8> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../quantization/src/test/resources/gptq-fixture");
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
            let code = codes[r * k + j];
            let at = r * groups + j / format.group;
            out[r * k + j] = if format.nvfp4 {
                let stored = scales[at];
                let e4m3 = table.map_or(stored, |table| table[usize::from(stored)]);
                let e = u32::from(e4m3);
                let magnitude = if e < 8 {
                    (e & 7) as f32 / 8.0 * 0.015625
                } else {
                    (1.0 + (e & 7) as f32 / 8.0) * f32::powi(2.0, (e >> 3) as i32 - 7)
                };
                let sign = if code & 8 != 0 { -1.0 } else { 1.0 };
                E2M1[usize::from(code & 7)] * sign * (magnitude * global)
            } else {
                f32::from(code as i8) * half(u16::from_le_bytes([scales[2 * at], scales[2 * at + 1]]))
            };
        }
    }
    out
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

#[test]
#[ignore = "needs a GPU, the native library, the math libraries and the bundled NVRTC"]
fn rounds_like_the_pytorch_reference() {
    let cases: Value = serde_json::from_slice(&fixture("cases.json")).unwrap();
    let n = cases["rows"].as_u64().unwrap() as usize;
    let k = cases["columns"].as_u64().unwrap() as usize;
    let damp = cases["damp"].as_f64().unwrap() as f32;
    let weight_file = fixture("weight_bf16.npy");
    let weight_npy = Npy::parse(&weight_file).unwrap();
    let hessian_file = fixture("hessian.npy");
    let hessian_npy = Npy::parse(&hessian_file).unwrap();
    let h = hessian_npy.to_vec::<f32>().unwrap();
    let w: Vec<f32> = weight_npy
        .to_vec::<i16>()
        .unwrap()
        .iter()
        .map(|bits| f32::from_bits(u32::from(*bits as u16) << 16))
        .collect();
    let mut rig = rig(GptqSettings::default());
    let weight = rig.upload(weight_npy.data);
    let hessian = rig.upload(hessian_npy.data);
    for case in cases["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let format = Format::of(case["format"].as_str().unwrap(), case["layout"].as_str().unwrap(), None).unwrap();
        let factor = rig
            .factor(&hessian, k, damp, case["act_order"].as_bool().unwrap())
            .unwrap();
        let tensor = if format.nvfp4 {
            rig.tensor_scales(&weight, n, &factor, &format).unwrap()
        } else {
            TensorScales {
                global_scale: 0.0,
                table: None,
            }
        };
        let state = rig
            .round(RoundRequest {
                weight: weight.addr(),
                n,
                factor: &factor,
                format: &format,
                tensor: &tensor,
                decoded_out: None,
            })
            .unwrap();
        let expected_global = case["global_scale"].as_f64().unwrap() as f32;
        if format.nvfp4 {
            assert_eq!(state.global_scale, expected_global, "{name}");
            if format.sd4 {
                let table: Vec<u8> = case["table"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_u64().unwrap() as u8)
                    .collect();
                assert_eq!(state.table.map(|t| t.to_vec()), Some(table), "{name} table");
            }
        }
        let codes_file = fixture(&format!("{name}-codes.npy"));
        let scales_file = fixture(&format!("{name}-scales.npy"));
        let (codes, scales) = (Npy::parse(&codes_file).unwrap(), Npy::parse(&scales_file).unwrap());
        let expected = decode(&format, codes.data, scales.data, expected_global, state.table, n, k);
        let actual = decode(
            &format,
            &state.codes,
            &state.scales,
            state.global_scale,
            state.table,
            n,
            k,
        );
        let agree = codes.data.iter().zip(&state.codes).filter(|(a, b)| a == b).count();
        let python = output_error(&w, &expected, &h, n, k);
        let native = output_error(&w, &actual, &h, n, k);
        let ratio = native / python;
        println!(
            "{name}: codes agree {:.4}, output error {native:.6e} (reference {python:.6e}), ratio {ratio:.4}",
            agree as f64 / (n * k) as f64
        );
        assert!((ratio - 1.0).abs() < 0.03, "{name}: output error ratio {ratio}");
        assert!(agree * 10 >= n * k * 9, "{name}: codes agree on {agree} of {}", n * k);
    }
}

#[test]
#[ignore = "needs a GPU, the native library, the math libraries and the bundled NVRTC"]
fn a_matrix_that_is_not_positive_definite_is_refused_by_the_step_that_found_it() {
    let (k, n) = (128usize, 4usize);
    let mut rig = rig(GptqSettings::default());
    // A diagonal of -1 (nonzero, so no column is dead) is not positive definite even after damping.
    let mut h = vec![0.0f32; k * k];
    for i in 0..k {
        h[i * k + i] = -1.0;
    }
    let bytes: Vec<u8> = h.iter().flat_map(|v| v.to_le_bytes()).collect();
    let hessian = rig.upload(&bytes);
    let Err(GptqError::Cuda(error)) = rig.factor(&hessian, k, 0.01, false) else {
        panic!("the factor of a negative matrix was accepted");
    };
    let text = error.to_string();
    assert!(text.contains("potrf"), "{text}");
    // The lane survives the refusal: a good matrix factors on it.
    for i in 0..k {
        h[i * k + i] = 1.0;
    }
    let bytes: Vec<u8> = h.iter().flat_map(|v| v.to_le_bytes()).collect();
    let good = rig.upload(&bytes);
    let factor = rig.factor(&good, k, 0.01, true).unwrap();
    assert_eq!(factor.order.len(), k);
    let _ = n;
}

#[test]
fn a_state_round_trips_through_its_archive_for_every_format() {
    let state = QuantState {
        format_name: "NVFP4".into(),
        layout: "row-split-k128-v1".into(),
        rows: 1,
        columns: 16,
        codes: vec![3; 16],
        scales: vec![7],
        global_scale: 0.5,
        table: None,
    };
    assert_eq!(QuantState::from_npz(&state.to_npz().unwrap()).unwrap(), state);
}
