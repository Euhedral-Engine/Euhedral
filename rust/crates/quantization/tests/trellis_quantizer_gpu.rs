//! The trellis quantizer on the device: the fallbacks, the damping retries and what a quantization must hold.
#![cfg(target_os = "linux")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::disallowed_methods,
    reason = "a test: a failed unwrap is the failure"
)]

mod trellis_rig;

use euhedral_inference_quantization::trellis::{CHOLESKY_RETRIES, QuantizeError, TrellisSettings};
use trellis_rig::{Rig, bf16, f16, gaussians, halves, hessian, request, rig, to_bf16};

const N: usize = 128;
const K: usize = 128;

fn weights(seed: u64) -> Vec<u16> {
    gaussians(N * K, seed).into_iter().map(to_bf16).collect()
}

/// The root-mean-square of `decoded - weight` over the root-mean-square of the weight.
fn relative_error(rig: &Rig, decoded: &euhedral_inference_cuda::DeviceBuffer, weight: &[u16]) -> f64 {
    let decoded = halves(&rig.download(decoded, N * K * 2));
    let (mut error, mut energy) = (0f64, 0f64);
    for (d, w) in decoded.iter().zip(weight) {
        let (d, w) = (f64::from(bf16(*d)), f64::from(bf16(*w)));
        error += (d - w) * (d - w);
        energy += w * w;
    }
    (error / energy).sqrt()
}

fn all_finite(words: &[u16]) -> bool {
    words.iter().all(|w| f16(*w).is_finite())
}

/// With a Hessian, the result is a faithful 3-bit decoding: the input-side scales are finite and the error is small.
/// A mutant that decodes in the Hessian's rotated basis, or skips the refit, leaves the error above the bound.
#[test]
#[ignore = "needs the graphics card"]
fn a_calibrated_quantization_decodes_close_to_the_weight() {
    let mut rig = rig(TrellisSettings::default());
    let weight = weights(1);
    let w = rig.upload_halves(&weight);
    let h = rig.upload_floats(&hessian(K, 512, 2));
    let out = rig
        .quantize(&request(&w, Some(&h), N as u32, K as u32, 512, 3.0))
        .unwrap();
    assert!(all_finite(&out.suh) && all_finite(&out.svh));
    assert_eq!(out.refit_failures, 0);
    assert!(
        out.proxy_error.is_finite() && out.proxy_error > 0.0,
        "{}",
        out.proxy_error
    );
    let error = relative_error(&rig, &out.decoded, &weight);
    assert!(error < 0.35 && error > 0.01, "relative error {error}");
}

/// A request without a Hessian, or with no calibration rows, is quantized as if the Hessian were the identity.
/// A mutant that reads the missing Hessian, or divides by the zero count, gives non-finite scales.
#[test]
#[ignore = "needs the graphics card"]
fn without_a_hessian_or_rows_the_weight_is_quantized_plainly() {
    let mut rig = rig(TrellisSettings::default());
    let weight = weights(3);
    let w = rig.upload_halves(&weight);
    let none = rig.quantize(&request(&w, None, N as u32, K as u32, 512, 3.0)).unwrap();
    let h = rig.upload_floats(&hessian(K, 512, 4));
    let zero = rig
        .quantize(&request(&w, Some(&h), N as u32, K as u32, 0, 3.0))
        .unwrap();
    for out in [&none, &zero] {
        assert!(all_finite(&out.suh) && all_finite(&out.svh));
        assert!(relative_error(&rig, &out.decoded, &weight) < 0.35);
    }
    assert_eq!(none.trellis, zero.trellis, "both take the same path");
    assert_eq!(none.suh, zero.suh);
}

/// A Hessian that is all zeros has no usable diagonal: the weight is quantized plainly, with no failed refit.
/// A mutant that factors it anyway fails every Cholesky attempt.
#[test]
#[ignore = "needs the graphics card"]
fn a_zero_hessian_falls_back_to_the_plain_path() {
    let mut rig = rig(TrellisSettings::default());
    let weight = weights(5);
    let w = rig.upload_halves(&weight);
    let h = rig.upload_floats(&vec![0f32; K * K]);
    let out = rig
        .quantize(&request(&w, Some(&h), N as u32, K as u32, 100, 3.0))
        .unwrap();
    assert_eq!(out.refit_failures, 0);
    assert!(all_finite(&out.suh) && all_finite(&out.svh));
    assert!(relative_error(&rig, &out.decoded, &weight) < 0.35);
}

/// A Hessian whose leading rows and columns are zero (inputs that never fired) is still factored, after damping.
/// A mutant that damps only when the factorization fails would fail on the first attempt and run the whole cap.
#[test]
#[ignore = "needs the graphics card"]
fn a_zeroed_leading_block_is_damped_and_quantized() {
    let mut rig = rig(TrellisSettings::default());
    let weight = weights(6);
    let w = rig.upload_halves(&weight);
    let mut values = hessian(K, 512, 7);
    for i in 0..K {
        for j in 0..K {
            if i < 16 || j < 16 {
                values[i * K + j] = 0.0;
            }
        }
    }
    let h = rig.upload_floats(&values);
    let out = rig
        .quantize(&request(&w, Some(&h), N as u32, K as u32, 512, 3.0))
        .unwrap();
    assert_eq!(out.refit_failures, 0);
    assert!(all_finite(&out.suh) && all_finite(&out.svh));
    assert!(relative_error(&rig, &out.decoded, &weight) < 0.4);
}

/// A mildly indefinite Hessian fails the first factorization and succeeds with more damping; a hopeless one gives up
/// after the cap and names it. A mutant that never retries fails the first; one that retries forever hangs the second
/// (the pump's deadline fails it).
#[test]
#[ignore = "needs the graphics card"]
fn an_indefinite_hessian_retries_with_more_damping_then_gives_up() {
    let mut rig = rig(TrellisSettings::default());
    let weight = weights(8);
    let w = rig.upload_halves(&weight);
    let mean = {
        let h = hessian(K, 512, 9);
        (0..K).map(|i| f64::from(h[i * K + i])).sum::<f64>() / K as f64
    };
    let mut mild = hessian(K, 512, 9);
    mild[5 * K + 5] -= (0.6 * mean) as f32;
    let h = rig.upload_floats(&mild);
    let out = rig
        .quantize(&request(&w, Some(&h), N as u32, K as u32, 512, 3.0))
        .unwrap();
    assert!(all_finite(&out.suh) && all_finite(&out.svh));

    let mut hopeless = hessian(K, 512, 9);
    hopeless[5 * K + 5] -= (100.0 * mean) as f32;
    let h = rig.upload_floats(&hopeless);
    let Err(error) = rig.quantize(&request(&w, Some(&h), N as u32, K as u32, 512, 3.0)) else {
        panic!("a hopeless Hessian was quantized");
    };
    assert!(
        matches!(error, QuantizeError::NotPositiveDefinite { retries } if retries == CHOLESKY_RETRIES),
        "{error:?}"
    );
    // the lane is usable again after a failure
    let again = rig.quantize(&request(&w, None, N as u32, K as u32, 0, 3.0)).unwrap();
    assert!(all_finite(&again.suh));
}

/// Two requests with the same inputs and seed give the same words and scales, whatever the chunking of the Viterbi
/// scratch; another seed gives other signs. A mutant that ignores the seed, or whose chunks change a result, fails.
#[test]
#[ignore = "needs the graphics card"]
fn the_seed_picks_the_signs_and_the_scratch_size_changes_nothing() {
    let weight = weights(10);
    let values = hessian(K, 512, 11);
    let run = |scratch: u64, seed: u64| {
        let mut rig = rig(TrellisSettings {
            viterbi_scratch_bytes: scratch,
        });
        let w = rig.upload_halves(&weight);
        let h = rig.upload_floats(&values);
        let mut r = request(&w, Some(&h), N as u32, K as u32, 512, 2.5);
        r.seed = seed;
        rig.quantize(&r).unwrap()
    };
    let big = run(256 << 20, 0);
    // room for one tile at a time
    let small = run(9 << 20, 0);
    assert_eq!(big.trellis, small.trellis);
    assert_eq!(big.suh, small.suh);
    assert_eq!(big.svh, small.svh);
    let other = run(256 << 20, 1);
    assert_ne!(big.suh, other.suh);
}
