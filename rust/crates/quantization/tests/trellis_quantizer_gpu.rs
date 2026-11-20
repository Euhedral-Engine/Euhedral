//! The trellis quantizer on the device: the fallbacks, the damping retries and what a quantization must hold.
#![cfg(target_os = "linux")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::disallowed_methods,
    reason = "a test: a failed unwrap is the failure"
)]

mod trellis_rig;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use euhedral_inference_quantization::trellis::{CHOLESKY_RETRIES, QuantizeError, Rate, TrellisSettings};
use trellis_rig::{Rig, bf16, f16, gaussians, halves, hessian, request, rig, to_bf16};

const N: usize = 128;
const K: usize = 128;

fn weights(seed: u64) -> Vec<u16> {
    gaussians(N * K, seed).into_iter().map(to_bf16).collect()
}

/// The root-mean-square of `decoded - weight` over the root-mean-square of the weight.
fn relative_error(rig: &Rig, decoded: &euhedral_inference_quantization::memory::Device, weight: &[u16]) -> f64 {
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

/// A Hessian whose leading rows and columns are zero (inputs that never fired) is singular. The damping applied before
/// the first attempt makes it factor at once; a mutant without that damping needs a retry (one retry's extra damping
/// would also rescue it, so the result alone could not tell).
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
    assert_eq!(
        out.cholesky_retries, 0,
        "the initial damping alone makes it positive definite"
    );
    assert_eq!(out.refit_failures, 0);
    assert!(all_finite(&out.suh) && all_finite(&out.svh));
    assert!(relative_error(&rig, &out.decoded, &weight) < 0.4);
}

/// A mildly indefinite Hessian fails the first factorization and succeeds after retries with more damping; a hopeless
/// one gives up after the cap and names it. A mutant that never retries, or that restores the matrix wrongly between
/// retries, fails the first; one that retries forever hangs the second (the pump's deadline fails it).
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
    // after the initial damping its smallest eigenvalue is about -0.16 (the mean diagonal is 1); three retries make
    // it positive definite
    mild[5 * K + 5] -= mean as f32;
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
    let run = |tiles: u64, seed: u64| {
        let scratch = if tiles == 0 {
            TrellisSettings::default().viterbi_scratch_bytes
        } else {
            Rate::new(2.5).unwrap().scratch_bytes(tiles)
        };
        let mut rig = rig(TrellisSettings {
            viterbi_scratch_bytes: scratch,
        });
        let w = rig.upload_halves(&weight);
        let h = rig.upload_floats(&values);
        let mut r = request(&w, Some(&h), N as u32, K as u32, 512, 2.5);
        r.seed = seed;
        rig.quantize(&r).unwrap()
    };
    let whole = run(0, 0);
    assert!(whole.trellis.iter().any(|word| *word != 0));
    for tiles in 1..=3 {
        let chunked = run(tiles, 0);
        assert_eq!(whole.trellis, chunked.trellis, "{tiles} tiles");
        assert_eq!(whole.suh, chunked.suh, "{tiles} tiles");
        assert_eq!(whole.svh, chunked.svh, "{tiles} tiles");
    }
    let other = run(0, 1);
    assert_ne!(whole.suh, other.suh);
}

/// `X^T X / rows` over rows whose features share a component within each group of eight and differ in scale: a
/// Hessian the rounding errors can be steered by.
fn correlated_hessian(rows: usize, seed: u64) -> Vec<f32> {
    let noise = gaussians(rows * K, seed);
    let shared = gaussians(rows * (K / 8), seed + 1000);
    let scale = |j: usize| 0.3 * 1.03f32.powi(j as i32);
    let mut x = vec![0f32; rows * K];
    for r in 0..rows {
        for j in 0..K {
            x[r * K + j] = scale(j) * (0.5 * noise[r * K + j] + 0.85 * shared[r * (K / 8) + j / 8]);
        }
    }
    let mut h = vec![0f32; K * K];
    for r in 0..rows {
        let row = &x[r * K..(r + 1) * K];
        for i in 0..K {
            for j in 0..K {
                h[i * K + j] += row[i] * row[j] / rows as f32;
            }
        }
    }
    h
}

/// `tr(E H Eᵀ) / tr(W H Wᵀ)` for the error `E = decoded - weight`, in double precision on the host.
fn hessian_error(
    rig: &Rig,
    decoded: &euhedral_inference_quantization::memory::Device,
    weight: &[u16],
    h: &[f32],
) -> f64 {
    let decoded = halves(&rig.download(decoded, N * K * 2));
    let quadratic = |row: &[f64]| {
        let mut sum = 0f64;
        for i in 0..K {
            let mut inner = 0f64;
            for j in 0..K {
                inner += f64::from(h[i * K + j]) * row[j];
            }
            sum += row[i] * inner;
        }
        sum
    };
    let (mut error, mut energy) = (0f64, 0f64);
    for r in 0..N {
        let w: Vec<f64> = (0..K).map(|j| f64::from(bf16(weight[r * K + j]))).collect();
        let e: Vec<f64> = (0..K).map(|j| f64::from(bf16(decoded[r * K + j])) - w[j]).collect();
        error += quadratic(&e);
        energy += quadratic(&w);
    }
    error / energy
}

/// Under a correlated Hessian the error in its metric, measured on the host from the decoded weight, matches the
/// quantizer's own proxy error and is clearly below what the plain path gives on the same weight. A mutant that drops
/// the feedback of the rounding errors (a strip error launched over one row), skips the scale refit or fixes the global
/// scale at one leaves the error above the bound.
#[test]
#[ignore = "needs the graphics card"]
fn the_error_in_the_hessian_metric_is_what_the_quantizer_reports_and_beats_the_plain_path() {
    let mut rig = rig(TrellisSettings::default());
    let weight = weights(12);
    let values = correlated_hessian(512, 13);
    let w = rig.upload_halves(&weight);
    let h = rig.upload_floats(&values);
    let calibrated = rig
        .quantize(&request(&w, Some(&h), N as u32, K as u32, 512, 3.0))
        .unwrap();
    let plain = rig.quantize(&request(&w, None, N as u32, K as u32, 0, 3.0)).unwrap();
    let measured = hessian_error(&rig, &calibrated.decoded, &weight, &values);
    let plain_measured = hessian_error(&rig, &plain.decoded, &weight, &values);
    let relative = (measured - calibrated.proxy_error).abs() / calibrated.proxy_error;
    assert!(
        relative < 0.05,
        "measured {measured}, reported {}",
        calibrated.proxy_error
    );
    assert!(
        measured < 0.4 * plain_measured,
        "{measured} against plain {plain_measured}"
    );
    // this input gives 0.0040 with the scale search and the refit; without the refit it gives 0.0045 and with the
    // global scale fixed at one 0.0052
    assert!(measured < 0.0043, "{measured}");
}

/// A quantization takes exactly the bytes it says it takes, charges all of them, and gives them back once its result
/// is dropped. A mutant that allocates outside the claim, or never frees the result, leaves the planner short.
#[test]
#[ignore = "needs the graphics card"]
fn a_quantization_takes_the_bytes_it_charges_and_gives_them_back() {
    let mut rig = rig(TrellisSettings::default());
    let capacity = rig.capacity();
    let w = rig.upload_halves(&weights(5));
    let h = rig.upload_floats(&hessian(K, 512, 6));
    let out = rig
        .quantize(&request(&w, Some(&h), N as u32, K as u32, 512, 3.0))
        .unwrap();
    // The decoded weight is the only buffer that outlives the step; its claim is still held.
    let held = rig.free();
    assert!(held.0 < capacity, "the result's device bytes are no longer charged");
    assert_eq!(held.1, capacity, "staging stayed charged after the step");
    drop(out);
    rig.settle();
    assert_eq!(rig.free(), (capacity, capacity));
}

/// A step refused for a short claim, of either kind, allocates nothing and charges nothing. A mutant that allocates
/// before it charges leaves the pool or the planner holding bytes of a step that never ran.
#[test]
#[ignore = "needs the graphics card"]
fn a_step_refused_for_a_short_claim_allocates_nothing() {
    let mut rig = rig(TrellisSettings::default());
    let capacity = rig.capacity();
    let w = rig.upload_halves(&weights(7));
    let h = rig.upload_floats(&hessian(K, 512, 8));
    let asked = request(&w, Some(&h), N as u32, K as u32, 512, 3.0);
    for short in [(1, 0), (0, 1), (u64::MAX, 0), (0, u64::MAX)] {
        rig.short = short;
        let refused = rig.quantize(&asked);
        assert!(
            matches!(&refused, Err(QuantizeError::Request(text)) if text.contains("claim")),
            "a claim short by {short:?} was accepted"
        );
        rig.settle();
        assert_eq!(rig.free(), (capacity, capacity), "{short:?}: bytes stayed charged");
        assert_eq!(
            rig.pinned.allocated_bytes(),
            0,
            "{short:?}: the pool allocated for a refused step"
        );
    }
    rig.short = (0, 0);
    rig.quantize(&asked).unwrap();
}

/// A step given no claims is refused before it does anything.
#[test]
#[ignore = "needs the graphics card"]
fn a_step_given_no_claims_is_refused() {
    let mut rig = rig(TrellisSettings::default());
    let w = rig.upload_halves(&weights(9));
    let refused = rig.quantize_with(
        &request(&w, None, N as u32, K as u32, 0, 3.0),
        euhedral_inference_quantization::memory::Claims::default(),
    );
    assert!(matches!(&refused, Err(QuantizeError::Request(text)) if text.contains("claim")));
    assert_eq!(rig.pinned.allocated_bytes(), 0);
}

/// A chain that cannot arm its retirement keeps its buffers parked until a later retirement is confirmed, because
/// work of its may still be queued on the stream. A mutant that frees them at the failure hands queued copies back
/// to the pool.
#[test]
#[ignore = "needs the graphics card"]
fn a_chain_that_cannot_arm_keeps_its_staging_until_a_retirement_is_confirmed() {
    let mut rig = rig(TrellisSettings::default());
    let w = rig.upload_halves(&weights(11));
    let h = rig.upload_floats(&hessian(K, 512, 12));
    let asked = request(&w, Some(&h), N as u32, K as u32, 512, 3.0);
    assert_eq!(rig.pinned.allocated_bytes(), 0);
    rig.block_arming();
    let failed = rig.quantize(&asked);
    assert!(matches!(failed, Err(QuantizeError::Gpu(_))), "{:?}", failed.map(|_| ()));
    assert_eq!(rig.unsettled(), 1);
    assert!(rig.pinned.allocated_bytes() > 0, "the chain took staging");
    assert_eq!(
        rig.pinned.held_bytes(),
        0,
        "the staging went back to the pool while its work was still queued"
    );
    rig.unblock_arming();
    rig.quantize(&asked).unwrap();
    assert_eq!(rig.unsettled(), 0);
    assert!(rig.pinned.held_bytes() > 0, "the staging came back to the pool");
}

/// Shutdown waits for the stream before it releases a failed chain's buffers. A mutant that releases first hands the
/// pool staging the device may still be writing.
#[test]
#[ignore = "needs the graphics card"]
fn shutdown_waits_for_the_stream_before_it_releases_a_failed_chains_staging() {
    let mut rig = rig(TrellisSettings::default());
    let w = rig.upload_halves(&weights(13));
    let h = rig.upload_floats(&hessian(K, 512, 14));
    rig.block_arming();
    let failed = rig.quantize(&request(&w, Some(&h), N as u32, K as u32, 512, 3.0));
    assert!(matches!(failed, Err(QuantizeError::Gpu(_))), "{:?}", failed.map(|_| ()));
    assert_eq!(rig.unsettled(), 1);
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
    assert_eq!(calls.load(Ordering::SeqCst), 1, "shutdown did not wait for the stream");
    assert!(
        rig.pinned.held_bytes() > 0,
        "the staging came back to the pool after the wait"
    );
}
