//! Every available instruction-set variant equals the scalar twin bit for bit.

mod common;

use common::{Rng, Weights, quantized, random_activations, random_weights};
use euhedral_inference_host_gemv::contract::{MAX_PAIR_SUM, f32_to_bf16};
use euhedral_inference_host_gemv::{Isa, available, detected, gemv_nvfp4, is_available};

fn bits(y: &[f32]) -> Vec<u32> {
    y.iter().map(|v| v.to_bits()).collect()
}

fn run(isa: Isa, w: &Weights, x_bf16: &[u16], m: usize) -> Vec<f32> {
    let q = quantized(x_bf16, m, w.k);
    let mut y = vec![0.0; m * w.n];
    gemv_nvfp4(isa, &w.view(), &q, 0..w.n, &mut y);
    y
}

#[test]
fn scalar_is_always_available_and_detected_is_available() {
    assert!(available().contains(&Isa::Scalar));
    assert!(is_available(detected()));
    assert_eq!(available().first(), Some(&Isa::Scalar));
}

#[test]
fn every_available_isa_matches_scalar_bitwise() {
    let mut rng = Rng::new(7);
    for k in [64usize, 5120, 17408] {
        for n in [1usize, 17, 256] {
            let w = random_weights(&mut rng, n, k);
            let x = random_activations(&mut rng, 8, k);
            // The scalar twin computes all eight rows at once; each M is a prefix of it (row exactness).
            let reference = run(Isa::Scalar, &w, &x, 8);
            assert!(reference.iter().any(|v| *v != 0.0), "degenerate fixture k={k} n={n}");
            for isa in available() {
                for m in 1..=8 {
                    let y = run(isa, &w, &x[..m * k], m);
                    assert_eq!(
                        bits(&y),
                        bits(&reference[..m * n]),
                        "{} differs from the scalar twin at k={k} n={n} m={m}",
                        isa.name()
                    );
                }
            }
        }
    }
}

#[test]
fn m_rows_equal_m_single_rows_bitwise() {
    let mut rng = Rng::new(11);
    for k in [64usize, 1152, 5120] {
        let n = 33;
        let w = random_weights(&mut rng, n, k);
        let x = random_activations(&mut rng, 11, k);
        for isa in available() {
            // Eleven rows exercise a block of eight and a block of three.
            let all = run(isa, &w, &x, 11);
            for m in 0..11 {
                let one = run(isa, &w, &x[m * k..(m + 1) * k], 1);
                assert_eq!(
                    bits(&all[m * n..(m + 1) * n]),
                    bits(&one),
                    "{} row {m} k={k}",
                    isa.name()
                );
            }
        }
    }
}

#[test]
fn chunking_does_not_change_bits() {
    let mut rng = Rng::new(13);
    let (k, n, m) = (2176usize, 97usize, 5usize);
    let w = random_weights(&mut rng, n, k);
    let x = random_activations(&mut rng, m, k);
    let q = quantized(&x, m, k);
    for isa in available() {
        let mut whole = vec![0.0; m * n];
        gemv_nvfp4(isa, &w.view(), &q, 0..n, &mut whole);
        for _ in 0..6 {
            // A random partition of 0..n into consecutive chunks of random sizes.
            let mut y = vec![0.0; m * n];
            let mut start = 0;
            while start < n {
                let end = (start + 1 + rng.below(40)).min(n);
                let len = end - start;
                let mut part = vec![0.0; m * len];
                gemv_nvfp4(isa, &w.view(), &q, start..end, &mut part);
                for r in 0..m {
                    y[r * n + start..r * n + end].copy_from_slice(&part[r * len..(r + 1) * len]);
                }
                start = end;
            }
            assert_eq!(bits(&y), bits(&whole), "{} chunked differs", isa.name());
        }
    }
}

#[test]
fn empty_range_writes_nothing() {
    let mut rng = Rng::new(17);
    let w = random_weights(&mut rng, 4, 128);
    let x = random_activations(&mut rng, 2, 128);
    let q = quantized(&x, 2, 128);
    for isa in available() {
        gemv_nvfp4(isa, &w.view(), &q, 2..2, &mut []);
    }
}

#[test]
fn avx2_pair_sums_never_saturate() {
    // The worst case of the int16 steps: weight magnitude 12 (code 7 or 15) against an activation of magnitude 127, every
    // product of one sign.
    assert!(
        MAX_PAIR_SUM < i32::from(i16::MAX) / 2,
        "the added pair sums must fit int16"
    );
    let k = 256;
    // The sign of the low-nibble (even) and high-nibble (odd) weight of each code byte.
    for (code_byte, even_sign, odd_sign) in [
        (0x77u8, 1.0f32, 1.0f32),
        (0xff, -1.0, -1.0),
        (0x7f, -1.0, 1.0),
        (0xf7, 1.0, -1.0),
    ] {
        for direction in [1.0f32, -1.0] {
            let n = 3;
            let mut rng = Rng::new(19);
            let mut w = random_weights(&mut rng, n, k);
            w.codes.fill(code_byte);
            w.scales.fill(0x38); // 1.0
            w.global = 1.0;
            // Activations of magnitude 1.0 quantise to +-127. Their signs follow the weights', so every product has
            // the same sign (`direction`) and magnitude 12 * 127.
            let x: Vec<u16> = (0..2 * k)
                .map(|i| f32_to_bf16(direction * if i % 2 == 0 { even_sign } else { odd_sign }))
                .collect();
            let reference = run(Isa::Scalar, &w, &x, 2);
            for isa in available() {
                let y = run(isa, &w, &x, 2);
                assert_eq!(
                    bits(&y),
                    bits(&reference),
                    "{} code {code_byte:#x} direction {direction}",
                    isa.name()
                );
            }
            // Per group the dot is exactly 16 * 12 * 127 and the coefficient (1 * (1 / 127)) * 0.5.
            let dot = 16.0 * 12.0 * 127.0 * f64::from(direction);
            let expected = dot * f64::from((1.0f32 / 127.0) * 0.5) * (k / 16) as f64;
            for &v in &reference {
                assert!(
                    (f64::from(v) - expected).abs() <= expected.abs() * 1e-6,
                    "{v} vs {expected}"
                );
            }
        }
    }
}
