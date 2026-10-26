//! The error of the integer-activation GEMV against an f64 reference is within the contract's bound.

mod common;

use common::{Rng, quantized, random_activations, random_weights};
use euhedral_inference_host_gemv::contract::bf16_to_f32;
use euhedral_inference_host_gemv::{available, gemv_nvfp4};

/// The spacing of f32 numbers at `v`.
fn ulp_f32(v: f64) -> f64 {
    let a = (v.abs() as f32).max(f32::MIN_POSITIVE);
    f64::from(f32::from_bits(a.to_bits() + 1)) - f64::from(a)
}

#[test]
fn error_within_contract_bound() {
    let mut rng = Rng::new(23);
    let mut worst_relative = 0.0f64;
    for (k, n, m) in [(64usize, 40usize, 3usize), (5120, 24, 3), (17408, 8, 2)] {
        let w = random_weights(&mut rng, n, k);
        let x = random_activations(&mut rng, m, k);
        let q = quantized(&x, m, k);
        for isa in available() {
            let mut y = vec![0.0; m * n];
            gemv_nvfp4(isa, &w.view(), &q, 0..n, &mut y);
            for r in 0..m {
                let xr: Vec<f64> = x[r * k..(r + 1) * k]
                    .iter()
                    .map(|&b| f64::from(bf16_to_f32(b)))
                    .collect();
                let mut norm = 0.0f64;
                let mut worst_row = 0.0f64;
                for row in 0..n {
                    let (mut exact, mut magnitude, mut bound) = (0.0f64, 0.0f64, 0.0f64);
                    for block in 0..k / 32 {
                        let (mut max_x, mut sum_w) = (0.0f64, 0.0f64);
                        for (i, &xi) in xr.iter().enumerate().skip(block * 32).take(32) {
                            let weight = w.value(row, i);
                            exact += weight * xi;
                            magnitude += (weight * xi).abs();
                            max_x = max_x.max(xi.abs());
                            sum_w += weight.abs();
                        }
                        bound += max_x / 254.0 * sum_w;
                    }
                    bound += 4.0 * ulp_f32(magnitude);
                    let got = f64::from(y[r * n + row]);
                    let error = (got - exact).abs();
                    assert!(
                        error <= bound,
                        "{} k={k} row {row}/{r}: error {error} exceeds the bound {bound} (exact {exact}, got {got})",
                        isa.name()
                    );
                    norm += exact * exact;
                    worst_row = worst_row.max(error);
                }
                if norm > 0.0 {
                    worst_relative = worst_relative.max(worst_row / norm.sqrt());
                }
            }
        }
    }
    eprintln!("max error relative to the output vector norm: {worst_relative:.3e}");
}
