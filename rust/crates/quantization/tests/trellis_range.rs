//! The pure parts of the trellis activation-range check, against direct computations on small blocks.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    reason = "a test file: a failed unwrap is the failure, and the values compared are exact by construction"
)]

mod range_support;

use range_support::{BLOCK, Extremes, FP16_MAX, hadamard, half_to_f32, prepass_extremes, report_json, segment_of};
use serde_json::Value;

fn bf16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

/// `H v` with `H[i][j] = (-1)^popcount(i & j)`, summed in float64.
fn direct_hadamard(values: &[f32]) -> Vec<f64> {
    (0..values.len())
        .map(|i| {
            values
                .iter()
                .enumerate()
                .map(|(j, value)| {
                    let sign = if (i & j).count_ones().is_multiple_of(2) {
                        1.0
                    } else {
                        -1.0
                    };
                    sign * f64::from(*value)
                })
                .sum()
        })
        .collect()
}

fn pattern(count: usize, seed: usize) -> Vec<f32> {
    (0..count)
        .map(|i| (((i * 37 + seed * 11) % 23) as f32 - 11.0) / 4.0)
        .collect()
}

#[test]
fn the_transform_is_the_product_with_the_sign_matrix() {
    for size in [2usize, 16, BLOCK] {
        let values = pattern(size, size);
        let mut fast = values.clone();
        hadamard(&mut fast);
        for (at, want) in direct_hadamard(&values).iter().enumerate() {
            assert_eq!(f64::from(fast[at]), *want, "size {size}, entry {at}");
        }
    }
}

/// Two rows of two blocks: each block has its own `suh`, the largest output sits in the second block of the second row
/// and is negative, so a signed maximum, a `suh` that is not applied, one that is read from the wrong block and an
/// output that is not scaled each give another number.
#[test]
fn the_extremes_are_the_scaled_rotation_of_each_block_times_its_suh() {
    let k = 2 * BLOCK;
    let mut x = pattern(2 * k, 3);
    x[k + BLOCK + 5] = -9.0;
    let suh: Vec<f32> = (0..k)
        .map(|i| if i < BLOCK { 1.0 } else { -2.0 + (i % 3) as f32 })
        .collect();
    let got = prepass_extremes("layer.linear", &bf16_bytes(&x), 2, k, &suh);

    let scale = 1.0 / (BLOCK as f64).sqrt();
    let mut want = 0f64;
    for row in x.chunks(k) {
        for (block, signs) in row.chunks(BLOCK).zip(suh.chunks(BLOCK)) {
            let scaled: Vec<f32> = block.iter().zip(signs).map(|(a, b)| a * b).collect();
            for value in direct_hadamard(&scaled) {
                want = want.max((value * scale).abs());
            }
        }
    }
    assert_eq!(got.linear, "layer.linear");
    assert_eq!(f64::from(got.max_prepass), want as f32 as f64);
    assert_eq!(got.max_x, 9.0);
    assert_eq!(got.max_suh, 2.0);
    assert!(got.max_prepass > 0.0 && got.fits_fp16());
}

#[test]
fn an_output_past_fp16_or_not_a_number_does_not_fit() {
    let mut extremes = Extremes {
        linear: "l".to_owned(),
        max_x: 1.0,
        max_suh: 1.0,
        max_prepass: FP16_MAX - 1.0,
    };
    assert!(extremes.fits_fp16());
    extremes.max_prepass = FP16_MAX;
    assert!(
        !extremes.fits_fp16(),
        "the largest FP16 value is the edge: it is not below it"
    );
    extremes.max_prepass = f32::INFINITY;
    assert!(!extremes.fits_fp16());

    // A NaN input makes every output of its block NaN, which the maximum must not drop.
    let mut x = vec![0.0f32; BLOCK];
    x[3] = f32::NAN;
    let found = prepass_extremes("l", &bf16_bytes(&x), 1, BLOCK, &[1.0; BLOCK]);
    assert!(!found.fits_fp16(), "a NaN prepass fits: {found:?}");
    assert!(found.max_x.is_nan(), "a NaN input is dropped from the largest |x|");
}

/// A block whose outputs are all negative still has a largest magnitude: the maximum is of `|output|`.
#[test]
fn the_largest_output_is_taken_by_magnitude() {
    let mut x = vec![0.0f32; BLOCK];
    x[0] = -9.0;
    let found = prepass_extremes("l", &bf16_bytes(&x), 1, BLOCK, &[1.0; BLOCK]);
    let want = (9.0 / (BLOCK as f64).sqrt()) as f32;
    assert!((found.max_prepass - want).abs() < 1e-6, "{found:?}");
}

#[test]
fn a_range_of_rows_belongs_to_the_segment_that_holds_all_of_it() {
    let ends = [6144, 7168];
    assert_eq!(segment_of(&ends, 0, 6144), Ok(0));
    assert_eq!(segment_of(&ends, 6143, 1), Ok(0));
    assert_eq!(segment_of(&ends, 6144, 1024), Ok(1));
    assert!(segment_of(&ends, 6000, 200).unwrap_err().contains("straddle"));
    assert!(segment_of(&ends, 7168, 1).is_err());
}

#[test]
fn half_precision_values_widen_exactly() {
    assert_eq!(half_to_f32(0x3c00), 1.0);
    assert_eq!(half_to_f32(0xc000), -2.0);
    assert_eq!(half_to_f32(0x7bff), FP16_MAX);
    assert_eq!(half_to_f32(0x0001), 2f32.powi(-24));
    assert_eq!(half_to_f32(0x0400), 2f32.powi(-14));
    assert_eq!(half_to_f32(0x7c00), f32::INFINITY);
    assert!(half_to_f32(0x7e00).is_nan());
}

#[test]
fn the_report_holds_each_linear_with_six_significant_digits() {
    let found = [
        Extremes {
            linear: "a.b".to_owned(),
            max_x: 0.5,
            max_suh: 1234567.0,
            max_prepass: 0.000012345678,
        },
        Extremes {
            linear: "c".to_owned(),
            max_x: 0.0,
            max_suh: 100000.0,
            max_prepass: f32::NAN,
        },
    ];
    let text = report_json(&found);
    let parsed: Value = serde_json::from_str(&text).expect("the report is JSON");
    let cases = parsed["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 2);
    assert_eq!(cases[0]["linear"], "a.b");
    assert_eq!(cases[0]["max_abs_x"].as_f64(), Some(0.5));
    assert_eq!(cases[0]["max_abs_suh"].as_f64(), Some(1_234_570.0));
    assert_eq!(cases[0]["max_abs_prepass"].as_f64(), Some(0.0000123457));
    assert_eq!(cases[1]["max_abs_x"].as_f64(), Some(0.0));
    assert_eq!(cases[1]["max_abs_suh"].as_f64(), Some(100_000.0));
    assert!(cases[1]["max_abs_prepass"].is_null());
    assert!(text.contains("1.23457e+06"), "{text}");
}
