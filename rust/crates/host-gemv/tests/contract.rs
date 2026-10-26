//! The pieces of the numeric contract that the twin and bound tests take for granted.

mod common;

use common::{Rng, e4m3, random_weights};
use euhedral_inference_host_gemv::contract::{bf16_to_f32, decode_e4m3, f32_to_bf16, padded_k, x_position};
use euhedral_inference_host_gemv::{LayoutError, Nvfp4View, QuantizedRows, quantize_rows, to_bf16};

fn bf16(v: f32) -> u16 {
    f32_to_bf16(v)
}

#[test]
fn e4m3_decode_is_exact_for_every_code() {
    for code in 0u8..=0x7f {
        assert_eq!(f64::from(decode_e4m3(code)), e4m3(code), "code {code:#x}");
        assert_eq!(decode_e4m3(code), decode_e4m3(code | 0x80), "the sign bit is ignored");
    }
}

#[test]
fn x_position_is_a_permutation_within_each_chunk() {
    let mut seen = vec![false; 384];
    for k in 0..384 {
        let p = x_position(k);
        assert_eq!(p / 128, k / 128);
        assert!(!seen[p]);
        seen[p] = true;
    }
    assert_eq!(
        (x_position(0), x_position(1), x_position(2), x_position(127)),
        (0, 64, 1, 127)
    );
}

#[test]
fn quantisation_follows_the_contract() {
    // Block 0 has max 127 (scale 1), so values are their own step counts and halves round to even.
    let mut x = vec![bf16(0.0); 96];
    x[0] = bf16(127.0); // s = 1
    x[1] = bf16(2.5); // 2.5 -> 2
    x[2] = bf16(3.5); // 3.5 -> 4
    x[3] = bf16(-2.5); // -2 (half to even)
    x[4] = bf16(-0.5); // -0
    // Block 1 is all zero: scale 0, values 0.
    // Block 2: one spike and small values.
    x[64] = bf16(-254.0); // s = 2
    x[65] = bf16(3.0); // 1.5 -> 2
    x[66] = bf16(5.0); // 2.5 -> 2
    let mut q = QuantizedRows::new();
    quantize_rows(&x, 1, 96, &mut q);
    assert_eq!((q.rows(), q.k()), (1, 96));
    assert_eq!(q.scale(0, 0), 1.0);
    assert_eq!([0, 1, 2, 3, 4].map(|i| q.value(0, i)), [127, 2, 4, -2, 0]);
    assert_eq!(q.scale(0, 1), 0.0);
    assert!((32..64).all(|i| q.value(0, i) == 0));
    assert_eq!(q.scale(0, 2), 2.0);
    assert_eq!([64, 65, 66].map(|i| q.value(0, i)), [-127, 2, 2]);
    // Reusing the buffer for fewer rows leaves no stale state.
    quantize_rows(&x[..32], 1, 32, &mut q);
    assert_eq!(q.scale(0, 0), 1.0);
}

#[test]
fn quantisation_error_is_half_a_step() {
    let mut rng = Rng::new(5);
    let k = 320;
    let x: Vec<u16> = (0..2 * k).map(|_| bf16(rng.unit() * 37.0)).collect();
    let mut q = QuantizedRows::new();
    quantize_rows(&x, 2, k, &mut q);
    for r in 0..2 {
        for c in 0..k {
            let s = q.scale(r, c / 32);
            let error = (bf16_to_f32(x[r * k + c]) - f32::from(q.value(r, c)) * s).abs();
            assert!(error <= s * 0.5 * (1.0 + 1e-6), "row {r} col {c}");
        }
    }
}

#[test]
fn bf16_conversion_rounds_to_nearest_even() {
    assert_eq!(f32_to_bf16(1.0), 0x3f80);
    // Exactly halfway between two BF16 values rounds to the even one.
    assert_eq!(f32_to_bf16(f32::from_bits(0x3f80_8000)), 0x3f80);
    assert_eq!(f32_to_bf16(f32::from_bits(0x3f81_8000)), 0x3f82);
    assert_eq!(f32_to_bf16(f32::from_bits(0x3f80_8001)), 0x3f81);
    let y = [1.0f32, -2.5, 0.0, 3.3333333];
    let mut out = [0u16; 4];
    to_bf16(&y, &mut out);
    assert_eq!(out, y.map(f32_to_bf16));
}

#[test]
fn row_split_payload_parses_planes_and_global() {
    let mut rng = Rng::new(3);
    let (n, k) = (5usize, 96usize);
    let w = random_weights(&mut rng, n, k);
    let kp = padded_k(k);
    let scale_offset = (n * kp / 2).div_ceil(256) * 256;
    let global_offset = (scale_offset + n * kp / 16).div_ceil(256) * 256;
    let mut payload = vec![0u8; global_offset + 4];
    payload[..n * kp / 2].copy_from_slice(&w.codes);
    payload[scale_offset..scale_offset + n * kp / 16].copy_from_slice(&w.scales);
    payload[global_offset..].copy_from_slice(&w.global.to_le_bytes());
    let view = Nvfp4View::from_row_split(&payload, n, k).expect("valid payload");
    assert_eq!((view.n(), view.k(), view.global()), (n, k, w.global));
    assert_eq!(view.plane_bytes(), n * (kp / 2 + kp / 16));
    assert_eq!(
        Nvfp4View::from_row_split(&payload[..payload.len() - 1], n, k).unwrap_err(),
        LayoutError::TooShort {
            what: "payload",
            need: payload.len(),
            have: payload.len() - 1
        }
    );
    assert_eq!(
        Nvfp4View::from_row_split(&payload, n, 100).unwrap_err(),
        LayoutError::Width(100)
    );
    assert!(Nvfp4View::new(&w.codes[..10], &w.scales, 1.0, n, k).is_err());
}

#[test]
fn views_reject_what_the_java_validation_rejects() {
    let mut rng = Rng::new(9);
    let (n, k) = (3usize, 160usize);
    let w = random_weights(&mut rng, n, k);
    let new = |scales: &[u8], global: f32| Nvfp4View::new(&w.codes, scales, global, n, k).map(|_| ());
    assert_eq!(new(&w.scales, w.global), Ok(()));
    // The global scale: NaN, infinite, negative, and so large that e4m3 * global overflows.
    for global in [
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        -1.0,
        -f32::MIN_POSITIVE,
        f32::MAX,
    ] {
        assert!(
            matches!(new(&w.scales, global), Err(LayoutError::Global(_))),
            "global {global}"
        );
    }
    assert_eq!(new(&w.scales, 0.0), Ok(()));
    // Both NaN scale codes, in the first byte, in the middle and in the padding of the last row.
    for code in [0x7fu8, 0xff] {
        for offset in [0, 7, n * padded_k(k) / 16 - 1] {
            let mut scales = w.scales.clone();
            scales[offset] = code;
            assert_eq!(
                new(&scales, w.global),
                Err(LayoutError::ScaleNan { offset }),
                "code {code:#x} at {offset}"
            );
        }
    }
    // The same checks through the payload parser.
    let kp = padded_k(k);
    let scale_offset = (n * kp / 2).div_ceil(256) * 256;
    let global_offset = (scale_offset + n * kp / 16).div_ceil(256) * 256;
    let mut payload = vec![0u8; global_offset + 4];
    payload[..n * kp / 2].copy_from_slice(&w.codes);
    payload[scale_offset..scale_offset + n * kp / 16].copy_from_slice(&w.scales);
    payload[global_offset..].copy_from_slice(&w.global.to_le_bytes());
    assert!(Nvfp4View::from_row_split(&payload, n, k).is_ok());
    payload[global_offset..].copy_from_slice(&f32::NAN.to_le_bytes());
    assert!(matches!(
        Nvfp4View::from_row_split(&payload, n, k),
        Err(LayoutError::Global(_))
    ));
    payload[global_offset..].copy_from_slice(&w.global.to_le_bytes());
    payload[scale_offset + 5] = 0x7f;
    assert_eq!(
        Nvfp4View::from_row_split(&payload, n, k).unwrap_err(),
        LayoutError::ScaleNan { offset: 5 }
    );
}
