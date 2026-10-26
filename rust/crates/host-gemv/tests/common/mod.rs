//! Shared fixtures: a seeded generator and random NVFP4 weights and BF16 activations that include zero blocks and the
//! extreme values.
#![allow(dead_code)]

use euhedral_inference_host_gemv::contract::{CHUNK, GROUP, f32_to_bf16, padded_k};
use euhedral_inference_host_gemv::{Nvfp4View, QuantizedRows, quantize_rows};

/// xorshift64*: small, seeded, and the same on every machine.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    /// Uniform in [-1, 1).
    pub fn unit(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / (1u64 << 23) as f32 - 1.0
    }
}

/// Owned weight planes in the row-split layout (rows padded to 128 values).
pub struct Weights {
    pub codes: Vec<u8>,
    pub scales: Vec<u8>,
    pub global: f32,
    pub n: usize,
    pub k: usize,
}

impl Weights {
    pub fn view(&self) -> Nvfp4View<'_> {
        Nvfp4View::new(&self.codes, &self.scales, self.global, self.n, self.k).expect("valid view")
    }

    /// The dequantised weight `e2m1 * e4m3 * global` of element `col` of row `row`, in f64.
    pub fn value(&self, row: usize, col: usize) -> f64 {
        let kp = padded_k(self.k);
        let byte = self.codes[row * kp / 2 + col / 2];
        let code = if col.is_multiple_of(2) { byte & 15 } else { byte >> 4 };
        let magnitude = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0][usize::from(code & 7)];
        let e2m1 = if code & 8 != 0 { -magnitude } else { magnitude };
        let scale = e4m3(self.scales[row * kp / GROUP + col / GROUP]);
        e2m1 * scale * f64::from(self.global)
    }
}

/// An independent E4M3 decode in f64 (no sign bit).
pub fn e4m3(code: u8) -> f64 {
    let e = i32::from((code >> 3) & 15);
    let m = f64::from(code & 7);
    if e == 0 {
        m / 8.0 * 2f64.powi(-6)
    } else {
        (1.0 + m / 8.0) * 2f64.powi(e - 7)
    }
}

/// A valid scale byte: never NaN (0x7f) and no sign bit.
fn scale_byte(rng: &mut Rng) -> u8 {
    match rng.below(8) {
        0 => 0,
        1 => 0x7e,
        2 => 1 + rng.below(7) as u8,
        _ => (rng.next_u64() as u8 & 0x7f).min(0x7e),
    }
}

pub fn random_weights(rng: &mut Rng, n: usize, k: usize) -> Weights {
    let kp = padded_k(k);
    let mut codes = vec![0u8; n * kp / 2];
    let mut scales = vec![0u8; n * kp / GROUP];
    for row in 0..n {
        for group in 0..kp / GROUP {
            let bytes = &mut codes[row * kp / 2 + group * 8..][..8];
            match rng.below(10) {
                0 => bytes.fill(0x00),
                1 => bytes.fill([0x77, 0xff, 0x7f, 0xf7][rng.below(4)]),
                _ => bytes.iter_mut().for_each(|b| *b = rng.next_u64() as u8),
            }
            // The padding past k carries arbitrary valid values; the result must not depend on them.
            scales[row * kp / GROUP + group] = if rng.below(12) == 0 { 0 } else { scale_byte(rng) };
        }
    }
    let global = [0.25, 1.0, 0.0371, 3.5, 0.0009765625][rng.below(5)];
    Weights {
        codes,
        scales,
        global,
        n,
        k,
    }
}

/// `rows` rows of `k` BF16 activations: ordinary values, zero blocks, blocks of identical extremes and single spikes.
pub fn random_activations(rng: &mut Rng, rows: usize, k: usize) -> Vec<u16> {
    let mut x = vec![0u16; rows * k];
    for row in 0..rows {
        for block in 0..k / 32 {
            let range = [0.01f32, 1.0, 40.0, 3000.0][rng.below(4)];
            let values = &mut x[row * k + block * 32..][..32];
            match rng.below(10) {
                0 => {}
                1 => {
                    let sign = if rng.below(2) == 0 { 1.0 } else { -1.0 };
                    values.fill(f32_to_bf16(sign * range));
                }
                2 => {
                    values
                        .iter_mut()
                        .for_each(|v| *v = f32_to_bf16(rng.unit() * range * 1e-3));
                    values[rng.below(32)] = f32_to_bf16(range);
                }
                _ => values.iter_mut().for_each(|v| *v = f32_to_bf16(rng.unit() * range)),
            }
        }
    }
    x
}

pub fn quantized(x: &[u16], rows: usize, k: usize) -> QuantizedRows {
    let mut q = QuantizedRows::new();
    quantize_rows(x, rows, k, &mut q);
    q
}

/// The padding granule, for tests that size buffers.
pub const PAD: usize = CHUNK;
