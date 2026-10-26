//! Activation quantisation: BF16 rows to int8 blocks of 32 with an f32 scale each.
//!
//! For each 32-element block `s = max|x| / 127` (f32), `q = round_half_even(x / s)` clamped to +-127; a zero block has
//! `s = 0` and `q = 0`. The inputs must be finite. Every step is correctly rounded or exact, so the AVX2 build of the
//! same code (selected at run time) gives the same bits as the portable build.

use crate::contract::{BLOCK, CHUNK, bf16_to_f32, padded_k};

/// Quantised activation rows. The int8 values of a row are stored per 128-value chunk de-interleaved (see
/// [`crate::contract::x_position`]) and padded with zeros to a whole chunk; the storage is private to the crate.
#[derive(Debug, Clone, Default)]
pub struct QuantizedRows {
    rows: usize,
    k: usize,
    q: Vec<i8>,
    scales: Vec<f32>,
}

impl QuantizedRows {
    /// An empty buffer, to be filled by [`quantize_rows`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Activation rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Columns per row.
    pub fn k(&self) -> usize {
        self.k
    }

    /// The scale of block `block` (32 elements) of row `row`.
    pub fn scale(&self, row: usize, block: usize) -> f32 {
        self.scales[row * self.scale_stride() + block]
    }

    /// The int8 value of element `col` of row `row`.
    pub fn value(&self, row: usize, col: usize) -> i8 {
        self.q[row * padded_k(self.k) + crate::contract::x_position(col)]
    }

    pub(crate) fn q(&self) -> &[i8] {
        &self.q
    }

    pub(crate) fn scales(&self) -> &[f32] {
        &self.scales
    }

    /// Int8 values per row in storage.
    pub(crate) fn q_stride(&self) -> usize {
        padded_k(self.k)
    }

    /// Scales per row in storage.
    pub(crate) fn scale_stride(&self) -> usize {
        padded_k(self.k) / BLOCK
    }
}

/// Quantises `rows` rows of `k` BF16 values (`x_bf16`, row-major) into `out`, reusing its buffers.
///
/// # Panics
/// If `k` is not a positive multiple of 32 or `x_bf16.len() != rows * k`.
pub fn quantize_rows(x_bf16: &[u16], rows: usize, k: usize, out: &mut QuantizedRows) {
    assert!(
        k > 0 && k.is_multiple_of(BLOCK),
        "k must be a positive multiple of {BLOCK}"
    );
    assert_eq!(x_bf16.len(), rows * k, "x must hold rows * k values");
    out.rows = rows;
    out.k = k;
    let kp = padded_k(k);
    out.q.clear();
    out.q.resize(rows * kp, 0);
    out.scales.clear();
    out.scales.resize(rows * (kp / BLOCK), 0.0);
    for r in 0..rows {
        let x = &x_bf16[r * k..(r + 1) * k];
        let q = &mut out.q[r * kp..(r + 1) * kp];
        let s = &mut out.scales[r * (kp / BLOCK)..(r + 1) * (kp / BLOCK)];
        quantize_row(x, q, s);
    }
}

fn quantize_row(x: &[u16], q: &mut [i8], s: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 was detected on this CPU.
        unsafe { quantize_row_avx2(x, q, s) };
        return;
    }
    quantize_row_portable(x, q, s);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn quantize_row_avx2(x: &[u16], q: &mut [i8], s: &mut [f32]) {
    quantize_row_generic(x, q, s);
}

fn quantize_row_portable(x: &[u16], q: &mut [i8], s: &mut [f32]) {
    quantize_row_generic(x, q, s);
}

#[inline(always)]
fn quantize_row_generic(x: &[u16], q: &mut [i8], s: &mut [f32]) {
    for (b, block) in x.as_chunks::<BLOCK>().0.iter().enumerate() {
        let mut values = [0.0f32; BLOCK];
        let mut max = 0.0f32;
        for (v, &bits) in values.iter_mut().zip(block) {
            *v = bf16_to_f32(bits);
            max = max.max(v.abs());
        }
        let scale = max / 127.0;
        s[b] = scale;
        if scale == 0.0 {
            continue;
        }
        let mut natural = [0i8; BLOCK];
        for (n, &v) in natural.iter_mut().zip(&values) {
            *n = (v / scale).round_ties_even().clamp(-127.0, 127.0) as i8;
        }
        // De-interleave into the chunk: evens to the first half, odds to the second.
        let chunk = b * BLOCK / CHUNK * CHUNK;
        let in_chunk = b * BLOCK % CHUNK / 2;
        for j in 0..BLOCK / 2 {
            q[chunk + in_chunk + j] = natural[2 * j];
            q[chunk + CHUNK / 2 + in_chunk + j] = natural[2 * j + 1];
        }
    }
}
