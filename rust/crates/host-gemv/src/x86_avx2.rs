//! The 256-bit x86 kernel, instantiated for AVX2 here (`vpmaddubsw` + `vpmaddwd`) and for AVX-VNNI in
//! [`crate::x86_avxvnni`] (`vpdpbusd`). Both are bit for bit the scalar twin.
//!
//! Per 128-value chunk (eight groups) of a weight row:
//!
//! * 64 code bytes load as two vectors. Masking the low and the high nibbles and looking each up in a 16-entry
//!   `vpshufb` table gives the weights as int8 (`2 * e2m1`), even elements and odd elements apart, which line up with
//!   the de-interleaved activations of [`crate::contract::x_position`].
//! * The weight magnitude is the unsigned operand; the sign moves onto the activation (`vpsignb`), so a product is at
//!   most 12 * 127 and a pair sum at most 3048 (below the int16 saturation point of `vpmaddubsw`).
//! * Two 32-byte dot products (even and odd nibbles) give two int32 lanes per group; one `vphaddd` of the two halves of
//!   the chunk gives one int32 per group in the lane order [g0, g1, g4, g5 | g2, g3, g6, g7].
//! * The coefficients `(t * s) * 0.5` use the same lane order, and one `vfmadd` adds each group into its accumulator
//!   lane, which is `g mod 8` of the contract. The lane order makes the contract's final tree a pair of lane adds.

use std::arch::x86_64::*;
use std::ops::Range;

use crate::contract::{E2M1_X2, E2M1_X2_ABS};
use crate::{Nvfp4View, QuantizedRows};

/// The int8 dot product of two nibble halves: `sum` over 4 consecutive bytes of `a0 * b0 + a1 * b1` per int32 lane,
/// with `a` the unsigned weight magnitudes and `b` the signed activations.
pub(crate) trait Dot {
    /// # Safety
    /// The CPU must support the instruction set of the implementation.
    unsafe fn pair(a0: __m256i, b0: __m256i, a1: __m256i, b1: __m256i) -> __m256i;
}

/// `vpmaddubsw` + `vpmaddwd`. Each product is at most 12 * 127, a pair at most 3048, and the pair sums of both halves
/// added in int16 at most 6096: nothing saturates.
pub(crate) struct Avx2Dot;

impl Dot for Avx2Dot {
    #[inline(always)]
    unsafe fn pair(a0: __m256i, b0: __m256i, a1: __m256i, b1: __m256i) -> __m256i {
        // SAFETY: the caller guarantees AVX2.
        unsafe {
            let sums = _mm256_add_epi16(_mm256_maddubs_epi16(a0, b0), _mm256_maddubs_epi16(a1, b1));
            _mm256_madd_epi16(sums, _mm256_set1_epi16(1))
        }
    }
}

/// The E4M3 scale codes (one per u32 lane) as f32, exact.
#[inline(always)]
pub(crate) unsafe fn e4m3_lanes(codes: __m256i) -> __m256 {
    // SAFETY: the caller guarantees AVX2.
    unsafe {
        let v = _mm256_and_si256(codes, _mm256_set1_epi32(0x7f));
        let e = _mm256_and_si256(_mm256_srli_epi32::<3>(v), _mm256_set1_epi32(15));
        let m = _mm256_and_si256(v, _mm256_set1_epi32(7));
        let normal = _mm256_cmpgt_epi32(e, _mm256_setzero_si256());
        let mantissa = _mm256_or_si256(m, _mm256_and_si256(normal, _mm256_set1_epi32(8)));
        let exponent = _mm256_max_epi32(e, _mm256_set1_epi32(1));
        let power = _mm256_slli_epi32::<23>(_mm256_add_epi32(exponent, _mm256_set1_epi32(117)));
        _mm256_mul_ps(_mm256_cvtepi32_ps(mantissa), _mm256_castsi256_ps(power))
    }
}

/// `t_g = e4m3 * global` for the eight groups of a chunk, in the lane order [g0, g1, g4, g5 | g2, g3, g6, g7].
#[inline(always)]
pub(crate) unsafe fn group_scales(scales: *const u8, global: __m256) -> __m256 {
    // SAFETY: the caller guarantees AVX2 and eight readable bytes at `scales`.
    unsafe {
        let raw = _mm_loadl_epi64(scales.cast());
        let order = _mm_setr_epi8(0, 1, 4, 5, 2, 3, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0);
        let lanes = _mm256_cvtepu8_epi32(_mm_shuffle_epi8(raw, order));
        _mm256_mul_ps(e4m3_lanes(lanes), global)
    }
}

/// The coefficient `(t * s) * 0.5` per lane, from the four activation block scales of the chunk (at `scales`).
#[inline(always)]
pub(crate) unsafe fn coefficients(t: __m256, scales: *const f32) -> __m256 {
    // SAFETY: the caller guarantees AVX2 and four readable floats at `scales`.
    unsafe {
        let blocks = _mm256_castps128_ps256(_mm_loadu_ps(scales));
        let s = _mm256_permutevar8x32_ps(blocks, _mm256_setr_epi32(0, 0, 2, 2, 1, 1, 3, 3));
        _mm256_mul_ps(_mm256_mul_ps(t, s), _mm256_set1_ps(0.5))
    }
}

/// The contract's tree `((a0+a4)+(a2+a6))+((a1+a5)+(a3+a7))` over the lanes [a0, a1, a4, a5 | a2, a3, a6, a7].
#[inline(always)]
pub(crate) unsafe fn reduce(acc: __m256) -> f32 {
    // SAFETY: the caller guarantees AVX2.
    unsafe {
        let lo = _mm256_castps256_ps128(acc);
        let hi = _mm256_extractf128_ps::<1>(acc);
        let lo2 = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
        let hi2 = _mm_add_ps(hi, _mm_movehl_ps(hi, hi));
        let u = _mm_add_ps(lo2, hi2);
        _mm_cvtss_f32(_mm_add_ss(u, _mm_shuffle_ps::<0b01>(u, u)))
    }
}

/// The 16-entry table of `table` replicated into both 128-bit lanes.
#[inline(always)]
unsafe fn table256(table: &[i8; 16]) -> __m256i {
    // SAFETY: the caller guarantees AVX2; the table is 16 readable bytes.
    unsafe { _mm256_broadcastsi128_si256(_mm_loadu_si128(table.as_ptr().cast())) }
}

/// Output rows `rows_out` for every activation row (`y[m * len + (n - start)]`).
///
/// # Safety
/// The CPU must support the instruction set of `D` and AVX2 + FMA. `rows_out` must lie within `w`, and `y` must hold
/// `x.rows() * rows_out.len()` values.
#[inline(always)]
pub(crate) unsafe fn gemv<D: Dot>(w: &Nvfp4View<'_>, x: &QuantizedRows, rows_out: &Range<usize>, y: &mut [f32]) {
    let mut m0 = 0;
    while m0 < x.rows() {
        let mb = (x.rows() - m0).min(8);
        // SAFETY: forwarded from this function's contract.
        unsafe {
            match mb {
                1 => block::<D, 1>(w, x, m0, rows_out, y),
                2 => block::<D, 2>(w, x, m0, rows_out, y),
                3 => block::<D, 3>(w, x, m0, rows_out, y),
                4 => block::<D, 4>(w, x, m0, rows_out, y),
                5 => block::<D, 5>(w, x, m0, rows_out, y),
                6 => block::<D, 6>(w, x, m0, rows_out, y),
                7 => block::<D, 7>(w, x, m0, rows_out, y),
                _ => block::<D, 8>(w, x, m0, rows_out, y),
            }
        }
        m0 += mb;
    }
}

/// `MB` activation rows (from `m0`) against `rows_out`: each weight chunk is decoded once and used for all `MB` rows.
#[inline(always)]
unsafe fn block<D: Dot, const MB: usize>(
    w: &Nvfp4View<'_>,
    x: &QuantizedRows,
    m0: usize,
    rows_out: &Range<usize>,
    y: &mut [f32],
) {
    let len = rows_out.len();
    let chunks = w.chunks();
    // SAFETY: AVX2 + FMA are guaranteed by the caller. Every pointer below stays inside its slice: a weight row has
    // `chunks * 64` code bytes and `chunks * 8` scale bytes, an activation row `chunks * 128` int8 values and
    // `chunks * 4` scales (the views and `quantize_rows` pad to whole chunks), and `rows_out` lies within `w`.
    unsafe {
        let signed = table256(&E2M1_X2);
        let magnitude = table256(&E2M1_X2_ABS);
        let nibble = _mm256_set1_epi8(0x0f);
        let global = _mm256_set1_ps(w.global());
        for (i, n) in rows_out.clone().enumerate() {
            let codes = w.codes().as_ptr().add(n * w.code_stride());
            let scales = w.scales().as_ptr().add(n * w.scale_stride());
            let mut acc = [_mm256_setzero_ps(); MB];
            for c in 0..chunks {
                let b0 = _mm256_loadu_si256(codes.add(c * 64).cast());
                let b1 = _mm256_loadu_si256(codes.add(c * 64 + 32).cast());
                let lo0 = _mm256_and_si256(b0, nibble);
                let hi0 = _mm256_and_si256(_mm256_srli_epi16::<4>(b0), nibble);
                let lo1 = _mm256_and_si256(b1, nibble);
                let hi1 = _mm256_and_si256(_mm256_srli_epi16::<4>(b1), nibble);
                let (sl0, sh0) = (_mm256_shuffle_epi8(signed, lo0), _mm256_shuffle_epi8(signed, hi0));
                let (sl1, sh1) = (_mm256_shuffle_epi8(signed, lo1), _mm256_shuffle_epi8(signed, hi1));
                let (al0, ah0) = (_mm256_shuffle_epi8(magnitude, lo0), _mm256_shuffle_epi8(magnitude, hi0));
                let (al1, ah1) = (_mm256_shuffle_epi8(magnitude, lo1), _mm256_shuffle_epi8(magnitude, hi1));
                let t = group_scales(scales.add(c * 8), global);
                for (j, a) in acc.iter_mut().enumerate() {
                    let m = m0 + j;
                    let xq = x.q().as_ptr().add(m * x.q_stride() + c * 128);
                    let xl0 = _mm256_sign_epi8(_mm256_loadu_si256(xq.cast()), sl0);
                    let xl1 = _mm256_sign_epi8(_mm256_loadu_si256(xq.add(32).cast()), sl1);
                    let xh0 = _mm256_sign_epi8(_mm256_loadu_si256(xq.add(64).cast()), sh0);
                    let xh1 = _mm256_sign_epi8(_mm256_loadu_si256(xq.add(96).cast()), sh1);
                    let e0 = D::pair(al0, xl0, ah0, xh0);
                    let e1 = D::pair(al1, xl1, ah1, xh1);
                    let dots = _mm256_cvtepi32_ps(_mm256_hadd_epi32(e0, e1));
                    let cf = coefficients(t, x.scales().as_ptr().add(m * x.scale_stride() + c * 4));
                    *a = _mm256_fmadd_ps(dots, cf, *a);
                }
            }
            for (j, &a) in acc.iter().enumerate() {
                y[(m0 + j) * len + i] = reduce(a);
            }
        }
    }
}

/// # Safety
/// AVX2 and FMA must be available; see [`gemv`].
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn gemv_avx2(w: &Nvfp4View<'_>, x: &QuantizedRows, rows_out: &Range<usize>, y: &mut [f32]) {
    // SAFETY: forwarded from this function's contract.
    unsafe { gemv::<Avx2Dot>(w, x, rows_out, y) }
}
