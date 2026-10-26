//! The AVX-512 VNNI kernel: the data path of [`crate::x86_avx2`] on 512-bit vectors. One vector holds a whole chunk's
//! 64 code bytes, so the nibble halves are two vectors of 64 weights and one `vpdpbusd` pair gives 16 int32 lanes (two
//! per group). The two 256-bit halves then meet in the same `vphaddd` as the 256-bit kernel, so the lane order, the
//! coefficients and the final tree are shared with it.
//!
//! AVX-512 has no `vpsignb`: the activations of negative weights are negated under a mask taken from the weights'
//! sign bits (`vpmovb2m`), which gives the same operand.

use std::arch::x86_64::*;
use std::ops::Range;

use crate::contract::{E2M1_X2, E2M1_X2_ABS};
use crate::x86_avx2::{coefficients, group_scales, reduce};
use crate::{Nvfp4View, QuantizedRows};

/// # Safety
/// AVX2, FMA, AVX-512 F + BW + VNNI must be available. `rows_out` must lie within `w`, and `y` must hold
/// `x.rows() * rows_out.len()` values.
#[target_feature(enable = "avx2,fma,avx512f,avx512bw,avx512vnni")]
pub(crate) unsafe fn gemv_avx512vnni(w: &Nvfp4View<'_>, x: &QuantizedRows, rows_out: &Range<usize>, y: &mut [f32]) {
    let mut m0 = 0;
    while m0 < x.rows() {
        let mb = (x.rows() - m0).min(8);
        // SAFETY: forwarded from this function's contract.
        unsafe {
            match mb {
                1 => block::<1>(w, x, m0, rows_out, y),
                2 => block::<2>(w, x, m0, rows_out, y),
                3 => block::<3>(w, x, m0, rows_out, y),
                4 => block::<4>(w, x, m0, rows_out, y),
                5 => block::<5>(w, x, m0, rows_out, y),
                6 => block::<6>(w, x, m0, rows_out, y),
                7 => block::<7>(w, x, m0, rows_out, y),
                _ => block::<8>(w, x, m0, rows_out, y),
            }
        }
        m0 += mb;
    }
}

/// The activations negated where the corresponding weight is negative.
#[inline(always)]
unsafe fn apply_sign(x: __m512i, weights: __m512i) -> __m512i {
    // SAFETY: the caller guarantees AVX-512 F + BW.
    unsafe { _mm512_mask_sub_epi8(x, _mm512_movepi8_mask(weights), _mm512_setzero_si512(), x) }
}

/// # Safety
/// As [`gemv_avx512vnni`].
#[inline(always)]
unsafe fn block<const MB: usize>(
    w: &Nvfp4View<'_>,
    x: &QuantizedRows,
    m0: usize,
    rows_out: &Range<usize>,
    y: &mut [f32],
) {
    let len = rows_out.len();
    let chunks = w.chunks();
    // SAFETY: the features are guaranteed by the caller, and the pointers stay inside their slices exactly as in the
    // 256-bit kernel (whole padded chunks of 64 code bytes, 8 scale bytes, 128 int8 values and 4 block scales).
    unsafe {
        let signed = _mm512_broadcast_i32x4(_mm_loadu_si128(E2M1_X2.as_ptr().cast()));
        let magnitude = _mm512_broadcast_i32x4(_mm_loadu_si128(E2M1_X2_ABS.as_ptr().cast()));
        let nibble = _mm512_set1_epi8(0x0f);
        let global = _mm256_set1_ps(w.global());
        for (i, n) in rows_out.clone().enumerate() {
            let codes = w.codes().as_ptr().add(n * w.code_stride());
            let scales = w.scales().as_ptr().add(n * w.scale_stride());
            let mut acc = [_mm256_setzero_ps(); MB];
            for c in 0..chunks {
                let bytes = _mm512_loadu_si512(codes.add(c * 64).cast());
                let lo = _mm512_and_si512(bytes, nibble);
                let hi = _mm512_and_si512(_mm512_srli_epi16::<4>(bytes), nibble);
                let (sl, sh) = (_mm512_shuffle_epi8(signed, lo), _mm512_shuffle_epi8(signed, hi));
                let (al, ah) = (_mm512_shuffle_epi8(magnitude, lo), _mm512_shuffle_epi8(magnitude, hi));
                let t = group_scales(scales.add(c * 8), global);
                for (j, a) in acc.iter_mut().enumerate() {
                    let m = m0 + j;
                    let xq = x.q().as_ptr().add(m * x.q_stride() + c * 128);
                    let xl = apply_sign(_mm512_loadu_si512(xq.cast()), sl);
                    let xh = apply_sign(_mm512_loadu_si512(xq.add(64).cast()), sh);
                    let dots = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(_mm512_setzero_si512(), al, xl), ah, xh);
                    let halves = _mm256_hadd_epi32(_mm512_castsi512_si256(dots), _mm512_extracti64x4_epi64::<1>(dots));
                    let cf = coefficients(t, x.scales().as_ptr().add(m * x.scale_stride() + c * 4));
                    *a = _mm256_fmadd_ps(_mm256_cvtepi32_ps(halves), cf, *a);
                }
            }
            for (j, &a) in acc.iter().enumerate() {
                y[(m0 + j) * len + i] = reduce(a);
            }
        }
    }
}
