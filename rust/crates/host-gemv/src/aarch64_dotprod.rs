//! The aarch64 kernel: NEON with the `sdot` dot-product instructions. `sdot` multiplies signed bytes, so the weights
//! (looked up with `tbl` in the signed table) and the activations are used as they are.
//!
//! A 16-byte vector of code bytes holds two groups: its low nibbles are their even elements and its high nibbles
//! their odd elements, which line up with 16-byte slices of the de-interleaved activations of
//! [`crate::contract::x_position`]. Two `sdot` give four int32 lanes (two per group); two pairwise adds over the four
//! vectors of a chunk give one int32 per group in natural order, which is the accumulator order of the contract.

use std::arch::aarch64::*;
use std::ops::Range;

use crate::contract::E2M1_X2;
use crate::{Nvfp4View, QuantizedRows};

/// # Safety
/// NEON and dotprod must be available. `rows_out` must lie within `w`, and `y` must hold
/// `x.rows() * rows_out.len()` values.
#[target_feature(enable = "neon,dotprod")]
pub(crate) unsafe fn gemv_dotprod(w: &Nvfp4View<'_>, x: &QuantizedRows, rows_out: &Range<usize>, y: &mut [f32]) {
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

/// The E4M3 scale codes (one per u32 lane) as f32, exact.
#[inline(always)]
unsafe fn e4m3_lanes(codes: uint32x4_t) -> float32x4_t {
    // SAFETY: the caller guarantees NEON.
    unsafe {
        let v = vandq_u32(codes, vdupq_n_u32(0x7f));
        let e = vandq_u32(vshrq_n_u32::<3>(v), vdupq_n_u32(15));
        let m = vandq_u32(v, vdupq_n_u32(7));
        let normal = vcgtq_u32(e, vdupq_n_u32(0));
        let mantissa = vorrq_u32(m, vandq_u32(normal, vdupq_n_u32(8)));
        let exponent = vmaxq_u32(e, vdupq_n_u32(1));
        let power = vshlq_n_u32::<23>(vaddq_u32(exponent, vdupq_n_u32(117)));
        vmulq_f32(vcvtq_f32_u32(mantissa), vreinterpretq_f32_u32(power))
    }
}

/// # Safety
/// As [`gemv_dotprod`].
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
    // SAFETY: NEON and dotprod are guaranteed by the caller. The pointers stay inside their slices: whole padded
    // chunks of 64 code bytes, 8 scale bytes, 128 int8 values and 4 block scales per row.
    unsafe {
        let table = vld1q_s8(E2M1_X2.as_ptr());
        let nibble = vdupq_n_u8(0x0f);
        let global = vdupq_n_f32(w.global());
        let half = vdupq_n_f32(0.5);
        let zero = vdupq_n_s32(0);
        for (i, n) in rows_out.clone().enumerate() {
            let codes = w.codes().as_ptr().add(n * w.code_stride());
            let scales = w.scales().as_ptr().add(n * w.scale_stride());
            let mut acc = [(vdupq_n_f32(0.0), vdupq_n_f32(0.0)); MB];
            for c in 0..chunks {
                let mut lo = [vdupq_n_s8(0); 4];
                let mut hi = [vdupq_n_s8(0); 4];
                for p in 0..4 {
                    let bytes = vld1q_u8(codes.add(c * 64 + p * 16));
                    lo[p] = vqtbl1q_s8(table, vandq_u8(bytes, nibble));
                    hi[p] = vqtbl1q_s8(table, vshrq_n_u8::<4>(bytes));
                }
                let raw = vld1_u8(scales.add(c * 8));
                let wide = vmovl_u8(raw);
                let t_lo = vmulq_f32(e4m3_lanes(vmovl_u16(vget_low_u16(wide))), global);
                let t_hi = vmulq_f32(e4m3_lanes(vmovl_u16(vget_high_u16(wide))), global);
                for (j, a) in acc.iter_mut().enumerate() {
                    let m = m0 + j;
                    let xq = x.q().as_ptr().add(m * x.q_stride() + c * 128);
                    let mut d = [zero; 4];
                    for p in 0..4 {
                        let xe = vld1q_s8(xq.add(p * 16));
                        let xo = vld1q_s8(xq.add(64 + p * 16));
                        d[p] = vdotq_s32(vdotq_s32(zero, lo[p], xe), hi[p], xo);
                    }
                    let groups_lo = vpaddq_s32(d[0], d[1]);
                    let groups_hi = vpaddq_s32(d[2], d[3]);
                    let s = vld1q_f32(x.scales().as_ptr().add(m * x.scale_stride() + c * 4));
                    let s_lo = vzip1q_f32(s, s);
                    let s_hi = vzip2q_f32(s, s);
                    let c_lo = vmulq_f32(vmulq_f32(t_lo, s_lo), half);
                    let c_hi = vmulq_f32(vmulq_f32(t_hi, s_hi), half);
                    a.0 = vfmaq_f32(a.0, vcvtq_f32_s32(groups_lo), c_lo);
                    a.1 = vfmaq_f32(a.1, vcvtq_f32_s32(groups_hi), c_hi);
                }
            }
            for (j, &(lo, hi)) in acc.iter().enumerate() {
                // ((a0+a4)+(a2+a6))+((a1+a5)+(a3+a7)) over lanes a0..a3 and a4..a7.
                let sum = vaddq_f32(lo, hi);
                let pairs = vaddq_f32(sum, vextq_f32::<2>(sum, sum));
                y[(m0 + j) * len + i] = vgetq_lane_f32::<0>(pairs) + vgetq_lane_f32::<1>(pairs);
            }
        }
    }
}
