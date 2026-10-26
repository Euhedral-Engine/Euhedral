//! The 256-bit kernel of [`crate::x86_avx2`] with the AVX-VNNI `vpdpbusd` (non-saturating, never `vpdpbusds`).

use std::arch::x86_64::*;
use std::ops::Range;

use crate::x86_avx2::{Dot, gemv};
use crate::{Nvfp4View, QuantizedRows};

struct VnniDot;

impl Dot for VnniDot {
    #[inline(always)]
    unsafe fn pair(a0: __m256i, b0: __m256i, a1: __m256i, b1: __m256i) -> __m256i {
        // SAFETY: the caller guarantees AVX-VNNI.
        unsafe { _mm256_dpbusd_avx_epi32(_mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), a0, b0), a1, b1) }
    }
}

/// # Safety
/// AVX2, FMA and AVX-VNNI must be available; see [`crate::x86_avx2::gemv`].
#[target_feature(enable = "avx2,fma,avxvnni")]
pub(crate) unsafe fn gemv_avxvnni(w: &Nvfp4View<'_>, x: &QuantizedRows, rows_out: &Range<usize>, y: &mut [f32]) {
    // SAFETY: forwarded from this function's contract.
    unsafe { gemv::<VnniDot>(w, x, rows_out, y) }
}
