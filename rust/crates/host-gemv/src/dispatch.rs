//! Run-time selection of the kernel variant by detected CPU features.

use std::ops::Range;

use crate::{Nvfp4View, QuantizedRows, scalar};

/// An instruction-set variant of the GEMV kernel. Every variant computes the same bits as [`Isa::Scalar`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Isa {
    /// Portable code, the exact reference twin.
    Scalar,
    /// x86-64 AVX2 + FMA: `vpmaddubsw` + `vpmaddwd`.
    Avx2,
    /// x86-64 AVX-VNNI: `vpdpbusd` on 256-bit vectors.
    AvxVnni,
    /// x86-64 AVX-512 (F, BW) VNNI: `vpdpbusd` on 512-bit vectors.
    Avx512Vnni,
    /// aarch64 NEON with the dot-product extension: `sdot`.
    NeonDotprod,
}

impl Isa {
    /// A short name for logs and tables.
    pub fn name(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::Avx2 => "avx2",
            Self::AvxVnni => "avx-vnni",
            Self::Avx512Vnni => "avx512-vnni",
            Self::NeonDotprod => "neon-dotprod",
        }
    }
}

/// Whether this CPU can run `isa`.
pub fn is_available(isa: Isa) -> bool {
    match isa {
        Isa::Scalar => true,
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"),
        #[cfg(target_arch = "x86_64")]
        Isa::AvxVnni => is_available(Isa::Avx2) && is_x86_feature_detected!("avxvnni"),
        #[cfg(target_arch = "x86_64")]
        Isa::Avx512Vnni => {
            is_available(Isa::Avx2)
                && is_x86_feature_detected!("avx512f")
                && is_x86_feature_detected!("avx512bw")
                && is_x86_feature_detected!("avx512vnni")
        }
        #[cfg(target_arch = "aarch64")]
        Isa::NeonDotprod => std::arch::is_aarch64_feature_detected!("dotprod"),
        #[allow(unreachable_patterns)]
        _ => false,
    }
}

/// Every variant this CPU can run, scalar first, then in increasing preference.
pub fn available() -> Vec<Isa> {
    [Isa::Scalar, Isa::Avx2, Isa::AvxVnni, Isa::Avx512Vnni, Isa::NeonDotprod]
        .into_iter()
        .filter(|&isa| is_available(isa))
        .collect()
}

/// The preferred variant of this CPU: the widest dot-product instruction it has, else the portable one. The engine
/// may instead pick the fastest of [`available`] from a startup measurement.
pub fn detected() -> Isa {
    available().pop().unwrap_or(Isa::Scalar)
}

/// Computes output rows `rows_out` of `w` for every activation row of `x` into `y`:
/// `y[m * rows_out.len() + (n - rows_out.start)]` for activation row `m` and output row `n`. This is the chunk unit the
/// engine publishes as frames; the bits of a result do not depend on the chunking, on the other activation rows, or on
/// `isa`.
///
/// # Panics
/// If `isa` is not available on this CPU ([`is_available`]), `x.k() != w.k()`, `rows_out` is not within `0..w.n()`, or
/// `y.len() != x.rows() * rows_out.len()`.
pub fn gemv_nvfp4(isa: Isa, w: &Nvfp4View<'_>, x: &QuantizedRows, rows_out: Range<usize>, y: &mut [f32]) {
    assert!(is_available(isa), "{} is not available on this CPU", isa.name());
    assert_eq!(x.k(), w.k(), "activation width differs from the weights'");
    assert!(
        rows_out.start <= rows_out.end && rows_out.end <= w.n(),
        "rows_out lies outside the weights"
    );
    assert_eq!(
        y.len(),
        x.rows() * rows_out.len(),
        "y must hold rows * rows_out.len() values"
    );
    match isa {
        Isa::Scalar => scalar::gemv(w, x, rows_out, y),
        #[cfg(target_arch = "x86_64")]
        // SAFETY: availability was checked above; the bounds were asserted above.
        Isa::Avx2 => unsafe { crate::x86_avx2::gemv_avx2(w, x, &rows_out, y) },
        #[cfg(target_arch = "x86_64")]
        // SAFETY: as above.
        Isa::AvxVnni => unsafe { crate::x86_avxvnni::gemv_avxvnni(w, x, &rows_out, y) },
        #[cfg(target_arch = "x86_64")]
        // SAFETY: as above.
        Isa::Avx512Vnni => unsafe { crate::x86_avx512vnni::gemv_avx512vnni(w, x, &rows_out, y) },
        #[cfg(target_arch = "aarch64")]
        // SAFETY: as above.
        Isa::NeonDotprod => unsafe { crate::aarch64_dotprod::gemv_dotprod(w, x, &rows_out, y) },
        #[allow(unreachable_patterns)]
        _ => unreachable!("is_available rejects variants of other architectures"),
    }
}
