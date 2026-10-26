//! CPU GEMV over NVFP4 weights that stay in host RAM.
//!
//! The kernel picks its instruction set at run time ([`detected`], [`available`]). Every variant is bit for bit equal to
//! the portable scalar variant, which is the exact reference twin of the numeric contract in [`contract`]. A call
//! computes a range of output rows for every activation row; the engine publishes such ranges as frames.

#[cfg(target_arch = "aarch64")]
mod aarch64_dotprod;
pub mod contract;
mod dispatch;
mod quantize;
mod scalar;
#[cfg(target_arch = "x86_64")]
mod x86_avx2;
#[cfg(target_arch = "x86_64")]
mod x86_avx512vnni;
#[cfg(target_arch = "x86_64")]
mod x86_avxvnni;

use std::error::Error;
use std::fmt;

pub use contract::to_bf16;
pub use dispatch::{Isa, available, detected, gemv_nvfp4, is_available};
pub use quantize::{QuantizedRows, quantize_rows};

use contract::{BLOCK, CHUNK, GROUP, padded_k};

/// The alignment of the planes of a row-split NVFP4 tensor.
const PLANE_ALIGNMENT: usize = 256;

fn align_up(value: usize, alignment: usize) -> usize {
    value.div_ceil(alignment) * alignment
}

/// A weight tensor that does not fit the layout it was given.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayoutError {
    /// `k` is not a positive multiple of 32 (the activation block).
    Width(usize),
    /// A plane or payload is shorter than the layout needs.
    TooShort {
        what: &'static str,
        need: usize,
        have: usize,
    },
}

impl fmt::Display for LayoutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Width(k) => write!(f, "row width must be a positive multiple of {BLOCK}: {k}"),
            Self::TooShort { what, need, have } => write!(f, "{what} needs {need} bytes, has {have}"),
        }
    }
}

impl Error for LayoutError {}

/// An NVFP4 weight tensor of `n` rows by `k` columns in the row-split layout (`ROW_SPLIT_K128_V1`): rows of `k`
/// padded to 128 values. The code plane holds E2M1 codes two per byte (the even column in the low nibble), the
/// scale plane one E4M3 scale per 16 values. A weight is `e2m1(code) * e4m3(scale) * global`.
///
/// The GEMV reads whole 128-value chunks, so both planes must cover the padded row (the padding is never part of the
/// result).
#[derive(Debug, Clone, Copy)]
pub struct Nvfp4View<'a> {
    codes: &'a [u8],
    scales: &'a [u8],
    global: f32,
    n: usize,
    k: usize,
}

impl<'a> Nvfp4View<'a> {
    /// A view over the two planes. `k` must be a positive multiple of 32.
    pub fn new(codes: &'a [u8], scales: &'a [u8], global: f32, n: usize, k: usize) -> Result<Self, LayoutError> {
        if k == 0 || !k.is_multiple_of(BLOCK) {
            return Err(LayoutError::Width(k));
        }
        let kp = padded_k(k);
        if codes.len() < n * (kp / 2) {
            return Err(LayoutError::TooShort {
                what: "code plane",
                need: n * (kp / 2),
                have: codes.len(),
            });
        }
        if scales.len() < n * (kp / GROUP) {
            return Err(LayoutError::TooShort {
                what: "scale plane",
                need: n * (kp / GROUP),
                have: scales.len(),
            });
        }
        Ok(Self {
            codes,
            scales,
            global,
            n,
            k,
        })
    }

    /// A view over a whole `ROW_SPLIT_K128_V1` payload: the code plane at 0, the scale plane at the next 256-byte
    /// boundary, then at the next 256-byte boundary the little-endian f32 global scale.
    pub fn from_row_split(payload: &'a [u8], n: usize, k: usize) -> Result<Self, LayoutError> {
        if k == 0 || !k.is_multiple_of(BLOCK) {
            return Err(LayoutError::Width(k));
        }
        let kp = padded_k(k);
        let scale_offset = align_up(n * (kp / 2), PLANE_ALIGNMENT);
        let global_offset = align_up(scale_offset + n * (kp / GROUP), PLANE_ALIGNMENT);
        let need = global_offset + size_of::<f32>();
        let Some(global_bytes) = payload.get(global_offset..need) else {
            return Err(LayoutError::TooShort {
                what: "payload",
                need,
                have: payload.len(),
            });
        };
        let global = f32::from_le_bytes(global_bytes.try_into().expect("four bytes"));
        Self::new(
            &payload[..scale_offset],
            &payload[scale_offset..global_offset],
            global,
            n,
            k,
        )
    }

    /// Output rows.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Columns (the dot length).
    pub fn k(&self) -> usize {
        self.k
    }

    /// The tensor's global scale.
    pub fn global(&self) -> f32 {
        self.global
    }

    /// Bytes of the code and scale planes a full pass reads (the padded rows).
    pub fn plane_bytes(&self) -> usize {
        self.n * (padded_k(self.k) / 2 + padded_k(self.k) / GROUP)
    }

    pub(crate) fn codes(&self) -> &'a [u8] {
        self.codes
    }

    pub(crate) fn scales(&self) -> &'a [u8] {
        self.scales
    }

    /// Code bytes per row.
    pub(crate) fn code_stride(&self) -> usize {
        padded_k(self.k) / 2
    }

    /// Scale bytes per row.
    pub(crate) fn scale_stride(&self) -> usize {
        padded_k(self.k) / GROUP
    }

    /// Whole 128-value chunks per row.
    pub(crate) fn chunks(&self) -> usize {
        padded_k(self.k) / CHUNK
    }
}
