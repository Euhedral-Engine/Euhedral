//! The rates the trellis kernels support and what each one needs.

use super::TrellisError;

/// Weights in one tile.
pub const TILE_WEIGHTS: usize = 256;

/// The most dynamic shared memory a block gets without opting in.
pub(super) const DEFAULT_SHARED_BYTES: u32 = 48 * 1024;

/// A rate in bits per weight: an integer from 2 to 6, or 2.5 or 3.5, which alternate K and K + 1 bits (K the lower
/// rate).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Rate {
    lower: u32,
    half_step: bool,
}

impl Rate {
    /// The rate `bits`, when the kernels support it.
    pub fn new(bits: f64) -> Result<Self, TrellisError> {
        let lower = bits.floor();
        let half_step = bits != lower;
        let supported = if half_step {
            bits == 2.5 || bits == 3.5
        } else {
            (2.0..=6.0).contains(&lower)
        };
        if !supported {
            return Err(TrellisError::UnsupportedRate(bits));
        }
        Ok(Self {
            lower: lower as u32,
            half_step,
        })
    }

    /// Bits per weight.
    pub fn bits(self) -> f64 {
        f64::from(self.lower) + if self.half_step { 0.5 } else { 0.0 }
    }

    /// The integer rate, or the lower of the two a half-step rate alternates.
    pub fn lower(self) -> u32 {
        self.lower
    }

    /// Whether this is a half-step rate.
    pub fn is_half_step(self) -> bool {
        self.half_step
    }

    /// The 16-bit words of one packed tile.
    pub fn words_per_tile(self) -> usize {
        16 * self.lower as usize + if self.half_step { 8 } else { 0 }
    }

    /// States per trellis column that a tile's history keeps: `65536 >> K`.
    fn edges(self) -> u64 {
        65536u64 >> self.lower
    }

    /// Bytes of Viterbi scratch for `count` tiles. An integer rate keeps one history byte per position and state;
    /// a half-step rate keeps two cost arrays of half floats and 16-bit history.
    pub fn scratch_bytes(self, count: u64) -> u64 {
        let edges = self.edges();
        if self.half_step {
            count * (2 * 2 * edges + 2 * 256 * edges)
        } else {
            count * 256 * edges
        }
    }

    /// The tiles whose Viterbi scratch fits in `bytes`.
    pub fn scratch_tiles(self, bytes: u64) -> Result<u32, TrellisError> {
        let tile = self.scratch_bytes(1);
        if bytes < tile {
            return Err(TrellisError::ScratchTooSmall {
                bytes,
                bits: self.bits(),
                needed: tile,
            });
        }
        Ok(u32::try_from(bytes / tile).unwrap_or(u32::MAX))
    }

    /// Threads of a Viterbi block.
    pub(super) fn viterbi_threads(self) -> u32 {
        if self.half_step || self.lower >= 5 { 512 } else { 1024 }
    }

    /// Dynamic shared memory of a Viterbi block. An integer rate holds two half-float cost arrays, the tile and
    /// the reduction slots.
    pub(super) fn viterbi_shared_bytes(self) -> u32 {
        if self.half_step {
            256 * 2 + 32 * 4 + 128
        } else {
            let edges = 65536u32 >> self.lower;
            2 * edges * 2 + 256 * 2 + 64 + 128
        }
    }
}

/// Position in a tensor-core ordered tile to its row-major element (`row * 16 + column`) of the 16x16 tile.
pub const TENSOR_CORE_ORDER: [u16; TILE_WEIGHTS] = tensor_core_order();

const fn tensor_core_order() -> [u16; TILE_WEIGHTS] {
    let mut order = [0u16; TILE_WEIGHTS];
    let mut t = 0;
    while t < 32 {
        let r0 = (t % 4) * 2;
        let c0 = t / 4;
        let rows = [r0, r0 + 1, r0 + 8, r0 + 9];
        let mut j = 0;
        while j < 8 {
            order[8 * t + j] = (rows[j % 4] * 16 + c0 + (j / 4) * 8) as u16;
            j += 1;
        }
        t += 1;
    }
    order
}
