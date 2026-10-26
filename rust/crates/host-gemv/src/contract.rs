//! The numeric contract shared by every variant (spec section 9.3).
//!
//! For output row `n` and activation row `m`:
//!
//! 1. x is quantised per 32-element block (see [`crate::quantize`]).
//! 2. Weight codes decode to the int8 value `2 * e2m1` (range +-12). The per-16 E4M3 scale and the tensor's global
//!    scale combine to the f32 `t_g = e4m3 * global` of group `g`.
//! 3. The dot product of a 16-element group is exact in int32.
//! 4. Group `g` contributes `(dot_g as f32) * c_g` with `c_g = (t_g * s_b) * 0.5` (`b = g / 2`, the x block of the
//!    group), into accumulator `a[g mod 8]` by one fused multiply-add: `a = fma(dot, c, a)`.
//! 5. The accumulators combine in the fixed tree [`reduce8`].
//!
//! Every operation above is correctly rounded IEEE-754 binary32, so every variant produces the same bits.

/// Elements per scale group of the weights.
pub const GROUP: usize = 16;
/// Elements per quantisation block of the activations.
pub const BLOCK: usize = 32;
/// Elements per kernel chunk: eight groups, one per accumulator.
pub const CHUNK: usize = 128;
/// Accumulators of a row.
pub const ACCUMULATORS: usize = 8;

/// Activation values are stored per [`CHUNK`] de-interleaved: the 64 even-indexed elements first, then the 64
/// odd-indexed ones. A code byte holds the even element in its low nibble and the odd one in its high nibble, so a
/// vector of code bytes masked by nibble lines up with a vector of activations without any lane shuffling. The
/// integer dot of a group does not depend on the order of its elements.
#[inline]
pub const fn x_position(k: usize) -> usize {
    let r = k % CHUNK;
    k - r + (r & 1) * (CHUNK / 2) + r / 2
}

/// `k` rounded up to the 128-element padding of the row-split weight layout.
#[inline]
pub const fn padded_k(k: usize) -> usize {
    k.div_ceil(CHUNK) * CHUNK
}

/// The int8 value (`2 * e2m1`) of each 4-bit code: sign in bit 3, magnitude codes 0..=7 are 0, 0.5, 1, 1.5, 2, 3, 4, 6.
pub const E2M1_X2: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

/// The magnitudes of [`E2M1_X2`], the unsigned operand of the integer dot.
pub const E2M1_X2_ABS: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, 1, 2, 3, 4, 6, 8, 12];

/// The largest magnitude of a pair sum of the AVX2 `vpmaddubsw` step: two products of a weight magnitude (at most 12)
/// and an activation (at most 127). The pair sums of both nibble halves added in int16 stay below `i16::MAX` too.
pub const MAX_PAIR_SUM: i32 = 2 * 127 * 12;

/// An E4M3 scale code as f32. The sign bit is ignored (a valid tensor has none) and code 0x7F is a finite 480 as on
/// the GPU. Exact: at most four significant bits.
#[inline]
pub fn decode_e4m3(code: u8) -> f32 {
    let e = u32::from(code >> 3) & 15;
    let m = u32::from(code) & 7;
    let mantissa = if e > 0 { m + 8 } else { m };
    let exponent = e.max(1);
    // 2^(exponent - 10), built from its biased exponent field.
    mantissa as f32 * f32::from_bits((exponent + 117) << 23)
}

/// The contribution coefficient of a group: `(t_g * s_b) * 0.5`, with the product order fixed as written.
#[inline]
pub fn coefficient(t: f32, s: f32) -> f32 {
    (t * s) * 0.5
}

/// The fixed combine tree of the eight accumulators.
#[inline]
pub fn reduce8(a: [f32; ACCUMULATORS]) -> f32 {
    let (p, q) = (a[0] + a[4], a[2] + a[6]);
    let (r, s) = (a[1] + a[5], a[3] + a[7]);
    (p + q) + (r + s)
}

/// f32 to BF16, round to nearest even, bit for bit the GPU epilogue (`q3::float_to_bf16`).
#[inline]
pub fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

/// BF16 to f32 (exact).
#[inline]
pub fn bf16_to_f32(value: u16) -> f32 {
    f32::from_bits(u32::from(value) << 16)
}

/// Converts f32 results to BF16 with round-to-nearest-even, matching the GPU epilogue.
///
/// # Panics
/// If the slices differ in length.
pub fn to_bf16(y: &[f32], out: &mut [u16]) {
    assert_eq!(y.len(), out.len(), "to_bf16 needs equal lengths");
    for (o, &v) in out.iter_mut().zip(y) {
        *o = f32_to_bf16(v);
    }
}
