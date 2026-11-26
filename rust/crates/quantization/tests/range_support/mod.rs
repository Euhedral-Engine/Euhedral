//! The pure parts of the trellis activation-range check: what the engine's prepass does to a linear's input rows
//! before it rounds them to FP16 (multiply by `suh`, rotate each 128-wide block by the Hadamard transform), the
//! extremes of that, the segment a linear's rows sit in, and the report file.
#![allow(dead_code, reason = "the host test and the device test each use some of these")]

use std::fmt::Write;

/// The largest finite FP16 value.
pub const FP16_MAX: f32 = 65504.0;

/// The width the prepass rotates at a time.
pub const BLOCK: usize = 128;

/// The extremes of one linear: the largest `|x|`, the largest `|suh|` and the largest `|H128(x * suh)|`. A NaN in
/// any of them stays NaN.
#[derive(Clone, Debug, PartialEq)]
pub struct Extremes {
    pub linear: String,
    pub max_x: f32,
    pub max_suh: f32,
    pub max_prepass: f32,
}

impl Extremes {
    /// Whether the prepass output fits FP16: finite and below its largest value.
    pub fn fits_fp16(&self) -> bool {
        self.max_prepass.is_finite() && self.max_prepass < FP16_MAX
    }
}

/// The larger of two values, NaN when either is (`f32::max` would drop a NaN, and a NaN prepass must fail the check).
fn widest(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() { f32::NAN } else { a.max(b) }
}

/// The value of the half-precision float `bits`.
pub fn half_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 == 0 { 1.0 } else { -1.0 };
    let exponent = i32::from((bits >> 10) & 0x1f);
    let fraction = f32::from(bits & 0x3ff);
    match exponent {
        0 => sign * fraction * 2f32.powi(-24),
        0x1f if fraction == 0.0 => sign * f32::INFINITY,
        0x1f => f32::NAN,
        _ => sign * (1.0 + fraction / 1024.0) * 2f32.powi(exponent - 15),
    }
}

/// The unnormalised fast Walsh-Hadamard transform of `values` (a power of two long), in place.
pub fn hadamard(values: &mut [f32]) {
    let mut half = 1;
    while half < values.len() {
        for start in (0..values.len()).step_by(half * 2) {
            for at in start..start + half {
                let (a, b) = (values[at], values[at + half]);
                values[at] = a + b;
                values[at + half] = a - b;
            }
        }
        half *= 2;
    }
}

/// The extremes of `rows` rows of `k` BF16 values (`x`, little-endian bytes) under `suh` (`k` values): per row and
/// per block of 128, `x * suh` rotated by the Hadamard transform and scaled by `1 / sqrt(128)`.
pub fn prepass_extremes(linear: &str, x: &[u8], rows: usize, k: usize, suh: &[f32]) -> Extremes {
    assert!(
        k.is_multiple_of(BLOCK) && suh.len() == k && x.len() == 2 * rows * k,
        "{linear}: {rows} rows of {k} do not fit {} suh values and {} bytes",
        suh.len(),
        x.len()
    );
    let max_suh = suh.iter().fold(0.0, |max, value| widest(max, value.abs()));
    let (mut max_x, mut max_prepass) = (0.0f32, 0.0f32);
    let mut block = [0.0f32; BLOCK];
    let scale = 1.0 / (BLOCK as f32).sqrt();
    for row in x.chunks_exact(2 * k) {
        let (blocks, _) = row.as_chunks::<{ 2 * BLOCK }>();
        let (signs, _) = suh.as_chunks::<BLOCK>();
        for (blocks, signs) in blocks.iter().zip(signs) {
            for ((slot, pair), sign) in block.iter_mut().zip(blocks.as_chunks::<2>().0).zip(signs) {
                let value = f32::from_bits(u32::from(u16::from_le_bytes(*pair)) << 16);
                max_x = widest(max_x, value.abs());
                *slot = value * sign;
            }
            hadamard(&mut block);
            for value in block {
                max_prepass = widest(max_prepass, (value * scale).abs());
            }
        }
    }
    Extremes {
        linear: linear.to_owned(),
        max_x,
        max_suh,
        max_prepass,
    }
}

/// The index of the segment of `ends` (exclusive row ends) that holds rows `[first, first + rows)`, or why none does.
pub fn segment_of(ends: &[u32], first: u64, rows: u64) -> Result<usize, String> {
    let segment = ends
        .iter()
        .position(|end| u64::from(*end) > first)
        .ok_or_else(|| format!("row {first} is past the last segment end {:?}", ends.last()))?;
    if first + rows > u64::from(ends[segment]) {
        return Err(format!(
            "rows {first}..{} straddle the segment end {}",
            first + rows,
            ends[segment]
        ));
    }
    Ok(segment)
}

/// `value` as C's `%.6g` writes it: six significant digits, no trailing zeros, an exponent below -4 or from 6 up.
/// Not a finite value is `null`.
fn six_digits(value: f32) -> String {
    if !value.is_finite() {
        return "null".to_owned();
    }
    if value == 0.0 {
        return "0".to_owned();
    }
    let scientific = format!("{:.5e}", f64::from(value));
    let (mantissa, exponent) = scientific.split_once('e').unwrap_or((&scientific, "0"));
    let exponent: i32 = exponent.parse().unwrap_or(0);
    if !(-4..6).contains(&exponent) {
        let mantissa = mantissa.trim_end_matches('0').trim_end_matches('.');
        return format!("{mantissa}e{exponent:+03}");
    }
    let places = usize::try_from(5 - exponent).unwrap_or(0);
    let fixed = format!("{:.places$}", f64::from(value));
    if fixed.contains('.') {
        fixed.trim_end_matches('0').trim_end_matches('.').to_owned()
    } else {
        fixed
    }
}

/// The report `native/tests/test_trellis.py` reads: a `cases` array with one record per linear.
pub fn report_json(found: &[Extremes]) -> String {
    let mut json = String::from("{\"cases\": [\n");
    for (index, e) in found.iter().enumerate() {
        let comma = if index + 1 < found.len() { "," } else { "" };
        let _ = writeln!(
            json,
            "  {{\"linear\": \"{}\", \"max_abs_x\": {}, \"max_abs_suh\": {}, \"max_abs_prepass\": {}}}{comma}",
            e.linear,
            six_digits(e.max_x),
            six_digits(e.max_suh),
            six_digits(e.max_prepass)
        );
    }
    json.push_str("]}\n");
    json
}
