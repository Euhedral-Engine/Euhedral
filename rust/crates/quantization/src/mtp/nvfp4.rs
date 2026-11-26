//! NVFP4 round to nearest as the converter writes it (`tools/euhedral_artifacts/nvfp4.py`, layout row-split-k128-v1),
//! in float32: the global scale is the largest magnitude over 6 x 448; each block of 16 along a row takes the E4M3 code
//! nearest to its largest magnitude / 6 / global, each value the E2M1 code nearest to |value| / (scale x global); ties
//! go to the even code, values past a table's end to its last entry. The training kernel (`drafter.cu` `nvfp4_round`)
//! computes the same bytes.

use thiserror::Error;

/// The E2M1 magnitudes, by code.
pub const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// The finite E4M3 magnitudes, by code 0 to 126: the codes under 8 are subnormal (a multiple of 2^-9), the others
/// `(1 + m / 8) x 2^(e - 7)`. Each power of two is built from its exponent bits, so no transcendental is involved.
pub const E4M3: [f32; 127] = e4m3_table();

/// 6 x 448, the largest value an NVFP4 weight decodes to under a global scale of 1.
const REACH: f32 = 2688.0;

const fn e4m3_table() -> [f32; 127] {
    let mut table = [0.0; 127];
    let mut code = 0;
    while code < 127 {
        let mantissa = (code & 7) as f32 / 8.0;
        table[code] = if code < 8 {
            mantissa * 0.015625
        } else {
            let exponent = (code >> 3) as u32 + 120;
            (1.0 + mantissa) * f32::from_bits(exponent << 23)
        };
        code += 1;
    }
    table
}

/// Why rows cannot be rounded.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum Nvfp4Error {
    #[error("row length {0} is not a multiple of 128")]
    RowLength(u32),
    #[error("{values} values for {rows} rows of {columns}")]
    Count { values: usize, rows: u64, columns: u32 },
}

/// The index of the entry of the ascending `table` nearest to `value` (numpy `searchsorted` side left, then the nearer
/// neighbour).
pub fn round_to_table(value: f32, table: &[f32]) -> u8 {
    let low = table.partition_point(|entry| *entry < value);
    let upper = low.clamp(1, table.len() - 1);
    let lower = upper - 1;
    let (below, above) = (value - table[lower], table[upper] - value);
    let mut index = if above < below || (above == below && upper % 2 == 0) {
        upper
    } else {
        lower
    };
    if value >= table[table.len() - 1] {
        index = table.len() - 1;
    }
    if value <= table[0] {
        index = 0;
    }
    index as u8
}

fn align(value: u64) -> u64 {
    value.next_multiple_of(256)
}

/// Where the scale plane and the global scale lie in an NVFP4 object of `rows` rows of `columns`.
fn offsets(rows: u64, columns: u32) -> (u64, u64) {
    let count = rows * u64::from(columns);
    let scales = align(count / 2);
    (scales, align(scales + count / 16))
}

/// The bytes of an NVFP4 object of `rows` rows of `columns` (a multiple of 128): the packed codes, the scale plane and
/// the global scale, each plane on a 256-byte boundary.
pub fn bytes(rows: u64, columns: u32) -> u64 {
    offsets(rows, columns).1 + 4
}

/// The NVFP4 object of `values`, `rows` rows of `columns` in row-major order.
pub fn round(values: &[f32], rows: u64, columns: u32) -> Result<Vec<u8>, Nvfp4Error> {
    if !columns.is_multiple_of(128) {
        return Err(Nvfp4Error::RowLength(columns));
    }
    if values.len() as u64 != rows * u64::from(columns) {
        return Err(Nvfp4Error::Count {
            values: values.len(),
            rows,
            columns,
        });
    }
    let amax = values.iter().fold(0.0f32, |largest, value| largest.max(value.abs()));
    let global = amax / REACH;
    let (scales, at) = offsets(rows, columns);
    let mut out = vec![0u8; (at + 4) as usize];
    let (blocks, _) = values.as_chunks::<16>();
    for (block, values) in blocks.iter().enumerate() {
        let largest = values.iter().fold(0.0f32, |largest, value| largest.max(value.abs()));
        let scale = if global > 0.0 {
            round_to_table(largest / 6.0 / global, &E4M3)
        } else {
            0
        };
        let decode = E4M3[usize::from(scale)] * global;
        let mut codes = [0u8; 16];
        for (code, value) in codes.iter_mut().zip(values) {
            let magnitude = round_to_table(if decode > 0.0 { value.abs() / decode } else { 0.0 }, &E2M1);
            *code = magnitude | if *value < 0.0 && magnitude > 0 { 8 } else { 0 };
        }
        for (pair, byte) in codes.as_chunks::<2>().0.iter().zip(&mut out[block * 8..block * 8 + 8]) {
            *byte = pair[0] | (pair[1] << 4);
        }
        out[scales as usize + block] = scale;
    }
    out[at as usize..].copy_from_slice(&global.to_le_bytes());
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two rows of 128 whose blocks span the scale range: full scale, 1/64, 1/65536 (subnormal scales) and zero.
    fn converter_input() -> Vec<f32> {
        let scale = [1.0f32, 1.0 / 64.0, 1.0 / 65536.0, 0.0];
        (0..2u32)
            .flat_map(|row| (0..128u32).map(move |column| ((column * 37 + row * 101) % 61) as f32 - 30.0))
            .enumerate()
            .map(|(index, value)| value / 8.0 * scale[(index % 128 / 16 + index / 128) % 4])
            .collect()
    }

    /// The bytes of the converter's `quantize_nvfp4_rows` for `converter_input`.
    #[rustfmt::skip]
    const CODES: &str = "3f6dfad4964f6de0c5175e7ce2b5276efae4964f7df0d5070000000000000000a64f6df9d4075f7ce1c5275e7be2a63f6ef9d4075f7df0c50000000000000000d4065f7de0c5175e7bd2a52f6de9d3960000000000000000b5276efad3964f6de0c4065e7ce1b5275dfad2963e6de9c40000000000000000e2a63f6df9d4064e";
    const SCALES: [u8; 16] = [126, 78, 3, 0, 125, 78, 3, 0, 77, 4, 0, 126, 78, 4, 0, 126];

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    #[test]
    fn the_rows_are_the_bytes_the_converter_makes() {
        let out = round(&converter_input(), 2, 128).unwrap();
        assert_eq!(out.len() as u64, bytes(2, 128));
        assert_eq!(hex(&out[..128]), CODES);
        assert_eq!(out[128..256], [0u8; 128], "the gap before the scales is zero");
        assert_eq!(out[256..272], SCALES);
        assert_eq!(out[272..512], [0u8; 240]);
        assert_eq!(out[512..], 985_062_254u32.to_le_bytes());
    }

    #[test]
    fn zeros_make_a_zero_object() {
        let out = round(&[0.0; 256], 2, 128).unwrap();
        assert!(out.iter().all(|byte| *byte == 0));
    }

    #[test]
    fn a_block_at_the_maximum_takes_the_largest_scale_code() {
        let mut values = [0.0f32; 128];
        values[0] = -2688.0;
        values[1] = 2688.0;
        let out = round(&values, 1, 128).unwrap();
        assert_eq!(out[128..136], [0u8; 8], "no block but the first has a scale");
        // global 1, largest 2688 / 6 / 1 = 448 is the last table entry.
        assert_eq!(out[256], 126);
        assert_eq!(out[0], 0x7f, "-6 is code 15, +6 is code 7: low nibble first");
    }

    #[test]
    fn a_tie_goes_to_the_even_index() {
        // 0.25 lies halfway between 0.0 (index 0) and 0.5 (index 1); 2.5 between 2.0 (4) and 3.0 (5); 5.0 between 4.0
        // (6) and 6.0 (7).
        assert_eq!(round_to_table(0.25, &E2M1), 0);
        assert_eq!(round_to_table(0.75, &E2M1), 2);
        assert_eq!(round_to_table(2.5, &E2M1), 4);
        assert_eq!(round_to_table(5.0, &E2M1), 6);
        assert_eq!(round_to_table(100.0, &E2M1), 7);
        assert_eq!(round_to_table(-1.0, &E2M1), 0);
    }

    #[test]
    fn the_scale_table_is_ascending_and_ends_at_448() {
        assert!(E4M3.windows(2).all(|pair| pair[0] < pair[1]));
        assert_eq!((E4M3[0], E4M3[126]), (0.0, 448.0));
        assert_eq!(E4M3[1], 2.0f32.powi(-9));
    }

    #[test]
    fn a_row_that_is_not_a_multiple_of_128_or_a_wrong_count_is_refused() {
        assert_eq!(round(&[0.0; 64], 1, 64), Err(Nvfp4Error::RowLength(64)));
        assert_eq!(
            round(&[0.0; 100], 1, 128),
            Err(Nvfp4Error::Count {
                values: 100,
                rows: 1,
                columns: 128
            })
        );
    }
}
