use crate::error::EdrlError;

/// The values one E4M3 block scale covers.
const BLOCK: u64 = 16;
/// The FP32 global scale starts on this boundary after the last row.
const TRAILER_ALIGNMENT: u64 = 256;

/// The geometry of `row-interleaved-nvfp4-v1` tables (the n-gram embedding shards): `rows` rows of
/// [`NgramLayout::row_bytes`] bytes with no padding, each the row's `k / 2` bytes of E2M1 codes (even column in the low
/// nibble) followed by its `k / 16` E4M3 block scales; then, at the next 256-byte boundary after the last row, the
/// table's little-endian FP32 global scale. A value decodes to `e2m1(code) * e4m3(scale) * global`.
///
/// Unlike the row-split layout, one row is one contiguous range, so a lookup reads one row's bytes from one place.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NgramLayout {
    rows: u64,
    k: u64,
}

impl NgramLayout {
    /// Whether rows of `k` values can be stored: a positive multiple of 16.
    pub fn supports(k: u64) -> bool {
        k > 0 && k.is_multiple_of(BLOCK)
    }

    /// The layout of `rows` rows of `k` values; refuses an empty table, an unsupported width and a size that
    /// overflows.
    pub fn new(rows: u64, k: u64) -> Result<NgramLayout, EdrlError> {
        if rows == 0 || !NgramLayout::supports(k) {
            return Err(EdrlError::invalid(format!(
                "an interleaved NVFP4 table needs rows and a row width that is a positive multiple of 16: {rows} x {k}"
            )));
        }
        let layout = NgramLayout { rows, k };
        layout
            .rows
            .checked_mul(layout.row_bytes())
            .and_then(|body| body.checked_add(TRAILER_ALIGNMENT - 1))
            .ok_or_else(|| EdrlError::invalid(format!("interleaved NVFP4 table size overflows: {rows} x {k}")))?;
        Ok(layout)
    }

    /// The bytes of one row: codes, then block scales.
    pub fn row_bytes(&self) -> u64 {
        self.k / 2 + self.k / BLOCK
    }

    /// Where row `row` starts, or `None` past the last row.
    pub fn row_offset(&self, row: u64) -> Option<u64> {
        (row < self.rows).then(|| row * self.row_bytes())
    }

    /// Where the global scale starts.
    pub fn trailer_offset(&self) -> u64 {
        (self.rows * self.row_bytes()).div_ceil(TRAILER_ALIGNMENT) * TRAILER_ALIGNMENT
    }

    /// The payload size: the rows, padding, and the 4-byte global scale.
    pub fn byte_size(&self) -> u64 {
        self.trailer_offset() + 4
    }
}
