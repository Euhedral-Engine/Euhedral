//! The tensor data types, weight formats and layouts of the container, and the payload sizes they allow
//! (`CompactTensorLayout.acceptsByteSize` and the registered `LinearFormat`s).

use crate::error::EdrlError;

/// The source element type of a tensor (`TensorDataType`); the ordinal is what the table stores.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TensorDataType {
    Bf16,
    Fp16,
    Fp32,
    Fp8E4m3,
    Uint8,
    Int8,
    Int32,
}

impl TensorDataType {
    pub(crate) fn from_ordinal(ordinal: i32) -> Result<Self, EdrlError> {
        Ok(match ordinal {
            0 => Self::Bf16,
            1 => Self::Fp16,
            2 => Self::Fp32,
            3 => Self::Fp8E4m3,
            4 => Self::Uint8,
            5 => Self::Int8,
            6 => Self::Int32,
            _ => return Err(out_of_range("tensor data type", ordinal)),
        })
    }
}

/// How a payload's values are stored (`WeightFormat`); the ordinal is what the table stores.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WeightFormat {
    Bf16,
    Fp16,
    Fp32,
    I32,
    Q3G64Fp16,
    Q4G64Fp16,
    Q5G64Fp16,
    Nvfp4,
    Trellis,
}

impl WeightFormat {
    pub(crate) fn from_ordinal(ordinal: i32) -> Result<Self, EdrlError> {
        Ok(match ordinal {
            0 => Self::Bf16,
            1 => Self::Fp16,
            2 => Self::Fp32,
            3 => Self::I32,
            4 => Self::Q3G64Fp16,
            5 => Self::Q4G64Fp16,
            6 => Self::Q5G64Fp16,
            7 => Self::Nvfp4,
            8 => Self::Trellis,
            _ => return Err(out_of_range("tensor weight format", ordinal)),
        })
    }
}

/// The persistent byte layout of a payload (`WeightLayout`); the ordinal is what the table stores.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WeightLayout {
    ContiguousLeV1,
    RowSplitK128V1,
    RowSplitP2e2V1,
    RowSplitK128Sd4V1,
    RowInterleavedNvfp4V1,
    TrellisTile16V1,
}

impl WeightLayout {
    pub(crate) fn from_ordinal(ordinal: i32) -> Result<Self, EdrlError> {
        Ok(match ordinal {
            0 => Self::ContiguousLeV1,
            1 => Self::RowSplitK128V1,
            2 => Self::RowSplitP2e2V1,
            3 => Self::RowSplitK128Sd4V1,
            4 => Self::RowInterleavedNvfp4V1,
            5 => Self::TrellisTile16V1,
            _ => return Err(out_of_range("tensor weight layout", ordinal)),
        })
    }
}

fn out_of_range(what: &str, ordinal: i32) -> EdrlError {
    EdrlError::invalid(format!("{what} value is outside the supported range: {ordinal}"))
}

const GROUP_SIZE: u64 = 64;
const PLANE_ALIGNMENT: u64 = 256;
const K_ALIGNMENT: u64 = 128;
const NVFP4_BLOCK: u64 = 16;
const NVFP4_TABLE: u64 = 16;
const P2E2_SLICE: u64 = 1024;
const P2E2_PAD_WORDS: u64 = 80;
const TRELLIS_BLOCK: u64 = 128;

/// Why a shape, type, format and layout cannot be sized.
type Refusal = String;

/// Whether `bytes` is a valid payload size for a tensor, as `CompactTensorLayout.acceptsByteSize`: exactly the
/// registered format's size, or P2E2's bounds for P2E2 (its size depends on the codes). Combinations the engine has
/// no format for, and sizes that overflow, are refusals.
pub(crate) fn accepts_byte_size(
    shape: &[u64],
    dtype: TensorDataType,
    format: WeightFormat,
    layout: WeightLayout,
    bytes: u64,
) -> Result<bool, Refusal> {
    use WeightFormat as F;
    use WeightLayout as L;
    match (format, layout) {
        (F::Q3G64Fp16 | F::Q4G64Fp16 | F::Q5G64Fp16, L::RowSplitK128V1) => {
            grouped_size(shape, dtype, format).map(|size| size == bytes)
        }
        (F::Q3G64Fp16, L::RowSplitP2e2V1) => p2e2_accepts(shape, dtype, bytes),
        (F::Nvfp4, L::RowSplitK128V1 | L::RowSplitK128Sd4V1) => {
            nvfp4_size(shape, dtype, layout).map(|size| size == bytes)
        }
        (F::Bf16, L::ContiguousLeV1) => direct_size(shape, dtype, format).map(|size| size == bytes),
        (F::Trellis, L::TrellisTile16V1) => trellis_size(shape, dtype).map(|size| size == bytes),
        (_, L::RowSplitP2e2V1) => Err(p2e2_requirement()),
        (_, L::ContiguousLeV1) => direct_size(shape, dtype, format).map(|size| size == bytes),
        (_, L::RowSplitK128V1) => Err(if dtype != TensorDataType::Bf16 {
            "grouped compact storage requires BF16 source metadata".into()
        } else if shape.len() != 2 || shape.contains(&0) {
            "row-split layout requires a positive rank-2 shape".into()
        } else {
            format!("unsupported row-split format {format:?}")
        }),
        (_, L::RowSplitK128Sd4V1) => Err("SD4 layout requires NVFP4 storage of BF16 source metadata".into()),
        (_, L::RowInterleavedNvfp4V1) => Err("interleaved NVFP4 rows belong to version 3 artifacts".into()),
        (_, L::TrellisTile16V1) => Err("trellis tensors are sized by the registered trellis format".into()),
    }
}

fn overflow() -> Refusal {
    "compact size overflows".into()
}

fn mul(left: u64, right: u64) -> Result<u64, Refusal> {
    left.checked_mul(right).ok_or_else(overflow)
}

fn add(left: u64, right: u64) -> Result<u64, Refusal> {
    left.checked_add(right).ok_or_else(overflow)
}

fn align_up(value: u64, alignment: u64) -> Result<u64, Refusal> {
    Ok(add(value, alignment - 1)? / alignment * alignment)
}

/// `(rows, k)` of a positive rank-2 shape.
fn matrix(shape: &[u64]) -> Result<(u64, u64), Refusal> {
    match shape {
        [rows, k] if *rows > 0 && *k > 0 => Ok((*rows, *k)),
        _ => Err("row-split layout requires a positive rank-2 shape".into()),
    }
}

fn direct_size(shape: &[u64], dtype: TensorDataType, format: WeightFormat) -> Result<u64, Refusal> {
    let word_bytes = match format {
        WeightFormat::Bf16 | WeightFormat::Fp16 => 2,
        WeightFormat::Fp32 | WeightFormat::I32 => 4,
        _ => return Err(format!("contiguous layout does not support {format:?}")),
    };
    let mut elements: u64 = 1;
    for dimension in shape {
        if *dimension == 0 {
            return Err("compact shape is invalid or overflows".into());
        }
        elements = elements
            .checked_mul(*dimension)
            .ok_or("compact shape is invalid or overflows")?;
    }
    if dtype == TensorDataType::Int32 && format != WeightFormat::I32 {
        return Err("INT32 source dtype requires I32 storage metadata".into());
    }
    let required = match format {
        WeightFormat::Bf16 => Some((TensorDataType::Bf16, "BF16", "BF16")),
        WeightFormat::Fp16 => Some((TensorDataType::Fp16, "FP16", "FP16")),
        WeightFormat::I32 => Some((TensorDataType::Int32, "I32", "INT32")),
        _ => None,
    };
    if let Some((required, storage, source)) = required
        && dtype != required
    {
        return Err(format!("{storage} storage requires {source} source metadata"));
    }
    elements
        .checked_mul(word_bytes)
        .ok_or_else(|| "compact payload size overflows".into())
}

/// Q3, Q4 or Q5 at 64-value groups, row-split at K=128: base plane, high plane, then FP16 scales, each plane
/// starting on a 256-byte boundary.
fn grouped_size(shape: &[u64], dtype: TensorDataType, format: WeightFormat) -> Result<u64, Refusal> {
    if dtype != TensorDataType::Bf16 {
        return Err("grouped compact storage requires BF16 source metadata".into());
    }
    let (n, k) = matrix(shape)?;
    let bits = match format {
        WeightFormat::Q3G64Fp16 => 3,
        WeightFormat::Q4G64Fp16 => 4,
        _ => 5,
    };
    let k_pad = align_up(k, K_ALIGNMENT)?;
    let groups = k_pad / GROUP_SIZE;
    let base_per_group = if bits == 3 { 24 } else { 32 };
    let high_per_group = if bits <= 4 { 0 } else { 8 };
    let base_bytes = mul(n, mul(groups, base_per_group)?)?;
    let high_bytes = mul(n, mul(groups, high_per_group)?)?;
    let high_offset = align_up(base_bytes, PLANE_ALIGNMENT)?;
    let scale_offset = add(high_offset, align_up(high_bytes, PLANE_ALIGNMENT)?)?;
    add(scale_offset, mul(mul(n, groups)?, 2)?)
}

/// NVFP4: E2M1 codes, E4M3 block scales (or, in the SD4 layout, 4-bit indices into a 16-entry table), and an FP32
/// global scale, each plane starting on a 256-byte boundary.
fn nvfp4_size(shape: &[u64], dtype: TensorDataType, layout: WeightLayout) -> Result<u64, Refusal> {
    if dtype != TensorDataType::Bf16 {
        return Err(if layout == WeightLayout::RowSplitK128Sd4V1 {
            "SD4 layout requires NVFP4 storage of BF16 source metadata"
        } else {
            "grouped compact storage requires BF16 source metadata"
        }
        .into());
    }
    let (rows, k) = matrix(shape)?;
    let sd4 = layout == WeightLayout::RowSplitK128Sd4V1;
    let k_pad = align_up(k, K_ALIGNMENT)?;
    let row_scale_bytes = k_pad / if sd4 { 2 * NVFP4_BLOCK } else { NVFP4_BLOCK };
    let scale_offset = align_up(mul(rows, k_pad)? / 2, PLANE_ALIGNMENT)?;
    let trailer_offset = align_up(add(scale_offset, mul(rows, row_scale_bytes)?)?, PLANE_ALIGNMENT)?;
    let trailer_bytes = if sd4 { NVFP4_TABLE } else { 0 } + 4;
    add(trailer_offset, trailer_bytes)
}

/// Q3 entropy coded as P2E2: the size depends on the codes, so any 4-byte multiple of payload between an empty and a
/// full payload is valid.
fn p2e2_accepts(shape: &[u64], dtype: TensorDataType, bytes: u64) -> Result<bool, Refusal> {
    let supported =
        matches!(shape, [rows, k] if *rows > 0 && *k > 0 && k % P2E2_SLICE == 0) && dtype == TensorDataType::Bf16;
    if !supported {
        return Err(p2e2_requirement());
    }
    let (rows, k) = (shape[0], shape[1]);
    let row_base = align_up(mul(rows, k)? / 4, PLANE_ALIGNMENT)?;
    let scale = align_up(add(row_base, mul(rows, 4)?)?, PLANE_ALIGNMENT)?;
    let payload = align_up(add(scale, mul(rows, k)? / 32)?, PLANE_ALIGNMENT)?;
    let minimum = add(payload, 4 * P2E2_PAD_WORDS)?;
    let maximum = add(minimum, mul(4, add(mul(rows, k)?, 15)? / 16)?)?;
    Ok(bytes >= minimum && bytes <= maximum && (bytes - payload).is_multiple_of(4))
}

fn p2e2_requirement() -> Refusal {
    format!("P2E2 layout requires Q3 BF16 metadata and a rank-2 shape whose K is a multiple of {P2E2_SLICE}")
}

/// Trellis tiles `[k/16, n/16, w]` of int16 words: `w` is 16*K (K in 1..=8) or 16*K+8 (K in 1..=7); `k` and `n` are
/// multiples of 128.
fn trellis_size(shape: &[u64], dtype: TensorDataType) -> Result<u64, Refusal> {
    if dtype != TensorDataType::Bf16 {
        return Err("trellis storage requires BF16 source metadata".into());
    }
    let [k_tiles, n_tiles, words] = shape else {
        return Err("trellis layout requires a positive [k/16, n/16, w] shape".into());
    };
    if shape.contains(&0) {
        return Err("trellis layout requires a positive [k/16, n/16, w] shape".into());
    }
    if !words_valid(*words) {
        return Err(format!("trellis tile width {words} is not a stored width"));
    }
    let (k, n) = (mul(*k_tiles, 16)?, mul(*n_tiles, 16)?);
    if k % TRELLIS_BLOCK != 0 || n % TRELLIS_BLOCK != 0 {
        return Err(format!("trellis k={k} and n={n} must be multiples of 128"));
    }
    mul(mul(mul(*k_tiles, *n_tiles)?, *words)?, 2)
}

/// Whether `w` words per tile is a stored width.
fn words_valid(w: u64) -> bool {
    (16..=128).contains(&w) && (w.is_multiple_of(16) || (w % 16 == 8 && w <= 120))
}

/// The `.segments` companion of a trellis object: the exclusive row ends of its segments, one little-endian `int32`
/// each (unlike the big-endian table, because the device reads the payload as is).
#[derive(Debug)]
pub struct TrellisCompanions;

impl TrellisCompanions {
    /// The most segments one object carries.
    pub const MAX_SEGMENTS: usize = 8;
    /// Every segment end, `n` and `k` are multiples of this.
    pub const BLOCK: u32 = 128;

    /// The segment ends in `segments`: 1 to 8 strictly increasing multiples of 128.
    pub fn parse(segments: &[u8]) -> Result<Vec<u32>, EdrlError> {
        if !segments.len().is_multiple_of(4) {
            return Err(EdrlError::invalid(
                "trellis segments are not a whole number of int32 values",
            ));
        }
        let ends: Vec<i32> = segments
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| i32::from_le_bytes(*word))
            .collect();
        if ends.is_empty() || ends.len() > Self::MAX_SEGMENTS {
            return Err(EdrlError::invalid(format!(
                "segment count {} is outside 1..{}",
                ends.len(),
                Self::MAX_SEGMENTS
            )));
        }
        let mut previous = 0;
        for (index, end) in ends.iter().enumerate() {
            if *end <= previous {
                return Err(EdrlError::invalid(format!(
                    "segment ends are not strictly increasing at {index}"
                )));
            }
            if end % Self::BLOCK as i32 != 0 {
                return Err(EdrlError::invalid(format!(
                    "segment end {end} is not a multiple of {}",
                    Self::BLOCK
                )));
            }
            previous = *end;
        }
        Ok(ends.into_iter().map(|end| end as u32).collect())
    }

    /// Checks that `ends` fit an object of `n` rows whose `.suh` companion has `suh_rows` rows: one segment per
    /// `suh` row, the last ending at `n`.
    pub fn check_rows(ends: &[u32], n: u64, suh_rows: u64) -> Result<(), EdrlError> {
        if ends.len() as u64 != suh_rows {
            return Err(EdrlError::invalid(format!(
                "segment count {} differs from suh rows {suh_rows}",
                ends.len()
            )));
        }
        match ends.last() {
            Some(last) if u64::from(*last) == n => Ok(()),
            Some(last) => Err(EdrlError::invalid(format!("last segment end {last} is not n={n}"))),
            None => Err(EdrlError::invalid("segment count 0 is outside 1..8")),
        }
    }
}
