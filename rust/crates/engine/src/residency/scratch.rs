//! The scratch the linear routes take on the device, as a plan sizes it before any launch
//! (`TrellisFormat.planScratchBytes` and `TrellisSettings`).

/// The thresholds of the trellis linear route on one device (`TrellisSettings`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrellisThresholds {
    /// The most rows a launch other than a verification keeps on the tensor-core route.
    pub mma_max_rows: u64,
    /// The thread blocks the K split of the tensor-core route aims for.
    pub target_ctas: u64,
    /// The fewest rows for which each block runs every K split of its columns itself, with no split partials.
    pub whole_k_rows: u64,
}

/// The most segments one trellis object carries; the bound where only the payload is known.
pub const MAX_SEGMENTS: u64 = 8;

/// The Hadamard block: every segment boundary, `n` and `k` are multiples of it.
pub const TRELLIS_BLOCK: u64 = 128;

/// Regions of the scratch buffers are aligned to this many bytes.
const SCRATCH_ALIGNMENT: u64 = 256;

fn align_up(value: u64) -> u64 {
    value.div_ceil(SCRATCH_ALIGNMENT) * SCRATCH_ALIGNMENT
}

/// The K splits of the tensor-core route for an object of `k` inputs and `n` outputs: enough to fill `target_ctas`
/// blocks, at most one per 256 inputs (`TrellisFormat.splits`).
pub fn splits(k: u64, n: u64, target_ctas: u64) -> u64 {
    let blocks = (n / TRELLIS_BLOCK).max(1);
    let wanted = target_ctas.div_ceil(blocks);
    wanted.min(k / 256).max(1)
}

/// The tensor-core route's scratch for launches of up to `rows` rows: the activations, and the split partials of the
/// most rows below `whole_k_rows` (`TrellisFormat.mmaBytes`).
pub fn trellis_mma_bytes(thresholds: TrellisThresholds, rows: u64, k: u64, n: u64, segments: u64) -> u64 {
    let activations = align_up(segments * rows * k * 2);
    let splits = splits(k, n, thresholds.target_ctas);
    let split_rows = rows.min(thresholds.whole_k_rows.saturating_sub(1));
    if splits == 1 || split_rows == 0 {
        return activations;
    }
    activations + splits * split_rows * n * 4
}

/// The prefill route's scratch: the rotated activations, the object decoded to BF16 `[n, k]` and the GEMM outputs
/// `[rows, n]`, each region aligned (`TrellisFormat.prefillBytes`).
pub fn trellis_prefill_bytes(rows: u64, k: u64, n: u64, segments: u64) -> u64 {
    align_up(segments * rows * k * 2) + align_up(n * k * 2) + align_up(rows * n * 2)
}

/// The scratch an object of `n` outputs and `k` inputs needs at up to `max_rows` rows when its segment count is
/// unknown: the tensor-core route with at most [`MAX_SEGMENTS`] segments and, when `max_rows` exceeds the
/// tensor-core row limit, the larger prefill route (`TrellisFormat.planScratchBytes`).
pub fn trellis_plan_bytes(thresholds: TrellisThresholds, max_rows: u64, k: u64, n: u64) -> u64 {
    let mut bytes = trellis_mma_bytes(thresholds, max_rows, k, n, MAX_SEGMENTS);
    if max_rows > thresholds.mma_max_rows {
        bytes = bytes.max(trellis_prefill_bytes(max_rows, k, n, MAX_SEGMENTS));
    }
    bytes
}
