//! The launches of the Flash-Next kernel table, one function per entry of Java's `Ops` (and of `GdnLayer.Qwen4GdnOps`):
//! the same kernel, grid, block and argument order. Each refuses a zero extent, as Java refuses a non-positive one, and
//! the alignment its kernel reads with; the value is in the message.
//!
//! A function returns the [`Launch`] and queues nothing: what a test records is what the device runs.

use euhedral_inference_cuda::DevicePtr;

use super::FlashError;
use super::kernels::Qwen4Kernel;
use crate::formats::shapes::{KernelArguments, Launch};

const WARP_BLOCK: u32 = 256;
const TC_BLOCK: u32 = 128;
const TC_MIN_ROWS: u32 = 9;

fn positive(op: &'static str, extents: &[(&str, u32)]) -> Result<(), FlashError> {
    match extents.iter().find(|(_, value)| *value == 0) {
        Some((name, _)) => Err(FlashError::refused(op, format!("{name} must be positive, got 0"))),
        None => Ok(()),
    }
}

fn aligned(op: &'static str, alignment: u64, addresses: &[(&str, DevicePtr)]) -> Result<(), FlashError> {
    match addresses
        .iter()
        .find(|(_, address)| address.is_null() || address.get() % alignment != 0)
    {
        Some((name, address)) => Err(FlashError::refused(
            op,
            format!("{name} {:#x} is not {alignment}-byte aligned", address.get()),
        )),
        None => Ok(()),
    }
}

/// `value / divisor` rounded up, as a launch dimension.
fn blocks(op: &'static str, value: u64, divisor: u64) -> Result<u32, FlashError> {
    u32::try_from(value.div_ceil(divisor))
        .map_err(|_| FlashError::refused(op, format!("{value} items in blocks of {divisor} exceed a grid")))
}

/// The K slices of a decode linear (`Ops.slices`): 8 for a narrow output or a long K, 4 for a mid K, else 1.
pub fn slices(k: u32, n: u32) -> u32 {
    if n < 64 || k >= 8192 {
        8
    } else if k >= 2048 && n < 4096 {
        4
    } else {
        1
    }
}

/// The BF16 linear `output[rows, n] = input[rows, k] * weights[n, k]^T` (`Ops.linearBf16`): the tensor-core kernels
/// from nine rows on, the K-split kernels below it, and the one-warp-per-column kernel when `exact` numerics are
/// selected or neither fits.
pub fn linear_bf16(
    input: DevicePtr,
    weights: DevicePtr,
    output: DevicePtr,
    rows: u32,
    k: u32,
    n: u32,
    exact: bool,
) -> Result<Launch, FlashError> {
    const OP: &str = "linear_bf16";
    positive(OP, &[("rows", rows), ("k", k), ("n", n)])?;
    if k % 8 != 0 {
        return Err(FlashError::refused(
            OP,
            format!("a BF16 linear needs K divisible by 8, got {k}"),
        ));
    }
    aligned(OP, 16, &[("input", input), ("weights", weights)])?;
    let args = KernelArguments::new()
        .pointer(input)
        .pointer(weights)
        .pointer(output)
        .int32(rows)
        .int32(k)
        .int32(n);
    if rows >= TC_MIN_ROWS && !exact && k % 32 == 0 && n % 8 == 0 && output.get() & 3 == 0 {
        let split = n <= 1536 || n % 32 != 0;
        let tiles = rows > 16;
        let kernel = match (split, tiles) {
            (true, true) => Qwen4Kernel::LinearTcRowsSplit,
            (true, false) => Qwen4Kernel::LinearTcSplit,
            (false, true) => Qwen4Kernel::LinearTcRows,
            (false, false) => Qwen4Kernel::LinearTc,
        };
        let columns = if split { n / 8 } else { n / 32 };
        let row_blocks = blocks(OP, u64::from(rows), if tiles { 64 } else { 16 })?;
        return Ok(kernel.launch([columns, row_blocks, 1], [TC_BLOCK, 1, 1], args));
    }
    let slices = if rows < TC_MIN_ROWS && !exact { slices(k, n) } else { 1 };
    let row_blocks = blocks(OP, u64::from(rows), 8)?;
    let (kernel, columns) = match slices {
        8 => (Qwen4Kernel::LinearSplit8, blocks(OP, u64::from(n), 1)?),
        4 => (Qwen4Kernel::LinearSplit4, blocks(OP, u64::from(n), 2)?),
        _ => (Qwen4Kernel::LinearBf16, blocks(OP, u64::from(n), 8)?),
    };
    Ok(kernel.launch([columns, row_blocks, 1], [WARP_BLOCK, 1, 1], args))
}

/// RMS norm over `groups` groups of `width` per row (`Ops.groupedRmsNorm`).
pub fn grouped_rms_norm(
    input: DevicePtr,
    weight: DevicePtr,
    output: DevicePtr,
    rows: u32,
    groups: u32,
    width: u32,
    epsilon: f32,
) -> Result<Launch, FlashError> {
    const OP: &str = "grouped_rms_norm";
    positive(OP, &[("rows", rows), ("groups", groups), ("width", width)])?;
    let grid = rows
        .checked_mul(groups)
        .ok_or_else(|| FlashError::refused(OP, format!("{rows} rows of {groups} groups exceed a grid")))?;
    Ok(Qwen4Kernel::GroupedRmsNorm.launch(
        [grid, 1, 1],
        [WARP_BLOCK, 1, 1],
        KernelArguments::new()
            .pointer(input)
            .pointer(weight)
            .pointer(output)
            .int32(rows)
            .int32(groups)
            .int32(width)
            .float32(epsilon),
    ))
}

/// `output = silu(input) / divisor` over `count` values (`Ops.scaledSilu`).
pub fn scaled_silu(input: DevicePtr, output: DevicePtr, count: u32, divisor: f32) -> Result<Launch, FlashError> {
    const OP: &str = "scaled_silu";
    positive(OP, &[("count", count)])?;
    Ok(Qwen4Kernel::ScaledSilu.launch(
        [blocks(OP, u64::from(count), 256)?, 1, 1],
        [WARP_BLOCK, 1, 1],
        KernelArguments::new()
            .pointer(input)
            .pointer(output)
            .int32(count)
            .float32(divisor),
    ))
}

/// The mix of the streams by the low-rank weights (`Ops.hcMix`). `width` is the hidden size of one stream.
pub fn hc_mix(
    normed: DevicePtr,
    up: DevicePtr,
    mixed: DevicePtr,
    rows: u32,
    streams: u32,
    width: u32,
) -> Result<Launch, FlashError> {
    const OP: &str = "hc_mix";
    positive(OP, &[("rows", rows), ("streams", streams), ("width", width)])?;
    Ok(Qwen4Kernel::HcMix.launch(
        [blocks(OP, u64::from(width), 256)?, rows, 1],
        [WARP_BLOCK, 1, 1],
        KernelArguments::new()
            .pointer(normed)
            .pointer(up)
            .pointer(mixed)
            .int32(rows)
            .int32(streams)
            .int32(width),
    ))
}

/// The block's output added into each stream by the raw injection weights (`Ops.hcInject`).
pub fn hc_inject(
    hyper: DevicePtr,
    block: DevicePtr,
    raw_injection: DevicePtr,
    output: DevicePtr,
    rows: u32,
    streams: u32,
    width: u32,
) -> Result<Launch, FlashError> {
    const OP: &str = "hc_inject";
    positive(OP, &[("rows", rows), ("streams", streams), ("width", width)])?;
    Ok(Qwen4Kernel::HcInject.launch(
        [blocks(OP, u64::from(width), 256)?, rows, 1],
        [WARP_BLOCK, 1, 1],
        KernelArguments::new()
            .pointer(hyper)
            .pointer(block)
            .pointer(raw_injection)
            .pointer(output)
            .int32(rows)
            .int32(streams)
            .int32(width),
    ))
}

/// One row copied into each of `streams` streams (`Ops.repeatStreams`).
pub fn repeat_streams(
    input: DevicePtr,
    output: DevicePtr,
    rows: u32,
    streams: u32,
    width: u32,
) -> Result<Launch, FlashError> {
    const OP: &str = "repeat_streams";
    positive(OP, &[("rows", rows), ("streams", streams), ("width", width)])?;
    let total = u64::from(rows) * u64::from(streams) * u64::from(width);
    Ok(Qwen4Kernel::RepeatStreams.launch(
        [blocks(OP, total, 256)?, 1, 1],
        [WARP_BLOCK, 1, 1],
        KernelArguments::new()
            .pointer(input)
            .pointer(output)
            .int32(rows)
            .int32(streams)
            .int32(width),
    ))
}

/// The NVFP4 n-gram records expanded to BF16 rows of `width` values (`Ops.ngramExpand`).
pub fn ngram_expand(
    records: DevicePtr,
    output: DevicePtr,
    count: u32,
    width: u32,
    record_bytes: u32,
) -> Result<Launch, FlashError> {
    const OP: &str = "ngram_expand";
    positive(
        OP,
        &[("count", count), ("width", width), ("record_bytes", record_bytes)],
    )?;
    if record_bytes % 4 != 0 {
        return Err(FlashError::refused(
            OP,
            format!("records are multiples of four bytes, got {record_bytes}"),
        ));
    }
    Ok(Qwen4Kernel::NgramExpand.launch(
        [count, 1, 1],
        [WARP_BLOCK, 1, 1],
        KernelArguments::new()
            .pointer(records)
            .pointer(output)
            .int32(count)
            .int32(width)
            .int32(record_bytes),
    ))
}

/// `gated = gate(key . query) * value` per stream (`Ops.pleGate`).
pub fn ple_gate(
    key: DevicePtr,
    query: DevicePtr,
    value: DevicePtr,
    output: DevicePtr,
    rows: u32,
    streams: u32,
    width: u32,
) -> Result<Launch, FlashError> {
    const OP: &str = "ple_gate";
    positive(OP, &[("rows", rows), ("streams", streams), ("width", width)])?;
    let grid = rows
        .checked_mul(streams)
        .ok_or_else(|| FlashError::refused(OP, format!("{rows} rows of {streams} streams exceed a grid")))?;
    Ok(Qwen4Kernel::PleGate.launch(
        [grid, 1, 1],
        [WARP_BLOCK, 1, 1],
        KernelArguments::new()
            .pointer(key)
            .pointer(query)
            .pointer(value)
            .pointer(output)
            .int32(rows)
            .int32(streams)
            .int32(width),
    ))
}

/// The dilated depthwise convolution of the PLE over its history rows (`Ops.pleConv`).
#[allow(clippy::too_many_arguments, reason = "the kernel's own arguments, in Java's order")]
pub fn ple_conv(
    normed: DevicePtr,
    gated: DevicePtr,
    history: DevicePtr,
    weights: DevicePtr,
    output: DevicePtr,
    rows: u32,
    channels: u32,
    taps: u32,
    dilation: u32,
) -> Result<Launch, FlashError> {
    const OP: &str = "ple_conv";
    positive(
        OP,
        &[
            ("rows", rows),
            ("channels", channels),
            ("taps", taps),
            ("dilation", dilation),
        ],
    )?;
    Ok(Qwen4Kernel::PleConv.launch(
        [blocks(OP, u64::from(channels), 256)?, rows, 1],
        [WARP_BLOCK, 1, 1],
        KernelArguments::new()
            .pointer(normed)
            .pointer(gated)
            .pointer(history)
            .pointer(weights)
            .pointer(output)
            .int32(rows)
            .int32(channels)
            .int32(taps)
            .int32(dilation),
    ))
}

/// The last `history` rows of `x` kept as the next call's convolution history (`Ops.convHistory`): at most 32.
pub fn conv_history(
    x: DevicePtr,
    history_rows: DevicePtr,
    rows: u32,
    channels: u32,
    history: u32,
) -> Result<Launch, FlashError> {
    const OP: &str = "conv_history";
    positive(OP, &[("rows", rows), ("channels", channels), ("history", history)])?;
    if history > 32 {
        return Err(FlashError::refused(
            OP,
            format!("the history kernel keeps at most 32 rows, got {history}"),
        ));
    }
    Ok(Qwen4Kernel::ConvHistory.launch(
        [blocks(OP, u64::from(channels), 256)?, 1, 1],
        [WARP_BLOCK, 1, 1],
        KernelArguments::new()
            .pointer(x)
            .pointer(history_rows)
            .int32(rows)
            .int32(channels)
            .int32(history),
    ))
}

/// The causal depthwise convolution of the GDN input (`Qwen4GdnOps.convolution`).
pub fn gdn_conv(
    x: DevicePtr,
    history: DevicePtr,
    weights: DevicePtr,
    output: DevicePtr,
    rows: u32,
    channels: u32,
    taps: u32,
) -> Result<Launch, FlashError> {
    const OP: &str = "gdn_conv";
    positive(OP, &[("rows", rows), ("channels", channels), ("taps", taps)])?;
    Ok(Qwen4Kernel::GdnConv.launch(
        [blocks(OP, u64::from(channels), 256)?, rows, 1],
        [WARP_BLOCK, 1, 1],
        KernelArguments::new()
            .pointer(x)
            .pointer(history)
            .pointer(weights)
            .pointer(output)
            .int32(rows)
            .int32(channels)
            .int32(taps),
    ))
}

/// The decay and update gates of the recurrence from the `a` and `b` projections (`Qwen4GdnOps.control`).
#[allow(clippy::too_many_arguments, reason = "the kernel's own arguments, in Java's order")]
pub fn gdn_control(
    a: DevicePtr,
    b: DevicePtr,
    a_log: DevicePtr,
    dt_bias: DevicePtr,
    alpha: DevicePtr,
    beta: DevicePtr,
    rows: u32,
    heads: u32,
) -> Result<Launch, FlashError> {
    const OP: &str = "gdn_control";
    positive(OP, &[("rows", rows), ("heads", heads)])?;
    let items = u64::from(rows) * u64::from(heads);
    Ok(Qwen4Kernel::GdnControl.launch(
        [blocks(OP, items, 256)?, 1, 1],
        [WARP_BLOCK, 1, 1],
        KernelArguments::new()
            .pointer(a)
            .pointer(b)
            .pointer(a_log)
            .pointer(dt_bias)
            .pointer(alpha)
            .pointer(beta)
            .int32(rows)
            .int32(heads),
    ))
}

/// The gate applied to the recurrence's output, normed per head (`Qwen4GdnOps.gatedNorm`): `activation` 0 is SiLU, 1 is
/// sigmoid.
#[allow(clippy::too_many_arguments, reason = "the kernel's own arguments, in Java's order")]
pub fn gdn_gated_norm(
    core: DevicePtr,
    z: DevicePtr,
    weight: DevicePtr,
    output: DevicePtr,
    rows: u32,
    heads: u32,
    head_dim: u32,
    epsilon: f32,
    activation: u32,
) -> Result<Launch, FlashError> {
    const OP: &str = "gdn_gated_norm";
    positive(OP, &[("rows", rows), ("heads", heads), ("head_dim", head_dim)])?;
    let grid = rows
        .checked_mul(heads)
        .ok_or_else(|| FlashError::refused(OP, format!("{rows} rows of {heads} heads exceed a grid")))?;
    Ok(Qwen4Kernel::GdnGatedNorm.launch(
        [grid, 1, 1],
        [128, 1, 1],
        KernelArguments::new()
            .pointer(core)
            .pointer(z)
            .pointer(weight)
            .pointer(output)
            .int32(rows)
            .int32(heads)
            .int32(head_dim)
            .float32(epsilon)
            .int32(activation),
    ))
}

/// The rows of a BF16 table by token id (`Ops.embedding`); a row is a multiple of 8 values, the table and the output
/// 16-byte aligned.
pub fn embedding(
    table: DevicePtr,
    ids: DevicePtr,
    output: DevicePtr,
    rows: u32,
    width: u32,
    vocabulary: u32,
) -> Result<Launch, FlashError> {
    const OP: &str = "embedding";
    positive(OP, &[("rows", rows), ("width", width), ("vocabulary", vocabulary)])?;
    if width % 8 != 0 {
        return Err(FlashError::refused(
            OP,
            format!("embedding rows are multiples of 8 values, got {width}"),
        ));
    }
    aligned(OP, 16, &[("table", table), ("output", output)])?;
    Ok(Qwen4Kernel::Embedding.launch(
        [rows, 1, 1],
        [WARP_BLOCK, 1, 1],
        KernelArguments::new()
            .pointer(table)
            .pointer(ids)
            .pointer(output)
            .int32(rows)
            .int32(width)
            .int32(vocabulary),
    ))
}

#[cfg(test)]
mod tests;
