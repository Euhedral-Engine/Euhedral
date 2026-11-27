//! The launches of the mixture of experts (`MoeOps` and `ExpertOps`): the router, the shared expert's SwiGLU, the
//! per-expert gate-up and down kernels, the ordered combine and the finish, with Java's grid, block and argument order.
//! Each refuses what its kernel cannot take, with the value in the message, and returns the [`Launch`] without queueing it.

use euhedral_inference_cuda::DevicePtr;

use crate::flash::FlashError;
use crate::flash::kernels::Qwen4Kernel;
use crate::flash::ops::{aligned, blocks, positive};
use crate::formats::shapes::{KernelArguments, Launch};

/// The block of the element-wise kernels, the router and the finish.
const WIDE_BLOCK: u32 = 256;
/// The block of the expert kernels.
const EXPERT_BLOCK: u32 = 128;
/// Output columns of one block of the gate-up kernel, rows of one block of the down kernel.
const GATE_UP_COLUMNS: u32 = 32;
const DOWN_ROWS: u32 = 128;
/// The most experts the router's per-row selection takes, and the most it picks.
pub const ROUTER_MAX_EXPERTS: u32 = 1024;
pub const ROUTER_MAX_TOP_K: u32 = 16;
/// The row alignment the expert kernels need of every activation, scratch and output address.
pub const EXPERT_ALIGNMENT: u64 = 16;

/// Shapes and record offsets of an expert record: `gate_up` `[2 * inter][hidden]` at `gate_up_offset`, `down`
/// `[hidden][inter]` at `down_offset`, both complete NVFP4 tensors (`ExpertOps.Geometry`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpertGeometry {
    hidden: u32,
    inter: u32,
    gate_up_offset: u32,
    down_offset: u32,
}

impl ExpertGeometry {
    /// Refuses extents that are not positive multiples of 128 (`inter` of 32 for the gate-up columns and `hidden` of
    /// 128 for the down rows), offsets that are not multiples of 256, and offsets past 32 bits.
    pub fn new(hidden: u32, inter: u32, gate_up_offset: u64, down_offset: u64) -> Result<Self, FlashError> {
        const OP: &str = "expert geometry";
        positive(OP, &[("hidden", hidden), ("inter", inter)])?;
        if !hidden.is_multiple_of(128) || !inter.is_multiple_of(128) {
            return Err(FlashError::refused(
                OP,
                format!("hidden and inter must be multiples of 128, got {hidden}, {inter}"),
            ));
        }
        let offset = |name: &str, value: u64| {
            u32::try_from(value)
                .ok()
                .filter(|value| value.is_multiple_of(256))
                .ok_or_else(|| {
                    FlashError::refused(
                        OP,
                        format!("{name} offset {value} is not a 256-byte aligned 32-bit offset"),
                    )
                })
        };
        Ok(Self {
            hidden,
            inter,
            gate_up_offset: offset("gate_up", gate_up_offset)?,
            down_offset: offset("down", down_offset)?,
        })
    }

    /// Flash-Next's experts: 2560 hidden, 640 intermediate, `gate_up` at 0 and `down` at 1,843,456.
    pub fn flash_next() -> Self {
        Self {
            hidden: 2560,
            inter: 640,
            gate_up_offset: 0,
            down_offset: 1_843_456,
        }
    }

    pub fn hidden(&self) -> u32 {
        self.hidden
    }

    pub fn inter(&self) -> u32 {
        self.inter
    }

    /// The bytes of `act` for `pairs` pairs, which is where `weighted` starts (`ExpertOps.Scratch`).
    pub fn weighted_offset(&self, pairs: u32) -> u64 {
        (2 * u64::from(pairs) * u64::from(self.inter)).next_multiple_of(EXPERT_ALIGNMENT)
    }

    /// The device scratch of a block of at most `max_pairs` pairs: `act` `[pairs][inter]`, then `weighted`
    /// `[pairs][hidden]`, both BF16 and 16-byte aligned.
    pub fn scratch_bytes(&self, max_pairs: u32) -> u64 {
        self.weighted_offset(max_pairs)
            + (2 * u64::from(max_pairs) * u64::from(self.hidden)).next_multiple_of(EXPERT_ALIGNMENT)
    }
}

/// The device addresses of one planned block's description, as an expert kernel reads them: the slot table, the first
/// item of the expert and how many it has, and the pairs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpertItems {
    pub slots: DevicePtr,
    pub items: DevicePtr,
    pub item_count: u32,
    pub pairs: DevicePtr,
}

/// The router (`MoeOps.router`): the `k` experts of each of `rows` rows from the BF16 `logits`, as `ids` (int32) and
/// `weights` (BF16 bits), `rows` x `k` each.
pub fn router(
    logits: DevicePtr,
    ids: DevicePtr,
    weights: DevicePtr,
    rows: u32,
    experts: u32,
    k: u32,
) -> Result<Launch, FlashError> {
    const OP: &str = "router";
    positive(OP, &[("rows", rows), ("experts", experts), ("k", k)])?;
    if experts > ROUTER_MAX_EXPERTS || k > ROUTER_MAX_TOP_K || k > experts {
        return Err(FlashError::refused(
            OP,
            format!(
                "router shape {rows} x {experts} top {k}: at most {ROUTER_MAX_EXPERTS} experts and {ROUTER_MAX_TOP_K} picks, never more than the experts"
            ),
        ));
    }
    Ok(Qwen4Kernel::Router.launch(
        [rows, 1, 1],
        [WIDE_BLOCK, 1, 1],
        KernelArguments::new()
            .pointer(logits)
            .pointer(ids)
            .pointer(weights)
            .int32(rows)
            .int32(experts)
            .int32(k),
    ))
}

/// `output = bf16(bf16(silu(gate)) * up)` over `count` values (`MoeOps.swiGlu`).
pub fn swiglu(gate: DevicePtr, up: DevicePtr, output: DevicePtr, count: u32) -> Result<Launch, FlashError> {
    const OP: &str = "swiglu";
    positive(OP, &[("count", count)])?;
    Ok(Qwen4Kernel::Swiglu.launch(
        [blocks(OP, u64::from(count), u64::from(WIDE_BLOCK))?, 1, 1],
        [WIDE_BLOCK, 1, 1],
        KernelArguments::new()
            .pointer(gate)
            .pointer(up)
            .pointer(output)
            .int32(count),
    ))
}

/// The SwiGLU of [`swiglu`] for `rows` rows of `cols` values, written in rows of `padded` values whose columns from
/// `cols` on are 0 (`MoeOps.swiGluPadded`).
pub fn swiglu_padded(
    gate: DevicePtr,
    up: DevicePtr,
    output: DevicePtr,
    rows: u32,
    cols: u32,
    padded: u32,
) -> Result<Launch, FlashError> {
    const OP: &str = "swiglu padded";
    positive(OP, &[("rows", rows), ("cols", cols)])?;
    if padded < cols {
        return Err(FlashError::refused(
            OP,
            format!("the padded width {padded} is below the {cols} columns"),
        ));
    }
    Ok(Qwen4Kernel::SwigluPadded.launch(
        [blocks(OP, u64::from(padded), u64::from(WIDE_BLOCK))?, rows, 1],
        [WIDE_BLOCK, 1, 1],
        KernelArguments::new()
            .pointer(gate)
            .pointer(up)
            .pointer(output)
            .int32(cols)
            .int32(padded),
    ))
}

/// The bytes of a plain NVFP4 tensor of `rows` rows of `k` values: the codes, the scales aligned to 256 bytes and the
/// FP32 global (`MoeOps.nvfp4Bytes`).
pub fn nvfp4_bytes(rows: u32, k: u32) -> u64 {
    let align = |value: u64| (value + 255) & !255;
    let (rows, k) = (u64::from(rows), u64::from(k));
    let scales = align(rows * k / 2);
    align(scales + rows * k / 16) + 4
}

/// The plain NVFP4 tensor `source` of `rows` rows of `k` values copied to `target` in rows of `padded` values: the
/// added columns are zero codes with zero scales (`MoeOps.nvfp4PadK`). `target` holds [`nvfp4_bytes`] of `padded`.
pub fn nvfp4_pad_k(
    source: DevicePtr,
    target: DevicePtr,
    rows: u32,
    k: u32,
    padded: u32,
) -> Result<Launch, FlashError> {
    const OP: &str = "nvfp4 pad";
    positive(OP, &[("rows", rows), ("k", k)])?;
    if !k.is_multiple_of(128) || !padded.is_multiple_of(128) || padded < k {
        return Err(FlashError::refused(
            OP,
            format!("padding {k} values to {padded}: both must be multiples of 128, the padded no less than the width"),
        ));
    }
    Ok(Qwen4Kernel::Nvfp4PadK.launch(
        [rows + 1, 1, 1],
        [128, 1, 1],
        KernelArguments::new()
            .pointer(source)
            .pointer(target)
            .int32(rows)
            .int32(k)
            .int32(padded),
    ))
}

/// `output = routed + sigmoid(gate) * shared` per row, in BF16 (`MoeOps.finish`).
pub fn moe_finish(
    routed: DevicePtr,
    shared: DevicePtr,
    gate: DevicePtr,
    output: DevicePtr,
    rows: u32,
    width: u32,
) -> Result<Launch, FlashError> {
    const OP: &str = "moe finish";
    positive(OP, &[("rows", rows), ("width", width)])?;
    Ok(Qwen4Kernel::MoeFinish.launch(
        [blocks(OP, u64::from(width), u64::from(WIDE_BLOCK))?, rows, 1],
        [WIDE_BLOCK, 1, 1],
        KernelArguments::new()
            .pointer(routed)
            .pointer(shared)
            .pointer(gate)
            .pointer(output)
            .int32(rows)
            .int32(width),
    ))
}

/// Gate and up of each pair's row against the work items' expert, then SwiGLU, into `act` `[pairs][inter]`
/// (`ExpertOps.gateUpSwiGlu`).
pub fn expert_gate_up(
    geometry: &ExpertGeometry,
    items: &ExpertItems,
    input: DevicePtr,
    act: DevicePtr,
) -> Result<Launch, FlashError> {
    const OP: &str = "expert gate up";
    positive(OP, &[("item count", items.item_count)])?;
    aligned(
        OP,
        EXPERT_ALIGNMENT,
        &[
            ("slots", items.slots),
            ("items", items.items),
            ("pairs", items.pairs),
            ("input", input),
            ("act", act),
        ],
    )?;
    Ok(Qwen4Kernel::ExpertGateUpSwiglu.launch(
        [geometry.inter / GATE_UP_COLUMNS, items.item_count, 1],
        [EXPERT_BLOCK, 1, 1],
        KernelArguments::new()
            .pointer(items.slots)
            .pointer(items.items)
            .pointer(items.pairs)
            .pointer(input)
            .pointer(act)
            .int32(geometry.gate_up_offset)
            .int32(geometry.hidden)
            .int32(geometry.inter),
    ))
}

/// The down projection of each pair's `act` row times its routing weight, into `weighted` `[pairs][hidden]`
/// (`ExpertOps.downWeighted`).
pub fn expert_down(
    geometry: &ExpertGeometry,
    items: &ExpertItems,
    act: DevicePtr,
    weighted: DevicePtr,
) -> Result<Launch, FlashError> {
    const OP: &str = "expert down";
    positive(OP, &[("item count", items.item_count)])?;
    aligned(
        OP,
        EXPERT_ALIGNMENT,
        &[
            ("slots", items.slots),
            ("items", items.items),
            ("pairs", items.pairs),
            ("act", act),
            ("weighted", weighted),
        ],
    )?;
    Ok(Qwen4Kernel::ExpertDown.launch(
        [geometry.hidden / DOWN_ROWS, items.item_count, 1],
        [EXPERT_BLOCK, 1, 1],
        KernelArguments::new()
            .pointer(items.slots)
            .pointer(items.items)
            .pointer(items.pairs)
            .pointer(act)
            .pointer(weighted)
            .int32(geometry.down_offset)
            .int32(geometry.inter)
            .int32(geometry.hidden),
    ))
}

/// The rows' weighted outputs added into `output` one BF16 addition at a time, in the order of each row's pair list
/// (`ExpertOps.combine`, with the sums starting at zero: a row without pairs is zeroed).
pub fn expert_combine(
    geometry: &ExpertGeometry,
    weighted: DevicePtr,
    row_offsets: DevicePtr,
    row_pairs: DevicePtr,
    output: DevicePtr,
    rows: u32,
) -> Result<Launch, FlashError> {
    const OP: &str = "expert combine";
    positive(OP, &[("rows", rows)])?;
    aligned(
        OP,
        EXPERT_ALIGNMENT,
        &[
            ("weighted", weighted),
            ("row offsets", row_offsets),
            ("row pairs", row_pairs),
            ("output", output),
        ],
    )?;
    let vectors = geometry.hidden / 8;
    Ok(Qwen4Kernel::ExpertCombine.launch(
        [blocks(OP, u64::from(vectors), u64::from(WIDE_BLOCK))?, rows, 1],
        [WIDE_BLOCK, 1, 1],
        KernelArguments::new()
            .pointer(weighted)
            .pointer(row_offsets)
            .pointer(row_pairs)
            .pointer(output)
            .int32(geometry.hidden)
            .int32(1),
    ))
}

#[cfg(test)]
mod tests;
