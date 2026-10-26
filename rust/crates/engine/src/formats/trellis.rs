//! Trellis (EXL3) linears: the routes of `TrellisFormat.linear`.
//!
//! One linear is a list of launches chosen from the object's shape, its rows and the thresholds:
//!
//! - **Exact numerics**: the scalar reference alone, one block per output element.
//! - **Tensor-core route** (a verification, or at most `kernels.trellis.tensor-core-max-tokens` rows): the prepass
//!   rotates the rows by `suh` into FP16 (or block-scaled E4M3), then the GEMM decodes the trellis tiles in registers.
//!   Below `whole-k-min-tokens` rows the K is split over `kernels.trellis.target-blocks` blocks and an epilogue sums the
//!   partials in order; from there each block runs every split itself, in blocks of 32 rows, or of 64 from
//!   `wide-block-min-tokens`. A verification never takes the FP8 kernels: its rows must equal one-row launches bit for
//!   bit.
//! - **Prefill route** (more rows, not a verification): the object decoded to BF16 in rotated space, multiplied per
//!   segment on the row-exact BF16 GEMM, then the epilogue.
//!
//! The split depends on the shape and the target alone, never on the rows.

use std::sync::atomic::{AtomicU8, Ordering};

use euhedral_inference_cuda::{CudaError, CudaLibrary, DevicePtr, Submission};
use euhedral_inference_edrl::WeightFormat;
use euhedral_inference_settings::{Auto, TrellisKernelSettings};

use crate::formats::shapes::{KernelArguments, Launch, LinearKernel, ShapeRecorder, Table};
use crate::formats::{LinearFormat, LinearMode, ScratchRegion};
use crate::model::{WeightRef, trellis_logical};
use crate::residency::scratch::{
    MAX_SEGMENTS, TRELLIS_BLOCK, TrellisThresholds, splits, trellis_mma_bytes, trellis_prefill_bytes,
};

/// The most rows the launch grids' Y extent holds.
const MAX_GRID_ROWS: u64 = 65535;

/// Columns per block of the prepass: one 128-column block per warp, four warps.
const PREPASS_BLOCK_COLUMNS: u64 = 512;

/// Columns per block of the epilogue: one 128-column block per warp, four warps.
const EPILOGUE_BLOCK_COLUMNS: u64 = 512;

/// Rows per block of the whole-K GEMM (`kSplitRows`, native/src/trellis/mma.cuh).
const WHOLE_K_ROW_BLOCK: u64 = 32;

/// Rows per block of the wide whole-K GEMM (`kSplitRows64`).
const WIDE_WHOLE_K_ROW_BLOCK: u64 = 64;

/// Regions of the scratch are aligned to this many bytes.
const SCRATCH_ALIGNMENT: u64 = 256;

/// Threads of a block of the GEMMs and of the decode.
const GEMM_THREADS: u32 = 256;

/// Threads of a block of the prepass, the epilogues and the reference.
const ROW_THREADS: u32 = 128;

/// Layouts of the prepass's output: FP16 in the GEMM's fragment order, BF16 in column order (the prefill route), or
/// block-scaled E4M3 for the FP8 GEMM.
const FRAGMENT_FP16: u32 = 0;
const COLUMN_BF16: u32 = 1;
const BLOCK_FP8: u32 = 2;

/// The row-exact BF16 GEMMs of the Flash-Next table, at their table positions (`RowExactBf16.Kernel`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RowExactKernel {
    Tc,
    TcSplit,
    TcRows,
    TcRowsSplit,
}

impl RowExactKernel {
    fn index(self) -> i32 {
        match self {
            RowExactKernel::Tc => 30,
            RowExactKernel::TcSplit => 31,
            RowExactKernel::TcRows => 32,
            RowExactKernel::TcRowsSplit => 33,
        }
    }

    fn symbol(self) -> &'static str {
        match self {
            RowExactKernel::Tc => "euhedral_q4_linear_tc_bf16",
            RowExactKernel::TcSplit => "euhedral_q4_linear_tc_split_bf16",
            RowExactKernel::TcRows => "euhedral_q4_linear_tc_rows_bf16",
            RowExactKernel::TcRowsSplit => "euhedral_q4_linear_tc_rows_split_bf16",
        }
    }
}

/// The positions and names of the row-exact BF16 GEMMs in the Flash-Next table: `(index, name)`.
pub const ROW_EXACT_KERNELS: [(usize, &str); 4] = [
    (30, "euhedral_q4_linear_tc_bf16"),
    (31, "euhedral_q4_linear_tc_split_bf16"),
    (32, "euhedral_q4_linear_tc_rows_bf16"),
    (33, "euhedral_q4_linear_tc_rows_split_bf16"),
];

/// The geometry of one trellis object.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrellisShape<'a> {
    /// Outputs.
    pub n: u64,
    /// Inputs.
    pub k: u64,
    /// Words per 16x16 tile.
    pub w: u64,
    /// Segments: the rows of `suh`.
    pub segments: u64,
    /// The segment ends known on the host (the prefill route cuts the object by them).
    pub ends: &'a [u32],
}

/// The device addresses of one launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrellisOperands {
    pub input: DevicePtr,
    pub output: DevicePtr,
    /// The tile words.
    pub words: DevicePtr,
    pub suh: DevicePtr,
    pub svh: DevicePtr,
    /// The segment ends as the device reads them.
    pub ends: DevicePtr,
    pub scratch: ScratchRegion,
}

/// The trellis linears with the thresholds of one device.
#[derive(Debug)]
pub struct TrellisLinear {
    target_blocks: u64,
    tensor_core_max_rows: u64,
    whole_k_rows: u64,
    wide_rows: u64,
    exact_numerics: bool,
    record: bool,
    /// Whether the device has the block-scaled FP8 kernels: 0 not asked yet, 1 yes, 2 no.
    fp8: AtomicU8,
}

impl TrellisLinear {
    /// Linears with `target_blocks` thread blocks to aim the K split for (`kernels.trellis.target-blocks`, 8 per
    /// streaming multiprocessor unless stated), keeping up to `tensor_core_max_tokens` rows of a launch other than a
    /// verification on the tensor-core route, with whole-K blocks from `whole_k_min_tokens` rows and blocks of 64
    /// rows from `wide_block_min_tokens`.
    pub fn new(
        target_blocks: usize,
        tensor_core_max_tokens: usize,
        whole_k_min_tokens: usize,
        wide_block_min_tokens: usize,
    ) -> Self {
        Self {
            target_blocks: target_blocks as u64,
            tensor_core_max_rows: tensor_core_max_tokens as u64,
            whole_k_rows: whole_k_min_tokens as u64,
            wide_rows: wide_block_min_tokens as u64,
            exact_numerics: false,
            record: false,
            fp8: AtomicU8::new(0),
        }
    }

    /// Linears with the thresholds `settings` state on a device of `sm_count` streaming multiprocessors: the target
    /// blocks are 8 per multiprocessor unless stated (`TrellisSettings.of`).
    pub fn from_settings(settings: &TrellisKernelSettings, sm_count: u32) -> Self {
        let target = match settings.target_blocks {
            Auto::Value(blocks) => blocks as usize,
            Auto::Auto => 8 * sm_count as usize,
        };
        Self::new(
            target,
            settings.tensor_core_max_tokens as usize,
            settings.whole_k_min_tokens as usize,
            settings.wide_block_min_tokens as usize,
        )
    }

    /// Linears that run the scalar reference alone (the oracle of the tests; production never selects it).
    pub fn with_exact_numerics(mut self) -> Self {
        self.exact_numerics = true;
        self
    }

    /// Linears that list each launch they queue on [`ShapeRecorder`] (for tests).
    pub fn with_shape_recording(mut self) -> Self {
        self.record = true;
        self
    }

    /// The thresholds as the scratch sizing takes them.
    pub fn thresholds(&self) -> TrellisThresholds {
        TrellisThresholds {
            mma_max_rows: self.tensor_core_max_rows,
            target_ctas: self.target_blocks,
            whole_k_rows: self.whole_k_rows,
        }
    }

    /// The scratch bytes of a launch of up to `rows` rows over an object of `shape`: the tensor-core route's, and, for
    /// a prefill-capable launch above the tensor-core row limit, the larger prefill route's
    /// (`TrellisFormat.scratchBytes` and `decodeScratchBytes`).
    pub fn scratch_for(&self, shape: &TrellisShape<'_>, rows: u64, prefill: bool) -> u64 {
        let thresholds = self.thresholds();
        let mut bytes = trellis_mma_bytes(thresholds, rows, shape.k, shape.n, shape.segments);
        if prefill && rows > self.tensor_core_max_rows {
            bytes = bytes.max(trellis_prefill_bytes(rows, shape.k, shape.n, shape.segments));
        }
        bytes
    }

    /// The launches of one linear of `rows` rows over an object of `shape`, in stream order, given to `sink`.
    /// `block_fp8` is whether the device has the block-scaled FP8 kernels.
    pub fn plan(
        &self,
        shape: &TrellisShape<'_>,
        operands: &TrellisOperands,
        rows: usize,
        mode: LinearMode,
        block_fp8: bool,
        sink: &mut dyn FnMut(Launch) -> Result<(), CudaError>,
    ) -> Result<(), CudaError> {
        self.plan_with(self.exact_numerics, shape, operands, rows, mode, block_fp8, sink)
    }

    /// [`TrellisLinear::plan`] with `exact` deciding whether the scalar reference runs: the format's own flag, or the
    /// library's process-wide selection that Java's `gpu.exactNumerics()` reads.
    #[allow(clippy::too_many_arguments, reason = "plan's arguments and the exact selection")]
    fn plan_with(
        &self,
        exact: bool,
        shape: &TrellisShape<'_>,
        operands: &TrellisOperands,
        rows: usize,
        mode: LinearMode,
        block_fp8: bool,
        sink: &mut dyn FnMut(Launch) -> Result<(), CudaError>,
    ) -> Result<(), CudaError> {
        let rows = rows as u64;
        if rows == 0 || rows > MAX_GRID_ROWS {
            return Err(argument(format!(
                "a trellis launch takes 1..={MAX_GRID_ROWS} rows: {rows}"
            )));
        }
        let run = Run {
            linear: self,
            shape,
            operands,
            rows,
            row_exact: mode.row_exact,
            block_fp8,
        };
        if exact {
            run.reference(sink)
        } else if mode.row_exact || rows <= self.tensor_core_max_rows {
            run.tensor_core(sink)
        } else {
            run.prefill(sink)
        }
    }

    /// Whether the device has the block-scaled FP8 kernels, asked of the library once.
    ///
    /// Java asks for the one kernel the launch would take (`TRELLIS_MMA_ROWS_FP8` below `wide-block-min-tokens`
    /// rows, `TRELLIS_MMA_ROWS64_FP8` from there). Both live in the one architecture-specific module
    /// (`trellis_fp8/kernels.cu`, sm_12xa), so a device has both or neither; asking for both makes the answer
    /// independent of the rows and cacheable, and the two never differ.
    fn block_fp8(&self, library: &CudaLibrary) -> Result<bool, CudaError> {
        match self.fp8.load(Ordering::Relaxed) {
            1 => Ok(true),
            2 => Ok(false),
            _ => {
                let available = library.linear_kernel_available(LinearKernel::TrellisMmaRowsFp8.index())?
                    && library.linear_kernel_available(LinearKernel::TrellisMmaRows64Fp8.index())?;
                self.fp8.store(if available { 1 } else { 2 }, Ordering::Relaxed);
                Ok(available)
            }
        }
    }
}

/// The shape and operands of a trellis weight, or why it is not one this format runs.
fn parts<'w>(
    weight: &'w WeightRef,
    x: DevicePtr,
    y: DevicePtr,
    scratch: ScratchRegion,
) -> Result<(TrellisShape<'w>, TrellisOperands), CudaError> {
    let (Some(companions), WeightFormat::Trellis) = (weight.trellis.as_ref(), weight.format) else {
        return Err(argument(format!("{} is not a trellis object", weight.name)));
    };
    let (n, k) = trellis_logical(&weight.shape)
        .ok_or_else(|| argument(format!("{} has no [k/16, n/16, w] shape", weight.name)))?;
    let w = weight.shape[2];
    let shape = TrellisShape {
        n,
        k,
        w,
        segments: companions.suh_rows,
        ends: &companions.segment_ends,
    };
    let operands = TrellisOperands {
        input: x,
        output: y,
        words: weight.ptr,
        suh: companions.suh,
        svh: companions.svh,
        ends: companions.segments,
        scratch,
    };
    Ok((shape, operands))
}

impl LinearFormat for TrellisLinear {
    fn submit(
        &self,
        submission: &Submission<'_>,
        weight: &WeightRef,
        x: DevicePtr,
        y: DevicePtr,
        rows: usize,
        scratch: ScratchRegion,
        mode: LinearMode,
    ) -> Result<(), CudaError> {
        let (shape, operands) = parts(weight, x, y, scratch)?;
        // A verification may be selected on the lane as well as asked for by the stage.
        let mode = LinearMode {
            row_exact: mode.row_exact || submission.mode().row_exact,
            ..mode
        };
        let library = submission.lane().library();
        let exact = self.exact_numerics || library.exact_numerics();
        let block_fp8 = !exact && self.block_fp8(library)?;
        self.plan_with(exact, &shape, &operands, rows, mode, block_fp8, &mut |launch| {
            if self.record {
                ShapeRecorder::record(launch.line());
            }
            launch.queue(submission)
        })
    }

    fn scratch_bytes(&self, weight: &WeightRef, rows: usize, mode: LinearMode) -> usize {
        let Ok((shape, _)) = parts(weight, DevicePtr::NULL, DevicePtr::NULL, ScratchRegion::NONE) else {
            return 0;
        };
        self.scratch_for(&shape, rows as u64, mode.prefill) as usize
    }
}

/// One launch's operands and its route chains.
struct Run<'a> {
    linear: &'a TrellisLinear,
    shape: &'a TrellisShape<'a>,
    operands: &'a TrellisOperands,
    rows: u64,
    row_exact: bool,
    block_fp8: bool,
}

type Sink<'s> = &'s mut dyn FnMut(Launch) -> Result<(), CudaError>;

impl Run<'_> {
    fn activations_bytes(&self) -> u64 {
        align_up(self.shape.segments * self.rows * self.shape.k * 2)
    }

    fn reference(&self, sink: Sink<'_>) -> Result<(), CudaError> {
        let shape = self.shape;
        let operands = self.operands;
        sink(
            LinearKernel::TrellisReference.launch(
                [dim(self.rows * shape.n)?, 1, 1],
                [ROW_THREADS, 1, 1],
                dim(4 * shape.k)?,
                KernelArguments::new()
                    .pointer(operands.input)
                    .pointer(operands.words)
                    .pointer(operands.suh)
                    .pointer(operands.svh)
                    .pointer(operands.ends)
                    .int32(dim(shape.segments)?)
                    .pointer(operands.output)
                    .int32(dim(self.rows)?)
                    .int32(dim(shape.k)?)
                    .int32(dim(shape.n)?)
                    .int32(dim(shape.w)?),
            ),
        )
    }

    /// `layout`: [`FRAGMENT_FP16`], [`COLUMN_BF16`] or [`BLOCK_FP8`].
    fn prepass(&self, activations: DevicePtr, layout: u32, sink: Sink<'_>) -> Result<(), CudaError> {
        let shape = self.shape;
        sink(
            LinearKernel::TrellisPrepass.launch(
                [
                    dim(shape.k.div_ceil(PREPASS_BLOCK_COLUMNS))?,
                    dim(self.rows)?,
                    dim(shape.segments)?,
                ],
                [ROW_THREADS, 1, 1],
                0,
                KernelArguments::new()
                    .pointer(self.operands.input)
                    .pointer(self.operands.suh)
                    .pointer(activations)
                    .int32(dim(self.rows)?)
                    .int32(dim(shape.k)?)
                    .int32(dim(shape.segments)?)
                    .int32(layout),
            ),
        )
    }

    fn tensor_core(&self, sink: Sink<'_>) -> Result<(), CudaError> {
        let shape = self.shape;
        let operands = self.operands;
        let thresholds = self.linear.thresholds();
        let splits = splits(shape.k, shape.n, thresholds.target_ctas);
        let activations = self.activations_bytes();
        let scratch = operands.scratch.ptr;
        if self.rows >= self.linear.whole_k_rows {
            self.require_scratch(activations)?;
            let wide = self.rows >= self.linear.wide_rows;
            // A verification's rows must equal one-row launches bit for bit: FP16 operands and FP32 sums. Other
            // launches take block-scaled FP8 operands where the device has them.
            let block_fp8 = !self.row_exact && self.block_fp8;
            self.prepass(scratch, if block_fp8 { BLOCK_FP8 } else { FRAGMENT_FP16 }, sink)?;
            let kernel = match (block_fp8, wide) {
                (true, true) => LinearKernel::TrellisMmaRows64Fp8,
                (true, false) => LinearKernel::TrellisMmaRowsFp8,
                (false, true) => LinearKernel::TrellisMmaRows64,
                (false, false) => LinearKernel::TrellisMmaRows,
            };
            let row_block = if wide {
                WIDE_WHOLE_K_ROW_BLOCK
            } else {
                WHOLE_K_ROW_BLOCK
            };
            return sink(
                kernel.launch(
                    [dim(shape.n / TRELLIS_BLOCK)?, dim(self.rows.div_ceil(row_block))?, 1],
                    [GEMM_THREADS, 1, 1],
                    0,
                    KernelArguments::new()
                        .pointer(scratch)
                        .pointer(operands.words)
                        .pointer(operands.ends)
                        .pointer(operands.svh)
                        .pointer(operands.output)
                        .int32(dim(self.rows)?)
                        .int32(dim(shape.k)?)
                        .int32(dim(shape.n)?)
                        .int32(dim(shape.w)?)
                        .int32(dim(splits)?)
                        .int32(dim(shape.segments)?),
                ),
            );
        }
        let partials = if splits == 1 {
            0
        } else {
            splits * self.rows * shape.n * 4
        };
        self.require_scratch(activations + partials)?;
        let partials_at = scratch.offset(activations);
        self.prepass(scratch, FRAGMENT_FP16, sink)?;
        sink(
            LinearKernel::TrellisMma.launch(
                [dim(shape.n / TRELLIS_BLOCK)?, dim(splits)?, 1],
                [GEMM_THREADS, 1, 1],
                0,
                KernelArguments::new()
                    .pointer(scratch)
                    .pointer(operands.words)
                    .pointer(operands.ends)
                    .pointer(operands.svh)
                    .pointer(partials_at)
                    .pointer(operands.output)
                    .int32(dim(self.rows)?)
                    .int32(dim(shape.k)?)
                    .int32(dim(shape.n)?)
                    .int32(dim(shape.w)?)
                    .int32(dim(splits)?)
                    .int32(dim(shape.segments)?),
            ),
        )?;
        if splits == 1 {
            return Ok(());
        }
        sink(
            LinearKernel::TrellisEpilogue.launch(
                [dim(shape.n.div_ceil(EPILOGUE_BLOCK_COLUMNS))?, dim(self.rows)?, 1],
                [ROW_THREADS, 1, 1],
                0,
                KernelArguments::new()
                    .pointer(partials_at)
                    .int32(dim(splits)?)
                    .pointer(operands.svh)
                    .pointer(operands.output)
                    .int32(dim(self.rows)?)
                    .int32(dim(shape.n)?),
            ),
        )
    }

    fn prefill(&self, sink: Sink<'_>) -> Result<(), CudaError> {
        let shape = self.shape;
        let operands = self.operands;
        let bounds = shape.ends;
        if bounds.len() as u64 != shape.segments {
            return Err(argument(
                "a trellis prefill launch needs the host segment ends of its object".to_string(),
            ));
        }
        validate_segments(bounds, shape.n)?;
        let activations = self.activations_bytes();
        let decoded_bytes = align_up(shape.n * shape.k * 2);
        let rotated_bytes = align_up(self.rows * shape.n * 2);
        self.require_scratch(activations + decoded_bytes + rotated_bytes)?;
        let scratch = operands.scratch.ptr;
        let decoded = scratch.offset(activations);
        let rotated = decoded.offset(decoded_bytes);
        self.prepass(scratch, COLUMN_BF16, sink)?;
        sink(
            LinearKernel::TrellisDecodeBf16.launch(
                [dim((shape.k / 16) * (shape.n / 16))?, 1, 1],
                [GEMM_THREADS, 1, 1],
                0,
                KernelArguments::new()
                    .pointer(operands.words)
                    .pointer(decoded)
                    .int32(dim(shape.k)?)
                    .int32(dim(shape.n)?)
                    .int32(dim(shape.w)?),
            ),
        )?;
        let mut start = 0u64;
        for (segment, end) in bounds.iter().enumerate() {
            let width = u64::from(*end) - start;
            row_exact_bf16(
                scratch.offset(segment as u64 * self.rows * shape.k * 2),
                decoded.offset(start * shape.k * 2),
                rotated.offset(self.rows * start * 2),
                self.rows,
                shape.k,
                width,
                sink,
            )?;
            start = u64::from(*end);
        }
        sink(
            LinearKernel::TrellisEpilogueBf16.launch(
                [dim(shape.n / TRELLIS_BLOCK)?, dim(self.rows)?, 1],
                [ROW_THREADS, 1, 1],
                0,
                KernelArguments::new()
                    .pointer(rotated)
                    .pointer(operands.ends)
                    .int32(dim(shape.segments)?)
                    .pointer(operands.svh)
                    .pointer(operands.output)
                    .int32(dim(self.rows)?)
                    .int32(dim(shape.n)?),
            ),
        )
    }

    fn require_scratch(&self, needed: u64) -> Result<(), CudaError> {
        let scratch = self.operands.scratch;
        if scratch.ptr.is_null() || (scratch.bytes as u64) < needed {
            return Err(argument(format!(
                "a trellis linear of {} rows needs {needed} bytes of scratch; the bound region holds {}",
                self.rows, scratch.bytes
            )));
        }
        Ok(())
    }
}

/// The row-exact BF16 linear: `output[r][j] = bf16(sum_k input[r][k] * weights[j][k])` for BF16 `weights` of `n` rows
/// by `k` columns (`RowExactBf16.linear`). Tensor-core kernels for every row count, K split by the shape alone and
/// four row tiles per weight load above 16 rows, so each row's bits are those of its one-row launch. Needs K divisible
/// by 32 and N by 8.
fn row_exact_bf16(
    input: DevicePtr,
    weights: DevicePtr,
    output: DevicePtr,
    rows: u64,
    k: u64,
    n: u64,
    sink: Sink<'_>,
) -> Result<(), CudaError> {
    if rows == 0 || k == 0 || n == 0 {
        return Err(argument("extents must be positive".to_string()));
    }
    if !k.is_multiple_of(32) || !n.is_multiple_of(8) {
        return Err(argument(format!(
            "a row-exact BF16 linear needs K divisible by 32 and N by 8, got {k} x {n}"
        )));
    }
    for address in [input, weights] {
        if address.is_null() || !address.get().is_multiple_of(16) {
            return Err(argument(format!("address {} is not 16-byte aligned", address.get())));
        }
    }
    if !output.get().is_multiple_of(4) {
        return Err(argument("the output must be 4-byte aligned".to_string()));
    }
    let split = n <= 1536 || !n.is_multiple_of(32);
    let tiles = rows > 16;
    let kernel = match (split, tiles) {
        (true, true) => RowExactKernel::TcRowsSplit,
        (true, false) => RowExactKernel::TcSplit,
        (false, true) => RowExactKernel::TcRows,
        (false, false) => RowExactKernel::Tc,
    };
    let row_tile = if tiles { 64 } else { 16 };
    sink(Launch {
        table: Table::Qwen4,
        kernel: kernel.index(),
        symbol: kernel.symbol(),
        grid: [
            dim(if split { n / 8 } else { n / 32 })?,
            dim(rows.div_ceil(row_tile))?,
            1,
        ],
        block: [128, 1, 1],
        shared_bytes: 0,
        args: KernelArguments::new()
            .pointer(input)
            .pointer(weights)
            .pointer(output)
            .int32(dim(rows)?)
            .int32(dim(k)?)
            .int32(dim(n)?),
    })
}

/// Checks the segment ends of an object of `n` rows (`TrellisLayout.validateSegments`): between 1 and
/// [`MAX_SEGMENTS`] of them, strictly increasing multiples of 128, the last `n`.
fn validate_segments(ends: &[u32], n: u64) -> Result<(), CudaError> {
    if ends.is_empty() || ends.len() as u64 > MAX_SEGMENTS {
        return Err(argument(format!(
            "segment count {} is outside 1..={MAX_SEGMENTS}",
            ends.len()
        )));
    }
    let mut previous = 0u64;
    for (index, end) in ends.iter().enumerate() {
        let end = u64::from(*end);
        if end <= previous {
            return Err(argument(format!("segment ends are not strictly increasing at {index}")));
        }
        if end % TRELLIS_BLOCK != 0 {
            return Err(argument(format!(
                "segment end {end} is not a multiple of {TRELLIS_BLOCK}"
            )));
        }
        previous = end;
    }
    if previous != n {
        return Err(argument(format!("last segment end {previous} is not n={n}")));
    }
    Ok(())
}

fn align_up(value: u64) -> u64 {
    value.div_ceil(SCRATCH_ALIGNMENT) * SCRATCH_ALIGNMENT
}

/// A value as a launch argument or extent: Java's `int`, so a value past 2^31 - 1 is refused as it is there.
fn dim(value: u64) -> Result<u32, CudaError> {
    i32::try_from(value)
        .map(i32::unsigned_abs)
        .map_err(|_| argument(format!("{value} does not fit a 32-bit launch argument")))
}

fn argument(message: String) -> CudaError {
    CudaError::Argument(message)
}
