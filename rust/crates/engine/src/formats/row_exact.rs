//! The row-exact BF16 linear every format that multiplies by BF16 weights launches (`RowExactBf16`): the BF16
//! projections, and a rotated trellis segment of the prefill route.

use euhedral_inference_cuda::{CudaError, DevicePtr};

use crate::formats::shapes::{KernelArguments, Launch, Table};

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

/// The row-exact BF16 linear: `output[r][j] = bf16(sum_k input[r][k] * weights[j][k])` for BF16 `weights` of `n` rows
/// by `k` columns (`RowExactBf16.linear`). Tensor-core kernels for every row count, K split by the shape alone and
/// four row tiles per weight load above 16 rows, so each row's bits are those of its one-row launch. Needs K divisible
/// by 32 and N by 8.
pub(super) fn row_exact_bf16(
    input: DevicePtr,
    weights: DevicePtr,
    output: DevicePtr,
    rows: u64,
    k: u64,
    n: u64,
    sink: &mut dyn FnMut(Launch) -> Result<(), CudaError>,
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

/// A value as a launch argument or extent: a 32-bit signed integer, so a value past 2^31 - 1 is refused.
pub(super) fn dim(value: u64) -> Result<u32, CudaError> {
    i32::try_from(value)
        .map(i32::unsigned_abs)
        .map_err(|_| argument(format!("{value} does not fit a 32-bit launch argument")))
}

pub(super) fn argument(message: String) -> CudaError {
    CudaError::Argument(message)
}
