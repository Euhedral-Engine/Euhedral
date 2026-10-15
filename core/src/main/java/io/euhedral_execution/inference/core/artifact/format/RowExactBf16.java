package io.euhedral_execution.inference.core.artifact.format;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.gpu.KernelArguments;
import io.euhedral_execution.inference.core.gpu.KernelTable;
import io.euhedral_execution.inference.core.gpu.TableKernel;

/// The row-exact BF16 linear: tensor-core kernels for every row count, K split by the shape alone and four row
/// tiles per weight load above 16 rows, so each row's bits are those of its one-row launch. Every format that
/// multiplies by BF16 weights (BF16 projections, a rotated trellis segment) launches it from here.
public final class RowExactBf16 {

    /// The kernels of the Flash-Next table this launch uses, at their table positions; a test compares the names.
    enum Kernel implements TableKernel {
        TC(30, "euhedral_q4_linear_tc_bf16"),
        TC_SPLIT(31, "euhedral_q4_linear_tc_split_bf16"),
        TC_ROWS(32, "euhedral_q4_linear_tc_rows_bf16"),
        TC_ROWS_SPLIT(33, "euhedral_q4_linear_tc_rows_split_bf16");

        private final int index;
        private final String symbol;

        Kernel(int index, String symbol) {
            this.index = index;
            this.symbol = symbol;
        }

        @Override
        public KernelTable table() {
            return KernelTable.QWEN4;
        }

        @Override
        public int index() {
            return this.index;
        }

        @Override
        public String symbol() {
            return this.symbol;
        }
    }

    private static final ThreadLocal<KernelArguments> ARGUMENTS = ThreadLocal.withInitial(KernelArguments::new);

    private RowExactBf16() {}

    /// `output[r][j] = bf16(sum_k input[r][k] * weights[j][k])` for BF16 `weights` of `n` rows by `k` columns.
    /// Needs K divisible by 32 and N by 8.
    public static void linear(ExecutionGpu gpu, long input, long weights, long output, int rows, int k, int n) {
        if (rows <= 0 || k <= 0 || n <= 0) throw new IllegalArgumentException("extents must be positive");
        if (k % 32 != 0 || n % 8 != 0)
            throw new IllegalArgumentException(
                    "a row-exact BF16 linear needs K divisible by 32 and N by 8, got " + k + " x " + n);
        requireAligned(16, input, weights);
        if ((output & 3) != 0) throw new IllegalArgumentException("the output must be 4-byte aligned");
        boolean split = n <= 1536 || n % 32 != 0;
        boolean tiles = rows > 16;
        Kernel kernel = split ? (tiles ? Kernel.TC_ROWS_SPLIT : Kernel.TC_SPLIT) : (tiles ? Kernel.TC_ROWS : Kernel.TC);
        gpu.launchTableKernel(
                kernel,
                split ? n / 8 : n / 32,
                (rows + (tiles ? 64 : 16) - 1) / (tiles ? 64 : 16),
                1,
                128,
                1,
                1,
                0,
                ARGUMENTS
                        .get()
                        .clear()
                        .pointer(input)
                        .pointer(weights)
                        .pointer(output)
                        .int32(rows)
                        .int32(k)
                        .int32(n));
    }

    private static void requireAligned(int alignment, long... addresses) {
        for (long address : addresses) {
            if (address == 0 || address % alignment != 0)
                throw new IllegalArgumentException("address " + address + " is not " + alignment + "-byte aligned");
        }
    }
}
