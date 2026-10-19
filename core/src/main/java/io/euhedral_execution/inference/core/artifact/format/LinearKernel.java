package io.euhedral_execution.inference.core.artifact.format;

import io.euhedral_execution.inference.core.gpu.KernelTable;
import io.euhedral_execution.inference.core.gpu.TableKernel;

/// The kernels of the shared linear table (`euhedral_cuda_linear_launch`, native/src/host/linear_kernels.c). The
/// declaration order is the native table's order, which is its ABI; a test compares the names.
public enum LinearKernel implements TableKernel {
    /// out[g][r] = round(H128(x[r] * suh[g])): FP16 in fragment order, BF16 in column order, or block-scaled E4M3.
    /// Grid (ceil(k/512), rows, s), block 128.
    TRELLIS_PREPASS("euhedral_trellis_prepass"),
    /// Tensor-core GEMM decoding trellis tiles in registers; fused epilogue when S == 1. Grid (n/128, S), block 256.
    TRELLIS_MMA("euhedral_trellis_mma"),
    /// The same GEMM with every K split in one CTA and the epilogue fused, bitwise the split launch's result.
    /// Grid (n/128, ceil(rows/32)), block 256.
    TRELLIS_MMA_ROWS("euhedral_trellis_mma_rows"),
    /// [#TRELLIS_MMA_ROWS] in blocks of 64 rows, two per SM, for the most rows. Grid (n/128, ceil(rows/64)),
    /// block 256.
    TRELLIS_MMA_ROWS64("euhedral_trellis_mma_rows64"),
    /// Sums the K-split partials in order, Hadamard, svh, BF16. Grid (ceil(n/512), rows), block 128.
    TRELLIS_EPILOGUE("euhedral_trellis_epilogue"),
    /// The payload as a BF16 [n][k] rotated weight. Grid (k/16)*(n/16), block 256.
    TRELLIS_DECODE_BF16("euhedral_trellis_decode_bf16"),
    /// Hadamard, svh and BF16 of the prefill route's rotated products. Grid (n/128, rows), block 128.
    TRELLIS_EPILOGUE_BF16("euhedral_trellis_epilogue_bf16"),
    /// The scalar reference, one CTA per output element. Grid rows*n, block 128, dynamic shared 4*k.
    TRELLIS_REFERENCE("euhedral_trellis_reference"),
    /// [#TRELLIS_MMA_ROWS] on block-scaled E4M3 operands (sm_12xa only): row-exact, not the FP16 operands' bits.
    /// Grid (n/128, ceil(rows/32)), block 256.
    TRELLIS_MMA_ROWS_FP8("euhedral_trellis_mma_rows_fp8"),
    /// [#TRELLIS_MMA_ROWS_FP8] in blocks of 64 rows, two per SM. Grid (n/128, ceil(rows/64)), block 256.
    TRELLIS_MMA_ROWS64_FP8("euhedral_trellis_mma_rows64_fp8");

    private final String symbol;

    LinearKernel(String symbol) {
        this.symbol = symbol;
    }

    @Override
    public KernelTable table() {
        return KernelTable.LINEAR;
    }

    @Override
    public int index() {
        return ordinal();
    }

    @Override
    public String symbol() {
        return this.symbol;
    }
}
