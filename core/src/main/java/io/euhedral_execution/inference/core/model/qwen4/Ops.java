package io.euhedral_execution.inference.core.model.qwen4;

import io.euhedral_execution.inference.core.artifact.WeightLayout;
import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.gpu.KernelArguments;
import io.euhedral_execution.inference.core.gpu.ScratchUse;

/// Typed launches of the Flash-Next kernels (native/src/qwen4): the geometry each kernel is written for and the
/// argument checks the native launcher cannot make (alignment, extents). Every method queues on the GPU's selected
/// stream; none allocates on the hot path.
///
/// The kernels reproduce upstream's BF16 arithmetic, rounding after every tensor operation the way PyTorch does
/// (docs/FLASH_NEXT_EXECUTION.md).
public final class Ops {

    private static final ThreadLocal<KernelArguments> ARGUMENTS = ThreadLocal.withInitial(KernelArguments::new);

    private Ops() {}

    private static KernelArguments arguments() {
        return ARGUMENTS.get().clear();
    }

    /// Whether BF16 linears whose shape allows it run on tensor cores; `EUHEDRAL_QWEN4_LINEAR_TC=0` keeps the
    /// FP32 kernel, which exact numerics always use.
    static final boolean LINEAR_TC = !"0".equals(System.getenv("EUHEDRAL_QWEN4_LINEAR_TC"));

    /// Rows from which a BF16 linear runs on tensor cores: the row count from which the NVFP4 linears leave the
    /// decode kernels too.
    static final int TC_MIN_ROWS = 9;

    /// Whether decode's BF16 linears split K across warps where one warp per column would leave the device idle
    /// (`EUHEDRAL_QWEN4_LINEAR_SPLIT=0`: one warp per column, the exact route's kernel).
    static final boolean LINEAR_SPLIT = !"0".equals(System.getenv("EUHEDRAL_QWEN4_LINEAR_SPLIT"));

    /// The K slices of a decode linear of `k` inputs and `n` outputs, by its shape alone: 8 for outputs too few to
    /// fill the device or reductions of 8192 and more, 4 for reductions of 2048 and more, else 1.
    static int slices(int k, int n) {
        if (n < 64 || k >= 8192) return 8;
        if (k >= 2048 && n < 4096) return 4;
        return 1;
    }

    /// `output[r][j] = bf16(sum_k input[r][k] * weights[j][k])` for BF16 `weights` of `n` rows by `k` columns.

    public static void linearBf16(ExecutionGpu gpu, long input, long weights, long output, int rows, int k, int n) {
        requirePositive(rows, k, n);
        if (k % 8 != 0) throw new IllegalArgumentException("a BF16 linear needs K divisible by 8, got " + k);
        requireAligned(16, input, weights);
        if (LINEAR_TC
                && rows >= TC_MIN_ROWS
                && !gpu.exactNumerics()
                && k % 32 == 0
                && n % 8 == 0
                && (output & 3) == 0) {
            // Tensor cores, for prefill chunks: a decode row would use one of an MMA's 16 rows, and the FP32 kernel
            // streams its weights faster. K is split in four for narrow outputs (a function of the shape alone, so a
            // row's bits do not depend on how many rows run with it), four row tiles per weight load above 16 rows.
            boolean split = n <= 1536 || n % 32 != 0;
            boolean tiles = rows > 16;
            Kernel kernel = split
                    ? (tiles ? Kernel.LINEAR_TC_ROWS_SPLIT_BF16 : Kernel.LINEAR_TC_SPLIT_BF16)
                    : (tiles ? Kernel.LINEAR_TC_ROWS_BF16 : Kernel.LINEAR_TC_BF16);
            gpu.launchTableKernel(
                    kernel,
                    split ? n / 8 : n / 32,
                    ceilDiv(rows, tiles ? 64 : 16),
                    1,
                    128,
                    1,
                    1,
                    0,
                    arguments()
                            .pointer(input)
                            .pointer(weights)
                            .pointer(output)
                            .int32(rows)
                            .int32(k)
                            .int32(n));
            return;
        }
        int slices = rows < TC_MIN_ROWS && LINEAR_SPLIT && !gpu.exactNumerics() ? slices(k, n) : 1;
        if (slices > 1) {
            gpu.launchTableKernel(
                    slices == 8 ? Kernel.LINEAR_SPLIT8_BF16 : Kernel.LINEAR_SPLIT4_BF16,
                    ceilDiv(n, 8 / slices),
                    ceilDiv(rows, 8),
                    1,
                    256,
                    1,
                    1,
                    0,
                    arguments()
                            .pointer(input)
                            .pointer(weights)
                            .pointer(output)
                            .int32(rows)
                            .int32(k)
                            .int32(n));
            return;
        }
        gpu.launchTableKernel(
                Kernel.LINEAR_BF16,
                ceilDiv(n, 8),
                ceilDiv(rows, 8),
                1,
                256,
                1,
                1,
                0,
                arguments()
                        .pointer(input)
                        .pointer(weights)
                        .pointer(output)
                        .int32(rows)
                        .int32(k)
                        .int32(n));
    }

    /// Qwen4ExpTextRMSNorm with a group size: every one of `groups` groups of `width` values in each of `rows`
    /// rows is normalized separately and scaled by `1 + weight`, the weight covering the whole row.
    public static void groupedRmsNorm(
            ExecutionGpu gpu, long input, long weight, long output, int rows, int groups, int width, float epsilon) {
        requirePositive(rows, groups, width);
        gpu.launchTableKernel(
                Kernel.GROUPED_RMS_NORM_BF16,
                Math.multiplyExact(rows, groups),
                1,
                1,
                256,
                1,
                1,
                0,
                arguments()
                        .pointer(input)
                        .pointer(weight)
                        .pointer(output)
                        .int32(rows)
                        .int32(groups)
                        .int32(width)
                        .float32(epsilon));
    }

    /// `output = bf16(silu(bf16(input / divisor)))` over `count` values.
    public static void scaledSilu(ExecutionGpu gpu, long input, long output, int count, float divisor) {
        requirePositive(count);
        gpu.launchTableKernel(
                Kernel.SCALED_SILU_BF16,
                ceilDiv(count, 256),
                1,
                1,
                256,
                1,
                1,
                0,
                arguments().pointer(input).pointer(output).int32(count).float32(divisor));
    }

    /// The mixed block input of a gated residual: `mean over streams of bf16(sigmoid(up) * normed)`.
    public static void hcMix(ExecutionGpu gpu, long normed, long up, long mixed, int rows, int streams, int width) {
        requirePositive(rows, streams, width);
        gpu.launchTableKernel(
                Kernel.HC_MIX_BF16,
                ceilDiv(width, 256),
                rows,
                1,
                256,
                1,
                1,
                0,
                arguments()
                        .pointer(normed)
                        .pointer(up)
                        .pointer(mixed)
                        .int32(rows)
                        .int32(streams)
                        .int32(width));
    }

    /// Injects a block's result into every stream: `hyper + block * 2 * sigmoid(raw / streams)`.
    public static void hcInject(
            ExecutionGpu gpu,
            long hyper,
            long block,
            long rawInjection,
            long output,
            int rows,
            int streams,
            int width) {
        requirePositive(rows, streams, width);
        gpu.launchTableKernel(
                Kernel.HC_INJECT_BF16,
                ceilDiv(width, 256),
                rows,
                1,
                256,
                1,
                1,
                0,
                arguments()
                        .pointer(hyper)
                        .pointer(block)
                        .pointer(rawInjection)
                        .pointer(output)
                        .int32(rows)
                        .int32(streams)
                        .int32(width));
    }

    /// `output[r][s] = input[r]` for every stream `s`: the embedding row becomes the initial residual state.
    public static void repeatStreams(ExecutionGpu gpu, long input, long output, int rows, int streams, int width) {
        requirePositive(rows, streams, width);
        long total = (long) rows * streams * width;
        gpu.launchTableKernel(
                Kernel.REPEAT_STREAMS_BF16,
                Math.toIntExact(ceilDiv(total, 256)),
                1,
                1,
                256,
                1,
                1,
                0,
                arguments()
                        .pointer(input)
                        .pointer(output)
                        .int32(rows)
                        .int32(streams)
                        .int32(width));
    }

    /// Expands staged n-gram records (see [io.euhedral_execution.inference.core.model.qwen4.loader.NgramStore]) into
    /// BF16 rows of `width` values, one after another.
    public static void ngramExpand(ExecutionGpu gpu, long records, long output, int count, int width, int recordBytes) {
        requirePositive(count, width, recordBytes);
        if (recordBytes % 4 != 0) throw new IllegalArgumentException("records are multiples of four bytes");
        gpu.launchTableKernel(
                Kernel.NGRAM_EXPAND_BF16,
                count,
                1,
                1,
                256,
                1,
                1,
                0,
                arguments()
                        .pointer(records)
                        .pointer(output)
                        .int32(count)
                        .int32(width)
                        .int32(recordBytes));
    }

    /// The PLE gate: every stream's share of the shared value, `sigmoid(signed sqrt(key . query / sqrt(width))) *
    /// value`.
    public static void pleGate(
            ExecutionGpu gpu, long key, long query, long value, long output, int rows, int streams, int width) {
        requirePositive(rows, streams, width);
        gpu.launchTableKernel(
                Kernel.PLE_GATE_BF16,
                Math.multiplyExact(rows, streams),
                1,
                1,
                256,
                1,
                1,
                0,
                arguments()
                        .pointer(key)
                        .pointer(query)
                        .pointer(value)
                        .pointer(output)
                        .int32(rows)
                        .int32(streams)
                        .int32(width));
    }

    /// The PLE short convolution with `(taps - 1) * dilation` rows of history: `gated + silu(conv(normed))`. The
    /// history is read, not written; call [#convHistory] afterwards.
    public static void pleConv(
            ExecutionGpu gpu,
            long normed,
            long gated,
            long history,
            long weights,
            long output,
            int rows,
            int channels,
            int taps,
            int dilation) {
        requirePositive(rows, channels, taps, dilation);
        gpu.launchTableKernel(
                Kernel.PLE_CONV_BF16,
                ceilDiv(channels, 256),
                rows,
                1,
                256,
                1,
                1,
                0,
                arguments()
                        .pointer(normed)
                        .pointer(gated)
                        .pointer(history)
                        .pointer(weights)
                        .pointer(output)
                        .int32(rows)
                        .int32(channels)
                        .int32(taps)
                        .int32(dilation));
    }

    /// Replaces the `history` rows at `historyRows` with the last `history` rows of (history ++ x).
    public static void convHistory(ExecutionGpu gpu, long x, long historyRows, int rows, int channels, int history) {
        requirePositive(rows, channels, history);
        if (history > 32) throw new IllegalArgumentException("the history kernel keeps at most 32 rows");
        gpu.launchTableKernel(
                Kernel.CONV_HISTORY_BF16,
                ceilDiv(channels, 256),
                1,
                1,
                256,
                1,
                1,
                0,
                arguments()
                        .pointer(x)
                        .pointer(historyRows)
                        .int32(rows)
                        .int32(channels)
                        .int32(history));
    }

    /// `output[r] = table[ids[r]]`: embedding rows of `width` BF16 values from a table of `vocabulary` rows that is in
    /// device memory or mapped host memory; `ids` are int32 token ids on the device.
    public static void embedding(
            ExecutionGpu gpu, long table, long ids, long output, int rows, int width, int vocabulary) {
        requirePositive(rows, width, vocabulary);
        if (width % 8 != 0) throw new IllegalArgumentException("embedding rows are multiples of 8 values");
        requireAligned(16, table, output);
        gpu.launchTableKernel(
                Kernel.EMBEDDING_BF16,
                rows,
                1,
                1,
                256,
                1,
                1,
                0,
                arguments()
                        .pointer(table)
                        .pointer(ids)
                        .pointer(output)
                        .int32(rows)
                        .int32(width)
                        .int32(vocabulary));
    }

    private static int ceilDiv(int value, int divisor) {
        return (value + divisor - 1) / divisor;
    }

    private static long ceilDiv(long value, long divisor) {
        return (value + divisor - 1) / divisor;
    }

    private static void requirePositive(int... values) {
        for (int value : values) if (value <= 0) throw new IllegalArgumentException("extents must be positive");
    }

    private static void requireAligned(int alignment, long... addresses) {
        for (long address : addresses) {
            if (address == 0 || address % alignment != 0)
                throw new IllegalArgumentException("address " + address + " is not " + alignment + "-byte aligned");
        }
    }

    /// Expansion scratch the NVFP4 linears `(inFeatures, outFeatures)` of `shapes` take at `rows` rows: the
    /// largest of them, since a stage runs its linears one after another.
    static long nvfp4Scratch(ExecutionGpu gpu, int rows, int... shapes) {
        long bytes = 0;
        for (int i = 0; i < shapes.length; i += 2)
            bytes = Math.max(
                    bytes,
                    gpu.scratchBytes(
                            ScratchUse.NVFP4_LINEAR, rows, shapes[i], shapes[i + 1], WeightLayout.ROW_SPLIT_K128_V1));
        return bytes;
    }
}
