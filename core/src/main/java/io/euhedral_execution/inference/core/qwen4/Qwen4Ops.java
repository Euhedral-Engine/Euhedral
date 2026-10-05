package io.euhedral_execution.inference.core.qwen4;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.gpu.Qwen4Kernel;
import io.euhedral_execution.inference.core.gpu.Qwen4KernelArguments;

/// Typed launches of the Flash-Next kernels (native/src/qwen4): the geometry each kernel is written for and the
/// argument checks the native launcher cannot make (alignment, extents). Every method queues on the GPU's selected
/// stream; none allocates on the hot path.
///
/// The kernels reproduce upstream's BF16 arithmetic, rounding after every tensor operation the way PyTorch does
/// (docs/FLASH_NEXT_EXECUTION.md).
public final class Qwen4Ops {

    private static final ThreadLocal<Qwen4KernelArguments> ARGUMENTS =
            ThreadLocal.withInitial(Qwen4KernelArguments::new);

    private Qwen4Ops() {}

    private static Qwen4KernelArguments arguments() {
        return ARGUMENTS.get().clear();
    }

    /// `output[r][j] = bf16(sum_k input[r][k] * weights[j][k])` for BF16 `weights` of `n` rows by `k` columns.
    public static void linearBf16(ExecutionGpu gpu, long input, long weights, long output, int rows, int k, int n) {
        requirePositive(rows, k, n);
        if (k % 8 != 0) throw new IllegalArgumentException("a BF16 linear needs K divisible by 8, got " + k);
        requireAligned(16, input, weights);
        gpu.launchQwen4(
                Qwen4Kernel.LINEAR_BF16,
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
        gpu.launchQwen4(
                Qwen4Kernel.GROUPED_RMS_NORM_BF16,
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
        gpu.launchQwen4(
                Qwen4Kernel.SCALED_SILU_BF16,
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
        gpu.launchQwen4(
                Qwen4Kernel.HC_MIX_BF16,
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
        gpu.launchQwen4(
                Qwen4Kernel.HC_INJECT_BF16,
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
        gpu.launchQwen4(
                Qwen4Kernel.REPEAT_STREAMS_BF16,
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

    /// Expands staged n-gram records (see [io.euhedral_execution.inference.core.model_loader.qwen4.NgramStore]) into
    /// BF16 rows of `width` values, one after another.
    public static void ngramExpand(ExecutionGpu gpu, long records, long output, int count, int width, int recordBytes) {
        requirePositive(count, width, recordBytes);
        if (recordBytes % 4 != 0) throw new IllegalArgumentException("records are multiples of four bytes");
        gpu.launchQwen4(
                Qwen4Kernel.NGRAM_EXPAND_BF16,
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
        gpu.launchQwen4(
                Qwen4Kernel.PLE_GATE_BF16,
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
        gpu.launchQwen4(
                Qwen4Kernel.PLE_CONV_BF16,
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
        gpu.launchQwen4(
                Qwen4Kernel.CONV_HISTORY_BF16,
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
}
