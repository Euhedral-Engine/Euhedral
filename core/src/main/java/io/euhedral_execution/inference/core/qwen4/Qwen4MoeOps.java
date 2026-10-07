package io.euhedral_execution.inference.core.qwen4;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.gpu.Qwen4Kernel;
import io.euhedral_execution.inference.core.gpu.Qwen4KernelArguments;

/// Launches of the routing and shared-expert kernels of the MoE block.
public final class Qwen4MoeOps {

    private static final ThreadLocal<Qwen4KernelArguments> ARGUMENTS =
            ThreadLocal.withInitial(Qwen4KernelArguments::new);

    private Qwen4MoeOps() {}

    private static Qwen4KernelArguments arguments() {
        return ARGUMENTS.get().clear();
    }

    /// Selects the `k` experts of every row from the BF16 router logits: `ids` (int32) and `weights` (BF16 bits),
    /// `rows` x `k` each. Ties go to the lower expert.
    public static void router(ExecutionGpu gpu, long logits, long ids, long weights, int rows, int experts, int k) {
        if (rows <= 0 || experts <= 0 || experts > 1024 || k <= 0 || k > 16 || k > experts)
            throw new IllegalArgumentException("router shape " + rows + " x " + experts + " top " + k);
        gpu.launchQwen4(
                Qwen4Kernel.ROUTER_BF16,
                rows,
                1,
                1,
                256,
                1,
                1,
                0,
                arguments()
                        .pointer(logits)
                        .pointer(ids)
                        .pointer(weights)
                        .int32(rows)
                        .int32(experts)
                        .int32(k));
    }

    /// `output = bf16(bf16(silu(gate)) * up)` over `count` values.
    public static void swiGlu(ExecutionGpu gpu, long gate, long up, long output, int count) {
        if (count <= 0) throw new IllegalArgumentException("count must be positive");
        gpu.launchQwen4(
                Qwen4Kernel.SWIGLU_BF16,
                (count + 255) / 256,
                1,
                1,
                256,
                1,
                1,
                0,
                arguments().pointer(gate).pointer(up).pointer(output).int32(count));
    }

    /// The SwiGLU of [#swiGlu] for `rows` rows of `cols` values, written in rows of `padded` values whose columns at
    /// and past `cols` are 0.
    public static void swiGluPadded(ExecutionGpu gpu, long gate, long up, long output, int rows, int cols, int padded) {
        if (rows <= 0 || cols <= 0 || padded < cols) throw new IllegalArgumentException("invalid padded SwiGLU shape");
        gpu.launchQwen4(
                Qwen4Kernel.SWIGLU_PADDED_BF16,
                (padded + 255) / 256,
                rows,
                1,
                256,
                1,
                1,
                0,
                arguments()
                        .pointer(gate)
                        .pointer(up)
                        .pointer(output)
                        .int32(cols)
                        .int32(padded));
    }

    /// Copies the plain NVFP4 tensor `source` (`rows` rows of `k` values) into `target`, rows of `padded` values whose
    /// added columns are zero codes with zero scales. `target` holds [#nvfp4Bytes]`(rows, padded)` bytes.
    public static void nvfp4PadK(ExecutionGpu gpu, long source, long target, int rows, int k, int padded) {
        if (rows <= 0 || k <= 0 || k % 128 != 0 || padded % 128 != 0 || padded < k)
            throw new IllegalArgumentException("invalid NVFP4 padding " + k + " to " + padded);
        gpu.launchQwen4(
                Qwen4Kernel.NVFP4_PAD_K,
                rows + 1,
                1,
                1,
                128,
                1,
                1,
                0,
                arguments().pointer(source).pointer(target).int32(rows).int32(k).int32(padded));
    }

    /// Bytes of a plain NVFP4 tensor of `rows` rows of `k` values (`k` a multiple of 128): codes, 256-aligned E4M3
    /// scales, and the 256-aligned FP32 global.
    public static long nvfp4Bytes(int rows, int k) {
        long scales = align256((long) rows * k / 2);
        return align256(scales + (long) rows * k / 16) + 4;
    }

    private static long align256(long value) {
        return (value + 255) & ~255L;
    }

    /// `output = routed + sigmoid(gate) * shared` per row, in BF16 as upstream rounds it.
    public static void finish(ExecutionGpu gpu, long routed, long shared, long gate, long output, int rows, int width) {
        if (rows <= 0 || width <= 0) throw new IllegalArgumentException("extents must be positive");
        gpu.launchQwen4(
                Qwen4Kernel.MOE_FINISH_BF16,
                (width + 255) / 256,
                rows,
                1,
                256,
                1,
                1,
                0,
                arguments()
                        .pointer(routed)
                        .pointer(shared)
                        .pointer(gate)
                        .pointer(output)
                        .int32(rows)
                        .int32(width));
    }
}
