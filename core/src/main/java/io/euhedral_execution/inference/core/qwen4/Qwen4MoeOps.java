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
