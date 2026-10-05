package io.euhedral_execution.inference.core.qwen4;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.gpu.Qwen4Kernel;
import io.euhedral_execution.inference.core.gpu.Qwen4KernelArguments;

/// The Gated DeltaNet token mixer of a Flash-Next layer (Qwen4ExpTextGatedDeltaNet), over the dense engine's
/// delta-rule recurrence:
///
/// ```
/// qkv = in_proj_qkv(x)  z = in_proj_z(x)  a = in_proj_a(x)  b = in_proj_b(x)
/// conv   = silu(depthwise causal conv(qkv))               carries the last (taps - 1) rows of qkv
/// alpha  = exp(-exp(A_log) * softplus(a + dt_bias))       beta = sigmoid(b)
/// core   = recurrence(conv.q, conv.k, conv.v; alpha, beta)   state [value head][column][key] in FP32
/// out    = out_proj(rmsnorm(core) * act(z))               act is the model's output gate (sigmoid)
/// ```
///
/// Rows are processed in order with the state carried, so any chunking of a sequence gives the same result.
public final class Qwen4GdnLayer {

    /// The layer's weights: NVFP4 projections and BF16 small tensors.
    public record Weights(
            Qwen4Weight inProjQkv,
            Qwen4Weight inProjZ,
            Qwen4Weight outProj,
            Qwen4Weight inProjA,
            Qwen4Weight inProjB,
            Qwen4Weight convolution,
            Qwen4Weight aLog,
            Qwen4Weight dtBias,
            Qwen4Weight norm) {}

    /// What a sequence carries: the convolution history (BF16 rows) and the recurrent state (FP32).
    public record State(long convolutionHistory, long recurrent) {}

    /// The device buffers one call writes ([#scratchBytes]).
    public record Scratch(
            long qkv, long z, long a, long b, long convolved, long alpha, long beta, long core, long normed) {}

    private final int hidden;
    private final int keyHeads;
    private final int valueHeads;
    private final int keyHeadDim;
    private final int valueHeadDim;
    private final int taps;
    private final float epsilon;
    private final int outputGateActivation;

    /// `outputGate` is the config's activation name of the output gate: "sigmoid" or "silu".
    public Qwen4GdnLayer(
            int hidden,
            int keyHeads,
            int valueHeads,
            int keyHeadDim,
            int valueHeadDim,
            int taps,
            float epsilon,
            String outputGate) {
        if (keyHeadDim != 128 || valueHeadDim != 128)
            throw new IllegalArgumentException("the recurrence kernels are written for 128-wide heads");
        if (valueHeads % keyHeads != 0)
            throw new IllegalArgumentException("value heads must be a multiple of key heads");
        if (taps < 2 || taps > 32) throw new IllegalArgumentException("convolution kernel size " + taps);
        this.hidden = hidden;
        this.keyHeads = keyHeads;
        this.valueHeads = valueHeads;
        this.keyHeadDim = keyHeadDim;
        this.valueHeadDim = valueHeadDim;
        this.taps = taps;
        this.epsilon = epsilon;
        this.outputGateActivation = switch (outputGate) {
            case "silu" -> 0;
            case "sigmoid" -> 1;
            default -> throw new IllegalArgumentException("unsupported GDN output gate " + outputGate);
        };
    }

    public int convolutionChannels() {
        return 2 * this.keyHeads * this.keyHeadDim + this.valueHeads * this.valueHeadDim;
    }

    public int valueWidth() {
        return this.valueHeads * this.valueHeadDim;
    }

    /// Rows of convolution history a sequence keeps.
    public int historyRows() {
        return this.taps - 1;
    }

    public long convolutionHistoryBytes() {
        return (long) historyRows() * convolutionChannels() * Short.BYTES;
    }

    public long recurrentBytes() {
        return (long) this.valueHeads * this.valueHeadDim * this.keyHeadDim * Float.BYTES;
    }

    /// Allocates a zeroed state on the device.
    public State allocateState(ExecutionGpu gpu) {
        long history = gpu.allocate(convolutionHistoryBytes());
        long recurrent = 0;
        try {
            recurrent = gpu.allocate(recurrentBytes());
            reset(gpu, new State(history, recurrent));
        } catch (Throwable failure) {
            if (recurrent != 0) gpu.free(recurrent);
            gpu.free(history);
            throw failure;
        }
        return new State(history, recurrent);
    }

    public void freeState(ExecutionGpu gpu, State state) {
        gpu.free(state.recurrent());
        gpu.free(state.convolutionHistory());
    }

    /// Zeroes a state for a new sequence.
    public void reset(ExecutionGpu gpu, State state) {
        gpu.zeroDeviceMemory(state.convolutionHistory(), convolutionHistoryBytes());
        gpu.zeroDeviceMemory(state.recurrent(), recurrentBytes());
    }

    public long scratchBytes(int rows) {
        long bf16 = Short.BYTES;
        return (long) rows
                * (2L * convolutionChannels() * bf16 // qkv, convolved
                        + 3L * valueWidth() * bf16 // z, core, normed
                        + 2L * this.valueHeads * bf16 // a, b
                        + 2L * this.valueHeads * Float.BYTES); // alpha, beta
    }

    public Scratch scratch(long base, int rows) {
        long channels = convolutionChannels();
        long qkv = base;
        long convolved = qkv + (long) rows * channels * Short.BYTES;
        long z = convolved + (long) rows * channels * Short.BYTES;
        long core = z + (long) rows * valueWidth() * Short.BYTES;
        long normed = core + (long) rows * valueWidth() * Short.BYTES;
        long a = normed + (long) rows * valueWidth() * Short.BYTES;
        long b = a + (long) rows * this.valueHeads * Short.BYTES;
        long alpha = b + (long) rows * this.valueHeads * Short.BYTES;
        long beta = alpha + (long) rows * this.valueHeads * Float.BYTES;
        return new Scratch(qkv, z, a, b, convolved, alpha, beta, core, normed);
    }

    /// Mixes `rows` rows at `input` (rows x hidden) into `output` (rows x hidden), advancing `state`.
    public void run(
            ExecutionGpu gpu, Weights weights, State state, long input, int rows, Scratch scratch, long output) {
        int channels = convolutionChannels();
        int valueWidth = valueWidth();
        Qwen4Weight qkv = weights.inProjQkv();
        gpu.linearNvfp4Bf16(input, qkv.address(), scratch.qkv(), rows, this.hidden, channels, qkv.bytes());
        Qwen4Weight z = weights.inProjZ();
        gpu.linearNvfp4Bf16(input, z.address(), scratch.z(), rows, this.hidden, valueWidth, z.bytes());
        Qwen4Ops.linearBf16(gpu, input, weights.inProjA().address(), scratch.a(), rows, this.hidden, this.valueHeads);
        Qwen4Ops.linearBf16(gpu, input, weights.inProjB().address(), scratch.b(), rows, this.hidden, this.valueHeads);
        Qwen4GdnOps.convolution(
                gpu,
                scratch.qkv(),
                state.convolutionHistory(),
                weights.convolution().address(),
                scratch.convolved(),
                rows,
                channels,
                this.taps);
        Qwen4Ops.convHistory(gpu, scratch.qkv(), state.convolutionHistory(), rows, channels, historyRows());
        Qwen4GdnOps.control(
                gpu,
                scratch.a(),
                scratch.b(),
                weights.aLog().address(),
                weights.dtBias().address(),
                scratch.alpha(),
                scratch.beta(),
                rows,
                this.valueHeads);
        gpu.gdnRecurrenceBf16(
                scratch.convolved(),
                scratch.alpha(),
                scratch.beta(),
                state.recurrent(),
                scratch.core(),
                rows,
                this.keyHeads,
                this.valueHeads,
                this.keyHeadDim,
                this.valueHeadDim,
                (float) (1.0 / Math.sqrt(this.keyHeadDim)));
        Qwen4GdnOps.gatedNorm(
                gpu,
                scratch.core(),
                scratch.z(),
                weights.norm().address(),
                scratch.normed(),
                rows,
                this.valueHeads,
                this.valueHeadDim,
                this.epsilon,
                this.outputGateActivation);
        Qwen4Weight out = weights.outProj();
        gpu.linearNvfp4Bf16(scratch.normed(), out.address(), output, rows, valueWidth, this.hidden, out.bytes());
    }

    /// Launch wrappers of the layer's own kernels.
    static final class Qwen4GdnOps {
        private static final ThreadLocal<Qwen4KernelArguments> ARGUMENTS =
                ThreadLocal.withInitial(Qwen4KernelArguments::new);

        private Qwen4GdnOps() {}

        static void convolution(
                ExecutionGpu gpu, long x, long history, long weights, long output, int rows, int channels, int taps) {
            gpu.launchQwen4(
                    Qwen4Kernel.GDN_CONV_BF16,
                    (channels + 255) / 256,
                    rows,
                    1,
                    256,
                    1,
                    1,
                    0,
                    ARGUMENTS
                            .get()
                            .clear()
                            .pointer(x)
                            .pointer(history)
                            .pointer(weights)
                            .pointer(output)
                            .int32(rows)
                            .int32(channels)
                            .int32(taps));
        }

        static void control(
                ExecutionGpu gpu, long a, long b, long aLog, long dtBias, long alpha, long beta, int rows, int heads) {
            gpu.launchQwen4(
                    Qwen4Kernel.GDN_CONTROL_BF16,
                    (rows * heads + 255) / 256,
                    1,
                    1,
                    256,
                    1,
                    1,
                    0,
                    ARGUMENTS
                            .get()
                            .clear()
                            .pointer(a)
                            .pointer(b)
                            .pointer(aLog)
                            .pointer(dtBias)
                            .pointer(alpha)
                            .pointer(beta)
                            .int32(rows)
                            .int32(heads));
        }

        static void gatedNorm(
                ExecutionGpu gpu,
                long core,
                long z,
                long weight,
                long output,
                int rows,
                int heads,
                int headDim,
                float epsilon,
                int activation) {
            gpu.launchQwen4(
                    Qwen4Kernel.GDN_GATED_NORM_BF16,
                    rows * heads,
                    1,
                    1,
                    128,
                    1,
                    1,
                    0,
                    ARGUMENTS
                            .get()
                            .clear()
                            .pointer(core)
                            .pointer(z)
                            .pointer(weight)
                            .pointer(output)
                            .int32(rows)
                            .int32(heads)
                            .int32(headDim)
                            .float32(epsilon)
                            .int32(activation));
        }
    }
}
