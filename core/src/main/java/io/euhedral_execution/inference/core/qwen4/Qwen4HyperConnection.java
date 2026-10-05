package io.euhedral_execution.inference.core.qwen4;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;

/// Qwen4ExpTextGatedResidual: the residual state of a token is `streams` parallel rows of `hidden` values; before
/// each attention and MoE block the streams are normalized, mixed down by a low-rank gate into the block's one input,
/// and afterwards the block's result is injected into every stream with its own weight.
///
/// ```
/// normed = group RMSNorm(streams)                                   [streams * hidden]
/// up     = up_proj(silu(down_proj(normed) / streams))               down: [lowrank], up: [streams * hidden]
/// mixed  = mean over streams of sigmoid(up) * normed                [hidden]
/// raw    = inject_proj(normed)                                      [streams]
/// ... block(mixed) ...
/// state  = streams + block_result * 2 * sigmoid(raw / streams)      every stream
/// ```
///
/// The final mixer of the model has no injection projection: it only mixes.
public final class Qwen4HyperConnection {

    /// The BF16 weights of one gated residual, as device addresses. `inject` is 0 for the final mixer.
    public record Weights(long norm, long down, long up, long inject) {}

    /// Device buffers one mix writes, [#scratchBytes] in all: the normalized streams, the low-rank projection, its
    /// activation, the up projection and the raw injection weights.
    public record Scratch(long normed, long down, long activated, long up, long rawInjection) {}

    private final int streams;
    private final int hidden;
    private final int lowrank;
    private final float epsilon;

    public Qwen4HyperConnection(int streams, int hidden, int lowrank, float epsilon) {
        if (streams <= 1) throw new IllegalArgumentException("hyper-connections need more than one stream");
        this.streams = streams;
        this.hidden = hidden;
        this.lowrank = lowrank;
        this.epsilon = epsilon;
    }

    public int streams() {
        return this.streams;
    }

    public int hidden() {
        return this.hidden;
    }

    /// Width of the residual state of one token.
    public int stateWidth() {
        return this.streams * this.hidden;
    }

    /// Bytes of the scratch for `rows` rows.
    public long scratchBytes(int rows) {
        return (long) rows * (2L * stateWidth() + 2L * this.lowrank + this.streams) * Short.BYTES;
    }

    /// Lays a scratch out over `base`, which must hold [#scratchBytes] bytes and be 16-byte aligned.
    public Scratch scratch(long base, int rows) {
        long normed = base;
        long down = normed + (long) rows * stateWidth() * Short.BYTES;
        long activated = down + (long) rows * this.lowrank * Short.BYTES;
        long up = activated + (long) rows * this.lowrank * Short.BYTES;
        long raw = up + (long) rows * stateWidth() * Short.BYTES;
        return new Scratch(normed, down, activated, up, raw);
    }

    /// Mixes the `rows` residual states at `state` down to `mixed` (`rows` x hidden). Leaves the normalized streams
    /// and, when the weights carry an injection projection, the raw injection weights in `scratch`.
    public void mix(ExecutionGpu gpu, Weights weights, long state, Scratch scratch, long mixed, int rows) {
        int width = stateWidth();
        Qwen4Ops.groupedRmsNorm(
                gpu, state, weights.norm(), scratch.normed(), rows, this.streams, this.hidden, this.epsilon);
        Qwen4Ops.linearBf16(gpu, scratch.normed(), weights.down(), scratch.down(), rows, width, this.lowrank);
        Qwen4Ops.scaledSilu(gpu, scratch.down(), scratch.activated(), rows * this.lowrank, this.streams);
        Qwen4Ops.linearBf16(gpu, scratch.activated(), weights.up(), scratch.up(), rows, this.lowrank, width);
        Qwen4Ops.hcMix(gpu, scratch.normed(), scratch.up(), mixed, rows, this.streams, this.hidden);
        if (weights.inject() != 0)
            Qwen4Ops.linearBf16(
                    gpu, scratch.normed(), weights.inject(), scratch.rawInjection(), rows, width, this.streams);
    }

    /// Injects `block` (rows x hidden) into the states at `state`, writing the new states to `output` (which may be
    /// `state`). `scratch` is the one [#mix] filled for the same rows.
    public void inject(ExecutionGpu gpu, long state, long block, Scratch scratch, long output, int rows) {
        Qwen4Ops.hcInject(gpu, state, block, scratch.rawInjection(), output, rows, this.streams, this.hidden);
    }
}
