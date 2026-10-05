package io.euhedral_execution.inference.core.qwen4;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.model_loader.qwen4.NgramStore;

/// Qwen4ExpTextPLELayer: per-layer embeddings from hashed n-grams of the prompt's tokens, injected into every
/// residual stream.
///
/// ```
/// tokens --host--> n-gram row ids --NgramStore--> staged records --> embedding [embedDim]
/// key   = norm_key(key_proj(embedding))        per stream      value = value_proj(embedding)   shared
/// query = norm_query(streams)                  per stream
/// gate  = sigmoid(signed sqrt(key . query / sqrt(hidden)));  gated = gate * value
/// out   = gated + silu(depthwise dilated conv(norm_conv(gated)))     history of (taps - 1) * dilation rows
/// ```
///
/// The caller adds `out` to the residual streams. The n-gram table never reaches the device whole: only the rows the
/// tokens hash to are gathered, staged and expanded.
///
/// The state a sequence carries between calls is the convolution history (device) and the tokens before the next one
/// (host), [State].
public final class Qwen4Ple {

    /// Where the layer's weights are on the device: NVFP4 projections with their sizes, BF16 norms and convolution.
    public record Weights(
            long keyProj,
            long keyProjBytes,
            long valueProj,
            long valueProjBytes,
            long normKey,
            long normQuery,
            long normConv,
            long convolution) {}

    /// What a sequence carries: the convolution history rows and the token context.
    public static final class State {
        private final long history;
        private final Qwen4NgramIds.Context context;

        public State(long historyAddress, Qwen4NgramIds.Context context) {
            this.history = historyAddress;
            this.context = context;
        }

        public long history() {
            return this.history;
        }

        public Qwen4NgramIds.Context context() {
            return this.context;
        }
    }

    /// The device buffers one call writes ([#scratchBytes]).
    public record Scratch(
            long records,
            long embedding,
            long key,
            long keyNormed,
            long value,
            long query,
            long gated,
            long gatedNormed) {}

    private final NgramStore store;
    private final Qwen4NgramIds ids;
    private final int hidden;
    private final int streams;
    private final int embedDim;
    private final int taps;
    private final int dilation;
    private final float epsilon;
    private long[] rowIds = new long[0];

    public Qwen4Ple(
            NgramStore store,
            Qwen4NgramIds ids,
            int hidden,
            int streams,
            int embedDim,
            int taps,
            int dilation,
            float epsilon) {
        if (store.heads() != ids.heads()) throw new IllegalArgumentException("the id function and the store disagree");
        if ((long) store.heads() * store.rowWidth() != embedDim)
            throw new IllegalArgumentException("the heads' rows must concatenate to the embedding width");
        this.store = store;
        this.ids = ids;
        this.hidden = hidden;
        this.streams = streams;
        this.embedDim = embedDim;
        this.taps = taps;
        this.dilation = dilation;
        this.epsilon = epsilon;
    }

    public int stateWidth() {
        return this.streams * this.hidden;
    }

    /// Rows of convolution history a sequence keeps.
    public int historyRows() {
        return (this.taps - 1) * this.dilation;
    }

    /// Bytes of the convolution history of one sequence.
    public long historyBytes() {
        return (long) historyRows() * stateWidth() * Short.BYTES;
    }

    /// A state for a new sequence over the zeroed history at `historyAddress` ([#historyBytes] bytes).
    public State newState(long historyAddress) {
        return new State(historyAddress, this.ids.newContext());
    }

    /// Resets `state` for a new sequence; the caller zeroes the history.
    public void reset(State state, int endOfSequence) {
        state.context().reset(endOfSequence);
    }

    public long scratchBytes(int rows) {
        long bf16 = Short.BYTES;
        return (long) rows * this.store.heads() * this.store.recordBytes()
                + (long) rows * this.embedDim * bf16
                + (long) rows * (2L * stateWidth() + this.hidden + 3L * stateWidth()) * bf16;
    }

    public Scratch scratch(long base, int rows) {
        long records = base;
        long embedding = records + (long) rows * this.store.heads() * this.store.recordBytes();
        long key = embedding + (long) rows * this.embedDim * Short.BYTES;
        long keyNormed = key + (long) rows * stateWidth() * Short.BYTES;
        long value = keyNormed + (long) rows * stateWidth() * Short.BYTES;
        long query = value + (long) rows * this.hidden * Short.BYTES;
        long gated = query + (long) rows * stateWidth() * Short.BYTES;
        long gatedNormed = gated + (long) rows * stateWidth() * Short.BYTES;
        return new Scratch(records, embedding, key, keyNormed, value, query, gated, gatedNormed);
    }

    /// Computes the PLE output of `rows` tokens (`tokens[offset ..]`) whose residual states are at `streamsAddress`
    /// into `output` (rows x streams x hidden), advancing the state past them. Returns the staging buffer of the
    /// n-gram rows, which the caller closes once the work has retired.
    public ExecutionGpu.UploadBuffer apply(
            ExecutionGpu gpu,
            Weights weights,
            State state,
            int[] tokens,
            int offset,
            int rows,
            long streamsAddress,
            Scratch scratch,
            long output) {
        int heads = this.store.heads();
        int count = rows * heads;
        if (this.rowIds.length < count) this.rowIds = new long[count];
        this.ids.compute(state.context(), tokens, offset, rows, this.rowIds);
        ExecutionGpu.UploadBuffer staged = this.store.stageRecords(gpu, this.rowIds, count, scratch.records());
        try {
            int width = stateWidth();
            Qwen4Ops.ngramExpand(
                    gpu,
                    scratch.records(),
                    scratch.embedding(),
                    count,
                    this.store.rowWidth(),
                    this.store.recordBytes());
            gpu.linearNvfp4Bf16(
                    scratch.embedding(),
                    weights.keyProj(),
                    scratch.key(),
                    rows,
                    this.embedDim,
                    width,
                    weights.keyProjBytes());
            Qwen4Ops.groupedRmsNorm(
                    gpu,
                    scratch.key(),
                    weights.normKey(),
                    scratch.keyNormed(),
                    rows,
                    this.streams,
                    this.hidden,
                    this.epsilon);
            gpu.linearNvfp4Bf16(
                    scratch.embedding(),
                    weights.valueProj(),
                    scratch.value(),
                    rows,
                    this.embedDim,
                    this.hidden,
                    weights.valueProjBytes());
            Qwen4Ops.groupedRmsNorm(
                    gpu,
                    streamsAddress,
                    weights.normQuery(),
                    scratch.query(),
                    rows,
                    this.streams,
                    this.hidden,
                    this.epsilon);
            Qwen4Ops.pleGate(
                    gpu,
                    scratch.keyNormed(),
                    scratch.query(),
                    scratch.value(),
                    scratch.gated(),
                    rows,
                    this.streams,
                    this.hidden);
            Qwen4Ops.groupedRmsNorm(
                    gpu,
                    scratch.gated(),
                    weights.normConv(),
                    scratch.gatedNormed(),
                    rows,
                    this.streams,
                    this.hidden,
                    this.epsilon);
            Qwen4Ops.pleConv(
                    gpu,
                    scratch.gatedNormed(),
                    scratch.gated(),
                    state.history(),
                    weights.convolution(),
                    output,
                    rows,
                    width,
                    this.taps,
                    this.dilation);
            Qwen4Ops.convHistory(gpu, scratch.gatedNormed(), state.history(), rows, width, historyRows());
        } catch (Throwable failure) {
            staged.close();
            throw failure;
        }
        return staged;
    }
}
