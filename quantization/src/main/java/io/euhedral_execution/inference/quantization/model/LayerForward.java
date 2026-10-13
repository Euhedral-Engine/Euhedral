package io.euhedral_execution.inference.quantization.model;

import io.euhedral_execution.inference.core.model.qwen38.GdnState;
import io.euhedral_execution.inference.core.state.AttentionKvState;
import io.euhedral_execution.inference.quantization.checkpoint.Qwen38Objects;
import io.euhedral_execution.inference.quantization.gpu.Device;
import io.euhedral_execution.inference.quantization.gpu.Upload;

/// One Qwen3.8 decoder layer (or the MTP block) over a batch of whole sequences on the engine's kernels, with BF16
/// projection weights. Each quantized linear's input rows go to the observer first (for its Hessian). Sequences start
/// at position 0 with fresh GDN state; attention keeps its keys and values in the engine's NVFP4 cache, as the deployed
/// engine does. The hidden states [sequences * length, hidden] are updated in place.
public final class LayerForward implements AutoCloseable {

    /// Receives the input rows (BF16 [rows, columns], device) of a quantized object before it is applied.
    @FunctionalInterface
    public interface Observer {
        void observe(String object, long rows, int count, int columns);
    }

    private final Device device;
    private final Qwen38Objects.Config c;
    private final int maxRows;
    private final long normalized, first, second, mixed, context, gateUp, swiglu, delta, alpha, beta, position;
    private final java.util.List<GdnState> gdnStates = new java.util.ArrayList<>();
    private final java.util.List<AttentionKvState> kvStates = new java.util.ArrayList<>();
    private final int qkWidth, vzWidth, gdnValueWidth, convolutionWidth, attentionQ, attentionKv;

    public LayerForward(Device device, Qwen38Objects.Config config, int maxRows) {
        this.device = device;
        this.c = config;
        this.maxRows = maxRows;
        int hidden = c.hidden();
        attentionQ = c.heads() * c.headDim();
        attentionKv = c.kvHeads() * c.headDim();
        qkWidth = 2 * c.keyHeads() * c.keyDim();
        gdnValueWidth = c.valueHeads() * c.valueDim();
        vzWidth = 2 * gdnValueWidth;
        convolutionWidth = qkWidth + gdnValueWidth;
        int widest = Math.max(Math.max(vzWidth, attentionQ + attentionKv), convolutionWidth);
        normalized = alloc(2L * maxRows * hidden);
        first = alloc(2L * maxRows * widest);
        second = alloc(2L * maxRows * widest);
        mixed = alloc(2L * maxRows * Math.max(convolutionWidth, attentionQ + attentionKv));
        context = alloc(2L * maxRows * Math.max(gdnValueWidth, attentionQ));
        gateUp = alloc(2L * maxRows * 2 * c.intermediate());
        swiglu = alloc(2L * maxRows * c.intermediate());
        delta = alloc(2L * maxRows * hidden);
        alpha = alloc(4L * maxRows * c.valueHeads());
        beta = alloc(4L * maxRows * c.valueHeads());
        position = alloc(8);
        Upload.longs(device.gpu, position, new long[] {0});
    }

    public int hidden() {
        return c.hidden();
    }

    private long alloc(long bytes) {
        return device.gpu.allocate(bytes);
    }

    /// Runs `layer` over `sequences` sequences of `length` tokens whose hidden states are `hidden` (BF16, device),
    /// every step on the calling thread's stream: [#head], then [#sequence] for each sequence, then [#tail].
    public void forward(LayerWeights layer, long hidden, int sequences, int length, Observer observer) {
        prepareSlots(sequences);
        head(layer, hidden, sequences * length, observer);
        for (int s = 0; s < sequences; s++) sequence(layer, s, length);
        tail(layer, hidden, sequences * length, observer);
    }

    /// Makes the GDN states and attention caches of `sequences` slots (before frames use them concurrently).
    public void prepareSlots(int sequences) {
        while (gdnStates.size() < sequences) {
            gdnStates.add(GdnState.allocate(
                    device.gpu, c.keyHeads(), c.valueHeads(), c.keyDim(), c.valueDim(), c.convKernel()));
        }
        while (kvStates.size() < sequences) kvStates.add(new AttentionKvState(device.gpu, attentionKv));
    }

    /// The rows of all sequences at once: the input norm and the mixer's input projections (with GDN's a/b control).
    public void head(LayerWeights layer, long hidden, int rows, Observer observer) {
        if (rows > maxRows) throw new IllegalArgumentException(rows + " rows exceed the workspace's " + maxRows);
        int h = c.hidden();
        device.gpu.rmsNormUnitOffsetBf16(hidden, layer.inputNorm, normalized, rows, h, (float) c.eps());
        if (layer.attention) {
            int width = attentionQ + attentionKv;
            String fused = layer.prefix + "attention/query_key_gate_value";
            if (layer.objects.containsKey(fused)) {
                // The MTP block's one object holds the query_key rows, then the gate_value rows.
                if (observer != null) observer.observe(fused, normalized, rows, h);
                long weights = layer.objects.get(fused);
                device.linear(normalized, weights, first, rows, h, width);
                device.linear(normalized, weights + 2L * width * h, second, rows, h, width);
            } else {
                apply(layer, "attention/query_key", normalized, first, rows, h, width, observer);
                apply(layer, "attention/gate_value", normalized, second, rows, h, width, observer);
            }
            return;
        }
        // query_key and value_z read the same rows: the observer sees them under both names.
        apply(layer, "gdn/query_key", normalized, first, rows, h, qkWidth, observer);
        apply(layer, "gdn/value_z", normalized, second, rows, h, vzWidth, observer);
        device.gpu.gdnProjectControlFp32(
                normalized,
                layer.aProjection,
                layer.bProjection,
                layer.aLog,
                layer.dtBias,
                alpha,
                beta,
                rows,
                h,
                c.valueHeads());
    }

    /// Sequence `s` of `length` tokens: the GDN convolution and recurrence from fresh state, or the Q/K norm, RoPE and
    /// causal attention over a fresh cache. Sequences are independent, so each can run on its own stream.
    public void sequence(LayerWeights layer, int s, int length) {
        var gpu = device.gpu;
        long offset = (long) s * length;
        if (!layer.attention) {
            GdnState state = gdnStates.get(s);
            gpu.zeroDeviceMemory(state.convolutionStateAddress(), 2L * convolutionWidth * (c.convKernel() - 1));
            gpu.zeroDeviceMemory(state.recurrentStateAddress(), 4L * c.valueHeads() * c.keyDim() * c.valueDim());
            gpu.gdnConvolutionBf16(
                    first + 2 * offset * qkWidth,
                    second + 2 * offset * vzWidth,
                    layer.convolution,
                    state.convolutionStateAddress(),
                    mixed + 2 * offset * convolutionWidth,
                    length,
                    qkWidth,
                    gdnValueWidth,
                    convolutionWidth,
                    c.convKernel());
            gpu.gdnRecurrenceBf16(
                    mixed + 2 * offset * convolutionWidth,
                    alpha + 4 * offset * c.valueHeads(),
                    beta + 4 * offset * c.valueHeads(),
                    state.recurrentStateAddress(),
                    context + 2 * offset * gdnValueWidth,
                    length,
                    c.keyHeads(),
                    c.valueHeads(),
                    c.keyDim(),
                    c.valueDim(),
                    (float) (1.0 / Math.sqrt(c.keyDim())));
            return;
        }
        int width = attentionQ + attentionKv;
        long qk = first + 2 * offset * width, gv = second + 2 * offset * width, normed = mixed + 2 * offset * width;
        gpu.attentionQkNormRopeBf16(
                qk,
                layer.queryNorm,
                layer.keyNorm,
                normed,
                length,
                c.heads(),
                c.kvHeads(),
                c.headDim(),
                c.rotaryDim(),
                0,
                position,
                (float) c.eps(),
                c.theta());
        AttentionKvState state = kvStates.get(s);
        state.truncate(0);
        state.prepareAppend(0, length);
        gpu.attentionKvAppendNvfp4(
                normed,
                gv,
                state.keyCacheAddress(),
                state.valueCacheAddress(),
                length,
                attentionQ,
                attentionKv,
                0,
                position);
        state.appendSubmitted(length);
        gpu.attentionCausalNvfp4(
                normed,
                gv,
                state.keyCacheAddress(),
                state.valueCacheAddress(),
                context + 2 * offset * attentionQ,
                length,
                c.heads(),
                c.kvHeads(),
                c.headDim(),
                length,
                0,
                position,
                0);
        state.commitAppended(length);
    }

    /// The rows of all sequences at once: the mixer's output (GDN's gated norm first), the residual and the MLP.
    public void tail(LayerWeights layer, long hidden, int rows, Observer observer) {
        var gpu = device.gpu;
        int h = c.hidden();
        float eps = (float) c.eps();
        if (layer.attention) {
            apply(layer, "attention/output", context, delta, rows, attentionQ, h, observer);
        } else {
            gpu.gdnGatedRmsNormBf16(context, second, layer.gdnNorm, first, rows, c.valueHeads(), c.valueDim(), eps);
            apply(layer, "gdn/output", first, delta, rows, gdnValueWidth, h, observer);
        }
        gpu.residualAddBf16(hidden, delta, hidden, rows, h);
        gpu.rmsNormUnitOffsetBf16(hidden, layer.postNorm, normalized, rows, h, eps);
        apply(layer, "mlp/gate_up", normalized, gateUp, rows, h, 2 * c.intermediate(), observer);
        gpu.swiGluBf16(gateUp, swiglu, rows, c.intermediate());
        apply(layer, "mlp/down", swiglu, delta, rows, c.intermediate(), h, observer);
        gpu.residualAddBf16(hidden, delta, hidden, rows, h);
    }

    private void apply(
            LayerWeights layer, String role, long input, long output, int rows, int k, int n, Observer observer) {
        String name = layer.prefix + role;
        if (observer != null) observer.observe(name, input, rows, k);
        device.linear(input, layer.objects.get(name), output, rows, k, n);
    }

    @Override
    public void close() {
        gdnStates.forEach(GdnState::close);
        kvStates.forEach(AttentionKvState::close);
        for (long address :
                new long[] {normalized, first, second, mixed, context, gateUp, swiglu, delta, alpha, beta, position}) {
            device.gpu.free(address);
        }
    }
}
