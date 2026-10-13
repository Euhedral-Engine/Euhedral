package io.euhedral_execution.inference.quantization.model;

import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import io.euhedral_execution.inference.quantization.checkpoint.Qwen38Objects;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.LinkedHashMap;
import java.util.Map;

/// One decoder layer's (or the MTP block's) weights on the device in BF16, read from the checkpoint: the quantized
/// objects fused as the converter fuses them, which calibration replaces with their decoded rounding, and the rest
/// (norms, GDN convolution, a/b projections, FP32 A_log and dt_bias) as the engine's kernels take them.
public final class LayerWeights implements AutoCloseable {

    public final String prefix;
    public final boolean attention;
    /// Quantized objects by name (`text/layers/N/mlp/down`, ...): device address of BF16 [rows, columns].
    public final Map<String, Long> objects = new LinkedHashMap<>();
    public final Map<String, long[]> shapes = new LinkedHashMap<>();
    public long inputNorm, postNorm, queryNorm, keyNorm;
    public long aProjection, bProjection, aLog, dtBias, convolution, gdnNorm;
    private final CudaGpuMemory gpu;
    private final Map<String, Long> owned = new LinkedHashMap<>();

    private LayerWeights(CudaGpuMemory gpu, String prefix, boolean attention) {
        this.gpu = gpu;
        this.prefix = prefix;
        this.attention = attention;
    }

    /// Decoder layer `index`.
    public static LayerWeights layer(CudaGpuMemory gpu, Qwen38Objects checkpoint, int index) throws IOException {
        boolean attention = checkpoint.config.fullAttention()[index];
        String p = "model.language_model.layers." + index + ".";
        var weights = new LayerWeights(gpu, "text/layers/" + index + "/", attention);
        for (String name : checkpoint.layerObjects(index)) weights.object(checkpoint, name);
        weights.common(checkpoint, p, attention ? null : "linear_attn.");
        return weights;
    }

    /// The MTP block (a full-attention layer with one fused q/k/gate/v object).
    public static LayerWeights mtp(CudaGpuMemory gpu, Qwen38Objects checkpoint) throws IOException {
        var weights = new LayerWeights(gpu, "mtp/layer/", true);
        for (String name : Qwen38Objects.MTP_OBJECTS) {
            if (name.startsWith("mtp/layer/")) weights.object(checkpoint, name);
        }
        weights.common(checkpoint, "mtp.layers.0.", null);
        return weights;
    }

    private void common(Qwen38Objects checkpoint, String p, String gdn) throws IOException {
        inputNorm = tensor(checkpoint, p + "input_layernorm.weight");
        postNorm = tensor(checkpoint, p + "post_attention_layernorm.weight");
        if (attention) {
            queryNorm = tensor(checkpoint, p + "self_attn.q_norm.weight");
            keyNorm = tensor(checkpoint, p + "self_attn.k_norm.weight");
            return;
        }
        aProjection = tensor(checkpoint, p + gdn + "in_proj_a.weight");
        bProjection = tensor(checkpoint, p + gdn + "in_proj_b.weight");
        gdnNorm = tensor(checkpoint, p + gdn + "norm.weight");
        aLog = floats(checkpoint, p + gdn + "A_log");
        dtBias = floats(checkpoint, p + gdn + "dt_bias");
        convolution = transposedConvolution(checkpoint, p + gdn + "conv1d.weight");
    }

    private void object(Qwen38Objects checkpoint, String name) throws IOException {
        var source = checkpoint.object(name);
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment host = arena.allocate(source.bytes(), 16);
            checkpoint.read(source, host);
            long address = upload(name, host);
            objects.put(name, address);
            shapes.put(name, new long[] {source.rows(), source.columns()});
        }
    }

    private long tensor(Qwen38Objects checkpoint, String tensor) throws IOException {
        long bytes = checkpoint.store().tensor(tensor).bytes();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment host = arena.allocate(bytes, 16);
            checkpoint.readTensor(tensor, host);
            return upload(tensor, host);
        }
    }

    /// A BF16 vector as FP32 (the engine's GDN control takes A_log and dt_bias in FP32).
    private long floats(Qwen38Objects checkpoint, String tensor) throws IOException {
        long elements = checkpoint.store().tensor(tensor).elements();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment bf16 = arena.allocate(2 * elements, 16);
            checkpoint.readTensor(tensor, bf16);
            MemorySegment fp32 = arena.allocate(4 * elements, 16);
            for (long i = 0; i < elements; i++) {
                fp32.setAtIndex(
                        ValueLayout.JAVA_FLOAT,
                        i,
                        Float.intBitsToFloat(bf16.getAtIndex(ValueLayout.JAVA_SHORT, i) << 16));
            }
            return upload(tensor, fp32);
        }
    }

    /// conv1d.weight [channels, 1, taps] as [taps, channels] (the converter's gdn/convolution layout).
    private long transposedConvolution(Qwen38Objects checkpoint, String tensor) throws IOException {
        long[] shape = checkpoint.store().tensor(tensor).shape();
        int channels = (int) shape[0], taps = (int) shape[2];
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment source = arena.allocate(2L * channels * taps, 16);
            checkpoint.readTensor(tensor, source);
            MemorySegment transposed = arena.allocate(2L * channels * taps, 16);
            for (int c = 0; c < channels; c++) {
                for (int t = 0; t < taps; t++) {
                    transposed.setAtIndex(
                            ValueLayout.JAVA_SHORT,
                            (long) t * channels + c,
                            source.getAtIndex(ValueLayout.JAVA_SHORT, (long) c * taps + t));
                }
            }
            return upload(tensor, transposed);
        }
    }

    private long upload(String name, MemorySegment host) {
        long address = gpu.allocate(host.byteSize());
        gpu.copyHostToDevice(address, host, host.byteSize());
        owned.put(name, address);
        return address;
    }

    /// Replaces object `name` with `decoded` (BF16, same shape, device), which this layer then owns.
    public void replace(String name, long decoded) {
        gpu.free(objects.get(name));
        objects.put(name, decoded);
        owned.put(name, decoded);
    }

    @Override
    public void close() {
        for (long address : owned.values()) gpu.free(address);
        owned.clear();
    }
}
