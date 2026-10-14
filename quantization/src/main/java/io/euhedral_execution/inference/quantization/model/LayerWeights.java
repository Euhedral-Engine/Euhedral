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
///
/// Loading has two steps so that a worker can read the next layer while the GPU runs the current one: [#stage]
/// reads and converts everything into host memory, [#upload] copies it to the device.
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

    /// A layer read into host memory, not yet on the device.
    public static final class Staged implements AutoCloseable {
        final String prefix;
        final boolean attention;
        final Map<String, MemorySegment> parts = new LinkedHashMap<>();
        final Map<String, long[]> shapes = new LinkedHashMap<>();
        private final Arena arena = Arena.ofShared();

        Staged(String prefix, boolean attention) {
            this.prefix = prefix;
            this.attention = attention;
        }

        /// The layer's tensors in host memory by role (norms, GDN tensors) or object name, as they are uploaded.
        public Map<String, MemorySegment> parts() {
            return java.util.Collections.unmodifiableMap(parts);
        }

        @Override
        public void close() {
            arena.close();
        }
    }

    private LayerWeights(CudaGpuMemory gpu, String prefix, boolean attention) {
        this.gpu = gpu;
        this.prefix = prefix;
        this.attention = attention;
    }

    /// Decoder layer `index` (read and uploaded).
    public static LayerWeights layer(CudaGpuMemory gpu, Qwen38Objects checkpoint, int index) throws IOException {
        try (Staged staged = stage(checkpoint, index)) {
            return upload(gpu, staged);
        }
    }

    /// The MTP block (a full-attention layer with one fused q/k/gate/v object).
    public static LayerWeights mtp(CudaGpuMemory gpu, Qwen38Objects checkpoint) throws IOException {
        try (Staged staged = new Staged("mtp/layer/", true)) {
            for (String name : Qwen38Objects.MTP_OBJECTS) {
                if (name.startsWith("mtp/layer/")) object(staged, checkpoint, name);
            }
            common(staged, checkpoint, "mtp.layers.0.", null);
            return upload(gpu, staged);
        }
    }

    /// Reads decoder layer `index` into host memory.
    public static Staged stage(Qwen38Objects checkpoint, int index) throws IOException {
        boolean attention = checkpoint.config.fullAttention()[index];
        var staged = new Staged("text/layers/" + index + "/", attention);
        for (String name : checkpoint.layerObjects(index)) object(staged, checkpoint, name);
        common(staged, checkpoint, "model.language_model.layers." + index + ".", attention ? null : "linear_attn.");
        return staged;
    }

    /// Copies a staged layer to the device.
    public static LayerWeights upload(CudaGpuMemory gpu, Staged staged) {
        var weights = new LayerWeights(gpu, staged.prefix, staged.attention);
        for (var part : staged.parts.entrySet()) {
            long address = gpu.allocate(part.getValue().byteSize());
            gpu.copyHostToDevice(address, part.getValue(), part.getValue().byteSize());
            weights.owned.put(part.getKey(), address);
            switch (part.getKey()) {
                case "inputNorm" -> weights.inputNorm = address;
                case "postNorm" -> weights.postNorm = address;
                case "queryNorm" -> weights.queryNorm = address;
                case "keyNorm" -> weights.keyNorm = address;
                case "aProjection" -> weights.aProjection = address;
                case "bProjection" -> weights.bProjection = address;
                case "gdnNorm" -> weights.gdnNorm = address;
                case "aLog" -> weights.aLog = address;
                case "dtBias" -> weights.dtBias = address;
                case "convolution" -> weights.convolution = address;
                default -> {
                    weights.objects.put(part.getKey(), address);
                    weights.shapes.put(part.getKey(), staged.shapes.get(part.getKey()));
                }
            }
        }
        return weights;
    }

    private static void common(Staged staged, Qwen38Objects checkpoint, String p, String gdn) throws IOException {
        tensor(staged, checkpoint, "inputNorm", p + "input_layernorm.weight");
        tensor(staged, checkpoint, "postNorm", p + "post_attention_layernorm.weight");
        if (staged.attention) {
            tensor(staged, checkpoint, "queryNorm", p + "self_attn.q_norm.weight");
            tensor(staged, checkpoint, "keyNorm", p + "self_attn.k_norm.weight");
            return;
        }
        tensor(staged, checkpoint, "aProjection", p + gdn + "in_proj_a.weight");
        tensor(staged, checkpoint, "bProjection", p + gdn + "in_proj_b.weight");
        tensor(staged, checkpoint, "gdnNorm", p + gdn + "norm.weight");
        floats(staged, checkpoint, "aLog", p + gdn + "A_log");
        floats(staged, checkpoint, "dtBias", p + gdn + "dt_bias");
        transposedConvolution(staged, checkpoint, "convolution", p + gdn + "conv1d.weight");
    }

    private static void object(Staged staged, Qwen38Objects checkpoint, String name) throws IOException {
        var source = checkpoint.object(name);
        MemorySegment host = staged.arena.allocate(source.bytes(), 16);
        checkpoint.read(source, host);
        staged.parts.put(name, host);
        staged.shapes.put(name, new long[] {source.rows(), source.columns()});
    }

    private static void tensor(Staged staged, Qwen38Objects checkpoint, String role, String tensor) throws IOException {
        MemorySegment host =
                staged.arena.allocate(checkpoint.store().tensor(tensor).bytes(), 16);
        checkpoint.readTensor(tensor, host);
        staged.parts.put(role, host);
    }

    /// A BF16 vector as FP32 (the engine's GDN control takes A_log and dt_bias in FP32).
    private static void floats(Staged staged, Qwen38Objects checkpoint, String role, String tensor) throws IOException {
        long elements = checkpoint.store().tensor(tensor).elements();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment bf16 = arena.allocate(2 * elements, 16);
            checkpoint.readTensor(tensor, bf16);
            MemorySegment fp32 = staged.arena.allocate(4 * elements, 16);
            for (long i = 0; i < elements; i++) {
                fp32.setAtIndex(
                        ValueLayout.JAVA_FLOAT,
                        i,
                        Float.intBitsToFloat(bf16.getAtIndex(ValueLayout.JAVA_SHORT, i) << 16));
            }
            staged.parts.put(role, fp32);
        }
    }

    /// conv1d.weight [channels, 1, taps] as [taps, channels] (the converter's gdn/convolution layout).
    private static void transposedConvolution(Staged staged, Qwen38Objects checkpoint, String role, String tensor)
            throws IOException {
        long[] shape = checkpoint.store().tensor(tensor).shape();
        int channels = (int) shape[0], taps = (int) shape[2];
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment source = arena.allocate(2L * channels * taps, 16);
            checkpoint.readTensor(tensor, source);
            MemorySegment transposed = staged.arena.allocate(2L * channels * taps, 16);
            for (int c = 0; c < channels; c++) {
                for (int t = 0; t < taps; t++) {
                    transposed.setAtIndex(
                            ValueLayout.JAVA_SHORT,
                            (long) t * channels + c,
                            source.getAtIndex(ValueLayout.JAVA_SHORT, (long) c * taps + t));
                }
            }
            staged.parts.put(role, transposed);
        }
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
