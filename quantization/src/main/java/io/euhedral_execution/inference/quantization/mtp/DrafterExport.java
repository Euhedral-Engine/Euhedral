package io.euhedral_execution.inference.quantization.mtp;

import io.euhedral_execution.inference.core.artifact.TensorDataType;
import io.euhedral_execution.inference.core.artifact.TensorDescriptor;
import io.euhedral_execution.inference.core.artifact.WeightFormat;
import io.euhedral_execution.inference.core.artifact.WeightLayout;
import io.euhedral_execution.inference.core.model.qwen38.artifact.Artifact;
import io.euhedral_execution.inference.core.model.qwen38.artifact.ArtifactReader;
import io.euhedral_execution.inference.quantization.checkpoint.Qwen38Objects;
import io.euhedral_execution.inference.quantization.checkpoint.ResumeFile;
import io.euhedral_execution.inference.quantization.checkpoint.Safetensors;
import io.euhedral_execution.inference.quantization.mixed.Assemble;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/// Writes an artifact whose MTP layer is a trained drafter's: every object of the base artifact, except that the MTP
/// layer's projections are NVFP4 (round to nearest, [Nvfp4Rounding], as the drafter trained them) and its norms BF16,
/// from a `train-drafter` parameter file (FP32 values in [Drafter]'s parameter order after the file's header) or,
/// without one, from the checkpoint's MTP weights. The base's own MTP objects and their companions are left out.
public final class DrafterExport {

    public record Settings(Path parameters, Path model, Path artifact, Path out) {}

    /// The MTP objects in the order the converter writes them.
    private static final List<String> ORDER = List.of(
            Drafter.INPUT_PROJECTION,
            Drafter.EMBEDDING_NORM,
            Drafter.HIDDEN_NORM,
            Drafter.INPUT_NORM,
            Drafter.QKGV,
            Drafter.QUERY_NORM,
            Drafter.KEY_NORM,
            Drafter.OUTPUT,
            Drafter.POST_NORM,
            Drafter.GATE_UP,
            Drafter.DOWN,
            Drafter.FINAL_NORM);

    private static final Map<String, String> NORM_TENSORS = Map.of(
            Drafter.EMBEDDING_NORM, "mtp.pre_fc_norm_embedding.weight",
            Drafter.HIDDEN_NORM, "mtp.pre_fc_norm_hidden.weight",
            Drafter.INPUT_NORM, "mtp.layers.0.input_layernorm.weight",
            Drafter.POST_NORM, "mtp.layers.0.post_attention_layernorm.weight",
            Drafter.QUERY_NORM, "mtp.layers.0.self_attn.q_norm.weight",
            Drafter.KEY_NORM, "mtp.layers.0.self_attn.k_norm.weight",
            Drafter.FINAL_NORM, "mtp.norm.weight");

    private static final long ALIGNMENT = 4096;

    private DrafterExport() {}

    /// The checkpoint tensor of each MTP norm.
    static String normTensor(String norm) {
        return NORM_TENSORS.get(norm);
    }

    public static void run(Settings settings) throws IOException {
        Artifact base = ArtifactReader.read(settings.artifact());
        Path payloads = settings.out().resolveSibling(settings.out().getFileName() + ".mtp-payloads");
        Map<String, TensorDescriptor> written = new LinkedHashMap<>();
        try (Safetensors store = Safetensors.open(settings.model());
                FileChannel out = FileChannel.open(
                        payloads,
                        StandardOpenOption.CREATE,
                        StandardOpenOption.WRITE,
                        StandardOpenOption.TRUNCATE_EXISTING);
                Arena arena = Arena.ofConfined()) {
            Qwen38Objects checkpoint = new Qwen38Objects(store, settings.model());
            Map<String, long[]> shapes = shapes(checkpoint);
            Map<String, Long> fileOffsets = parameterOffsets(shapes);
            ResumeFile parameters = settings.parameters() == null ? null : new ResumeFile(settings.parameters());
            try {
                long offset = 0;
                for (String name : ORDER) {
                    long[] shape = shapes.get(name);
                    boolean projection = shape.length == 2;
                    long count = projection ? shape[0] * shape[1] : shape[0];
                    MemorySegment values = arena.allocate(4 * count, 64);
                    if (parameters != null) {
                        parameters.read(fileOffsets.get(name), values);
                    } else {
                        MemorySegment bf16 = arena.allocate(2 * count, 64);
                        if (projection) checkpoint.read(checkpoint.object(name), bf16);
                        else checkpoint.readTensor(NORM_TENSORS.get(name), bf16);
                        for (long i = 0; i < count; i++)
                            values.setAtIndex(
                                    ValueLayout.JAVA_INT,
                                    i,
                                    (bf16.getAtIndex(ValueLayout.JAVA_SHORT, i) & 0xffff) << 16);
                    }
                    MemorySegment payload;
                    if (projection) {
                        payload = arena.allocate(Nvfp4Rounding.bytes(shape[0], (int) shape[1]), 64);
                        Nvfp4Rounding.round(values, shape[0], (int) shape[1], payload);
                    } else {
                        payload = arena.allocate(2 * count, 64);
                        for (long i = 0; i < count; i++)
                            payload.setAtIndex(
                                    ValueLayout.JAVA_SHORT, i, bf16(values.getAtIndex(ValueLayout.JAVA_FLOAT, i)));
                    }
                    var buffer = payload.asByteBuffer();
                    for (long at = offset; buffer.hasRemaining(); ) at += out.write(buffer, at);
                    written.put(
                            name,
                            new TensorDescriptor(
                                    name,
                                    shape,
                                    TensorDataType.BF16,
                                    projection ? WeightFormat.NVFP4 : WeightFormat.BF16,
                                    projection ? WeightLayout.ROW_SPLIT_K128_V1 : WeightLayout.CONTIGUOUS_LE_V1,
                                    offset,
                                    payload.byteSize()));
                    offset = (offset + payload.byteSize() + ALIGNMENT - 1) / ALIGNMENT * ALIGNMENT;
                }
            } finally {
                if (parameters != null) parameters.close();
            }
            out.force(true);
        }
        List<TensorDescriptor> tensors = new ArrayList<>();
        Map<String, Path> sources = new HashMap<>();
        boolean placed = false;
        for (TensorDescriptor tensor : base.tensors()) {
            if (tensor.name().startsWith("mtp/")) {
                if (!placed) {
                    for (TensorDescriptor mtp : written.values()) {
                        tensors.add(mtp);
                        sources.put(mtp.name(), payloads);
                    }
                    placed = true;
                }
                continue;
            }
            tensors.add(tensor);
            sources.put(tensor.name(), settings.artifact());
        }
        if (!placed) throw new IllegalArgumentException(settings.artifact() + " has no MTP layer");
        try {
            Assemble.write(
                    new Artifact(base.header(), base.config(), tensors.toArray(TensorDescriptor[]::new), sources),
                    settings.out(),
                    Assemble.DEFAULT_COPY_BUFFER_MIB << 20);
        } finally {
            Files.deleteIfExists(payloads);
        }
        System.out.println("wrote " + settings.out() + " with the MTP layer of "
                + (settings.parameters() == null ? settings.model() : settings.parameters()));
    }

    /// Each MTP object's shape: [rows, columns] for a projection, [width] for a norm.
    static Map<String, long[]> shapes(Qwen38Objects checkpoint) {
        Map<String, long[]> shapes = new HashMap<>();
        for (String name : Drafter.PROJECTIONS) {
            var source = checkpoint.object(name);
            shapes.put(name, new long[] {source.rows(), source.columns()});
        }
        var config = checkpoint.config;
        for (String norm : Drafter.NORMS)
            shapes.put(norm, new long[] {
                norm.equals(Drafter.QUERY_NORM) || norm.equals(Drafter.KEY_NORM) ? config.headDim() : config.hidden()
            });
        return shapes;
    }

    /// Where each parameter's FP32 values start in a parameter file: after its header, in [Drafter]'s order.
    private static Map<String, Long> parameterOffsets(Map<String, long[]> shapes) {
        Map<String, Long> offsets = new HashMap<>();
        long offset = ResumeFile.HEADER;
        List<String> order = new ArrayList<>(Drafter.PROJECTIONS);
        order.addAll(Drafter.NORMS);
        for (String name : order) {
            offsets.put(name, offset);
            long[] shape = shapes.get(name);
            offset += 4 * (shape.length == 2 ? shape[0] * shape[1] : shape[0]);
        }
        return offsets;
    }

    private static short bf16(float value) {
        int bits = Float.floatToRawIntBits(value);
        if (Float.isNaN(value)) return (short) 0x7fc0;
        return (short) ((bits + 0x7fff + ((bits >> 16) & 1)) >>> 16);
    }
}
