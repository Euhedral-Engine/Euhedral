package io.euhedral_execution.inference.quantization.mixed;

import io.euhedral_execution.inference.core.artifact.CompactTensorLayout;
import io.euhedral_execution.inference.core.artifact.TensorDataType;
import io.euhedral_execution.inference.core.artifact.TensorDescriptor;
import io.euhedral_execution.inference.core.artifact.WeightFormat;
import io.euhedral_execution.inference.core.artifact.WeightLayout;
import io.euhedral_execution.inference.core.model.qwen38.LayerType;
import io.euhedral_execution.inference.core.model.qwen38.Qwen38Config;
import io.euhedral_execution.inference.core.model.qwen38.artifact.Artifact;
import io.euhedral_execution.inference.core.model.qwen38.artifact.ArtifactHeader;
import io.euhedral_execution.inference.core.model.qwen38.artifact.CompactArtifactWriter;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/// Tiny artifacts shaped like calibrated ones: raw objects shared by every source, projections and heads that each
/// source fills with its own bytes, and, for a source with companions, `.suh`, `.svh` and `.segments` after each
/// projection. Projections are stored as 64 x 64 Q3 objects, a quantized format.
final class TinyArtifacts {

    static final List<String> PROJECTIONS = List.of(
            "text/layers/0/mlp/gate_up",
            "text/layers/0/mlp/down",
            "text/layers/1/attention/output",
            "text/layers/1/mlp/down");
    static final String DRAFTER_PROJECTION = "mtp/layer/mlp/down";

    private TinyArtifacts() {}

    /// Writes an artifact: shared raw objects from a fixed seed, everything else from `seed`. `companions` adds the
    /// trellis companions to the layers' and the drafter's projections.
    static Path write(Path path, long seed, boolean companions) throws IOException {
        return write(path, seed, companions, 99);
    }

    /// As [#write], the shared raw objects filled from `sharedSeed`.
    static Path write(Path path, long seed, boolean companions, long sharedSeed) throws IOException {
        return write(path, seed, companions, sharedSeed, "");
    }

    /// As [#write], without the object `omitted`.
    static Path writeWithout(Path path, long seed, String omitted) throws IOException {
        return write(path, seed, false, 99, omitted);
    }

    private static Path write(Path path, long seed, boolean companions, long sharedSeed, String omitted)
            throws IOException {
        List<TensorDescriptor> tensors = new ArrayList<>();
        List<byte[]> data = new ArrayList<>();
        raw(tensors, data, "text/layers/0/input_norm", TensorDataType.BF16, WeightFormat.BF16, 32, sharedSeed);
        raw(tensors, data, "text/layers/0/gdn/a_log", TensorDataType.FP32, WeightFormat.FP32, 8, sharedSeed + 1);
        quantized(tensors, data, "text/token_embedding", seed, false);
        for (String name : PROJECTIONS) if (!name.equals(omitted)) quantized(tensors, data, name, seed, companions);
        raw(tensors, data, "text/final_norm", TensorDataType.BF16, WeightFormat.BF16, 32, sharedSeed + 2);
        quantized(tensors, data, "text/output_head", seed, false);
        raw(tensors, data, "text/draft_head_token_ids", TensorDataType.INT32, WeightFormat.I32, 16, sharedSeed + 3);
        raw(tensors, data, "mtp/layer/input_norm", TensorDataType.BF16, WeightFormat.BF16, 32, seed);
        quantized(tensors, data, DRAFTER_PROJECTION, seed, companions);

        Qwen38Config config = config();
        long metadataSize = CompactArtifactWriter.metadataSize(config);
        long tableOffset = ArtifactHeader.BYTE_SIZE + metadataSize;
        TensorDescriptor[] table = tensors.toArray(TensorDescriptor[]::new);
        long provisional = 0;
        TensorDescriptor[] sized = new TensorDescriptor[table.length];
        for (int i = 0; i < table.length; i++) {
            TensorDescriptor t = table[i];
            sized[i] = new TensorDescriptor(
                    t.name(), t.shape(), t.dataType(), t.format(), t.layout(), provisional, t.byteSize());
            provisional += t.byteSize();
        }
        long dataOffset = tableOffset + CompactArtifactWriter.tensorTableSize(sized);
        long next = (dataOffset + 255) / 256 * 256;
        for (int i = 0; i < table.length; i++) {
            TensorDescriptor t = table[i];
            table[i] =
                    new TensorDescriptor(t.name(), t.shape(), t.dataType(), t.format(), t.layout(), next, t.byteSize());
            next = (next + t.byteSize() + 255) / 256 * 256;
        }
        var header = new ArtifactHeader(
                ArtifactHeader.MAGIC,
                ArtifactHeader.COMPACT_VERSION,
                ArtifactHeader.BYTE_SIZE,
                metadataSize,
                tableOffset,
                table.length,
                dataOffset);
        CompactArtifactWriter.write(path, new Artifact(header, config, table), data.toArray(byte[][]::new));
        return path;
    }

    private static void quantized(
            List<TensorDescriptor> tensors, List<byte[]> data, String name, long seed, boolean companions) {
        long size = CompactTensorLayout.expectedByteSize(
                new long[] {64, 64}, TensorDataType.BF16, WeightFormat.Q3_G64_FP16, WeightLayout.ROW_SPLIT_K128_V1);
        add(
                tensors,
                data,
                name,
                new long[] {64, 64},
                TensorDataType.BF16,
                WeightFormat.Q3_G64_FP16,
                WeightLayout.ROW_SPLIT_K128_V1,
                size,
                seed + name.hashCode());
        if (!companions || name.equals("text/token_embedding") || name.equals("text/output_head")) return;
        raw(tensors, data, name + ".suh", TensorDataType.FP16, WeightFormat.FP16, 64, seed + name.hashCode() + 1);
        raw(tensors, data, name + ".svh", TensorDataType.FP16, WeightFormat.FP16, 64, seed + name.hashCode() + 2);
        raw(tensors, data, name + ".segments", TensorDataType.INT32, WeightFormat.I32, 4, seed + name.hashCode() + 3);
    }

    private static void raw(
            List<TensorDescriptor> tensors,
            List<byte[]> data,
            String name,
            TensorDataType type,
            WeightFormat format,
            long elements,
            long seed) {
        long size = CompactTensorLayout.expectedByteSize(
                new long[] {elements}, type, format, WeightLayout.CONTIGUOUS_LE_V1);
        add(tensors, data, name, new long[] {elements}, type, format, WeightLayout.CONTIGUOUS_LE_V1, size, seed);
    }

    private static void add(
            List<TensorDescriptor> tensors,
            List<byte[]> data,
            String name,
            long[] shape,
            TensorDataType type,
            WeightFormat format,
            WeightLayout layout,
            long size,
            long seed) {
        byte[] bytes = new byte[(int) size];
        new Random(seed).nextBytes(bytes);
        tensors.add(new TensorDescriptor(name, shape, type, format, layout, 0, size));
        data.add(bytes);
    }

    static Qwen38Config config() {
        return new Qwen38Config(
                1,
                1,
                0,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1.0e-6,
                1.0,
                1.0,
                1,
                "silu",
                new LayerType[0],
                0,
                0,
                0,
                0,
                false,
                false,
                0);
    }
}
