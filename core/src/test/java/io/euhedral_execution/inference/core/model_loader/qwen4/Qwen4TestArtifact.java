package io.euhedral_execution.inference.core.model_loader.qwen4;

import io.euhedral_execution.inference.core.model_loader.artifact.CompactTensorLayout;
import io.euhedral_execution.inference.core.model_loader.artifact.Nvfp4Layout;
import io.euhedral_execution.inference.core.model_loader.layer_weights.TensorDataType;
import io.euhedral_execution.inference.core.model_loader.layer_weights.WeightFormat;
import io.euhedral_execution.inference.core.model_loader.layer_weights.WeightLayout;
import io.euhedral_execution.inference.core.model_loader.qwen4.expert.ExpertBank;
import io.euhedral_execution.inference.core.model_loader.qwen4.expert.ExpertProjection;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/// Builds small `qwen4_exp` artifacts with the real object inventory and deterministic random payloads.
public final class Qwen4TestArtifact {

    private Qwen4TestArtifact() {}

    /// A miniature of the real topology: every kind of object exists, with sizes in the kilobytes.
    public static Qwen4Config miniConfig() {
        int hidden = 256;
        return new Qwen4Config(
                new Qwen4Config.Text(64, hidden, 4, 4096, 1e-6, "silu", false, 1, 2, new Qwen4LayerType[] {
                    Qwen4LayerType.GATED_DELTA_NET,
                    Qwen4LayerType.GATED_DELTA_NET,
                    Qwen4LayerType.SPARSE_ATTENTION,
                    Qwen4LayerType.GATED_DELTA_NET
                }),
                new Qwen4Config.Attention(4, 2, 64, 0.25, 1e7, new int[] {3, 3, 2}, true, "sigmoid", 4),
                new Qwen4Config.Gdn(2, 4, 64, 64, 4, "float32"),
                new Qwen4Config.Qsa(16, 4, 64, 1, 2),
                new Qwen4Config.Moe(8, 2, 128, 128),
                new Qwen4Config.HyperConnection(2, 32),
                new Qwen4Config.Ngram(
                        3,
                        40,
                        1,
                        4,
                        4,
                        21,
                        new long[] {10_000_000_000_001L, 10_000_000_000_002L, 10_000_000_000_003L},
                        new long[] {0, 41},
                        new long[] {41, 43}),
                new Qwen4Config.Ple(new int[] {1}, hidden, 4),
                new Qwen4Config.Mtp(1, true, new Qwen4LayerType[] {Qwen4LayerType.SPARSE_ATTENTION}, 1e7, false, -1),
                new Qwen4Config.Vision(
                        1, 32, 64, 2, 3, 4, 2, 2, hidden, 16, "gelu_pytorch_tanh", new int[0], 9, 10, 11, 12));
    }

    public static Qwen4Artifact write(Path path, Qwen4Config config, long seed) throws IOException {
        Qwen4Inventory.Inventory inventory = Qwen4Inventory.expected(config);
        List<Qwen4Tensor> tensors = new ArrayList<>();
        List<ExpertBank> banks = new ArrayList<>();
        long cursor = 1 << 20; // generous preamble: the data section starts after the tables, on 4096
        for (Map.Entry<String, Qwen4Inventory.Expected> entry :
                inventory.tensors().entrySet()) {
            Qwen4Inventory.Expected want = entry.getValue();
            long size = size(want);
            long alignment = want.layout() == WeightLayout.ROW_INTERLEAVED_NVFP4_V1 ? 4096 : 256;
            cursor = alignUp(cursor, alignment);
            tensors.add(new Qwen4Tensor(
                    entry.getKey(),
                    want.shape(),
                    TensorDataType.BF16,
                    want.format(),
                    want.layout(),
                    want.group(),
                    cursor,
                    size,
                    0));
            cursor += size;
        }
        for (Qwen4Inventory.ExpectedBank want : inventory.banks()) {
            List<ExpertProjection> projections = new ArrayList<>();
            long record = 0;
            for (Map.Entry<String, long[]> projection : want.projections().entrySet()) {
                long size = Nvfp4Layout.byteSize(
                        projection.getValue()[0], projection.getValue()[1]);
                projections.add(new ExpertProjection(
                        projection.getKey(),
                        projection.getValue(),
                        TensorDataType.BF16,
                        WeightFormat.NVFP4,
                        WeightLayout.ROW_SPLIT_K128_V1,
                        record,
                        size));
                record = alignUp(record + size, 256);
            }
            long stride = alignUp(record, 4096);
            cursor = alignUp(cursor, 4096);
            long[] offsets = new long[want.experts()];
            long[] sizes = new long[want.experts()];
            for (int e = 0; e < offsets.length; e++) {
                offsets[e] = cursor + e * stride;
                sizes[e] = stride;
            }
            cursor += want.experts() * stride;
            banks.add(new ExpertBank(
                    want.name(), want.group(), want.layer(), projections, offsets, sizes, new int[offsets.length]));
        }
        long fileSize = alignUp(cursor, 4096);
        return Qwen4ArtifactWriter.write(
                path,
                config,
                tensors.toArray(Qwen4Tensor[]::new),
                banks.toArray(ExpertBank[]::new),
                4096L * 256,
                Math.max(fileSize, 4096L * 256 + 4096),
                tensor -> payload(
                        seed, tensor.name(), tensor.shape(), tensor.format(), tensor.layout(), (int) tensor.byteSize()),
                (bank, expert) -> record(seed, bank, expert));
    }

    private static long size(Qwen4Inventory.Expected want) {
        if (want.layout() == WeightLayout.ROW_INTERLEAVED_NVFP4_V1)
            return NgramLayout.byteSize(want.shape()[0], want.shape()[1]);
        return CompactTensorLayout.expectedByteSize(want.shape(), TensorDataType.BF16, want.format(), want.layout());
    }

    static byte[] payload(long seed, String name, long[] shape, WeightFormat format, WeightLayout layout, int size) {
        byte[] bytes = new byte[size];
        new SplittableRandom(seed ^ name.hashCode()).nextBytes(bytes);
        if (layout == WeightLayout.ROW_SPLIT_K128_V1 && format == WeightFormat.NVFP4)
            fixRowSplit(bytes, 0, shape[0], shape[1]);
        if (layout == WeightLayout.ROW_INTERLEAVED_NVFP4_V1) fixInterleaved(bytes, shape[0], shape[1]);
        return bytes;
    }

    static byte[] record(long seed, ExpertBank bank, int expert) {
        byte[] bytes = new byte[(int) bank.recordBytes(expert)];
        new SplittableRandom(seed ^ (bank.name().hashCode() * 31L + expert)).nextBytes(bytes);
        for (ExpertProjection projection : bank.projections()) {
            // Padding between projections stays as random bytes: only the structured fields are fixed.
            fixRowSplit(
                    bytes,
                    (int) projection.recordOffset(),
                    projection.shape()[0],
                    projection.shape()[1]);
        }
        return bytes;
    }

    /// Valid block scales (never the NaN code) and a finite positive global scale.
    private static void fixRowSplit(byte[] bytes, int base, long rows, long k) {
        int scales = base + (int) Nvfp4Layout.scaleOffset(rows, k);
        int end = scales + (int) (rows * Nvfp4Layout.rowScaleBytes(k, WeightLayout.ROW_SPLIT_K128_V1));
        for (int i = scales; i < end; i++) bytes[i] &= 0x7e;
        putFloat(bytes, base + (int) Nvfp4Layout.globalScaleOffset(rows, k), 0.0625f + (bytes[base] & 0x3f) / 1024f);
    }

    private static void fixInterleaved(byte[] bytes, long rows, long k) {
        long rowBytes = NgramLayout.rowBytes(k);
        for (long row = 0; row < rows; row++) {
            int scales = (int) (row * rowBytes + k / 2);
            for (int i = 0; i < k / 16; i++) bytes[scales + i] &= 0x7e;
        }
        putFloat(bytes, (int) NgramLayout.trailerOffset(rows, k), 0.125f);
    }

    private static void putFloat(byte[] bytes, int offset, float value) {
        ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).putFloat(offset, value);
    }

    private static long alignUp(long value, long alignment) {
        return (value + alignment - 1) / alignment * alignment;
    }
}
