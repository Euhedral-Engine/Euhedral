package io.euhedral_execution.inference.core.model.qwen4.loader;

import io.euhedral_execution.inference.core.artifact.CompactTensorLayout;
import io.euhedral_execution.inference.core.artifact.Nvfp4Layout;
import io.euhedral_execution.inference.core.artifact.TensorDataType;
import io.euhedral_execution.inference.core.artifact.WeightFormat;
import io.euhedral_execution.inference.core.artifact.WeightLayout;
import io.euhedral_execution.inference.core.model.qwen4.Qwen4Config;
import io.euhedral_execution.inference.core.model.qwen4.expert.ExpertBank;
import io.euhedral_execution.inference.core.model.qwen4.expert.ExpertProjection;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/// Builds small `qwen4_exp` artifacts with the real object inventory and deterministic random payloads.
public final class TestArtifact {

    private TestArtifact() {}

    /// A miniature of the real topology: every kind of object exists, with sizes in the kilobytes.
    public static Qwen4Config miniConfig() {
        int hidden = 256;
        return new Qwen4Config(
                new Qwen4Config.Text(64, hidden, 4, 4096, 1e-6, "silu", false, 1, 2, new LayerType[] {
                    LayerType.GATED_DELTA_NET,
                    LayerType.GATED_DELTA_NET,
                    LayerType.SPARSE_ATTENTION,
                    LayerType.GATED_DELTA_NET
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
                new Qwen4Config.Mtp(1, true, new LayerType[] {LayerType.SPARSE_ATTENTION}, 1e7, false, -1),
                new Qwen4Config.Vision(
                        1, 32, 64, 2, 3, 4, 2, 2, hidden, 16, "gelu_pytorch_tanh", new int[0], 9, 10, 11, 12));
    }

    /// The real model's topology (config.json of Qwen3.8-Flash-Next), for planning against real object sizes.
    public static Qwen4Config realConfig() {
        LayerType[] types = new LayerType[48];
        for (int i = 0; i < types.length; i++)
            types[i] = i % 4 == 3 ? LayerType.SPARSE_ATTENTION : LayerType.GATED_DELTA_NET;
        return new Qwen4Config(
                new Qwen4Config.Text(248320, 2560, 48, 262144, 1e-6, "silu", false, 248044, 248044, types),
                new Qwen4Config.Attention(24, 2, 256, 0.25, 1e7, new int[] {11, 11, 10}, true, "sigmoid", 4),
                new Qwen4Config.Gdn(16, 48, 128, 128, 4, "float32"),
                new Qwen4Config.Qsa(2048, 4, 128, 1, 4),
                new Qwen4Config.Moe(512, 10, 640, 640),
                new Qwen4Config.HyperConnection(4, 320),
                new Qwen4Config.Ngram(
                        3,
                        20_000_000,
                        8,
                        128,
                        128,
                        2_500_012,
                        new long[] {23703573157769L, 20109073645365L, 8052911324071L},
                        new long[] {
                            0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297, 160000374,
                            180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275
                        },
                        new long[] {
                            20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077, 20000081,
                            20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171
                        }),
                new Qwen4Config.Ple(new int[] {1}, 2560, 4),
                new Qwen4Config.Mtp(1, true, new LayerType[] {LayerType.SPARSE_ATTENTION}, 1e7, false, -1),
                new Qwen4Config.Vision(
                        27,
                        1152,
                        4304,
                        16,
                        3,
                        16,
                        2,
                        2,
                        2560,
                        2304,
                        "gelu_pytorch_tanh",
                        new int[0],
                        248056,
                        248057,
                        248053,
                        248054));
    }

    /// An artifact's tables for `config` with the real converter's object geometry, and no file: for planning and
    /// cache tests at the real model's sizes.
    public static Artifact virtual(Qwen4Config config) {
        Layout layout = layout(config);
        return new Artifact(
                new Header(
                        Header.MAGIC,
                        Header.VERSION,
                        Header.ARCHITECTURE_QWEN4_EXP,
                        64,
                        0,
                        64,
                        0,
                        4096L * 256,
                        layout.fileSize),
                config,
                layout.tensors,
                layout.banks);
    }

    private record Layout(Tensor[] tensors, ExpertBank[] banks, long fileSize) {}

    public static Artifact write(Path path, Qwen4Config config, long seed) throws IOException {
        Layout layout = layout(config);
        return ArtifactWriter.write(
                path,
                config,
                layout.tensors,
                layout.banks,
                4096L * 256,
                Math.max(layout.fileSize, 4096L * 256 + 4096),
                tensor -> payload(
                        seed, tensor.name(), tensor.shape(), tensor.format(), tensor.layout(), (int) tensor.byteSize()),
                (bank, expert) -> record(seed, bank, expert));
    }

    private static Layout layout(Qwen4Config config) {
        ExpectedInventory.Inventory inventory = ExpectedInventory.expected(config);
        List<Tensor> tensors = new ArrayList<>();
        List<ExpertBank> banks = new ArrayList<>();
        long cursor = 1 << 20; // generous preamble: the data section starts after the tables, on 4096
        for (Map.Entry<String, ExpectedInventory.Expected> entry :
                inventory.tensors().entrySet()) {
            ExpectedInventory.Expected want = entry.getValue();
            long size = size(want);
            long alignment = want.layout() == WeightLayout.ROW_INTERLEAVED_NVFP4_V1 ? 4096 : 256;
            cursor = alignUp(cursor, alignment);
            tensors.add(new Tensor(
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
        for (ExpectedInventory.ExpectedBank want : inventory.banks()) {
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
        return new Layout(tensors.toArray(Tensor[]::new), banks.toArray(ExpertBank[]::new), alignUp(cursor, 4096));
    }

    private static long size(ExpectedInventory.Expected want) {
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
