package io.euhedral_execution.inference.core.model.qwen38.loader;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.artifact.TensorDataType;
import io.euhedral_execution.inference.core.artifact.TensorDescriptor;
import io.euhedral_execution.inference.core.artifact.TensorHandle;
import io.euhedral_execution.inference.core.artifact.WeightFormat;
import io.euhedral_execution.inference.core.artifact.WeightLayout;
import io.euhedral_execution.inference.core.artifact.WeightLoadException;
import io.euhedral_execution.inference.core.model.qwen38.ArtifactProfile;
import io.euhedral_execution.inference.core.model.qwen38.LayerType;
import io.euhedral_execution.inference.core.model.qwen38.Qwen38Config;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class CompactWeightLoaderTest {

    @Test
    void assemblesFusedRuntimeObjectsWithoutSplittingHandles() throws Exception {
        Qwen38Config config = config();
        Map<String, TensorHandle> handles = new LinkedHashMap<>();
        handles.put("text/token_embedding", handle("text/token_embedding"));
        handles.put("text/final_norm", handle("text/final_norm"));
        handles.put("text/output_head", handle("text/output_head"));
        for (int index = 0; index < 64; index++) {
            String prefix = "text/layers/" + index;
            handles.put(prefix + "/input_norm", handle(prefix + "/input_norm"));
            handles.put(prefix + "/post_attention_norm", handle(prefix + "/post_attention_norm"));
            handles.put(prefix + "/mlp/gate_up", handle(prefix + "/mlp/gate_up"));
            handles.put(prefix + "/mlp/down", handle(prefix + "/mlp/down"));
            if (config.layerTypes()[index] == LayerType.FULL_ATTENTION) {
                for (String suffix : new String[] {
                    "/attention/query_key",
                    "/attention/gate_value",
                    "/attention/query_norm",
                    "/attention/key_norm",
                    "/attention/output"
                }) {
                    handles.put(prefix + suffix, handle(prefix + suffix));
                }
            } else {
                for (String suffix : new String[] {
                    "/gdn/a_log",
                    "/gdn/dt_bias",
                    "/gdn/convolution",
                    "/gdn/a_projection",
                    "/gdn/b_projection",
                    "/gdn/query_key",
                    "/gdn/value_z",
                    "/gdn/norm",
                    "/gdn/output"
                }) {
                    handles.put(prefix + suffix, handle(prefix + suffix));
                }
            }
        }
        handles.put("mtp/input_projection", handle("mtp/input_projection"));
        handles.put("mtp/embedding_norm", handle("mtp/embedding_norm"));
        handles.put("mtp/hidden_norm", handle("mtp/hidden_norm"));
        handles.put("mtp/final_norm", handle("mtp/final_norm"));
        handles.put("mtp/layer/input_norm", handle("mtp/layer/input_norm"));
        handles.put("mtp/layer/post_attention_norm", handle("mtp/layer/post_attention_norm"));
        for (String suffix : new String[] {
            "/attention/query_key_gate_value",
            "/attention/query_norm",
            "/attention/key_norm",
            "/attention/output",
            "/mlp/gate_up",
            "/mlp/down"
        }) {
            handles.put("mtp/layer" + suffix, handle("mtp/layer" + suffix));
        }
        handles.put("text/draft_head", handle("text/draft_head"));
        handles.put("text/draft_head_token_ids", handle("text/draft_head_token_ids"));
        handles.put("vision/sentinel", handle("vision/sentinel"));

        Weights weights = CompactWeightLoader.assemble(config, handles);

        LayerWeights gdnLayer = weights.layers()[0];
        assertInstanceOf(GdnWeights.class, gdnLayer.mixer());
        assertInstanceOf(DenseFfnWeights.class, gdnLayer.ffn());
        assertInstanceOf(AttentionWeights.class, weights.layers()[3].mixer());
        assertEquals(
                "text/layers/3/attention/query_key",
                ((AttentionWeights) weights.layers()[3].mixer()).queryKey().name());
        assertInstanceOf(MtpAttentionWeights.class, weights.mtp().layer().mixer());
        assertEquals(handles.size(), weights.runtimeObjects().size());
        assertThrows(
                UnsupportedOperationException.class,
                () -> weights.runtimeObjects().put("unexpected", handle("unexpected")));
    }

    private static TensorHandle handle(String name) {
        return new TensorHandle(name, new long[] {1}, TensorDataType.BF16, WeightFormat.BF16, 1L, 2L);
    }

    @Test
    void acceptsTheNvfp4TextInventoryWithoutVision() throws Exception {
        CompactWeightLoader.validateInventory(config(), nvfp4Inventory());
    }

    private static final io.euhedral_execution.inference.core.model.qwen38.loader.DFlash2Config DFLASH2 =
            io.euhedral_execution.inference.core.model.qwen38.loader.DFlash2Config.parse(new int[] {
                1,
                5,
                5120,
                17408,
                32,
                8,
                128,
                8,
                248070,
                2,
                16,
                16,
                256,
                2048,
                248320,
                Float.floatToIntBits(1e-6f),
                Float.floatToIntBits(1e7f),
                5,
                5,
                19,
                33,
                47,
                61
            });

    private static Map<String, TensorDescriptor> withDFlash2(boolean nvfp4Projections) {
        Map<String, TensorDescriptor> descriptors = nvfp4Inventory();
        descriptors.put(
                "dflash2/config",
                new TensorDescriptor(
                        "dflash2/config",
                        new long[] {23},
                        TensorDataType.INT32,
                        WeightFormat.I32,
                        WeightLayout.CONTIGUOUS_LE_V1,
                        0,
                        92));
        DFlash2Inventory.expected(DFLASH2)
                .forEach((name, entry) -> descriptors.put(
                        name,
                        nvfp4Projections && entry.projection()
                                ? descriptor(name, entry.shape(), WeightFormat.NVFP4)
                                : new TensorDescriptor(
                                        name,
                                        entry.shape(),
                                        TensorDataType.BF16,
                                        WeightFormat.BF16,
                                        WeightLayout.CONTIGUOUS_LE_V1,
                                        0,
                                        1)));
        return descriptors;
    }

    @Test
    void acceptsTheDFlash2DrafterBesideTheTextInventory() throws Exception {
        CompactWeightLoader.validateInventory(config(), withDFlash2(false), DFLASH2);
        CompactWeightLoader.validateInventory(config(), withDFlash2(true), DFLASH2);
        assertEquals(785 + 72, withDFlash2(false).size(), "5 x 13 layer objects, 3 shared, 3 selector, config");
    }

    @Test
    void rejectsADFlash2ArtifactReadWithoutItsConfigurationOrWithAnUnknownObject() {
        assertThrows(
                WeightLoadException.class, () -> CompactWeightLoader.validateInventory(config(), withDFlash2(false)));
        Map<String, TensorDescriptor> extra = withDFlash2(false);
        extra.put(
                "dflash2/layers/5/input_norm",
                descriptor("dflash2/layers/5/input_norm", new long[] {5120}, WeightFormat.BF16));
        extra.remove("dflash2/final_norm");
        assertThrows(WeightLoadException.class, () -> CompactWeightLoader.validateInventory(config(), extra, DFLASH2));
        Map<String, TensorDescriptor> quantizedNorm = withDFlash2(false);
        quantizedNorm.put(
                "dflash2/hidden_norm", descriptor("dflash2/hidden_norm", new long[] {5120}, WeightFormat.NVFP4));
        assertThrows(
                WeightLoadException.class,
                () -> CompactWeightLoader.validateInventory(config(), quantizedNorm, DFLASH2));
    }

    @Test
    void rejectsAPartialVisionTower() throws Exception {
        Map<String, TensorDescriptor> descriptors = nvfp4Inventory();
        descriptors.put(
                "vision/merger/fc1", descriptor("vision/merger/fc1", new long[] {4608, 4608}, WeightFormat.NVFP4));
        assertThrows(WeightLoadException.class, () -> CompactWeightLoader.validateInventory(config(), descriptors));
    }

    @Test
    void rejectsNvfp4WhereADirectTensorIsRegistered() throws Exception {
        Map<String, TensorDescriptor> descriptors = nvfp4Inventory();
        descriptors.put("text/final_norm", descriptor("text/final_norm", new long[] {5120}, WeightFormat.NVFP4));
        assertThrows(WeightLoadException.class, () -> CompactWeightLoader.validateInventory(config(), descriptors));
    }

    @Test
    void acceptsAnyGroupedFormatWhereAGroupedOneIsRegistered() throws Exception {
        Map<String, TensorDescriptor> descriptors = nvfp4Inventory();
        // Q3 down and output slots in Q4, a Q5 value_z slot in Q4, a Q3 slot in Q5.
        descriptors.put(
                "text/layers/0/mlp/down",
                descriptor("text/layers/0/mlp/down", new long[] {5120, 17408}, WeightFormat.Q4_G64_FP16));
        descriptors.put(
                "text/layers/0/gdn/output",
                descriptor("text/layers/0/gdn/output", new long[] {5120, 6144}, WeightFormat.Q4_G64_FP16));
        descriptors.put(
                "text/layers/0/gdn/value_z",
                descriptor("text/layers/0/gdn/value_z", new long[] {12288, 5120}, WeightFormat.Q4_G64_FP16));
        descriptors.put(
                "text/layers/3/attention/output",
                descriptor("text/layers/3/attention/output", new long[] {5120, 6144}, WeightFormat.Q5_G64_FP16));
        CompactWeightLoader.validateInventory(config(), descriptors);

        // P2E2 stores Q3 only.
        descriptors.put(
                "text/layers/0/mlp/down",
                new TensorDescriptor(
                        "text/layers/0/mlp/down",
                        new long[] {5120, 17408},
                        TensorDataType.BF16,
                        WeightFormat.Q4_G64_FP16,
                        WeightLayout.ROW_SPLIT_P2E2_V1,
                        0,
                        1));
        assertThrows(WeightLoadException.class, () -> CompactWeightLoader.validateInventory(config(), descriptors));
    }

    /// The registered text and MTP inventory with every row-split object except the embedding in NVFP4.
    private static Map<String, TensorDescriptor> nvfp4Inventory() {
        Qwen38Config config = config();
        List<String> names = new ArrayList<>(List.of(
                "text/token_embedding",
                "text/final_norm",
                "text/output_head",
                "text/draft_head",
                "text/draft_head_token_ids",
                "mtp/input_projection",
                "mtp/embedding_norm",
                "mtp/hidden_norm",
                "mtp/final_norm",
                "mtp/layer/input_norm",
                "mtp/layer/post_attention_norm",
                "mtp/layer/attention/query_key_gate_value",
                "mtp/layer/attention/query_norm",
                "mtp/layer/attention/key_norm",
                "mtp/layer/attention/output",
                "mtp/layer/mlp/gate_up",
                "mtp/layer/mlp/down"));
        for (int index = 0; index < 64; index++) {
            String prefix = "text/layers/" + index;
            List<String> suffixes =
                    new ArrayList<>(List.of("/input_norm", "/post_attention_norm", "/mlp/gate_up", "/mlp/down"));
            suffixes.addAll(
                    config.layerTypes()[index] == LayerType.FULL_ATTENTION
                            ? List.of(
                                    "/attention/query_key",
                                    "/attention/gate_value",
                                    "/attention/query_norm",
                                    "/attention/key_norm",
                                    "/attention/output")
                            : List.of(
                                    "/gdn/a_log",
                                    "/gdn/dt_bias",
                                    "/gdn/convolution",
                                    "/gdn/a_projection",
                                    "/gdn/b_projection",
                                    "/gdn/query_key",
                                    "/gdn/value_z",
                                    "/gdn/norm",
                                    "/gdn/output"));
            for (String suffix : suffixes) names.add(prefix + suffix);
        }
        Map<String, TensorDescriptor> descriptors = new LinkedHashMap<>();
        for (String name : names) {
            var expected = CompactWeightLoader.expectedDescriptor(name);
            WeightFormat format =
                    expected.layout() == WeightLayout.ROW_SPLIT_K128_V1 && !name.equals("text/token_embedding")
                            ? WeightFormat.NVFP4
                            : expected.format();
            descriptors.put(
                    name,
                    new TensorDescriptor(name, expected.shape(), expected.dataType(), format, expected.layout(), 0, 1));
        }
        assertEquals(785, descriptors.size());
        return descriptors;
    }

    private static TensorDescriptor descriptor(String name, long[] shape, WeightFormat format) {
        return new TensorDescriptor(name, shape, TensorDataType.BF16, format, WeightLayout.ROW_SPLIT_K128_V1, 0, 1);
    }

    private static Qwen38Config config() {
        LayerType[] types = new LayerType[64];
        for (int index = 0; index < types.length; index++) {
            types[index] = index % 4 == 3 ? LayerType.FULL_ATTENTION : LayerType.GATED_DELTA_NET;
        }
        return new Qwen38Config(
                248320,
                5120,
                64,
                24,
                4,
                256,
                17408,
                16,
                48,
                128,
                128,
                4,
                1.0e-6,
                10_000_000.0,
                0.25,
                262144,
                "silu",
                types,
                0,
                0,
                0,
                0,
                false,
                true,
                1);
    }

    @TempDir
    Path directory;

    /// Three BF16 objects in a file: the embedding (host memory), layer 0's down projection (disk) and the final
    /// norm (device).
    private Map<String, TensorDescriptor> smallFile(Path file) throws Exception {
        String[] names = {"text/token_embedding", "text/layers/0/mlp/down", "text/final_norm"};
        Map<String, TensorDescriptor> descriptors = new LinkedHashMap<>();
        byte[] contents = new byte[3 * 8192];
        for (int index = 0; index < contents.length; index++) contents[index] = (byte) (index * 7);
        for (int index = 0; index < names.length; index++)
            descriptors.put(
                    names[index],
                    new TensorDescriptor(
                            names[index],
                            new long[] {64, 32},
                            TensorDataType.BF16,
                            WeightFormat.BF16,
                            WeightLayout.CONTIGUOUS_LE_V1,
                            256 + 8192L * index,
                            4096));
        Files.write(file, contents);
        return descriptors;
    }

    @Test
    void diskBackedObjectsStayInTheFile() throws Exception {
        Path file = directory.resolve("small.edrl");
        Map<String, TensorDescriptor> descriptors = smallFile(file);
        var gpu = new CountingGpuMemory();
        Map<String, TensorHandle> handles = CompactWeightLoader.loadObjects(
                file,
                descriptors.values(),
                gpu,
                ArtifactProfile.Speculation.NONE,
                Set.of("text/token_embedding"),
                Set.of("text/layers/0/mlp/down"));
        TensorHandle down = handles.get("text/layers/0/mlp/down");
        assertTrue(down.diskBacked());
        assertFalse(down.hostBacked());
        assertEquals(descriptors.get("text/layers/0/mlp/down").dataOffset(), down.fileOffset());
        assertEquals(0L, down.deviceAddress());
        assertEquals(4096, down.byteSize());
        assertFalse(handles.get("text/final_norm").diskBacked());
        assertFalse(handles.get("text/token_embedding").diskBacked());
        // The final norm on the device, the embedding in a one-object arena; nothing for the disk-backed object.
        assertEquals(4096, gpu.deviceBytes);
        assertEquals(4096, gpu.hostBytes);
    }

    @Test
    void aHostBackedOutputHeadIsMappedNotStaged() throws Exception {
        Path file = directory.resolve("small.edrl");
        smallFile(file);
        TensorDescriptor head = new TensorDescriptor(
                "text/output_head",
                new long[] {64, 32},
                TensorDataType.BF16,
                WeightFormat.BF16,
                WeightLayout.CONTIGUOUS_LE_V1,
                256,
                4096);
        var gpu = new CountingGpuMemory();
        Map<String, TensorHandle> handles = CompactWeightLoader.loadObjects(
                file,
                java.util.List.of(head),
                gpu,
                ArtifactProfile.Speculation.NONE,
                Set.of("text/output_head"),
                Set.of());
        TensorHandle loaded = handles.get("text/output_head");
        assertTrue(loaded.hostMapped());
        assertFalse(loaded.hostBacked());
        assertEquals(4096, gpu.hostBytes);
        assertEquals(0, gpu.deviceBytes);
    }

    @Test
    void anObjectThatCannotBeStagedIsNeverDiskBacked() throws Exception {
        Path file = directory.resolve("small.edrl");
        Map<String, TensorDescriptor> descriptors = smallFile(file);
        var failure = assertThrows(
                IllegalArgumentException.class,
                () -> CompactWeightLoader.loadObjects(
                        file,
                        descriptors.values(),
                        new CountingGpuMemory(),
                        ArtifactProfile.Speculation.NONE,
                        Set.of(),
                        Set.of("text/final_norm")));
        assertTrue(failure.getMessage().contains("text/final_norm cannot be read from disk"));
    }

    private static final class CountingGpuMemory implements io.euhedral_execution.inference.core.gpu.GpuMemory {
        long deviceBytes;
        long hostBytes;
        private long next = 0x10000;

        @Override
        public long allocate(long byteSize) {
            this.deviceBytes += byteSize;
            return this.next += byteSize + 256;
        }

        @Override
        public long allocateHostWeights(long byteSize) {
            this.hostBytes += byteSize;
            return java.lang.foreign.Arena.global().allocate(byteSize, 4096).address();
        }

        @Override
        public long hostWeightsDeviceAddress(long hostAddress) {
            return hostAddress;
        }

        @Override
        public void copyHostToDevice(long destination, java.lang.foreign.MemorySegment source, long byteSize) {}

        @Override
        public void copyDeviceToHost(java.lang.foreign.MemorySegment destination, long source, long byteSize) {}

        @Override
        public void free(long address) {}
    }

    @Test
    void bf16ValuesStandForAnyQuantizedObjectOfTheSameShape() {
        var gateUp = CompactWeightLoader.expectedDescriptor("text/layers/0/mlp/gate_up");
        var head = CompactWeightLoader.expectedDescriptor("text/output_head");
        org.junit.jupiter.api.Assertions.assertTrue(CompactWeightLoader.accepts(gateUp, bf16(gateUp.shape())));
        org.junit.jupiter.api.Assertions.assertTrue(CompactWeightLoader.accepts(head, bf16(head.shape())));
        org.junit.jupiter.api.Assertions.assertFalse(
                CompactWeightLoader.accepts(gateUp, bf16(new long[] {gateUp.shape()[0], 4096})));
        org.junit.jupiter.api.Assertions.assertFalse(CompactWeightLoader.accepts(
                gateUp,
                new io.euhedral_execution.inference.core.artifact.TensorDescriptor(
                        "text/layers/0/mlp/gate_up",
                        gateUp.shape(),
                        io.euhedral_execution.inference.core.artifact.TensorDataType.BF16,
                        io.euhedral_execution.inference.core.artifact.WeightFormat.BF16,
                        io.euhedral_execution.inference.core.artifact.WeightLayout.ROW_SPLIT_K128_V1,
                        0,
                        2)));
    }

    private static io.euhedral_execution.inference.core.artifact.TensorDescriptor bf16(long[] shape) {
        return new io.euhedral_execution.inference.core.artifact.TensorDescriptor(
                "object",
                shape,
                io.euhedral_execution.inference.core.artifact.TensorDataType.BF16,
                io.euhedral_execution.inference.core.artifact.WeightFormat.BF16,
                io.euhedral_execution.inference.core.artifact.WeightLayout.CONTIGUOUS_LE_V1,
                0,
                2 * shape[0] * shape[1]);
    }
}
