package io.euhedral_execution.inference.core.model.qwen38.loader;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.artifact.TensorDataType;
import io.euhedral_execution.inference.core.artifact.TensorDescriptor;
import io.euhedral_execution.inference.core.artifact.TensorHandle;
import io.euhedral_execution.inference.core.artifact.WeightFormat;
import io.euhedral_execution.inference.core.artifact.WeightLayout;
import io.euhedral_execution.inference.core.artifact.WeightLoadException;
import io.euhedral_execution.inference.core.artifact.format.LinearFormats;
import io.euhedral_execution.inference.core.model.qwen38.ArtifactProfile;
import io.euhedral_execution.inference.core.model.qwen38.LayerType;
import io.euhedral_execution.inference.core.model.qwen38.MemoryGpu;
import io.euhedral_execution.inference.core.model.qwen38.Qwen38Config;
import io.euhedral_execution.inference.core.model.qwen38.artifact.Artifact;
import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/// A trellis inventory has the registered objects plus the three companions of every trellis object, and anything
/// else is refused by name before a byte reaches the device.
class TrellisInventoryTest {

    private static final int W = 56;
    private static final String MTP_QKGV = "mtp/layer/attention/query_key_gate_value";

    @TempDir
    Path directory;

    private Path file;
    private Map<String, Long> segmentOffsets;

    /// Counts allocations: a refused artifact makes none.
    private static final class CountingGpu extends MemoryGpu {
        int allocations;

        @Override
        public synchronized long allocate(long byteSize) {
            this.allocations++;
            return super.allocate(byteSize);
        }
    }

    @BeforeEach
    void writeSegments() throws Exception {
        this.file = this.directory.resolve("segments.edrl");
        writeFile(Map.of());
    }

    /// The file holds one segment list per distinct list; `overrides` replaces the list of the named objects.
    private void writeFile(Map<String, int[]> overrides) throws Exception {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        this.segmentOffsets = new LinkedHashMap<>();
        for (String name : trellisNames()) {
            int[] ends = overrides.getOrDefault(name, segmentsOf(name));
            this.segmentOffsets.put(name, (long) bytes.size());
            ByteBuffer buffer = ByteBuffer.allocate(ends.length * 4).order(ByteOrder.LITTLE_ENDIAN);
            for (int end : ends) buffer.putInt(end);
            bytes.write(buffer.array());
        }
        Files.write(this.file, bytes.toByteArray());
    }

    private static int[] segmentsOf(String name) {
        if (name.equals(MTP_QKGV)) return new int[] {6144, 7168, 13312, 14336};
        if (name.endsWith("/attention/query_key") || name.endsWith("/attention/gate_value"))
            return new int[] {6144, 7168};
        if (name.endsWith("/gdn/query_key")) return new int[] {4096};
        if (name.endsWith("/gdn/value_z")) return new int[] {6144, 12288};
        if (name.endsWith("/mlp/gate_up")) return new int[] {17408, 34816};
        return new int[] {5120};
    }

    private static boolean isTrellis(String name) {
        return !name.equals("text/token_embedding")
                && !name.equals("text/output_head")
                && !name.equals("text/draft_head")
                && CompactWeightLoader.expectedDescriptor(name) != null
                && CompactWeightLoader.expectedDescriptor(name).layout() == WeightLayout.ROW_SPLIT_K128_V1;
    }

    private static List<String> objectNames() {
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
                MTP_QKGV,
                "mtp/layer/attention/query_norm",
                "mtp/layer/attention/key_norm",
                "mtp/layer/attention/output",
                "mtp/layer/mlp/gate_up",
                "mtp/layer/mlp/down"));
        for (int index = 0; index < 64; index++) {
            String prefix = "text/layers/" + index;
            names.addAll(List.of(
                    prefix + "/input_norm",
                    prefix + "/post_attention_norm",
                    prefix + "/mlp/gate_up",
                    prefix + "/mlp/down"));
            if (config.layerTypes()[index] == LayerType.FULL_ATTENTION)
                names.addAll(List.of(
                        prefix + "/attention/query_key",
                        prefix + "/attention/gate_value",
                        prefix + "/attention/query_norm",
                        prefix + "/attention/key_norm",
                        prefix + "/attention/output"));
            else
                names.addAll(List.of(
                        prefix + "/gdn/a_log",
                        prefix + "/gdn/dt_bias",
                        prefix + "/gdn/convolution",
                        prefix + "/gdn/a_projection",
                        prefix + "/gdn/b_projection",
                        prefix + "/gdn/query_key",
                        prefix + "/gdn/value_z",
                        prefix + "/gdn/norm",
                        prefix + "/gdn/output"));
        }
        return names;
    }

    private static List<String> trellisNames() {
        return objectNames().stream().filter(TrellisInventoryTest::isTrellis).toList();
    }

    /// Descriptors of every registered object: the projections trellis with their companions, the rest as the
    /// registry expects them. The payloads are not in the file; only `.segments` are.
    private Map<String, TensorDescriptor> inventory() {
        Map<String, TensorDescriptor> descriptors = new LinkedHashMap<>();
        for (String name : objectNames()) {
            var expected = CompactWeightLoader.expectedDescriptor(name);
            if (!isTrellis(name)) {
                descriptors.put(
                        name,
                        new TensorDescriptor(
                                name,
                                expected.shape(),
                                expected.dataType(),
                                expected.format(),
                                expected.layout(),
                                0,
                                1));
                continue;
            }
            long n = expected.shape()[0], k = expected.shape()[1];
            long[] shape = {k / 16, n / 16, W};
            descriptors.put(
                    name,
                    new TensorDescriptor(
                            name,
                            shape,
                            TensorDataType.BF16,
                            WeightFormat.TRELLIS,
                            WeightLayout.TRELLIS_TILE16_V1,
                            0,
                            k / 16 * (n / 16) * W * 2));
            int segments = segmentsOf(name).length;
            descriptors.put(name + ".suh", fp16(name + ".suh", new long[] {segments, k}));
            descriptors.put(name + ".svh", fp16(name + ".svh", new long[] {n}));
            descriptors.put(
                    name + ".segments",
                    new TensorDescriptor(
                            name + ".segments",
                            new long[] {segments},
                            TensorDataType.INT32,
                            WeightFormat.I32,
                            WeightLayout.CONTIGUOUS_LE_V1,
                            this.segmentOffsets.get(name),
                            segments * 4L));
        }
        return descriptors;
    }

    private static TensorDescriptor fp16(String name, long[] shape) {
        long elements = 1;
        for (long dimension : shape) elements *= dimension;
        return new TensorDescriptor(
                name, shape, TensorDataType.FP16, WeightFormat.FP16, WeightLayout.CONTIGUOUS_LE_V1, 0, elements * 2);
    }

    private static Qwen38Config config() {
        LayerType[] types = new LayerType[64];
        for (int index = 0; index < types.length; index++)
            types[index] = index % 4 == 3 ? LayerType.FULL_ATTENTION : LayerType.GATED_DELTA_NET;
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

    /// Loading `descriptors` must fail naming `tensor`, with no device allocation made.
    private void assertRefused(Map<String, TensorDescriptor> descriptors, String tensor) {
        CountingGpu gpu = new CountingGpu();
        Artifact artifact = new Artifact(null, config(), descriptors.values().toArray(new TensorDescriptor[0]));
        var failure = assertThrows(
                WeightLoadException.class,
                () -> CompactWeightLoader.load(
                        this.file, artifact, gpu, descriptors, ArtifactProfile.Speculation.MTP, Set.of(), Set.of()));
        assertTrue(failure.getMessage().contains(tensor), failure.getMessage());
        assertEquals(0, gpu.allocations);
    }

    @Test
    void aFullTrellisInventoryIsAcceptedWithItsCompanionsGrouped() throws Exception {
        Map<String, TensorDescriptor> descriptors = inventory();
        int objects = trellisNames().size();
        assertEquals(785 + 3 * objects, descriptors.size());
        CompactWeightLoader.validateInventory(config(), descriptors);
        CompactWeightLoader.validateTrellisSegments(this.file, descriptors);
        assertEquals(
                3 * objects, CompactWeightLoader.trellisCompanions(descriptors).size());

        Map<String, TensorHandle> handles = new LinkedHashMap<>();
        for (String name : List.of("text/layers/0/mlp/down", "text/layers/0/mlp/down.suh")) {
            TensorDescriptor descriptor = descriptors.get(name);
            handles.put(
                    name,
                    new TensorHandle(
                            name,
                            descriptor.shape(),
                            descriptor.dataType(),
                            descriptor.format(),
                            descriptor.layout(),
                            4096,
                            descriptor.byteSize()));
        }
        assertThrows(
                IllegalArgumentException.class, () -> CompactWeightLoader.group(handles, "text/layers/0/mlp/down"));
        for (String suffix : List.of(".svh", ".segments")) {
            TensorDescriptor descriptor = descriptors.get("text/layers/0/mlp/down" + suffix);
            handles.put(
                    descriptor.name(),
                    new TensorHandle(
                            descriptor.name(),
                            descriptor.shape(),
                            descriptor.dataType(),
                            descriptor.format(),
                            descriptor.layout(),
                            4096,
                            descriptor.byteSize()));
        }
        List<String> grouped = CompactWeightLoader.group(handles, "text/layers/0/mlp/down").stream()
                .map(TensorHandle::name)
                .toList();
        assertEquals(
                List.of(
                        "text/layers/0/mlp/down",
                        "text/layers/0/mlp/down.suh",
                        "text/layers/0/mlp/down.svh",
                        "text/layers/0/mlp/down.segments"),
                grouped);
    }

    @Test
    void theEmbeddingAndHeadsCannotBeTrellis() {
        for (String name : List.of("text/token_embedding", "text/output_head")) {
            Map<String, TensorDescriptor> descriptors = inventory();
            TensorDescriptor original = descriptors.get(name);
            long[] shape = {5120 / 16, original.shape()[0] / 16, 48};
            descriptors.put(
                    name,
                    new TensorDescriptor(
                            name,
                            shape,
                            TensorDataType.BF16,
                            WeightFormat.TRELLIS,
                            WeightLayout.TRELLIS_TILE16_V1,
                            original.dataOffset(),
                            shape[0] * shape[1] * shape[2] * 2));
            assertRefused(descriptors, name);
        }
    }

    @Test
    void aMissingCompanionIsRefusedByName() {
        for (String suffix : List.of(".suh", ".svh", ".segments")) {
            Map<String, TensorDescriptor> descriptors = inventory();
            descriptors.remove("text/layers/3/attention/gate_value" + suffix);
            assertRefused(descriptors, "text/layers/3/attention/gate_value" + suffix);
        }
    }

    @Test
    void anOrphanCompanionIsRefusedByName() {
        Map<String, TensorDescriptor> descriptors = inventory();
        TensorDescriptor down = descriptors.get("text/layers/5/mlp/down");
        long[] shape = {5120, 17408};
        descriptors.put(
                down.name(),
                new TensorDescriptor(
                        down.name(),
                        shape,
                        TensorDataType.BF16,
                        WeightFormat.NVFP4,
                        WeightLayout.ROW_SPLIT_K128_V1,
                        0,
                        1));
        assertRefused(descriptors, "text/layers/5/mlp/down.suh");

        Map<String, TensorDescriptor> stray = inventory();
        stray.put("text/layers/5/mlp/other.svh", fp16("text/layers/5/mlp/other.svh", new long[] {128}));
        assertRefused(stray, "text/layers/5/mlp/other.svh");
    }

    @Test
    void aMalformedSegmentsPayloadIsRefusedByNameWhateverTheFault() throws Exception {
        String name = "text/layers/2/gdn/value_z";
        for (int[] bad : new int[][] {
            {12288, 6144}, // not increasing
            {6000, 12288}, // unaligned
            {6144, 12160}, // last is not n
            {6144, 6144} // repeated
        }) {
            writeFile(Map.of(name, bad));
            assertRefused(inventory(), name + ".segments");
        }
        // The count must equal the suh rows.
        writeFile(Map.of(name, new int[] {6144, 12288}));
        Map<String, TensorDescriptor> descriptors = inventory();
        TensorDescriptor suh = descriptors.get(name + ".suh");
        descriptors.put(name + ".suh", fp16(suh.name(), new long[] {3, 5120}));
        assertRefused(descriptors, name + ".segments");
    }

    @Test
    void aMtpAttentionProjectionMustBeCuttableAtHalf() throws Exception {
        writeFile(Map.of(MTP_QKGV, new int[] {6144, 13312, 14336}));
        Map<String, TensorDescriptor> descriptors = inventory();
        descriptors.put(MTP_QKGV + ".suh", fp16(MTP_QKGV + ".suh", new long[] {3, 5120}));
        descriptors.put(
                MTP_QKGV + ".segments",
                new TensorDescriptor(
                        MTP_QKGV + ".segments",
                        new long[] {3},
                        TensorDataType.INT32,
                        WeightFormat.I32,
                        WeightLayout.CONTIGUOUS_LE_V1,
                        this.segmentOffsets.get(MTP_QKGV),
                        12));
        assertRefused(descriptors, MTP_QKGV + ".segments");
    }

    @Test
    void aWrongCompanionShapeOrTypeIsRefusedByName() {
        String name = "text/layers/0/mlp/gate_up";
        Map<String, TensorDescriptor> suh = inventory();
        suh.put(name + ".suh", fp16(name + ".suh", new long[] {2, 5248}));
        assertRefused(suh, name + ".suh");

        Map<String, TensorDescriptor> svh = inventory();
        svh.put(name + ".svh", fp16(name + ".svh", new long[] {34688}));
        assertRefused(svh, name + ".svh");

        Map<String, TensorDescriptor> type = inventory();
        TensorDescriptor segments = type.get(name + ".segments");
        type.put(
                name + ".segments",
                new TensorDescriptor(
                        segments.name(),
                        segments.shape(),
                        TensorDataType.FP16,
                        WeightFormat.FP16,
                        WeightLayout.CONTIGUOUS_LE_V1,
                        segments.dataOffset(),
                        segments.byteSize()));
        assertRefused(type, name + ".segments");
    }

    @Test
    void aCompanionNeverLeavesTheDevice() {
        Map<String, TensorDescriptor> descriptors = inventory();
        String suh = "text/layers/0/mlp/down.suh";
        assertThrows(
                IllegalArgumentException.class,
                () -> CompactWeightLoader.loadObjects(
                        this.file,
                        descriptors.values(),
                        new CountingGpu(),
                        ArtifactProfile.Speculation.NONE,
                        Set.of(),
                        Set.of(suh)));
        assertThrows(
                IllegalArgumentException.class,
                () -> CompactWeightLoader.loadObjects(
                        this.file,
                        descriptors.values(),
                        new CountingGpu(),
                        ArtifactProfile.Speculation.NONE,
                        Set.of(suh),
                        Set.of()));
    }

    @Test
    void theProfileIsTrellisNotCompressed() {
        Map<String, TensorDescriptor> descriptors = inventory();
        var profile = ArtifactProfile.of(
                new Artifact(null, config(), descriptors.values().toArray(new TensorDescriptor[0])));
        assertEquals(ArtifactProfile.Quantization.TRELLIS, profile.quantization());
        assertFalse(profile.compressed());
        assertEquals("trellis", profile.artifactName());
        assertEquals(4, profile.draftLength().most());
        assertEquals(io.euhedral_execution.inference.core.state.KvFormat.NVFP4, profile.kvFormat());
    }

    @Test
    void theResidencyPlanReservesThePrefillScratchOfTheLargestTrellisObject() {
        Map<String, TensorDescriptor> descriptors = inventory();
        Artifact artifact = new Artifact(null, config(), descriptors.values().toArray(new TensorDescriptor[0]));
        var profile =
                new ArtifactProfile(ArtifactProfile.Quantization.TRELLIS, false, ArtifactProfile.Speculation.NONE);
        long expected = LinearFormats.of(descriptors.get("text/layers/0/mlp/gate_up"))
                .planScratchBytes(
                        descriptors.get("text/layers/0/mlp/gate_up"),
                        io.euhedral_execution.inference.core.InferenceConfig.PREFILL_CHUNK_TOKENS);
        assertEquals(expected, ResidencyPlanner.trellisScratchBytes(artifact, profile));
        assertTrue(expected > 34816L * 5120 * 2);

        Map<String, TensorDescriptor> plain = new LinkedHashMap<>(descriptors);
        plain.values().removeIf(d -> d.format() == WeightFormat.TRELLIS);
        assertEquals(
                0,
                ResidencyPlanner.trellisScratchBytes(
                        new Artifact(null, config(), plain.values().toArray(new TensorDescriptor[0])), profile));
    }

    @Test
    void onlyThePayloadIsStageable() {
        Map<String, TensorDescriptor> descriptors = inventory();
        String name = "text/layers/4/mlp/gate_up";
        assertTrue(HostWeightSelection.stageable(descriptors.get(name)));
        for (String suffix : List.of(".suh", ".svh", ".segments"))
            assertFalse(HostWeightSelection.stageable(descriptors.get(name + suffix)), suffix);
        Artifact artifact = new Artifact(null, config(), descriptors.values().toArray(new TensorDescriptor[0]));
        Set<String> selected = HostWeightSelection.select(artifact, Long.MAX_VALUE);
        assertTrue(selected.contains(name));
        for (String selectedName : selected)
            assertFalse(
                    selectedName.endsWith(".suh")
                            || selectedName.endsWith(".svh")
                            || selectedName.endsWith(".segments"),
                    selectedName);
    }

    @Test
    void theMtpProjectionSplitsAtHalfWithItsCompanionsCutAndRebased() {
        var format = LinearFormats.find(WeightFormat.TRELLIS, WeightLayout.TRELLIS_TILE16_V1)
                .orElseThrow();
        MemoryGpu gpu = new MemoryGpu();
        int k = 128, n = 14336, w = 40;
        int tilesK = k / 16, tilesN = n / 16;
        int[] ends = {6144, 7168, 13312, 14336};
        byte[] payload = new byte[tilesK * tilesN * w * 2];
        for (int index = 0; index < payload.length; index++) payload[index] = (byte) (index * 31 + (index >>> 9));
        byte[] suh = new byte[ends.length * k * 2];
        for (int index = 0; index < suh.length; index++) suh[index] = (byte) (index / (k * 2) * 40 + 1 + index % 7);
        byte[] svh = new byte[n * 2];
        for (int index = 0; index < svh.length; index++) svh[index] = (byte) (index * 13 + 5);
        ByteBuffer segments = ByteBuffer.allocate(16).order(ByteOrder.LITTLE_ENDIAN);
        for (int end : ends) segments.putInt(end);

        List<TensorHandle> object = List.of(
                upload(
                        gpu,
                        MTP_QKGV,
                        new long[] {tilesK, tilesN, w},
                        TensorDataType.BF16,
                        WeightFormat.TRELLIS,
                        WeightLayout.TRELLIS_TILE16_V1,
                        payload),
                upload(
                        gpu,
                        MTP_QKGV + ".suh",
                        new long[] {4, k},
                        TensorDataType.FP16,
                        WeightFormat.FP16,
                        WeightLayout.CONTIGUOUS_LE_V1,
                        suh),
                upload(
                        gpu,
                        MTP_QKGV + ".svh",
                        new long[] {n},
                        TensorDataType.FP16,
                        WeightFormat.FP16,
                        WeightLayout.CONTIGUOUS_LE_V1,
                        svh),
                upload(
                        gpu,
                        MTP_QKGV + ".segments",
                        new long[] {4},
                        TensorDataType.INT32,
                        WeightFormat.I32,
                        WeightLayout.CONTIGUOUS_LE_V1,
                        segments.array()));

        for (int half = 0; half < 2; half++) {
            String name = half == 0 ? "mtp/layer/attention/query_key" : "mtp/layer/attention/gate_value";
            long first = half * 7168L;
            List<TensorHandle> cut = format.rows(object, first, 7168, name, gpu, ends);
            assertEquals(
                    List.of(name, name + ".suh", name + ".svh", name + ".segments"),
                    cut.stream().map(TensorHandle::name).toList());
            assertEquals(
                    List.of(tilesK, tilesN / 2, w),
                    java.util.Arrays.stream(cut.get(0).shape())
                            .boxed()
                            .map(Long::intValue)
                            .toList());
            assertEquals(WeightFormat.TRELLIS, cut.get(0).format());
            assertArrayEquals(new long[] {2, k}, cut.get(1).shape());
            assertArrayEquals(new long[] {7168}, cut.get(2).shape());
            assertArrayEquals(new long[] {2}, cut.get(3).shape());

            ByteBuffer cutSegments =
                    ByteBuffer.wrap(gpu.bytes(cut.get(3).deviceAddress(), 8)).order(ByteOrder.LITTLE_ENDIAN);
            assertEquals(6144, cutSegments.getInt());
            assertEquals(7168, cutSegments.getInt());
            assertArrayEquals(
                    java.util.Arrays.copyOfRange(suh, half * 2 * k * 2, (half * 2 + 2) * k * 2),
                    gpu.bytes(cut.get(1).deviceAddress(), 2 * k * 2));
            assertArrayEquals(
                    java.util.Arrays.copyOfRange(svh, (int) first * 2, (int) (first + 7168) * 2),
                    gpu.bytes(cut.get(2).deviceAddress(), 7168 * 2));
            // Each k-tile row keeps its half of the tile columns.
            byte[] cutPayload =
                    gpu.bytes(cut.get(0).deviceAddress(), (int) cut.get(0).byteSize());
            int rowBytes = tilesN / 2 * w * 2;
            for (int row = 0; row < tilesK; row++)
                assertArrayEquals(
                        java.util.Arrays.copyOfRange(
                                payload,
                                row * tilesN * w * 2 + half * rowBytes,
                                row * tilesN * w * 2 + (half + 1) * rowBytes),
                        java.util.Arrays.copyOfRange(cutPayload, row * rowBytes, (row + 1) * rowBytes),
                        "tile row " + row);
        }
        // A cut inside a segment is refused.
        assertThrows(IllegalArgumentException.class, () -> format.rows(object, 0, 6400, "x", gpu, ends));
        assertDoesNotThrow(() -> format.rows(object, 0, 6144, "y", gpu, ends));
    }

    private static TensorHandle upload(
            MemoryGpu gpu,
            String name,
            long[] shape,
            TensorDataType dataType,
            WeightFormat format,
            WeightLayout layout,
            byte[] contents) {
        long address = gpu.allocate(contents.length);
        try (var arena = java.lang.foreign.Arena.ofConfined()) {
            var segment = arena.allocate(contents.length);
            java.lang.foreign.MemorySegment.copy(
                    contents, 0, segment, java.lang.foreign.ValueLayout.JAVA_BYTE, 0, contents.length);
            gpu.copyHostToDevice(address, segment, contents.length);
        }
        return new TensorHandle(name, shape, dataType, format, layout, address, contents.length);
    }
}
