package io.euhedral_execution.inference.core.model.qwen38.loader;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.artifact.TensorDataType;
import io.euhedral_execution.inference.core.artifact.TensorDescriptor;
import io.euhedral_execution.inference.core.artifact.WeightFormat;
import io.euhedral_execution.inference.core.artifact.WeightLayout;
import io.euhedral_execution.inference.core.artifact.WeightLoadException;
import io.euhedral_execution.inference.core.gpu.GpuMemory;
import io.euhedral_execution.inference.core.model.qwen38.LayerType;
import io.euhedral_execution.inference.core.model.qwen38.Qwen38Config;
import io.euhedral_execution.inference.core.model.qwen38.artifact.Artifact;
import io.euhedral_execution.inference.core.model.qwen38.artifact.ArtifactHeader;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.Arrays;
import org.junit.jupiter.api.Test;

/// The loader's entry points validate the artifact before touching the device.
class WeightLoaderTest {

    private static final Path UNREAD = Path.of("unread.edrl");

    @Test
    void rejectsAnArtifactWithoutACompactHeader() {
        Qwen38Config config = config();
        ArtifactHeader legacy = header(1);
        for (ArtifactHeader header : new ArtifactHeader[] {null, legacy, header(3)}) {
            Artifact artifact = new Artifact(header, config, new TensorDescriptor[0]);
            FakeGpuMemory gpu = new FakeGpuMemory();

            WeightLoadException load =
                    assertThrows(WeightLoadException.class, () -> WeightLoader.load(UNREAD, artifact, gpu));
            WeightLoadException firstLayer =
                    assertThrows(WeightLoadException.class, () -> WeightLoader.loadFirstLayer(UNREAD, artifact, gpu));

            assertTrue(load.getMessage().contains("compact version 2"));
            assertTrue(firstLayer.getMessage().contains("compact version 2"));
            assertEquals(0, gpu.allocateCalls);
        }
    }

    @Test
    void rejectsAMissingConfigOrTensorTable() {
        FakeGpuMemory gpu = new FakeGpuMemory();

        WeightLoadException noConfig = assertThrows(
                WeightLoadException.class,
                () -> WeightLoader.load(UNREAD, new Artifact(header(2), null, new TensorDescriptor[0]), gpu));
        WeightLoadException noTable = assertThrows(
                WeightLoadException.class,
                () -> WeightLoader.load(UNREAD, new Artifact(header(2), config(), null), gpu));

        assertTrue(noConfig.getMessage().contains("config is missing"));
        assertTrue(noTable.getMessage().contains("tensor table is missing"));
        assertEquals(0, gpu.allocateCalls);
    }

    @Test
    void rejectsDuplicateTensorNameBeforeGpuLoad() {
        FakeGpuMemory gpu = new FakeGpuMemory();
        Artifact artifact = artifact("text/token_embedding", "text/token_embedding");

        WeightLoadException failure =
                assertThrows(WeightLoadException.class, () -> WeightLoader.load(UNREAD, artifact, gpu));

        assertTrue(failure.getMessage().contains("duplicate tensor name"));
        assertEquals(0, gpu.allocateCalls);
    }

    @Test
    void rejectsAnInventoryThatIsNotTheCompactTextModelBeforeGpuLoad() {
        FakeGpuMemory gpu = new FakeGpuMemory();
        Artifact artifact = artifact("text/token_embedding", "text/final_norm", "text/output_head");

        WeightLoadException load =
                assertThrows(WeightLoadException.class, () -> WeightLoader.load(UNREAD, artifact, gpu));
        WeightLoadException firstLayer =
                assertThrows(WeightLoadException.class, () -> WeightLoader.loadFirstLayer(UNREAD, artifact, gpu));

        assertTrue(load.getMessage().contains("785 runtime objects"));
        assertTrue(firstLayer.getMessage().contains("785 runtime objects"));
        assertEquals(0, gpu.allocateCalls);
    }

    @Test
    void rejectsADenseOrMoeTopologyThatIsNotTheReferenceModel() {
        LayerType[] types = new LayerType[2];
        Arrays.fill(types, LayerType.FULL_ATTENTION);
        Qwen38Config small = config(types, 0);
        Qwen38Config moe = config(config().layerTypes(), 8);
        FakeGpuMemory gpu = new FakeGpuMemory();

        for (Qwen38Config config : new Qwen38Config[] {small, moe}) {
            Artifact artifact = new Artifact(header(2), config, new TensorDescriptor[0]);
            assertThrows(WeightLoadException.class, () -> WeightLoader.load(UNREAD, artifact, gpu));
        }
        assertEquals(0, gpu.allocateCalls);
    }

    private static ArtifactHeader header(int version) {
        return new ArtifactHeader(
                ArtifactHeader.MAGIC,
                version,
                ArtifactHeader.BYTE_SIZE,
                0,
                ArtifactHeader.BYTE_SIZE,
                0,
                ArtifactHeader.BYTE_SIZE);
    }

    private static Artifact artifact(String... names) {
        TensorDescriptor[] descriptors = Arrays.stream(names)
                .map(name -> new TensorDescriptor(
                        name,
                        new long[] {1},
                        TensorDataType.BF16,
                        WeightFormat.BF16,
                        WeightLayout.CONTIGUOUS_LE_V1,
                        0,
                        2))
                .toArray(TensorDescriptor[]::new);
        return new Artifact(header(2), config(), descriptors);
    }

    private static Qwen38Config config() {
        LayerType[] types = new LayerType[64];
        for (int index = 0; index < types.length; index++) {
            types[index] = index % 4 == 3 ? LayerType.FULL_ATTENTION : LayerType.GATED_DELTA_NET;
        }
        return config(types, 0);
    }

    private static Qwen38Config config(LayerType[] types, int experts) {
        return new Qwen38Config(
                248320,
                5120,
                types.length,
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
                experts,
                experts == 0 ? 0 : 2,
                experts == 0 ? 0 : 4,
                0,
                false,
                true,
                1);
    }

    private static final class FakeGpuMemory implements GpuMemory {

        private int allocateCalls;

        @Override
        public long allocate(long byteSize) {
            allocateCalls++;
            return 0x1000L + allocateCalls;
        }

        @Override
        public void copyHostToDevice(long destination, MemorySegment source, long byteSize) {}

        @Override
        public void copyDeviceToHost(MemorySegment destination, long source, long byteSize) {
            throw new UnsupportedOperationException();
        }

        @Override
        public void free(long address) {}
    }
}
