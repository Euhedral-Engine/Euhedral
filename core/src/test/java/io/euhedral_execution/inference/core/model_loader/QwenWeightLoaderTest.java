package io.euhedral_execution.inference.core.model_loader;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.gpu.GpuMemory;
import io.euhedral_execution.inference.core.model_loader.artifact.QwenArtifact;
import io.euhedral_execution.inference.core.model_loader.artifact.QwenArtifactHeader;
import io.euhedral_execution.inference.core.model_loader.artifact.TensorDescriptor;
import io.euhedral_execution.inference.core.model_loader.config.QwenConfig;
import io.euhedral_execution.inference.core.model_loader.config.QwenLayerType;
import io.euhedral_execution.inference.core.model_loader.layer_weights.TensorDataType;
import io.euhedral_execution.inference.core.model_loader.layer_weights.WeightFormat;
import io.euhedral_execution.inference.core.model_loader.layer_weights.WeightLayout;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.Arrays;
import org.junit.jupiter.api.Test;

/// The loader's entry points validate the artifact before touching the device.
class QwenWeightLoaderTest {

    private static final Path UNREAD = Path.of("unread.edrl");

    @Test
    void rejectsAnArtifactWithoutACompactHeader() {
        QwenConfig config = config();
        QwenArtifactHeader legacy = header(1);
        for (QwenArtifactHeader header : new QwenArtifactHeader[] {null, legacy, header(3)}) {
            QwenArtifact artifact = new QwenArtifact(header, config, new TensorDescriptor[0]);
            FakeGpuMemory gpu = new FakeGpuMemory();

            QwenWeightLoadException load =
                    assertThrows(QwenWeightLoadException.class, () -> QwenWeightLoader.load(UNREAD, artifact, gpu));
            QwenWeightLoadException firstLayer = assertThrows(
                    QwenWeightLoadException.class, () -> QwenWeightLoader.loadFirstLayer(UNREAD, artifact, gpu));

            assertTrue(load.getMessage().contains("compact version 2"));
            assertTrue(firstLayer.getMessage().contains("compact version 2"));
            assertEquals(0, gpu.allocateCalls);
        }
    }

    @Test
    void rejectsAMissingConfigOrTensorTable() {
        FakeGpuMemory gpu = new FakeGpuMemory();

        QwenWeightLoadException noConfig = assertThrows(
                QwenWeightLoadException.class,
                () -> QwenWeightLoader.load(UNREAD, new QwenArtifact(header(2), null, new TensorDescriptor[0]), gpu));
        QwenWeightLoadException noTable = assertThrows(
                QwenWeightLoadException.class,
                () -> QwenWeightLoader.load(UNREAD, new QwenArtifact(header(2), config(), null), gpu));

        assertTrue(noConfig.getMessage().contains("config is missing"));
        assertTrue(noTable.getMessage().contains("tensor table is missing"));
        assertEquals(0, gpu.allocateCalls);
    }

    @Test
    void rejectsDuplicateTensorNameBeforeGpuLoad() {
        FakeGpuMemory gpu = new FakeGpuMemory();
        QwenArtifact artifact = artifact("text/token_embedding", "text/token_embedding");

        QwenWeightLoadException failure =
                assertThrows(QwenWeightLoadException.class, () -> QwenWeightLoader.load(UNREAD, artifact, gpu));

        assertTrue(failure.getMessage().contains("duplicate tensor name"));
        assertEquals(0, gpu.allocateCalls);
    }

    @Test
    void rejectsAnInventoryThatIsNotTheCompactTextModelBeforeGpuLoad() {
        FakeGpuMemory gpu = new FakeGpuMemory();
        QwenArtifact artifact = artifact("text/token_embedding", "text/final_norm", "text/output_head");

        QwenWeightLoadException load =
                assertThrows(QwenWeightLoadException.class, () -> QwenWeightLoader.load(UNREAD, artifact, gpu));
        QwenWeightLoadException firstLayer = assertThrows(
                QwenWeightLoadException.class, () -> QwenWeightLoader.loadFirstLayer(UNREAD, artifact, gpu));

        assertTrue(load.getMessage().contains("785 runtime objects"));
        assertTrue(firstLayer.getMessage().contains("785 runtime objects"));
        assertEquals(0, gpu.allocateCalls);
    }

    @Test
    void rejectsADenseOrMoeTopologyThatIsNotTheReferenceModel() {
        QwenLayerType[] types = new QwenLayerType[2];
        Arrays.fill(types, QwenLayerType.FULL_ATTENTION);
        QwenConfig small = config(types, 0);
        QwenConfig moe = config(config().layerTypes(), 8);
        FakeGpuMemory gpu = new FakeGpuMemory();

        for (QwenConfig config : new QwenConfig[] {small, moe}) {
            QwenArtifact artifact = new QwenArtifact(header(2), config, new TensorDescriptor[0]);
            assertThrows(QwenWeightLoadException.class, () -> QwenWeightLoader.load(UNREAD, artifact, gpu));
        }
        assertEquals(0, gpu.allocateCalls);
    }

    private static QwenArtifactHeader header(int version) {
        return new QwenArtifactHeader(
                QwenArtifactHeader.MAGIC,
                version,
                QwenArtifactHeader.BYTE_SIZE,
                0,
                QwenArtifactHeader.BYTE_SIZE,
                0,
                QwenArtifactHeader.BYTE_SIZE);
    }

    private static QwenArtifact artifact(String... names) {
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
        return new QwenArtifact(header(2), config(), descriptors);
    }

    private static QwenConfig config() {
        QwenLayerType[] types = new QwenLayerType[64];
        for (int index = 0; index < types.length; index++) {
            types[index] = index % 4 == 3 ? QwenLayerType.FULL_ATTENTION : QwenLayerType.GATED_DELTA_NET;
        }
        return config(types, 0);
    }

    private static QwenConfig config(QwenLayerType[] types, int experts) {
        return new QwenConfig(
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
