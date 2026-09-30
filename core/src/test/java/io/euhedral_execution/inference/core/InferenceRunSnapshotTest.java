package io.euhedral_execution.inference.core;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.euhedral_execution.inference.core.model_loader.config.QwenConfig;
import io.euhedral_execution.inference.core.model_loader.config.QwenLayerType;
import io.euhedral_execution.inference.core.sampling.GenerationConfig;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

class InferenceRunSnapshotTest {
    static QwenConfig config(QwenLayerType[] layers) {
        return new QwenConfig(
                248320, 5120, 4, 24, 4, 256, 17408, 16, 48, 128, 128, 4, 1e-6, 1e7, 0.25, 262144, "silu", layers, 0, 0,
                0, 0, false, true, 1);
    }

    static InferenceRunSnapshot snapshot(GenerationConfig generation) {
        QwenLayerType[] layers = {
            QwenLayerType.GATED_DELTA_NET,
            QwenLayerType.GATED_DELTA_NET,
            QwenLayerType.GATED_DELTA_NET,
            QwenLayerType.FULL_ATTENTION
        };
        return new InferenceRunSnapshot(
                InferenceRunSnapshot.SCHEMA_VERSION,
                InferenceRunSnapshot.Tuning.of(new InferenceTuning(ProcessorTopology.bits(9, 1, 8, 0), 256)),
                List.of(0, 1),
                new InferenceRunSnapshot.Model(
                        "/models/model.edrl", 12859040768L, 2, InferenceRunSnapshot.Dimensions.of(config(layers))),
                generation,
                new InferenceRunSnapshot.RuntimeIdentity(
                        "25.0.2+10",
                        "Eclipse Adoptium",
                        "OpenJDK 64-Bit Server VM",
                        "Linux",
                        "amd64",
                        null,
                        "euhedral-core-0.0.7.jar",
                        "/opt/euhedral/lib/libeuhedral_cuda.so",
                        InferenceRunSnapshot.UNAVAILABLE));
    }

    @Test
    void serializesAStableJsonContract() {
        String expected = "{\"schemaVersion\":2,"
                + "\"tuning\":{\"workerProcessorIds\":[0,1,8,9],\"prefillChunkTokens\":256,\"q3DispatchMode\":\"AUTO\",\"q3SmallRowThreshold\":8},"
                + "\"workerCoreIds\":[0,1],"
                + "\"model\":{\"artifactPath\":\"/models/model.edrl\",\"artifactBytes\":12859040768,"
                + "\"artifactFormatVersion\":2,\"dimensions\":{\"vocabSize\":248320,\"hiddenSize\":5120,"
                + "\"numHiddenLayers\":4,\"fullAttentionLayers\":1,\"gatedDeltaNetLayers\":3,"
                + "\"numAttentionHeads\":24,\"numKeyValueHeads\":4,\"attentionHeadDim\":256,"
                + "\"intermediateSize\":17408,\"linearNumKeyHeads\":16,\"linearNumValueHeads\":48,"
                + "\"linearKeyHeadDim\":128,\"linearValueHeadDim\":128,\"maxPositionEmbeddings\":262144}},"
                + "\"generation\":{\"temperature\":0.7,\"topK\":20,\"topP\":0.8,\"seed\":42,\"greedy\":false},"
                + "\"runtime\":{\"javaVersion\":\"25.0.2+10\",\"javaVendor\":\"Eclipse Adoptium\","
                + "\"javaVmName\":\"OpenJDK 64-Bit Server VM\",\"osName\":\"Linux\",\"osArch\":\"amd64\","
                + "\"euhedralCoreVersion\":\"unavailable\",\"euhedralCoreArtifact\":\"euhedral-core-0.0.7.jar\","
                + "\"nativeLibraryPath\":\"/opt/euhedral/lib/libeuhedral_cuda.so\","
                + "\"nativeRuntimeVersion\":\"unavailable\"}}";
        var generation = new GenerationConfig(0.7f, 20, 0.8f, 42L, false);
        assertEquals(expected, snapshot(generation).toJson());
        assertEquals(expected, snapshot(generation).toJson(), "serialization must be repeatable");
    }

    @Test
    void absentGenerationIsAnExplicitNull() {
        String json = snapshot(null).toJson();
        assertEquals(1, json.split("\"generation\":null,", -1).length - 1, json);
    }

    @Test
    void projectsQwenConfigWithoutRetainingItsMutableLayerArray() {
        QwenLayerType[] layers = {QwenLayerType.FULL_ATTENTION, QwenLayerType.GATED_DELTA_NET};
        var dimensions = InferenceRunSnapshot.Dimensions.of(config(layers));
        layers[1] = QwenLayerType.FULL_ATTENTION;
        assertEquals(1, dimensions.fullAttentionLayers());
        assertEquals(1, dimensions.gatedDeltaNetLayers());
    }

    @Test
    void listsAreImmutableCopies() {
        List<Integer> cores = new ArrayList<>(List.of(0, 1));
        var original = snapshot(null);
        var copy = new InferenceRunSnapshot(
                original.schemaVersion(),
                original.tuning(),
                cores,
                original.model(),
                original.generation(),
                original.runtime());
        cores.add(2);
        assertEquals(List.of(0, 1), copy.workerCoreIds());
        assertThrows(
                UnsupportedOperationException.class, () -> copy.workerCoreIds().add(3));
        assertThrows(
                UnsupportedOperationException.class,
                () -> copy.tuning().workerProcessorIds().add(3));
    }
}
