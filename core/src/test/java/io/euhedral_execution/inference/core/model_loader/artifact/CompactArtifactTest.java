package io.euhedral_execution.inference.core.model_loader.artifact;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.euhedral_execution.inference.core.model_loader.config.QwenConfig;
import io.euhedral_execution.inference.core.model_loader.config.QwenLayerType;
import io.euhedral_execution.inference.core.model_loader.layer_weights.TensorDataType;
import io.euhedral_execution.inference.core.model_loader.layer_weights.WeightFormat;
import io.euhedral_execution.inference.core.model_loader.layer_weights.WeightLayout;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class CompactArtifactTest {

    @TempDir
    Path tempDirectory;

    @Test
    void q3GeometryAndCompactDescriptorRoundTrip() throws Exception {
        long byteSize = CompactTensorLayout.expectedByteSize(
                new long[] {64, 64}, TensorDataType.BF16, WeightFormat.Q3_G64_FP16, WeightLayout.ROW_SPLIT_K128_V1);
        assertEquals(3328, byteSize);
        TensorDescriptor descriptor = new TensorDescriptor(
                "text/test_fused",
                new long[] {64, 64},
                TensorDataType.BF16,
                WeightFormat.Q3_G64_FP16,
                WeightLayout.ROW_SPLIT_K128_V1,
                0,
                byteSize);
        QwenConfig config = emptyConfig();
        long metadataSize = QwenCompactArtifactWriter.metadataSize(config);
        long tableOffset = QwenArtifactHeader.BYTE_SIZE + metadataSize;
        long tableSize = QwenArtifactCodec.encodeTensorTable(new TensorDescriptor[] {descriptor}).length;
        long dataOffset = tableOffset + tableSize;
        descriptor = new TensorDescriptor(
                descriptor.name(),
                descriptor.shape(),
                descriptor.dataType(),
                descriptor.format(),
                descriptor.layout(),
                dataOffset,
                byteSize);
        Path path = tempDirectory.resolve("compact.edrl");
        QwenCompactArtifactWriter.write(
                path,
                new QwenArtifact(
                        new QwenArtifactHeader(
                                QwenArtifactHeader.MAGIC,
                                QwenArtifactHeader.COMPACT_VERSION,
                                QwenArtifactHeader.BYTE_SIZE,
                                metadataSize,
                                tableOffset,
                                1,
                                dataOffset),
                        config,
                        new TensorDescriptor[] {descriptor}),
                new byte[][] {new byte[(int) byteSize]});

        QwenArtifact decoded = QwenArtifactReader.read(path);
        assertEquals(QwenArtifactHeader.COMPACT_VERSION, decoded.header().version());
        assertEquals(WeightLayout.ROW_SPLIT_K128_V1, decoded.tensors()[0].layout());
        assertEquals(byteSize, decoded.tensors()[0].byteSize());
    }

    @Test
    void rejectsCompactDescriptorWithIncorrectPayloadSize() {
        TensorDescriptor malformed = new TensorDescriptor(
                "text/malformed",
                new long[] {64, 64},
                TensorDataType.BF16,
                WeightFormat.Q3_G64_FP16,
                WeightLayout.ROW_SPLIT_K128_V1,
                0,
                3327);
        assertThrows(
                QwenArtifactFormatException.class,
                () -> QwenArtifactCodec.encodeTensorTable(new TensorDescriptor[] {malformed}));
    }

    @Test
    void rejectsGroupedDescriptorWithConflictingSourceDtype() {
        assertThrows(
                IllegalArgumentException.class,
                () -> CompactTensorLayout.expectedByteSize(
                        new long[] {64, 64},
                        TensorDataType.INT32,
                        WeightFormat.Q3_G64_FP16,
                        WeightLayout.ROW_SPLIT_K128_V1));
    }

    private static QwenConfig emptyConfig() {
        return new QwenConfig(
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
                new QwenLayerType[0],
                0,
                0,
                0,
                0,
                false,
                false,
                0);
    }
}
