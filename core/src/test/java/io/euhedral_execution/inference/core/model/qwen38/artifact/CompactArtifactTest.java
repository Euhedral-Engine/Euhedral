package io.euhedral_execution.inference.core.model.qwen38.artifact;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.euhedral_execution.inference.core.artifact.ArtifactFormatException;
import io.euhedral_execution.inference.core.artifact.CompactTensorLayout;
import io.euhedral_execution.inference.core.artifact.TensorDataType;
import io.euhedral_execution.inference.core.artifact.TensorDescriptor;
import io.euhedral_execution.inference.core.artifact.WeightFormat;
import io.euhedral_execution.inference.core.artifact.WeightLayout;
import io.euhedral_execution.inference.core.model.qwen38.LayerType;
import io.euhedral_execution.inference.core.model.qwen38.Qwen38Config;
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
        Qwen38Config config = emptyConfig();
        long metadataSize = CompactArtifactWriter.metadataSize(config);
        long tableOffset = ArtifactHeader.BYTE_SIZE + metadataSize;
        long tableSize = ArtifactCodec.encodeTensorTable(new TensorDescriptor[] {descriptor}).length;
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
        CompactArtifactWriter.write(
                path,
                new Artifact(
                        new ArtifactHeader(
                                ArtifactHeader.MAGIC,
                                ArtifactHeader.COMPACT_VERSION,
                                ArtifactHeader.BYTE_SIZE,
                                metadataSize,
                                tableOffset,
                                1,
                                dataOffset),
                        config,
                        new TensorDescriptor[] {descriptor}),
                new byte[][] {new byte[(int) byteSize]});

        Artifact decoded = ArtifactReader.read(path);
        assertEquals(ArtifactHeader.COMPACT_VERSION, decoded.header().version());
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
                ArtifactFormatException.class,
                () -> ArtifactCodec.encodeTensorTable(new TensorDescriptor[] {malformed}));
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

    private static Qwen38Config emptyConfig() {
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
