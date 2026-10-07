package io.euhedral_execution.inference.core.model.qwen38.artifact;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.euhedral_execution.inference.core.artifact.ArtifactFormatException;
import io.euhedral_execution.inference.core.artifact.TensorDataType;
import io.euhedral_execution.inference.core.artifact.TensorDescriptor;
import io.euhedral_execution.inference.core.artifact.WeightFormat;
import io.euhedral_execution.inference.core.artifact.WeightLayout;
import io.euhedral_execution.inference.core.model.qwen38.LayerType;
import io.euhedral_execution.inference.core.model.qwen38.Qwen38Config;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class ArtifactBinaryTest {

    private static final byte[][] PAYLOAD = {{1, 2, 3, 4}, {5, 6, 7, 8}};

    @TempDir
    Path tempDirectory;

    @Test
    void writesAndReadsDeterministicArtifact() throws Exception {
        Qwen38Config config = sampleConfig();
        TensorDescriptor[] tensors = descriptors(config);
        Artifact artifact = artifact(config, tensors);
        Path firstPath = tempDirectory.resolve("first.edrl");
        Path secondPath = tempDirectory.resolve("second.edrl");

        CompactArtifactWriter.write(firstPath, artifact, PAYLOAD);
        CompactArtifactWriter.write(secondPath, artifact, PAYLOAD);

        Artifact read = ArtifactReader.read(firstPath);
        assertEquals(artifact.header(), read.header());
        assertConfigEquals(artifact.config(), read.config());
        assertTensorDescriptorsEqual(artifact.tensors(), read.tensors());
        byte[] firstBytes = Files.readAllBytes(firstPath);
        assertArrayEquals(firstBytes, Files.readAllBytes(secondPath));
        assertArrayEquals(
                new byte[] {1, 2, 3, 4},
                java.util.Arrays.copyOfRange(
                        firstBytes, (int) tensors[0].dataOffset(), (int) tensors[0].dataOffset() + 4));
    }

    @Test
    void rejectsInvalidMagic() throws Exception {
        Artifact artifact = artifact(sampleConfig(), descriptors(sampleConfig()));
        Path path = tempDirectory.resolve("invalid-magic.edrl");
        CompactArtifactWriter.write(path, artifact, PAYLOAD);

        byte[] bytes = Files.readAllBytes(path);
        ByteBuffer.wrap(bytes).order(ByteOrder.BIG_ENDIAN).putInt(0, 0);
        Files.write(path, bytes);

        assertThrows(ArtifactFormatException.class, () -> ArtifactReader.read(path));
    }

    @Test
    void rejectsTruncatedArtifact() throws Exception {
        Artifact artifact = artifact(sampleConfig(), descriptors(sampleConfig()));
        Path path = tempDirectory.resolve("truncated.edrl");
        CompactArtifactWriter.write(path, artifact, PAYLOAD);

        byte[] bytes = Files.readAllBytes(path);
        Files.write(path, java.util.Arrays.copyOf(bytes, bytes.length - 1));

        assertThrows(ArtifactFormatException.class, () -> ArtifactReader.read(path));
    }

    @Test
    void rejectsTensorDataOutsideFileBounds() throws Exception {
        Qwen38Config config = sampleConfig();
        TensorDescriptor[] tensors = descriptors(config);
        Artifact artifact = artifact(config, tensors);
        Path path = tempDirectory.resolve("invalid-offset.edrl");
        CompactArtifactWriter.write(path, artifact, PAYLOAD);

        byte[] bytes = Files.readAllBytes(path);
        int dataOffsetField = (int) tensorsTableOffset(artifact) + tensorDataOffsetWithinEntry(tensors[0]);
        ByteBuffer.wrap(bytes).order(ByteOrder.BIG_ENDIAN).putLong(dataOffsetField, Long.MAX_VALUE);
        Files.write(path, bytes);

        assertThrows(ArtifactFormatException.class, () -> ArtifactReader.read(path));
    }

    @Test
    void rejectsTensorCountThatDoesNotMatchTable() throws Exception {
        Artifact artifact = artifact(sampleConfig(), descriptors(sampleConfig()));
        Path path = tempDirectory.resolve("invalid-count.edrl");
        CompactArtifactWriter.write(path, artifact, PAYLOAD);

        byte[] bytes = Files.readAllBytes(path);
        ByteBuffer.wrap(bytes).order(ByteOrder.BIG_ENDIAN).putInt(32, 3);
        Files.write(path, bytes);

        assertThrows(ArtifactFormatException.class, () -> ArtifactReader.read(path));
    }

    @Test
    void rejectsLegacyAndUnknownVersions() throws Exception {
        Artifact artifact = artifact(sampleConfig(), descriptors(sampleConfig()));
        Path path = tempDirectory.resolve("invalid-version.edrl");
        CompactArtifactWriter.write(path, artifact, PAYLOAD);
        byte[] original = Files.readAllBytes(path);

        for (int version : new int[] {1, ArtifactHeader.COMPACT_VERSION + 1}) {
            byte[] bytes = original.clone();
            ByteBuffer.wrap(bytes).order(ByteOrder.BIG_ENDIAN).putInt(4, version);
            Files.write(path, bytes);

            ArtifactFormatException failure =
                    assertThrows(ArtifactFormatException.class, () -> ArtifactReader.read(path));
            assertEquals(
                    "unsupported artifact version " + version + ": only compact version "
                            + ArtifactHeader.COMPACT_VERSION + " artifacts are supported",
                    failure.getMessage());
        }
    }

    @Test
    void writerRejectsNonCompactHeader() {
        Artifact compact = artifact(sampleConfig(), descriptors(sampleConfig()));
        ArtifactHeader header = compact.header();
        Artifact legacy = new Artifact(
                new ArtifactHeader(
                        header.magic(),
                        1,
                        header.metadataOffset(),
                        header.metadataSize(),
                        header.tensorTableOffset(),
                        header.tensorCount(),
                        header.tensorDataOffset()),
                compact.config(),
                compact.tensors());

        assertThrows(
                ArtifactFormatException.class,
                () -> CompactArtifactWriter.write(tempDirectory.resolve("legacy.edrl"), legacy, PAYLOAD));
    }

    @Test
    void rejectsNonZeroReservedHeaderField() throws Exception {
        Artifact artifact = artifact(sampleConfig(), descriptors(sampleConfig()));
        Path path = tempDirectory.resolve("invalid-reserved.edrl");
        CompactArtifactWriter.write(path, artifact, PAYLOAD);

        byte[] bytes = Files.readAllBytes(path);
        ByteBuffer.wrap(bytes).order(ByteOrder.BIG_ENDIAN).putInt(36, 1);
        Files.write(path, bytes);

        assertThrows(ArtifactFormatException.class, () -> ArtifactReader.read(path));
    }

    @Test
    void rejectsInvalidTensorDataType() throws Exception {
        Qwen38Config config = sampleConfig();
        TensorDescriptor[] tensors = descriptors(config);
        Artifact artifact = artifact(config, tensors);
        Path path = tempDirectory.resolve("invalid-data-type.edrl");
        CompactArtifactWriter.write(path, artifact, PAYLOAD);

        byte[] bytes = Files.readAllBytes(path);
        int dataTypeField =
                (int) tensorsTableOffset(artifact) + tensorDataOffsetWithinEntry(tensors[0]) - Integer.BYTES * 3;
        ByteBuffer.wrap(bytes).order(ByteOrder.BIG_ENDIAN).putInt(dataTypeField, Integer.MAX_VALUE);
        Files.write(path, bytes);

        assertThrows(ArtifactFormatException.class, () -> ArtifactReader.read(path));
    }

    @Test
    void rejectsInvalidTensorLayout() throws Exception {
        Qwen38Config config = sampleConfig();
        TensorDescriptor[] tensors = descriptors(config);
        Artifact artifact = artifact(config, tensors);
        Path path = tempDirectory.resolve("invalid-layout.edrl");
        CompactArtifactWriter.write(path, artifact, PAYLOAD);

        byte[] bytes = Files.readAllBytes(path);
        int layoutField = (int) tensorsTableOffset(artifact) + tensorDataOffsetWithinEntry(tensors[0]) - Integer.BYTES;
        ByteBuffer.wrap(bytes).order(ByteOrder.BIG_ENDIAN).putInt(layoutField, Integer.MAX_VALUE);
        Files.write(path, bytes);

        assertThrows(ArtifactFormatException.class, () -> ArtifactReader.read(path));
    }

    @Test
    void rejectsTensorByteSizeThatDoesNotMatchItsShape() throws Exception {
        Qwen38Config config = sampleConfig();
        TensorDescriptor[] tensors = descriptors(config);
        Artifact artifact = artifact(config, tensors);
        Path path = tempDirectory.resolve("mismatched-byte-size.edrl");
        CompactArtifactWriter.write(path, artifact, PAYLOAD);

        byte[] bytes = Files.readAllBytes(path);
        int byteSizeField = (int) tensorsTableOffset(artifact) + tensorDataOffsetWithinEntry(tensors[0]) + Long.BYTES;
        ByteBuffer.wrap(bytes).order(ByteOrder.BIG_ENDIAN).putLong(byteSizeField, 3);
        Files.write(path, bytes);

        assertThrows(ArtifactFormatException.class, () -> ArtifactReader.read(path));
    }

    @Test
    void rejectsInvalidTensorNameLength() throws Exception {
        Artifact artifact = artifact(sampleConfig(), descriptors(sampleConfig()));
        Path path = tempDirectory.resolve("invalid-name-length.edrl");
        CompactArtifactWriter.write(path, artifact, PAYLOAD);

        byte[] bytes = Files.readAllBytes(path);
        ByteBuffer.wrap(bytes)
                .order(ByteOrder.BIG_ENDIAN)
                .putInt((int) artifact.header().tensorTableOffset(), -1);
        Files.write(path, bytes);

        assertThrows(ArtifactFormatException.class, () -> ArtifactReader.read(path));
    }

    @Test
    void rejectsNegativeTensorByteSize() throws Exception {
        Qwen38Config config = sampleConfig();
        TensorDescriptor[] tensors = descriptors(config);
        Artifact artifact = artifact(config, tensors);
        Path path = tempDirectory.resolve("invalid-byte-size.edrl");
        CompactArtifactWriter.write(path, artifact, PAYLOAD);

        byte[] bytes = Files.readAllBytes(path);
        int byteSizeField = (int) tensorsTableOffset(artifact) + tensorDataOffsetWithinEntry(tensors[0]) + Long.BYTES;
        ByteBuffer.wrap(bytes).order(ByteOrder.BIG_ENDIAN).putLong(byteSizeField, -1);
        Files.write(path, bytes);

        assertThrows(ArtifactFormatException.class, () -> ArtifactReader.read(path));
    }

    @Test
    void rejectsMalformedMetadataUtf8() throws Exception {
        Qwen38Config config = sampleConfig();
        TensorDescriptor[] tensors = descriptors(config);
        Artifact artifact = artifact(config, tensors);
        Path path = tempDirectory.resolve("invalid-utf8.edrl");
        CompactArtifactWriter.write(path, artifact, PAYLOAD);

        byte[] bytes = Files.readAllBytes(path);
        int activationOffset = (int) artifact.header().metadataOffset()
                + Integer.BYTES * 12
                + Double.BYTES * 3
                + Integer.BYTES
                + Integer.BYTES;
        bytes[activationOffset] = (byte) 0xC3;
        Files.write(path, bytes);

        assertThrows(ArtifactFormatException.class, () -> ArtifactReader.read(path));
    }

    @Test
    void rejectsInvalidMetadataBoolean() throws Exception {
        Qwen38Config config = sampleConfig();
        TensorDescriptor[] tensors = descriptors(config);
        Artifact artifact = artifact(config, tensors);
        Path path = tempDirectory.resolve("invalid-boolean.edrl");
        CompactArtifactWriter.write(path, artifact, PAYLOAD);

        byte[] bytes = Files.readAllBytes(path);
        int booleanOffset = (int) artifact.header().metadataOffset()
                + Integer.BYTES * 12
                + Double.BYTES * 3
                + Integer.BYTES
                + Integer.BYTES
                + config.hiddenActivation().getBytes(java.nio.charset.StandardCharsets.UTF_8).length
                + Integer.BYTES
                + Integer.BYTES * config.layerTypes().length
                + Integer.BYTES * 4;
        bytes[booleanOffset] = 2;
        Files.write(path, bytes);

        assertThrows(ArtifactFormatException.class, () -> ArtifactReader.read(path));
    }

    @Test
    void rejectsOverlappingTensorPayloads() throws Exception {
        Qwen38Config config = sampleConfig();
        Artifact validArtifact = artifact(config, descriptors(config));
        long dataOffset = validArtifact.header().tensorDataOffset();
        TensorDescriptor[] tensors = {
            descriptor("text/test_a", new long[] {1}, TensorDataType.FP32, WeightFormat.FP32, dataOffset),
            descriptor("text/test_b", new long[] {2}, TensorDataType.BF16, WeightFormat.BF16, dataOffset)
        };
        Artifact artifact = new Artifact(validArtifact.header(), config, tensors);
        Path path = tempDirectory.resolve("overlapping-data.edrl");

        assertThrows(
                ArtifactFormatException.class,
                () -> CompactArtifactWriter.write(path, artifact, new byte[][] {{1, 2, 3, 4}, {5, 6, 7, 8}}));
    }

    @Test
    void rejectsOverlappingTensorPayloadsWhenReading() throws Exception {
        Qwen38Config config = sampleConfig();
        TensorDescriptor[] tensors = descriptors(config);
        Artifact artifact = artifact(config, tensors);
        Path path = tempDirectory.resolve("overlapping-data-on-read.edrl");
        CompactArtifactWriter.write(path, artifact, PAYLOAD);

        byte[] bytes = Files.readAllBytes(path);
        int secondDataOffsetField = (int) artifact.header().tensorTableOffset()
                + tensorEntrySize(tensors[0])
                + tensorDataOffsetWithinEntry(tensors[1]);
        ByteBuffer.wrap(bytes)
                .order(ByteOrder.BIG_ENDIAN)
                .putLong(secondDataOffsetField, tensors[0].dataOffset())
                .putLong(secondDataOffsetField + Long.BYTES, tensors[0].byteSize());
        Files.write(path, bytes);

        assertThrows(ArtifactFormatException.class, () -> ArtifactReader.read(path));
    }

    @Test
    void rejectsMetadataLargerThanReaderLimitBeforeDecoding() throws Exception {
        long metadataSize = ArtifactCodec.MAX_METADATA_BYTES + 1;
        long tableOffset = ArtifactHeader.BYTE_SIZE + metadataSize;
        Path path = tempDirectory.resolve("oversized-metadata.edrl");
        ByteBuffer header = ByteBuffer.allocate(ArtifactHeader.BYTE_SIZE).order(ByteOrder.BIG_ENDIAN);
        header.putInt(ArtifactHeader.MAGIC);
        header.putInt(ArtifactHeader.COMPACT_VERSION);
        header.putLong(ArtifactHeader.BYTE_SIZE);
        header.putLong(metadataSize);
        header.putLong(tableOffset);
        header.putInt(0);
        header.putInt(0);
        header.putLong(tableOffset);
        header.flip();
        try (FileChannel channel = FileChannel.open(
                path, StandardOpenOption.CREATE, StandardOpenOption.WRITE, StandardOpenOption.TRUNCATE_EXISTING)) {
            channel.write(header);
            channel.position(tableOffset - 1);
            channel.write(ByteBuffer.wrap(new byte[] {0}));
        }

        ArtifactFormatException exception =
                assertThrows(ArtifactFormatException.class, () -> ArtifactReader.read(path));
        assertEquals("metadata is too large", exception.getMessage());
    }

    private static Qwen38Config sampleConfig() {
        return new Qwen38Config(
                32_000,
                4_096,
                2,
                32,
                8,
                128,
                11_008,
                16,
                8,
                64,
                128,
                4,
                1.0e-6,
                1_000_000.0,
                1.0,
                32_768,
                "silu",
                new LayerType[] {LayerType.FULL_ATTENTION, LayerType.GATED_DELTA_NET},
                0,
                0,
                0,
                0,
                true,
                false,
                0);
    }

    private static TensorDescriptor descriptor(
            String name, long[] shape, TensorDataType dataType, WeightFormat format, long dataOffset) {
        return new TensorDescriptor(name, shape, dataType, format, WeightLayout.CONTIGUOUS_LE_V1, dataOffset, 4);
    }

    private static TensorDescriptor[] descriptors(Qwen38Config config) {
        long tableOffset = ArtifactHeader.BYTE_SIZE + CompactArtifactWriter.metadataSize(config);
        long tableSize = CompactArtifactWriter.tensorTableSize(new TensorDescriptor[] {
            descriptor("text/test_a", new long[] {1}, TensorDataType.FP32, WeightFormat.FP32, 0),
            descriptor("text/test_b", new long[] {2}, TensorDataType.BF16, WeightFormat.BF16, 4)
        });
        long dataOffset = tableOffset + tableSize;
        return new TensorDescriptor[] {
            descriptor("text/test_a", new long[] {1}, TensorDataType.FP32, WeightFormat.FP32, dataOffset),
            descriptor("text/test_b", new long[] {2}, TensorDataType.BF16, WeightFormat.BF16, dataOffset + 4)
        };
    }

    private static Artifact artifact(Qwen38Config config, TensorDescriptor[] tensors) {
        long metadataOffset = ArtifactHeader.BYTE_SIZE;
        long metadataSize = CompactArtifactWriter.metadataSize(config);
        long tableOffset = metadataOffset + metadataSize;
        long tensorDataOffset = tableOffset + CompactArtifactWriter.tensorTableSize(tensors);
        return new Artifact(
                new ArtifactHeader(
                        ArtifactHeader.MAGIC,
                        ArtifactHeader.COMPACT_VERSION,
                        metadataOffset,
                        metadataSize,
                        tableOffset,
                        tensors.length,
                        tensorDataOffset),
                config,
                tensors);
    }

    private static long tensorsTableOffset(Artifact artifact) {
        return artifact.header().tensorTableOffset();
    }

    private static int tensorDataOffsetWithinEntry(TensorDescriptor descriptor) {
        int nameLength = descriptor.name().getBytes(java.nio.charset.StandardCharsets.UTF_8).length;
        return Integer.BYTES + nameLength + Integer.BYTES + Long.BYTES * descriptor.shape().length + Integer.BYTES * 3;
    }

    private static int tensorEntrySize(TensorDescriptor descriptor) {
        return tensorDataOffsetWithinEntry(descriptor) + Long.BYTES * 2;
    }

    private static void assertConfigEquals(Qwen38Config expected, Qwen38Config actual) {
        assertEquals(expected.vocabSize(), actual.vocabSize());
        assertEquals(expected.hiddenSize(), actual.hiddenSize());
        assertEquals(expected.numHiddenLayers(), actual.numHiddenLayers());
        assertEquals(expected.numAttentionHeads(), actual.numAttentionHeads());
        assertEquals(expected.numKeyValueHeads(), actual.numKeyValueHeads());
        assertEquals(expected.attentionHeadDim(), actual.attentionHeadDim());
        assertEquals(expected.intermediateSize(), actual.intermediateSize());
        assertEquals(expected.linearNumKeyHeads(), actual.linearNumKeyHeads());
        assertEquals(expected.linearNumValueHeads(), actual.linearNumValueHeads());
        assertEquals(expected.linearKeyHeadDim(), actual.linearKeyHeadDim());
        assertEquals(expected.linearValueHeadDim(), actual.linearValueHeadDim());
        assertEquals(expected.linearConvKernelDim(), actual.linearConvKernelDim());
        assertEquals(expected.rmsNormEpsilon(), actual.rmsNormEpsilon());
        assertEquals(expected.ropeTheta(), actual.ropeTheta());
        assertEquals(expected.partialRotaryFactor(), actual.partialRotaryFactor());
        assertEquals(expected.maxPositionEmbeddings(), actual.maxPositionEmbeddings());
        assertEquals(expected.hiddenActivation(), actual.hiddenActivation());
        assertArrayEquals(expected.layerTypes(), actual.layerTypes());
        assertEquals(expected.numExperts(), actual.numExperts());
        assertEquals(expected.numExpertsPerToken(), actual.numExpertsPerToken());
        assertEquals(expected.moeIntermediateSize(), actual.moeIntermediateSize());
        assertEquals(expected.sharedExpertIntermediateSize(), actual.sharedExpertIntermediateSize());
        assertEquals(expected.tieWordEmbeddings(), actual.tieWordEmbeddings());
        assertEquals(expected.attentionOutputGate(), actual.attentionOutputGate());
        assertEquals(expected.mtpLayerCount(), actual.mtpLayerCount());
    }

    private static void assertTensorDescriptorsEqual(TensorDescriptor[] expected, TensorDescriptor[] actual) {
        assertEquals(expected.length, actual.length);
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i].name(), actual[i].name());
            assertArrayEquals(expected[i].shape(), actual[i].shape());
            assertEquals(expected[i].dataType(), actual[i].dataType());
            assertEquals(expected[i].format(), actual[i].format());
            assertEquals(expected[i].layout(), actual[i].layout());
            assertEquals(expected[i].dataOffset(), actual[i].dataOffset());
            assertEquals(expected[i].byteSize(), actual[i].byteSize());
        }
    }
}
