package io.euhedral_execution.inference.core.model_loader.artifact;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.euhedral_execution.inference.core.model_loader.layer_weights.TensorDataType;
import io.euhedral_execution.inference.core.model_loader.layer_weights.WeightFormat;
import io.euhedral_execution.inference.core.model_loader.layer_weights.WeightLayout;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class TensorDataReaderTest {

    @TempDir
    Path tempDirectory;

    @Test
    void readsExactPayloadIntoNativeMemoryFromAbsoluteOffset() throws Exception {
        Path path = tempDirectory.resolve("weights.edrl");
        int dataOffset = 7;
        byte[] payload = {(byte) 0x00, (byte) 0x7F, (byte) 0x80, (byte) 0xFF};
        byte[] file = new byte[dataOffset + payload.length];
        System.arraycopy(payload, 0, file, dataOffset, payload.length);
        Files.write(path, file);

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment segment = TensorDataReader.read(path, descriptor(dataOffset, payload.length), arena);

            assertEquals(payload.length, segment.byteSize());
            assertArrayEquals(payload, segment.toArray(ValueLayout.JAVA_BYTE));
        }
    }

    @Test
    void rejectsTruncatedPayload() throws Exception {
        Path path = tempDirectory.resolve("truncated.edrl");
        Files.write(path, new byte[10]);

        try (Arena arena = Arena.ofConfined()) {
            assertThrows(QwenArtifactFormatException.class, () -> TensorDataReader.read(path, descriptor(7, 4), arena));
        }
    }

    @Test
    void readsZeroLengthTensorAsEmptyNativeSegment() throws Exception {
        Path path = tempDirectory.resolve("empty.edrl");
        Files.write(path, new byte[8]);

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment segment = TensorDataReader.read(path, descriptor(8, 0), arena);

            assertEquals(0, segment.byteSize());
        }
    }

    @Test
    void rejectsNullInputs() throws Exception {
        Path path = tempDirectory.resolve("null-inputs.edrl");
        Files.write(path, new byte[1]);
        TensorDescriptor descriptor = descriptor(0, 0);

        try (Arena arena = Arena.ofConfined()) {
            assertThrows(QwenArtifactFormatException.class, () -> TensorDataReader.read(null, descriptor, arena));
            assertThrows(QwenArtifactFormatException.class, () -> TensorDataReader.read(path, null, arena));
            assertThrows(QwenArtifactFormatException.class, () -> TensorDataReader.read(path, descriptor, null));
        }
    }

    @Test
    void rejectsNegativeOffsetAndSize() throws Exception {
        Path path = tempDirectory.resolve("invalid-range.edrl");
        Files.write(path, new byte[1]);

        try (Arena arena = Arena.ofConfined()) {
            assertThrows(
                    QwenArtifactFormatException.class, () -> TensorDataReader.read(path, descriptor(-1, 0), arena));
            assertThrows(
                    QwenArtifactFormatException.class, () -> TensorDataReader.read(path, descriptor(0, -1), arena));
        }
    }

    @Test
    void rejectsOverflowingRange() throws Exception {
        Path path = tempDirectory.resolve("overflowing-range.edrl");
        Files.write(path, new byte[1]);

        try (Arena arena = Arena.ofConfined()) {
            assertThrows(
                    QwenArtifactFormatException.class,
                    () -> TensorDataReader.read(path, descriptor(Long.MAX_VALUE, 1), arena));
        }
    }

    @Test
    void rejectsTruncatedOversizedRange() throws Exception {
        Path path = tempDirectory.resolve("oversized.edrl");
        Files.write(path, new byte[1]);

        try (Arena arena = Arena.ofConfined()) {
            QwenArtifactFormatException exception = assertThrows(
                    QwenArtifactFormatException.class,
                    () -> TensorDataReader.read(path, descriptor(0, (long) Integer.MAX_VALUE + 1), arena));
            assertEquals("tensor payload extends beyond the file: test.tensor", exception.getMessage());
        }
    }

    private static TensorDescriptor descriptor(long dataOffset, long byteSize) {
        return new TensorDescriptor(
                "test.tensor",
                new long[] {4},
                TensorDataType.UINT8,
                WeightFormat.Q4_G64_FP16,
                WeightLayout.ROW_SPLIT_K128_V1,
                dataOffset,
                byteSize);
    }
}
