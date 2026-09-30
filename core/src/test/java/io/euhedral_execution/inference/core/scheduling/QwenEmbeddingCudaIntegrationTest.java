package io.euhedral_execution.inference.core.scheduling;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import io.euhedral_execution.inference.core.model_loader.QwenModel;
import io.euhedral_execution.inference.core.model_loader.QwenWeights;
import io.euhedral_execution.inference.core.model_loader.artifact.CompactTensorLayout;
import io.euhedral_execution.inference.core.model_loader.artifact.QwenArtifact;
import io.euhedral_execution.inference.core.model_loader.artifact.QwenArtifactReader;
import io.euhedral_execution.inference.core.model_loader.artifact.TensorDescriptor;
import io.euhedral_execution.inference.core.model_loader.layer_weights.TensorDataType;
import io.euhedral_execution.inference.core.model_loader.layer_weights.WeightFormat;
import io.euhedral_execution.inference.core.model_loader.layer_weights.WeightLayout;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;

class QwenEmbeddingCudaIntegrationTest {

    private static final int[] MODEL_TOKEN_IDS = {0, 1, 2, 42, 1234, 8192, 248319};
    private static final float MODEL_MAX_ABSOLUTE_TOLERANCE = 0.06f;
    private static final float MODEL_RMS_TOLERANCE = 0.01f;
    private static final float Q3_ABSOLUTE_TOLERANCE = 1.0f / 128.0f;

    @Test
    void q3EmbeddingKernelMatchesKnownCpuReference() throws Exception {
        Path libraryPath = nativeLibraryPath();
        int vocabularySize = 17;
        int hiddenSize = 5_120;
        int[] tokenIds = {16, 5, 1, 5};
        byte[] packedWeights = syntheticQ3Weights(vocabularySize, hiddenSize);
        long weightBytes = packedWeights.length;
        long tokenBytes = (long) tokenIds.length * Integer.BYTES;
        long outputBytes = (long) tokenIds.length * hiddenSize * Short.BYTES;
        float[] expected = decodeSyntheticReference(packedWeights, vocabularySize, tokenIds, hiddenSize);

        try (CudaGpuMemory gpu = new CudaGpuMemory(libraryPath);
                Arena hostArena = Arena.ofConfined()) {
            long deviceWeights = gpu.allocate(weightBytes);
            long deviceTokenIds = 0;
            long deviceOutput = 0;
            try {
                deviceTokenIds = gpu.allocate(tokenBytes);
                deviceOutput = gpu.allocate(outputBytes);
                MemorySegment hostWeights = hostArena.allocate(weightBytes, 1);
                MemorySegment hostTokenIds = hostArena.allocate(tokenBytes, Integer.BYTES);
                MemorySegment hostOutput = hostArena.allocate(outputBytes, Short.BYTES);
                for (int index = 0; index < packedWeights.length; index++) {
                    hostWeights.set(ValueLayout.JAVA_BYTE, index, packedWeights[index]);
                }
                for (int index = 0; index < tokenIds.length; index++) {
                    hostTokenIds.set(ValueLayout.JAVA_INT, (long) index * Integer.BYTES, tokenIds[index]);
                }
                gpu.copyHostToDevice(deviceWeights, hostWeights, weightBytes);
                gpu.copyHostToDevice(deviceTokenIds, hostTokenIds, tokenBytes);

                gpu.embedQ3(
                        deviceTokenIds,
                        deviceWeights,
                        weightBytes,
                        deviceOutput,
                        tokenIds.length,
                        vocabularySize,
                        hiddenSize);
                gpu.synchronize();
                gpu.copyDeviceToHost(hostOutput, deviceOutput, outputBytes);

                assertCloseBfloat16(hostOutput, expected, Q3_ABSOLUTE_TOLERANCE);
            } finally {
                gpu.free(deviceOutput);
                gpu.free(deviceTokenIds);
                gpu.free(deviceWeights);
            }
        }
    }

    @Test
    void compactModelEmbeddingStageMatchesSourceBfloat16Reference() throws Exception {
        Path libraryPath = nativeLibraryPath();
        Path compactPath = Path.of(System.getProperty("euhedral.qwen.artifact"));
        Path referencePath = Path.of(System.getProperty("euhedral.qwen.reference-artifact"));
        assertTrue(Files.isRegularFile(compactPath), "compact Qwen artifact is missing: " + compactPath);
        assertTrue(Files.isRegularFile(referencePath), "BF16 reference artifact is missing: " + referencePath);

        QwenArtifact compactArtifact = QwenArtifactReader.read(compactPath);
        QwenArtifact referenceArtifact = QwenArtifactReader.read(referencePath);
        TensorDescriptor referenceEmbedding = findBfloat16Embedding(referenceArtifact);
        assertTrue(compactArtifact.tensors().length > 0, "compact artifact has no persistent objects");

        long expectedWeightBytes = Arrays.stream(compactArtifact.tensors())
                .mapToLong(TensorDescriptor::byteSize)
                .sum();
        int hiddenSize = compactArtifact.config().hiddenSize();
        int vocabularySize = compactArtifact.config().vocabSize();
        assertEquals(vocabularySize, Math.toIntExact(referenceEmbedding.shape()[0]));
        assertEquals(hiddenSize, Math.toIntExact(referenceEmbedding.shape()[1]));

        float[] reference = readBfloat16Rows(referencePath, referenceEmbedding, MODEL_TOKEN_IDS, hiddenSize);
        AtomicReference<float[]> output = new AtomicReference<>();

        try (CudaGpuMemory gpu = new CudaGpuMemory(libraryPath)) {
            CudaGpuMemory.DeviceMemoryInfo beforeLoad = gpu.deviceMemoryInfo();
            assertTrue(
                    beforeLoad.freeBytes() >= expectedWeightBytes + (64L << 20),
                    "insufficient free VRAM for compact model: free=" + beforeLoad.freeBytes() + ", model="
                            + expectedWeightBytes);

            try (QwenModel model = QwenModel.load(compactPath, compactArtifact, gpu)) {
                QwenWeights weights = model.weights();
                assertEquals(
                        compactArtifact.tensors().length,
                        weights.runtimeObjects().size());
                assertEquals(64, weights.layers().length);
                CudaGpuMemory.DeviceMemoryInfo resident = gpu.deviceMemoryInfo();
                assertTrue(resident.freeBytes() > 0, "model weights did not fit on the CUDA device");

                QwenExecutionPlan plan = QwenExecutionPlan.embeddingOnly(weights);
                QwenSequenceState sequence = new QwenSequenceState(501L);
                QwenExecutionContext context = new QwenExecutionContext(
                        plan, sequence, QwenExecutionContext.ExecutionKind.PREFILL, 0L, MODEL_TOKEN_IDS);
                QwenExecutionContext.Outcome outcome = TestExecution.run(
                        plan,
                        gpu,
                        context,
                        terminalContext -> {
                            QwenExecutionWorkspace workspace = terminalContext.workspace();
                            try (Arena outputArena = Arena.ofConfined()) {
                                MemorySegment hostOutput = outputArena.allocate(workspace.byteSize(), Short.BYTES);
                                gpu.copyDeviceToHost(hostOutput, workspace.hiddenStateAddress(), workspace.byteSize());
                                output.set(decodeBfloat16(hostOutput));
                            }
                        },
                        600);
                assertEquals(QwenExecutionContext.Status.SUCCESS, outcome.status());

                assertTrue(context.workspace().isClosed(), "submission workspace survived terminal completion");
                assertEmbeddingCloseToSource(output.get(), reference);
                assertTrue(
                        gpu.deviceMemoryInfo().freeBytes() >= resident.freeBytes() - (16L << 20),
                        "submission completion changed model-weight residency");
            }

            assertTrue(
                    gpu.deviceMemoryInfo().freeBytes() >= beforeLoad.freeBytes() - (128L << 20),
                    "model teardown did not restore free VRAM near its baseline");
        }
    }

    private static byte[] syntheticQ3Weights(int vocabularySize, int hiddenSize) {
        long byteSize = CompactTensorLayout.expectedByteSize(
                new long[] {vocabularySize, hiddenSize},
                TensorDataType.BF16,
                WeightFormat.Q3_G64_FP16,
                WeightLayout.ROW_SPLIT_K128_V1);
        byte[] packed = new byte[Math.toIntExact(byteSize)];
        int groupsPerRow = ((hiddenSize + 127) / 128) * 2;
        int activeGroups = hiddenSize / 64;
        int baseBytes = vocabularySize * groupsPerRow * 24;
        int scaleOffset = (baseBytes + 255) & ~255;

        for (int row = 0; row < vocabularySize; row++) {
            for (int group = 0; group < activeGroups; group++) {
                int rowGroup = row * groupsPerRow + group;
                int codeGroupOffset = rowGroup * 24;
                for (int index = 0; index < 64; index++) {
                    writeThreeBitCode(packed, codeGroupOffset, index, (row + group + index) & 7);
                }
                int scale = scaleOffset + rowGroup * Short.BYTES;
                packed[scale] = 0;
                packed[scale + 1] = 0x38;
            }
        }
        return packed;
    }

    private static float[] decodeSyntheticReference(byte[] packed, int vocabularySize, int[] tokenIds, int hiddenSize) {
        int groupsPerRow = ((hiddenSize + 127) / 128) * 2;
        int baseBytes = vocabularySize * groupsPerRow * 24;
        int scaleOffset = (baseBytes + 255) & ~255;
        float[] expected = new float[tokenIds.length * hiddenSize];
        for (int tokenIndex = 0; tokenIndex < tokenIds.length; tokenIndex++) {
            int row = tokenIds[tokenIndex];
            for (int index = 0; index < hiddenSize; index++) {
                int group = index / 64;
                int laneIndex = index % 64;
                int rowGroup = row * groupsPerRow + group;
                int groupOffset = rowGroup * 24;
                int scale = scaleOffset + rowGroup * Short.BYTES;
                float multiplier = Float.float16ToFloat(
                        (short) (Byte.toUnsignedInt(packed[scale]) | (Byte.toUnsignedInt(packed[scale + 1]) << 8)));
                int code = readThreeBitCode(packed, groupOffset, laneIndex);
                expected[tokenIndex * hiddenSize + index] = (code < 4 ? code : code - 8) * multiplier;
            }
        }
        return expected;
    }

    private static void writeThreeBitCode(byte[] packed, int groupOffset, int index, int code) {
        int bitOffset = index * 3;
        int byteOffset = groupOffset + bitOffset / Byte.SIZE;
        int shift = bitOffset % Byte.SIZE;
        packed[byteOffset] |= (byte) (code << shift);
        if (shift > 5) {
            packed[byteOffset + 1] |= (byte) (code >>> (Byte.SIZE - shift));
        }
    }

    private static int readThreeBitCode(byte[] packed, int groupOffset, int index) {
        int bitOffset = index * 3;
        int byteOffset = groupOffset + bitOffset / Byte.SIZE;
        int shift = bitOffset % Byte.SIZE;
        int bits = Byte.toUnsignedInt(packed[byteOffset]);
        if (shift > 5) {
            bits |= Byte.toUnsignedInt(packed[byteOffset + 1]) << Byte.SIZE;
        }
        return (bits >>> shift) & 7;
    }

    private static void assertCloseBfloat16(MemorySegment actualBytes, float[] expected, float tolerance) {
        float[] actual = decodeBfloat16(actualBytes);
        assertEquals(expected.length, actual.length);
        for (int index = 0; index < expected.length; index++) {
            assertTrue(
                    Math.abs(actual[index] - expected[index]) <= tolerance * Math.max(1.0f, Math.abs(expected[index])),
                    "embedding mismatch at index " + index + ": actual=" + actual[index] + ", expected="
                            + expected[index]);
        }
    }

    private static float[] decodeBfloat16(MemorySegment bytes) {
        int count = Math.toIntExact(bytes.byteSize() / Short.BYTES);
        float[] values = new float[count];
        for (int index = 0; index < count; index++) {
            int bits = Short.toUnsignedInt(bytes.get(ValueLayout.JAVA_SHORT, (long) index * Short.BYTES));
            values[index] = Float.intBitsToFloat(bits << 16);
        }
        return values;
    }

    private static TensorDescriptor findBfloat16Embedding(QwenArtifact artifact) {
        List<TensorDescriptor> matches = Arrays.stream(artifact.tensors())
                .filter(tensor -> tensor.name().endsWith("embed_tokens.weight"))
                .toList();
        assertEquals(1, matches.size(), "expected one source embedding tensor");
        TensorDescriptor embedding = matches.getFirst();
        assertEquals(TensorDataType.BF16, embedding.dataType());
        assertEquals(WeightFormat.BF16, embedding.format());
        return embedding;
    }

    private static float[] readBfloat16Rows(
            Path artifactPath, TensorDescriptor embedding, int[] tokenIds, int hiddenSize) throws IOException {
        long rowBytes = (long) hiddenSize * Short.BYTES;
        float[] values = new float[tokenIds.length * hiddenSize];
        try (FileChannel channel = FileChannel.open(artifactPath, StandardOpenOption.READ)) {
            for (int tokenIndex = 0; tokenIndex < tokenIds.length; tokenIndex++) {
                ByteBuffer row = ByteBuffer.allocate(Math.toIntExact(rowBytes)).order(ByteOrder.LITTLE_ENDIAN);
                long offset = embedding.dataOffset() + tokenIds[tokenIndex] * rowBytes;
                while (row.hasRemaining()) {
                    int read = channel.read(row, offset + row.position());
                    if (read < 0) {
                        throw new IOException(
                                "truncated BF16 reference embedding row for token " + tokenIds[tokenIndex]);
                    }
                }
                row.flip();
                for (int index = 0; index < hiddenSize; index++) {
                    int bits = Short.toUnsignedInt(row.getShort());
                    values[tokenIndex * hiddenSize + index] = Float.intBitsToFloat(bits << 16);
                }
            }
        }
        return values;
    }

    private static void assertEmbeddingCloseToSource(float[] actual, float[] reference) {
        assertEquals(reference.length, actual.length);
        double squaredError = 0;
        float maximumAbsoluteError = 0;
        for (int index = 0; index < reference.length; index++) {
            float error = Math.abs(actual[index] - reference[index]);
            maximumAbsoluteError = Math.max(maximumAbsoluteError, error);
            squaredError += (double) error * error;
        }
        double rootMeanSquareError = Math.sqrt(squaredError / reference.length);
        assertTrue(
                maximumAbsoluteError <= MODEL_MAX_ABSOLUTE_TOLERANCE,
                "Q3 embedding max absolute error " + maximumAbsoluteError);
        assertTrue(rootMeanSquareError <= MODEL_RMS_TOLERANCE, "Q3 embedding RMSE " + rootMeanSquareError);
    }

    private static Path nativeLibraryPath() {
        return Path.of(System.getProperty("euhedral.cuda.library"));
    }
}
