package io.euhedral_execution.inference.core.gpu;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.model_loader.QwenWeights;
import io.euhedral_execution.inference.core.model_loader.TensorLoader;
import io.euhedral_execution.inference.core.model_loader.artifact.CompactTensorLayout;
import io.euhedral_execution.inference.core.model_loader.artifact.QwenArtifact;
import io.euhedral_execution.inference.core.model_loader.artifact.QwenArtifactHeader;
import io.euhedral_execution.inference.core.model_loader.artifact.QwenArtifactReader;
import io.euhedral_execution.inference.core.model_loader.artifact.TensorDataReader;
import io.euhedral_execution.inference.core.model_loader.artifact.TensorDescriptor;
import io.euhedral_execution.inference.core.model_loader.config.QwenConfig;
import io.euhedral_execution.inference.core.model_loader.layer_weights.QwenLayerWeights;
import io.euhedral_execution.inference.core.model_loader.layer_weights.TensorDataType;
import io.euhedral_execution.inference.core.model_loader.layer_weights.TensorHandle;
import io.euhedral_execution.inference.core.model_loader.layer_weights.WeightFormat;
import io.euhedral_execution.inference.core.model_loader.layer_weights.WeightLayout;
import io.euhedral_execution.inference.core.scheduling.QwenExecutionContext;
import io.euhedral_execution.inference.core.scheduling.QwenExecutionPlan;
import io.euhedral_execution.inference.core.scheduling.QwenSequenceState;
import io.euhedral_execution.inference.core.scheduling.TestExecution;
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
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;

class QwenCompactCudaExecutionIntegrationTest {

    private static final int PREFIX_OUTPUT_ROWS = 64;
    private static final long REQUIRED_FREE_VRAM_RESERVE = 128L * 1024L * 1024L;

    @Test
    void realCompactWeightsMatchStandaloneOperatorsAndEuhedralFrames() throws Exception {
        Path artifactPath = Path.of(System.getProperty("euhedral.qwen.artifact"));
        assertTrue(Files.isRegularFile(artifactPath), "compact Qwen artifact is missing: " + artifactPath);
        QwenArtifact artifact = QwenArtifactReader.read(artifactPath);
        assertEquals(QwenArtifactHeader.COMPACT_VERSION, artifact.header().version());
        Map<String, TensorDescriptor> descriptors = descriptorsByName(artifact.tensors());
        TensorDescriptor embeddingDescriptor = descriptors.get("text/token_embedding");
        TensorDescriptor normDescriptor = descriptors.get("text/layers/0/input_norm");
        TensorDescriptor projectionDescriptor = descriptors.get("text/layers/0/mlp/gate_up");
        assertTrue(embeddingDescriptor != null, "compact token embedding is missing");
        assertTrue(normDescriptor != null, "first-layer input RMSNorm is missing");
        assertTrue(projectionDescriptor != null, "first-layer Q3 gate/up weight is missing");

        QwenConfig config = artifact.config();
        int hiddenSize = config.hiddenSize();
        assertArrayEquals(new long[] {config.vocabSize(), hiddenSize}, embeddingDescriptor.shape());
        assertArrayEquals(new long[] {hiddenSize}, normDescriptor.shape());
        assertEquals(WeightFormat.Q3_G64_FP16, embeddingDescriptor.format());
        assertEquals(WeightLayout.ROW_SPLIT_K128_V1, embeddingDescriptor.layout());
        assertEquals(TensorDataType.BF16, normDescriptor.dataType());
        assertEquals(WeightFormat.BF16, normDescriptor.format());
        assertEquals(WeightLayout.CONTIGUOUS_LE_V1, normDescriptor.layout());
        assertEquals(WeightFormat.Q3_G64_FP16, projectionDescriptor.format());
        assertEquals(WeightLayout.ROW_SPLIT_K128_V1, projectionDescriptor.layout());
        assertEquals(hiddenSize, projectionDescriptor.shape()[1]);

        Path libraryPath = Path.of(System.getProperty("euhedral.cuda.library"));
        try (CudaGpuMemory gpu = new CudaGpuMemory(libraryPath);
                Arena arena = Arena.ofShared()) {
            CudaGpuMemory.DeviceMemoryInfo before = gpu.deviceMemoryInfo();
            long allocatedBefore = gpu.allocatedBytes();
            long requiredModelBytes = Math.addExact(
                    Math.addExact(embeddingDescriptor.byteSize(), normDescriptor.byteSize()),
                    projectionDescriptor.byteSize());
            assertTrue(
                    before.freeBytes() >= requiredModelBytes + REQUIRED_FREE_VRAM_RESERVE,
                    "insufficient free VRAM for the real execution slice: weights=" + requiredModelBytes + ", free="
                            + before.freeBytes());

            List<Long> ownedModelAddresses = new ArrayList<>();
            Throwable failure = null;
            try {
                TensorHandle embedding = loadTensor(artifactPath, embeddingDescriptor, gpu, ownedModelAddresses);
                TensorHandle norm = loadTensor(artifactPath, normDescriptor, gpu, ownedModelAddresses);
                TensorHandle projection = loadTensor(artifactPath, projectionDescriptor, gpu, ownedModelAddresses);
                byte[] projectionPrefix = prefixQ3Rows(artifactPath, projectionDescriptor, PREFIX_OUTPUT_ROWS);

                CudaGpuMemory.DeviceMemoryInfo resident = gpu.deviceMemoryInfo();
                assertTrue(resident.freeBytes() > REQUIRED_FREE_VRAM_RESERVE, "model slice exhausted device memory");
                byte[] normPayload = readTensor(artifactPath, normDescriptor);
                short[] normWeights = bf16Values(normPayload);
                byte[] fullProjectionPayload = readTensor(artifactPath, projectionDescriptor);
                int[] tokenCounts = {1, 4, 32};
                for (int tokenCount : tokenCounts) {
                    int[] tokenIds = tokenIds(tokenCount, config.vocabSize());
                    short[] expectedEmbedding = embeddingReference(artifactPath, embeddingDescriptor, tokenIds);
                    short[] expectedNorm = CudaGpuOperationsIntegrationTest.rmsNorm(
                            expectedEmbedding, normWeights, tokenCount, hiddenSize, (float) config.rmsNormEpsilon());
                    int outputWidth = Math.toIntExact(projection.shape()[0]);
                    int referenceOutputWidth = tokenCount == 1 ? outputWidth : PREFIX_OUTPUT_ROWS;
                    byte[] referenceProjectionPayload = tokenCount == 1 ? fullProjectionPayload : projectionPrefix;
                    short[] expectedProjection = CudaGpuOperationsIntegrationTest.linearReference(
                            expectedNorm, referenceProjectionPayload, tokenCount, hiddenSize, referenceOutputWidth);

                    validateStandaloneOperations(
                            gpu,
                            arena,
                            embedding,
                            norm,
                            projection,
                            tokenIds,
                            expectedEmbedding,
                            expectedNorm,
                            expectedProjection,
                            outputWidth,
                            referenceOutputWidth,
                            config);
                    validateEuhedralFrames(
                            gpu,
                            arena,
                            config,
                            embedding,
                            norm,
                            projection,
                            tokenIds,
                            expectedEmbedding,
                            expectedNorm,
                            expectedProjection,
                            outputWidth,
                            referenceOutputWidth,
                            tokenCount);
                    assertTrue(
                            gpu.deviceMemoryInfo().freeBytes() > REQUIRED_FREE_VRAM_RESERVE,
                            "resident model weights left insufficient workspace headroom at T=" + tokenCount);
                }
                System.out.printf(
                        "Real compact CUDA execution passed: artifact=%s residentWeightBytes=%d freeBefore=%d "
                                + "freeWithWeights=%d shapes=T1/full,T4/full,T32/full "
                                + "reference=full@T1,first64@T4/T32%n",
                        artifactPath, requiredModelBytes, before.freeBytes(), resident.freeBytes());
            } catch (Throwable executionFailure) {
                failure = executionFailure;
            } finally {
                failure = freeAll(gpu, ownedModelAddresses, failure);
            }

            gpu.releaseQ3Scratch(); // the shared scratch of the quantized-activation routes outlives a slice
            long allocatedAfter = gpu.allocatedBytes();
            if (allocatedAfter != allocatedBefore) {
                IllegalStateException restoreFailure = new IllegalStateException(
                        "real execution slice did not release GPU allocations: allocated before=" + allocatedBefore
                                + ", after=" + allocatedAfter);
                if (failure == null) failure = restoreFailure;
                else failure.addSuppressed(restoreFailure);
            }
            if (failure != null) rethrow(failure);
            System.out.printf("Real compact CUDA execution cleanup passed: allocated bytes=%d%n", allocatedAfter);
        }
    }

    private static void validateStandaloneOperations(
            CudaGpuMemory gpu,
            Arena arena,
            TensorHandle embedding,
            TensorHandle norm,
            TensorHandle projection,
            int[] tokenIds,
            short[] expectedEmbedding,
            short[] expectedNorm,
            short[] expectedProjection,
            int outputWidth,
            int referenceOutputWidth,
            QwenConfig config)
            throws Exception {
        List<Long> addresses = new ArrayList<>(4);
        Throwable failure = null;
        try {
            long tokenAddress = uploadInts(gpu, arena, tokenIds);
            addresses.add(tokenAddress);
            long inputAddress = gpu.allocate((long) expectedEmbedding.length * Short.BYTES);
            addresses.add(inputAddress);
            long normalizedAddress = gpu.allocate((long) expectedNorm.length * Short.BYTES);
            addresses.add(normalizedAddress);
            long projectionAddress = gpu.allocate((long) tokenIds.length * outputWidth * Short.BYTES);
            addresses.add(projectionAddress);

            gpu.embedQ3(
                    tokenAddress,
                    embedding.deviceAddress(),
                    embedding.byteSize(),
                    inputAddress,
                    tokenIds.length,
                    config.vocabSize(),
                    config.hiddenSize());
            short[] actualEmbedding =
                    CudaGpuOperationsIntegrationTest.download(gpu, arena, inputAddress, expectedEmbedding.length);
            assertBf16Close("standalone embedding T=" + tokenIds.length, expectedEmbedding, actualEmbedding, 0.001f);

            gpu.rmsNormBf16(
                    inputAddress, norm.deviceAddress(), normalizedAddress, tokenIds.length, config.hiddenSize(), (float)
                            config.rmsNormEpsilon());
            short[] actualNorm =
                    CudaGpuOperationsIntegrationTest.download(gpu, arena, normalizedAddress, expectedNorm.length);
            assertBf16Close("standalone RMSNorm T=" + tokenIds.length, expectedNorm, actualNorm, 0.02f);

            gpu.linearQ3Bf16(
                    normalizedAddress,
                    projection.deviceAddress(),
                    projectionAddress,
                    tokenIds.length,
                    config.hiddenSize(),
                    outputWidth,
                    projection.byteSize());
            short[] actualProjection = CudaGpuOperationsIntegrationTest.download(
                    gpu, arena, projectionAddress, tokenIds.length * outputWidth);
            assertBf16Close(
                    "standalone Q3 linear T=" + tokenIds.length,
                    expectedProjection,
                    prefixOutputRows(actualProjection, tokenIds.length, outputWidth, referenceOutputWidth),
                    0.05f);
        } catch (Throwable operationFailure) {
            failure = operationFailure;
        } finally {
            failure = freeAll(gpu, addresses, failure);
        }
        if (failure != null) rethrow(failure);
    }

    private static void validateEuhedralFrames(
            CudaGpuMemory gpu,
            Arena arena,
            QwenConfig config,
            TensorHandle embedding,
            TensorHandle norm,
            TensorHandle projection,
            int[] tokenIds,
            short[] expectedEmbedding,
            short[] expectedNorm,
            short[] expectedProjection,
            int outputWidth,
            int referenceOutputWidth,
            int tokenCount)
            throws Exception {
        QwenWeights sliceWeights = new QwenWeights(config, embedding, new QwenLayerWeights[0], norm, projection, null);
        QwenExecutionPlan plan = new QwenExecutionPlan(sliceWeights, norm, List.of(projection));
        QwenSequenceState sequence = new QwenSequenceState(9000L + tokenCount);
        QwenExecutionContext.ExecutionKind kind = tokenCount == 1
                ? QwenExecutionContext.ExecutionKind.DECODE
                : QwenExecutionContext.ExecutionKind.PREFILL;
        QwenExecutionContext context = new QwenExecutionContext(plan, sequence, kind, 0, tokenIds);
        AtomicReference<FrameOutputs> captured = new AtomicReference<>();
        QwenExecutionContext.Outcome outcome = TestExecution.run(
                plan,
                gpu,
                context,
                completed -> {
                    try (Arena outputArena = Arena.ofShared()) {
                        captured.set(new FrameOutputs(
                                CudaGpuOperationsIntegrationTest.download(
                                        gpu,
                                        outputArena,
                                        completed.workspace().hiddenStateAddress(),
                                        tokenCount * config.hiddenSize()),
                                CudaGpuOperationsIntegrationTest.download(
                                        gpu,
                                        outputArena,
                                        completed.workspace().normalizedStateAddress(),
                                        tokenCount * config.hiddenSize()),
                                CudaGpuOperationsIntegrationTest.download(
                                        gpu,
                                        outputArena,
                                        completed.workspace().projectionAddress(0),
                                        tokenCount * outputWidth)));
                    }
                },
                120);
        assertEquals(QwenExecutionContext.Status.SUCCESS, outcome.status(), "Euhedral slice failed at T=" + tokenCount);
        assertTrue(context.workspace().isClosed(), "submission workspace survived terminal completion");
        assertEquals(tokenCount, sequence.currentTokenPosition());
        assertTrue(captured.get() != null, "terminal frame outputs were not captured");
        assertBf16Close(
                "frame EmbeddingFrame T=" + tokenCount,
                expectedEmbedding,
                captured.get().embedding(),
                0.001f);
        assertBf16Close(
                "frame RmsNormFrame T=" + tokenCount,
                expectedNorm,
                captured.get().normalized(),
                0.02f);
        assertBf16Close(
                "frame LinearFrame T=" + tokenCount,
                expectedProjection,
                prefixOutputRows(captured.get().projection(), tokenCount, outputWidth, referenceOutputWidth),
                0.05f);
    }

    private static short[] embeddingReference(Path artifactPath, TensorDescriptor descriptor, int[] tokenIds)
            throws IOException {
        int rows = Math.toIntExact(descriptor.shape()[0]);
        int width = Math.toIntExact(descriptor.shape()[1]);
        int groups = ((width + 127) / 128) * 2;
        int codeBytesPerRow = groups * 24;
        int scaleBytesPerRow = groups * Short.BYTES;
        long scaleOffset = align256((long) rows * codeBytesPerRow);
        short[] result = new short[tokenIds.length * width];
        try (FileChannel channel = FileChannel.open(artifactPath, StandardOpenOption.READ)) {
            for (int row = 0; row < tokenIds.length; row++) {
                int tokenId = tokenIds[row];
                byte[] codes =
                        readRange(channel, descriptor.dataOffset() + (long) tokenId * codeBytesPerRow, codeBytesPerRow);
                byte[] scaleBytes = readRange(
                        channel,
                        descriptor.dataOffset() + scaleOffset + (long) tokenId * scaleBytesPerRow,
                        scaleBytesPerRow);
                ByteBuffer scales = ByteBuffer.wrap(scaleBytes).order(ByteOrder.LITTLE_ENDIAN);
                for (int column = 0; column < width; column++) {
                    int group = column / 64;
                    int bit = (column % 64) * 3;
                    int byteOffset = group * 24 + (bit >>> 3);
                    int shift = bit & 7;
                    int packed = codes[byteOffset] & 0xff;
                    if (shift > 5) packed |= (codes[byteOffset + 1] & 0xff) << 8;
                    int code = (packed >>> shift) & 7;
                    if (code >= 4) code -= 8;
                    float scale = Float.float16ToFloat(scales.getShort(group * Short.BYTES));
                    result[row * width + column] = CudaGpuOperationsIntegrationTest.floatToBf16(code * scale);
                }
            }
        }
        return result;
    }

    private static byte[] prefixQ3Rows(Path artifactPath, TensorDescriptor descriptor, int prefixRows)
            throws IOException {
        int inputWidth = Math.toIntExact(descriptor.shape()[1]);
        int groups = ((inputWidth + 127) / 128) * 2;
        long sourceCodeBytes = descriptor.shape()[0] * groups * 24L;
        long sourceScaleOffset = align256(sourceCodeBytes);
        int codeBytesPerRow = groups * 24;
        int prefixCodeBytes = Math.multiplyExact(prefixRows, codeBytesPerRow);
        int scaleBytes = Math.multiplyExact(prefixRows, groups * Short.BYTES);
        int prefixScaleOffset = Math.toIntExact(align256(prefixCodeBytes));
        byte[] result = new byte[Math.addExact(prefixScaleOffset, scaleBytes)];
        try (FileChannel channel = FileChannel.open(artifactPath, StandardOpenOption.READ)) {
            readRange(channel, descriptor.dataOffset(), result, 0, prefixCodeBytes);
            readRange(channel, descriptor.dataOffset() + sourceScaleOffset, result, prefixScaleOffset, scaleBytes);
        }
        long expectedSize = CompactTensorLayout.expectedByteSize(
                new long[] {prefixRows, inputWidth}, descriptor.dataType(), descriptor.format(), descriptor.layout());
        if (result.length != expectedSize) {
            throw new IOException("Q3 row prefix does not match compact format geometry");
        }
        return result;
    }

    private static byte[] readTensor(Path artifactPath, TensorDescriptor descriptor) throws IOException {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment segment = TensorDataReader.read(artifactPath, descriptor, arena);
            return segment.toArray(ValueLayout.JAVA_BYTE);
        }
    }

    private static short[] bf16Values(byte[] payload) {
        if ((payload.length & 1) != 0) throw new IllegalArgumentException("BF16 payload has an odd byte size");
        ByteBuffer values = ByteBuffer.wrap(payload).order(ByteOrder.LITTLE_ENDIAN);
        short[] result = new short[payload.length / Short.BYTES];
        for (int index = 0; index < result.length; index++) result[index] = values.getShort();
        return result;
    }

    private static short[] prefixOutputRows(short[] values, int rows, int width, int prefixWidth) {
        short[] result = new short[rows * prefixWidth];
        for (int row = 0; row < rows; row++) {
            System.arraycopy(values, row * width, result, row * prefixWidth, prefixWidth);
        }
        return result;
    }

    private static int[] tokenIds(int count, int vocabularySize) {
        int[] result = new int[count];
        for (int index = 0; index < count; index++) result[index] = 101 + (index * 7919 % (vocabularySize - 101));
        return result;
    }

    private static Map<String, TensorDescriptor> descriptorsByName(TensorDescriptor[] descriptors) {
        Map<String, TensorDescriptor> result = new HashMap<>();
        for (TensorDescriptor descriptor : descriptors) result.put(descriptor.name(), descriptor);
        return result;
    }

    private static TensorHandle loadTensor(
            Path artifactPath, TensorDescriptor descriptor, CudaGpuMemory gpu, List<Long> ownedAddresses)
            throws IOException {
        TensorHandle handle = TensorLoader.load(artifactPath, descriptor, gpu);
        ownedAddresses.add(handle.deviceAddress());
        return handle;
    }

    private static long uploadInts(CudaGpuMemory gpu, Arena arena, int[] values) {
        MemorySegment host = arena.allocate((long) values.length * Integer.BYTES, Integer.BYTES);
        for (int index = 0; index < values.length; index++) {
            host.set(ValueLayout.JAVA_INT, (long) index * Integer.BYTES, values[index]);
        }
        long address = gpu.allocate(host.byteSize());
        try {
            gpu.copyHostToDevice(address, host, host.byteSize());
            return address;
        } catch (RuntimeException | Error failure) {
            gpu.free(address);
            throw failure;
        }
    }

    private static byte[] readRange(FileChannel channel, long offset, int size) throws IOException {
        byte[] result = new byte[size];
        readRange(channel, offset, result, 0, size);
        return result;
    }

    private static void readRange(FileChannel channel, long offset, byte[] destination, int destinationOffset, int size)
            throws IOException {
        ByteBuffer buffer = ByteBuffer.wrap(destination, destinationOffset, size);
        long position = offset;
        while (buffer.hasRemaining()) {
            int read = channel.read(buffer, position);
            if (read < 0) throw new IOException("unexpected end of compact tensor payload");
            if (read == 0) continue;
            position += read;
        }
    }

    private static long align256(long size) {
        return (size + 255L) & ~255L;
    }

    private static void assertBf16Close(String operation, short[] expected, short[] actual, float tolerance) {
        assertEquals(expected.length, actual.length, operation + " output size");
        float maxError = 0;
        int maxIndex = -1;
        for (int index = 0; index < expected.length; index++) {
            float difference = Math.abs(CudaGpuOperationsIntegrationTest.bf16ToFloat(expected[index])
                    - CudaGpuOperationsIntegrationTest.bf16ToFloat(actual[index]));
            if (difference > maxError) {
                maxError = difference;
                maxIndex = index;
            }
        }
        assertTrue(
                maxError <= tolerance,
                operation + " max absolute error " + maxError + " at index " + maxIndex + " exceeds " + tolerance);
        System.out.printf("%s maxAbsError=%.8f%n", operation, maxError);
    }

    private static Throwable freeAll(CudaGpuMemory gpu, List<Long> addresses, Throwable failure) {
        for (int index = addresses.size() - 1; index >= 0; index--) {
            try {
                gpu.free(addresses.get(index));
            } catch (Throwable cleanupFailure) {
                if (failure == null) failure = cleanupFailure;
                else failure.addSuppressed(cleanupFailure);
            }
        }
        return failure;
    }

    private static void rethrow(Throwable failure) throws Exception {
        if (failure instanceof Exception exception) throw exception;
        if (failure instanceof Error error) throw error;
        throw new AssertionError(failure);
    }

    private record FrameOutputs(short[] embedding, short[] normalized, short[] projection) {}
}
