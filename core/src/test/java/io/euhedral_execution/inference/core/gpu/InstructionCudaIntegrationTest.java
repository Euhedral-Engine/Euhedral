package io.euhedral_execution.inference.core.gpu;

import static io.euhedral_execution.inference.core.gpu.CudaGpuOperationsIntegrationTest.assertBf16Equals;
import static io.euhedral_execution.inference.core.gpu.CudaGpuOperationsIntegrationTest.download;
import static io.euhedral_execution.inference.core.gpu.CudaGpuOperationsIntegrationTest.floatToBf16;
import static io.euhedral_execution.inference.core.gpu.CudaGpuOperationsIntegrationTest.linearReference;
import static io.euhedral_execution.inference.core.gpu.CudaGpuOperationsIntegrationTest.q3Weights;
import static io.euhedral_execution.inference.core.gpu.CudaGpuOperationsIntegrationTest.rmsNorm;
import static io.euhedral_execution.inference.core.gpu.CudaGpuOperationsIntegrationTest.upload;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.artifact.TensorDataType;
import io.euhedral_execution.inference.core.artifact.TensorHandle;
import io.euhedral_execution.inference.core.artifact.WeightFormat;
import io.euhedral_execution.inference.core.artifact.WeightLayout;
import io.euhedral_execution.inference.core.model.qwen38.ExecutionPlan;
import io.euhedral_execution.inference.core.model.qwen38.LayerType;
import io.euhedral_execution.inference.core.model.qwen38.Quantum;
import io.euhedral_execution.inference.core.model.qwen38.Qwen38Config;
import io.euhedral_execution.inference.core.model.qwen38.Sequence;
import io.euhedral_execution.inference.core.model.qwen38.TestExecution;
import io.euhedral_execution.inference.core.model.qwen38.loader.LayerWeights;
import io.euhedral_execution.inference.core.model.qwen38.loader.Weights;
import java.lang.foreign.Arena;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;

class InstructionCudaIntegrationTest {

    @Test
    void embeddingReferenceDecodesZeroAndSubnormalScales() {
        int width = 128;
        byte[] weights = q3Weights(1, width);
        int scaleOffset = (2 * 24 + 255) & ~255;
        ByteBuffer scales = ByteBuffer.wrap(weights).order(ByteOrder.LITTLE_ENDIAN);
        scales.putShort(scaleOffset, (short) 0);
        scales.putShort(scaleOffset + Short.BYTES, (short) 1);

        short[] rows = embeddingRows(weights, new int[] {0}, 1, width);

        assertEquals(floatToBf16(-0.0f), rows[0]);
        assertEquals(floatToBf16(Float.float16ToFloat((short) 1)), rows[64]);
    }

    @Test
    void euhedralInstructionsProduceCpuReferencedProjection() throws Exception {
        int width = 128;
        int vocabulary = 8;
        int[] tokenIds = {1, 3};
        int outputs = 3;
        float epsilon = 1.0e-5f;
        byte[] embedding = q3Weights(vocabulary, width);
        byte[] projection = q3Weights(outputs, width);
        short[] normWeights = new short[width];
        for (int col = 0; col < width; col++) normWeights[col] = floatToBf16(0.75f + (col % 7) * 0.03125f);
        short[] expectedHidden = embeddingRows(embedding, tokenIds, vocabulary, width);
        short[] expectedNorm = rmsNorm(expectedHidden, normWeights, tokenIds.length, width, epsilon);
        short[] expectedProjection = linearReference(expectedNorm, projection, tokenIds.length, width, outputs);

        // The terminal consumer runs on the lattice worker that retires the quantum.
        try (CudaGpuMemory gpu = new CudaGpuMemory(Path.of(System.getProperty("euhedral.cuda.library")));
                Arena arena = Arena.ofShared()) {
            long embeddingAddress = upload(gpu, arena, embedding);
            long projectionAddress = upload(gpu, arena, projection);
            long normAddress = upload(gpu, arena, normWeights);
            try {
                TensorHandle embeddingHandle =
                        q3Handle("embedding", vocabulary, width, embeddingAddress, embedding.length);
                TensorHandle projectionHandle =
                        q3Handle("projection", outputs, width, projectionAddress, projection.length);
                TensorHandle normHandle = new TensorHandle(
                        "norm",
                        new long[] {width},
                        TensorDataType.BF16,
                        WeightFormat.BF16,
                        WeightLayout.CONTIGUOUS_LE_V1,
                        normAddress,
                        width * Short.BYTES);
                Qwen38Config config = new Qwen38Config(
                        vocabulary,
                        width,
                        0,
                        1,
                        1,
                        width,
                        width,
                        1,
                        1,
                        1,
                        1,
                        1,
                        epsilon,
                        1_000_000.0,
                        1.0,
                        128,
                        "silu",
                        new LayerType[0],
                        0,
                        0,
                        0,
                        0,
                        false,
                        false,
                        0);
                Weights weights =
                        new Weights(config, embeddingHandle, new LayerWeights[0], normHandle, projectionHandle, null);
                ExecutionPlan plan = new ExecutionPlan(weights, normHandle, List.of(projectionHandle));
                Quantum context = new Quantum(plan, new Sequence(30), Quantum.ExecutionKind.PREFILL, 0, tokenIds);
                AtomicReference<short[]> actualNorm = new AtomicReference<>();
                AtomicReference<short[]> actualProjection = new AtomicReference<>();
                Quantum.Outcome outcome = TestExecution.run(
                        plan,
                        gpu,
                        context,
                        completed -> {
                            actualNorm.set(download(
                                    gpu,
                                    arena,
                                    completed.workspace().normalizedStateAddress(),
                                    tokenIds.length * width));
                            actualProjection.set(download(
                                    gpu, arena, completed.workspace().projectionAddress(0), tokenIds.length * outputs));
                        },
                        30);
                assertEquals(Quantum.Status.SUCCESS, outcome.status(), () -> String.valueOf(outcome.failure()));
                assertTrue(context.workspace().isClosed());
                assertBf16Equals(expectedNorm, actualNorm.get(), 0.01f);
                assertBf16Equals(expectedProjection, actualProjection.get(), 0.03f);
            } finally {
                gpu.free(normAddress);
                gpu.free(projectionAddress);
                gpu.free(embeddingAddress);
            }
        }
    }

    private static TensorHandle q3Handle(String name, int rows, int width, long address, long byteSize) {
        return new TensorHandle(
                name,
                new long[] {rows, width},
                TensorDataType.BF16,
                WeightFormat.Q3_G64_FP16,
                WeightLayout.ROW_SPLIT_K128_V1,
                address,
                byteSize);
    }

    private static short[] embeddingRows(byte[] weights, int[] tokenIds, int vocabulary, int width) {
        int groups = ((width + 127) / 128) * 2;
        int scaleOffset = (vocabulary * groups * 24 + 255) & ~255;
        ByteBuffer scales = ByteBuffer.wrap(weights).order(ByteOrder.LITTLE_ENDIAN);
        short[] result = new short[tokenIds.length * width];
        for (int row = 0; row < tokenIds.length; row++) {
            for (int col = 0; col < width; col++) {
                int group = col / 64;
                int bit = (col % 64) * 3;
                int offset = (tokenIds[row] * groups + group) * 24 + (bit >>> 3);
                int packed = (weights[offset] & 255) | ((weights[offset + 1] & 255) << 8);
                int code = (packed >>> (bit & 7)) & 7;
                if (code >= 4) code -= 8;
                float scale = Float.float16ToFloat(scales.getShort(scaleOffset + (tokenIds[row] * groups + group) * 2));
                result[row * width + col] = floatToBf16(code * scale);
            }
        }
        return result;
    }
}
