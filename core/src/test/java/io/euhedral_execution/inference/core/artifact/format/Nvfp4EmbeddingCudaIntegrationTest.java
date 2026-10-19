package io.euhedral_execution.inference.core.artifact.format;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.artifact.Nvfp4Layout;
import io.euhedral_execution.inference.core.artifact.TensorDataType;
import io.euhedral_execution.inference.core.artifact.TensorHandle;
import io.euhedral_execution.inference.core.artifact.WeightFormat;
import io.euhedral_execution.inference.core.artifact.WeightLayout;
import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;

/// A row-split NVFP4 token embedding, looked up through its [LinearFormat], gives each id's row as
/// bf16(e2m1(code) * (e4m3(scale) * global)) bit for bit, the vocabulary's last row included, and leaves the row of
/// an id outside the vocabulary as it was.
class Nvfp4EmbeddingCudaIntegrationTest {

    private static final float[] E2M1 = {0f, 0.5f, 1f, 1.5f, 2f, 3f, 4f, 6f};

    @Test
    void rowsAreTheDequantizedTable() {
        int vocabulary = 1024, hidden = 5120;
        int[] ids = {0, 1, 1023, 1023, 512, -1, 1024};
        long scaleOffset = Nvfp4Layout.scaleOffset(vocabulary, hidden);
        long globalOffset = Nvfp4Layout.globalScaleOffset(vocabulary, hidden);
        long bytes = Nvfp4Layout.byteSize(vocabulary, hidden);
        float global = 3.0e-3f;
        SplittableRandom random = new SplittableRandom(5);
        byte[] table = new byte[Math.toIntExact(bytes)];
        for (int i = 0; i < scaleOffset; i++) table[i] = (byte) random.nextInt(256);
        // Finite E4M3 scales, subnormals included (0x7f is NaN).
        for (long i = scaleOffset; i < scaleOffset + (long) vocabulary * hidden / 16; i++)
            table[(int) i] = (byte) random.nextInt(0x7f);
        MemorySegment.ofArray(table)
                .set(ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN), globalOffset, global);
        LinearFormat format = LinearFormats.find(WeightFormat.NVFP4, WeightLayout.ROW_SPLIT_K128_V1)
                .orElseThrow();
        assertTrue(format.embeds());
        try (CudaGpuMemory gpu = new CudaGpuMemory(Path.of(System.getProperty("euhedral.cuda.library")));
                Arena arena = Arena.ofConfined()) {
            long tableAddress = upload(gpu, arena, MemorySegment.ofArray(table), bytes);
            long idAddress = upload(gpu, arena, MemorySegment.ofArray(ids), 4L * ids.length);
            short[] sentinel = new short[ids.length * hidden];
            Arrays.fill(sentinel, (short) 0x7e7e);
            long output = upload(gpu, arena, MemorySegment.ofArray(sentinel), 2L * sentinel.length);
            TensorHandle handle = new TensorHandle(
                    "text/token_embedding",
                    new long[] {vocabulary, hidden},
                    TensorDataType.BF16,
                    WeightFormat.NVFP4,
                    WeightLayout.ROW_SPLIT_K128_V1,
                    tableAddress,
                    bytes);
            format.embed(gpu, new EmbeddingLaunch(idAddress, handle, output, ids.length, vocabulary, hidden));
            gpu.synchronize();
            MemorySegment host = arena.allocate(2L * sentinel.length, 16);
            gpu.copyDeviceToHost(host, output, host.byteSize());
            short[] rows = host.toArray(ValueLayout.JAVA_SHORT);
            for (int t = 0; t < ids.length; t++) {
                short[] expected = ids[t] < 0 || ids[t] >= vocabulary
                        ? Arrays.copyOfRange(sentinel, t * hidden, (t + 1) * hidden)
                        : row(table, ids[t], hidden, scaleOffset, global);
                assertArrayEquals(expected, Arrays.copyOfRange(rows, t * hidden, (t + 1) * hidden), "token " + t);
            }
            for (long address : new long[] {tableAddress, idAddress, output}) gpu.free(address);
        }
    }

    private static short[] row(byte[] table, int row, int hidden, long scaleOffset, float global) {
        short[] values = new short[hidden];
        for (int c = 0; c < hidden; c++) {
            int packed = table[row * (hidden / 2) + c / 2] & 0xff;
            int code = c % 2 == 0 ? packed & 15 : packed >> 4;
            float e2m1 = (code & 8) != 0 ? -E2M1[code & 7] : E2M1[code & 7];
            float scale = e4m3(table[(int) (scaleOffset + (long) row * (hidden / 16) + c / 16)] & 0xff) * global;
            values[c] = bf16(e2m1 * scale);
        }
        return values;
    }

    private static float e4m3(int code) {
        int exponent = (code >> 3) & 15, mantissa = code & 7;
        float magnitude = exponent == 0 ? mantissa * 0x1p-9f : (1 + mantissa / 8f) * (float) Math.pow(2, exponent - 7);
        return (code & 0x80) != 0 ? -magnitude : magnitude;
    }

    /// Round to nearest even (the kernels' euhedral_float_to_bfloat16; no NaN arises here).
    private static short bf16(float value) {
        int bits = Float.floatToRawIntBits(value);
        bits += 0x7fff + ((bits >>> 16) & 1);
        return (short) (bits >>> 16);
    }

    private static long upload(CudaGpuMemory gpu, Arena arena, MemorySegment values, long bytes) {
        MemorySegment host = arena.allocate(bytes, 16);
        host.copyFrom(values);
        long address = gpu.allocate(bytes);
        gpu.copyHostToDevice(address, host, bytes);
        return address;
    }
}
