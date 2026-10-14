package io.euhedral_execution.inference.core.gpu;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;

/// The BF16 token embedding gathers each id's row bit for bit, the vocabulary's last row included, and leaves the
/// row of an id outside the vocabulary as it was.
class Bf16EmbeddingCudaIntegrationTest {

    @Test
    void rowsGatherExactly() {
        int vocabulary = 1024, hidden = 5120;
        int[] ids = {0, 1, 1023, 1023, 512, -1, 1024};
        try (CudaGpuMemory gpu = new CudaGpuMemory(Path.of(System.getProperty("euhedral.cuda.library")));
                Arena arena = Arena.ofConfined()) {
            short[] table = new short[vocabulary * hidden];
            SplittableRandom random = new SplittableRandom(3);
            for (int i = 0; i < table.length; i++) table[i] = (short) random.nextInt(1 << 16);
            long tableAddress = upload(gpu, arena, MemorySegment.ofArray(table), 2L * table.length);
            long idAddress = upload(gpu, arena, MemorySegment.ofArray(ids), 4L * ids.length);
            short[] sentinel = new short[ids.length * hidden];
            Arrays.fill(sentinel, (short) 0x7e7e);
            long output = upload(gpu, arena, MemorySegment.ofArray(sentinel), 2L * sentinel.length);
            gpu.embedBf16(idAddress, tableAddress, 2L * table.length, output, ids.length, vocabulary, hidden);
            gpu.synchronize();
            MemorySegment host = arena.allocate(2L * sentinel.length, 16);
            gpu.copyDeviceToHost(host, output, host.byteSize());
            short[] rows = host.toArray(ValueLayout.JAVA_SHORT);
            for (int t = 0; t < ids.length; t++) {
                short[] expected = ids[t] < 0 || ids[t] >= vocabulary
                        ? Arrays.copyOfRange(sentinel, t * hidden, (t + 1) * hidden)
                        : Arrays.copyOfRange(table, ids[t] * hidden, (ids[t] + 1) * hidden);
                assertArrayEquals(expected, Arrays.copyOfRange(rows, t * hidden, (t + 1) * hidden), "token " + t);
            }
            for (long address : new long[] {tableAddress, idAddress, output}) gpu.free(address);
        }
    }

    private static long upload(CudaGpuMemory gpu, Arena arena, MemorySegment values, long bytes) {
        MemorySegment host = arena.allocate(bytes, 16);
        host.copyFrom(values);
        long address = gpu.allocate(bytes);
        gpu.copyHostToDevice(address, host, bytes);
        return address;
    }
}
