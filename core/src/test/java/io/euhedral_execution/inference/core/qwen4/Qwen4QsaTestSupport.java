package io.euhedral_execution.inference.core.qwen4;

import static io.euhedral_execution.inference.core.qwen4.Qwen4Reference.bits;

import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import io.euhedral_execution.inference.core.scheduling.AttentionKvState;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;
import java.util.SplittableRandom;

/// Device helpers of the QSA GPU tests: opening the GPU, uploads and downloads of arrays, the decoded
/// contents of a
/// sequence's KV pages.
final class Qwen4QsaTestSupport {

    private Qwen4QsaTestSupport() {}

    static CudaGpuMemory open() {
        return new CudaGpuMemory(Path.of(System.getProperty("euhedral.cuda.library")));
    }

    static short[] random(SplittableRandom rng, int count, double scale) {
        short[] values = new short[count];
        for (int i = 0; i < count; i++) {
            double u = rng.nextDouble(), v = rng.nextDouble();
            double normal = Math.sqrt(-2 * Math.log(Math.max(u, 1e-12))) * Math.cos(2 * Math.PI * v);
            values[i] = bits((float) (normal * scale));
        }
        return values;
    }

    static long upload(CudaGpuMemory gpu, Arena arena, short[] values) {
        long address = gpu.allocate(Math.max(16, (long) values.length * Short.BYTES));
        MemorySegment host = arena.allocate((long) values.length * Short.BYTES, 16);
        MemorySegment.copy(values, 0, host, ValueLayout.JAVA_SHORT, 0, values.length);
        gpu.copyHostToDevice(address, host, (long) values.length * Short.BYTES);
        return address;
    }

    static long upload(CudaGpuMemory gpu, Arena arena, int[] values) {
        long address = gpu.allocate(Math.max(16, (long) values.length * Integer.BYTES));
        MemorySegment host = arena.allocate((long) values.length * Integer.BYTES, 16);
        MemorySegment.copy(values, 0, host, ValueLayout.JAVA_INT, 0, values.length);
        gpu.copyHostToDevice(address, host, (long) values.length * Integer.BYTES);
        return address;
    }

    static short[] downloadShorts(CudaGpuMemory gpu, Arena arena, long address, long count) {
        MemorySegment host = arena.allocate(count * Short.BYTES, 16);
        gpu.copyDeviceToHost(host, address, count * Short.BYTES);
        short[] values = new short[Math.toIntExact(count)];
        MemorySegment.copy(host, ValueLayout.JAVA_SHORT, 0, values, 0, values.length);
        return values;
    }

    static int[] downloadInts(CudaGpuMemory gpu, Arena arena, long address, long count) {
        MemorySegment host = arena.allocate(count * Integer.BYTES, 16);
        gpu.copyDeviceToHost(host, address, count * Integer.BYTES);
        int[] values = new int[Math.toIntExact(count)];
        MemorySegment.copy(host, ValueLayout.JAVA_INT, 0, values, 0, values.length);
        return values;
    }

    static float[] downloadFloats(CudaGpuMemory gpu, Arena arena, long address, long count) {
        MemorySegment host = arena.allocate(count * Float.BYTES, 16);
        gpu.copyDeviceToHost(host, address, count * Float.BYTES);
        float[] values = new float[Math.toIntExact(count)];
        MemorySegment.copy(host, ValueLayout.JAVA_FLOAT, 0, values, 0, values.length);
        return values;
    }

    static byte[] downloadBytes(CudaGpuMemory gpu, Arena arena, long address, long count) {
        MemorySegment host = arena.allocate(count, 16);
        gpu.copyDeviceToHost(host, address, count);
        byte[] values = new byte[Math.toIntExact(count)];
        MemorySegment.copy(host, ValueLayout.JAVA_BYTE, 0, values, 0, values.length);
        return values;
    }

    /// The effective (unrotated, dequantized) K or V rows of one KV head of the first `tokens` positions
    /// held by a
    /// state's page table (`table`: the device table of page addresses), `[token][256]`.
    static double[][] decodeCache(CudaGpuMemory gpu, Arena arena, long table, int tokens, int heads, int head) {
        int pages = (tokens + AttentionKvState.PAGE_TOKENS - 1) / AttentionKvState.PAGE_TOKENS;
        MemorySegment addresses = arena.allocate((long) pages * Long.BYTES, 8);
        gpu.copyDeviceToHost(addresses, table, (long) pages * Long.BYTES);
        double[][] rows = new double[tokens][];
        long pageBytes = (long) AttentionKvState.PAGE_TOKENS * heads * AttentionKvState.HEAD_ROW_BYTES;
        for (int page = 0; page < pages; page++) {
            byte[] bytes = downloadBytes(gpu, arena, addresses.getAtIndex(ValueLayout.JAVA_LONG, page), pageBytes);
            for (int t = page * AttentionKvState.PAGE_TOKENS;
                    t < Math.min(tokens, (page + 1) * AttentionKvState.PAGE_TOKENS);
                    t++) {
                int offset = ((t % AttentionKvState.PAGE_TOKENS) * heads + head) * AttentionKvState.HEAD_ROW_BYTES;
                rows[t] = Qwen4QsaReference.decodeRow(bytes, offset);
            }
        }
        return rows;
    }
}
