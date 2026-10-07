package io.euhedral_execution.inference.core.model.qwen38;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Map;
import java.util.TreeMap;

/// Test GPU whose allocations are byte arrays, so copies move real bytes. Every address is distinct and
/// never reused, and a copy outside an allocation fails.
public class MemoryGpu extends ExecutionGpu {
    private final TreeMap<Long, byte[]> memory = new TreeMap<>();
    private long next = 1 << 20;

    @Override
    public synchronized long allocate(long byteSize) {
        long address = this.next;
        this.next += byteSize + 4096;
        this.memory.put(address, new byte[Math.toIntExact(byteSize)]);
        return address;
    }

    @Override
    public synchronized void free(long address) {
        if (this.memory.remove(address) == null) throw new IllegalStateException("unknown or duplicate free");
    }

    @Override
    public synchronized void copyHostToDevice(long destination, MemorySegment source, long byteSize) {
        byte[] bytes = new byte[Math.toIntExact(byteSize)];
        MemorySegment.copy(source, ValueLayout.JAVA_BYTE, 0, bytes, 0, bytes.length);
        write(destination, bytes);
    }

    @Override
    public synchronized void copyDeviceToHost(MemorySegment destination, long source, long byteSize) {
        byte[] bytes = read(source, Math.toIntExact(byteSize));
        MemorySegment.copy(bytes, 0, destination, ValueLayout.JAVA_BYTE, 0, bytes.length);
    }

    @Override
    public synchronized void copyDeviceToDevice(long destination, long source, long byteSize) {
        write(destination, read(source, Math.toIntExact(byteSize)));
    }

    @Override
    public synchronized void zeroDeviceMemory(long address, long byteSize) {
        write(address, new byte[Math.toIntExact(byteSize)]);
    }

    /// Fills `bytes` at `address` with a pattern derived from `seed`, so tests can tell buffers apart.
    public synchronized void fill(long address, int bytes, int seed) {
        byte[] pattern = new byte[bytes];
        for (int i = 0; i < bytes; i++) pattern[i] = (byte) (seed * 31 + i * 7 + (i >>> 8));
        write(address, pattern);
    }

    public synchronized byte[] bytes(long address, int count) {
        return read(address, count);
    }

    @Override
    public void embedQ3(long a, long b, long c, long d, int e, int f, int g) {}

    @Override
    public void synchronize() {}

    private byte[] read(long address, int count) {
        Map.Entry<Long, byte[]> entry = this.memory.floorEntry(address);
        int offset = entry == null ? -1 : Math.toIntExact(address - entry.getKey());
        if (entry == null || offset + count > entry.getValue().length)
            throw new IllegalStateException("read outside an allocation at " + address);
        byte[] out = new byte[count];
        System.arraycopy(entry.getValue(), offset, out, 0, count);
        return out;
    }

    private void write(long address, byte[] bytes) {
        Map.Entry<Long, byte[]> entry = this.memory.floorEntry(address);
        int offset = entry == null ? -1 : Math.toIntExact(address - entry.getKey());
        if (entry == null || offset + bytes.length > entry.getValue().length)
            throw new IllegalStateException("write outside an allocation at " + address);
        System.arraycopy(bytes, 0, entry.getValue(), offset, bytes.length);
    }
}
