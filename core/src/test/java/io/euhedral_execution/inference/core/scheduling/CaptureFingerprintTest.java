package io.euhedral_execution.inference.core.scheduling;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;

import io.euhedral_execution.inference.core.gpu.GpuMemory;
import java.lang.foreign.MemorySegment;
import java.util.HashMap;
import java.util.Map;
import org.junit.jupiter.api.Test;

/// A capture key identifies allocations, not addresses: a closed session's address that a new allocation
/// takes over must key differently, or a replay would run a graph whose nodes reference freed memory.
class CaptureFingerprintTest {

    /// Hands out one address again after it was freed, with a new allocation identity each time.
    private static final class ReusingGpu implements GpuMemory {
        private final Map<Long, Long> ids = new HashMap<>();
        private long serial;

        @Override
        public long allocate(long byteSize) {
            this.ids.put(4096L, ++this.serial);
            return 4096L;
        }

        @Override
        public void free(long address) {
            this.ids.remove(address);
        }

        @Override
        public long allocationId(long address) {
            return this.ids.getOrDefault(address, 0L);
        }

        @Override
        public void copyHostToDevice(long destination, MemorySegment source, long byteSize) {}

        @Override
        public void copyDeviceToHost(MemorySegment destination, long source, long byteSize) {}
    }

    private static long fingerprint(QwenWorkspaceStorage storage) {
        CaptureFingerprint fingerprint = new CaptureFingerprint();
        storage.fingerprint(fingerprint);
        return fingerprint.value();
    }

    @Test
    void anAddressTakenOverByANewAllocationKeysDifferently() {
        var gpu = new ReusingGpu();
        var first = new QwenWorkspaceStorage(gpu);
        assertEquals(4096L, first.acquire(0, 64));
        long captured = fingerprint(first);
        assertEquals(captured, fingerprint(first), "the same allocations key the same");
        first.close();

        var second = new QwenWorkspaceStorage(gpu);
        assertEquals(4096L, second.acquire(0, 64), "the new storage took over the freed address");
        assertNotEquals(captured, fingerprint(second));
    }
}
