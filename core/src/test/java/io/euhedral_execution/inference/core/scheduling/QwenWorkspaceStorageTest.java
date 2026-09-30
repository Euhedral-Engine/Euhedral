package io.euhedral_execution.inference.core.scheduling;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.gpu.GpuMemory;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class QwenWorkspaceStorageTest {

    @Test
    void aSlotWhoseCapacityCoversTheRequestIsReusedWithoutAllocating() {
        var gpu = new SizedGpu();
        var storage = new QwenWorkspaceStorage(gpu);
        long first = storage.acquire(3, 4096);
        assertEquals(first, storage.acquire(3, 4096));
        assertEquals(first, storage.acquire(3, 1024), "a smaller binding fits the retained allocation");
        assertEquals(List.of(first), gpu.allocations);
        assertTrue(gpu.frees.isEmpty());
        assertEquals(4096, storage.retainedBytes());
    }

    @Test
    void onlyAGrowingRequestReplacesItsSlotAndNeverReturnsAnUndersizedAllocation() {
        var gpu = new SizedGpu();
        var storage = new QwenWorkspaceStorage(gpu);
        long small = storage.acquire(0, 100);
        long other = storage.acquire(1, 300);
        long grown = storage.acquire(0, 250);
        assertNotEquals(small, grown);
        assertEquals(List.of(small), gpu.frees, "only the undersized slot is replaced");
        assertTrue(gpu.sizes.get(grown) >= 250);
        assertEquals(grown, storage.acquire(0, 200));
        assertEquals(other, storage.acquire(1, 300));
        assertEquals(250 + 300, storage.retainedBytes());
    }

    @Test
    void detachTransfersOwnershipSoCloseNeverFreesTheDetachedAllocation() {
        var gpu = new SizedGpu();
        var storage = new QwenWorkspaceStorage(gpu);
        long logits = storage.acquire(5, 512);
        assertEquals(logits, storage.detach(5));
        assertEquals(0, storage.retainedBytes());
        assertThrows(IllegalStateException.class, () -> storage.detach(5));
        long replacement = storage.acquire(5, 512);
        assertNotEquals(logits, replacement, "the next binding allocates its own buffer");
        storage.close();
        assertEquals(List.of(replacement), gpu.frees);
    }

    @Test
    void closeFreesEachAllocationOnceAndAFailedFreeKeepsItsSlotForARetry() {
        var gpu = new SizedGpu();
        var storage = new QwenWorkspaceStorage(gpu);
        long first = storage.acquire(0, 64);
        long second = storage.acquire(2, 64);
        gpu.failFreeOf = second;
        assertThrows(IllegalStateException.class, storage::close);
        assertFalse(storage.isClosed());
        assertEquals(List.of(first), gpu.frees);
        assertEquals(64, storage.retainedBytes());

        storage.close();
        storage.close();
        assertTrue(storage.isClosed());
        assertEquals(List.of(first, second), gpu.frees);
        assertEquals(0, storage.retainedBytes());
        assertThrows(IllegalStateException.class, () -> storage.acquire(0, 64));
    }

    @Test
    void rejectsInvalidRequestsWithoutAllocating() {
        var gpu = new SizedGpu();
        var storage = new QwenWorkspaceStorage(gpu);
        assertThrows(IllegalArgumentException.class, () -> storage.acquire(-1, 64));
        assertThrows(IllegalArgumentException.class, () -> storage.acquire(0, 0));
        assertTrue(gpu.allocations.isEmpty());
    }

    private static final class SizedGpu implements GpuMemory {
        final List<Long> allocations = new ArrayList<>();
        final List<Long> frees = new ArrayList<>();
        final Map<Long, Long> sizes = new HashMap<>();
        long failFreeOf;
        private long next = 1;

        @Override
        public long allocate(long byteSize) {
            long address = this.next++;
            this.allocations.add(address);
            this.sizes.put(address, byteSize);
            return address;
        }

        @Override
        public void copyHostToDevice(long destination, MemorySegment source, long byteSize) {
            throw new UnsupportedOperationException();
        }

        @Override
        public void copyDeviceToHost(MemorySegment destination, long source, long byteSize) {
            throw new UnsupportedOperationException();
        }

        @Override
        public void free(long address) {
            if (address == this.failFreeOf) {
                this.failFreeOf = 0;
                throw new IllegalStateException("injected free failure");
            }
            this.frees.add(address);
        }
    }
}
