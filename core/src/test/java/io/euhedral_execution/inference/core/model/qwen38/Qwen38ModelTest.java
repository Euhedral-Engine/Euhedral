package io.euhedral_execution.inference.core.model.qwen38;

import static org.junit.jupiter.api.Assertions.*;

import io.euhedral_execution.inference.core.gpu.GpuMemory;
import io.euhedral_execution.inference.core.model.qwen38.loader.Weights;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

class Qwen38ModelTest {
    @Test
    void ownsOnlyAllocationsStillLiveAfterLoadingAndClosesOnce() throws Exception {
        var gpu = new Memory();
        var model = Qwen38Model.load(gpu, memory -> {
            long temporary = memory.allocate(4);
            memory.allocate(8);
            memory.free(temporary);
            return new Weights(null, null, null, null, null, null);
        });
        assertEquals(List.of(1L), gpu.freed);
        model.close();
        model.close();
        assertEquals(List.of(1L, 2L), gpu.freed);
    }

    @Test
    void partialLoadFailureReleasesAllocations() {
        var gpu = new Memory();
        var failure = new IllegalStateException("load");
        assertSame(
                failure,
                assertThrows(
                        IllegalStateException.class,
                        () -> Qwen38Model.load(gpu, memory -> {
                            memory.allocate(4);
                            throw failure;
                        })));
        assertEquals(List.of(1L), gpu.freed);
    }

    @Test
    void failedRollbackRetainsAllocationsForExplicitRetry() {
        var failFree = new java.util.concurrent.atomic.AtomicBoolean(true);
        var gpu = new Memory() {
            @Override
            public void free(long address) {
                if (failFree.get()) throw new IllegalStateException("free failed");
                super.free(address);
            }
        };
        var failed = assertThrows(
                Qwen38Model.LoadFailure.class,
                () -> Qwen38Model.load(gpu, memory -> {
                    memory.allocate(8);
                    throw new IllegalStateException("upload failed");
                }));
        assertTrue(gpu.freed.isEmpty());
        assertEquals("upload failed", failed.getCause().getMessage());
        failFree.set(false);
        failed.close();
        failed.close();
        assertEquals(List.of(1L), gpu.freed);
    }

    static class Memory implements GpuMemory {
        long next;
        List<Long> freed = new ArrayList<>();

        public long allocate(long bytes) {
            return ++next;
        }

        public void free(long address) {
            freed.add(address);
        }

        public void copyHostToDevice(long address, MemorySegment source, long bytes) {}

        public void copyDeviceToHost(MemorySegment destination, long address, long bytes) {}
    }
}
