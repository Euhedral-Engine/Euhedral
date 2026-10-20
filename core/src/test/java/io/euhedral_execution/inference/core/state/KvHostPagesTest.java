package io.euhedral_execution.inference.core.state;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.gpu.GpuMemory;
import java.lang.foreign.MemorySegment;
import java.util.HashMap;
import java.util.Map;
import org.junit.jupiter.api.Test;

class KvHostPagesTest {
    private static final long PAGE = 4096;

    @Test
    void slotsCarveTheArenaPerLayerAndShareOneRegionPerSlot() {
        var gpu = new Memory();
        try (KvHostPages pages = KvHostPages.allocate(gpu, 2, 3, 4, 5, PAGE)) {
            assertEquals(2 * 3 * 5 * PAGE, gpu.host.get(Memory.HOST));
            assertEquals(2 * 5 * PAGE, gpu.device.get(Memory.DEVICE));
            KvHostPages.Slot first = pages.take();
            KvHostPages.Slot second = pages.take();
            assertNotNull(first);
            assertNotNull(second);
            AttentionKvState.HostRun run = second.run(2);
            assertEquals(4, run.devicePages());
            assertEquals(5, run.pages());
            assertEquals(Memory.HOST + (1 * 3 + 2) * 5 * PAGE, run.host());
            assertEquals(Memory.DEVICE + 5 * PAGE, run.region());
            assertEquals(second.run(0).region(), run.region(), "a slot's layers share its region");
        }
        assertTrue(gpu.host.isEmpty());
        assertTrue(gpu.device.isEmpty());
    }

    @Test
    void aFullArenaRefusesInsteadOfWaitingAndAReleasedSlotIsTakenAgain() {
        try (KvHostPages pages = KvHostPages.allocate(new Memory(), 1, 1, 1, 1, PAGE)) {
            KvHostPages.Slot slot = pages.take();
            assertNotNull(slot);
            assertNull(pages.take());
            slot.release();
            slot.release();
            assertNotNull(pages.take());
            assertNull(pages.take());
        }
    }

    private static final class Memory implements GpuMemory {
        static final long HOST = 0x1000_0000L;
        static final long DEVICE = 0x2000_0000L;
        final Map<Long, Long> host = new HashMap<>();
        final Map<Long, Long> device = new HashMap<>();

        @Override
        public long allocate(long byteSize) {
            this.device.put(DEVICE, byteSize);
            return DEVICE;
        }

        @Override
        public void free(long address) {
            this.device.remove(address);
        }

        @Override
        public long allocateHostWeights(long byteSize) {
            this.host.put(HOST, byteSize);
            return HOST;
        }

        @Override
        public void freeHostWeights(long address) {
            this.host.remove(address);
        }

        @Override
        public void copyHostToDevice(long destination, MemorySegment source, long byteSize) {}

        @Override
        public void copyDeviceToHost(MemorySegment destination, long source, long byteSize) {}
    }
}
