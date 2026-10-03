package io.euhedral_execution.inference.core.scheduling;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.gpu.GpuMemory;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

class QwenExecutionWorkspaceTest {

    @Test
    void allocatesOneBfloat16HiddenVectorPerInputTokenAndReleasesItOnce() {
        RecordingGpuMemory gpu = new RecordingGpuMemory();

        QwenExecutionWorkspace workspace = new QwenExecutionWorkspace(gpu, 3, 4096);
        workspace.allocateBuffers();
        long hiddenStateAddress = workspace.hiddenStateAddress();
        long inputAddress = workspace.tokenIdsAddress();

        assertEquals(3, workspace.tokenCount());
        assertEquals(4096, workspace.hiddenSize());
        assertEquals(3L * 4096 * Short.BYTES, workspace.byteSize());
        assertEquals(List.of(inputAddress, hiddenStateAddress), gpu.allocations());
        assertFalse(workspace.isClosed());

        workspace.close();
        workspace.close();

        assertTrue(workspace.isClosed());
        assertEquals(List.of(inputAddress, hiddenStateAddress), gpu.frees());
        assertThrows(IllegalStateException.class, workspace::hiddenStateAddress);
    }

    @Test
    void rejectsInvalidDimensionsBeforeAllocatingDeviceMemory() {
        RecordingGpuMemory gpu = new RecordingGpuMemory();

        assertThrows(IllegalArgumentException.class, () -> new QwenExecutionWorkspace(gpu, 0, 4096));
        assertThrows(IllegalArgumentException.class, () -> new QwenExecutionWorkspace(gpu, 2, 0));

        assertTrue(gpu.allocations().isEmpty());
    }

    @Test
    void freeFailureLeavesWorkspaceOpenForRetry() {
        RecordingGpuMemory gpu = new RecordingGpuMemory();
        QwenExecutionWorkspace workspace = new QwenExecutionWorkspace(gpu, 2, 64);
        workspace.allocateBuffers();
        long hiddenStateAddress = workspace.hiddenStateAddress();
        long inputAddress = workspace.tokenIdsAddress();
        gpu.freeFailuresRemaining = 1;

        // The input record's free fails; the hidden state is still freed, and the record kept for a retry.
        assertThrows(IllegalStateException.class, workspace::close);
        assertFalse(workspace.isClosed());
        assertEquals(hiddenStateAddress, workspace.hiddenStateAddress());
        assertEquals(List.of(hiddenStateAddress), gpu.frees());

        workspace.close();

        assertTrue(workspace.isClosed());
        assertEquals(List.of(hiddenStateAddress, inputAddress), gpu.frees());
    }

    private static final class RecordingGpuMemory implements GpuMemory {

        private final List<Long> allocations = new ArrayList<>();
        private final List<Long> frees = new ArrayList<>();
        private long nextAddress = 100;
        private int freeFailuresRemaining;

        @Override
        public long allocate(long byteSize) {
            assertTrue(byteSize > 0);
            long address = this.nextAddress++;
            this.allocations.add(address);
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
            if (this.freeFailuresRemaining > 0) {
                this.freeFailuresRemaining--;
                throw new IllegalStateException("synthetic free failure");
            }
            this.frees.add(address);
        }

        private List<Long> allocations() {
            return List.copyOf(this.allocations);
        }

        private List<Long> frees() {
            return List.copyOf(this.frees);
        }
    }
}
