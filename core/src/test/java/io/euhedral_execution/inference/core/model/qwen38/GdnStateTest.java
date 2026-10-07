package io.euhedral_execution.inference.core.model.qwen38;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import java.lang.foreign.MemorySegment;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

class GdnStateTest {

    @Test
    void allocatesAndZerosOnlyTheSequenceOwnedGdnStateThenClosesIdempotently() throws Exception {
        TrackingGpu gpu = new TrackingGpu();
        Class<?> stateType = stateType();
        Object state = allocator(stateType).invoke(null, gpu, 16, 48, 128, 128, 4);
        long convolutionState =
                (long) stateType.getMethod("convolutionStateAddress").invoke(state);
        long recurrentState =
                (long) stateType.getMethod("recurrentStateAddress").invoke(state);

        assertEquals(List.of(61_440L, 3_145_728L), gpu.allocationSizes);
        assertEquals(List.of(convolutionState, recurrentState), gpu.zeroedAddresses);
        assertTrue(convolutionState != recurrentState);
        ((AutoCloseable) state).close();
        assertEquals(List.of(convolutionState, recurrentState), gpu.freedAddresses);
        assertThrows(
                InvocationTargetException.class,
                () -> stateType.getMethod("recurrentStateAddress").invoke(state));
    }

    @Test
    void freesTheFirstAllocationWhenTheSecondSequenceStateAllocationFails() {
        TrackingGpu gpu = new TrackingGpu();
        gpu.failAllocation = 2;
        Class<?> stateType = stateType();
        InvocationTargetException failure = assertThrows(
                InvocationTargetException.class, () -> allocator(stateType).invoke(null, gpu, 16, 48, 128, 128, 4));

        assertTrue(failure.getCause() instanceof IllegalStateException);
        assertEquals(List.of(0x2000L), gpu.freedAddresses);
        assertTrue(gpu.zeroedAddresses.isEmpty());
    }

    private static Class<?> stateType() {
        return assertDoesNotThrow(() -> Class.forName("io.euhedral_execution.inference.core.model.qwen38.GdnState"));
    }

    private static Method allocator(Class<?> stateType) {
        return assertDoesNotThrow(() -> stateType.getMethod(
                "allocate", ExecutionGpu.class, int.class, int.class, int.class, int.class, int.class));
    }

    private static final class TrackingGpu extends ExecutionGpu {
        private final List<Long> allocationSizes = new ArrayList<>();
        private final List<Long> freedAddresses = new ArrayList<>();
        private final List<Long> zeroedAddresses = new ArrayList<>();
        private int failAllocation;

        @Override
        public long allocate(long byteSize) {
            this.allocationSizes.add(byteSize);
            if (this.failAllocation == this.allocationSizes.size()) {
                throw new IllegalStateException("injected allocation failure");
            }
            return 0x1000L + this.allocationSizes.size() * 0x1000L;
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
            this.freedAddresses.add(address);
        }

        @Override
        public void embedQ3(
                long tokenIdsAddress,
                long embeddingAddress,
                long embeddingByteSize,
                long hiddenStateAddress,
                int tokenCount,
                int vocabularySize,
                int hiddenSize) {
            throw new UnsupportedOperationException();
        }

        @Override
        public void synchronize() {
            throw new UnsupportedOperationException();
        }

        @Override
        public void zeroDeviceMemory(long address, long byteSize) {
            this.zeroedAddresses.add(address);
        }
    }
}
