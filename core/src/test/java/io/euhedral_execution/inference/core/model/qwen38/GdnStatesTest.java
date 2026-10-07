package io.euhedral_execution.inference.core.model.qwen38;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

class GdnStatesTest {

    @Test
    void allocatesOnlyGdnLayerStateAndReleasesEveryLayer() {
        TrackingGpu gpu = new TrackingGpu();
        GdnStates states = GdnStates.allocate(
                gpu,
                new LayerType[] {LayerType.GATED_DELTA_NET, LayerType.FULL_ATTENTION, LayerType.GATED_DELTA_NET},
                16,
                48,
                128,
                128,
                4);

        assertTrue(states.forLayer(0) != states.forLayer(2));
        assertThrows(IllegalArgumentException.class, () -> states.forLayer(1));
        assertEquals(4, gpu.allocations.size());

        states.close();

        assertEquals(4, gpu.frees.size());
        assertThrows(IllegalStateException.class, () -> states.forLayer(0));
    }

    private static final class TrackingGpu extends ExecutionGpu {
        private final List<Long> allocations = new ArrayList<>();
        private final List<Long> frees = new ArrayList<>();
        private long nextAddress = 0x1000;

        @Override
        public long allocate(long byteSize) {
            allocations.add(byteSize);
            return nextAddress += 0x1000;
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
            frees.add(address);
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
        public void zeroDeviceMemory(long address, long byteSize) {}
    }
}
