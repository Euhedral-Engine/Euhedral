package io.euhedral_execution.inference.core.model.qwen38;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.model.qwen38.loader.DFlash2Config;
import io.euhedral_execution.inference.core.model.qwen38.speculative.DFlash2State;
import java.util.List;
import org.junit.jupiter.api.Test;

/// Sequence state that grows while a quantum runs frees its outgrown buffer in stream order and allocates the
/// larger one in stream order: nothing on the hot path frees device memory synchronously.
class GrowthInStreamOrderTest {

    private static void assertGrewInStreamOrder(ExecutionFixtures.RecordingGpu gpu, long outgrown, long grown) {
        assertTrue(gpu.asyncFrees.contains(outgrown), "the outgrown buffer is freed in stream order");
        assertEquals(gpu.asyncFrees, gpu.frees, "every free while running is stream-ordered");
        assertTrue(gpu.asyncAllocations.contains(grown), "the grown buffer is a stream-ordered allocation");
    }

    @Test
    void decodeScratchGrowsInStreamOrder() {
        var gpu = new ExecutionFixtures.RecordingGpu();
        var states = AttentionStates.allocate(gpu, new LayerType[] {LayerType.FULL_ATTENTION}, 256);
        long first = states.decodeScratch(4, 1);
        long grown = states.decodeScratch(4, 4);
        assertGrewInStreamOrder(gpu, first, grown);
    }

    @Test
    void draftSeedRowsGrowInStreamOrder() {
        var gpu = new ExecutionFixtures.RecordingGpu();
        var states = AttentionStates.allocate(gpu, new LayerType[] {LayerType.FULL_ATTENTION}, 256, true);
        long first = states.draftSeedRows(8, 64);
        long grown = states.draftSeedRows(16, 64);
        assertGrewInStreamOrder(gpu, first, grown);
    }

    @Test
    void speculativeGdnStateGrowsInStreamOrder() {
        var gpu = new ExecutionFixtures.RecordingGpu() {
            @Override
            public void zeroDeviceMemory(long address, long byteSize) {}
        };
        var state = GdnState.allocate(gpu, 2, 4, 16, 16, 4);
        long first = state.speculative(8, 64, 128, 4, 64, 64).queryKeyRows();
        long grown = state.speculative(16, 64, 128, 4, 64, 64).queryKeyRows();
        assertGrewInStreamOrder(gpu, first, grown);
    }

    @Test
    void dflash2TapsGrowInStreamOrder() {
        var gpu = new ExecutionFixtures.RecordingGpu();
        var config = new DFlash2Config(1, 64, 128, 2, 1, 32, 16, 0, 4, 1, 16, 8, 64, 32, 1e-6f, 10000f, new int[] {0});
        var state = new DFlash2State(gpu, config);
        long first = state.taps(8);
        long grown = state.taps(64);
        assertGrewInStreamOrder(gpu, first, grown);
        assertEquals(List.of(first), gpu.asyncFrees);
    }
}
