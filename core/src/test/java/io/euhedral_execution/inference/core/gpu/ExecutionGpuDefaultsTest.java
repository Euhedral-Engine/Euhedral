package io.euhedral_execution.inference.core.gpu;

import static org.junit.jupiter.api.Assertions.assertEquals;

import io.euhedral_execution.inference.core.model.qwen38.ExecutionFixtures;
import java.util.List;
import org.junit.jupiter.api.Test;

class ExecutionGpuDefaultsTest {
    @Test
    void streamOrderedAllocationDefaultsToOrdinaryAllocation() {
        var gpu = new ExecutionFixtures.RecordingGpu();
        long address = gpu.allocateAsync(256);
        gpu.freeAsync(address);
        assertEquals(List.of(address), gpu.frees);
    }
}
