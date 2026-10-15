package io.euhedral_execution.inference.core.model.qwen38;

import static org.junit.jupiter.api.Assertions.assertEquals;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import org.junit.jupiter.api.Test;

class MappedHeadRoutingTest {

    @Test
    void deviceWeightsRunInOneCall() {
        assertEquals(1000, Stages.nvfp4RowsPerCall(false, 1000));
    }

    @Test
    void mappedWeightsNeverExceedThePlainLoadRowLimit() {
        int limit = ExecutionGpu.NVFP4_PLAIN_LOAD_MAX_ROWS;
        assertEquals(1, Stages.nvfp4RowsPerCall(true, 1));
        assertEquals(limit, Stages.nvfp4RowsPerCall(true, limit));
        assertEquals(limit, Stages.nvfp4RowsPerCall(true, limit + 1));
        assertEquals(limit, Stages.nvfp4RowsPerCall(true, 1000));
    }
}
