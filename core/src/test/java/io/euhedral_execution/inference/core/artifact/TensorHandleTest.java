package io.euhedral_execution.inference.core.artifact;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class TensorHandleTest {

    @Test
    void aDiskHandleKnowsItsFileOffsetAndIsNotHostBacked() {
        var descriptor = new TensorDescriptor(
                "text/layers/0/mlp/down",
                new long[] {5120, 17408},
                TensorDataType.BF16,
                WeightFormat.Q3_G64_FP16,
                WeightLayout.ROW_SPLIT_K128_V1,
                1L << 30,
                4096);
        TensorHandle disk = TensorHandle.disk(descriptor);
        assertTrue(disk.diskBacked());
        assertFalse(disk.hostBacked());
        assertEquals(1L << 30, disk.fileOffset());
        assertEquals(4096, disk.byteSize());
    }

    @Test
    void otherHandlesAreNotDiskBacked() {
        assertFalse(new TensorHandle("x", new long[] {1}, TensorDataType.BF16, WeightFormat.BF16, 16, 2).diskBacked());
        assertFalse(new TensorHandle(
                        "y",
                        new long[] {1},
                        TensorDataType.BF16,
                        WeightFormat.BF16,
                        WeightLayout.CONTIGUOUS_LE_V1,
                        16,
                        2,
                        4096,
                        true)
                .diskBacked());
    }
}
