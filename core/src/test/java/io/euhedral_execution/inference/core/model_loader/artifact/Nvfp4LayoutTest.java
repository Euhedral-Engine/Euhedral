package io.euhedral_execution.inference.core.model_loader.artifact;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.euhedral_execution.inference.core.model_loader.layer_weights.TensorDataType;
import io.euhedral_execution.inference.core.model_loader.layer_weights.WeightFormat;
import io.euhedral_execution.inference.core.model_loader.layer_weights.WeightLayout;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import org.junit.jupiter.api.Test;

class Nvfp4LayoutTest {

    private static final ValueLayout.OfFloat GLOBAL =
            ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

    @Test
    void geometryFollowsTheConverter() {
        // tools/convert_qwen_safetensors_to_compact_edrl.py nvfp4_offsets((3, 1024)) == (1536, 1792, 1796)
        assertEquals(1536, Nvfp4Layout.scaleOffset(3, 1024));
        assertEquals(1792, Nvfp4Layout.globalScaleOffset(3, 1024));
        assertEquals(1796, Nvfp4Layout.byteSize(3, 1024));
        assertEquals(
                1796,
                CompactTensorLayout.expectedByteSize(
                        new long[] {3, 1024}, TensorDataType.BF16, WeightFormat.NVFP4, WeightLayout.ROW_SPLIT_K128_V1));
        assertEquals(Nvfp4Layout.byteSize(2, 1088), Nvfp4Layout.byteSize(2, 1152));
    }

    @Test
    void validTensorsPassAndNanScalesFail() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment tensor = arena.allocate(Nvfp4Layout.byteSize(4, 1024));
            tensor.set(GLOBAL, Nvfp4Layout.globalScaleOffset(4, 1024), 1.5e-4f);
            assertDoesNotThrow(() -> Nvfp4Layout.validate(tensor, 4, 1024));
            long scale = Nvfp4Layout.scaleOffset(4, 1024) + 37;
            tensor.set(ValueLayout.JAVA_BYTE, scale, (byte) 0x7e);
            assertDoesNotThrow(() -> Nvfp4Layout.validate(tensor, 4, 1024));
            tensor.set(ValueLayout.JAVA_BYTE, scale, (byte) 0xff);
            assertThrows(IllegalArgumentException.class, () -> Nvfp4Layout.validate(tensor, 4, 1024));
        }
    }

    @Test
    void nonFiniteOrNegativeGlobalScalesAndWrongSizesFail() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment tensor = arena.allocate(Nvfp4Layout.byteSize(2, 1024));
            long global = Nvfp4Layout.globalScaleOffset(2, 1024);
            for (float value : new float[] {Float.NaN, Float.POSITIVE_INFINITY, -1.0f}) {
                tensor.set(GLOBAL, global, value);
                assertThrows(IllegalArgumentException.class, () -> Nvfp4Layout.validate(tensor, 2, 1024));
            }
            tensor.set(GLOBAL, global, 0.0f);
            assertThrows(IllegalArgumentException.class, () -> Nvfp4Layout.validate(tensor, 3, 1024));
        }
    }
}
