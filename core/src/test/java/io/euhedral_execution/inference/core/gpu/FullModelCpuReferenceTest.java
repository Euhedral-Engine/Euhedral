package io.euhedral_execution.inference.core.gpu;

import static io.euhedral_execution.inference.core.gpu.CudaGpuOperationsIntegrationTest.bf16ToFloat;
import static io.euhedral_execution.inference.core.gpu.CudaGpuOperationsIntegrationTest.floatToBf16;
import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

class FullModelCpuReferenceTest {

    @Test
    void nvfp4ReferenceSaturatesLargeFiniteValuesBeforeDistanceComparison() {
        for (float sign : new float[] {-1, 1}) {
            short[] row = new short[256];
            java.util.Arrays.fill(row, floatToBf16(sign * 1e30f));
            // A constant row maps to its DC coefficient. Saturation gives
            // E2M1 maximum * E4M3 maximum, then inverse H256 divides by 16.
            double expected = sign * 6.0 * 448.0 / 16.0;
            for (double value : Nvfp4KvReference.represented(row, 0)) assertEquals(expected, value);
        }
    }

    @Test
    void singleTokenAttentionMapsGroupedValuesAndAppliesPerQueryGate() {
        int queryHeads = 4;
        int keyValueHeads = 2;
        int headDim = 256;
        short[] gateValue = new short[(queryHeads + keyValueHeads) * headDim];
        for (int i = 0; i < queryHeads * headDim; i++) gateValue[i] = floatToBf16((i % 7) / 4.0f);
        // H256 maps each constant row to one coefficient; these scales/codes are exact.
        for (int h = 0; h < keyValueHeads; h++)
            java.util.Arrays.fill(
                    gateValue, (queryHeads + h) * headDim, (queryHeads + h + 1) * headDim, floatToBf16(3.0f * (h + 1)));

        short[] output =
                FullModelCpuReference.singleTokenAttentionContext(gateValue, queryHeads, keyValueHeads, headDim);

        for (int queryHead = 0; queryHead < queryHeads; queryHead++) {
            int valueHead = queryHead / (queryHeads / keyValueHeads);
            for (int lane = 0; lane < headDim; lane++) {
                float gate = bf16ToFloat(gateValue[queryHead * headDim + lane]);
                float value = bf16ToFloat(gateValue[queryHeads * headDim + valueHead * headDim + lane]);
                float expected = value / (1.0f + (float) Math.exp(-gate));
                assertEquals(expected, bf16ToFloat(output[queryHead * headDim + lane]), 0.06f);
            }
        }
    }
}
