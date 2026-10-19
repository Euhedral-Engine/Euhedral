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
            // A constant row maps to its DC coefficient, the largest of the first group: saturation gives the largest
            // magnitude times the largest E4M3 scale. The group's 15 zero coefficients take the smallest magnitude (the
            // midrise grid has no zero) at that scale, and the other groups are zero. Inverse H256: entry i sums the
            // coefficients j < 16 with signs (-1)^popcount(i & j), over 16.
            double[] represented = Nvfp4KvReference.represented(row, 0);
            for (int i = 0; i < 256; i++) {
                double sum = sign * Nvfp4KvReference.MAXIMUM * 448.0;
                for (int j = 1; j < 16; j++) sum += ((Integer.bitCount(i & j) & 1) == 0 ? 1 : -1) * 0.5 * 448.0;
                assertEquals(sum / 16, represented[i], 1e-9);
            }
        }
    }

    @Test
    void singleTokenAttentionMapsGroupedValuesAndAppliesPerQueryGate() {
        int queryHeads = 4;
        int keyValueHeads = 2;
        int headDim = 256;
        short[] gateValue = new short[(queryHeads + keyValueHeads) * headDim];
        for (int i = 0; i < queryHeads * headDim; i++) gateValue[i] = floatToBf16((i % 7) / 4.0f);
        for (int h = 0; h < keyValueHeads; h++)
            java.util.Arrays.fill(
                    gateValue, (queryHeads + h) * headDim, (queryHeads + h + 1) * headDim, floatToBf16(3.0f * (h + 1)));

        short[] output =
                FullModelCpuReference.singleTokenAttentionContext(gateValue, queryHeads, keyValueHeads, headDim);

        for (int queryHead = 0; queryHead < queryHeads; queryHead++) {
            int valueHead = queryHead / (queryHeads / keyValueHeads);
            // One key: the output is the value head's cached representation, gated.
            double[] cached = Nvfp4KvReference.represented(gateValue, queryHeads * headDim + valueHead * headDim);
            for (int lane = 0; lane < headDim; lane++) {
                float gate = bf16ToFloat(gateValue[queryHead * headDim + lane]);
                float expected = (float) cached[lane] / (1.0f + (float) Math.exp(-gate));
                assertEquals(expected, bf16ToFloat(output[queryHead * headDim + lane]), 0.06f);
            }
        }
    }
}
