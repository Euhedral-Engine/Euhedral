package io.euhedral_execution.inference.quantization.mixed;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.artifact.TensorDataType;
import io.euhedral_execution.inference.core.artifact.TensorDescriptor;
import io.euhedral_execution.inference.core.artifact.WeightFormat;
import io.euhedral_execution.inference.core.artifact.WeightLayout;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class CostTableTest {

    private static final Cost.Shape SHAPE = new Cost.Shape(64, 64);

    @TempDir
    Path directory;

    private Sources sources() throws Exception {
        return Sources.open(
                Map.of(
                        Option.TRELLIS_3_0, TinyArtifacts.write(directory.resolve("trellis.edrl"), 1, true),
                        Option.NVFP4, TinyArtifacts.write(directory.resolve("nvfp4.edrl"), 2, false)),
                Option.TRELLIS_3_0);
    }

    private static CostTable table() {
        Map<Cost.Key, Double> micros = new HashMap<>();
        micros.put(new Cost.Key(SHAPE, Option.NVFP4, 512), 51.2);
        micros.put(new Cost.Key(SHAPE, Option.TRELLIS_3_0, 512), 102.4);
        micros.put(new Cost.Key(SHAPE, Option.NVFP4, 4), 3.0);
        micros.put(new Cost.Key(SHAPE, Option.TRELLIS_3_0, 4), 5.0);
        return new CostTable(micros, 0, Map.of());
    }

    @Test
    void aKnownTableAndAllocationGiveExactSums() throws Exception {
        Sources sources = sources();
        // four projections: two on NVFP4, two left on the base (trellis 3.0)
        Allocation allocation = new Allocation(Map.of(
                "text/layers/0/mlp/down", Option.NVFP4,
                "text/layers/1/mlp/down", Option.NVFP4));
        assertEquals(2 * 51.2 / 512 + 2 * 102.4 / 512, table().prefillMicrosPerToken(sources, allocation), 1e-12);
        assertEquals(2 * 3.0 + 2 * 5.0, table().verifyMicros(sources, allocation, 4), 1e-12);
    }

    @Test
    void fixedTermsAreAddedOnce() throws Exception {
        Sources sources = sources();
        Allocation allocation = new Allocation(Map.of());
        CostTable fixed = table().withFixed(7.5, Map.of(4, 11.0));
        assertEquals(4 * 102.4 / 512 + 7.5, fixed.prefillMicrosPerToken(sources, allocation), 1e-12);
        assertEquals(4 * 5.0 + 11.0, fixed.verifyMicros(sources, allocation, 4), 1e-12);
        // the original is unchanged
        assertEquals(4 * 5.0, table().verifyMicros(sources, allocation, 4), 1e-12);
    }

    @Test
    void aMissingEntryIsRefusedNamingIt() throws Exception {
        Sources sources = sources();
        IllegalArgumentException refused = assertThrows(
                IllegalArgumentException.class, () -> table().verifyMicros(sources, new Allocation(Map.of()), 3));
        assertTrue(refused.getMessage().contains("trellis-3.0"), refused.getMessage());
        assertTrue(refused.getMessage().contains("64"), refused.getMessage());
        assertTrue(refused.getMessage().contains("3"), refused.getMessage());
    }

    @Test
    void theTableRoundTripsThroughJson() throws Exception {
        CostTable fixed = table().withFixed(7.5, Map.of(4, 11.0, 1, 2.5));
        Path file = directory.resolve("costs.json");
        fixed.write(file);
        CostTable read = CostTable.read(file);
        assertEquals(51.2, read.micros(new Cost.Key(SHAPE, Option.NVFP4, 512)), 0);
        assertEquals(5.0, read.micros(new Cost.Key(SHAPE, Option.TRELLIS_3_0, 4)), 0);
        Sources sources = sources();
        Allocation allocation = new Allocation(Map.of());
        assertEquals(
                fixed.prefillMicrosPerToken(sources, allocation), read.prefillMicrosPerToken(sources, allocation), 0);
        assertEquals(fixed.verifyMicros(sources, allocation, 4), read.verifyMicros(sources, allocation, 4), 0);
    }

    @Test
    void featuresComeFromTheStoredShapeOfEachFormat() {
        TensorDescriptor trellis = new TensorDescriptor(
                "t",
                new long[] {320, 1120, 56},
                TensorDataType.BF16,
                WeightFormat.TRELLIS,
                WeightLayout.TRELLIS_TILE16_V1,
                0,
                0);
        assertEquals(new Cost.Shape(5120, 17920), Cost.shape(trellis));
        TensorDescriptor rows = new TensorDescriptor(
                "n",
                new long[] {17920, 5120},
                TensorDataType.BF16,
                WeightFormat.NVFP4,
                WeightLayout.ROW_SPLIT_K128_V1,
                0,
                0);
        assertEquals(new Cost.Shape(5120, 17920), Cost.shape(rows));
    }
}
