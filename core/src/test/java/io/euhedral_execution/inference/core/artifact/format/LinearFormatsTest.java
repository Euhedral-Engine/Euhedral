package io.euhedral_execution.inference.core.artifact.format;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.artifact.CompactTensorLayout;
import io.euhedral_execution.inference.core.artifact.Nvfp4Layout;
import io.euhedral_execution.inference.core.artifact.TensorDataType;
import io.euhedral_execution.inference.core.artifact.TensorDescriptor;
import io.euhedral_execution.inference.core.artifact.TensorHandle;
import io.euhedral_execution.inference.core.artifact.WeightFormat;
import io.euhedral_execution.inference.core.artifact.WeightLayout;
import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.gpu.ScratchUse;
import io.euhedral_execution.inference.core.model.qwen38.ArtifactProfile;
import io.euhedral_execution.inference.core.model.qwen38.ArtifactProfile.Quantization;
import io.euhedral_execution.inference.core.model.qwen38.ExecutionPlan.Kind;
import io.euhedral_execution.inference.core.model.qwen38.artifact.Artifact;
import io.euhedral_execution.inference.core.model.qwen38.loader.HostWeightSelection;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;

/// Each registered format answers what the scattered switches answered before the registry: the rows below are
/// the old rules restated, and every registered key is checked against them.
class LinearFormatsTest {

    private static final WeightFormat Q3 = WeightFormat.Q3_G64_FP16;
    private static final WeightFormat Q4 = WeightFormat.Q4_G64_FP16;
    private static final WeightFormat Q5 = WeightFormat.Q5_G64_FP16;
    private static final WeightFormat NVFP4 = WeightFormat.NVFP4;
    private static final WeightFormat BF16 = WeightFormat.BF16;
    private static final WeightLayout K128 = WeightLayout.ROW_SPLIT_K128_V1;
    private static final WeightLayout P2E2 = WeightLayout.ROW_SPLIT_P2E2_V1;
    private static final WeightLayout SD4 = WeightLayout.ROW_SPLIT_K128_SD4_V1;
    private static final WeightLayout CONTIGUOUS = WeightLayout.CONTIGUOUS_LE_V1;

    /// One registered key and the old rules' answers for it.
    private record Row(
            String label,
            WeightFormat format,
            WeightLayout layout,
            boolean stageable,
            Quantization quantization,
            boolean compressed,
            boolean gateUpRegion,
            boolean splitsRows) {}

    private static final List<Row> ROWS = List.of(
            new Row("q3", Q3, K128, true, Quantization.Q3, false, true, false),
            new Row("q4", Q4, K128, true, Quantization.Q3, false, false, false),
            new Row("q5", Q5, K128, true, Quantization.Q3, false, false, false),
            new Row("p2e2", Q3, P2E2, false, Quantization.Q3, true, true, false),
            new Row("nvfp4", NVFP4, K128, true, Quantization.NVFP4, false, true, true),
            new Row("sd4", NVFP4, SD4, true, Quantization.NVFP4, true, true, true),
            new Row("bf16", BF16, CONTIGUOUS, true, Quantization.BF16, false, false, true));

    private static final List<Kind> KINDS = List.of(
            Kind.Q3_LINEAR,
            Kind.Q4_LINEAR,
            Kind.Q5_LINEAR,
            Kind.BF16_PROJECTION,
            Kind.BF16_LINEAR,
            Kind.Q3_GATE_UP_SWIGLU,
            Kind.DFLASH_LINEAR,
            Kind.DFLASH_LM_HEAD,
            Kind.RMS_NORM);

    private static LinearFormat format(Row row) {
        return LinearFormats.find(row.format(), row.layout()).orElseThrow();
    }

    @Test
    void theRegistryHoldsExactlyTheSevenKeys() {
        assertEquals(ROWS.size(), LinearFormats.all().size());
        for (Row row : ROWS) {
            LinearFormat format = format(row);
            assertEquals(row.format(), format.format(), row.label());
            assertEquals(row.layout(), format.layout(), row.label());
        }
        assertTrue(LinearFormats.find(WeightFormat.FP16, CONTIGUOUS).isEmpty());
        assertTrue(LinearFormats.find(WeightFormat.FP32, CONTIGUOUS).isEmpty());
        assertTrue(
                LinearFormats.find(NVFP4, WeightLayout.ROW_INTERLEAVED_NVFP4_V1).isEmpty());
        assertTrue(LinearFormats.find(Q4, P2E2).isEmpty());
        assertTrue(LinearFormats.find(BF16, K128).isEmpty());
        var message = assertThrows(
                        IllegalArgumentException.class,
                        () -> LinearFormats.of(handle("text/x", WeightFormat.FP32, CONTIGUOUS, 8, 8, 4)))
                .getMessage();
        assertEquals("unsupported linear weight format FP32/CONTIGUOUS_LE_V1: text/x", message);
        assertTrue(LinearFormats.quantized(Q3));
        assertTrue(LinearFormats.quantized(NVFP4));
        assertFalse(LinearFormats.quantized(BF16));
        assertFalse(LinearFormats.quantized(WeightFormat.FP16));
    }

    @Test
    void storageAnswersMatchTheOldSizingStageabilityAndProfile() {
        long[][] shapes = {{64, 128}, {128, 5120}, {17408, 5120}, {4096, 1024}, {7, 200}};
        for (Row row : ROWS) {
            LinearFormat format = format(row);
            assertEquals(row.stageable(), format.stageable(), row.label());
            assertEquals(row.quantization(), format.quantization(), row.label());
            assertEquals(row.compressed(), format.compressed(), row.label());
            assertEquals(row.splitsRows(), format.splitsRows(), row.label());
            assertEquals(List.of(), format.companions("text/layers/0/mlp/gate_up"));
            assertEquals(0, format.planScratchBytes(descriptor("n", row.format(), row.layout(), 64, 128, 0), 8));
            // The old public entry point for stageability.
            assertEquals(
                    row.stageable(),
                    HostWeightSelection.stageable(
                            descriptor("text/layers/3/mlp/gate_up", row.format(), row.layout(), 64, 128, 0)),
                    row.label());
            assertFalse(
                    HostWeightSelection.stageable(
                            descriptor("text/layers/3/mlp/unknown", row.format(), row.layout(), 64, 128, 0)),
                    row.label());
            for (long[] shape : shapes) {
                if (row.layout() == P2E2) {
                    var failure = assertThrows(
                            IllegalArgumentException.class, () -> format.byteSize(shape, TensorDataType.BF16));
                    assertEquals("P2E2 tensor sizes depend on their codes", failure.getMessage());
                    continue;
                }
                long expected = oldSize(row, shape);
                assertEquals(expected, format.byteSize(shape, TensorDataType.BF16), row.label());
                assertEquals(
                        expected,
                        CompactTensorLayout.expectedByteSize(shape, TensorDataType.BF16, row.format(), row.layout()));
                assertTrue(format.acceptsByteSize(shape, TensorDataType.BF16, expected));
                assertFalse(format.acceptsByteSize(shape, TensorDataType.BF16, expected + 1));
                assertTrue(CompactTensorLayout.acceptsByteSize(
                        shape, TensorDataType.BF16, row.format(), row.layout(), expected));
            }
        }
    }

    @Test
    void sizingKeepsItsMessages() {
        assertEquals(
                "grouped compact storage requires BF16 source metadata",
                assertThrows(
                                IllegalArgumentException.class,
                                () -> CompactTensorLayout.expectedByteSize(
                                        new long[] {64, 128}, TensorDataType.FP16, Q3, K128))
                        .getMessage());
        assertEquals(
                "row-split layout requires a positive rank-2 shape",
                assertThrows(
                                IllegalArgumentException.class,
                                () -> CompactTensorLayout.expectedByteSize(
                                        new long[] {64}, TensorDataType.BF16, NVFP4, K128))
                        .getMessage());
        assertEquals(
                "SD4 layout requires NVFP4 storage of BF16 source metadata",
                assertThrows(
                                IllegalArgumentException.class,
                                () -> CompactTensorLayout.expectedByteSize(
                                        new long[] {64, 128}, TensorDataType.FP16, NVFP4, SD4))
                        .getMessage());
        assertEquals(
                "unsupported row-split format FP16",
                assertThrows(
                                IllegalArgumentException.class,
                                () -> CompactTensorLayout.expectedByteSize(
                                        new long[] {64, 128}, TensorDataType.BF16, WeightFormat.FP16, K128))
                        .getMessage());
        assertEquals(
                "interleaved NVFP4 rows belong to version 3 artifacts",
                assertThrows(
                                IllegalArgumentException.class,
                                () -> CompactTensorLayout.expectedByteSize(
                                        new long[] {64, 128},
                                        TensorDataType.BF16,
                                        NVFP4,
                                        WeightLayout.ROW_INTERLEAVED_NVFP4_V1))
                        .getMessage());
        assertEquals(
                "compact descriptor metadata is incomplete",
                assertThrows(
                                IllegalArgumentException.class,
                                () -> CompactTensorLayout.expectedByteSize(null, TensorDataType.BF16, Q3, K128))
                        .getMessage());
        assertEquals(
                "P2E2 layout requires Q3 BF16 metadata and a rank-2 shape whose K is a multiple of 1024",
                assertThrows(
                                IllegalArgumentException.class,
                                () -> CompactTensorLayout.acceptsByteSize(
                                        new long[] {64, 100}, TensorDataType.BF16, Q3, P2E2, 10))
                        .getMessage());
        assertEquals(
                "P2E2 layout requires Q3 BF16 metadata and a rank-2 shape whose K is a multiple of 1024",
                assertThrows(
                                IllegalArgumentException.class,
                                () -> CompactTensorLayout.acceptsByteSize(
                                        new long[] {64, 1024}, TensorDataType.BF16, Q4, P2E2, 10))
                        .getMessage());
    }

    @Test
    void acceptanceMatchesTheOldRuleForEveryDescriptorAndExpectedObject() {
        List<WeightFormat> formats = List.of(Q3, Q4, Q5, NVFP4, BF16, WeightFormat.FP32);
        List<WeightLayout> layouts = List.of(K128, P2E2, SD4, CONTIGUOUS);
        List<long[]> shapes = List.of(new long[] {64, 128}, new long[] {64, 256});
        List<TensorDataType> types = List.of(TensorDataType.BF16, TensorDataType.FP16);
        int compared = 0;
        for (Row row : ROWS) {
            LinearFormat format = format(row);
            for (long[] descriptorShape : shapes)
                for (TensorDataType descriptorType : types) {
                    TensorDescriptor descriptor = new TensorDescriptor(
                            "n", descriptorShape, descriptorType, row.format(), row.layout(), 0, 1);
                    for (WeightFormat expectedFormat : formats)
                        for (WeightLayout expectedLayout : layouts)
                            for (long[] expectedShape : shapes) {
                                ExpectedTensor expected = new ExpectedTensor(
                                        expectedShape, TensorDataType.BF16, expectedFormat, expectedLayout);
                                assertEquals(
                                        oldAccepts(expected, descriptor),
                                        format.accepts(descriptor, expected),
                                        row.label() + " vs " + expectedFormat + "/" + expectedLayout);
                                compared++;
                            }
                }
        }
        assertTrue(compared > 1000);
    }

    @Test
    void theProfileComesFromTheFormat() {
        for (Row row : ROWS) {
            var artifact = new Artifact(null, null, new TensorDescriptor[] {
                descriptor("text/layers/0/mlp/gate_up", row.format(), row.layout(), 1, 1, 1)
            });
            // Only Q3 names a grouped artifact: Q4 and Q5 are widths inside one.
            if (row.format() == Q4 || row.format() == Q5) continue;
            var profile = ArtifactProfile.of(artifact);
            assertEquals(row.quantization(), profile.quantization(), row.label());
            assertEquals(row.compressed(), profile.compressed(), row.label());
        }
        assertEquals(
                "unsupported projection format FP16; expected Q3, NVFP4 or BF16",
                assertThrows(
                                IllegalArgumentException.class,
                                () -> ArtifactProfile.of(new Artifact(null, null, new TensorDescriptor[] {
                                    descriptor("text/layers/0/mlp/gate_up", WeightFormat.FP16, CONTIGUOUS, 1, 1, 1)
                                })))
                        .getMessage());
        assertEquals(
                "layout ROW_SPLIT_K128_V1 does not belong to BF16",
                assertThrows(
                                IllegalArgumentException.class,
                                () -> ArtifactProfile.of(new Artifact(null, null, new TensorDescriptor[] {
                                    descriptor("text/layers/0/mlp/gate_up", BF16, K128, 1, 1, 1)
                                })))
                        .getMessage());
    }

    @Test
    void kindsFollowTheFormatAndKeepTheRoleWhereTheOldCodeDid() {
        for (Row row : ROWS) {
            LinearFormat format = format(row);
            for (Kind role : List.of(Kind.Q3_LINEAR, Kind.Q4_LINEAR, Kind.Q5_LINEAR))
                assertEquals(oldKind(row.format(), role), format.kind(role), row.label() + " " + role);
            assertEquals(row.gateUpRegion(), format.gateUpRegion(), row.label());
        }
    }

    @Test
    void rowExactnessFollowsTheGpuFlagForQuantizedKindsOnly() {
        for (boolean flag : new boolean[] {true, false}) {
            Gpu gpu = new Gpu(flag);
            for (Row row : ROWS)
                for (Kind kind : KINDS) {
                    // A BF16 object never takes a grouped kind: its role maps to BF16_PROJECTION.
                    boolean groupedKind = kind == Kind.Q3_LINEAR || kind == Kind.Q4_LINEAR || kind == Kind.Q5_LINEAR;
                    if (row.format() == BF16 && groupedKind) continue;
                    assertEquals(
                            oldRowExact(row.format(), kind, flag),
                            format(row).rowExact(gpu, kind),
                            row.label() + " " + kind + " " + flag);
                }
        }
    }

    @Test
    void scratchUseMatchesTheOldRoutes() {
        for (Row row : ROWS)
            for (Kind kind : KINDS)
                for (boolean multiRow : new boolean[] {false, true})
                    assertEquals(
                            oldScratchUse(row.format() == NVFP4, kind, multiRow),
                            format(row).scratchUse(kind, multiRow),
                            row.label() + " " + kind + " " + multiRow);
    }

    @Test
    void scratchBytesAskTheGpuWithTheWeightsLayout() {
        for (Row row : ROWS) {
            Gpu gpu = new Gpu(false);
            TensorHandle weight = handle("w", row.format(), row.layout(), 64, 128, 16);
            long bytes = format(row).scratchBytes(gpu, ScratchUse.Q3_LINEAR, 9, 128, 64, List.of(weight));
            assertEquals(1, gpu.scratchQueries.size());
            assertEquals(row.layout() + ":9:128:64", gpu.scratchQueries.getFirst());
            assertEquals(4242, bytes);
        }
    }

    @Test
    void launchesRouteToTheOldKernels() {
        Gpu gpu = new Gpu(false);
        launch(gpu, Q3, K128, Kind.Q3_LINEAR, 3, false);
        launch(gpu, Q3, P2E2, Kind.Q3_LINEAR, 3, false);
        launch(gpu, Q4, K128, Kind.Q4_LINEAR, 3, false);
        launch(gpu, Q5, K128, Kind.Q5_LINEAR, 3, false);
        launch(gpu, BF16, CONTIGUOUS, Kind.BF16_LINEAR, 3, false);
        assertEquals(
                List.of(
                        "q3:ROW_SPLIT_K128_V1:3:in100:out200",
                        "q3:ROW_SPLIT_P2E2_V1:3:in100:out200",
                        "q4:3",
                        "q5:3",
                        "bf16float:3"),
                gpu.calls);
    }

    @Test
    void anNvfp4LinearOnHostMappedWeightsRunsInCallsOfAtMostThePlainLoadLimit() {
        int limit = ExecutionGpu.NVFP4_PLAIN_LOAD_MAX_ROWS;
        Gpu resident = new Gpu(false);
        launch(resident, NVFP4, K128, Kind.Q3_LINEAR, 2 * limit + 8, false);
        assertEquals(List.of("nvfp4:" + (2 * limit + 8) + ":in100:out200"), resident.calls);

        Gpu mapped = new Gpu(false);
        launch(mapped, NVFP4, K128, Kind.Q4_LINEAR, 2 * limit + 8, true);
        long inputStep = 128L * Short.BYTES * limit, outputStep = 64L * Short.BYTES * limit;
        assertEquals(
                List.of(
                        "nvfp4:" + limit + ":in" + 100 + ":out" + 200,
                        "nvfp4:" + limit + ":in" + (100 + inputStep) + ":out" + (200 + outputStep),
                        "nvfp4:8:in" + (100 + 2 * inputStep) + ":out" + (200 + 2 * outputStep)),
                mapped.calls);
    }

    @Test
    void gateUpRegionsLaunchTheirFormatsKernel() {
        Gpu gpu = new Gpu(false);
        for (Row row : ROWS) {
            if (!row.gateUpRegion()) continue;
            TensorHandle weight = handle("w", row.format(), row.layout(), 64, 128, 16);
            format(row).gateUpSwiGlu(gpu, launch(Kind.Q3_GATE_UP_SWIGLU, weight, 5, false));
        }
        assertEquals(
                List.of(
                        "q3gateup:ROW_SPLIT_K128_V1:5",
                        "q3gateup:ROW_SPLIT_P2E2_V1:5",
                        "nvfp4gateup:5",
                        "nvfp4gateup:5"),
                gpu.calls);
    }

    @Test
    void thePackedMtpProjectionSplitsOnlyWhereTheFormatCutsRows() {
        for (Row row : ROWS) assertEquals(row.splitsRows(), format(row).splitsRows(), row.label());
        assertThrows(
                UnsupportedOperationException.class,
                () -> format(ROWS.getFirst()).rows(List.of(), 0, 1, "n", new Gpu(false)));
    }

    // The old rules, restated.

    private static long oldSize(Row row, long[] shape) {
        if (row.format() == BF16) return shape[0] * shape[1] * 2;
        if (row.format() == NVFP4) return Nvfp4Layout.byteSize(shape[0], shape[1], row.layout());
        int bits = row.format() == Q3 ? 3 : row.format() == Q4 ? 4 : 5;
        long groups = (shape[1] + 127) / 128 * 128 / 64;
        long base = shape[0] * groups * (bits == 3 ? 24 : 32);
        long high = shape[0] * groups * (bits <= 4 ? 0 : 8);
        long highOffset = (base + 255) / 256 * 256;
        long scaleOffset = highOffset + (high + 255) / 256 * 256;
        return scaleOffset + shape[0] * groups * 2;
    }

    private static boolean grouped(WeightFormat format) {
        return format == Q3 || format == Q4 || format == Q5;
    }

    private static boolean oldAccepts(ExpectedTensor expected, TensorDescriptor descriptor) {
        boolean p2e2 = descriptor.layout() == P2E2 && descriptor.format() == Q3;
        boolean sd4 = descriptor.layout() == SD4 && descriptor.format() == NVFP4;
        boolean bf16 = descriptor.format() == BF16
                && descriptor.layout() == CONTIGUOUS
                && (grouped(expected.format()) || expected.format() == NVFP4);
        if (bf16)
            return Arrays.equals(expected.shape(), descriptor.shape()) && expected.dataType() == descriptor.dataType();
        return !(!Arrays.equals(expected.shape(), descriptor.shape())
                || expected.dataType() != descriptor.dataType()
                || !(expected.format() == descriptor.format()
                        || (descriptor.format() == NVFP4 && expected.layout() == K128)
                        || (grouped(expected.format()) && grouped(descriptor.format())))
                || !(expected.layout() == descriptor.layout() || ((p2e2 || sd4) && expected.layout() == K128)));
    }

    private static Kind oldKind(WeightFormat format, Kind role) {
        return switch (format) {
            case Q3_G64_FP16 -> Kind.Q3_LINEAR;
            case Q4_G64_FP16 -> Kind.Q4_LINEAR;
            case Q5_G64_FP16 -> Kind.Q5_LINEAR;
            case BF16 -> Kind.BF16_PROJECTION;
            default -> role;
        };
    }

    private static boolean oldRowExact(WeightFormat format, Kind kind, boolean flag) {
        boolean quantized = kind == Kind.Q3_LINEAR || kind == Kind.Q4_LINEAR || kind == Kind.Q5_LINEAR;
        // An NVFP4 projection launches once over all its rows; the grouped ones follow the GPU's flag.
        if (format == NVFP4) return quantized || kind == Kind.BF16_PROJECTION;
        return (quantized && flag) || kind == Kind.BF16_PROJECTION;
    }

    private static ScratchUse oldScratchUse(boolean nvfp4, Kind kind, boolean multiRow) {
        if (kind == Kind.Q3_GATE_UP_SWIGLU) return nvfp4 ? ScratchUse.NVFP4_GATE_UP : ScratchUse.Q3_GATE_UP;
        if (!multiRow) return null;
        return switch (kind) {
            case Q3_LINEAR -> nvfp4 ? ScratchUse.NVFP4_LINEAR : ScratchUse.Q3_LINEAR;
            case Q4_LINEAR, Q5_LINEAR -> nvfp4 ? ScratchUse.NVFP4_LINEAR : ScratchUse.MX_LINEAR;
            case BF16_PROJECTION -> null;
            case DFLASH_LINEAR -> nvfp4 ? ScratchUse.NVFP4_LINEAR : null;
            case DFLASH_LM_HEAD -> nvfp4 ? ScratchUse.NVFP4_LINEAR : ScratchUse.Q3_LINEAR;
            default -> null;
        };
    }

    // Fixtures.

    private static TensorDescriptor descriptor(
            String name, WeightFormat format, WeightLayout layout, long rows, long columns, long bytes) {
        return new TensorDescriptor(name, new long[] {rows, columns}, TensorDataType.BF16, format, layout, 0, bytes);
    }

    private static TensorHandle handle(
            String name, WeightFormat format, WeightLayout layout, long rows, long columns, long bytes) {
        return new TensorHandle(name, new long[] {rows, columns}, TensorDataType.BF16, format, layout, 4096, bytes);
    }

    private static LinearLaunch launch(Kind kind, TensorHandle weight, int rows, boolean hostMapped) {
        return new LinearLaunch(kind, List.of(weight), 100, 200, rows, 128, 64, 0, 0, false, hostMapped);
    }

    private static void launch(Gpu gpu, WeightFormat format, WeightLayout layout, Kind kind, int rows, boolean mapped) {
        TensorHandle weight = handle("w", format, layout, 64, 128, 16);
        LinearFormats.find(format, layout).orElseThrow().linear(gpu, launch(kind, weight, rows, mapped));
    }

    /// Records the kernels a format launches and the scratch sizes it asks for.
    private static final class Gpu extends ExecutionGpu {
        final boolean rowExactQuantized;
        final List<String> calls = new ArrayList<>();
        final List<String> scratchQueries = new ArrayList<>();

        Gpu(boolean rowExactQuantized) {
            this.rowExactQuantized = rowExactQuantized;
        }

        @Override
        public boolean rowExactQuantizedLinears() {
            return this.rowExactQuantized;
        }

        @Override
        public long scratchBytes(ScratchUse use, int rows, int inFeatures, int outFeatures, WeightLayout layout) {
            this.scratchQueries.add(layout + ":" + rows + ":" + inFeatures + ":" + outFeatures);
            return 4242;
        }

        @Override
        public void linearQ3Bf16(
                long input,
                long weights,
                long output,
                int rows,
                int inFeatures,
                int outFeatures,
                long weightsByteSize,
                WeightLayout layout) {
            this.calls.add("q3:" + layout + ":" + rows + ":in" + input + ":out" + output);
        }

        @Override
        public void linearQ4Bf16(
                long input, long weights, long output, int rows, int inFeatures, int outFeatures, long bytes) {
            this.calls.add("q4:" + rows);
        }

        @Override
        public void linearQ5Bf16(
                long input, long weights, long output, int rows, int inFeatures, int outFeatures, long bytes) {
            this.calls.add("q5:" + rows);
        }

        @Override
        public void linearNvfp4Bf16(
                long input, long weights, long output, int rows, int inFeatures, int outFeatures, long bytes) {
            this.calls.add("nvfp4:" + rows + ":in" + input + ":out" + output);
        }

        @Override
        public void linearBf16ToFloat(
                long input, long weights, long output, int rows, int inFeatures, int outFeatures) {
            this.calls.add("bf16float:" + rows);
        }

        @Override
        public void nvfp4GateUpSwiGluBf16(
                long input, long weights, long output, int rows, int width, int outputs, long bytes) {
            this.calls.add("nvfp4gateup:" + rows);
        }

        @Override
        public void q3GateUpSwiGluBf16(
                long input,
                long weights,
                long output,
                int rows,
                int width,
                int outputs,
                long bytes,
                WeightLayout layout) {
            this.calls.add("q3gateup:" + layout + ":" + rows);
        }

        @Override
        public long allocate(long byteSize) {
            return 1;
        }

        @Override
        public void copyHostToDevice(long destination, MemorySegment source, long byteSize) {}

        @Override
        public void copyDeviceToHost(MemorySegment destination, long source, long byteSize) {}

        @Override
        public void free(long address) {}

        @Override
        public void embedQ3(
                long tokenIdsAddress,
                long embeddingAddress,
                long embeddingByteSize,
                long hiddenStateAddress,
                int tokenCount,
                int vocabularySize,
                int hiddenSize) {}

        @Override
        public void synchronize() {}
    }
}
