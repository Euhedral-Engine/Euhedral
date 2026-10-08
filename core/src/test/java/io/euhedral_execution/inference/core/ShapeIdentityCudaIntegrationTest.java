package io.euhedral_execution.inference.core;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.hardware_utils.topology.SystemInfo;
import io.euhedral_execution.inference.core.model.qwen38.Qwen38Runtime;
import io.euhedral_execution.inference.core.model.qwen38.ShapeDescription;
import io.euhedral_execution.inference.core.testing.ModelGroup;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import java.util.stream.Stream;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

/// Every view of each real Qwen3.8 artifact, as production loads it (residency, staging, MTP), keeps the topology and
/// specs recorded before 3.8's views moved into shapes, and its prompt graph the shape recorded with it. Run one
/// artifact per JVM:
/// `-Peuhedral.shapes.artifacts=q3`, then `nvfp4`, `nvfp4-compressed`; `-Peuhedral.shapes.record=true` records. Without
/// the filter the test skips (the CUDA suite runs it as a skip); a filter that names no artifact fails.
// Own JVM: loads one of several artifacts chosen per JVM, and sets a global residency budget property.
@ModelGroup.OwnJvm
class ShapeIdentityCudaIntegrationTest {

    private static final Path GOLDENS = Path.of("src/test/resources/shapes");
    private static final String TOKENIZER = "/mnt/shared/qwen38-quant/source/qwen";
    private static final List<String> NAMES = List.of("q3", "nvfp4", "nvfp4-compressed");
    /// The device memory the residency planner sees: below what is free on the 16 GB card, whatever else runs.
    private static final long DEVICE_BUDGET = 14L << 30;

    static Stream<Arguments> artifacts() {
        String only = System.getProperty("euhedral.shapes.artifacts");
        if (only != null)
            for (String name : only.strip().split("\\s*,\\s*"))
                if (!NAMES.contains(name))
                    throw new IllegalArgumentException("euhedral.shapes.artifacts names no artifact: " + name);
        return Stream.of(
                Arguments.of(
                        "q3",
                        System.getProperty(
                                "euhedral.qwen.artifact", "/mnt/shared/qwen38-quant/artifacts/qwen3_8_27b_q3.edrl")),
                Arguments.of(
                        "nvfp4",
                        System.getProperty(
                                "euhedral.qwen.nvfp4-artifact",
                                "/mnt/shared/qwen38-quant/artifacts/qwen3_8_27b_nvfp4.edrl")),
                Arguments.of(
                        "nvfp4-compressed",
                        System.getProperty(
                                "euhedral.qwen.nvfp4-compressed-artifact",
                                "/mnt/shared/qwen38-quant/artifacts/qwen3_8_27b_nvfp4_compressed.edrl")));
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("artifacts")
    void everyViewKeepsItsRecordedShape(String name, String artifact) throws Exception {
        String only = System.getProperty("euhedral.shapes.artifacts");
        assumeTrue(only != null, "select an artifact with -Peuhedral.shapes.artifacts");
        assumeTrue(List.of(only.strip().split("\\s*,\\s*")).contains(name), "not selected: " + name);
        String library = System.getProperty("euhedral.cuda.library");
        assumeTrue(library != null && Files.isRegularFile(Path.of(library)), "no CUDA library");
        assumeTrue(Files.isRegularFile(Path.of(artifact)), "no artifact " + artifact);
        var config = new InferenceConfig(
                Path.of(artifact),
                Path.of(TOKENIZER),
                Path.of(library),
                SystemInfo.getPCpuSet(),
                Duration.ofSeconds(30));
        // Residency follows the GPU's free memory at load; a fixed budget below it makes every load place the same
        // weights in host memory, so the staged views compare.
        System.setProperty("euhedral.residency.device-budget-bytes", Long.toString(DEVICE_BUDGET));
        try (InferenceEngine engine = InferenceEngine.load(config)) {
            var plan = ((Qwen38Runtime) engine.modelRuntime()).plan();
            ShapeDescription.check(GOLDENS, name, ShapeDescription.views(plan));
            // Three full chunks and a short last one, at the sessions' chunk size.
            ShapeDescription.check(
                    GOLDENS,
                    name + "-prompt",
                    ShapeDescription.prompt(plan, InferenceConfig.PREFILL_CHUNK_TOKENS, 100, 4));
        }
    }
}
