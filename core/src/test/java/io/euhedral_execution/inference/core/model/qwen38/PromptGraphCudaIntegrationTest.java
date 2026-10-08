package io.euhedral_execution.inference.core.model.qwen38;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.hardware_utils.topology.SystemInfo;
import io.euhedral_execution.inference.core.InferenceConfig;
import io.euhedral_execution.inference.core.InferenceEngine;
import io.euhedral_execution.inference.core.generation.HostLogits;
import io.euhedral_execution.inference.core.generation.LogitsRequirement;
import java.lang.foreign.ValueLayout;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/// A prompt run as one graph of its chunks leaves exactly the state and the last logits row that the same chunks
/// leave run one quantum after another: every layer's GDN state and KV rows, and the sampled row. Runs on the dense
/// artifact `-Peuhedral.qwen.artifact` (Q3 by default).
class PromptGraphCudaIntegrationTest {

    private static final int CHUNK = 512;

    @Test
    @Timeout(600)
    void aPromptGraphLeavesTheStateOfItsChunksRunOneAfterAnother() throws Exception {
        compare(1300);
    }

    /// A last chunk below the region view's 64 rows selects the small prefill view, which the prompt graph's
    /// region workspace does not bind: the graph refuses it, and a session runs that chunk as a quantum of its own,
    /// sampling the token the serial chunks sample.
    @Test
    @Timeout(600)
    void aSessionsPromptWithAShortLastChunkSamplesTheSerialToken() throws Exception {
        String library = System.getProperty("euhedral.cuda.library");
        assumeTrue(library != null && Files.isRegularFile(Path.of(library)), "no CUDA library");
        Path artifact = Path.of(
                System.getProperty("euhedral.qwen.artifact", "/mnt/shared/qwen38-quant/artifacts/qwen3_8_27b_q3.edrl"));
        Path tokenizer =
                Path.of(System.getProperty("euhedral.qwen.tokenizer-dir", "/mnt/shared/qwen38-quant/source/qwen"));
        assumeTrue(Files.isRegularFile(artifact), "no artifact");
        int length = 1054;
        var config = new InferenceConfig(
                artifact, tokenizer, Path.of(library), SystemInfo.getPCpuSet(), Duration.ofSeconds(30));
        try (InferenceEngine engine = InferenceEngine.load(config)) {
            var runtime = (Qwen38Runtime) io.euhedral_execution.inference.core.TestEngines.modelRuntime(engine);
            ExecutionPlan plan = runtime.plan();
            Execution execution = runtime.execution();
            int[] prompt = Arrays.copyOf(
                    engine.tokenizer().encodeWithModelSpecialTokens("Count the facts. ".repeat(400)), length);
            assertThrows(
                    IllegalArgumentException.class,
                    () -> Quantum.prompt(plan, new Sequence(3), 0, prompt, CHUNK, LogitsRequirement.NONE, null));
            int vocabulary = plan.weights().config().vocabSize();
            int expected;
            try (var serialLogits = new HostLogits(runtime.gpu(), vocabulary)) {
                var serial = new Sequence(1);
                for (int offset = 0; offset < length; offset += CHUNK) {
                    int end = Math.min(offset + CHUNK, length);
                    boolean last = end == length;
                    var chunk = new Quantum(
                            plan,
                            serial,
                            Quantum.ExecutionKind.PREFILL,
                            offset,
                            Arrays.copyOfRange(prompt, offset, end),
                            last ? LogitsRequirement.LAST_TOKEN : LogitsRequirement.NONE,
                            last ? serialLogits : null);
                    assertEquals(
                            Quantum.Status.SUCCESS,
                            execution.submit(chunk).get(5, TimeUnit.MINUTES).status());
                }
                expected = argmax(serialLogits.row().toArray(ValueLayout.JAVA_SHORT));
            }
            try (var session = io.euhedral_execution.inference.core.TestEngines.createSession(
                    engine, io.euhedral_execution.inference.core.sampling.GenerationConfig.greedy(1))) {
                assertEquals(List.of(expected), session.generate(prompt, 1, text -> {}, null));
            }
        }
    }

    /// The greedy token of a BF16 row: the first of equal maxima.
    private static int argmax(short[] row) {
        int best = 0;
        for (int i = 1; i < row.length; i++)
            if (Float.intBitsToFloat(row[i] << 16) > Float.intBitsToFloat(row[best] << 16)) best = i;
        return best;
    }

    private static void compare(int promptTokens) throws Exception {
        final int PROMPT = promptTokens;
        String library = System.getProperty("euhedral.cuda.library");
        assumeTrue(library != null && Files.isRegularFile(Path.of(library)), "no CUDA library");
        Path artifact = Path.of(
                System.getProperty("euhedral.qwen.artifact", "/mnt/shared/qwen38-quant/artifacts/qwen3_8_27b_q3.edrl"));
        Path tokenizer =
                Path.of(System.getProperty("euhedral.qwen.tokenizer-dir", "/mnt/shared/qwen38-quant/source/qwen"));
        assumeTrue(Files.isRegularFile(artifact), "no artifact");
        var config = new InferenceConfig(
                artifact, tokenizer, Path.of(library), SystemInfo.getPCpuSet(), Duration.ofSeconds(30));
        try (InferenceEngine engine = InferenceEngine.load(config)) {
            var runtime = (Qwen38Runtime) io.euhedral_execution.inference.core.TestEngines.modelRuntime(engine);
            ExecutionPlan plan = runtime.plan();
            Execution execution = runtime.execution();
            int[] prompt = Arrays.copyOf(
                    engine.tokenizer().encodeWithModelSpecialTokens("Count the facts. ".repeat(400)), PROMPT);
            int vocabulary = plan.weights().config().vocabSize();
            try (var serialLogits = new HostLogits(runtime.gpu(), vocabulary);
                    var graphLogits = new HostLogits(runtime.gpu(), vocabulary)) {
                var serial = new Sequence(1);
                for (int offset = 0; offset < PROMPT; offset += CHUNK) {
                    int end = Math.min(offset + CHUNK, PROMPT);
                    boolean last = end == PROMPT;
                    var chunk = new Quantum(
                            plan,
                            serial,
                            Quantum.ExecutionKind.PREFILL,
                            offset,
                            Arrays.copyOfRange(prompt, offset, end),
                            last ? LogitsRequirement.LAST_TOKEN : LogitsRequirement.NONE,
                            last ? serialLogits : null);
                    assertEquals(
                            Quantum.Status.SUCCESS,
                            execution.submit(chunk).get(5, TimeUnit.MINUTES).status());
                }
                var graph = new Sequence(2);
                var whole = Quantum.prompt(plan, graph, 0, prompt, CHUNK, LogitsRequirement.LAST_TOKEN, graphLogits);
                assertEquals(
                        Quantum.Status.SUCCESS,
                        execution.submit(whole).get(5, TimeUnit.MINUTES).status());
                var layers = plan.weights().config().layerTypes();
                List<String> expected = SequenceStateProbe.committedDigests(runtime.gpu(), serial, layers);
                List<String> actual = SequenceStateProbe.committedDigests(runtime.gpu(), graph, layers);
                assertEquals(expected, actual, "every layer's GDN state and KV rows");
                assertArrayEquals(
                        serialLogits.row().toArray(ValueLayout.JAVA_SHORT),
                        graphLogits.row().toArray(ValueLayout.JAVA_SHORT),
                        "the last chunk's logits row");
            }
        }
    }
}
