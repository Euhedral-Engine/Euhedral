package io.euhedral_execution.inference.core.model.qwen38;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import io.euhedral_execution.inference.core.generation.DeviceLogits;
import io.euhedral_execution.inference.core.generation.LogitsRequirement;
import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import io.euhedral_execution.inference.core.runtime.PullingLattice;
import io.euhedral_execution.inference.core.testing.ModelGroup;
import io.euhedral_execution.inference.core.testing.SharedQwen38;
import io.euhedral_execution.inference.core.tokenizer.QwenTokenizer;
import java.lang.foreign.Arena;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/// A batched verification ([Quantum#batch]) must compute every sequence's rows bit for bit as that sequence's own
/// verification, and leave each sequence the same state: three sequences with prompts of different lengths verify
/// their forced tokens together, twice (8 + 8 + 8 rows, then 5 + 8 + 3), while three twins verify the same tokens one
/// sequence at a time. Every logits row, the positions, and every GDN and attention state must match.
@ModelGroup.Nvfp4
class BatchedVerifyCudaIntegrationTest {

    @Test
    @Timeout(value = 600, unit = TimeUnit.SECONDS)
    void batchedRowsAndStatesEqualEachSequenceVerifiedAlone() throws Throwable {
        Path tokenizerDirectory =
                Path.of(System.getProperty("euhedral.qwen.tokenizer-dir", "/mnt/shared/qwen38-quant/source/qwen"));
        int[] text = QwenTokenizer.load(tokenizerDirectory)
                .encodeText(Files.readString(
                        SpeculativeVerifyCudaIntegrationTest.repositoryRoot().resolve("docs/FRAME_MODEL.md")));
        int[] prefixes = {300, 411, 530};
        int[][] rounds = {{8, 8, 8}, {5, 8, 3}};
        var loaded = SharedQwen38.nvfp4();
        CudaGpuMemory gpu = loaded.gpu();
        Qwen38Model model = loaded.model();
        try (var lattice = new PullingLattice()) {
            var plan = new ExecutionPlan(model.weights(), model.staging());
            var runtime = new Execution(lattice, plan, gpu);
            List<Sequence> batched = new ArrayList<>(), alone = new ArrayList<>();
            try {
                int[][] documents = new int[prefixes.length][];
                for (int s = 0; s < prefixes.length; s++) {
                    // Each sequence reads the text from its own offset.
                    documents[s] = Arrays.copyOfRange(text, 97 * s, 97 * s + prefixes[s] + 64);
                    batched.add(new Sequence(100 + s));
                    alone.add(new Sequence(200 + s));
                    int[] prompt = Arrays.copyOf(documents[s], prefixes[s]);
                    SpeculativeVerifyCudaIntegrationTest.prefill(runtime, plan, batched.get(s), prompt, gpu);
                    SpeculativeVerifyCudaIntegrationTest.prefill(runtime, plan, alone.get(s), prompt, gpu);
                }
                for (int[] counts : rounds) {
                    List<Quantum.Part> parts = new ArrayList<>();
                    List<List<short[]>> expected = new ArrayList<>();
                    for (int s = 0; s < prefixes.length; s++) {
                        int at = Math.toIntExact(alone.get(s).currentTokenPosition());
                        int[] forced = Arrays.copyOfRange(documents[s], at, at + counts[s]);
                        expected.add(SpeculativeVerifyCudaIntegrationTest.run(
                                runtime,
                                plan,
                                alone.get(s),
                                Quantum.ExecutionKind.VERIFY,
                                forced,
                                LogitsRequirement.ALL_TOKENS,
                                gpu));
                        parts.add(
                                new Quantum.Part(batched.get(s), batched.get(s).currentTokenPosition(), forced));
                    }
                    List<short[]> rows = batch(runtime, plan, parts, gpu);
                    int row = 0;
                    for (int s = 0; s < prefixes.length; s++) {
                        for (int r = 0; r < counts[s]; r++)
                            assertArrayEquals(
                                    expected.get(s).get(r),
                                    rows.get(row++),
                                    "sequence " + s + " row " + r + " of " + Arrays.toString(counts));
                        assertEquals(
                                alone.get(s).currentTokenPosition(),
                                batched.get(s).currentTokenPosition());
                    }
                }
                for (int s = 0; s < prefixes.length; s++)
                    SpeculativeVerifyCudaIntegrationTest.assertStatesEqual(
                            gpu, model.weights().config().layerTypes(), alone.get(s), batched.get(s));
            } finally {
                for (Sequence sequence : batched) sequence.complete();
                for (Sequence sequence : alone) sequence.complete();
                runtime.close();
            }
        }
    }

    /// Runs one batched verification; returns its logits rows in part order.
    private static List<short[]> batch(
            Execution runtime, ExecutionPlan plan, List<Quantum.Part> parts, CudaGpuMemory gpu) throws Exception {
        AtomicReference<List<short[]>> captured = new AtomicReference<>(List.of());
        var context = Quantum.batch(plan, Quantum.ExecutionKind.VERIFY, parts, LogitsRequirement.ALL_TOKENS);
        var outcome = runtime.submit(context, done -> {
                    try (Arena arena = Arena.ofConfined();
                            DeviceLogits output = done.logitsOutput().orElseThrow()) {
                        List<short[]> rows = new ArrayList<>();
                        int vocabulary = output.vocabularySize();
                        for (int row = 0; row < output.tokenCount(); row++)
                            rows.add(SpeculativeVerifyCudaIntegrationTest.shorts(
                                    gpu,
                                    arena,
                                    output.deviceAddress() + (long) row * vocabulary * Short.BYTES,
                                    vocabulary));
                        captured.set(rows);
                    }
                })
                .get(600, TimeUnit.SECONDS);
        if (outcome.status() != Quantum.Status.SUCCESS) throw new AssertionError(outcome.failure());
        return captured.get();
    }
}
