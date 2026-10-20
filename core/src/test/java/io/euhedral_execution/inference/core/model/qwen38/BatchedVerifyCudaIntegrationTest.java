package io.euhedral_execution.inference.core.model.qwen38;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.generation.DeviceLogits;
import io.euhedral_execution.inference.core.generation.HostLogits;
import io.euhedral_execution.inference.core.generation.LogitsRequirement;
import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import io.euhedral_execution.inference.core.model.qwen38.artifact.ArtifactReader;
import io.euhedral_execution.inference.core.model.qwen38.loader.HostWeightSelection;
import io.euhedral_execution.inference.core.model.qwen38.speculative.SpeculativeAcceptance;
import io.euhedral_execution.inference.core.runtime.PullingLattice;
import io.euhedral_execution.inference.core.testing.ModelGroup;
import io.euhedral_execution.inference.core.testing.SharedQwen38;
import io.euhedral_execution.inference.core.tokenizer.QwenTokenizer;
import io.euhedral_execution.inference.core.tokenizer.TokenizerFiles;
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
/// sequence at a time. Every logits row, the positions, and every GDN and attention state must match. The NVFP4
/// artifact keeps 2304 MiB of its weights in host memory (`euhedral.batched.host-mib`), so six sequences fit beside it
/// on a 16 GB card, and the staged weights are exercised too.
@ModelGroup.OwnJvm
class BatchedVerifyCudaIntegrationTest {

    @Test
    @Timeout(value = 600, unit = TimeUnit.SECONDS)
    void batchedRowsAndStatesEqualEachSequenceVerifiedAlone() throws Throwable {
        Path tokenizerDirectory =
                Path.of(System.getProperty("euhedral.qwen.tokenizer-dir", "unset-euhedral.qwen.tokenizer-dir"));
        int[] text = QwenTokenizer.load(TokenizerFiles.fromCheckpoint(tokenizerDirectory))
                .encodeText(Files.readString(
                        SpeculativeVerifyCudaIntegrationTest.repositoryRoot().resolve("docs/architecture/FRAME_MODEL.md")));
        int[] prefixes = {300, 411, 530};
        int[][] rounds = {{8, 8, 8}, {5, 8, 3}};
        Path artifact = Path.of(System.getProperty("euhedral.qwen.nvfp4-artifact", SharedQwen38.NVFP4_DEFAULT));
        long hostBytes = Long.getLong("euhedral.batched.host-mib", 2304L) << 20;
        var data = ArtifactReader.read(artifact);
        try (CudaGpuMemory gpu = new CudaGpuMemory(Path.of(System.getProperty("euhedral.cuda.library")));
                Qwen38Model model = Qwen38Model.load(
                        artifact,
                        data,
                        gpu,
                        ArtifactProfile.Speculation.NONE,
                        HostWeightSelection.select(data, hostBytes));
                var lattice = new PullingLattice()) {
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

    /// Everything a test body needs: the runtime, three batched and three alone sequences prefilled with the same
    /// prompts, and each sequence's document (its prompt followed by 64 more tokens).
    private interface Body {
        void run(
                Qwen38Model model,
                Execution runtime,
                ExecutionPlan plan,
                CudaGpuMemory gpu,
                int[][] documents,
                List<Sequence> batched,
                List<Sequence> alone)
                throws Exception;
    }

    private static void withSequences(Body body) throws Throwable {
        Path tokenizerDirectory =
                Path.of(System.getProperty("euhedral.qwen.tokenizer-dir", "unset-euhedral.qwen.tokenizer-dir"));
        int[] text = QwenTokenizer.load(TokenizerFiles.fromCheckpoint(tokenizerDirectory))
                .encodeText(Files.readString(
                        SpeculativeVerifyCudaIntegrationTest.repositoryRoot().resolve("docs/architecture/FRAME_MODEL.md")));
        int[] prefixes = {300, 411, 530};
        Path artifact = Path.of(System.getProperty("euhedral.qwen.nvfp4-artifact", SharedQwen38.NVFP4_DEFAULT));
        long hostBytes = Long.getLong("euhedral.batched.host-mib", 2304L) << 20;
        var data = ArtifactReader.read(artifact);
        try (CudaGpuMemory gpu = new CudaGpuMemory(Path.of(System.getProperty("euhedral.cuda.library")));
                Qwen38Model model = Qwen38Model.load(
                        artifact,
                        data,
                        gpu,
                        ArtifactProfile.Speculation.NONE,
                        HostWeightSelection.select(data, hostBytes));
                var lattice = new PullingLattice()) {
            var plan = new ExecutionPlan(model.weights(), model.staging());
            var runtime = new Execution(lattice, plan, gpu);
            List<Sequence> batched = new ArrayList<>(), alone = new ArrayList<>();
            try {
                int[][] documents = new int[prefixes.length][];
                for (int s = 0; s < prefixes.length; s++) {
                    documents[s] = Arrays.copyOfRange(text, 97 * s, 97 * s + prefixes[s] + 64);
                    batched.add(new Sequence(100 + s));
                    alone.add(new Sequence(200 + s));
                    int[] prompt = Arrays.copyOf(documents[s], prefixes[s]);
                    SpeculativeVerifyCudaIntegrationTest.prefill(runtime, plan, batched.get(s), prompt, gpu);
                    SpeculativeVerifyCudaIntegrationTest.prefill(runtime, plan, alone.get(s), prompt, gpu);
                }
                body.run(model, runtime, plan, gpu, documents, batched, alone);
            } finally {
                for (Sequence sequence : batched) sequence.complete();
                for (Sequence sequence : alone) sequence.complete();
                runtime.close();
            }
        }
    }

    /// Each sequence of a batched verification commits only its own first rows (3 of 8, all 8, 1 of 8; then 5 of 5,
    /// 2 of 8, 3 of 3; then one-row steps), and the batched rows, positions and states equal those of the twin
    /// that verified alone with the same committed count.
    @Test
    @Timeout(value = 900, unit = TimeUnit.SECONDS)
    void batchedSegmentsCommitOnlyTheirAcceptedRows() throws Throwable {
        withSequences((model, runtime, plan, gpu, documents, batched, alone) -> {
            int[][] counts = {{8, 8, 8}, {5, 8, 3}, {1, 1, 1}};
            int[][] committed = {{3, 8, 1}, {5, 2, 3}, {1, 1, 1}};
            for (int round = 0; round < counts.length; round++) {
                List<Quantum.Part> parts = new ArrayList<>();
                List<List<short[]>> expected = new ArrayList<>();
                for (int s = 0; s < documents.length; s++) {
                    int at = Math.toIntExact(alone.get(s).currentTokenPosition());
                    int[] forced = Arrays.copyOfRange(documents[s], at, at + counts[round][s]);
                    if (round < 2) {
                        expected.add(SpeculativeVerifyCudaIntegrationTest.verify(
                                runtime, plan, alone.get(s), forced, committed[round][s], gpu));
                    } else {
                        expected.add(SpeculativeVerifyCudaIntegrationTest.run(
                                runtime,
                                plan,
                                alone.get(s),
                                Quantum.ExecutionKind.DECODE,
                                forced,
                                LogitsRequirement.ALL_TOKENS,
                                gpu));
                    }
                    parts.add(new Quantum.Part(batched.get(s), batched.get(s).currentTokenPosition(), forced)
                            .withCommittedRows(committed[round][s]));
                }
                List<short[]> rows = batch(runtime, plan, parts, gpu);
                int row = 0;
                for (int s = 0; s < documents.length; s++) {
                    for (int r = 0; r < counts[round][s]; r++)
                        assertArrayEquals(
                                expected.get(s).get(r),
                                rows.get(row++),
                                "round " + round + " sequence " + s + " row " + r);
                    assertEquals(
                            alone.get(s).currentTokenPosition(),
                            batched.get(s).currentTokenPosition(),
                            "round " + round + " sequence " + s + " position");
                }
            }
            for (int s = 0; s < documents.length; s++)
                SpeculativeVerifyCudaIntegrationTest.assertStatesEqual(
                        gpu, model.weights().config().layerTypes(), alone.get(s), batched.get(s));
        });
    }

    /// Each sequence verifies the model's own next four tokens with one wrong draft (sequence 0 at draft 2, sequence
    /// 1 none, sequence 2 at draft 1); the batched greedy acceptance of every segment equals its twin's alone.
    @Test
    @Timeout(value = 900, unit = TimeUnit.SECONDS)
    void batchedSegmentsResolveTheirOwnGreedyAcceptance() throws Throwable {
        withSequences((model, runtime, plan, gpu, documents, batched, alone) -> {
            // Draft index of each sequence's wrong token; 0 means none.
            int[] wrongDraft = {2, 0, 1};
            int vocabulary = model.weights().config().vocabSize();
            List<HostLogits> logitsToClose = new ArrayList<>();
            try {
                List<Quantum.Part> parts = new ArrayList<>();
                List<SpeculativeAcceptance> aloneAcceptance = new ArrayList<>(), batchedAcceptance = new ArrayList<>();
                for (int s = 0; s < documents.length; s++) {
                    int at = Math.toIntExact(alone.get(s).currentTokenPosition());
                    int[] rows = greedyContinuation(runtime, plan, gpu, documents[s], at, 5);
                    if (wrongDraft[s] > 0) rows[wrongDraft[s]] = (rows[wrongDraft[s]] + 1) % vocabulary;
                    HostLogits hostAlone = new HostLogits(gpu, vocabulary);
                    HostLogits hostBatched = new HostLogits(gpu, vocabulary);
                    hostAlone.selectOnDevice(true);
                    hostBatched.selectOnDevice(true);
                    logitsToClose.add(hostAlone);
                    logitsToClose.add(hostBatched);
                    var acceptAlone = new SpeculativeAcceptance(rows, t -> false, 8);
                    var acceptBatched = new SpeculativeAcceptance(rows, t -> false, 8);
                    aloneAcceptance.add(acceptAlone);
                    batchedAcceptance.add(acceptBatched);
                    var quantum = new Quantum(
                                    plan,
                                    alone.get(s),
                                    Quantum.ExecutionKind.VERIFY,
                                    alone.get(s).currentTokenPosition(),
                                    rows,
                                    LogitsRequirement.ALL_TOKENS,
                                    hostAlone)
                            .withAcceptance(acceptAlone);
                    var outcome = runtime.submit(quantum).get(600, TimeUnit.SECONDS);
                    if (outcome.status() != Quantum.Status.SUCCESS) throw new AssertionError(outcome.failure());
                    parts.add(new Quantum.Part(batched.get(s), batched.get(s).currentTokenPosition(), rows)
                            .verifying(hostBatched, acceptBatched, SpeculativeAcceptance.Selector.GREEDY));
                }
                var context = Quantum.batch(plan, Quantum.ExecutionKind.VERIFY, parts, LogitsRequirement.ALL_TOKENS);
                var outcome = runtime.submit(context).get(600, TimeUnit.SECONDS);
                if (outcome.status() != Quantum.Status.SUCCESS) throw new AssertionError(outcome.failure());
                for (int s = 0; s < documents.length; s++) {
                    assertArrayEquals(
                            aloneAcceptance.get(s).outputs(),
                            batchedAcceptance.get(s).outputs(),
                            "outputs " + s);
                    assertEquals(
                            aloneAcceptance.get(s).committedRows(), context.committedRows(s), "committed rows " + s);
                    assertEquals(
                            batchedAcceptance.get(s).committedRows(),
                            context.committedRows(s),
                            "resolved committed rows " + s);
                    assertEquals(
                            aloneAcceptance.get(s).acceptedDrafts(),
                            batchedAcceptance.get(s).acceptedDrafts(),
                            "accepted drafts " + s);
                    assertEquals(
                            alone.get(s).currentTokenPosition(),
                            batched.get(s).currentTokenPosition(),
                            "position " + s);
                }
                assertTrue(
                        aloneAcceptance.get(0).acceptedDrafts()
                                        != aloneAcceptance.get(1).acceptedDrafts()
                                || aloneAcceptance.get(1).acceptedDrafts()
                                        != aloneAcceptance.get(2).acceptedDrafts(),
                        "the sequences must accept different numbers of drafts");
                for (int s = 0; s < documents.length; s++)
                    SpeculativeVerifyCudaIntegrationTest.assertStatesEqual(
                            gpu, model.weights().config().layerTypes(), alone.get(s), batched.get(s));
            } finally {
                for (HostLogits logits : logitsToClose) logits.close();
            }
        });
    }

    /// The document token at `at` followed by the model's greedy continuation, `count` tokens in all, decoded
    /// on a throwaway sequence so the compared sequences stay untouched.
    private static int[] greedyContinuation(
            Execution runtime, ExecutionPlan plan, CudaGpuMemory gpu, int[] document, int at, int count)
            throws Exception {
        Sequence probe = new Sequence(900);
        try (HostLogits logits = new HostLogits(gpu, plan.weights().config().vocabSize())) {
            logits.selectOnDevice(true);
            SpeculativeVerifyCudaIntegrationTest.prefill(runtime, plan, probe, Arrays.copyOf(document, at), gpu);
            int[] tokens = new int[count];
            tokens[0] = document[at];
            for (int i = 1; i < count; i++) {
                var acceptance = new SpeculativeAcceptance(new int[] {tokens[i - 1]}, t -> false, 1);
                var outcome = runtime.submit(new Quantum(
                                        plan,
                                        probe,
                                        Quantum.ExecutionKind.VERIFY,
                                        probe.currentTokenPosition(),
                                        new int[] {tokens[i - 1]},
                                        LogitsRequirement.ALL_TOKENS,
                                        logits)
                                .withAcceptance(acceptance))
                        .get(600, TimeUnit.SECONDS);
                if (outcome.status() != Quantum.Status.SUCCESS) throw new AssertionError(outcome.failure());
                tokens[i] = acceptance.outputs()[0];
            }
            return tokens;
        } finally {
            try {
                probe.complete();
            } catch (RuntimeException ignored) {
                // Completing the probe must not mask the failure that ended the decode.
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
