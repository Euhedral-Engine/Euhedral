package io.euhedral_execution.inference.core.model.qwen38;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import io.euhedral_execution.inference.core.generation.DeviceLogits;
import io.euhedral_execution.inference.core.generation.HostLogits;
import io.euhedral_execution.inference.core.generation.LogitsRequirement;
import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import io.euhedral_execution.inference.core.model.qwen38.speculative.DraftLength;
import io.euhedral_execution.inference.core.model.qwen38.speculative.MtpDecoder;
import io.euhedral_execution.inference.core.runtime.PullingLattice;
import io.euhedral_execution.inference.core.state.KvFormat;
import io.euhedral_execution.inference.core.testing.ModelGroup;
import io.euhedral_execution.inference.core.testing.SharedQwen38Mtp;
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

/// Sequences that keep their attention caches in BF16 on the compact Q3 artifact: MTP speculative decoding (drafts,
/// verification, rejected drafts truncated and overwritten) generates exactly the tokens of one-row greedy decode and
/// leaves exactly its state, and a batched verification of several BF16-cache sequences computes every row and state
/// as each sequence verified alone.
@ModelGroup.CompactQ3
@ModelGroup.Mtp
class KvFormatCudaIntegrationTest {

    @Test
    @Timeout(value = 600, unit = TimeUnit.SECONDS)
    void speculativeDecodeOverABf16CacheEqualsGreedyDecode() throws Throwable {
        QwenTokenizer tokenizer = tokenizer();
        int[] text = document(tokenizer);
        int budget = 64;
        List<int[]> prompts = List.of(
                tokenizer.encodeWithModelSpecialTokens("<|im_start|>user\nWhat is the capital of France? Answer with"
                        + " one word.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"),
                tokenizer.encodeWithModelSpecialTokens("<|im_start|>user\nWrite a Java method that reverses a"
                        + " singly linked list, with a short explanation.<|im_end|>\n<|im_start|>assistant\n"
                        + "<think>\n\n</think>\n\n"),
                Arrays.copyOf(text, 600),
                // Crosses a KV page boundary and the 2048-key attention switch while generating.
                Arrays.copyOf(text, 1990));
        var loaded = SharedQwen38Mtp.q3(Long.getLong("euhedral.speculative.host-mib", 1024L));
        CudaGpuMemory gpu = loaded.gpu();
        Qwen38Model model = loaded.model();
        DraftLength length = ArtifactProfile.of(loaded.artifact()).draftLength();
        try (var lattice = new PullingLattice()) {
            var plan = new ExecutionPlan(model.weights(), model.staging());
            var runtime = new Execution(lattice, plan, gpu);
            try {
                long id = 300;
                for (int[] prompt : prompts) {
                    var ordinary = bf16Sequence(++id);
                    List<Integer> expected;
                    List<byte[]> expectedState;
                    try (var logits =
                            new HostLogits(gpu, model.weights().config().vocabSize())) {
                        logits.selectOnDevice(true);
                        expected = SpeculativeDecodeCudaIntegrationTest.greedy(
                                runtime, plan, ordinary, prompt, budget, logits, tokenizer);
                        SpeculativeDecodeCudaIntegrationTest.probe(runtime, plan, ordinary, logits);
                        expectedState = SpeculativeDecodeCudaIntegrationTest.snapshot(
                                gpu, model.weights().config().layerTypes(), ordinary);
                    } finally {
                        ordinary.complete();
                    }
                    var speculative = bf16Sequence(++id);
                    try (var decoder = new MtpDecoder(
                            runtime,
                            plan,
                            gpu,
                            speculative,
                            tokenizer::isGenerationEosToken,
                            length,
                            512,
                            length.most())) {
                        List<Integer> actual = decoder.generate(prompt, budget, token -> {});
                        System.out.println(
                                "BF16 cache, prompt of " + prompt.length + " tokens: " + decoder.statistics());
                        assertEquals(expected, actual, "speculative tokens");
                        try (var logits =
                                new HostLogits(gpu, model.weights().config().vocabSize())) {
                            logits.selectOnDevice(true);
                            SpeculativeDecodeCudaIntegrationTest.probe(runtime, plan, speculative, logits);
                        }
                        assertEquals(KvFormat.BF16, ((AttentionStates) speculative.kvCacheState()).format());
                        List<byte[]> actualState = SpeculativeDecodeCudaIntegrationTest.snapshot(
                                gpu, model.weights().config().layerTypes(), speculative);
                        assertEquals(expectedState.size(), actualState.size());
                        for (int i = 0; i < expectedState.size(); i++)
                            assertArrayEquals(expectedState.get(i), actualState.get(i), "state block " + i);
                    } finally {
                        speculative.complete();
                    }
                }
            } finally {
                runtime.close();
            }
        }
    }

    @Test
    @Timeout(value = 600, unit = TimeUnit.SECONDS)
    void batchedVerificationOverBf16CachesEqualsEachSequenceAlone() throws Throwable {
        int[] text = document(tokenizer());
        int[] prefixes = {300, 411, 2040};
        int[][] rounds = {{8, 8, 8}, {5, 8, 3}};
        var loaded = SharedQwen38Mtp.q3(Long.getLong("euhedral.speculative.host-mib", 1024L));
        CudaGpuMemory gpu = loaded.gpu();
        Qwen38Model model = loaded.model();
        try (var lattice = new PullingLattice()) {
            var plan = new ExecutionPlan(model.weights(), model.staging());
            var runtime = new Execution(lattice, plan, gpu);
            List<Sequence> batched = new ArrayList<>(), alone = new ArrayList<>();
            try {
                int[][] documents = new int[prefixes.length][];
                for (int s = 0; s < prefixes.length; s++) {
                    documents[s] = Arrays.copyOfRange(text, 97 * s, 97 * s + prefixes[s] + 64);
                    batched.add(bf16Sequence(400 + s));
                    alone.add(bf16Sequence(500 + s));
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
                    for (int s = 0; s < prefixes.length; s++)
                        for (int r = 0; r < counts[s]; r++)
                            assertArrayEquals(
                                    expected.get(s).get(r),
                                    rows.get(row++),
                                    "sequence " + s + " row " + r + " of " + Arrays.toString(counts));
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

    private static Sequence bf16Sequence(long id) {
        var sequence = new Sequence(id);
        sequence.useKvFormat(KvFormat.BF16);
        return sequence;
    }

    private static QwenTokenizer tokenizer() throws Exception {
        return QwenTokenizer.load(
                Path.of(System.getProperty("euhedral.qwen.tokenizer-dir", "/mnt/shared/qwen38-quant/source/qwen")));
    }

    private static int[] document(QwenTokenizer tokenizer) throws Exception {
        return tokenizer.encodeText(Files.readString(
                SpeculativeVerifyCudaIntegrationTest.repositoryRoot().resolve("docs/FRAME_MODEL.md")));
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
