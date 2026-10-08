package io.euhedral_execution.inference.core.model.qwen38;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import io.euhedral_execution.inference.core.generation.HostLogits;
import io.euhedral_execution.inference.core.generation.LogitsRequirement;
import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import io.euhedral_execution.inference.core.model.qwen38.speculative.MtpDecoder;
import io.euhedral_execution.inference.core.runtime.PullingLattice;
import io.euhedral_execution.inference.core.state.AttentionKvState;
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
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/// MTP speculative decoding is an optimization of greedy decode: for each prompt, ordinary one-row greedy
/// decode and MTP speculative decode on fresh sequences must produce exactly the same token IDs and
/// leave exactly the same GDN and attention state. Prompts cover several kinds of text, long runs (many
/// steps with zero, partial and full acceptance, and context growth) and an end-of-generation stop.
@ModelGroup.Nvfp4
class SpeculativeDecodeCudaIntegrationTest {

    @Test
    @Timeout(value = 300, unit = TimeUnit.SECONDS)
    void speculativeDecodeEqualsGreedyDecode() throws Throwable {
        Path tokenizerDirectory =
                Path.of(System.getProperty("euhedral.qwen.tokenizer-dir", "/mnt/shared/qwen38-quant/source/qwen"));
        int budget = Integer.getInteger("euhedral.speculative.tokens", 160);
        QwenTokenizer tokenizer = QwenTokenizer.load(tokenizerDirectory);
        Path root = SpeculativeVerifyCudaIntegrationTest.repositoryRoot();
        String frameModel = Files.readString(root.resolve("docs/FRAME_MODEL.md"));
        int[] text = tokenizer.encodeText(frameModel);
        List<int[]> prompts = List.of(
                tokenizer.encodeWithModelSpecialTokens("<|im_start|>user\nWhat is the capital of France? Answer with"
                        + " one word.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"),
                Arrays.copyOf(text, 600),
                tokenizer.encodeWithModelSpecialTokens("<|im_start|>user\nWrite a Java method that reverses a"
                        + " singly linked list, with a short explanation.<|im_end|>\n<|im_start|>assistant\n"
                        + "<think>\n\n</think>\n\n"));
        long hostMiB = Long.getLong("euhedral.speculative.host-mib", 1024L);
        var loaded = SharedQwen38Mtp.nvfp4(hostMiB);
        CudaGpuMemory gpu = loaded.gpu();
        Qwen38Model model = loaded.model();
        try (var lattice = new PullingLattice()) {
            var plan = new ExecutionPlan(model.weights(), model.staging());
            var runtime = new Execution(lattice, plan, gpu);
            try {
                long id = 100;
                for (int[] prompt : prompts) {
                    var ordinary = new Sequence(++id);
                    List<Integer> expected;
                    List<byte[]> expectedState;
                    int expectedNext;
                    long ordinaryPosition;
                    try (var logits =
                            new HostLogits(gpu, model.weights().config().vocabSize())) {
                        logits.selectOnDevice(true);
                        expected = greedy(runtime, plan, ordinary, prompt, budget, logits, tokenizer);
                        expectedNext = probe(runtime, plan, ordinary, logits);
                        expectedState = snapshot(gpu, model.weights().config().layerTypes(), ordinary);
                    } finally {
                        ordinaryPosition = ordinary.currentTokenPosition() - 1;
                        ordinary.complete();
                    }
                    var speculative = new Sequence(++id);
                    try (var decoder =
                            new MtpDecoder(runtime, plan, gpu, speculative, tokenizer::isGenerationEosToken, 3, 512)) {
                        List<Integer> actual = decoder.generate(prompt, budget, token -> {});
                        System.out.println("prompt of " + prompt.length + " tokens, " + expected.size() + " generated: "
                                + decoder.statistics());
                        assertEquals(expected, actual, "speculative tokens");
                        assertEquals(
                                prompt.length
                                        + expected.size()
                                        - (tokenizer.isGenerationEosToken(expected.getLast()) ? 1 : 0),
                                speculative.currentTokenPosition());
                        assertEquals(ordinaryPosition, speculative.currentTokenPosition());
                        try (var logits =
                                new HostLogits(gpu, model.weights().config().vocabSize())) {
                            logits.selectOnDevice(true);
                            assertEquals(
                                    expectedNext, probe(runtime, plan, speculative, logits), "decode after generation");
                        }
                        List<byte[]> actualState =
                                snapshot(gpu, model.weights().config().layerTypes(), speculative);
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

    /// Ordinary greedy generation, as Session runs it: chunked prefill, then one decode
    /// quantum per token, feeding the last allowed token without sampling and never feeding the end token.
    static List<Integer> greedy(
            Execution runtime,
            ExecutionPlan plan,
            Sequence sequence,
            int[] prompt,
            int budget,
            HostLogits logits,
            QwenTokenizer tokenizer)
            throws Exception {
        for (int offset = 0; offset < prompt.length; offset += 512) {
            int end = Math.min(prompt.length, offset + 512);
            boolean last = end == prompt.length;
            execute(
                    runtime,
                    new Quantum(
                            plan,
                            sequence,
                            Quantum.ExecutionKind.PREFILL,
                            offset,
                            Arrays.copyOfRange(prompt, offset, end),
                            last ? LogitsRequirement.LAST_TOKEN : LogitsRequirement.NONE,
                            last ? logits : null));
        }
        List<Integer> tokens = new ArrayList<>();
        int token = logits.selectedToken();
        while (true) {
            tokens.add(token);
            if (tokenizer.isGenerationEosToken(token)) return tokens;
            boolean more = tokens.size() < budget;
            execute(
                    runtime,
                    new Quantum(
                            plan,
                            sequence,
                            Quantum.ExecutionKind.DECODE,
                            sequence.currentTokenPosition(),
                            new int[] {token},
                            more ? LogitsRequirement.LAST_TOKEN : LogitsRequirement.NONE,
                            more ? logits : null));
            if (!more) return tokens;
            token = logits.selectedToken();
        }
    }

    /// One more decode quantum with a fixed token: it applies any GDN replay a speculative sequence still
    /// holds, so both sequences' states are materialized, and returns its greedy token.
    static int probe(Execution runtime, ExecutionPlan plan, Sequence sequence, HostLogits logits) throws Exception {
        execute(
                runtime,
                new Quantum(
                        plan,
                        sequence,
                        Quantum.ExecutionKind.DECODE,
                        sequence.currentTokenPosition(),
                        new int[] {198},
                        LogitsRequirement.LAST_TOKEN,
                        logits));
        return logits.selectedToken();
    }

    static void execute(Execution runtime, Quantum context) throws Exception {
        var outcome = runtime.submit(context).get(600, TimeUnit.SECONDS);
        if (outcome.status() != Quantum.Status.SUCCESS) throw new AssertionError(outcome.failure());
    }

    /// The committed base state, as host bytes: every GDN layer's recurrent and convolution state, and
    /// every attention layer's committed K and V rows. A pending GDN replay counts as not yet applied,
    /// so the snapshot is taken only when none is pending.
    static List<byte[]> snapshot(CudaGpuMemory gpu, LayerType[] layerTypes, Sequence sequence) {
        var gdn = (GdnStates) sequence.recurrentState();
        var kv = (AttentionStates) sequence.kvCacheState();
        int length = Math.toIntExact(sequence.currentTokenPosition());
        List<byte[]> blocks = new ArrayList<>();
        try (Arena arena = Arena.ofConfined()) {
            for (int layer = 0; layer < layerTypes.length; layer++) {
                if (layerTypes[layer] == LayerType.FULL_ATTENTION) {
                    AttentionKvState state = kv.forLayer(layer);
                    assertEquals(length, state.length(), "KV length, layer " + layer);
                    long rowBytes = state.planePageBytes() / AttentionKvState.PAGE_TOKENS;
                    for (int page = 0; page * AttentionKvState.PAGE_TOKENS < length; page++) {
                        long valid = Math.min(
                                        AttentionKvState.PAGE_TOKENS,
                                        length - (long) page * AttentionKvState.PAGE_TOKENS)
                                * rowBytes;
                        for (int plane = 0; plane < 2; plane++)
                            blocks.add(SpeculativeVerifyCudaIntegrationTest.bytes(
                                    gpu,
                                    arena,
                                    state.pageAddresses().get(page) + plane * state.planePageBytes(),
                                    valid));
                    }
                } else {
                    GdnState state = gdn.forLayer(layer);
                    assertEquals(0, state.pendingReplayRows(), "snapshot with a pending replay, layer " + layer);
                    blocks.add(SpeculativeVerifyCudaIntegrationTest.bytes(
                            gpu, arena, state.recurrentStateAddress(), state.recurrentBytes()));
                    blocks.add(SpeculativeVerifyCudaIntegrationTest.bytes(
                            gpu, arena, state.convolutionStateAddress(), state.convolutionBytes()));
                }
            }
        }
        return blocks;
    }
}
