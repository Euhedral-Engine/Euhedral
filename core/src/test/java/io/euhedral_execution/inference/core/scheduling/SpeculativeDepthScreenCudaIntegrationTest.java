package io.euhedral_execution.inference.core.scheduling;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import io.euhedral_execution.inference.core.model_loader.HostWeightSelection;
import io.euhedral_execution.inference.core.model_loader.QwenModel;
import io.euhedral_execution.inference.core.model_loader.WeightResidency;
import io.euhedral_execution.inference.core.model_loader.artifact.QwenArtifactReader;
import io.euhedral_execution.inference.core.tokenizer.QwenTokenizer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/// Screening tool, not a regression test (skipped unless `euhedral.speculative.screen` is set): one process
/// loads the model once and runs speculative generation at every requested depth, exact and (NVFP4)
/// native verification, interleaved over rounds on the same chat prompts. Process-to-process variance
/// (host-backed transfer bandwidth, clocks) cancels, so depths and modes can be ranked by their per-step
/// verify and draft times and their acceptance without end-to-end benchmark forks.
class SpeculativeDepthScreenCudaIntegrationTest {

    private static final double GIB = 1L << 30;

    private static final List<String> TASKS = List.of(
            "Summarize the document above in detail, section by section.",
            "Write a Java implementation of the main mechanism the document above describes, with comments"
                    + " explaining each part.",
            "List the design decisions in the document above and explain the trade-off behind each one.",
            "Write a tutorial for a new engineer that explains the ideas in the document above, with examples.");

    @Test
    @Timeout(value = 7200, unit = TimeUnit.SECONDS)
    void screenDepthsAndModes() throws Throwable {
        assumeTrue(Boolean.getBoolean("euhedral.speculative.screen"), "screening tool");
        Path library = Path.of(System.getProperty("euhedral.cuda.library"));
        Path artifact = Path.of(System.getProperty("euhedral.speculative.artifact"));
        int[] depths = Arrays.stream(System.getProperty("euhedral.speculative.depths", "2,3,4")
                        .split(","))
                .mapToInt(Integer::parseInt)
                .toArray();
        boolean nativeToo = Boolean.getBoolean("euhedral.speculative.native");
        int contextTokens = Integer.getInteger("euhedral.speculative.prefix", 1024);
        int budget = Integer.getInteger("euhedral.speculative.tokens", 256);
        int rounds = Integer.getInteger("euhedral.speculative.rounds", 2);
        QwenTokenizer tokenizer = QwenTokenizer.load(
                Path.of(System.getProperty("euhedral.qwen.tokenizer-dir", "/mnt/shared/qwen38-quant/source/qwen")));
        String[] paragraphs = Files.readString(SpeculativeVerifyCudaIntegrationTest.repositoryRoot()
                        .resolve("benchmark/src/main/resources/io/euhedral_execution/inference/benchmark/prompt/"
                                + "chat-corpus-v1-document.md"))
                .split("\n\n");
        List<int[]> prompts = new ArrayList<>();
        for (String task : TASKS) {
            // The most paragraphs that fit: binary search, since tokenizing a long prefix is slow.
            int low = 1, high = paragraphs.length;
            while (low < high) {
                int middle = (low + high + 1) >>> 1;
                if (chat(tokenizer, paragraphs, middle, task).length <= contextTokens) low = middle;
                else high = middle - 1;
            }
            prompts.add(chat(tokenizer, paragraphs, low, task));
        }
        var data = QwenArtifactReader.read(artifact);
        long hostBytes = Long.getLong("euhedral.speculative.host-mib", 0L) << 20;
        try (CudaGpuMemory gpu = printFree(new CudaGpuMemory(library));
                QwenModel model = QwenModel.load(
                        artifact,
                        data,
                        gpu,
                        WeightResidency.SPECULATIVE,
                        HostWeightSelection.select(data, hostBytes),
                        QwenModel.DEFAULT_STAGING_SLOTS);
                var lattice = new PullingLattice()) {
            var plan = new QwenExecutionPlan(model.weights(), model.staging());
            long[] minFree = {gpu.deviceMemoryInfo().freeBytes()};
            System.out.printf(
                    Locale.ROOT,
                    "memory after load: allocated %.3f GiB, device free %.3f GiB%n",
                    gpu.allocatedBytes() / GIB,
                    minFree[0] / GIB);
            gpu.resetPeakAllocatedBytes();
            var runtime = new EuhedralInferenceRuntime(lattice, plan, gpu);
            List<String> modes = nativeToo ? List.of("exact", "native") : List.of("exact");
            // [mode][depth] accumulated statistics over rounds and prompts.
            long[][][] totals = new long[modes.size()][depths.length][7];
            long[][][] histograms = new long[modes.size()][depths.length][8];
            long sequenceId = 1000;
            try {
                for (int round = 0; round <= rounds; round++) { // round 0 warms up
                    for (int m = 0; m < modes.size(); m++) {
                        gpu.selectNvfp4NativeDecode(modes.get(m).equals("native"));
                        for (int d = 0; d < depths.length; d++) {
                            for (int[] prompt : prompts) {
                                var sequence = new QwenSequenceState(++sequenceId);
                                try (var decoder = new QwenSpeculativeDecoder(
                                        runtime,
                                        plan,
                                        gpu,
                                        sequence,
                                        tokenizer::isGenerationEosToken,
                                        depths[d],
                                        512)) {
                                    int[] emitted = {0};
                                    decoder.generate(prompt, budget, token -> {
                                        if (emitted[0]++ % 16 == 0)
                                            minFree[0] = Math.min(
                                                    minFree[0],
                                                    gpu.deviceMemoryInfo().freeBytes());
                                    });
                                    if (round == 0) continue;
                                    var s = decoder.statistics();
                                    long[] t = totals[m][d];
                                    t[0] += s.verifications;
                                    t[1] += s.outputTokens - 1;
                                    t[2] += s.verifyNanos;
                                    t[3] += s.catchUpNanos;
                                    t[4] += s.recursionNanos;
                                    t[5] += s.prefillNanos;
                                    t[6] += s.promptCatchUpNanos;
                                    for (int a = 0; a < s.acceptedDrafts.length; a++)
                                        histograms[m][d][a] += s.acceptedDrafts[a];
                                } finally {
                                    sequence.complete();
                                }
                            }
                        }
                    }
                }
            } finally {
                gpu.selectNvfp4NativeDecode(false);
                runtime.close();
            }
            System.out.printf(
                    Locale.ROOT,
                    "memory during runs: peak allocated %.3f GiB, min device free %.3f GiB (prompts %s tokens)%n",
                    gpu.peakAllocatedBytes() / GIB,
                    minFree[0] / GIB,
                    prompts.stream().map(p -> Integer.toString(p.length)).toList());
            System.out.printf(
                    Locale.ROOT,
                    "screen %s, %d-token prompts, %d tokens, %d rounds%n",
                    artifact.getFileName(),
                    contextTokens,
                    budget,
                    rounds);
            for (int m = 0; m < modes.size(); m++) {
                for (int d = 0; d < depths.length; d++) {
                    long[] t = totals[m][d];
                    double verifications = t[0];
                    double stepMs = (t[2] + t[3] + t[4]) / 1e6 / verifications;
                    double tokensPerStep = t[1] / verifications;
                    // Steps only: the prompt's catch-up (the MTP layer over every prompt token) and its first
                    // drafts belong to the time to first token.
                    System.out.printf(
                            Locale.ROOT,
                            "%-6s depth %d: verify %6.2f ms, catch-up %5.2f ms, recursion %5.2f ms, tokens/step %.3f,"
                                    + " step %6.2f ms -> %6.1f tok/s | accepted %s | prompt: prefill %.2f s,"
                                    + " MTP catch-up %.2f s%n",
                            modes.get(m),
                            depths[d],
                            t[2] / 1e6 / verifications,
                            t[3] / 1e6 / verifications,
                            t[4] / 1e6 / verifications,
                            tokensPerStep,
                            stepMs,
                            1000.0 * tokensPerStep / stepMs,
                            Arrays.toString(Arrays.copyOf(histograms[m][d], depths[d] + 1)),
                            t[5] / 1e9 / prompts.size() / rounds,
                            t[6] / 1e9 / prompts.size() / rounds);
                }
            }
        }
    }

    private static CudaGpuMemory printFree(CudaGpuMemory gpu) {
        System.out.printf(
                Locale.ROOT,
                "memory before load: device free %.3f GiB%n",
                gpu.deviceMemoryInfo().freeBytes() / GIB);
        return gpu;
    }

    private static int[] chat(QwenTokenizer tokenizer, String[] paragraphs, int count, String task) {
        String document = String.join("\n\n", Arrays.asList(paragraphs).subList(0, count));
        return tokenizer.encodeWithModelSpecialTokens("<|im_start|>user\nHere is a document:\n\n" + document + "\n\n"
                + task + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
    }
}
