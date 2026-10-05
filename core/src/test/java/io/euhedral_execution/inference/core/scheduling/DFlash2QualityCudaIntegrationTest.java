package io.euhedral_execution.inference.core.scheduling;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import io.euhedral_execution.inference.core.model_loader.QwenModel;
import io.euhedral_execution.inference.core.tokenizer.QwenTokenizer;
import java.io.BufferedWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/// DFlash2 drafting quality on fixed chat prompts, for comparing drafters (BF16 against NVFP4 projections): every
/// verification's anchor position, proposal, candidates and accepted drafts, and each prompt's generated tokens, as
/// JSON lines in `-Peuhedral.dflash2.report` (tools/dflash2_quality.py compares two reports). Greedy output is exact
/// whatever the drafter, so two reports hold the same tokens and differ only in their steps.
class DFlash2QualityCudaIntegrationTest {

    static final List<String> PROMPTS = List.of(
            "Write a Java method that reverses a singly linked list, with a short explanation.",
            "Explain how a hash map handles collisions, with an example in Python.",
            "A train leaves at 9:40 and travels 210 km at 84 km/h. When does it arrive? Show the steps.",
            "Summarize the causes of the French Revolution in five bullet points.",
            "Write a short story about a lighthouse keeper who finds a message in a bottle.",
            "What is the difference between TCP and UDP? Give two use cases for each.",
            "Write a SQL query that returns the three customers with the highest total order value.",
            "Translate into French: The meeting has been moved to Thursday afternoon because of the storm.");

    @Test
    @Timeout(value = 3600, unit = TimeUnit.SECONDS)
    void recordDraftingQuality() throws Throwable {
        Path artifact = Path.of(System.getProperty("euhedral.qwen.dflash2-artifact", ""));
        String report = System.getProperty("euhedral.dflash2.report", "");
        assumeTrue(Files.isRegularFile(artifact) && !report.isEmpty(), "needs a DFlash2 artifact and a report path");
        int budget = Integer.getInteger("euhedral.speculative.tokens", 256);
        QwenTokenizer tokenizer = QwenTokenizer.load(
                Path.of(System.getProperty("euhedral.qwen.tokenizer-dir", "/mnt/shared/qwen38-quant/source/qwen")));
        try (CudaGpuMemory gpu = new CudaGpuMemory(Path.of(System.getProperty("euhedral.cuda.library")));
                QwenModel model = DFlash2SpeculativeDecodeCudaIntegrationTest.load(artifact, gpu, 4096);
                var lattice = new PullingLattice();
                BufferedWriter out = Files.newBufferedWriter(Path.of(report))) {
            var plan = new QwenExecutionPlan(model.weights(), model.staging());
            var runtime = new EuhedralInferenceRuntime(lattice, plan, gpu);
            try {
                long id = 500;
                for (int index = 0; index < PROMPTS.size(); index++) {
                    int[] prompt = tokenizer.encodeWithModelSpecialTokens("<|im_start|>user\n" + PROMPTS.get(index)
                            + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
                    var sequence = new QwenSequenceState(++id);
                    int promptIndex = index;
                    try (var decoder =
                            new DFlash2Decoder(runtime, plan, gpu, sequence, tokenizer::isGenerationEosToken, 512)) {
                        decoder.observe((position, anchor, proposal, candidates, accepted) -> {
                            try {
                                out.write("{\"prompt\":" + promptIndex + ",\"position\":" + position + ",\"anchor\":"
                                        + anchor + ",\"proposal\":" + Arrays.toString(proposal) + ",\"candidates\":"
                                        + Arrays.toString(candidates) + ",\"accepted\":" + accepted + "}\n");
                            } catch (java.io.IOException failure) {
                                throw new java.io.UncheckedIOException(failure);
                            }
                        });
                        List<Integer> tokens = decoder.generate(prompt, budget, token -> {}, null);
                        out.write("{\"prompt\":" + index + ",\"tokens\":" + tokens + ",\"statistics\":\""
                                + decoder.statistics() + "\"}\n");
                        System.out.println("DFLASH2_QUALITY prompt " + index + ": " + decoder.statistics());
                    } finally {
                        sequence.complete();
                    }
                }
            } finally {
                runtime.close();
            }
        }
    }
}
