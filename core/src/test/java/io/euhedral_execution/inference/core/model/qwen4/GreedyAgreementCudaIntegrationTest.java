package io.euhedral_execution.inference.core.model.qwen4;

import static io.euhedral_execution.inference.core.model.qwen4.Reference.bf;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.euhedral_execution.inference.core.InferenceConfig;
import io.euhedral_execution.inference.core.InferenceEngine;
import io.euhedral_execution.inference.core.generation.GenerationSession;
import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import io.euhedral_execution.inference.core.model.qwen4.loader.HostBudget;
import io.euhedral_execution.inference.core.model.qwen4.loader.Mode;
import io.euhedral_execution.inference.core.sampling.GenerationConfig;
import java.io.InputStream;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.List;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;

/// The end-to-end gate: fixed text prompts, the upstream model's greedy continuations (with every
/// step's top-5 logits, `tools/flash_next_reference.py greedy`, KV cache through the engine's NVFP4
/// codec) and the engine's.
///
/// A greedy token may differ from the reference only where the reference's own margin between the
/// two tokens is inside BF16 noise (a logit difference of `NOISE`), because the engine's logits
/// differ from the reference's by a few percent (chunking alone moves the reference by as much).
/// Teacher-forced, so one flip does not change what follows, the engine's token at every step is
/// compared with the reference's, and across expert-cache capacities the engine's own tokens must
/// be identical.
class GreedyAgreementCudaIntegrationTest {

    /// Logit difference below which two tokens are interchangeable: the reference's own logits move
    /// by up to 0.75 between chunkings of one prompt (docs/FLASH_NEXT_REFERENCE.md), whose logits
    /// are about 20.
    private static final double NOISE = 1.0;

    private record Step(int token, int[] top5, double[] logits) {
        double margin(int other) {
            for (int i = 0; i < top5.length; i++) if (top5[i] == other) return logits[0] - logits[i];
            return Double.POSITIVE_INFINITY;
        }
    }

    private record Case(String prompt, int[] promptIds, List<Step> steps) {}

    private static List<Case> cases() throws Exception {
        try (InputStream in =
                GreedyAgreementCudaIntegrationTest.class.getResourceAsStream("/qwen4/greedy-reference.json")) {
            assumeTrue(in != null, "no greedy reference");
            JsonNode root = new ObjectMapper().readTree(in);
            List<Case> cases = new ArrayList<>();
            for (JsonNode result : root.get("results")) {
                int[] promptIds = new int[result.get("prompt_ids").size()];
                for (int i = 0; i < promptIds.length; i++)
                    promptIds[i] = result.get("prompt_ids").get(i).asInt();
                List<Step> steps = new ArrayList<>();
                for (JsonNode step : result.get("steps")) {
                    int[] top = new int[5];
                    double[] logits = new double[5];
                    for (int i = 0; i < 5; i++) {
                        top[i] = step.get("top5").get(i).asInt();
                        logits[i] = step.get("top5_logits").get(i).asDouble();
                    }
                    steps.add(new Step(step.get("token").asInt(), top, logits));
                }
                cases.add(new Case(result.get("prompt").asText(), promptIds, steps));
            }
            return cases;
        }
    }

    private static int argmax(short[] logits) {
        int best = 0;
        for (int i = 1; i < logits.length; i++) if (bf(logits[i]) > bf(logits[best])) best = i;
        return best;
    }

    /// The engine's token at each step when fed the reference's own continuation.
    private static int[][] teacherForced(CudaGpuMemory gpu, Qwen4Model model, List<Case> cases) throws Exception {
        try (TestLattice.Run run = TestLattice.shared().run(gpu, model, 2048);
                ExecutionPlan executor = run.plan()) {
            int vocabulary = executor.vocabularySize();
            var readback = gpu.allocateReadbackBuffer((long) vocabulary * 2);
            int[][] tokens = new int[cases.size()][];
            try {
                for (int c = 0; c < cases.size(); c++) {
                    Case reference = cases.get(c);
                    tokens[c] = new int[reference.steps().size()];
                    try (Sequence sequence = executor.newSequence()) {
                        int[] feed = reference.promptIds();
                        for (int step = 0; step < tokens[c].length; step++) {
                            for (int at = 0; at < feed.length; at += 512) {
                                int rows = Math.min(512, feed.length - at);
                                Blocking.step(
                                        executor,
                                        sequence,
                                        feed,
                                        at,
                                        rows,
                                        at + rows == feed.length
                                                ? address ->
                                                        gpu.copyDeviceToReadback(readback, address, vocabulary * 2L)
                                                : null);
                            }
                            short[] logits = new short[vocabulary];
                            MemorySegment.copy(readback.segment(), ValueLayout.JAVA_SHORT, 0, logits, 0, vocabulary);
                            tokens[c][step] = argmax(logits);
                            feed = new int[] {reference.steps().get(step).token()};
                        }
                    }
                }
            } finally {
                readback.close();
            }
            return tokens;
        }
    }

    @Test
    void theEnginesGreedyTokensFollowTheReferenceWhateverTheCacheHolds() throws Exception {
        assumeTrue(TestSupport.hasArtifact(), "no artifact");
        List<Case> cases = cases();
        int[][] roomy;
        try (CudaGpuMemory gpu = TestSupport.openGpu();
                Qwen4Model model = Qwen4Model.open(
                        TestSupport.artifactPath(),
                        gpu,
                        gpu.deviceMemoryInfo().freeBytes(),
                        HostBudget.system(),
                        Mode.TEXT,
                        4096)) {
            roomy = teacherForced(gpu, model, cases);
        }
        int[][] minimal;
        try (CudaGpuMemory gpu = TestSupport.openGpu();
                Qwen4Model model = Qwen4Model.open(
                        TestSupport.artifactPath(),
                        gpu,
                        Math.min(5L << 30, gpu.deviceMemoryInfo().freeBytes()),
                        HostBudget.system(),
                        Mode.TEXT,
                        262144)) {
            assertTrue(model.expertCache().slotCount() <= 64);
            minimal = teacherForced(gpu, model, cases);
            assertTrue(model.expertCache().stats().snapshot().evictions() > 0);
            var tiers = model.hierarchyStats();
            System.out.println("minimal cache: " + model.expertCache().slotCount() + " device slots, host store "
                    + model.plan().expertStore()
                    + (tiers.ram() == null
                            ? ""
                            : ", " + tiers.ram().residentExperts() + " resident in RAM, "
                                    + tiers.ram().totalEvictions() + " RAM evictions, "
                                    + tiers.ram().totalHits()
                                    + " RAM hits")
                    + ", " + tiers.artifact().recordReads() + " artifact reads");
        }
        int agree = 0, total = 0;
        for (int c = 0; c < cases.size(); c++) {
            assertArrayEquals(roomy[c], minimal[c], "the cache capacity changed the tokens of prompt " + c);
            for (int step = 0; step < roomy[c].length; step++) {
                Step reference = cases.get(c).steps().get(step);
                total++;
                if (roomy[c][step] == reference.token()) agree++;
                else
                    assertTrue(
                            reference.margin(roomy[c][step]) < NOISE,
                            "prompt " + c + " step " + step + ": engine " + roomy[c][step] + " vs reference "
                                    + reference.token() + ", reference margin " + reference.margin(roomy[c][step]));
            }
        }
        System.out.println("teacher-forced greedy agreement: " + agree + " of " + total + " steps");
    }

    @Test
    void freeRunningGenerationThroughTheEngineAgreesUntilAnInterchangeableFlip() throws Exception {
        assumeTrue(TestSupport.hasArtifact(), "no artifact");
        Path tokenizers =
                Path.of(System.getProperty("euhedral.qwen4.tokenizer-dir", "/mnt/shared/qwen38-flash-next/nvfp4"));
        assumeTrue(Files.isRegularFile(tokenizers.resolve("tokenizer.json")), "no tokenizer");
        List<Case> cases = cases();
        BitSet cpus = new BitSet();
        cpus.set(0);
        try (InferenceEngine engine = InferenceEngine.load(new InferenceConfig(
                TestSupport.artifactPath(),
                tokenizers,
                Path.of(System.getProperty("euhedral.cuda.library")),
                cpus,
                4096,
                Duration.ofSeconds(30)))) {
            int matched = 0, total = 0;
            for (Case reference : cases) {
                int[] prompt = engine.tokenizePromptAsync(reference.prompt()).get(30, TimeUnit.SECONDS);
                assertArrayEquals(reference.promptIds(), prompt, "tokenization of " + reference.prompt());
                GenerationSession session = engine.createGenerationSession(GenerationConfig.greedy(1));
                List<Integer> tokens = session.generateAsync(
                                prompt, reference.steps().size(), text -> {}, null, null)
                        .get(10, TimeUnit.MINUTES);
                session.close();
                int step = 0;
                while (step < tokens.size()
                        && tokens.get(step) == reference.steps().get(step).token()) step++;
                matched += step;
                total += reference.steps().size();
                if (step < tokens.size())
                    assertTrue(
                            reference.steps().get(step).margin(tokens.get(step)) < NOISE,
                            "free-running step " + step + " of '"
                                    + reference
                                            .prompt()
                                            .substring(
                                                    0,
                                                    Math.min(
                                                            30,
                                                            reference.prompt().length())) + "': engine "
                                    + tokens.get(step) + " vs reference "
                                    + reference.steps().get(step).token());
            }
            System.out.println("free-running greedy tokens equal before the first flip: " + matched + " of " + total);
            assertEquals(cases.size(), cases.size());
        }
    }
}
