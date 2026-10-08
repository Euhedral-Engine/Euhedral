package io.euhedral_execution.inference.core.model.qwen4;

import static io.euhedral_execution.inference.core.model.qwen4.Reference.bf;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import io.euhedral_execution.inference.core.testing.ModelGroup;
import java.io.InputStream;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

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
@ModelGroup.FlashNext
@Timeout(value = 5, unit = TimeUnit.MINUTES)
class GreedyAgreementCudaIntegrationTest {

    /// Logit difference below which two tokens are interchangeable: the reference's own logits move
    /// by up to 0.75 between chunkings of one prompt (docs/FLASH_NEXT_REFERENCE.md), whose logits
    /// are about 20.
    static final double NOISE = 1.0;

    record Step(int token, int[] top5, double[] logits) {
        double margin(int other) {
            for (int i = 0; i < top5.length; i++) if (top5[i] == other) return logits[0] - logits[i];
            return Double.POSITIVE_INFINITY;
        }
    }

    record Case(String prompt, int[] promptIds, List<Step> steps) {}

    static List<Case> cases() throws Exception {
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

    static int argmax(short[] logits) {
        int best = 0;
        for (int i = 1; i < logits.length; i++) if (bf(logits[i]) > bf(logits[best])) best = i;
        return best;
    }

    /// The engine's token at each step when fed the reference's own continuation.
    static int[][] teacherForced(SharedFlashNext.Loaded loaded, List<Case> cases) throws Exception {
        CudaGpuMemory gpu = loaded.gpu();
        ExecutionPlan executor = loaded.plan();
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
                                            ? address -> gpu.copyDeviceToReadback(readback, address, vocabulary * 2L)
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

    /// The engine's tokens when fed the reference's continuation, on the shared roomy model (computed once per JVM; the
    /// undersized-cache test compares against them).
    static int[][] roomyTokens(List<Case> cases) {
        return SharedFlashNext.baseline("greedy-roomy", () -> {
            try {
                var loaded = SharedFlashNext.model(SharedFlashNext.ROOMY);
                return teacherForced(loaded, cases);
            } catch (Exception failure) {
                throw new IllegalStateException(failure);
            }
        });
    }

    @Test
    void theEnginesGreedyTokensFollowTheReference() throws Exception {
        assumeTrue(TestSupport.hasArtifact(), "no artifact");
        List<Case> cases = cases();
        int[][] roomy = roomyTokens(cases);
        int agree = 0, total = 0;
        for (int c = 0; c < cases.size(); c++) {
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
}
