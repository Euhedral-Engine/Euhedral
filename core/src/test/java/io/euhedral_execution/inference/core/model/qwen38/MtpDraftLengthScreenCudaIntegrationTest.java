package io.euhedral_execution.inference.core.model.qwen38;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import io.euhedral_execution.inference.core.model.qwen38.artifact.ArtifactReader;
import io.euhedral_execution.inference.core.model.qwen38.loader.ResidencyPlanner;
import io.euhedral_execution.inference.core.model.qwen38.speculative.DraftLength;
import io.euhedral_execution.inference.core.model.qwen38.speculative.MtpDecoder;
import io.euhedral_execution.inference.core.runtime.PullingLattice;
import io.euhedral_execution.inference.core.testing.ModelGroup;
import io.euhedral_execution.inference.core.tokenizer.QwenTokenizer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/// In-process screen of MTP draft lengths: the model loads once and every arm generates every prompt, arms
/// alternating per prompt and rounds alternating their order, so host and transfer conditions are shared. Each
/// generation's decode rate is its committed tokens after the first over its verification, catch-up and drafting
/// time; per arm and prompt category the report sums tokens and time. Every arm must generate the same tokens.
///
/// Prompts: `-Peuhedral.mtp.prompts` (JSON lines from tools/mtp_draft_prompts.py). Arms: `-Peuhedral.mtp.arms`,
/// comma-separated `n` (fixed) or `least-most[/step]@threshold` (default `4,3-7@-0.5`). Rounds: `-Peuhedral.mtp.rounds`
/// (default 2). Tokens: `-Peuhedral.speculative.tokens` (default 256). Context: `-Peuhedral.mtp.context`.
@ModelGroup.OwnJvm // plans its own residency for the prompts' context
class MtpDraftLengthScreenCudaIntegrationTest {

    @Test
    @Timeout(value = 7200, unit = TimeUnit.SECONDS)
    void screenDraftLengths() throws Throwable {
        Path artifact = Path.of(System.getProperty("euhedral.speculative.artifact", ""));
        Path prompts = Path.of(System.getProperty("euhedral.mtp.prompts", ""));
        assumeTrue(Files.isRegularFile(artifact) && Files.isRegularFile(prompts), "needs an MTP artifact and prompts");
        int budget = Integer.getInteger("euhedral.speculative.tokens", 256);
        int rounds = Integer.getInteger("euhedral.mtp.rounds", 2);
        int context = Integer.getInteger("euhedral.mtp.context", 16384);
        List<DraftLength> arms = new ArrayList<>();
        for (String arm : System.getProperty("euhedral.mtp.arms", "4,3-7@-0.5").split(","))
            arms.add(parse(arm.strip()));
        QwenTokenizer tokenizer = QwenTokenizer.load(
                Path.of(System.getProperty("euhedral.qwen.tokenizer-dir", "/mnt/shared/qwen38-quant/source/qwen")));
        var mapper = new ObjectMapper();
        List<JsonNode> entries = new ArrayList<>();
        for (String line : Files.readAllLines(prompts)) if (!line.isBlank()) entries.add(mapper.readTree(line));
        // Per arm, per category: tokens, nanoseconds, verifications, drafts.
        Map<String, long[]> totals = new LinkedHashMap<>();
        try (CudaGpuMemory gpu = new CudaGpuMemory(Path.of(System.getProperty("euhedral.cuda.library")));
                Qwen38Model model = load(artifact, gpu, context);
                var lattice = new PullingLattice()) {
            var plan = new ExecutionPlan(model.weights(), model.staging());
            var runtime = new Execution(lattice, plan, gpu);
            try {
                long id = 900;
                for (int round = 0; round < rounds; round++) {
                    for (JsonNode entry : entries) {
                        int[] prompt = tokenizer.encodeWithModelSpecialTokens(
                                entry.get("text").asText());
                        if (prompt.length + budget + MtpDecoder.MAX_DRAFTS > context)
                            throw new IllegalArgumentException(entry.get("name").asText() + " needs more context");
                        String category = entry.get("category").asText();
                        List<Integer> reference = null;
                        for (int a = 0; a < arms.size(); a++) {
                            DraftLength arm = arms.get(round % 2 == 0 ? a : arms.size() - 1 - a);
                            var sequence = new Sequence(++id);
                            try (var decoder = new MtpDecoder(
                                    runtime,
                                    plan,
                                    gpu,
                                    sequence,
                                    tokenizer::isGenerationEosToken,
                                    arm,
                                    512,
                                    Math.min(arm.most(), MtpDecoder.MAX_VERIFIED))) {
                                long replayedBefore = runtime.replayedQuanta();
                                List<Integer> tokens = decoder.generate(prompt, budget, token -> {});
                                long replayed = runtime.replayedQuanta() - replayedBefore;
                                if (reference == null) reference = tokens;
                                else
                                    assertEquals(
                                            reference, tokens, entry.get("name").asText() + " under " + arm);
                                MtpDecoder.Statistics statistics = decoder.statistics();
                                long nanos =
                                        statistics.verifyNanos + statistics.catchUpNanos + statistics.recursionNanos;
                                long drafts = 0;
                                for (int n = 0; n < statistics.draftLengths.length; n++)
                                    drafts += n * statistics.draftLengths[n];
                                for (String key : List.of(label(arm) + " " + category, label(arm) + " all")) {
                                    long[] sums = totals.computeIfAbsent(key, k -> new long[4]);
                                    sums[0] += statistics.outputTokens - 1;
                                    sums[1] += nanos;
                                    sums[2] += statistics.verifications;
                                    sums[3] += drafts;
                                }
                                System.out.printf(
                                        Locale.ROOT,
                                        "MTP_SCREEN round %d %s %s: %d tokens, %.1f tok/s, %.2f tokens/verification, %d replayed; %s%n",
                                        round,
                                        entry.get("name").asText(),
                                        label(arm),
                                        statistics.outputTokens,
                                        (statistics.outputTokens - 1) / (nanos / 1e9),
                                        statistics.outputTokens / (double) Math.max(1, statistics.verifications),
                                        replayed,
                                        statistics);
                            } finally {
                                sequence.complete();
                            }
                        }
                    }
                }
            } finally {
                runtime.close();
            }
        }
        for (var total : totals.entrySet()) {
            long[] sums = total.getValue();
            System.out.printf(
                    Locale.ROOT,
                    "MTP_SCREEN_TOTAL %s: %.1f tok/s, %.2f tokens/verification, %.2f drafts/verification%n",
                    total.getKey(),
                    sums[0] / (sums[1] / 1e9),
                    (sums[0] + 0.0) / sums[2],
                    (sums[3] + 0.0) / sums[2]);
        }
    }

    private static DraftLength parse(String arm) {
        int at = arm.indexOf('@');
        if (at < 0) return DraftLength.fixed(Integer.parseInt(arm));
        int dash = arm.indexOf('-');
        int slash = arm.indexOf('/');
        int end = slash < 0 ? at : slash;
        return new DraftLength(
                Integer.parseInt(arm.substring(0, dash)),
                Integer.parseInt(arm.substring(dash + 1, end)),
                slash < 0 ? 1 : Integer.parseInt(arm.substring(slash + 1, at)),
                Float.parseFloat(arm.substring(at + 1)));
    }

    private static String label(DraftLength arm) {
        return arm.gated()
                ? arm.least() + "-" + arm.most() + "/" + arm.step() + "@" + arm.threshold()
                : "MTP" + arm.most();
    }

    private static Qwen38Model load(Path artifact, CudaGpuMemory gpu, int contextTokens) throws Exception {
        var data = ArtifactReader.read(artifact);
        var profile = ArtifactProfile.of(data);
        assumeTrue(profile.speculation() == ArtifactProfile.Speculation.MTP, "not an MTP artifact");
        var residency =
                ResidencyPlanner.plan(data, profile, gpu.deviceMemoryInfo().freeBytes(), contextTokens);
        return Qwen38Model.load(artifact, data, gpu, ArtifactProfile.Speculation.MTP, residency.hostBacked());
    }
}
