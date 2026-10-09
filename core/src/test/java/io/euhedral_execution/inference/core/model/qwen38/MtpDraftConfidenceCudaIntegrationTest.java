package io.euhedral_execution.inference.core.model.qwen38;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import io.euhedral_execution.inference.core.model.qwen38.artifact.ArtifactReader;
import io.euhedral_execution.inference.core.model.qwen38.loader.ResidencyPlanner;
import io.euhedral_execution.inference.core.model.qwen38.speculative.MtpDecoder;
import io.euhedral_execution.inference.core.runtime.PullingLattice;
import io.euhedral_execution.inference.core.testing.ModelGroup;
import io.euhedral_execution.inference.core.tokenizer.QwenTokenizer;
import java.io.BufferedWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/// MTP drafting confidence at every output position, for screening speculation policies offline
/// (tools/mtp_loop_screen.py): the decoder drafts `-Peuhedral.mtp.drafts` tokens (default 16) from each position and
/// commits one token per step, and the report holds, per position, the drafts with the draft head's log-probability
/// of each, then each prompt's generated tokens. Acceptance of any prefix of a position's drafts follows from the
/// generated tokens, which are greedy decode's.
///
/// Prompts come from `-Peuhedral.mtp.prompts` (JSON lines from tools/mtp_draft_prompts.py); the report goes to
/// `-Peuhedral.mtp.report`. The artifact is `-Peuhedral.speculative.artifact` (an MTP artifact).
@ModelGroup.OwnJvm // plans its own residency for the prompts' context
class MtpDraftConfidenceCudaIntegrationTest {

    @Test
    @Timeout(value = 7200, unit = TimeUnit.SECONDS)
    void recordDraftConfidence() throws Throwable {
        Path artifact = Path.of(System.getProperty("euhedral.speculative.artifact", ""));
        Path prompts = Path.of(System.getProperty("euhedral.mtp.prompts", ""));
        String report = System.getProperty("euhedral.mtp.report", "");
        assumeTrue(
                Files.isRegularFile(artifact) && Files.isRegularFile(prompts) && !report.isEmpty(),
                "needs an MTP artifact, a prompt file and a report path");
        int budget = Integer.getInteger("euhedral.speculative.tokens", 512);
        int drafts = Integer.getInteger("euhedral.mtp.drafts", MtpDecoder.MAX_DRAFTS);
        int context = Integer.getInteger("euhedral.mtp.context", 16384);
        QwenTokenizer tokenizer = QwenTokenizer.load(
                Path.of(System.getProperty("euhedral.qwen.tokenizer-dir", "/mnt/shared/qwen38-quant/source/qwen")));
        var mapper = new ObjectMapper();
        List<JsonNode> entries = Files.readAllLines(prompts).stream()
                .filter(line -> !line.isBlank())
                .map(line -> {
                    try {
                        return mapper.readTree(line);
                    } catch (java.io.IOException failure) {
                        throw new java.io.UncheckedIOException(failure);
                    }
                })
                .toList();
        try (CudaGpuMemory gpu = new CudaGpuMemory(Path.of(System.getProperty("euhedral.cuda.library")));
                Qwen38Model model = load(artifact, gpu, context);
                var lattice = new PullingLattice();
                BufferedWriter out = Files.newBufferedWriter(Path.of(report))) {
            var plan = new ExecutionPlan(model.weights(), model.staging());
            var runtime = new Execution(lattice, plan, gpu);
            try {
                long id = 700;
                for (int index = 0; index < entries.size(); index++) {
                    JsonNode entry = entries.get(index);
                    int[] prompt = tokenizer.encodeWithModelSpecialTokens(
                            entry.get("text").asText());
                    if (prompt.length + budget + drafts > context)
                        throw new IllegalArgumentException(entry.get("name").asText() + " needs more context");
                    var sequence = new Sequence(++id);
                    int promptIndex = index;
                    long started = System.nanoTime();
                    try (var decoder = new MtpDecoder(
                            runtime, plan, gpu, sequence, tokenizer::isGenerationEosToken, drafts, 512, 1)) {
                        decoder.observe(
                                (position, current, proposal, logProbabilities, accepted) -> {
                                    try {
                                        out.write("{\"prompt\":" + promptIndex + ",\"position\":" + position
                                                + ",\"current\":" + current + ",\"drafts\":"
                                                + Arrays.toString(proposal) + ",\"logp\":"
                                                + Arrays.toString(logProbabilities) + "}\n");
                                    } catch (java.io.IOException failure) {
                                        throw new java.io.UncheckedIOException(failure);
                                    }
                                },
                                true);
                        List<Integer> tokens = decoder.generate(prompt, budget, token -> {}, null);
                        out.write("{\"prompt\":" + index + ",\"name\":" + mapper.writeValueAsString(entry.get("name"))
                                + ",\"category\":" + mapper.writeValueAsString(entry.get("category"))
                                + ",\"thinking\":" + entry.get("thinking").asBoolean() + ",\"promptTokens\":"
                                + prompt.length + ",\"tokens\":" + tokens + "}\n");
                        out.flush();
                        System.out.printf(
                                "MTP_CONFIDENCE %s: %d prompt tokens, %d generated, %.1f s%n",
                                entry.get("name").asText(),
                                prompt.length,
                                tokens.size(),
                                (System.nanoTime() - started) / 1e9);
                    } finally {
                        sequence.complete();
                    }
                }
            } finally {
                runtime.close();
            }
        }
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
