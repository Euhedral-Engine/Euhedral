package io.euhedral_execution.inference.core;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.hardware_utils.topology.SystemInfo;
import io.euhedral_execution.inference.core.generation.GenerationSession;
import io.euhedral_execution.inference.core.generation.GenerationTimingListener;
import io.euhedral_execution.inference.core.generation.SessionOptions;
import io.euhedral_execution.inference.core.model.qwen38.Execution;
import io.euhedral_execution.inference.core.model.qwen38.Execution.PassStatistics;
import io.euhedral_execution.inference.core.model.qwen38.Quantum;
import io.euhedral_execution.inference.core.model.qwen38.Qwen38Runtime;
import io.euhedral_execution.inference.core.sampling.GenerationConfig;
import io.euhedral_execution.inference.core.testing.ModelGroup;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/// The unquantized Qwen3.8 model (a `bf16` artifact) read through the disk tier by one sequence: greedy decode with
/// speculation off and on (tokens per second, the disk and transfer bytes each quantum kind issues, tokens per
/// verification) and prompt prefill (a 2048-token prompt, a 512-token turn appended to a conversation, with the prefix
/// cache off and on). It prints the numbers a multi-sequence plan is sized from, and asserts that speculation does
/// not change greedy output. Skipped unless -Peuhedral.qwen.bf16-artifact names the artifact (local disk) and
/// -Peuhedral.qwen.tokenizer-dir its tokenizer.
// Own JVM: loads the engine several times.
@ModelGroup.OwnJvm
class Bf16BaselineCudaIntegrationTest {

    private static final int NEW_TOKENS = 64;
    private static final int PROMPT_TOKENS = 2048;
    private static final int TURN_TOKENS = 512;
    private static final String SENTENCE =
            "The heat pump moves thermal energy between a cold source and a warm sink by cycling a refrigerant"
                    + " through an evaporator, a compressor, a condenser and an expansion valve. ";

    @Test
    @Timeout(value = 7200, unit = TimeUnit.SECONDS)
    void decodeAndPrefillCostsOfTheUnquantizedModel() throws Exception {
        String library = System.getProperty("euhedral.cuda.library");
        assumeTrue(library != null && Files.isRegularFile(Path.of(library)), "no CUDA library");
        String artifactProperty = System.getProperty("euhedral.qwen.bf16-artifact");
        assumeTrue(artifactProperty != null && Files.isRegularFile(Path.of(artifactProperty)), "no BF16 artifact");
        Path artifact = Path.of(artifactProperty);
        Path tokenizer = Path.of(System.getProperty("euhedral.qwen.tokenizer-dir"));
        Path lib = Path.of(library);
        // The pinned host budget decides how much of the model is read from disk: 24 GiB leaves about half of it there.
        System.setProperty(
                "euhedral.residency.host-budget-bytes",
                Long.toString(Long.getLong("euhedral.baseline.host-budget-gib", 24L) << 30));
        try {
            run(artifact, tokenizer, lib);
        } finally {
            System.clearProperty("euhedral.residency.host-budget-bytes");
        }
    }

    private static void run(Path artifact, Path tokenizer, Path lib) throws Exception {
        // One engine serves both modes (statistics are read as differences) and the cold prefill; a second one
        // keeps a prefix cache.
        try (InferenceEngine engine = InferenceEngine.load(config(artifact, tokenizer, lib, 0L))) {
            List<Integer> plain = decode(engine, false);
            List<Integer> speculative = decode(engine, true);
            assertEquals(plain, speculative, "speculation must not change greedy output");
            prefill(engine, "prefix cache off", false);
        }
        try (InferenceEngine engine = InferenceEngine.load(config(artifact, tokenizer, lib, 1L << 30))) {
            prefill(engine, "prefix cache on", true);
        }
    }

    private static InferenceConfig config(Path artifact, Path tokenizer, Path library, long prefixCacheBytes) {
        return new InferenceConfig(
                artifact,
                tokenizer,
                library,
                SystemInfo.getPCpuSet(),
                4096,
                Duration.ofSeconds(60),
                prefixCacheBytes,
                2048);
    }

    private static List<Integer> decode(InferenceEngine engine, boolean speculation) throws Exception {
        String label = speculation ? "speculation on" : "speculation off";
        {
            var dense = (Qwen38Runtime) engine.modelRuntime();
            Execution execution = dense.execution();
            PassStatistics[] before = snapshot(execution);
            System.out.printf(
                    "[%s] %d MiB of weights read from disk; device after load: %s%n",
                    label, dense.diskBackedWeightBytes() >> 20, deviceMemory());
            int[] prompt = engine.tokenizer()
                    .encodeWithModelSpecialTokens("<|im_start|>user\nExplain how a heat pump works.<|im_end|>\n"
                            + "<|im_start|>assistant\n<think>\n\n</think>\n\n");
            var steps = new AtomicLong();
            var outputs = new AtomicLong();
            var timing = new GenerationTimingListener() {
                @Override
                public void promptEncoded(long nanos, int promptTokens) {}

                @Override
                public void prefillQuantum(long startNanos, long executedNanos, int tokens) {}

                @Override
                public void firstTokenSelected(long nanos, int tokenId) {}

                @Override
                public void decodeQuantum(
                        long startNanos,
                        long executedNanos,
                        long selectedNanos,
                        boolean sampled,
                        int selectedTokenId) {}

                @Override
                public void speculativeStep(long startNanos, long executedNanos, int out, int acceptedDrafts) {
                    steps.incrementAndGet();
                    outputs.addAndGet(out);
                }
            };
            List<Integer> tokens;
            long start = System.nanoTime();
            try (GenerationSession session =
                    engine.createGenerationSession(GenerationConfig.greedy(91L), new SessionOptions(speculation))) {
                tokens = session.generateAsync(prompt, NEW_TOKENS, text -> {}, null, timing)
                        .get(60, TimeUnit.MINUTES);
            }
            double seconds = (System.nanoTime() - start) / 1e9;
            System.out.printf(
                    "[%s] %d tokens (%d-token prompt) in %.1f s (%.3f tok/s, prefill included)%n",
                    label, tokens.size(), prompt.length, seconds, tokens.size() / seconds);
            if (steps.get() > 0)
                System.out.printf(
                        "[%s] %d verifications, %.2f tokens per verification%n",
                        label, steps.get(), (double) outputs.get() / steps.get());
            printStatistics(label, execution, before);
            return tokens;
        }
    }

    private static void prefill(InferenceEngine engine, String label, boolean cached) throws Exception {
        {
            Execution execution = ((Qwen38Runtime) engine.modelRuntime()).execution();
            int[] prompt = tokens(engine, PROMPT_TOKENS);
            int[] turn = tokens(engine, TURN_TOKENS);
            for (int round = 0; round < (cached ? 2 : 1); round++) {
                String what = label + ", " + PROMPT_TOKENS + "-token prompt, " + (round == 0 ? "cold" : "repeated");
                try (GenerationSession session =
                        engine.createGenerationSession(GenerationConfig.greedy(91L), new SessionOptions(false))) {
                    measure(what, execution, session, prompt);
                    if (round == 0)
                        measure(label + ", " + TURN_TOKENS + "-token turn appended", execution, session, turn);
                }
            }
        }
    }

    /// Generates one token from `ids` and prints what the prefill quanta of that call cost.
    private static void measure(String what, Execution execution, GenerationSession session, int[] ids)
            throws Exception {
        PassStatistics before = execution.statistics(Quantum.ExecutionKind.PREFILL);
        var prefillNanos = new AtomicLong();
        var timing = new GenerationTimingListener() {
            @Override
            public void promptEncoded(long nanos, int promptTokens) {}

            @Override
            public void prefillQuantum(long startNanos, long executedNanos, int tokens) {
                prefillNanos.addAndGet(executedNanos - startNanos);
            }

            @Override
            public void firstTokenSelected(long nanos, int tokenId) {}

            @Override
            public void decodeQuantum(
                    long startNanos, long executedNanos, long selectedNanos, boolean sampled, int selectedTokenId) {}
        };
        long start = System.nanoTime();
        session.generateAsync(ids, 1, text -> {}, null, timing).get(30, TimeUnit.MINUTES);
        double wall = (System.nanoTime() - start) / 1e9;
        PassStatistics after = execution.statistics(Quantum.ExecutionKind.PREFILL);
        long quanta = after.quanta() - before.quanta();
        System.out.printf(
                "[%s] %d tokens (%d restored): %d PREFILL quanta, %.1f s in prefill quanta (%.1f s call),"
                        + " disk %.0f MiB and transfer %.0f MiB per quantum%n",
                what,
                ids.length,
                session.restoredPromptTokens(),
                quanta,
                prefillNanos.get() / 1e9,
                wall,
                quanta == 0 ? 0 : (after.diskReadBytes() - before.diskReadBytes()) / (double) quanta / (1 << 20),
                quanta == 0 ? 0 : (after.transferBytes() - before.transferBytes()) / (double) quanta / (1 << 20));
    }

    private static int[] tokens(InferenceEngine engine, int count) {
        int[] ids = engine.tokenizer().encodeText(SENTENCE.repeat(count / 16 + 8));
        assertTrue(ids.length >= count, "the filler text encodes to at least " + count + " tokens");
        return Arrays.copyOf(ids, count);
    }

    private static PassStatistics[] snapshot(Execution execution) {
        return Arrays.stream(Quantum.ExecutionKind.values())
                .map(execution::statistics)
                .toArray(PassStatistics[]::new);
    }

    private static void printStatistics(String label, Execution execution, PassStatistics[] before) {
        long quanta = 0;
        long disk = 0;
        long transfer = 0;
        for (Quantum.ExecutionKind kind : Quantum.ExecutionKind.values()) {
            PassStatistics now = execution.statistics(kind);
            PassStatistics was = before[kind.ordinal()];
            var s = new PassStatistics(
                    now.quanta() - was.quanta(),
                    now.batchedQuanta() - was.batchedQuanta(),
                    now.segments() - was.segments(),
                    now.rows() - was.rows(),
                    now.diskReadBytes() - was.diskReadBytes(),
                    now.transferBytes() - was.transferBytes());
            if (s.quanta() == 0) continue;
            quanta += s.quanta();
            disk += s.diskReadBytes();
            transfer += s.transferBytes();
            System.out.printf(
                    "[%s] %s: %d quanta, %d rows, disk %.0f MiB and transfer %.0f MiB per quantum%n",
                    label,
                    kind,
                    s.quanta(),
                    s.rows(),
                    s.diskBytesPerQuantum() / (1 << 20),
                    s.transferBytesPerQuantum() / (1 << 20));
        }
        System.out.printf(
                "[%s] all kinds: %d quanta, disk %d MiB, transfer %d MiB%n", label, quanta, disk >> 20, transfer >> 20);
    }

    private static String deviceMemory() {
        try {
            Process process = new ProcessBuilder(
                            "nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader")
                    .redirectErrorStream(true)
                    .start();
            try (var reader = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
                return reader.readLine();
            }
        } catch (Exception failure) {
            return "unavailable: " + failure;
        }
    }
}
