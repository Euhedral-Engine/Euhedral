package io.euhedral_execution.inference.core;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.inference.core.generation.GenerationSession;
import io.euhedral_execution.inference.core.generation.SessionOptions;
import io.euhedral_execution.inference.core.model.qwen38.Qwen38Runtime;
import io.euhedral_execution.inference.core.sampling.GenerationConfig;
import io.euhedral_execution.inference.core.testing.ModelGroup;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/// The `bf16` artifact serving twelve speculative sequences at once on four workers, its weights read from disk every
/// pass and its LM head mapped from host memory: the configuration that stalled with the GPU idle and a worker
/// spinning in a CUDA call. Every session keeps generating for the run's duration (`euhedral.stress.minutes`,
/// default 8), and some token must arrive at least every `euhedral.stress.stall-seconds` (default 90; a pass takes
/// about 4 s). A stalled run cannot close its engine, so it ends the JVM. Skipped unless
/// -Peuhedral.qwen.bf16-artifact names the artifact and -Peuhedral.qwen.tokenizer-dir its tokenizer.
// Own JVM: sets the global residency properties.
@ModelGroup.OwnJvm
class DiskTierManySequencesCudaIntegrationTest {

    private static final int SEQUENCES = Integer.getInteger("euhedral.stress.sequences", 12);
    private static final int NEW_TOKENS = 192;
    private static final String[] TOPICS = {
        "how a heat pump moves thermal energy", "the life of a lighthouse keeper", "the planets of the solar system",
        "how bread rises", "the history of the printing press", "how a bicycle stays upright",
        "why the sky is blue", "the water cycle", "how vaccines train the immune system",
        "the rules of chess", "how a compiler works", "the migration of arctic terns",
    };

    @Test
    @Timeout(value = 60, unit = TimeUnit.MINUTES, threadMode = Timeout.ThreadMode.SEPARATE_THREAD)
    void twelveSpeculativeSequencesKeepGenerating() throws Exception {
        String library = System.getProperty("euhedral.cuda.library");
        assumeTrue(library != null && Files.isRegularFile(Path.of(library)), "no CUDA library");
        String artifact = System.getProperty("euhedral.qwen.bf16-artifact");
        assumeTrue(artifact != null && Files.isRegularFile(Path.of(artifact)), "no BF16 artifact");
        String tokenizer = System.getProperty("euhedral.qwen.tokenizer-dir");
        assumeTrue(tokenizer != null, "no tokenizer directory");
        long minutes = Long.getLong("euhedral.stress.minutes", 8L);
        long stallNanos = TimeUnit.SECONDS.toNanos(Long.getLong("euhedral.stress.stall-seconds", 90L));
        BitSet workers = new BitSet();
        workers.set(2, 6);
        var config = new InferenceConfig(
                Path.of(artifact),
                Path.of(tokenizer),
                Path.of(library),
                workers,
                6144,
                Duration.ofSeconds(60),
                0,
                2048,
                null,
                SEQUENCES);
        System.setProperty("euhedral.residency.host-budget-bytes", Long.toString(28L << 30));
        System.setProperty("euhedral.residency.map-output-head", "true");
        try (InferenceEngine engine = InferenceEngine.load(config)) {
            var runtime = (Qwen38Runtime) engine.modelRuntime();
            assertTrue(runtime.diskBackedWeightBytes() > 0, "weights beyond the host budget are read from disk");
            int count = Math.min(engine.maxSequences(), SEQUENCES);
            System.out.printf(
                    "[stress] pid %d, %d sequences, %d MiB from disk per pass%n",
                    ProcessHandle.current().pid(), count, runtime.diskBackedWeightBytes() >> 20);
            AtomicLong lastToken = new AtomicLong(System.nanoTime());
            AtomicLong tokens = new AtomicLong();
            long start = System.nanoTime();
            long end = start + TimeUnit.MINUTES.toNanos(minutes);
            List<GenerationSession> sessions = new ArrayList<>();
            List<CompletableFuture<?>> running = new ArrayList<>();
            int[] round = new int[count];
            for (int i = 0; i < count; i++) {
                sessions.add(engine.createGenerationSession(GenerationConfig.greedy(7L + i), SessionOptions.DEFAULT));
                running.add(null);
            }
            try {
                while (true) {
                    long now = System.nanoTime();
                    for (int i = 0; i < count; i++) {
                        CompletableFuture<?> future = running.get(i);
                        if (future != null && !future.isDone()) continue;
                        if (future != null) future.join();
                        if (now >= end) continue;
                        // A fresh conversation each round: a prefill, then speculative decode.
                        sessions.get(i).close();
                        sessions.set(
                                i,
                                engine.createGenerationSession(
                                        GenerationConfig.greedy(7L + i), SessionOptions.DEFAULT));
                        String text = "<|im_start|>user\nExplain " + TOPICS[(i + round[i]++) % TOPICS.length]
                                + " in detail.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
                        int[] prompt = engine.tokenizer().encodeWithModelSpecialTokens(text);
                        running.set(
                                i,
                                sessions.get(i)
                                        .generateAsync(
                                                prompt,
                                                NEW_TOKENS,
                                                piece -> {
                                                    tokens.incrementAndGet();
                                                    lastToken.set(System.nanoTime());
                                                },
                                                null,
                                                null));
                    }
                    if (now >= end && running.stream().allMatch(f -> f == null || f.isDone())) break;
                    if (now - lastToken.get() > stallNanos) {
                        System.out.printf(
                                "[stress] STALL: no token for %d s after %.0f s and %d tokens%n",
                                (now - lastToken.get()) / 1_000_000_000L, (now - start) / 1e9, tokens.get());
                        System.out.flush();
                        // The stuck device work holds the engine: it cannot close.
                        Runtime.getRuntime().halt(97);
                    }
                    Thread.sleep(1000);
                }
            } finally {
                for (GenerationSession session : sessions) session.close();
            }
            double seconds = (System.nanoTime() - start) / 1e9;
            System.out.printf(
                    "[stress] %d text pieces in %.0f s (%.2f per second), no stall%n",
                    tokens.get(), seconds, tokens.get() / seconds);
        } finally {
            System.clearProperty("euhedral.residency.host-budget-bytes");
            System.clearProperty("euhedral.residency.map-output-head");
        }
    }
}
