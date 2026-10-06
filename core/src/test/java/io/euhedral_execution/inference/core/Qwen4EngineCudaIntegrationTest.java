package io.euhedral_execution.inference.core;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.inference.core.sampling.GenerationConfig;
import io.euhedral_execution.inference.core.scheduling.GenerationSession;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.BitSet;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;

/// Flash-Next through the engine's own entry point: `InferenceEngine.load` takes the real EDRL v3 artifact, a session
/// generates greedily with the engine's tokenizer and sampler, and the tokens are the upstream model's (the
/// reference harness's `generate`: "The capital of France is" continues " Paris. The capital of Germany is Berlin").
class Qwen4EngineCudaIntegrationTest {

    private static final int[] PROMPT = {760, 6511, 314, 9338, 369};
    private static final int[] CONTINUATION = {11751, 13, 561, 6511, 314, 9564, 369, 19241};

    private static InferenceConfig config(int context) {
        Path artifact = Path.of(System.getProperty(
                "euhedral.qwen4.artifact", "/home/brandon/models/qwen3_8_flash_next/qwen3_8_flash_next_nvfp4.edrl"));
        assumeTrue(Files.isRegularFile(artifact), "no Flash-Next artifact at " + artifact);
        Path tokenizers =
                Path.of(System.getProperty("euhedral.qwen4.tokenizer-dir", "/mnt/shared/qwen38-flash-next/nvfp4"));
        assumeTrue(Files.isRegularFile(tokenizers.resolve("tokenizer.json")), "no tokenizer at " + tokenizers);
        BitSet cpus = new BitSet();
        cpus.set(0);
        return new InferenceConfig(
                artifact,
                tokenizers,
                Path.of(System.getProperty("euhedral.cuda.library")),
                cpus,
                context,
                Duration.ofSeconds(30));
    }

    @Test
    void generatesTheUpstreamContinuationThroughTheEngine() throws Exception {
        try (InferenceEngine engine = InferenceEngine.load(config(4096))) {
            assertEquals(248320, engine.vocabularySize());
            assertEquals(262144, engine.maxPositionEmbeddings());
            int[] prompt =
                    engine.tokenizePromptAsync("The capital of France is").get(30, TimeUnit.SECONDS);
            assertArrayEquals(PROMPT, prompt, "the Java tokenizer must encode as the reference's");
            GenerationSession session = engine.createGenerationSession(GenerationConfig.greedy(1));
            StringBuilder text = new StringBuilder();
            AtomicInteger quanta = new AtomicInteger();
            List<Integer> tokens = session.generateAsync(
                            prompt,
                            CONTINUATION.length,
                            text::append,
                            null,
                            new io.euhedral_execution.inference.core.scheduling.GenerationTimingListener() {
                                @Override
                                public void promptEncoded(long nanos, int promptTokens) {}

                                @Override
                                public void prefillQuantum(long startNanos, long executedNanos, int tokens) {
                                    quanta.incrementAndGet();
                                }

                                @Override
                                public void firstTokenSelected(long nanos, int tokenId) {}

                                @Override
                                public void decodeQuantum(
                                        long startNanos,
                                        long executedNanos,
                                        long selectedNanos,
                                        boolean sampled,
                                        int id) {
                                    quanta.incrementAndGet();
                                }
                            })
                    .get(5, TimeUnit.MINUTES);
            assertEquals(List.of(11751, 13, 561, 6511, 314, 9564, 369, 19241), tokens);
            assertEquals(" Paris. The capital of Germany is Berlin", text.toString());
            // Every generated token is committed to the sequence, the last one without sampling beyond the budget.
            assertEquals(PROMPT.length + CONTINUATION.length, session.currentTokenPosition());
            assertTrue(quanta.get() >= 1 + CONTINUATION.length - 1, "one prefill and a decode step per further token");
            assertFalse(session.expectsFirstPrompt());
            session.close();
            assertTrue(session.isClosed());
            assertNoLeftovers(engine);
        }
    }

    /// Nothing a generation used is still held: no lease, no pinned slot, no transfer in flight, no step, no
    /// model-specific thread.
    private static void assertNoLeftovers(InferenceEngine engine) {
        Qwen4Runtime runtime = engine.qwen4Runtime();
        var cache = runtime.storage().model().expertCache();
        cache.checkQuiescent();
        assertEquals(0, cache.openLeaseCount());
        assertEquals(0, runtime.plan().stepsInFlight());
        for (Thread thread : Thread.getAllStackTraces().keySet()) {
            String name = thread.getName();
            assertFalse(name.contains("flash-next"), "model-specific thread: " + name);
            assertFalse(name.contains("expert-transfer-completion"), name);
            assertFalse(name.contains("qwen4-expert-acquire"), name);
        }
    }

    @Test
    void aGenerationHoldsNoWorkerWhileItWaits() throws Exception {
        // One lattice worker: a step that blocked it while waiting for the device, an expert or the executor would
        // starve every other piece of host work, and the generation itself.
        try (InferenceEngine engine = InferenceEngine.load(config(4096))) {
            int[] prompt =
                    engine.tokenizePromptAsync("The capital of France is").get(30, TimeUnit.SECONDS);
            GenerationSession session = engine.createGenerationSession(GenerationConfig.greedy(1));
            var generation = session.generateAsync(prompt, 24, text -> {}, null, null);
            int completed = 0;
            long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(30);
            while (!generation.isDone() && System.nanoTime() < deadline && completed < 200) {
                int value = completed;
                assertEquals(value, engine.<Integer>onWorker(() -> value).get(10, TimeUnit.SECONDS));
                completed++;
            }
            assertTrue(completed >= 20, "host work progressed while the generation was in flight: " + completed);
            assertFalse(generation.isDone() && completed == 0);
            assertEquals(24, generation.get(5, TimeUnit.MINUTES).size());
            session.close();
            assertNoLeftovers(engine);
        }
    }

    @Test
    void cancellingInTheMiddleOfAPrefillLeavesNothingOpenAndTheEngineUsable() throws Exception {
        try (InferenceEngine engine = InferenceEngine.load(config(4096))) {
            int[] prompt = new int[1536];
            for (int i = 0; i < prompt.length; i++) prompt[i] = 1000 + (i * 7919) % 20000;
            GenerationSession session = engine.createGenerationSession(GenerationConfig.greedy(1));
            var generation = session.generateAsync(prompt, 8, text -> {}, null, null);
            Thread.sleep(1500);
            long begun = System.nanoTime();
            session.cancel();
            var tokens = generation.get(60, TimeUnit.SECONDS);
            long millis = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - begun);
            assertTrue(tokens.isEmpty());
            assertTrue(millis < 20_000, "the step stopped at a layer boundary, not after the chunk: " + millis + " ms");
            session.close();
            assertNoLeftovers(engine);
            // Experts, transfers and the executor are exactly as usable as before.
            int[] france =
                    engine.tokenizePromptAsync("The capital of France is").get(30, TimeUnit.SECONDS);
            GenerationSession again = engine.createGenerationSession(GenerationConfig.greedy(1));
            assertEquals(
                    List.of(11751, 13, 561, 6511, 314, 9564, 369, 19241),
                    again.generateAsync(france, 8, text -> {}, null, null).get(5, TimeUnit.MINUTES));
            again.close();
            assertNoLeftovers(engine);
        }
    }

    @Test
    void closingTheEngineLeavesNoAttachedSourceAndNoActiveWork() throws Exception {
        InferenceEngine engine = InferenceEngine.load(config(4096));
        Qwen4Runtime runtime = engine.qwen4Runtime();
        int[] prompt = engine.tokenizePromptAsync("The capital of France is").get(30, TimeUnit.SECONDS);
        GenerationSession session = engine.createGenerationSession(GenerationConfig.greedy(1));
        session.generateAsync(prompt, 4, text -> {}, null, null).get(5, TimeUnit.MINUTES);
        assertTrue(runtime.executionTasks().isAttached());
        engine.close();
        assertFalse(runtime.executionTasks().isAttached());
        assertFalse(runtime.hostTasks().isAttached());
        assertEquals(0, runtime.executionTasks().activeTasks());
        assertEquals(0, runtime.hostTasks().activeTasks());
    }

    @Test
    void cancellationStopsGenerationAndTheEngineClosesCleanly() throws Exception {
        InferenceEngine engine = InferenceEngine.load(config(4096));
        try {
            int[] prompt =
                    engine.tokenizePromptAsync("The capital of France is").get(30, TimeUnit.SECONDS);
            GenerationSession session = engine.createGenerationSession(GenerationConfig.greedy(1));
            AtomicInteger pieces = new AtomicInteger();
            var future = session.generateAsync(
                    prompt,
                    200,
                    text -> {
                        if (pieces.incrementAndGet() == 2) session.cancel();
                    },
                    null,
                    null);
            List<Integer> tokens = future.get(5, TimeUnit.MINUTES);
            assertTrue(session.isCancelled());
            assertTrue(tokens.size() < 200, "cancellation must stop the generation early, got " + tokens.size());
            // A cancelled session cannot take another prompt.
            try {
                session.generateAsync(prompt, 1, text -> {}, null, null);
                throw new AssertionError("a cancelled session accepted a prompt");
            } catch (IllegalStateException expected) {
                // as the dense session
            }
            session.close();
        } finally {
            engine.close();
        }
        assertTrue(engine.isClosed());
    }
}
