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
        }
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
