package io.euhedral_execution.inference.core.model.qwen38;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.core.config.LatticeConfig;
import io.euhedral_execution.core.control_plane.ControlPlaneLattice;
import io.euhedral_execution.core.control_plane.ControlPlaneShard;
import io.euhedral_execution.core.impl.BaseCloneableObject;
import io.euhedral_execution.core.impl.DefaultExecutor;
import io.euhedral_execution.hardware_utils.topology.SystemInfo;
import io.euhedral_execution.inference.core.generation.GenerationTimingListener;
import io.euhedral_execution.inference.core.model.qwen38.EngineExecutionFixture.SamplingGpu;
import io.euhedral_execution.inference.core.model.qwen38.prefix.PrefixCache;
import io.euhedral_execution.inference.core.runtime.EuhedralInferenceRuntime;
import io.euhedral_execution.inference.core.sampling.GenerationConfig;
import io.euhedral_execution.inference.core.tokenizer.QwenTokenizer;
import java.lang.foreign.Arena;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.BitSet;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.IntStream;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.api.parallel.Execution;
import org.junit.jupiter.api.parallel.ExecutionMode;

@Execution(ExecutionMode.SAME_THREAD)
class SessionPrefixCacheTest {

    private static final Path TOKENIZER_DIRECTORY =
            Path.of(System.getProperty("euhedral.qwen.tokenizer-dir", "/mnt/shared/qwen38-quant/source/qwen"));
    private static final AtomicLong LATTICE_ID = new AtomicLong();
    private static QwenTokenizer tokenizer;

    @BeforeAll
    static void loadTokenizer() throws Exception {
        assumeTrue(Files.isRegularFile(TOKENIZER_DIRECTORY.resolve("tokenizer.json")));
        tokenizer = QwenTokenizer.load(TOKENIZER_DIRECTORY);
    }

    private record Harness(
            ExecutionPlan plan,
            SamplingGpu gpu,
            ControlPlaneLattice lattice,
            EuhedralInferenceRuntime runtime,
            PrefixCache cache) {}

    private static int[] prompt(int tokens) {
        return IntStream.range(0, tokens).map(i -> 1 + i % 5).toArray();
    }

    private static Harness harness(long cacheBytes, int interval) throws InterruptedException {
        int vocabularySize = tokenizer.generationEosTokenIds().stream()
                        .mapToInt(Integer::intValue)
                        .max()
                        .orElseThrow()
                + 1;
        var weights = ExecutionFixtures.statefulCompactWeights(vocabularySize);
        var plan = new ExecutionPlan(weights);
        var gpu = new SamplingGpu(vocabularySize);
        gpu.selectTokens(1, 2, 3, 4, 5, 6, 7, 8);
        BitSet available = SystemInfo.getPCpuSet();
        int cpu = available.nextSetBit(0);
        assumeTrue(cpu >= 0, "requires one available physical CPU for the lattice worker");
        BitSet cpus = new BitSet();
        cpus.set(cpu);
        var workers = new BaseCloneableObject(new DefaultExecutor());
        var shard = ControlPlaneShard.createBaseShard("PrefixCacheSessionTestShard", workers);
        var lattice = ControlPlaneLattice.getOrCreate(new LatticeConfig(
                "PrefixCacheSessionTestLattice-" + LATTICE_ID.incrementAndGet(), cpus, Duration.ofSeconds(10), shard));
        var runtime = new EuhedralInferenceRuntime(lattice, plan, gpu);
        Arena arena = Arena.ofShared();
        var cache = new PrefixCache(gpu, weights.config(), arena.allocate(cacheBytes), arena::close, interval);
        lattice.start();
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(30);
        while (lattice.getActiveWorkers() < 1 && System.nanoTime() < deadline) Thread.onSpinWait();
        assertEquals(1, lattice.getActiveWorkers(), "lattice worker did not register");
        return new Harness(plan, gpu, lattice, runtime, cache);
    }

    private static Session session(Harness h, long id) {
        var session = new Session(tokenizer, h.plan(), h.runtime(), h.gpu(), id, GenerationConfig.greedy(id));
        session.usePrefixCache(h.cache());
        return session;
    }

    private static void close(Harness h) {
        h.cache().close();
        h.runtime().close();
        h.lattice().close();
    }

    @Test
    @Timeout(value = 90, unit = TimeUnit.SECONDS)
    void aSecondRequestPrefillsOnlyTheTokensAfterTheStoredPrefix() throws Exception {
        Harness h = harness(32L << 20, 1024);
        try {
            // Chunks end at 512, 1024 and 1300; only 1024 is on the interval, and 512 and 1300 are not wanted.
            int[] prompt = prompt(1300);
            try (var first = session(h, 1)) {
                first.generate(prompt, 2, text -> {}, null);
            }
            assertEquals(1, h.cache().stats().captured());
            h.gpu().embeddingInputs.clear();
            try (var second = session(h, 2)) {
                second.generate(prompt, 2, text -> {}, null);
            }
            assertEquals(1, h.cache().stats().hits());
            assertEquals(1024, h.cache().stats().reusedTokens());
            assertEquals(276, h.gpu().embeddingInputs.get(0).length, "only the tokens after position 1024");
        } finally {
            close(h);
        }
    }

    @Test
    @Timeout(value = 90, unit = TimeUnit.SECONDS)
    void aPromptThatDivergesAfterTheStoredPrefixReusesIt() throws Exception {
        Harness h = harness(32L << 20, 1024);
        try {
            try (var first = session(h, 1)) {
                first.generate(prompt(1300), 1, text -> {}, null);
            }
            int[] other = prompt(1500);
            other[1100] = 9;
            h.gpu().embeddingInputs.clear();
            try (var second = session(h, 2)) {
                second.generate(other, 1, text -> {}, null);
            }
            assertEquals(1024, h.cache().stats().reusedTokens());
            assertEquals(476, h.gpu().embeddingInputs.get(0).length);
        } finally {
            close(h);
        }
    }

    @Test
    @Timeout(value = 90, unit = TimeUnit.SECONDS)
    void aPromptThatSharesNothingPrefillsInFull() throws Exception {
        Harness h = harness(32L << 20, 1024);
        try {
            try (var first = session(h, 1)) {
                first.generate(prompt(1300), 1, text -> {}, null);
            }
            int[] other = prompt(1300);
            other[0] = 7;
            h.gpu().embeddingInputs.clear();
            try (var second = session(h, 2)) {
                second.generate(other, 1, text -> {}, null);
            }
            assertEquals(0, h.cache().stats().hits());
            assertEquals(512, h.gpu().embeddingInputs.get(0).length, "the first prefill chunk is whole");
        } finally {
            close(h);
        }
    }

    @Test
    @Timeout(value = 90, unit = TimeUnit.SECONDS)
    void anIdenticalPromptStillPrefillsAtLeastOneChunk() throws Exception {
        Harness h = harness(32L << 20, 1024);
        try {
            int[] prompt = prompt(1024);
            try (var first = session(h, 1)) {
                first.generate(prompt, 1, text -> {}, null);
            }
            // The checkpoint at 512 is the last chunk boundary before the prompt ends; 1024 is the prompt itself.
            assertEquals(2, h.cache().stats().captured());
            h.gpu().embeddingInputs.clear();
            try (var second = session(h, 2)) {
                second.generate(prompt, 1, text -> {}, null);
                assertEquals(1025L, second.currentTokenPosition(), "the prompt plus the committed first token");
            }
            assertEquals(512, h.cache().stats().reusedTokens(), "a prompt never restores its own length");
            assertEquals(512, h.gpu().embeddingInputs.get(0).length);
        } finally {
            close(h);
        }
    }

    @Test
    @Timeout(value = 90, unit = TimeUnit.SECONDS)
    void aShortPromptStoresItsLastFullChunkBoundary() throws Exception {
        Harness h = harness(32L << 20, 2048);
        try {
            int[] prompt = prompt(700);
            try (var first = session(h, 1)) {
                first.generate(prompt, 1, text -> {}, null);
            }
            assertEquals(1, h.cache().stats().captured(), "the boundary at 512 leaves one chunk of 188");
            h.gpu().embeddingInputs.clear();
            try (var second = session(h, 2)) {
                second.generate(prompt, 1, text -> {}, null);
            }
            assertEquals(512, h.cache().stats().reusedTokens());
            assertEquals(188, h.gpu().embeddingInputs.get(0).length);
        } finally {
            close(h);
        }
    }

    @Test
    @Timeout(value = 90, unit = TimeUnit.SECONDS)
    void aRestoreIsReportedToTheTimingListener() throws Exception {
        Harness h = harness(64L << 20, 512);
        try {
            String text = "The quick brown fox jumps over the lazy dog. ".repeat(300);
            try (var first = session(h, 1)) {
                first.generate(text, 1, ignored -> {});
            }
            AtomicInteger restored = new AtomicInteger(-1);
            var listener = new GenerationTimingListener() {
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
                public void prefixRestored(int tokens, long nanos) {
                    restored.set(tokens);
                }
            };
            try (var second = session(h, 2)) {
                assertEquals(0, second.restoredPromptTokens());
                second.generate(text, 1, ignored -> {}, null, listener);
                assertEquals(restored.get(), second.restoredPromptTokens(), "the session reports what it restored");
            }
            assertTrue(restored.get() >= 512 && restored.get() % 512 == 0, "restored " + restored.get());
            assertEquals(h.cache().stats().reusedTokens(), restored.get());
        } finally {
            close(h);
        }
    }

    @Test
    @Timeout(value = 90, unit = TimeUnit.SECONDS)
    void aCancelledGenerationLeavesNoQuantumOrLeaseBehind() throws Exception {
        Harness h = harness(32L << 20, 1024);
        try {
            int[] prompt = prompt(2000);
            try (var first = session(h, 1)) {
                first.generate(prompt, 1, text -> {}, null);
            }
            assertEquals(2, h.cache().stats().captured(), "1024 on the interval and 1536, the last chunk boundary");
            for (int attempt = 0; attempt < 5; attempt++) {
                try (var second = session(h, 10 + attempt)) {
                    var future = second.generateAsync(prompt, 50, text -> {}, null);
                    second.cancel();
                    // Cancelling before, during or after the restore ends the generation normally, never in an error.
                    future.get(30, TimeUnit.SECONDS);
                    assertFalse(second.sequenceState().isExecutionClaimed());
                }
            }
            assertEquals(0, h.runtime().activeQuanta());
            assertEquals(5, h.cache().stats().hits(), "each attempt began by restoring");
        } finally {
            close(h);
        }
    }

    @Test
    @Timeout(value = 90, unit = TimeUnit.SECONDS)
    void aSecondPromptOfOneSessionStoresNothingAndRestoresNothing() throws Exception {
        Harness h = harness(32L << 20, 1024);
        try (var session = session(h, 1)) {
            // The first prompt is short: no checkpoint. The second continues the same sequence, whose positions no
            // longer match the second prompt's offsets, so a checkpoint keyed by its tokens would be state of
            // other tokens.
            session.generate(prompt(100), 1, text -> {}, null);
            assertEquals(0, h.cache().stats().captured());
            session.generate(prompt(800), 1, text -> {}, null);
            assertEquals(0, h.cache().stats().captured(), "nothing is stored from a continuation prompt");
            assertEquals(0, h.cache().stats().hits());
        } finally {
            close(h);
        }
    }

    @Test
    @Timeout(value = 90, unit = TimeUnit.SECONDS)
    void aCacheAttachedAfterAnUncachedFirstPromptIsNotUsed() throws Exception {
        Harness h = harness(32L << 20, 512);
        try (var session = new Session(tokenizer, h.plan(), h.runtime(), h.gpu(), 5, GenerationConfig.greedy(5))) {
            session.generate(prompt(100), 1, text -> {}, null);
            session.usePrefixCache(h.cache());
            assertEquals(1, session.generate(prompt(800), 1, text -> {}, null).size());
            assertEquals(0, h.cache().stats().captured());
        } finally {
            close(h);
        }
    }

    @Test
    @Timeout(value = 90, unit = TimeUnit.SECONDS)
    void aSessionWithAnotherChunkSizeCannotUseTheCache() throws Exception {
        Harness h = harness(8L << 20, 1024);
        try (var session = new Session(
                tokenizer, h.plan(), h.runtime(), h.gpu(), 1, GenerationConfig.greedy(1), 256, ignored -> {})) {
            assertThrows(IllegalStateException.class, () -> session.usePrefixCache(h.cache()));
        } finally {
            close(h);
        }
    }

    @Test
    @Timeout(value = 90, unit = TimeUnit.SECONDS)
    void aSessionWithoutTheCacheChangesNothing() throws Exception {
        Harness h = harness(8L << 20, 1024);
        try {
            try (var session = new Session(tokenizer, h.plan(), h.runtime(), h.gpu(), 9, GenerationConfig.greedy(9))) {
                session.generate(prompt(1300), 1, text -> {}, null);
            }
            assertEquals(0, h.cache().stats().lookups());
            assertEquals(0, h.cache().stats().captured());
        } finally {
            close(h);
        }
    }
}
