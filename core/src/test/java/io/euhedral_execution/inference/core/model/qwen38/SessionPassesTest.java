package io.euhedral_execution.inference.core.model.qwen38;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.core.config.LatticeConfig;
import io.euhedral_execution.core.control_plane.ControlPlaneLattice;
import io.euhedral_execution.core.control_plane.ControlPlaneShard;
import io.euhedral_execution.core.impl.BaseCloneableObject;
import io.euhedral_execution.core.impl.DefaultExecutor;
import io.euhedral_execution.hardware_utils.topology.SystemInfo;
import io.euhedral_execution.inference.core.runtime.LatticeTesting;
import io.euhedral_execution.inference.core.sampling.GenerationConfig;
import io.euhedral_execution.inference.core.tokenizer.QwenTokenizer;
import io.euhedral_execution.inference.core.tokenizer.TokenizerFiles;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.BitSet;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.api.parallel.ExecutionMode;

/// Concurrent sessions on one execution offer their decode steps to shared passes.
@org.junit.jupiter.api.parallel.Execution(ExecutionMode.SAME_THREAD)
class SessionPassesTest {

    private static final Path TOKENIZER_DIRECTORY =
            Path.of(System.getProperty("euhedral.qwen.tokenizer-dir", "unset-euhedral.qwen.tokenizer-dir"));
    private static final AtomicLong LATTICE_ID = new AtomicLong();
    private static QwenTokenizer tokenizer;

    @BeforeAll
    static void loadTokenizer() throws Exception {
        assumeTrue(
                Files.isRegularFile(TOKENIZER_DIRECTORY.resolve("tokenizer.json")),
                "skipped: no tokenizer.json in " + TOKENIZER_DIRECTORY + "; pass -Peuhedral.qwen.tokenizer-dir=...");
        tokenizer = QwenTokenizer.load(TokenizerFiles.fromCheckpoint(TOKENIZER_DIRECTORY));
    }

    @Test
    @Timeout(value = 90, unit = TimeUnit.SECONDS)
    void overlappingSessionsShareDecodePassesAndAnEarlyEndDoesNotStallTheOthers() throws Exception {
        int vocabularySize = testVocabularySize();
        var plan = new ExecutionPlan(ExecutionFixtures.statefulCompactWeights(vocabularySize));
        var gpu = new SamplingGpu(vocabularySize);
        var lattice = createLattice();
        var runtime = new Execution(lattice, plan, gpu);
        var first = new Session(tokenizer, plan, runtime, gpu, 910, GenerationConfig.greedy(1L));
        var second = new Session(tokenizer, plan, runtime, gpu, 911, GenerationConfig.greedy(2L));
        var early = new Session(tokenizer, plan, runtime, gpu, 912, GenerationConfig.greedy(3L));
        var secondResult = new AtomicReference<CompletableFuture<List<Integer>>>();
        var earlyResult = new AtomicReference<CompletableFuture<List<Integer>>>();
        var started = new AtomicBoolean();
        // The others start from the first generation's first host-row copy, so their decodes overlap; the third
        // ends after two tokens while the first two still have steps to run.
        gpu.onRowCopy = () -> {
            if (!started.compareAndSet(false, true)) return;
            secondResult.set(second.generateAsync("!", 12, text -> {}, null, null));
            earlyResult.set(early.generateAsync("!", 2, text -> {}, null, null));
        };
        lattice.start();
        awaitWorker(lattice);
        try {
            var firstResult = first.generateAsync("!", 12, text -> {}, null, null);
            assertEquals(12, firstResult.get(60, TimeUnit.SECONDS).size());
            long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(30);
            while ((secondResult.get() == null || earlyResult.get() == null) && System.nanoTime() < deadline)
                Thread.onSpinWait();
            assertTrue(secondResult.get() != null && earlyResult.get() != null, "the overlapping generations started");
            assertEquals(12, secondResult.get().get(60, TimeUnit.SECONDS).size());
            assertEquals(2, earlyResult.get().get(60, TimeUnit.SECONDS).size());
            assertTrue(runtime.statistics(Quantum.ExecutionKind.VERIFY).batchedQuanta() > 0);
        } finally {
            first.close();
            second.close();
            early.close();
            runtime.close();
            LatticeTesting.closeAndAwait(lattice);
        }
    }

    private static ControlPlaneLattice createLattice() {
        BitSet available = SystemInfo.getPCpuSet();
        int cpu = available.nextSetBit(0);
        assumeTrue(cpu >= 0, "requires one available physical CPU for the lattice worker");
        BitSet cpus = new BitSet();
        cpus.set(cpu);
        var workers = new BaseCloneableObject(new DefaultExecutor());
        var shard = ControlPlaneShard.createBaseShard("QwenGenerationTestShard", workers);
        return ControlPlaneLattice.getOrCreate(new LatticeConfig(
                "QwenGenerationTestLattice-" + LATTICE_ID.incrementAndGet(), cpus, Duration.ofSeconds(10), shard));
    }

    private static void awaitWorker(ControlPlaneLattice lattice) throws InterruptedException {
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(30);
        while (lattice.getActiveWorkers() < 1 && System.nanoTime() < deadline) Thread.onSpinWait();
        assertEquals(1, lattice.getActiveWorkers(), "lattice worker did not register");
    }

    private static short bf16(float value) {
        return (short) (Float.floatToIntBits(value) >>> 16);
    }

    private static int testVocabularySize() {
        return tokenizer.generationEosTokenIds().stream()
                        .mapToInt(Integer::intValue)
                        .max()
                        .orElseThrow()
                + 1;
    }

    private static final class SamplingGpu extends ExecutionFixtures.RecordingGpu {
        private final int vocabularySize;
        private volatile Runnable onRowCopy = () -> {};

        private SamplingGpu(int vocabularySize) {
            this.vocabularySize = vocabularySize;
        }

        @Override
        public void copyDeviceToHost(MemorySegment destination, long source, long byteSize) {
            if (byteSize != (long) this.vocabularySize * Short.BYTES) return;
            this.onRowCopy.run();
            for (int tokenId = 0; tokenId < this.vocabularySize; tokenId++)
                destination.set(
                        ValueLayout.JAVA_SHORT, (long) tokenId * Short.BYTES, bf16(tokenId == 1 ? 10.0f : 0.0f));
        }

        @Override
        public void copyDeviceToDevice(long destination, long source, long byteSize) {}

        @Override
        public void zeroDeviceMemory(long address, long byteSize) {}

        @Override
        public void rmsNormUnitOffsetBf16(
                long inputAddress, long weightAddress, long outputAddress, int rows, int width, float epsilon) {}

        @Override
        public void linearQ4Bf16(
                long inputAddress,
                long weightsAddress,
                long outputAddress,
                int rows,
                int inFeatures,
                int outFeatures,
                long weightsByteSize) {}

        @Override
        public void linearQ5Bf16(
                long inputAddress,
                long weightsAddress,
                long outputAddress,
                int rows,
                int inFeatures,
                int outFeatures,
                long weightsByteSize) {}

        @Override
        public void linearBf16ToFloat(
                long inputAddress,
                long weightsAddress,
                long outputAddress,
                int rows,
                int inFeatures,
                int outFeatures) {}

        @Override
        public void gdnControlFp32(
                long aProjectionAddress,
                long bProjectionAddress,
                long aLogAddress,
                long dtBiasAddress,
                long alphaOutputAddress,
                long betaOutputAddress,
                int rows,
                int heads) {}

        @Override
        public void gdnConvolutionBf16(
                long queryKeyAddress,
                long valueZAddress,
                long convolutionWeightsAddress,
                long convolutionStateAddress,
                long outputAddress,
                int rows,
                int queryKeyWidth,
                int valueWidth,
                int convolutionWidth,
                int kernelSize) {
            // no kernel work
        }

        @Override
        public void gdnRecurrenceBf16(
                long convolvedAddress,
                long alphaAddress,
                long betaAddress,
                long recurrentStateAddress,
                long outputAddress,
                int rows,
                int keyHeads,
                int valueHeads,
                int keyHeadDim,
                int valueHeadDim,
                float outputScale) {
            // no kernel work
        }

        @Override
        public void gdnGatedRmsNormBf16(
                long recurrentAddress,
                long valueZAddress,
                long normWeightAddress,
                long outputAddress,
                int rows,
                int valueHeads,
                int headDim,
                float epsilon) {}

        @Override
        public void residualAddBf16(long residualAddress, long deltaAddress, long outputAddress, int rows, int width) {}

        @Override
        public void swiGluBf16(long gateUpAddress, long outputAddress, int rows, int intermediateSize) {}

        @Override
        public void attentionQkNormRopeBf16(
                long queryKeyAddress,
                long queryNormAddress,
                long keyNormAddress,
                long outputAddress,
                int rows,
                int queryHeads,
                int keyValueHeads,
                int headDim,
                int rotaryDim,
                long startPosition,
                long positionAddress,
                float epsilon,
                double ropeTheta) {
            // no kernel work
        }

        @Override
        public void attentionKvAppendNvfp4(
                long queryKeyAddress,
                long gateValueAddress,
                long keyCacheAddress,
                long valueCacheAddress,
                int rows,
                int queryWidth,
                int keyValueWidth,
                long startPosition,
                long positionAddress) {
            // no kernel work
        }

        @Override
        public void attentionCausalNvfp4(
                long queryKeyAddress,
                long gateValueAddress,
                long keyCacheAddress,
                long valueCacheAddress,
                long outputAddress,
                int rows,
                int queryHeads,
                int keyValueHeads,
                int headDim,
                int cacheLength,
                long startPosition,
                long positionAddress,
                long scratchAddress) {}
    }
}
