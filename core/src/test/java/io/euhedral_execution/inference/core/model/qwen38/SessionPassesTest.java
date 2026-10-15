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
import io.euhedral_execution.inference.core.sampling.GenerationConfig;
import io.euhedral_execution.inference.core.tokenizer.QwenTokenizer;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.BitSet;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.api.parallel.ExecutionMode;

/// Concurrent sessions on one execution offer their decode steps to shared passes.
@org.junit.jupiter.api.parallel.Execution(ExecutionMode.SAME_THREAD)
class SessionPassesTest {

    private static final Path TOKENIZER_DIRECTORY =
            Path.of(System.getProperty("euhedral.qwen.tokenizer-dir", "/mnt/shared/qwen38-quant/source/qwen"));
    private static final AtomicLong LATTICE_ID = new AtomicLong();
    private static QwenTokenizer tokenizer;

    @BeforeAll
    static void loadTokenizer() throws Exception {
        assumeTrue(Files.isRegularFile(TOKENIZER_DIRECTORY.resolve("tokenizer.json")));
        tokenizer = QwenTokenizer.load(TOKENIZER_DIRECTORY);
    }

    @Test
    @Timeout(value = 90, unit = TimeUnit.SECONDS)
    void twoOverlappingSessionsShareDecodePasses() throws Exception {
        int vocabularySize = testVocabularySize();
        var weights = ExecutionFixtures.statefulCompactWeights(vocabularySize);
        var plan = new ExecutionPlan(weights);
        var gpu = new SamplingGpu(vocabularySize);
        gpu.selectedTokenId = 1;
        var lattice = createLattice();
        var runtime = new Execution(lattice, plan, gpu);
        var first = new Session(tokenizer, plan, runtime, gpu, 910, GenerationConfig.greedy(1L));
        var second = new Session(tokenizer, plan, runtime, gpu, 911, GenerationConfig.greedy(2L));
        var secondResult = new java.util.concurrent.atomic.AtomicReference<
                java.util.concurrent.CompletableFuture<List<Integer>>>();
        var started = new java.util.concurrent.atomic.AtomicBoolean();
        // The second generation starts from the first one's first host-row copy, so their decodes overlap.
        gpu.onRowCopy = () -> {
            if (started.compareAndSet(false, true))
                secondResult.set(second.generateAsync("!", 12, text -> {}, null, null));
        };
        lattice.start();
        awaitWorker(lattice);
        try {
            var firstResult = first.generateAsync("!", 12, text -> {}, null, null);
            List<Integer> firstTokens = firstResult.get(60, TimeUnit.SECONDS);
            while (secondResult.get() == null) Thread.onSpinWait();
            List<Integer> secondTokens = secondResult.get().get(60, TimeUnit.SECONDS);
            assertEquals(12, firstTokens.size());
            assertEquals(12, secondTokens.size());
            assertTrue(runtime.statistics(Quantum.ExecutionKind.VERIFY).batchedQuanta() > 0);
        } finally {
            first.close();
            second.close();
            runtime.close();
            lattice.close();
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
        /// Each upload's ints by its device address; a later chunk's token ids sit inside a prompt's record.
        private final java.util.concurrent.ConcurrentSkipListMap<Long, int[]> uploadedTokenIds =
                new java.util.concurrent.ConcurrentSkipListMap<>();
        private final List<int[]> embeddingInputs = new java.util.concurrent.CopyOnWriteArrayList<>();
        private final List<Long> allocatedLogits = new java.util.concurrent.CopyOnWriteArrayList<>();
        private final List<Long> closedLogits = new java.util.concurrent.CopyOnWriteArrayList<>();
        private final List<Long> sampledLogitRows = new java.util.concurrent.CopyOnWriteArrayList<>();
        private final List<Long> sampledLogitCloses = new java.util.concurrent.CopyOnWriteArrayList<>();
        private final java.util.Set<Long> pendingLogits = java.util.concurrent.ConcurrentHashMap.newKeySet();
        private final java.util.Set<Long> liveLogits = java.util.concurrent.ConcurrentHashMap.newKeySet();
        private final List<Long> attentionStartPositions = new java.util.concurrent.CopyOnWriteArrayList<>();
        private final List<Long> convolutionStateAddresses = new java.util.concurrent.CopyOnWriteArrayList<>();
        private final List<Long> recurrentStateAddresses = new java.util.concurrent.CopyOnWriteArrayList<>();
        private final List<Long> keyCacheAddresses = new java.util.concurrent.CopyOnWriteArrayList<>();
        private volatile int lastTokenCount;
        private volatile int selectedTokenId = 1;
        private volatile int[] selectedTokenIds = new int[0];
        private int sampleIndex;
        private volatile boolean invalidLogits;
        private volatile RuntimeException embeddingFailure;
        private volatile Runnable onRowCopy = () -> {};

        private SamplingGpu(int vocabularySize) {
            this.vocabularySize = vocabularySize;
        }

        @Override
        public long allocate(long byteSize) {
            long address = super.allocate(byteSize);
            if (byteSize % ((long) this.vocabularySize * Short.BYTES) == 0) {
                this.allocatedLogits.add(address);
                this.liveLogits.add(address);
            }
            return address;
        }

        @Override
        public void copyHostToDevice(long destination, MemorySegment source, long byteSize) {
            int[] tokenIds = new int[Math.toIntExact(byteSize / Integer.BYTES)];
            for (int index = 0; index < tokenIds.length; index++) {
                tokenIds[index] = source.get(ValueLayout.JAVA_INT, (long) index * Integer.BYTES);
            }
            this.uploadedTokenIds.put(destination, tokenIds);
        }

        @Override
        public void copyDeviceToHost(MemorySegment destination, long source, long byteSize) {
            if (byteSize != (long) this.vocabularySize * Short.BYTES) return;
            this.onRowCopy.run();
            long logitsBase = source;
            this.sampledLogitRows.add(logitsBase);
            this.pendingLogits.add(logitsBase);
            int selected = this.selectedTokenIds.length == 0
                    ? this.selectedTokenId
                    : this.selectedTokenIds[Math.min(this.sampleIndex++, this.selectedTokenIds.length - 1)];
            for (int tokenId = 0; tokenId < this.vocabularySize; tokenId++) {
                destination.set(
                        ValueLayout.JAVA_SHORT,
                        (long) tokenId * Short.BYTES,
                        this.invalidLogits
                                ? bf16(Float.NEGATIVE_INFINITY)
                                : tokenId == selected ? bf16(10.0f) : bf16(0.0f));
            }
        }

        @Override
        public synchronized void free(long address) {
            if (this.pendingLogits.remove(address)) this.sampledLogitCloses.add(address);
            if (this.allocatedLogits.contains(address)) {
                assertTrue(this.liveLogits.remove(address), "logits allocation freed more than once");
                this.closedLogits.add(address);
            }
            super.free(address);
        }

        @Override
        public void copyDeviceToDevice(long destination, long source, long byteSize) {}

        @Override
        public void zeroDeviceMemory(long address, long byteSize) {}

        @Override
        public void embedQ3(
                long tokenIdsAddress,
                long embeddingAddress,
                long embeddingByteSize,
                long hiddenStateAddress,
                int tokenCount,
                int vocabularySize,
                int hiddenSize) {
            // The nearest upload at or below the address that holds every row: fixture addresses are not spaced
            // by allocation size, so another upload may sit between a prompt record's start and a later chunk.
            int[] tokenIds = null;
            for (var upload : this.uploadedTokenIds
                    .headMap(tokenIdsAddress, true)
                    .descendingMap()
                    .entrySet()) {
                long first = (tokenIdsAddress - upload.getKey()) / Integer.BYTES;
                if (first + tokenCount > upload.getValue().length) continue;
                tokenIds = java.util.Arrays.copyOfRange(upload.getValue(), (int) first, (int) first + tokenCount);
                break;
            }
            if (tokenIds == null) throw new IllegalStateException("embedding did not receive uploaded token IDs");
            this.embeddingInputs.add(tokenIds.clone());
            this.lastTokenCount = tokenCount;
            super.embedQ3(
                    tokenIdsAddress,
                    embeddingAddress,
                    embeddingByteSize,
                    hiddenStateAddress,
                    tokenCount,
                    vocabularySize,
                    hiddenSize);
            if (this.embeddingFailure != null) throw this.embeddingFailure;
        }

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
            this.convolutionStateAddresses.add(convolutionStateAddress);
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
            this.recurrentStateAddresses.add(recurrentStateAddress);
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
            this.attentionStartPositions.add(startPosition);
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
            this.keyCacheAddresses.add(keyCacheAddress);
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
