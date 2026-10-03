package io.euhedral_execution.inference.core.scheduling;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.model_loader.QwenWeights;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.List;
import java.util.Map;

public final class EngineExecutionFixture {
    public static QwenWeights weights() {
        return QwenExecutionFixtures.statefulCompactWeights(8);
    }

    public static QwenSequenceState sequence(QwenGenerationSession session) {
        return session.sequenceState();
    }

    private static short bf16(float value) {
        return (short) (Float.floatToIntBits(value) >>> 16);
    }

    public static class SamplingGpu extends QwenExecutionFixtures.RecordingGpu {
        public final List<Long> embeddingAddresses = new java.util.concurrent.CopyOnWriteArrayList<>();
        private final int vocabularySize;
        private final Map<Long, int[]> uploadedTokenIds = new java.util.concurrent.ConcurrentHashMap<>();
        public final List<int[]> embeddingInputs = new java.util.concurrent.CopyOnWriteArrayList<>();
        public final List<Long> allocatedLogits = new java.util.concurrent.CopyOnWriteArrayList<>();
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

        public SamplingGpu(int vocabularySize) {
            this.vocabularySize = vocabularySize;
        }

        /// Scripts sampled token IDs in order; the last ID repeats once the script is exhausted.
        public void selectTokens(int... tokenIds) {
            this.selectedTokenIds = tokenIds.clone();
            this.sampleIndex = 0;
        }

        public void afterEmbedding(Runnable callback) {
            this.afterEmbedding = callback;
        }

        public java.util.List<Long> freed() {
            return this.frees;
        }

        public final java.util.concurrent.atomic.AtomicInteger freeFailures =
                new java.util.concurrent.atomic.AtomicInteger();

        public java.util.List<Long> allocated() {
            return this.allocations;
        }

        @Override
        public long allocate(long byteSize) {
            long address = super.allocate(byteSize);
            if (byteSize == (long) this.vocabularySize * Short.BYTES) {
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
            long rowBytes = (long) this.vocabularySize * Short.BYTES;
            long logitsBase = source - (long) (this.lastTokenCount - 1) * rowBytes;
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
            if (this.freeFailures.getAndUpdate(value -> Math.max(0, value - 1)) > 0)
                throw new IllegalStateException("injected device free failure");
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
            int[] record = this.uploadedTokenIds.get(tokenIdsAddress);
            if (record == null) throw new IllegalStateException("embedding did not receive uploaded token IDs");
            // The input record carries the token IDs, then the start position.
            int[] tokenIds = java.util.Arrays.copyOf(record, tokenCount);
            this.embeddingInputs.add(tokenIds.clone());
            this.embeddingAddresses.add(embeddingAddress);
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
