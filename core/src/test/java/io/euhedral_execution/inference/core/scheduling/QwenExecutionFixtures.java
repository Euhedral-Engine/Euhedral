package io.euhedral_execution.inference.core.scheduling;

import io.euhedral_execution.core.generics.LatticeSource;
import io.euhedral_execution.core.generics.LatticeTerminal;
import io.euhedral_execution.core.impl.DefaultExecutor;
import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.gpu.GpuStream;
import io.euhedral_execution.inference.core.model_loader.QwenWeights;
import io.euhedral_execution.inference.core.model_loader.artifact.CompactTensorLayout;
import io.euhedral_execution.inference.core.model_loader.config.QwenConfig;
import io.euhedral_execution.inference.core.model_loader.config.QwenLayerType;
import io.euhedral_execution.inference.core.model_loader.layer_weights.QwenCompactAttentionWeights;
import io.euhedral_execution.inference.core.model_loader.layer_weights.QwenCompactDenseFfnWeights;
import io.euhedral_execution.inference.core.model_loader.layer_weights.QwenCompactGatedDeltaNetWeights;
import io.euhedral_execution.inference.core.model_loader.layer_weights.QwenLayerWeights;
import io.euhedral_execution.inference.core.model_loader.layer_weights.QwenMixerWeights;
import io.euhedral_execution.inference.core.model_loader.layer_weights.TensorDataType;
import io.euhedral_execution.inference.core.model_loader.layer_weights.TensorHandle;
import io.euhedral_execution.inference.core.model_loader.layer_weights.WeightFormat;
import io.euhedral_execution.inference.core.model_loader.layer_weights.WeightLayout;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

final class QwenExecutionFixtures {
    static final int WIDTH = 64;
    static final int VOCABULARY = 8;
    static final long MODEL_ADDRESS = 77;
    private static final AtomicLong NEXT_COMPACT_ADDRESS = new AtomicLong(20_000);

    static QwenWeights weights() {
        QwenConfig config = new QwenConfig(
                VOCABULARY,
                WIDTH,
                0,
                1,
                1,
                WIDTH,
                WIDTH,
                1,
                1,
                1,
                1,
                1,
                1.0e-6,
                1_000_000.0,
                1.0,
                128,
                "silu",
                new QwenLayerType[0],
                0,
                0,
                0,
                0,
                false,
                false,
                0);
        TensorHandle embedding = q3("embedding", VOCABULARY, MODEL_ADDRESS);
        return new QwenWeights(config, embedding, new QwenLayerWeights[0], norm(), embedding, null);
    }

    static TensorHandle norm() {
        return new TensorHandle(
                "norm",
                new long[] {WIDTH},
                TensorDataType.BF16,
                WeightFormat.BF16,
                WeightLayout.CONTIGUOUS_LE_V1,
                MODEL_ADDRESS + 1,
                WIDTH * Short.BYTES);
    }

    static TensorHandle q3(String name, int outputs, long address) {
        long[] shape = {outputs, WIDTH};
        long bytes = CompactTensorLayout.expectedByteSize(
                shape, TensorDataType.BF16, WeightFormat.Q3_G64_FP16, WeightLayout.ROW_SPLIT_K128_V1);
        return new TensorHandle(
                name,
                shape,
                TensorDataType.BF16,
                WeightFormat.Q3_G64_FP16,
                WeightLayout.ROW_SPLIT_K128_V1,
                address,
                bytes);
    }

    static QwenWeights statefulCompactWeights() {
        return statefulCompactWeights(VOCABULARY);
    }

    static QwenWeights statefulCompactWeights(int vocabularySize) {
        return statefulCompactWeights(vocabularySize, 128, 128);
    }

    static QwenWeights statefulCompactWeights(int vocabularySize, int hidden, int intermediate) {
        return statefulCompactWeights(vocabularySize, hidden, intermediate, 256, 1.0, 2, 1.0e-6);
    }

    static QwenWeights statefulCompactWeightsWithHeadDim(int attentionHeadDim) {
        return statefulCompactWeights(VOCABULARY, 128, 128, attentionHeadDim, 1.0, 2, 1.0e-6);
    }

    static QwenWeights statefulCompactWeightsWithRotaryFactor(double factor) {
        return statefulCompactWeights(VOCABULARY, 128, 128, 256, factor, 2, 1.0e-6);
    }

    static QwenWeights statefulCompactWeightsWithKernelWidth(int kernelWidth) {
        return statefulCompactWeights(VOCABULARY, 128, 128, 256, 1.0, kernelWidth, 1.0e-6);
    }

    static QwenWeights statefulCompactWeightsWithEpsilon(double epsilon) {
        return statefulCompactWeights(VOCABULARY, 128, 128, 256, 1.0, 2, epsilon);
    }

    private static QwenWeights statefulCompactWeights(
            int vocabularySize,
            int hidden,
            int intermediate,
            int attentionHeadDim,
            double rotaryFactor,
            int kernelWidth,
            double epsilon) {
        QwenLayerType[] types = {QwenLayerType.GATED_DELTA_NET, QwenLayerType.FULL_ATTENTION};
        QwenConfig config = new QwenConfig(
                vocabularySize,
                hidden,
                types.length,
                1,
                1,
                attentionHeadDim,
                intermediate,
                1,
                1,
                128,
                128,
                kernelWidth,
                epsilon,
                1_000_000.0,
                rotaryFactor,
                128,
                "silu",
                types,
                0,
                0,
                0,
                0,
                false,
                true,
                0);
        QwenLayerWeights[] layers = new QwenLayerWeights[types.length];
        layers[0] = layer(0, hidden, intermediate, gdnWeights(hidden, kernelWidth));
        layers[1] = layer(1, hidden, intermediate, attentionWeights(hidden, attentionHeadDim));
        return new QwenWeights(
                config,
                quantized("text/token_embedding", vocabularySize, hidden, WeightFormat.Q3_G64_FP16),
                layers,
                direct("text/final_norm", WeightFormat.BF16, hidden),
                quantized("text/output_head", vocabularySize, hidden, WeightFormat.Q3_G64_FP16),
                null);
    }

    private static QwenLayerWeights layer(int index, int hidden, int intermediate, QwenMixerWeights mixer) {
        return new QwenLayerWeights(
                index,
                direct("layer-" + index + "/input_norm", WeightFormat.BF16, hidden),
                direct("layer-" + index + "/post_norm", WeightFormat.BF16, hidden),
                mixer,
                new QwenCompactDenseFfnWeights(
                        quantized("layer-" + index + "/gate_up", 2 * intermediate, hidden, WeightFormat.Q3_G64_FP16),
                        quantized("layer-" + index + "/down", hidden, intermediate, WeightFormat.Q3_G64_FP16)));
    }

    private static QwenCompactGatedDeltaNetWeights gdnWeights(int hidden, int kernelWidth) {
        return new QwenCompactGatedDeltaNetWeights(
                direct("gdn/a_log", WeightFormat.FP32, 1),
                direct("gdn/dt_bias", WeightFormat.FP32, 1),
                direct("gdn/convolution", WeightFormat.BF16, kernelWidth, 384),
                direct("gdn/a_projection", WeightFormat.BF16, 1, hidden),
                direct("gdn/b_projection", WeightFormat.BF16, 1, hidden),
                quantized("gdn/query_key", 256, hidden, WeightFormat.Q4_G64_FP16),
                quantized("gdn/value_z", 256, hidden, WeightFormat.Q5_G64_FP16),
                direct("gdn/norm", WeightFormat.BF16, 128),
                quantized("gdn/output", hidden, 128, WeightFormat.Q3_G64_FP16));
    }

    private static QwenCompactAttentionWeights attentionWeights(int hidden, int headDim) {
        return new QwenCompactAttentionWeights(
                quantized("attention/query_key", 2 * headDim, hidden, WeightFormat.Q4_G64_FP16),
                quantized("attention/gate_value", 2 * headDim, hidden, WeightFormat.Q5_G64_FP16),
                direct("attention/query_norm", WeightFormat.BF16, headDim),
                direct("attention/key_norm", WeightFormat.BF16, headDim),
                quantized("attention/output", hidden, headDim, WeightFormat.Q3_G64_FP16));
    }

    private static TensorHandle direct(String name, WeightFormat format, int... dimensions) {
        long elements = 1;
        long[] shape = new long[dimensions.length];
        for (int index = 0; index < dimensions.length; index++) {
            shape[index] = dimensions[index];
            elements = Math.multiplyExact(elements, dimensions[index]);
        }
        int elementBytes = format == WeightFormat.FP32 ? Float.BYTES : Short.BYTES;
        return new TensorHandle(
                name,
                shape,
                TensorDataType.BF16,
                format,
                WeightLayout.CONTIGUOUS_LE_V1,
                NEXT_COMPACT_ADDRESS.getAndIncrement(),
                Math.multiplyExact(elements, elementBytes));
    }

    private static TensorHandle quantized(String name, int rows, int columns, WeightFormat format) {
        long[] shape = {rows, columns};
        return new TensorHandle(
                name,
                shape,
                TensorDataType.BF16,
                format,
                WeightLayout.ROW_SPLIT_K128_V1,
                NEXT_COMPACT_ADDRESS.getAndIncrement(),
                CompactTensorLayout.expectedByteSize(
                        shape, TensorDataType.BF16, format, WeightLayout.ROW_SPLIT_K128_V1));
    }

    static class RecordingGpu extends ExecutionGpu {
        final AtomicLong nextAddress = new AtomicLong(1000);
        final List<Long> allocations = new ArrayList<>();
        final List<Long> frees = new ArrayList<>();
        final List<String> operations = new ArrayList<>();
        Runnable afterEmbedding = () -> {};
        RuntimeException linearFailure;
        int synchronizations;
        int freeFailures;
        int failAllocationAt;
        int allocationAttempts;
        long failAddressOnce;

        @Override
        public long allocate(long byteSize) {
            if (++allocationAttempts == failAllocationAt) {
                throw new IllegalStateException("injected workspace allocation failure");
            }
            long address = nextAddress.getAndIncrement();
            allocations.add(address);
            return address;
        }

        @Override
        public void copyHostToDevice(long destination, MemorySegment source, long byteSize) {}

        @Override
        public void copyDeviceToHost(MemorySegment destination, long source, long byteSize) {}

        @Override
        public synchronized void free(long address) {
            if (address == failAddressOnce) {
                failAddressOnce = 0;
                throw new IllegalStateException("injected workspace free failure");
            }
            if (freeFailures-- > 0) throw new IllegalStateException("injected free failure");
            frees.add(address);
        }

        @Override
        public void embedQ3(
                long tokenIdsAddress,
                long embeddingAddress,
                long embeddingByteSize,
                long hiddenStateAddress,
                int tokenCount,
                int vocabularySize,
                int hiddenSize) {
            operations.add("embed");
            afterEmbedding.run();
        }

        @Override
        public void rmsNormBf16(
                long inputAddress, long weightAddress, long outputAddress, int rows, int width, float epsilon) {
            operations.add("norm");
        }

        @Override
        public void linearQ3Bf16(
                long inputAddress,
                long weightsAddress,
                long outputAddress,
                int rows,
                int inFeatures,
                int outFeatures,
                long weightsByteSize) {
            operations.add("linear:" + weightsAddress);
            if (linearFailure != null) throw linearFailure;
        }

        // Production prefill regions are no-ops for scheduling fakes; subclasses record as needed.
        @Override
        public void residualRmsNormBf16(
                long residual,
                long delta,
                long weight,
                long hidden,
                long normalized,
                int rows,
                int width,
                float epsilon) {}

        @Override
        public void gdnProjectControlFp32(
                long input, long a, long b, long log, long bias, long g, long beta, int rows, int width, int heads) {}

        @Override
        public void q3GateUpSwiGluBf16(
                long input, long weights, long output, int rows, int width, int outputs, long weightBytes) {}

        @Override
        public void q3FfnStreamedBf16(
                long input,
                long gateWeights,
                long downWeights,
                long output,
                long slots,
                long accumulators,
                int rows,
                int hidden,
                int intermediate,
                long gateBytes,
                long downBytes) {}

        @Override
        public void attentionProducersNvfp4(
                long input,
                long q4,
                long q5,
                long queryNorm,
                long keyNorm,
                long queryKey,
                long gate,
                long keys,
                long values,
                int rows,
                int hidden,
                int queryHeads,
                int keyHeads,
                int headDim,
                int rotaryDim,
                long start,
                float epsilon,
                double theta,
                long q4Bytes,
                long q5Bytes) {}

        @Override
        public void synchronize() {
            synchronizations++;
        }
    }

    /// Euhedral's execution terminal attached with unbounded demand: each published frame runs on the
    /// publishing thread, so a quantum on a synchronous fixture GPU finishes before `submit` returns.
    static LatticeTerminal inlineLattice() {
        return source -> {
            new DefaultExecutor().input(source);
            source.request(Long.MAX_VALUE);
        };
    }

    /// Euhedral's execution terminal with no standing demand: `drive` pulls and runs every ready frame,
    /// including retirement frames that a driver callback only enqueued, and leaves no demand behind.
    static final class ManualLattice implements LatticeTerminal {
        /// Each reusable graph attaches its own source once, when it is built.
        final List<LatticeSource> sources = new java.util.concurrent.CopyOnWriteArrayList<>();
        /// The most recently attached source.
        LatticeSource source;

        @Override
        public void addUpstream(LatticeSource attached) {
            new DefaultExecutor().input(attached);
            this.sources.add(attached);
            this.source = attached;
        }

        void drive() {
            for (LatticeSource attached : this.sources)
                attached.pull(PullingLattice::run, frame -> false, Long.MAX_VALUE);
        }

        /// Pulls from every attached source, as Euhedral's workers cycle through their upstreams.
        long pull(
                java.util.function.Consumer<io.euhedral_execution.core.frames.AbstractFrame> consumer,
                java.util.function.Function<io.euhedral_execution.core.frames.AbstractFrame, Boolean> stop,
                long demand) {
            long pulled = 0;
            for (LatticeSource attached : this.sources) {
                if (pulled >= demand) break;
                pulled += attached.pull(consumer, stop, demand - pulled);
            }
            return pulled;
        }
    }

    /// Every device allocation was freed exactly once, in any order.
    static void assertEachAllocationFreedOnce(RecordingGpu gpu) {
        org.junit.jupiter.api.Assertions.assertEquals(
                gpu.allocations.stream().sorted().toList(),
                gpu.frees.stream().sorted().toList());
    }

    static EuhedralInferenceRuntime runtime(QwenExecutionPlan plan, ExecutionGpu gpu) {
        return new EuhedralInferenceRuntime(inlineLattice(), plan, gpu);
    }

    /// A stream whose device work has finished when `submit` returns, but whose retirement boundaries
    /// stay unannounced until the test releases them.
    static final class HoldingStream implements GpuStream {
        final List<Long> tickets = new ArrayList<>();
        final List<RetirementListener> listeners = new ArrayList<>();
        final List<Throwable> failures = new ArrayList<>();
        int recoveries;
        private long nextTicket;

        @Override
        public synchronized void submit(Runnable launches, boolean overlapPredecessor) {
            launches.run();
        }

        @Override
        public synchronized long notifyRetired(RetirementListener listener) {
            long ticket = ++this.nextTicket;
            this.tickets.add(ticket);
            this.listeners.add(listener);
            return ticket;
        }

        synchronized int held() {
            return this.listeners.size();
        }

        /// Announces the oldest held boundary from a driver thread's point of view.
        void release(Throwable deviceFailure) {
            long ticket;
            RetirementListener listener;
            synchronized (this) {
                ticket = this.tickets.removeFirst();
                listener = this.listeners.removeFirst();
                if (deviceFailure != null) this.failed.put(ticket, deviceFailure);
            }
            listener.retired(ticket, true);
        }

        private final java.util.Map<Long, Throwable> failed = new java.util.HashMap<>();

        @Override
        public synchronized Throwable confirmRetired(long ticket) {
            Throwable failure = this.failed.remove(ticket);
            if (failure != null) this.failures.add(failure);
            return failure;
        }

        @Override
        public void synchronize() {}

        @Override
        public synchronized void recover(Throwable failure) {
            this.recoveries++;
        }

        @Override
        public void close() {}
    }

    private QwenExecutionFixtures() {}
}
