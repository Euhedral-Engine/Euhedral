package io.euhedral_execution.inference.core.gpu;

import io.euhedral_execution.inference.core.model_loader.layer_weights.WeightLayout;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.util.Objects;

/// GPU operations for Qwen stages. Production launches are ordered by a quantum-owned [GpuStream];
/// an operation called with no stream selected runs synchronously, which only tests and diagnostics use.
public abstract class ExecutionGpu implements GpuMemory {

    /// Host memory owned by one instruction until its GPU completion is proven.
    public record UploadBuffer(MemorySegment segment, Runnable release) implements AutoCloseable {
        public UploadBuffer {
            Objects.requireNonNull(segment, "segment");
            Objects.requireNonNull(release, "release");
        }

        @Override
        public void close() {
            release.run();
        }
    }

    /// Whether exact numerics (the scalar oracles) are selected.
    public boolean exactNumerics() {
        return false;
    }

    public UploadBuffer allocateUploadBuffer(long bytes) {
        Arena arena = Arena.ofShared();
        try {
            return new UploadBuffer(arena.allocate(bytes, Integer.BYTES), arena::close);
        } catch (RuntimeException | Error failure) {
            arena.close();
            throw failure;
        }
    }

    public void copyUploadToDevice(long destination, UploadBuffer upload) {
        copyHostToDevice(destination, upload.segment(), upload.segment().byteSize());
    }

    /// Host memory that receives queued device-to-host copies. Its owner keeps it, and reads it, only
    /// after every copy into it has retired; closing it earlier would leave DMA writing freed memory.
    public record ReadbackBuffer(MemorySegment segment, Runnable release) implements AutoCloseable {
        public ReadbackBuffer {
            Objects.requireNonNull(segment, "segment");
            Objects.requireNonNull(release, "release");
        }

        @Override
        public void close() {
            release.run();
        }
    }

    public ReadbackBuffer allocateReadbackBuffer(long bytes) {
        Arena arena = Arena.ofShared();
        try {
            return new ReadbackBuffer(arena.allocate(bytes, Long.BYTES), arena::close);
        } catch (RuntimeException | Error failure) {
            arena.close();
            throw failure;
        }
    }

    /// Copies `bytes` device bytes at `source` to the start of `destination`, queued on the selected
    /// stream; with no stream selected it completes before returning.
    public void copyDeviceToReadback(ReadbackBuffer destination, long source, long bytes) {
        copyDeviceToHost(destination.segment(), source, bytes);
    }

    public boolean completionProven() {
        return true;
    }

    /// Opens a device-ordering domain for one quantum at a time. Synchronous test GPUs retire work as
    /// it is submitted; the CUDA binding returns a real stream.
    public GpuStream openStream() {
        return new InlineGpuStream();
    }

    /// Permanently retains uncertain GPU ownership after failed recovery.
    public void poison(Throwable failure) {
        throw new UnsupportedOperationException("this GPU does not support asynchronous recovery");
    }

    /// Rejects new work after an unprovable GPU failure.
    public void ensureHealthy() {}

    protected static final FunctionDescriptor MALLOC =
            FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.JAVA_LONG);
    protected static final FunctionDescriptor FREE = FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS);
    protected static final FunctionDescriptor DEVICE_MEMORY_INFO =
            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS);
    protected static final FunctionDescriptor COPY = FunctionDescriptor.of(
            ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG);
    protected static final FunctionDescriptor EMBED_Q3 = FunctionDescriptor.of(
            ValueLayout.JAVA_INT,
            ValueLayout.ADDRESS,
            ValueLayout.ADDRESS,
            ValueLayout.ADDRESS,
            ValueLayout.JAVA_INT,
            ValueLayout.JAVA_INT,
            ValueLayout.JAVA_INT,
            ValueLayout.JAVA_LONG);
    protected static final FunctionDescriptor SYNCHRONIZE = FunctionDescriptor.of(ValueLayout.JAVA_INT);
    protected static final FunctionDescriptor RMS_NORM_BF16 = FunctionDescriptor.of(
            ValueLayout.JAVA_INT,
            ValueLayout.ADDRESS,
            ValueLayout.ADDRESS,
            ValueLayout.ADDRESS,
            ValueLayout.JAVA_INT,
            ValueLayout.JAVA_INT,
            ValueLayout.JAVA_FLOAT);
    protected static final FunctionDescriptor RMS_NORM_UNIT_OFFSET_BF16 = RMS_NORM_BF16;
    protected static final FunctionDescriptor LINEAR_Q3_BF16 = FunctionDescriptor.of(
            ValueLayout.JAVA_INT,
            ValueLayout.ADDRESS,
            ValueLayout.ADDRESS,
            ValueLayout.ADDRESS,
            ValueLayout.JAVA_INT,
            ValueLayout.JAVA_INT,
            ValueLayout.JAVA_INT,
            ValueLayout.JAVA_LONG);
    protected static final FunctionDescriptor LINEAR_QUANTIZED_BF16 = FunctionDescriptor.of(
            ValueLayout.JAVA_INT,
            ValueLayout.ADDRESS,
            ValueLayout.ADDRESS,
            ValueLayout.ADDRESS,
            ValueLayout.JAVA_INT,
            ValueLayout.JAVA_INT,
            ValueLayout.JAVA_INT,
            ValueLayout.JAVA_LONG,
            ValueLayout.JAVA_INT);
    protected static final FunctionDescriptor LINEAR_BF16_TO_FLOAT = FunctionDescriptor.of(
            ValueLayout.JAVA_INT,
            ValueLayout.ADDRESS,
            ValueLayout.ADDRESS,
            ValueLayout.ADDRESS,
            ValueLayout.JAVA_INT,
            ValueLayout.JAVA_INT,
            ValueLayout.JAVA_INT);
    protected static final FunctionDescriptor GDN_CONTROL_FP32 = FunctionDescriptor.of(
            ValueLayout.JAVA_INT,
            ValueLayout.ADDRESS,
            ValueLayout.ADDRESS,
            ValueLayout.ADDRESS,
            ValueLayout.ADDRESS,
            ValueLayout.ADDRESS,
            ValueLayout.ADDRESS,
            ValueLayout.JAVA_INT,
            ValueLayout.JAVA_INT);
    protected static final FunctionDescriptor GDN_CONVOLUTION_BF16 = FunctionDescriptor.of(
            ValueLayout.JAVA_INT,
            ValueLayout.ADDRESS,
            ValueLayout.ADDRESS,
            ValueLayout.ADDRESS,
            ValueLayout.ADDRESS,
            ValueLayout.ADDRESS,
            ValueLayout.JAVA_INT,
            ValueLayout.JAVA_INT,
            ValueLayout.JAVA_INT,
            ValueLayout.JAVA_INT,
            ValueLayout.JAVA_INT);
    protected static final FunctionDescriptor GDN_RECURRENCE_BF16 = FunctionDescriptor.of(
            ValueLayout.JAVA_INT,
            ValueLayout.ADDRESS,
            ValueLayout.ADDRESS,
            ValueLayout.ADDRESS,
            ValueLayout.ADDRESS,
            ValueLayout.ADDRESS,
            ValueLayout.JAVA_INT,
            ValueLayout.JAVA_INT,
            ValueLayout.JAVA_INT,
            ValueLayout.JAVA_INT,
            ValueLayout.JAVA_INT,
            ValueLayout.JAVA_FLOAT);
    protected static final FunctionDescriptor GDN_GATED_RMS_NORM_BF16 = FunctionDescriptor.of(
            ValueLayout.JAVA_INT,
            ValueLayout.ADDRESS,
            ValueLayout.ADDRESS,
            ValueLayout.ADDRESS,
            ValueLayout.ADDRESS,
            ValueLayout.JAVA_INT,
            ValueLayout.JAVA_INT,
            ValueLayout.JAVA_INT,
            ValueLayout.JAVA_FLOAT);
    protected static final FunctionDescriptor RESIDUAL_ADD_BF16 = FunctionDescriptor.of(
            ValueLayout.JAVA_INT,
            ValueLayout.ADDRESS,
            ValueLayout.ADDRESS,
            ValueLayout.ADDRESS,
            ValueLayout.JAVA_INT,
            ValueLayout.JAVA_INT);
    protected static final FunctionDescriptor ARGMAX_BF16 =
            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.ADDRESS);
    protected static final FunctionDescriptor SWIGLU_BF16 = FunctionDescriptor.of(
            ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT);
    protected static final FunctionDescriptor ZERO_DEVICE_MEMORY =
            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG);
    protected static final FunctionDescriptor ATTENTION_QK_NORM_ROPE_BF16 = FunctionDescriptor.of(
            ValueLayout.JAVA_INT,
            ValueLayout.ADDRESS,
            ValueLayout.ADDRESS,
            ValueLayout.ADDRESS,
            ValueLayout.ADDRESS,
            ValueLayout.JAVA_INT,
            ValueLayout.JAVA_INT,
            ValueLayout.JAVA_INT,
            ValueLayout.JAVA_INT,
            ValueLayout.JAVA_INT,
            ValueLayout.JAVA_LONG,
            ValueLayout.ADDRESS,
            ValueLayout.JAVA_FLOAT,
            ValueLayout.JAVA_DOUBLE);
    protected static final FunctionDescriptor ATTENTION_KV_APPEND_NVFP4 = FunctionDescriptor.of(
            ValueLayout.JAVA_INT,
            ValueLayout.ADDRESS,
            ValueLayout.ADDRESS,
            ValueLayout.ADDRESS,
            ValueLayout.ADDRESS,
            ValueLayout.JAVA_INT,
            ValueLayout.JAVA_INT,
            ValueLayout.JAVA_INT,
            ValueLayout.JAVA_LONG,
            ValueLayout.ADDRESS);
    protected static final FunctionDescriptor ATTENTION_CAUSAL_NVFP4 = FunctionDescriptor.of(
            ValueLayout.JAVA_INT,
            ValueLayout.ADDRESS,
            ValueLayout.ADDRESS,
            ValueLayout.ADDRESS,
            ValueLayout.ADDRESS,
            ValueLayout.ADDRESS,
            ValueLayout.JAVA_INT,
            ValueLayout.JAVA_INT,
            ValueLayout.JAVA_INT,
            ValueLayout.JAVA_INT,
            ValueLayout.JAVA_INT,
            ValueLayout.JAVA_LONG,
            ValueLayout.ADDRESS,
            ValueLayout.ADDRESS);
    protected static final int CUDA_FORMAT_MISMATCH = -3;

    protected static MethodHandle bind(
            Linker linker, SymbolLookup symbols, String name, FunctionDescriptor descriptor) {
        MemorySegment symbol =
                symbols.find(name).orElseThrow(() -> new GpuMemoryException("native symbol not found: " + name));
        return linker.downcallHandle(symbol, descriptor);
    }

    public abstract void embedQ3(
            long tokenIdsAddress,
            long embeddingAddress,
            long embeddingByteSize,
            long hiddenStateAddress,
            int tokenCount,
            int vocabularySize,
            int hiddenSize);

    public abstract void synchronize();

    /// [#embedQ3] for an embedding table in either Q3 layout. Backends without P2E2 support accept
    /// only [WeightLayout#ROW_SPLIT_K128_V1].
    public void embedQ3(
            long tokenIdsAddress,
            long embeddingAddress,
            long embeddingByteSize,
            long hiddenStateAddress,
            int tokenCount,
            int vocabularySize,
            int hiddenSize,
            WeightLayout layout) {
        requireRowSplit(layout, "Q3 embedding");
        embedQ3(
                tokenIdsAddress,
                embeddingAddress,
                embeddingByteSize,
                hiddenStateAddress,
                tokenCount,
                vocabularySize,
                hiddenSize);
    }

    protected static void requireRowSplit(WeightLayout layout, String operation) {
        if (layout != WeightLayout.ROW_SPLIT_K128_V1)
            throw new UnsupportedOperationException(operation + " does not support " + layout + " on this GPU");
    }

    public void rmsNormBf16(
            long inputAddress, long weightAddress, long outputAddress, int rows, int width, float epsilon) {
        throw new UnsupportedOperationException("BF16 RMS norm is not implemented by this GPU");
    }

    /// Applies Qwen's unit-offset RMSNorm convention: normalized * (1 + weight).
    public void rmsNormUnitOffsetBf16(
            long inputAddress, long weightAddress, long outputAddress, int rows, int width, float epsilon) {
        throw new UnsupportedOperationException("unit-offset BF16 RMS norm is not implemented by this GPU");
    }

    public void linearQ3Bf16(
            long inputAddress,
            long weightsAddress,
            long outputAddress,
            int rows,
            int inFeatures,
            int outFeatures,
            long weightsByteSize) {
        throw new UnsupportedOperationException("Q3 linear is not implemented by this GPU");
    }

    /// [#linearQ3Bf16] for weights in either Q3 layout; the results are the same values.
    public void linearQ3Bf16(
            long inputAddress,
            long weightsAddress,
            long outputAddress,
            int rows,
            int inFeatures,
            int outFeatures,
            long weightsByteSize,
            WeightLayout layout) {
        requireRowSplit(layout, "Q3 linear");
        linearQ3Bf16(inputAddress, weightsAddress, outputAddress, rows, inFeatures, outFeatures, weightsByteSize);
    }

    /// Queues a copy of `byteSize` bytes from pinned host weights at `source` to device memory on the
    /// selected stream.
    public void copyHostWeightsToDevice(long destination, long source, long byteSize) {
        throw new UnsupportedOperationException("host-backed weights are not implemented by this GPU");
    }

    /// NVFP4 linear (Nvfp4Layout): `outFeatures` weight rows of `inFeatures` values on BF16 rows.
    public void linearNvfp4Bf16(
            long input, long weights, long output, int rows, int inFeatures, int outFeatures, long weightBytes) {
        throw new UnsupportedOperationException("NVFP4 linear is not implemented by this GPU");
    }

    /// NVFP4 gate/up projection with SwiGLU: `outputs` weight rows, gate rows first, giving outputs / 2
    /// values per row.
    public void nvfp4GateUpSwiGluBf16(
            long input, long weights, long output, int rows, int width, int outputs, long weightBytes) {
        throw new UnsupportedOperationException("NVFP4 gate/up SwiGLU region is not implemented by this GPU");
    }

    /// Executes one row-split Q4 projection over BF16 activations.
    public void linearQ4Bf16(
            long inputAddress,
            long weightsAddress,
            long outputAddress,
            int rows,
            int inFeatures,
            int outFeatures,
            long weightsByteSize) {
        throw new UnsupportedOperationException("Q4 linear is not implemented by this GPU");
    }

    /// Executes one row-split Q5 projection over BF16 activations.
    public void linearQ5Bf16(
            long inputAddress,
            long weightsAddress,
            long outputAddress,
            int rows,
            int inFeatures,
            int outFeatures,
            long weightsByteSize) {
        throw new UnsupportedOperationException("Q5 linear is not implemented by this GPU");
    }

    /// Multiplies BF16 activations and weights, preserving FP32 projection results.
    public void linearBf16ToFloat(
            long inputAddress, long weightsAddress, long outputAddress, int rows, int inFeatures, int outFeatures) {
        throw new UnsupportedOperationException("BF16-to-FP32 linear is not implemented by this GPU");
    }

    public void gdnControlFp32(
            long aProjectionAddress,
            long bProjectionAddress,
            long aLogAddress,
            long dtBiasAddress,
            long alphaOutputAddress,
            long betaOutputAddress,
            int rows,
            int heads) {
        throw new UnsupportedOperationException("GDN control operation is not implemented by this GPU");
    }

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
        throw new UnsupportedOperationException("GDN convolution is not implemented by this GPU");
    }

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
        throw new UnsupportedOperationException("GDN recurrence is not implemented by this GPU");
    }

    public void gdnGatedRmsNormBf16(
            long recurrentAddress,
            long valueZAddress,
            long normWeightAddress,
            long outputAddress,
            int rows,
            int valueHeads,
            int headDim,
            float epsilon) {
        throw new UnsupportedOperationException("GDN gated RMSNorm is not implemented by this GPU");
    }

    public void residualAddBf16(long residualAddress, long deltaAddress, long outputAddress, int rows, int width) {
        throw new UnsupportedOperationException("BF16 residual add is not implemented by this GPU");
    }

    /// Copies `rows` contiguous rows of `sourcePitch` bytes to rows `destinationPitch` bytes apart, on the
    /// selected stream.
    public void copyRowsDeviceToDevice(
            long destination, long destinationPitch, long source, long sourcePitch, int rows) {
        throw new UnsupportedOperationException("pitched device copy is not implemented by this GPU");
    }

    /// Row-exact execution for launches from the calling thread: a multi-row operation computes every row
    /// bit for bit as a one-row operation at that row's position would (speculative verification).
    public void selectRowExact(boolean enabled) {}

    /// Whether Q3, Q4 and Q5 linears honour [#selectRowExact] themselves, so a row-exact quantum can
    /// launch them once over all its rows instead of once per row.
    public boolean rowExactQuantizedLinears() {
        return false;
    }

    /// Device bytes of the shared scratch that prefill and verification routes keep allocated between
    /// launches (expanded P2E2 weights, quantized activations); zero when this GPU keeps none.
    public long retainedScratchBytes() {
        return 0;
    }

    /// Greedy selection over `count` BF16 logits at `logitsAddress`: queues a kernel that writes one
    /// 64-bit key to `resultAddress`, whose low word is `0xFFFFFFFF` minus the host argmax's token ID
    /// (0 when no logit is selectable). False, with nothing queued, when this GPU cannot select on the
    /// device.
    public boolean argmaxBf16(long logitsAddress, int count, long resultAddress) {
        return false;
    }

    public void q3GateUpSwiGluBf16(
            long input, long weights, long output, int rows, int width, int outputs, long weightBytes) {
        throw new UnsupportedOperationException("Q3 gate/up SwiGLU region is not implemented");
    }

    /// The paired gate/up projection with SwiGLU in one region: `outputs` weight rows, gate rows first, giving
    /// outputs / 2 values per row. Backends without P2E2 support accept only [WeightLayout#ROW_SPLIT_K128_V1].
    public void q3GateUpSwiGluBf16(
            long input,
            long weights,
            long output,
            int rows,
            int width,
            int outputs,
            long weightBytes,
            WeightLayout layout) {
        requireRowSplit(layout, "Q3 gate/up SwiGLU region");
        q3GateUpSwiGluBf16(input, weights, output, rows, width, outputs, weightBytes);
    }

    /// Writes the rounded residual and normalizes that BF16 representation in one region.
    public void residualRmsNormBf16(
            long residual, long delta, long weight, long hidden, long normalized, int rows, int width, float epsilon) {
        throw new UnsupportedOperationException("residual RMSNorm region is not implemented");
    }

    /// Produces control values without exposing long-lived A/B projection buffers.
    public void gdnProjectControlFp32(
            long input,
            long aWeight,
            long bWeight,
            long aLog,
            long dtBias,
            long g,
            long beta,
            int rows,
            int width,
            int heads) {
        throw new UnsupportedOperationException("GDN projection/control region is not implemented");
    }

    public void swiGluBf16(long gateUpAddress, long outputAddress, int rows, int intermediateSize) {
        throw new UnsupportedOperationException("BF16 SwiGLU is not implemented by this GPU");
    }

    // DFlash2 drafter operators (native/src/dflash/kernels.cu). `position` addresses the quantum's uint64 start
    // position in device memory.

    public void dflashLinearBf16(long input, long weights, long output, int rows, int inFeatures, int outFeatures) {
        throw new UnsupportedOperationException("DFlash2 BF16 linear is not implemented by this GPU");
    }

    public void dflashRmsNormBf16(long input, long weight, long output, int rows, int width, float epsilon) {
        throw new UnsupportedOperationException("DFlash2 RMSNorm is not implemented by this GPU");
    }

    public void dflashConvBf16(
            long input, long dynamic, long base, long output, int rows, int width, int group, int taps, int part) {
        throw new UnsupportedOperationException("DFlash2 dynamic convolution is not implemented by this GPU");
    }

    public void dflashContextKvBf16(
            long kv,
            long keyNorm,
            long ringKeys,
            long ringValues,
            int rows,
            long position,
            int window,
            int keyValueHeads,
            int headDim,
            float epsilon,
            float theta) {
        throw new UnsupportedOperationException("DFlash2 context keys are not implemented by this GPU");
    }

    public void dflashBlockQkBf16(
            long query,
            long kv,
            long queryNorm,
            long keyNorm,
            long queryOut,
            long keyOut,
            int rows,
            long position,
            int heads,
            int keyValueHeads,
            int headDim,
            float epsilon,
            float theta) {
        throw new UnsupportedOperationException("DFlash2 block queries and keys are not implemented by this GPU");
    }

    public void dflashAttentionBf16(
            long query,
            long blockKeys,
            long kv,
            long ringKeys,
            long ringValues,
            long output,
            int rows,
            long position,
            int window,
            int heads,
            int keyValueHeads,
            int headDim) {
        throw new UnsupportedOperationException("DFlash2 attention is not implemented by this GPU");
    }

    public void dflashSwiGluBf16(long gateUp, long output, int rows, int intermediate) {
        throw new UnsupportedOperationException("DFlash2 SwiGLU is not implemented by this GPU");
    }

    public void dflashTopKBf16(long logits, int rows, int vocabulary, long values, long indices) {
        throw new UnsupportedOperationException("DFlash2 top-k is not implemented by this GPU");
    }

    public void dflashSelectBf16(
            long hidden,
            long values,
            long indices,
            long predecessor,
            long successor,
            long anchor,
            int positions,
            int rank,
            long tokens,
            long scores) {
        throw new UnsupportedOperationException("DFlash2 selector is not implemented by this GPU");
    }

    public void zeroDeviceMemory(long address, long byteSize) {
        throw new UnsupportedOperationException("device memory zeroing is not implemented by this GPU");
    }

    /// Applies per-head Q/K RMSNorm and partial RoPE to one compact projection result. Position-dependent
    /// kernels read the start position from `positionAddress` (one 64-bit value in device memory, written
    /// before the launch); `startPosition` is the host's copy, which validates and sizes the launch.
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
        throw new UnsupportedOperationException("Qwen attention Q/K normalization and RoPE are not implemented");
    }

    /// Appends the current compact K/V projections to one layer's sequence-owned cache.
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
        throw new UnsupportedOperationException("Qwen attention KV append is not implemented");
    }

    /// Evaluates causal GQA against the already-appended cache and applies the Q gate.
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
            long scratchAddress) {
        throw new UnsupportedOperationException("Qwen causal attention is not implemented");
    }
}
