package io.euhedral_execution.inference.core.gpu;

import java.lang.foreign.MemorySegment;
import java.nio.file.Path;

/// Test fixture: the CUDA binding with every stage launched on no selected stream.
///
/// Each native operation then runs on the default stream and synchronizes before it returns, which is
/// the retired synchronous execution mode. Stream-ordered execution is checked against it.
public final class SynchronousReferenceGpu extends ExecutionGpu implements AutoCloseable {

    private final CudaGpuMemory delegate;

    public SynchronousReferenceGpu(Path library) {
        this.delegate = new CudaGpuMemory(library);
    }

    public CudaGpuMemory delegate() {
        return this.delegate;
    }

    /// Stages submit on the calling thread with no stream selected, so every launch synchronizes.
    @Override
    public GpuStream openStream() {
        return new InlineGpuStream();
    }

    @Override
    public boolean completionProven() {
        return this.delegate.completionProven();
    }

    @Override
    public void ensureHealthy() {
        this.delegate.ensureHealthy();
    }

    @Override
    public void poison(Throwable failure) {
        this.delegate.poison(failure);
    }

    @Override
    public long allocate(long byteSize) {
        return this.delegate.allocate(byteSize);
    }

    @Override
    public void copyHostToDevice(long destination, MemorySegment source, long byteSize) {
        this.delegate.copyHostToDevice(destination, source, byteSize);
    }

    @Override
    public void copyDeviceToHost(MemorySegment destination, long source, long byteSize) {
        this.delegate.copyDeviceToHost(destination, source, byteSize);
    }

    @Override
    public void copyDeviceToDevice(long destination, long source, long byteSize) {
        this.delegate.copyDeviceToDevice(destination, source, byteSize);
    }

    @Override
    public void free(long address) {
        this.delegate.free(address);
    }

    public CudaGpuMemory.DeviceMemoryInfo deviceMemoryInfo() {
        return this.delegate.deviceMemoryInfo();
    }

    public long allocatedBytes() {
        return this.delegate.allocatedBytes();
    }

    @Override
    public void close() {
        this.delegate.close();
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
        this.delegate.embedQ3(
                tokenIdsAddress,
                embeddingAddress,
                embeddingByteSize,
                hiddenStateAddress,
                tokenCount,
                vocabularySize,
                hiddenSize);
    }

    @Override
    public void synchronize() {
        this.delegate.synchronize();
    }

    @Override
    public void rmsNormBf16(
            long inputAddress, long weightAddress, long outputAddress, int rows, int width, float epsilon) {
        this.delegate.rmsNormBf16(inputAddress, weightAddress, outputAddress, rows, width, epsilon);
    }

    @Override
    public void rmsNormUnitOffsetBf16(
            long inputAddress, long weightAddress, long outputAddress, int rows, int width, float epsilon) {
        this.delegate.rmsNormUnitOffsetBf16(inputAddress, weightAddress, outputAddress, rows, width, epsilon);
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
        this.delegate.linearQ3Bf16(
                inputAddress, weightsAddress, outputAddress, rows, inFeatures, outFeatures, weightsByteSize);
    }

    @Override
    public void gdnProjectionsBf16(
            long input,
            long q4Weights,
            long q5Weights,
            long queryKeyOutput,
            long valueZOutput,
            int rows,
            int hidden,
            int queryKeyWidth,
            int valueZWidth,
            long q4Bytes,
            long q5Bytes) {
        this.delegate.gdnProjectionsBf16(
                input,
                q4Weights,
                q5Weights,
                queryKeyOutput,
                valueZOutput,
                rows,
                hidden,
                queryKeyWidth,
                valueZWidth,
                q4Bytes,
                q5Bytes);
    }

    @Override
    public void linearQ4Bf16(
            long inputAddress,
            long weightsAddress,
            long outputAddress,
            int rows,
            int inFeatures,
            int outFeatures,
            long weightsByteSize) {
        this.delegate.linearQ4Bf16(
                inputAddress, weightsAddress, outputAddress, rows, inFeatures, outFeatures, weightsByteSize);
    }

    @Override
    public void linearQ5Bf16(
            long inputAddress,
            long weightsAddress,
            long outputAddress,
            int rows,
            int inFeatures,
            int outFeatures,
            long weightsByteSize) {
        this.delegate.linearQ5Bf16(
                inputAddress, weightsAddress, outputAddress, rows, inFeatures, outFeatures, weightsByteSize);
    }

    @Override
    public void linearBf16ToFloat(
            long inputAddress, long weightsAddress, long outputAddress, int rows, int inFeatures, int outFeatures) {
        this.delegate.linearBf16ToFloat(inputAddress, weightsAddress, outputAddress, rows, inFeatures, outFeatures);
    }

    @Override
    public void gdnControlFp32(
            long aProjectionAddress,
            long bProjectionAddress,
            long aLogAddress,
            long dtBiasAddress,
            long alphaOutputAddress,
            long betaOutputAddress,
            int rows,
            int heads) {
        this.delegate.gdnControlFp32(
                aProjectionAddress,
                bProjectionAddress,
                aLogAddress,
                dtBiasAddress,
                alphaOutputAddress,
                betaOutputAddress,
                rows,
                heads);
    }

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
        this.delegate.gdnConvolutionBf16(
                queryKeyAddress,
                valueZAddress,
                convolutionWeightsAddress,
                convolutionStateAddress,
                outputAddress,
                rows,
                queryKeyWidth,
                valueWidth,
                convolutionWidth,
                kernelSize);
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
        this.delegate.gdnRecurrenceBf16(
                convolvedAddress,
                alphaAddress,
                betaAddress,
                recurrentStateAddress,
                outputAddress,
                rows,
                keyHeads,
                valueHeads,
                keyHeadDim,
                valueHeadDim,
                outputScale);
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
            float epsilon) {
        this.delegate.gdnGatedRmsNormBf16(
                recurrentAddress, valueZAddress, normWeightAddress, outputAddress, rows, valueHeads, headDim, epsilon);
    }

    @Override
    public void residualAddBf16(long residualAddress, long deltaAddress, long outputAddress, int rows, int width) {
        this.delegate.residualAddBf16(residualAddress, deltaAddress, outputAddress, rows, width);
    }

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
            long q5Bytes) {
        this.delegate.attentionProducersNvfp4(
                input,
                q4,
                q5,
                queryNorm,
                keyNorm,
                queryKey,
                gate,
                keys,
                values,
                rows,
                hidden,
                queryHeads,
                keyHeads,
                headDim,
                rotaryDim,
                start,
                epsilon,
                theta,
                q4Bytes,
                q5Bytes);
    }

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
            long downBytes) {
        this.delegate.q3FfnStreamedBf16(
                input,
                gateWeights,
                downWeights,
                output,
                slots,
                accumulators,
                rows,
                hidden,
                intermediate,
                gateBytes,
                downBytes);
    }

    @Override
    public void q3FfnDownBf16(
            long input, long weights, long output, int rows, int width, int outputs, long weightBytes) {
        this.delegate.q3FfnDownBf16(input, weights, output, rows, width, outputs, weightBytes);
    }

    @Override
    public void q3GateUpSwiGluBf16(
            long input, long weights, long output, int rows, int width, int outputs, long weightBytes) {
        this.delegate.q3GateUpSwiGluBf16(input, weights, output, rows, width, outputs, weightBytes);
    }

    @Override
    public void residualRmsNormBf16(
            long residual, long delta, long weight, long hidden, long normalized, int rows, int width, float epsilon) {
        this.delegate.residualRmsNormBf16(residual, delta, weight, hidden, normalized, rows, width, epsilon);
    }

    @Override
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
        this.delegate.gdnProjectControlFp32(input, aWeight, bWeight, aLog, dtBias, g, beta, rows, width, heads);
    }

    @Override
    public void swiGluBf16(long gateUpAddress, long outputAddress, int rows, int intermediateSize) {
        this.delegate.swiGluBf16(gateUpAddress, outputAddress, rows, intermediateSize);
    }

    @Override
    public void zeroDeviceMemory(long address, long byteSize) {
        this.delegate.zeroDeviceMemory(address, byteSize);
    }

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
            float epsilon,
            double ropeTheta) {
        this.delegate.attentionQkNormRopeBf16(
                queryKeyAddress,
                queryNormAddress,
                keyNormAddress,
                outputAddress,
                rows,
                queryHeads,
                keyValueHeads,
                headDim,
                rotaryDim,
                startPosition,
                epsilon,
                ropeTheta);
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
            long startPosition) {
        this.delegate.attentionKvAppendNvfp4(
                queryKeyAddress,
                gateValueAddress,
                keyCacheAddress,
                valueCacheAddress,
                rows,
                queryWidth,
                keyValueWidth,
                startPosition);
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
            long scratchAddress) {
        this.delegate.attentionCausalNvfp4(
                queryKeyAddress,
                gateValueAddress,
                keyCacheAddress,
                valueCacheAddress,
                outputAddress,
                rows,
                queryHeads,
                keyValueHeads,
                headDim,
                cacheLength,
                startPosition,
                scratchAddress);
    }
}
