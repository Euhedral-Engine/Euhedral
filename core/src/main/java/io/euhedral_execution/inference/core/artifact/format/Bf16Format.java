package io.euhedral_execution.inference.core.artifact.format;

import io.euhedral_execution.inference.core.artifact.CompactTensorLayout;
import io.euhedral_execution.inference.core.artifact.TensorDataType;
import io.euhedral_execution.inference.core.artifact.TensorDescriptor;
import io.euhedral_execution.inference.core.artifact.TensorHandle;
import io.euhedral_execution.inference.core.artifact.WeightFormat;
import io.euhedral_execution.inference.core.artifact.WeightLayout;
import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.gpu.GpuMemory;
import io.euhedral_execution.inference.core.gpu.ScratchUse;
import io.euhedral_execution.inference.core.model.qwen38.ArtifactProfile;
import io.euhedral_execution.inference.core.model.qwen38.ExecutionPlan;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/// Unquantized BF16 values, contiguous: the unquantized model's projections, which run on tensor-core kernels whose
/// rows do not depend on how many run together, and (as `accepts` allows) any quantized role's object.
final class Bf16Format implements LinearFormat {

    @Override
    public WeightFormat format() {
        return WeightFormat.BF16;
    }

    @Override
    public WeightLayout layout() {
        return WeightLayout.CONTIGUOUS_LE_V1;
    }

    @Override
    public long byteSize(long[] shape, TensorDataType dataType) {
        return CompactTensorLayout.directSize(shape, dataType, WeightFormat.BF16);
    }

    @Override
    public boolean acceptsByteSize(long[] shape, TensorDataType dataType, long byteSize) {
        return byteSize(shape, dataType) == byteSize;
    }

    @Override
    public boolean accepts(TensorDescriptor descriptor, ExpectedTensor expected) {
        boolean same =
                Arrays.equals(expected.shape(), descriptor.shape()) && expected.dataType() == descriptor.dataType();
        if (GroupedLinearFormat.grouped(expected.format()) || expected.format() == WeightFormat.NVFP4) return same;
        return same && expected.format() == WeightFormat.BF16 && expected.layout() == WeightLayout.CONTIGUOUS_LE_V1;
    }

    @Override
    public boolean stageable() {
        return true;
    }

    @Override
    public ArtifactProfile.Quantization quantization() {
        return ArtifactProfile.Quantization.BF16;
    }

    @Override
    public boolean splitsRows() {
        return true;
    }

    @Override
    public List<TensorHandle> rows(List<TensorHandle> object, long first, long rows, String name, GpuMemory gpu) {
        TensorHandle source = object.getFirst();
        long k = source.shape()[1], bytes = rows * k * Short.BYTES;
        long address = gpu.allocate(bytes);
        try {
            gpu.copyDeviceToDevice(address, source.deviceAddress() + first * k * Short.BYTES, bytes);
        } catch (RuntimeException | Error failure) {
            gpu.free(address);
            throw failure;
        }
        return List.of(new TensorHandle(
                name,
                new long[] {rows, k},
                source.dataType(),
                WeightFormat.BF16,
                WeightLayout.CONTIGUOUS_LE_V1,
                address,
                bytes));
    }

    @Override
    public ExecutionPlan.Kind kind(ExecutionPlan.Kind role) {
        return ExecutionPlan.Kind.BF16_PROJECTION;
    }

    @Override
    public TensorHandle validateHandle(TensorHandle handle, long rows, int width, WeightFormat roleFormat) {
        Objects.requireNonNull(handle, "weight");
        long[] shape = handle.shape();
        if (shape == null
                || shape.length != 2
                || shape[0] != rows
                || shape[1] != width
                || width % 32 != 0
                || rows % 8 != 0
                || (handle.deviceAddress() == 0 && !handle.hostBacked() && !handle.diskBacked())
                || handle.dataType() != TensorDataType.BF16
                || handle.layout() != WeightLayout.CONTIGUOUS_LE_V1
                || handle.byteSize() != rows * width * Short.BYTES)
            throw new IllegalArgumentException("unsupported BF16 projection layout or dimensions: " + handle.name());
        return handle;
    }

    @Override
    public boolean rowExact(ExecutionGpu gpu, ExecutionPlan.Kind kind) {
        return kind == ExecutionPlan.Kind.BF16_PROJECTION;
    }

    @Override
    public ScratchUse scratchUse(ExecutionPlan.Kind kind, boolean multiRow) {
        return ScratchRules.nonNvfp4(kind, multiRow);
    }

    @Override
    public long scratchBytes(
            ExecutionGpu gpu, ScratchUse use, int rows, int inFeatures, int outFeatures, List<TensorHandle> weights) {
        return gpu.scratchBytes(
                use, rows, inFeatures, outFeatures, weights.getFirst().layout());
    }

    @Override
    public void linear(ExecutionGpu gpu, LinearLaunch launch) {
        long weights = launch.weights().getFirst().deviceAddress();
        switch (launch.kind()) {
            case BF16_PROJECTION ->
                io.euhedral_execution.inference.core.model.qwen4.Ops.linearBf16RowExact(
                        gpu,
                        launch.input(),
                        weights,
                        launch.output(),
                        launch.rows(),
                        launch.inFeatures(),
                        launch.outFeatures());
            case BF16_LINEAR ->
                gpu.linearBf16ToFloat(
                        launch.input(),
                        weights,
                        launch.output(),
                        launch.rows(),
                        launch.inFeatures(),
                        launch.outFeatures());
            default ->
                throw new IllegalArgumentException("linear stage received non-linear instruction: " + launch.kind());
        }
    }
}
