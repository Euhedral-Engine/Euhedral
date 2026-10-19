package io.euhedral_execution.inference.core.artifact.format;

import io.euhedral_execution.inference.core.artifact.TensorDataType;
import io.euhedral_execution.inference.core.artifact.TensorHandle;
import io.euhedral_execution.inference.core.artifact.WeightFormat;
import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.gpu.ScratchUse;
import io.euhedral_execution.inference.core.model.qwen38.ArtifactProfile;
import io.euhedral_execution.inference.core.model.qwen38.ExecutionPlan;
import java.util.List;
import java.util.Objects;

/// What the grouped formats (Q3, Q4 and Q5 at 64-value groups, and Q3's P2E2 layout) share: the width-specific
/// kinds and kernels, the scratch routes, and the dimension checks of a handle.
abstract class GroupedLinearFormat implements LinearFormat {

    static boolean grouped(WeightFormat format) {
        return format == WeightFormat.Q3_G64_FP16
                || format == WeightFormat.Q4_G64_FP16
                || format == WeightFormat.Q5_G64_FP16;
    }

    static boolean groupedKind(ExecutionPlan.Kind kind) {
        return kind == ExecutionPlan.Kind.Q3_LINEAR
                || kind == ExecutionPlan.Kind.Q4_LINEAR
                || kind == ExecutionPlan.Kind.Q5_LINEAR;
    }

    /// The checks every row-split quantized handle passes: rank-2 `rows` x `width` with a width of whole groups,
    /// a payload to read, BF16 source metadata, a role it can serve and a byte size its layout accepts.
    static TensorHandle validateQuantized(
            LinearFormat format, TensorHandle handle, long rows, int width, boolean roleOk) {
        Objects.requireNonNull(handle, "weight");
        long[] shape = handle.shape();
        if (shape == null
                || shape.length != 2
                || shape[0] != rows
                || shape[1] != width
                || width % 64 != 0
                || (handle.deviceAddress() == 0 && !handle.hostBacked() && !handle.diskBacked())
                || handle.dataType() != TensorDataType.BF16
                || !roleOk
                || !format.acceptsByteSize(shape, handle.dataType(), handle.byteSize())) {
            throw new IllegalArgumentException("unsupported quantized weight layout or dimensions: " + handle.name());
        }
        return handle;
    }

    @Override
    public ArtifactProfile.Quantization quantization() {
        return ArtifactProfile.Quantization.Q3;
    }

    @Override
    public boolean rowExact(ExecutionGpu gpu, ExecutionPlan.Kind kind) {
        // BF16 projections run on kernels whose rows do not depend on how many run together.
        return (groupedKind(kind) && gpu.rowExactQuantizedLinears()) || kind == ExecutionPlan.Kind.BF16_PROJECTION;
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
        TensorHandle weight = launch.weights().getFirst();
        switch (launch.kind()) {
            case Q3_LINEAR ->
                gpu.linearQ3Bf16(
                        launch.input(),
                        weight.deviceAddress(),
                        launch.output(),
                        launch.rows(),
                        launch.inFeatures(),
                        launch.outFeatures(),
                        weight.byteSize(),
                        weight.layout());
            case Q4_LINEAR ->
                gpu.linearQ4Bf16(
                        launch.input(),
                        weight.deviceAddress(),
                        launch.output(),
                        launch.rows(),
                        launch.inFeatures(),
                        launch.outFeatures(),
                        weight.byteSize());
            case Q5_LINEAR ->
                gpu.linearQ5Bf16(
                        launch.input(),
                        weight.deviceAddress(),
                        launch.output(),
                        launch.rows(),
                        launch.inFeatures(),
                        launch.outFeatures(),
                        weight.byteSize());
            default ->
                throw new IllegalArgumentException("linear stage received non-linear instruction: " + launch.kind());
        }
    }

    @Override
    public void gateUpSwiGlu(ExecutionGpu gpu, LinearLaunch launch) {
        TensorHandle weight = launch.weights().getFirst();
        gpu.q3GateUpSwiGluBf16(
                launch.input(),
                weight.deviceAddress(),
                launch.output(),
                launch.rows(),
                launch.inFeatures(),
                launch.outFeatures(),
                weight.byteSize(),
                weight.layout());
    }

    /// Q3 tables, in either layout ([ExecutionGpu#embedQ3]).
    @Override
    public boolean embeds() {
        return format() == WeightFormat.Q3_G64_FP16;
    }

    @Override
    public void embed(ExecutionGpu gpu, EmbeddingLaunch launch) {
        if (!embeds()) LinearFormat.super.embed(gpu, launch);
        else
            gpu.embedQ3(
                    launch.tokenIds(),
                    launch.table().deviceAddress(),
                    launch.table().byteSize(),
                    launch.output(),
                    launch.rows(),
                    launch.vocabulary(),
                    launch.hidden(),
                    layout());
    }
}
