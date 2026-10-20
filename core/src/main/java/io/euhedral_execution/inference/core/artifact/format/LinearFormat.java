package io.euhedral_execution.inference.core.artifact.format;

import io.euhedral_execution.inference.core.artifact.TensorDataType;
import io.euhedral_execution.inference.core.artifact.TensorDescriptor;
import io.euhedral_execution.inference.core.artifact.TensorHandle;
import io.euhedral_execution.inference.core.artifact.WeightFormat;
import io.euhedral_execution.inference.core.artifact.WeightLayout;
import io.euhedral_execution.inference.core.artifact.WeightLoadException;
import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.gpu.GpuMemory;
import io.euhedral_execution.inference.core.gpu.ScratchUse;
import io.euhedral_execution.inference.core.model.qwen38.ArtifactProfile;
import io.euhedral_execution.inference.core.model.qwen38.ExecutionPlan;
import java.lang.foreign.MemorySegment;
import java.util.List;

/// Everything the runtime decides per weight format: how a payload is sized, validated and accepted for a
/// registered object, whether it can be staged, which kind and scratch route its linears take, and how they launch.
/// One implementation per `(WeightFormat, WeightLayout)` key, found through [LinearFormats].
public interface LinearFormat {

    WeightFormat format();

    WeightLayout layout();

    // storage

    /// The payload size of an object of `shape`; IllegalArgumentException where the size depends on the contents.
    long byteSize(long[] shape, TensorDataType dataType);

    boolean acceptsByteSize(long[] shape, TensorDataType dataType, long byteSize);

    /// Whether `descriptor`, of this format, may stand for the registered `expected` object.
    boolean accepts(TensorDescriptor descriptor, ExpectedTensor expected);

    /// The companion tensors that accompany the object `name`.
    default List<String> companions(String name) {
        return List.of();
    }

    /// Whether the payload can be staged to the device whole.
    boolean stageable();

    ArtifactProfile.Quantization quantization();

    default boolean compressed() {
        return false;
    }

    default void validatePayload(TensorDescriptor descriptor, MemorySegment payload) throws WeightLoadException {}

    /// Whether [#rows] can cut the object into row ranges (the MTP split).
    default boolean splitsRows() {
        return false;
    }

    /// [object, companions...] of rows [first, first + rows) as new device allocations named `name` (MTP split).
    default List<TensorHandle> rows(List<TensorHandle> object, long first, long rows, String name, GpuMemory gpu) {
        throw new UnsupportedOperationException();
    }

    /// [#rows] for a format whose objects carry host-known segment `ends` (trellis); the others ignore them.
    default List<TensorHandle> rows(
            List<TensorHandle> object, long first, long rows, String name, GpuMemory gpu, int[] ends) {
        return rows(object, first, rows, name, gpu);
    }

    /// The scratch an object of `descriptor` needs at `maxRows` rows, planned before any launch with the trellis
    /// thresholds `trellis` (the formats that have none ignore them).
    default long planScratchBytes(TensorDescriptor descriptor, int maxRows, TrellisSettings trellis) {
        return 0;
    }

    // execution (qwen38)

    /// The kind that runs a linear of this format in `role`.
    ExecutionPlan.Kind kind(ExecutionPlan.Kind role);

    /// `handle` if it is a valid `rows` x `width` weight of this format for a role of `roleFormat`; else
    /// IllegalArgumentException.
    TensorHandle validateHandle(TensorHandle handle, long rows, int width, WeightFormat roleFormat);

    /// Whether a multi-row launch of `kind` computes each row as a one-row launch does.
    boolean rowExact(ExecutionGpu gpu, ExecutionPlan.Kind kind);

    ScratchUse scratchUse(ExecutionPlan.Kind kind, boolean multiRow);

    long scratchBytes(
            ExecutionGpu gpu, ScratchUse use, int rows, int inFeatures, int outFeatures, List<TensorHandle> weights);

    /// The scratch of a launch that is never a prefill: at most the decode-type views' rows, or a verification.
    default long decodeScratchBytes(
            ExecutionGpu gpu, ScratchUse use, int rows, int inFeatures, int outFeatures, List<TensorHandle> weights) {
        return scratchBytes(gpu, use, rows, inFeatures, outFeatures, weights);
    }

    void linear(ExecutionGpu gpu, LinearLaunch launch);

    // token embedding

    /// Whether a `[vocabulary, hidden]` token embedding table stored in this format can be looked up ([#embed]).
    default boolean embeds() {
        return false;
    }

    /// Queues the lookup `launch` describes, its table in this format, on the selected stream.
    default void embed(ExecutionGpu gpu, EmbeddingLaunch launch) {
        throw new UnsupportedOperationException("no token embedding lookup in " + format() + "/" + layout());
    }

    /// Whether the gate/up projection of this format fuses with SwiGLU in a prefill region.
    default boolean gateUpRegion() {
        return false;
    }

    default void gateUpSwiGlu(ExecutionGpu gpu, LinearLaunch launch) {
        throw new UnsupportedOperationException();
    }
}
