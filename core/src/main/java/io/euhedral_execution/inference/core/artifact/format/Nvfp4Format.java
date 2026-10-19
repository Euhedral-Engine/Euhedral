package io.euhedral_execution.inference.core.artifact.format;

import io.euhedral_execution.inference.core.artifact.Nvfp4Layout;
import io.euhedral_execution.inference.core.artifact.TensorDataType;
import io.euhedral_execution.inference.core.artifact.TensorDescriptor;
import io.euhedral_execution.inference.core.artifact.TensorHandle;
import io.euhedral_execution.inference.core.artifact.WeightFormat;
import io.euhedral_execution.inference.core.artifact.WeightLayout;
import io.euhedral_execution.inference.core.artifact.WeightLoadException;
import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.gpu.GpuMemory;
import io.euhedral_execution.inference.core.gpu.KernelArguments;
import io.euhedral_execution.inference.core.gpu.ScratchUse;
import io.euhedral_execution.inference.core.model.qwen38.ArtifactProfile;
import io.euhedral_execution.inference.core.model.qwen38.ExecutionPlan;
import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import java.util.List;

/// Block-scaled NVFP4 ([Nvfp4Layout]) in its plain layout or with SD4 scale tables (compressed). Both serve any
/// grouped or NVFP4 role and keep the role's kind; their tensor-core kernels take rows independently.
public final class Nvfp4Format implements LinearFormat {

    private static final ThreadLocal<KernelArguments> EMBEDDING_ARGUMENTS =
            ThreadLocal.withInitial(KernelArguments::new);

    private final WeightLayout layout;

    Nvfp4Format(WeightLayout layout) {
        if (!Nvfp4Layout.supports(layout)) throw new IllegalArgumentException("not an NVFP4 layout: " + layout);
        this.layout = layout;
    }

    private boolean sd4() {
        return this.layout == WeightLayout.ROW_SPLIT_K128_SD4_V1;
    }

    @Override
    public WeightFormat format() {
        return WeightFormat.NVFP4;
    }

    @Override
    public WeightLayout layout() {
        return this.layout;
    }

    @Override
    public long byteSize(long[] shape, TensorDataType dataType) {
        if (sd4()) {
            if (dataType != TensorDataType.BF16) {
                throw new IllegalArgumentException("SD4 layout requires NVFP4 storage of BF16 source metadata");
            }
        } else if (dataType != TensorDataType.BF16) {
            throw new IllegalArgumentException("grouped compact storage requires BF16 source metadata");
        }
        if (shape.length != 2 || shape[0] <= 0 || shape[1] <= 0) {
            throw new IllegalArgumentException("row-split layout requires a positive rank-2 shape");
        }
        return Nvfp4Layout.byteSize(shape[0], shape[1], this.layout);
    }

    @Override
    public boolean acceptsByteSize(long[] shape, TensorDataType dataType, long byteSize) {
        return byteSize(shape, dataType) == byteSize;
    }

    @Override
    public boolean accepts(TensorDescriptor descriptor, ExpectedTensor expected) {
        if (!Arrays.equals(expected.shape(), descriptor.shape()) || expected.dataType() != descriptor.dataType())
            return false;
        if (!sd4()) return expected.layout() == WeightLayout.ROW_SPLIT_K128_V1;
        return (expected.format() == WeightFormat.NVFP4 || expected.layout() == WeightLayout.ROW_SPLIT_K128_V1)
                && (expected.layout() == WeightLayout.ROW_SPLIT_K128_SD4_V1
                        || expected.layout() == WeightLayout.ROW_SPLIT_K128_V1);
    }

    @Override
    public boolean stageable() {
        return true;
    }

    @Override
    public ArtifactProfile.Quantization quantization() {
        return ArtifactProfile.Quantization.NVFP4;
    }

    @Override
    public boolean compressed() {
        return sd4();
    }

    @Override
    public void validatePayload(TensorDescriptor descriptor, MemorySegment payload) throws WeightLoadException {
        try {
            Nvfp4Layout.validate(payload, descriptor.shape()[0], descriptor.shape()[1], descriptor.layout());
        } catch (IllegalArgumentException exception) {
            throw new WeightLoadException(
                    "invalid NVFP4 tensor '" + descriptor.name() + "': " + exception.getMessage());
        }
    }

    @Override
    public boolean splitsRows() {
        return true;
    }

    @Override
    public List<TensorHandle> rows(List<TensorHandle> object, long first, long rows, String name, GpuMemory gpu) {
        TensorHandle source = object.getFirst();
        long k = source.shape()[1];
        WeightLayout layout = source.layout();
        long rowBytes = Nvfp4Layout.paddedK(k) / 2, rowScales = Nvfp4Layout.rowScaleBytes(k, layout);
        long sourceScales = Nvfp4Layout.scaleOffset(source.shape()[0], k);
        long sourceTrailer = Nvfp4Layout.trailerOffset(source.shape()[0], k, layout);
        long scales = Nvfp4Layout.scaleOffset(rows, k), trailer = Nvfp4Layout.trailerOffset(rows, k, layout);
        long bytes = Nvfp4Layout.byteSize(rows, k, layout);
        long address = gpu.allocate(bytes);
        try {
            long base = source.deviceAddress();
            gpu.copyDeviceToDevice(address, base + first * rowBytes, rows * rowBytes);
            gpu.copyDeviceToDevice(address + scales, base + sourceScales + first * rowScales, rows * rowScales);
            gpu.copyDeviceToDevice(address + trailer, base + sourceTrailer, Nvfp4Layout.trailerBytes(layout));
        } catch (RuntimeException | Error failure) {
            gpu.free(address);
            throw failure;
        }
        return List.of(new TensorHandle(
                name, new long[] {rows, k}, source.dataType(), WeightFormat.NVFP4, source.layout(), address, bytes));
    }

    @Override
    public ExecutionPlan.Kind kind(ExecutionPlan.Kind role) {
        return role;
    }

    @Override
    public TensorHandle validateHandle(TensorHandle handle, long rows, int width, WeightFormat roleFormat) {
        return GroupedLinearFormat.validateQuantized(this, handle, rows, width, true);
    }

    @Override
    public boolean rowExact(ExecutionGpu gpu, ExecutionPlan.Kind kind) {
        return GroupedLinearFormat.groupedKind(kind) || kind == ExecutionPlan.Kind.BF16_PROJECTION;
    }

    @Override
    public ScratchUse scratchUse(ExecutionPlan.Kind kind, boolean multiRow) {
        return ScratchRules.nvfp4(kind, multiRow);
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
        linear(
                gpu,
                launch.hostMapped(),
                launch.input(),
                weight.deviceAddress(),
                launch.output(),
                launch.rows(),
                launch.inFeatures(),
                launch.outFeatures(),
                weight.byteSize());
    }

    /// Rows per call of an NVFP4 linear: all of them, or at most [ExecutionGpu#NVFP4_PLAIN_LOAD_MAX_ROWS] when the
    /// weights are host-mapped, so that no call reaches the tile kernels (TMA over mapped host memory) or the
    /// scalar reference (a sweep of the weights over the bus per row).
    public static int rowsPerCall(boolean hostMapped, int rows) {
        return hostMapped ? Math.min(rows, ExecutionGpu.NVFP4_PLAIN_LOAD_MAX_ROWS) : rows;
    }

    /// An NVFP4 linear over `rows` rows in the calls [#rowsPerCall] allows.
    public static void linear(
            ExecutionGpu gpu,
            boolean hostMapped,
            long input,
            long weights,
            long output,
            int rows,
            int inFeatures,
            int outFeatures,
            long weightBytes) {
        int perCall = rowsPerCall(hostMapped, rows);
        for (int first = 0; first < rows; first += perCall)
            gpu.linearNvfp4Bf16(
                    input + (long) first * inFeatures * Short.BYTES,
                    weights,
                    output + (long) first * outFeatures * Short.BYTES,
                    Math.min(perCall, rows - first),
                    inFeatures,
                    outFeatures,
                    weightBytes);
    }

    @Override
    public boolean gateUpRegion() {
        return true;
    }

    @Override
    public void gateUpSwiGlu(ExecutionGpu gpu, LinearLaunch launch) {
        TensorHandle weight = launch.weights().getFirst();
        gpu.nvfp4GateUpSwiGluBf16(
                launch.input(),
                weight.deviceAddress(),
                launch.output(),
                launch.rows(),
                launch.inFeatures(),
                launch.outFeatures(),
                weight.byteSize());
    }

    /// Plain (not SD4) tables ([LinearKernel#NVFP4_EMBEDDING]).
    @Override
    public boolean embeds() {
        return !sd4();
    }

    @Override
    public void embed(ExecutionGpu gpu, EmbeddingLaunch launch) {
        if (!embeds()) {
            LinearFormat.super.embed(gpu, launch);
            return;
        }
        if (launch.hidden() % Nvfp4Layout.BLOCK != 0 || Nvfp4Layout.paddedK(launch.hidden()) != launch.hidden())
            throw new IllegalArgumentException("an NVFP4 embedding needs a width that is a multiple of 128");
        gpu.launchTableKernel(
                LinearKernel.NVFP4_EMBEDDING,
                launch.rows(),
                1,
                1,
                128,
                1,
                1,
                0,
                EMBEDDING_ARGUMENTS
                        .get()
                        .clear()
                        .pointer(launch.tokenIds())
                        .pointer(launch.table().deviceAddress())
                        .pointer(launch.output())
                        .int32(launch.rows())
                        .int32(launch.vocabulary())
                        .int32(launch.hidden())
                        .pointer(Nvfp4Layout.scaleOffset(launch.vocabulary(), launch.hidden()))
                        .pointer(Nvfp4Layout.globalScaleOffset(launch.vocabulary(), launch.hidden())));
    }
}
