package io.euhedral_execution.inference.core.artifact.format;

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
import java.lang.foreign.Arena;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/// Trellis (EXL3) weights: `[k/16, n/16, w]` int16 tile words ([TrellisLayout]) and three companions that stay on the
/// device (`<name>.suh` `[s, k]`, `<name>.svh` `[n]`, `<name>.segments` `[s]`). An object keeps its engine name and
/// logical shape `[n, k]`; its rows are cut into `s` segments that share one `suh` row. Only the payload is ever
/// host- or disk-backed and staged.
public final class TrellisFormat implements LinearFormat {

    /// Companion suffixes, in the order [#companions] lists them and [#rows] returns them.
    public static final String SUH = ".suh";

    public static final String SVH = ".svh";
    public static final String SEGMENTS = ".segments";

    /// Most segments an activation scratch must hold: the bound used where only the payload is known.
    private static final int SCRATCH_SEGMENTS = TrellisLayout.MAX_SEGMENTS;

    private static final long SCRATCH_ALIGNMENT = 256;

    TrellisFormat() {}

    @Override
    public WeightFormat format() {
        return WeightFormat.TRELLIS;
    }

    @Override
    public WeightLayout layout() {
        return WeightLayout.TRELLIS_TILE16_V1;
    }

    /// The logical `[n, k]` of a payload of `shape`, after checking the shape is a trellis one.
    private static long[] logical(long[] shape) {
        if (shape == null || shape.length != 3 || shape[0] <= 0 || shape[1] <= 0 || shape[2] <= 0)
            throw new IllegalArgumentException("trellis layout requires a positive [k/16, n/16, w] shape");
        if (shape[2] > Integer.MAX_VALUE || !TrellisLayout.wordsValid((int) shape[2]))
            throw new IllegalArgumentException("trellis tile width " + shape[2] + " is not a stored width");
        long k = PayloadSizes.multiply(shape[0], 16), n = PayloadSizes.multiply(shape[1], 16);
        if (k % TrellisLayout.BLOCK != 0 || n % TrellisLayout.BLOCK != 0)
            throw new IllegalArgumentException("trellis k=" + k + " and n=" + n + " must be multiples of 128");
        return new long[] {n, k};
    }

    @Override
    public long byteSize(long[] shape, TensorDataType dataType) {
        if (dataType != TensorDataType.BF16)
            throw new IllegalArgumentException("trellis storage requires BF16 source metadata");
        logical(shape);
        return PayloadSizes.multiply(PayloadSizes.multiply(PayloadSizes.multiply(shape[0], shape[1]), shape[2]), 2);
    }

    @Override
    public boolean acceptsByteSize(long[] shape, TensorDataType dataType, long byteSize) {
        return byteSize(shape, dataType) == byteSize;
    }

    @Override
    public boolean accepts(TensorDescriptor descriptor, ExpectedTensor expected) {
        if (descriptor.dataType() != TensorDataType.BF16 || expected.dataType() != TensorDataType.BF16) return false;
        if (!quantizedRole(expected.format())) return false;
        long[] nk;
        try {
            nk = logical(descriptor.shape());
        } catch (IllegalArgumentException refused) {
            return false;
        }
        return Arrays.equals(nk, expected.shape());
    }

    private static boolean quantizedRole(WeightFormat format) {
        return switch (format) {
            case Q3_G64_FP16, Q4_G64_FP16, Q5_G64_FP16, NVFP4 -> true;
            default -> false;
        };
    }

    @Override
    public List<String> companions(String name) {
        return List.of(name + SUH, name + SVH, name + SEGMENTS);
    }

    @Override
    public boolean stageable() {
        return true;
    }

    @Override
    public ArtifactProfile.Quantization quantization() {
        return ArtifactProfile.Quantization.TRELLIS;
    }

    @Override
    public boolean splitsRows() {
        return true;
    }

    /// The prefill route's bytes for an object of `descriptor` at `maxRows` rows: the rotated activations of at most
    /// [TrellisLayout#MAX_SEGMENTS] segments, the whole object decoded to BF16 `[n, k]`, and the GEMM outputs
    /// `[rows, n]`, each region 256-aligned.
    @Override
    public long planScratchBytes(TensorDescriptor descriptor, int maxRows) {
        long[] nk = logical(descriptor.shape());
        long n = nk[0], k = nk[1], rows = maxRows;
        long activations = PayloadSizes.multiply(PayloadSizes.multiply(SCRATCH_SEGMENTS, rows), k * 2);
        long decoded = PayloadSizes.multiply(n, k * 2);
        long outputs = PayloadSizes.multiply(rows, n * 2);
        return PayloadSizes.add(
                PayloadSizes.add(
                        PayloadSizes.alignUp(activations, SCRATCH_ALIGNMENT),
                        PayloadSizes.alignUp(decoded, SCRATCH_ALIGNMENT)),
                PayloadSizes.alignUp(outputs, SCRATCH_ALIGNMENT));
    }

    /// [object, suh, svh, segments] of rows `[first, first + rows)` as new device allocations named `name`, `name.suh`,
    /// `name.svh` and `name.segments`. Both ends of the range must be segment boundaries (or 0 and n). The
    /// payload's tile columns are copied per k-tile row; `svh`, the `suh` rows and the segment ends are cut and
    /// rebased.
    @Override
    public List<TensorHandle> rows(List<TensorHandle> object, long first, long rows, String name, GpuMemory gpu) {
        if (object.size() != 4) throw new IllegalArgumentException("a trellis object has 4 tensors: " + object.size());
        TensorHandle payload = object.get(0), suh = object.get(1), svh = object.get(2), segments = object.get(3);
        long[] nk = logical(payload.shape());
        long n = nk[0], k = nk[1];
        int w = (int) payload.shape()[2];
        int[] ends;
        try (Arena arena = Arena.ofConfined()) {
            var host = arena.allocate(segments.byteSize());
            gpu.copyDeviceToHost(host, segments.deviceAddress(), segments.byteSize());
            ends = TrellisLayout.readEnds(host);
        }
        long last = first + rows;
        if (first < 0 || rows <= 0 || last > n || first % 16 != 0 || rows % 16 != 0)
            throw new IllegalArgumentException("rows " + first + ".." + last + " are outside " + name);
        int start = first == 0 ? -1 : indexOf(ends, first), stop = indexOf(ends, last);
        if ((first != 0 && start < 0) || stop < 0 || stop <= start)
            throw new IllegalArgumentException(
                    "rows " + first + ".." + last + " of " + name + " do not begin and end on segment boundaries");
        int lo = start + 1, count = stop - start;
        int[] cut = new int[count];
        for (int index = 0; index < count; index++) cut[index] = (int) (ends[lo + index] - first);

        long tileRows = payload.shape()[0], tilesFirst = first / 16, tilesCut = rows / 16;
        long rowWords = tilesCut * w, sourceRowWords = payload.shape()[1] * w;
        long payloadBytes = tileRows * rowWords * 2, suhBytes = (long) count * k * 2, svhBytes = rows * 2;
        long segmentBytes = (long) count * Integer.BYTES;
        List<Long> allocated = new ArrayList<>();
        try {
            long payloadAddress = allocate(gpu, payloadBytes, allocated);
            long suhAddress = allocate(gpu, suhBytes, allocated);
            long svhAddress = allocate(gpu, svhBytes, allocated);
            long segmentsAddress = allocate(gpu, segmentBytes, allocated);
            for (long row = 0; row < tileRows; row++)
                gpu.copyDeviceToDevice(
                        payloadAddress + row * rowWords * 2,
                        payload.deviceAddress() + (row * sourceRowWords + tilesFirst * w) * 2,
                        rowWords * 2);
            gpu.copyDeviceToDevice(suhAddress, suh.deviceAddress() + (long) lo * k * 2, suhBytes);
            gpu.copyDeviceToDevice(svhAddress, svh.deviceAddress() + first * 2, svhBytes);
            try (Arena arena = Arena.ofConfined()) {
                var host = arena.allocate(segmentBytes);
                for (int index = 0; index < count; index++)
                    host.set(
                            ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN),
                            (long) index * Integer.BYTES,
                            cut[index]);
                gpu.copyHostToDevice(segmentsAddress, host, segmentBytes);
            }
            return List.of(
                    new TensorHandle(
                            name,
                            new long[] {tileRows, tilesCut, w},
                            payload.dataType(),
                            WeightFormat.TRELLIS,
                            WeightLayout.TRELLIS_TILE16_V1,
                            payloadAddress,
                            payloadBytes),
                    new TensorHandle(
                            name + SUH,
                            new long[] {count, k},
                            suh.dataType(),
                            suh.format(),
                            suh.layout(),
                            suhAddress,
                            suhBytes),
                    new TensorHandle(
                            name + SVH,
                            new long[] {rows},
                            svh.dataType(),
                            svh.format(),
                            svh.layout(),
                            svhAddress,
                            svhBytes),
                    new TensorHandle(
                            name + SEGMENTS,
                            new long[] {count},
                            segments.dataType(),
                            segments.format(),
                            segments.layout(),
                            segmentsAddress,
                            segmentBytes));
        } catch (RuntimeException | Error failure) {
            for (long address : allocated) {
                try {
                    gpu.free(address);
                } catch (RuntimeException | Error cleanup) {
                    failure.addSuppressed(cleanup);
                }
            }
            throw failure;
        }
    }

    private static long allocate(GpuMemory gpu, long bytes, List<Long> allocated) {
        long address = gpu.allocate(bytes);
        allocated.add(address);
        return address;
    }

    private static int indexOf(int[] ends, long value) {
        for (int index = 0; index < ends.length; index++) if (ends[index] == value) return index;
        return -1;
    }

    // execution: the engine wiring follows the storage half

    @Override
    public ExecutionPlan.Kind kind(ExecutionPlan.Kind role) {
        throw new UnsupportedOperationException("trellis execution is not wired yet");
    }

    @Override
    public TensorHandle validateHandle(TensorHandle handle, long rows, int width, WeightFormat roleFormat) {
        throw new UnsupportedOperationException("trellis execution is not wired yet");
    }

    @Override
    public boolean rowExact(ExecutionGpu gpu, ExecutionPlan.Kind kind) {
        throw new UnsupportedOperationException("trellis execution is not wired yet");
    }

    @Override
    public ScratchUse scratchUse(ExecutionPlan.Kind kind, boolean multiRow) {
        throw new UnsupportedOperationException("trellis execution is not wired yet");
    }

    @Override
    public long scratchBytes(
            ExecutionGpu gpu, ScratchUse use, int rows, int inFeatures, int outFeatures, List<TensorHandle> weights) {
        throw new UnsupportedOperationException("trellis execution is not wired yet");
    }

    @Override
    public void linear(ExecutionGpu gpu, LinearLaunch launch) {
        throw new UnsupportedOperationException("trellis execution is not wired yet");
    }
}
