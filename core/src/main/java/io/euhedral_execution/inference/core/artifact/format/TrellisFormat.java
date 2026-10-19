package io.euhedral_execution.inference.core.artifact.format;

import io.euhedral_execution.inference.core.artifact.TensorDataType;
import io.euhedral_execution.inference.core.artifact.TensorDescriptor;
import io.euhedral_execution.inference.core.artifact.TensorHandle;
import io.euhedral_execution.inference.core.artifact.WeightFormat;
import io.euhedral_execution.inference.core.artifact.WeightLayout;
import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.gpu.GpuMemory;
import io.euhedral_execution.inference.core.gpu.KernelArguments;
import io.euhedral_execution.inference.core.gpu.ScratchUse;
import io.euhedral_execution.inference.core.model.qwen38.ArtifactProfile;
import io.euhedral_execution.inference.core.model.qwen38.ExecutionPlan;
import java.lang.foreign.Arena;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

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

    private static final ThreadLocal<KernelArguments> ARGUMENTS = ThreadLocal.withInitial(KernelArguments::new);

    /// The most rows the launch grids' Y extent holds.
    private static final int MAX_GRID_ROWS = 65535;

    /// Columns per block of [LinearKernel#TRELLIS_PREPASS]: one 128-column block per warp, four warps.
    private static final int PREPASS_BLOCK_COLUMNS = 512;

    /// Columns per block of [LinearKernel#TRELLIS_EPILOGUE]: one 128-column block per warp, four warps.
    private static final int EPILOGUE_BLOCK_COLUMNS = 512;

    /// Rows per block of [LinearKernel#TRELLIS_MMA_ROWS] (`kSplitRows`, native/src/trellis/mma.cuh).
    private static final int WHOLE_K_ROW_BLOCK = 32;

    /// Rows per block of [LinearKernel#TRELLIS_MMA_ROWS64] (`kSplitRows64`, native/src/trellis/mma.cuh).
    private static final int WIDE_WHOLE_K_ROW_BLOCK = 64;

    /// Layouts of [LinearKernel#TRELLIS_PREPASS]'s output: FP16 in the tensor-core GEMM's fragment order, BF16 in
    /// column order (the prefill route), block-scaled E4M3 for the FP8 GEMM (`s * rows * (k + k / 32)` bytes,
    /// within the FP16 activations' region).
    private static final int FRAGMENT_FP16 = 0;

    private static final int COLUMN_BF16 = 1;
    private static final int BLOCK_FP8 = 2;

    private final TrellisSettings settings;

    TrellisFormat() {
        this(TrellisSettings.fromProperties());
    }

    public TrellisFormat(TrellisSettings settings) {
        this.settings = Objects.requireNonNull(settings, "settings");
    }

    public TrellisSettings settings() {
        return this.settings;
    }

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

    /// The scratch bytes an object of `descriptor` needs at `maxRows` rows when its segment count is unknown: the
    /// tensor-core route with at most [TrellisLayout#MAX_SEGMENTS] segments, and, when `maxRows` exceeds
    /// [TrellisSettings#mmaMaxRows], the larger prefill route (the rotated activations, the whole object decoded to
    /// BF16 `[n, k]`, and the GEMM outputs `[rows, n]`, each region 256-aligned).
    @Override
    public long planScratchBytes(TensorDescriptor descriptor, int maxRows) {
        long[] nk = logical(descriptor.shape());
        long bytes = mmaBytes(maxRows, nk[1], nk[0], SCRATCH_SEGMENTS);
        if (maxRows > this.settings.mmaMaxRows())
            bytes = Math.max(bytes, prefillBytes(maxRows, nk[1], nk[0], SCRATCH_SEGMENTS));
        return bytes;
    }

    /// Trellis rows need the object's host-known segment ends; see the overload with `ends`.
    @Override
    public List<TensorHandle> rows(List<TensorHandle> object, long first, long rows, String name, GpuMemory gpu) {
        throw new IllegalArgumentException("cutting trellis rows needs the object's segment ends");
    }

    /// The segment ends of the rows `[first, first + rows)` of an object of `ends`, rebased to the cut. Both ends of
    /// the range must be segment boundaries (or 0 and n).
    public static int[] cutEnds(int[] ends, long first, long rows, String name) {
        long last = first + rows;
        int start = first == 0 ? -1 : indexOf(ends, first), stop = indexOf(ends, last);
        if ((first != 0 && start < 0) || stop < 0 || stop <= start)
            throw new IllegalArgumentException(
                    "rows " + first + ".." + last + " of " + name + " do not begin and end on segment boundaries");
        int[] cut = new int[stop - start];
        for (int index = 0; index < cut.length; index++) cut[index] = (int) (ends[start + 1 + index] - first);
        return cut;
    }

    /// [object, suh, svh, segments] of rows `[first, first + rows)` as new device allocations named `name`, `name.suh`,
    /// `name.svh` and `name.segments`. `ends` are the object's segment ends, known on the host from the artifact;
    /// both ends of the range must be segment boundaries (or 0 and n). The payload's tile columns are copied per
    /// k-tile row; `svh`, the `suh` rows and the segment ends are cut and rebased.
    @Override
    public List<TensorHandle> rows(
            List<TensorHandle> object, long first, long rows, String name, GpuMemory gpu, int[] ends) {
        if (object.size() != 4) throw new IllegalArgumentException("a trellis object has 4 tensors: " + object.size());
        TensorHandle payload = object.get(0), suh = object.get(1), svh = object.get(2), segments = object.get(3);
        long[] nk = logical(payload.shape());
        long n = nk[0], k = nk[1];
        int w = (int) payload.shape()[2];
        long last = first + rows;
        if (first < 0 || rows <= 0 || last > n || first % 16 != 0 || rows % 16 != 0)
            throw new IllegalArgumentException("rows " + first + ".." + last + " are outside " + name);
        int[] cut = cutEnds(ends, first, rows, name);
        int count = cut.length;
        int lo = first == 0 ? 0 : indexOf(ends, first) + 1;
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

    // execution

    /// The K splits of the tensor-core route for an object of `k` inputs and `n` outputs: enough to fill `targetCtas`
    /// blocks, at most one per 256 inputs. It depends on the shape and the target alone, never on the rows.
    public static int splits(int k, int n, int targetCtas) {
        int blocks = n / TrellisLayout.BLOCK;
        int wanted = (targetCtas + blocks - 1) / blocks;
        return Math.max(1, Math.min(wanted, k / 256));
    }

    @Override
    public ExecutionPlan.Kind kind(ExecutionPlan.Kind role) {
        return ExecutionPlan.Kind.TRELLIS_LINEAR;
    }

    @Override
    public TensorHandle validateHandle(TensorHandle handle, long rows, int width, WeightFormat roleFormat) {
        Objects.requireNonNull(handle, "weight");
        long[] nk;
        try {
            nk = logical(handle.shape());
        } catch (IllegalArgumentException refused) {
            throw new IllegalArgumentException("unsupported trellis layout or dimensions: " + handle.name(), refused);
        }
        if (nk[0] != rows
                || nk[1] != width
                || handle.deviceAddress() == 0 && !handle.hostBacked() && !handle.diskBacked()
                || handle.dataType() != TensorDataType.BF16
                || handle.layout() != WeightLayout.TRELLIS_TILE16_V1
                || handle.byteSize() != byteSize(handle.shape(), handle.dataType()))
            throw new IllegalArgumentException("unsupported trellis layout or dimensions: " + handle.name());
        return handle;
    }

    /// A multi-row launch computes every row as a one-row launch does: the tensor-core route's results do not depend
    /// on the row count, and a verification never leaves it.
    @Override
    public boolean rowExact(ExecutionGpu gpu, ExecutionPlan.Kind kind) {
        return true;
    }

    @Override
    public ScratchUse scratchUse(ExecutionPlan.Kind kind, boolean multiRow) {
        return ScratchUse.TRELLIS_LINEAR;
    }

    /// The scratch of a launch of up to `rows` rows over an object with `weights` ([object, suh, svh, segments]): the
    /// larger of the tensor-core route at `rows` (a verification may reach every row) and, when `rows` exceeds the
    /// tensor-core row limit, the prefill route.
    @Override
    public long scratchBytes(
            ExecutionGpu gpu, ScratchUse use, int rows, int inFeatures, int outFeatures, List<TensorHandle> weights) {
        int segments = Math.toIntExact(weights.get(1).shape()[0]);
        long bytes = mmaBytes(rows, inFeatures, outFeatures, segments);
        if (rows > this.settings.mmaMaxRows())
            bytes = Math.max(bytes, prefillBytes(rows, inFeatures, outFeatures, segments));
        return bytes;
    }

    /// The tensor-core route's scratch alone: decode-type views never take the prefill route.
    @Override
    public long decodeScratchBytes(
            ExecutionGpu gpu, ScratchUse use, int rows, int inFeatures, int outFeatures, List<TensorHandle> weights) {
        return mmaBytes(
                rows, inFeatures, outFeatures, Math.toIntExact(weights.get(1).shape()[0]));
    }

    /// The tensor-core route's scratch for launches of up to `rows` rows: the activations, and the split partials of
    /// the most rows below [TrellisSettings#wholeKRows].
    private long mmaBytes(long rows, long k, long n, int segments) {
        long activations = PayloadSizes.alignUp(
                PayloadSizes.multiply(PayloadSizes.multiply(segments, rows), k * 2), SCRATCH_ALIGNMENT);
        int splits = splits(Math.toIntExact(k), Math.toIntExact(n), this.settings.targetCtas());
        long splitRows = Math.min(rows, this.settings.wholeKRows() - 1L);
        if (splits == 1 || splitRows == 0) return activations;
        return PayloadSizes.add(activations, PayloadSizes.multiply(PayloadSizes.multiply(splits, splitRows), n * 4));
    }

    private static long prefillBytes(long rows, long k, long n, int segments) {
        return PayloadSizes.add(
                PayloadSizes.add(
                        PayloadSizes.alignUp(
                                PayloadSizes.multiply(PayloadSizes.multiply(segments, rows), k * 2), SCRATCH_ALIGNMENT),
                        PayloadSizes.alignUp(PayloadSizes.multiply(n, k * 2), SCRATCH_ALIGNMENT)),
                PayloadSizes.alignUp(PayloadSizes.multiply(rows, n * 2), SCRATCH_ALIGNMENT));
    }

    /// Queues the launches of one trellis linear in stream order. Exact numerics run the scalar reference alone. A
    /// verification, or at most [TrellisSettings#mmaMaxRows] rows, runs prepass, tensor-core GEMM and (with a K
    /// split) the epilogue; more rows decode the object to BF16 in rotated space and multiply it per segment on the
    /// row-exact BF16 GEMM.
    @Override
    public void linear(ExecutionGpu gpu, LinearLaunch launch) {
        List<TensorHandle> weights = launch.weights();
        if (weights.size() != 4)
            throw new IllegalArgumentException(
                    "a trellis linear takes [object, suh, svh, segments]: " + weights.size() + " tensors");
        TensorHandle payload = weights.get(0), suh = weights.get(1), svh = weights.get(2), ends = weights.get(3);
        int rows = launch.rows(), k = launch.inFeatures(), n = launch.outFeatures();
        long[] nk = logical(payload.shape());
        if (nk[0] != n || nk[1] != k)
            throw new IllegalArgumentException("trellis object " + payload.name() + " is not " + n + " x " + k);
        if (rows <= 0 || rows > MAX_GRID_ROWS)
            throw new IllegalArgumentException("a trellis launch takes 1.." + MAX_GRID_ROWS + " rows: " + rows);
        int w = (int) payload.shape()[2];
        int segments = Math.toIntExact(suh.shape()[0]);
        Launch run = new Launch(
                gpu, launch, payload.deviceAddress(), suh.deviceAddress(), svh.deviceAddress(), ends.deviceAddress());
        run.rows = rows;
        run.k = k;
        run.n = n;
        run.w = w;
        run.segments = segments;
        if (gpu.exactNumerics()) run.reference();
        else if (launch.rowExact() || rows <= this.settings.mmaMaxRows()) run.tensorCore(this.settings);
        else run.prefill(launch.segmentEnds());
    }

    /// One launch's operands and its route chains.
    private static final class Launch {
        final ExecutionGpu gpu;
        final LinearLaunch launch;
        final long words, suh, svh, ends;
        int rows, k, n, w, segments;

        Launch(ExecutionGpu gpu, LinearLaunch launch, long words, long suh, long svh, long ends) {
            this.gpu = gpu;
            this.launch = launch;
            this.words = words;
            this.suh = suh;
            this.svh = svh;
            this.ends = ends;
        }

        KernelArguments arguments() {
            return ARGUMENTS.get().clear();
        }

        void reference() {
            this.gpu.launchTableKernel(
                    LinearKernel.TRELLIS_REFERENCE,
                    Math.multiplyExact(this.rows, this.n),
                    1,
                    1,
                    128,
                    1,
                    1,
                    Math.multiplyExact(4, this.k),
                    arguments()
                            .pointer(this.launch.input())
                            .pointer(this.words)
                            .pointer(this.suh)
                            .pointer(this.svh)
                            .pointer(this.ends)
                            .int32(this.segments)
                            .pointer(this.launch.output())
                            .int32(this.rows)
                            .int32(this.k)
                            .int32(this.n)
                            .int32(this.w));
        }

        /// `layout`: [#FRAGMENT_FP16], [#COLUMN_BF16] or [#BLOCK_FP8].
        void prepass(long activations, int layout) {
            this.gpu.launchTableKernel(
                    LinearKernel.TRELLIS_PREPASS,
                    Math.ceilDiv(this.k, PREPASS_BLOCK_COLUMNS),
                    this.rows,
                    this.segments,
                    128,
                    1,
                    1,
                    0,
                    arguments()
                            .pointer(this.launch.input())
                            .pointer(this.suh)
                            .pointer(activations)
                            .int32(this.rows)
                            .int32(this.k)
                            .int32(this.segments)
                            .int32(layout));
        }

        void tensorCore(TrellisSettings settings) {
            int splits = splits(this.k, this.n, settings.targetCtas());
            long activations = PayloadSizes.alignUp(
                    PayloadSizes.multiply(PayloadSizes.multiply(this.segments, this.rows), this.k * 2L),
                    SCRATCH_ALIGNMENT);
            if (this.rows >= settings.wholeKRows()) {
                requireScratch(activations);
                long scratch = this.launch.scratch();
                boolean wide = this.rows >= settings.wholeKWideRows();
                // A verification's rows must equal one-row launches bit for bit: FP16 operands and FP32 sums. Other
                // launches take block-scaled FP8 operands where the device has them.
                LinearKernel fp8 = wide ? LinearKernel.TRELLIS_MMA_ROWS64_FP8 : LinearKernel.TRELLIS_MMA_ROWS_FP8;
                boolean blockFp8 = !this.launch.rowExact() && this.gpu.tableKernelAvailable(fp8);
                prepass(scratch, blockFp8 ? BLOCK_FP8 : FRAGMENT_FP16);
                LinearKernel kernel =
                        blockFp8 ? fp8 : wide ? LinearKernel.TRELLIS_MMA_ROWS64 : LinearKernel.TRELLIS_MMA_ROWS;
                this.gpu.launchTableKernel(
                        kernel,
                        this.n / TrellisLayout.BLOCK,
                        Math.ceilDiv(this.rows, wide ? WIDE_WHOLE_K_ROW_BLOCK : WHOLE_K_ROW_BLOCK),
                        1,
                        256,
                        1,
                        1,
                        0,
                        arguments()
                                .pointer(scratch)
                                .pointer(this.words)
                                .pointer(this.ends)
                                .pointer(this.svh)
                                .pointer(this.launch.output())
                                .int32(this.rows)
                                .int32(this.k)
                                .int32(this.n)
                                .int32(this.w)
                                .int32(splits)
                                .int32(this.segments));
                return;
            }
            long partials =
                    splits == 1 ? 0 : PayloadSizes.multiply(PayloadSizes.multiply(splits, this.rows), this.n * 4L);
            requireScratch(PayloadSizes.add(activations, partials));
            long scratch = this.launch.scratch();
            prepass(scratch, FRAGMENT_FP16);
            this.gpu.launchTableKernel(
                    LinearKernel.TRELLIS_MMA,
                    this.n / TrellisLayout.BLOCK,
                    splits,
                    1,
                    256,
                    1,
                    1,
                    0,
                    arguments()
                            .pointer(scratch)
                            .pointer(this.words)
                            .pointer(this.ends)
                            .pointer(this.svh)
                            .pointer(scratch + activations)
                            .pointer(this.launch.output())
                            .int32(this.rows)
                            .int32(this.k)
                            .int32(this.n)
                            .int32(this.w)
                            .int32(splits)
                            .int32(this.segments));
            if (splits == 1) return;
            this.gpu.launchTableKernel(
                    LinearKernel.TRELLIS_EPILOGUE,
                    Math.ceilDiv(this.n, EPILOGUE_BLOCK_COLUMNS),
                    this.rows,
                    1,
                    128,
                    1,
                    1,
                    0,
                    arguments()
                            .pointer(scratch + activations)
                            .int32(splits)
                            .pointer(this.svh)
                            .pointer(this.launch.output())
                            .int32(this.rows)
                            .int32(this.n));
        }

        void prefill(int[] bounds) {
            if (bounds == null || bounds.length != this.segments)
                throw new IllegalArgumentException(
                        "a trellis prefill launch needs the host segment ends of its object");
            TrellisLayout.validateSegments(bounds, this.n, this.segments);
            long activations = PayloadSizes.alignUp(
                    PayloadSizes.multiply(PayloadSizes.multiply(this.segments, this.rows), this.k * 2L),
                    SCRATCH_ALIGNMENT);
            long decodedBytes = PayloadSizes.alignUp(PayloadSizes.multiply(this.n, this.k * 2L), SCRATCH_ALIGNMENT);
            long rotatedBytes = PayloadSizes.alignUp(PayloadSizes.multiply(this.rows, this.n * 2L), SCRATCH_ALIGNMENT);
            requireScratch(PayloadSizes.add(PayloadSizes.add(activations, decodedBytes), rotatedBytes));
            long scratch = this.launch.scratch(), decoded = scratch + activations, rotated = decoded + decodedBytes;
            prepass(scratch, COLUMN_BF16);
            this.gpu.launchTableKernel(
                    LinearKernel.TRELLIS_DECODE_BF16,
                    (this.k / 16) * (this.n / 16),
                    1,
                    1,
                    256,
                    1,
                    1,
                    0,
                    arguments()
                            .pointer(this.words)
                            .pointer(decoded)
                            .int32(this.k)
                            .int32(this.n)
                            .int32(this.w));
            int start = 0;
            for (int segment = 0; segment < bounds.length; segment++) {
                int width = bounds[segment] - start;
                RowExactBf16.linear(
                        this.gpu,
                        scratch + (long) segment * this.rows * this.k * 2,
                        decoded + (long) start * this.k * 2,
                        rotated + (long) this.rows * start * 2,
                        this.rows,
                        this.k,
                        width);
                start = bounds[segment];
            }
            this.gpu.launchTableKernel(
                    LinearKernel.TRELLIS_EPILOGUE_BF16,
                    this.n / TrellisLayout.BLOCK,
                    this.rows,
                    1,
                    128,
                    1,
                    1,
                    0,
                    arguments()
                            .pointer(rotated)
                            .pointer(this.ends)
                            .int32(this.segments)
                            .pointer(this.svh)
                            .pointer(this.launch.output())
                            .int32(this.rows)
                            .int32(this.n));
        }

        void requireScratch(long needed) {
            if (this.launch.scratch() == 0 || this.launch.scratchBytes() < needed)
                throw new IllegalStateException("a trellis linear of " + this.rows + " rows needs " + needed
                        + " bytes of scratch; the bound region holds " + this.launch.scratchBytes());
        }
    }
}
