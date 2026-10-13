package io.euhedral_execution.inference.quantization.gptq;

import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import io.euhedral_execution.inference.quantization.gpu.Cuda;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Comparator;
import java.util.List;
import java.util.stream.IntStream;

/// GPTQ for the artifact formats on the GPU (tools/gptq/gptq.py): each object's columns are rounded in order (or in
/// act-order) and each column's error carried into the columns not yet rounded through the upper Cholesky factor of
/// (H + damp * mean(diag H) I)^-1. Group scales are chosen by the converter's weighted searches on the error-updated
/// weights when a group's first column comes up; runs of columns between two choices round in one kernel launch.
/// Every call runs on `stream`.
public final class Gptq implements AutoCloseable {

    public static final int BLOCK = 128;
    private static final int SEARCH_BELOW = 6;
    private static final float E2M1_MAX_TIMES_E4M3_MAX = 2688.0f;

    /// A storage format: grouped (Q3/Q4/Q5 G64, codes qmin..qmax) or NVFP4 (block 16, optionally SD4).
    public record Format(String name, String layout, int group, int qmin, int qmax, boolean nvfp4, boolean sd4) {

        public static Format of(String name, String layout, Integer lowestCode) {
            boolean sd4 = layout.endsWith("sd4-v1");
            return switch (name) {
                case "NVFP4" -> new Format(name, layout, 16, 0, 0, true, sd4);
                case "Q3G64_F16S" -> grouped(name, layout, -4, 3, lowestCode);
                case "Q4G64_F16S" -> grouped(name, layout, -8, 7, lowestCode);
                case "Q5G64_F16S" -> grouped(name, layout, -16, 15, lowestCode);
                default -> throw new IllegalArgumentException("unknown format " + name);
            };
        }

        private static Format grouped(String name, String layout, int qmin, int qmax, Integer lowestCode) {
            return new Format(
                    name, layout, 64, lowestCode == null ? qmin : Math.max(qmin, lowestCode), qmax, false, false);
        }
    }

    /// An object's prepared Hessian: the column order, each column's position in it, the importance (diag H, in
    /// weight order), and the upper factor of the permuted, damped inverse (column-major K x K on the device).
    public record Factor(int k, int[] order, int[] position, float[] importance, long upper) {}

    /// An NVFP4 object's tensor-wide scales: FP32 global scale and, for SD4, the 16-entry table (else null).
    public record TensorScales(float globalScale, byte[] table) {}

    private final Cuda cuda;
    private final CudaGpuMemory gpu;
    private final long stream;
    private final long cublas, cusolver;
    private final Cuda.Function groupedChoose,
            nvfp4Choose,
            sd4Costs,
            segment,
            gatherGroup,
            bf16ToFloatPermuted,
            floatToBf16,
            permuteSymmetric,
            readDiagonal,
            writeDiagonal,
            mirrorLower,
            amaxBf16,
            sliceColumns,
            zeroColumns,
            fillBytes;

    public Gptq(Cuda cuda, CudaGpuMemory gpu, long stream) {
        this.cuda = cuda;
        this.gpu = gpu;
        this.stream = stream;
        List<Cuda.Function> f = cuda.compileResource(
                "gptq.cu",
                List.of(
                        "grouped_choose",
                        "nvfp4_choose",
                        "sd4_costs",
                        "segment",
                        "gather_group",
                        "bf16_to_float_permuted",
                        "float_to_bf16",
                        "permute_symmetric",
                        "read_diagonal",
                        "write_diagonal",
                        "mirror_lower",
                        "amax_bf16",
                        "slice_columns",
                        "zero_columns",
                        "fill_bytes"));
        groupedChoose = f.get(0);
        nvfp4Choose = f.get(1);
        sd4Costs = f.get(2);
        segment = f.get(3);
        gatherGroup = f.get(4);
        bf16ToFloatPermuted = f.get(5);
        floatToBf16 = f.get(6);
        permuteSymmetric = f.get(7);
        readDiagonal = f.get(8);
        writeDiagonal = f.get(9);
        mirrorLower = f.get(10);
        amaxBf16 = f.get(11);
        sliceColumns = f.get(12);
        zeroColumns = f.get(13);
        fillBytes = f.get(14);
        cublas = cuda.cublasHandle(stream);
        cusolver = cuda.cusolverHandle(stream);
    }

    private static int blocks(long count, int threads) {
        return (int) ((count + threads - 1) / threads);
    }

    private void zero(long address, long bytes) {
        cuda.launch(
                fillBytes,
                blocks(bytes, 256),
                1,
                256,
                0,
                stream,
                new Cuda.Arguments().pointer(address).pointer(bytes).integer(0));
    }

    /// Prepares the factor of `hessian` (FP32 [k, k] on the device, left unchanged): act-order (columns by decreasing
    /// diag H, stable) or the natural order; dead columns (diag 0) get a unit diagonal, the diagonal gains
    /// damp * mean(diag); then potrf (lower), potri (lower), mirror, potrf (upper).
    public Factor factor(long hessian, int k, float damp, boolean actOrder) {
        float[] importance = new float[k];
        try (Arena arena = Arena.ofConfined()) {
            long diagonal = gpu.allocate(4L * k);
            try {
                cuda.launch(
                        readDiagonal,
                        blocks(k, 256),
                        1,
                        256,
                        0,
                        stream,
                        new Cuda.Arguments().pointer(hessian).integer(k).pointer(diagonal));
                cuda.synchronize(stream);
                MemorySegment host = arena.allocate(4L * k);
                gpu.copyDeviceToHost(host, diagonal, 4L * k);
                MemorySegment.copy(host, ValueLayout.JAVA_FLOAT, 0, importance, 0, k);
            } finally {
                gpu.free(diagonal);
            }
        }
        int[] order = actOrder
                ? IntStream.range(0, k)
                        .boxed()
                        .sorted(Comparator.comparingDouble((Integer i) -> -importance[i]))
                        .mapToInt(Integer::intValue)
                        .toArray()
                : IntStream.range(0, k).toArray();
        int[] position = new int[k];
        for (int i = 0; i < k; i++) position[order[i]] = i;
        long bytes = 4L * k * k;
        long upper = gpu.allocate(bytes);
        long orderDevice = upload(order);
        cuda.launch(
                permuteSymmetric,
                blocks((long) k * k, 256),
                1,
                256,
                0,
                stream,
                new Cuda.Arguments()
                        .pointer(hessian)
                        .pointer(orderDevice)
                        .integer(k)
                        .pointer(upper));
        float[] diagonal = new float[k];
        double sum = 0;
        for (int i = 0; i < k; i++) {
            diagonal[i] = importance[order[i]] == 0.0f ? 1.0f : importance[order[i]];
        }
        // torch.mean over FP32 (pairwise in PyTorch; here in float64, then rounded): the damping only scales the
        // diagonal, so a rounding difference in it is far below the factor's own error.
        for (float d : diagonal) sum += d;
        float add = (float) (damp * (float) (sum / k));
        for (int i = 0; i < k; i++) diagonal[i] = diagonal[i] + add;
        long diagonalDevice = upload(diagonal);
        cuda.launch(
                writeDiagonal,
                blocks(k, 256),
                1,
                256,
                0,
                stream,
                new Cuda.Arguments().pointer(upper).integer(k).pointer(diagonalDevice));
        long info = gpu.allocate(4);
        int work = Math.max(
                Math.max(
                        cuda.solverWorkspace(cusolver, false, Cuda.FILL_LOWER, k, upper),
                        cuda.solverWorkspace(cusolver, true, Cuda.FILL_LOWER, k, upper)),
                cuda.solverWorkspace(cusolver, false, Cuda.FILL_UPPER, k, upper));
        long workspace = gpu.allocate(4L * Math.max(1, work));
        try {
            cuda.cholesky(cusolver, false, Cuda.FILL_LOWER, k, upper, workspace, work, info);
            checkInfo(info, "potrf (H)");
            cuda.cholesky(cusolver, true, Cuda.FILL_LOWER, k, upper, workspace, work, info);
            checkInfo(info, "potri");
            cuda.launch(
                    mirrorLower,
                    blocks((long) k * k, 256),
                    1,
                    256,
                    0,
                    stream,
                    new Cuda.Arguments().pointer(upper).integer(k));
            cuda.cholesky(cusolver, false, Cuda.FILL_UPPER, k, upper, workspace, work, info);
            checkInfo(info, "potrf (H^-1)");
        } finally {
            gpu.free(workspace);
            gpu.free(info);
            gpu.free(orderDevice);
            gpu.free(diagonalDevice);
        }
        return new Factor(k, order, position, importance, upper);
    }

    private void checkInfo(long info, String step) {
        cuda.synchronize(stream);
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment host = arena.allocate(4);
            gpu.copyDeviceToHost(host, info, 4);
            int value = host.get(ValueLayout.JAVA_INT, 0);
            if (value != 0) throw new IllegalStateException(step + " failed: cuSOLVER info " + value);
        }
    }

    /// The tensor-wide NVFP4 scales of a whole object (BF16 [rows, k] on the device): the global scale from its
    /// largest magnitude, and for SD4 the table chosen under diag H (nvfp4_tensor_scales).
    public TensorScales tensorScales(long weightBf16, long rows, int k, Format format, Factor factor) {
        long amax = gpu.allocate(4);
        zero(amax, 4);
        cuda.launch(
                amaxBf16,
                1024,
                1,
                256,
                0,
                stream,
                new Cuda.Arguments().pointer(weightBf16).pointer(rows * k).pointer(amax));
        cuda.synchronize(stream);
        float largest;
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment host = arena.allocate(4);
            gpu.copyDeviceToHost(host, amax, 4);
            largest = host.get(ValueLayout.JAVA_FLOAT, 0);
        } finally {
            gpu.free(amax);
        }
        float global = largest / E2M1_MAX_TIMES_E4M3_MAX;
        if (!format.sd4()) return new TensorScales(global, null);
        if (global <= 0) {
            byte[] identity = new byte[Sd4Table.ENTRIES];
            for (int i = 0; i < identity.length; i++) identity[i] = (byte) i;
            return new TensorScales(global, identity);
        }
        int codes = Sd4Table.CODES;
        long costs = gpu.allocate(8L * codes * codes),
                energy = gpu.allocate(8L * codes),
                counts = gpu.allocate(8L * codes);
        long weights = upload(factor.importance());
        long chunkRows = Math.max(1, (1L << 28) / (4L * k)); // FP32 rows per pass (256 MiB)
        long values = gpu.allocate(4L * chunkRows * k);
        long identityOrder = upload(IntStream.range(0, k).toArray());
        try {
            zero(costs, 8L * codes * codes);
            zero(energy, 8L * codes);
            zero(counts, 8L * codes);
            for (long begin = 0; begin < rows; begin += chunkRows) {
                long count = Math.min(chunkRows, rows - begin);
                cuda.launch(
                        bf16ToFloatPermuted,
                        blocks(count * k, 256),
                        1,
                        256,
                        0,
                        stream,
                        new Cuda.Arguments()
                                .pointer(weightBf16 + 2L * begin * k)
                                .pointer(count)
                                .pointer((long) k)
                                .pointer(identityOrder)
                                .pointer(values));
                long blockCount = count * k / 16;
                cuda.launch(
                        sd4Costs,
                        blocks(blockCount, 128),
                        1,
                        128,
                        0,
                        stream,
                        new Cuda.Arguments()
                                .pointer(values)
                                .pointer(blockCount)
                                .integer(k)
                                .real(global)
                                .pointer(weights)
                                .integer(SEARCH_BELOW)
                                .pointer(costs)
                                .pointer(energy)
                                .pointer(counts));
            }
            cuda.synchronize(stream);
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment hostCosts = arena.allocate(8L * codes * codes),
                        hostEnergy = arena.allocate(8L * codes),
                        hostCounts = arena.allocate(8L * codes);
                gpu.copyDeviceToHost(hostCosts, costs, hostCosts.byteSize());
                gpu.copyDeviceToHost(hostEnergy, energy, hostEnergy.byteSize());
                gpu.copyDeviceToHost(hostCounts, counts, hostCounts.byteSize());
                double[][] c = new double[codes][codes];
                for (int o = 0; o < codes; o++) {
                    MemorySegment.copy(hostCosts, ValueLayout.JAVA_DOUBLE, 8L * o * codes, c[o], 0, codes);
                }
                return new TensorScales(
                        global,
                        Sd4Table.choose(
                                c,
                                hostEnergy.toArray(ValueLayout.JAVA_DOUBLE),
                                hostCounts.toArray(ValueLayout.JAVA_LONG),
                                SEARCH_BELOW));
            }
        } finally {
            gpu.free(costs);
            gpu.free(energy);
            gpu.free(counts);
            gpu.free(weights);
            gpu.free(values);
            gpu.free(identityOrder);
        }
    }

    /// Rounds rows [0, n) of an object (BF16 [n, k] on the device at `weightBf16`) under `factor`; writes the decoded
    /// weights as BF16 [n, k] to `decodedBf16` unless it is 0, and returns the state (host memory from `arena`).
    public QuantState round(
            long weightBf16, int n, Factor factor, Format format, TensorScales tensor, Arena arena, long decodedBf16) {
        int k = factor.k();
        int group = format.group();
        int groups = k / group;
        long w = gpu.allocateAsync(4L * n * k);
        long decoded = gpu.allocateAsync(4L * n * k);
        long codes = gpu.allocateAsync(2L * n * k);
        long decodes = gpu.allocateAsync(4L * n * groups);
        long scales = gpu.allocateAsync((format.nvfp4() ? 1L : 2L) * n * groups);
        long block = gpu.allocateAsync(4L * n * BLOCK);
        long errors = gpu.allocateAsync(4L * n * BLOCK);
        long values = gpu.allocateAsync(4L * n * group);
        long orderDevice = uploadAsync(factor.order());
        long positionDevice = uploadAsync(factor.position());
        int[] groupOf = new int[k];
        for (int j = 0; j < k; j++) groupOf[j] = factor.order()[j] / group;
        long groupDevice = uploadAsync(groupOf);
        long tableDevice = format.sd4() ? uploadAsync(tensor.table()) : 0L;
        long weightsDevice = uploadAsync(factor.importance());
        try {
            cuda.launch(
                    bf16ToFloatPermuted,
                    blocks((long) n * k, 256),
                    1,
                    256,
                    0,
                    stream,
                    new Cuda.Arguments()
                            .pointer(weightBf16)
                            .pointer(n)
                            .pointer((long) k)
                            .pointer(orderDevice)
                            .pointer(w));
            int[] dead = IntStream.range(0, k)
                    .filter(j -> factor.importance()[factor.order()[j]] == 0.0f)
                    .toArray();
            if (dead.length > 0) {
                long deadDevice = uploadAsync(dead);
                cuda.launch(
                        zeroColumns,
                        blocks((long) n * dead.length, 256),
                        1,
                        256,
                        0,
                        stream,
                        new Cuda.Arguments()
                                .pointer(w)
                                .pointer((long) k)
                                .pointer(n)
                                .pointer(deadDevice)
                                .integer(dead.length));
                cuda.synchronize(stream);
                gpu.freeAsync(deadDevice);
            }
            zero(decodes, 4L * n * groups);
            boolean[] chosen = new boolean[groups];
            int segmentShared = 32 * (BLOCK + 1) * 4;
            for (int begin = 0; begin < k; begin += BLOCK) {
                int end = Math.min(k, begin + BLOCK), width = end - begin;
                cuda.launch(
                        sliceColumns,
                        blocks((long) n * width, 256),
                        1,
                        256,
                        0,
                        stream,
                        new Cuda.Arguments()
                                .pointer(w)
                                .pointer((long) k)
                                .integer(begin)
                                .integer(width)
                                .pointer(n)
                                .pointer(block));
                zero(errors, 4L * n * width);
                int first = 0;
                for (int i = 0; i < width; i++) {
                    int g = factor.order()[begin + i] / group;
                    if (chosen[g]) continue;
                    if (i > first)
                        launchSegment(
                                block,
                                errors,
                                factor,
                                decodes,
                                groupDevice,
                                orderDevice,
                                codes,
                                decoded,
                                n,
                                width,
                                k,
                                begin,
                                first,
                                i,
                                groups,
                                format,
                                segmentShared);
                    first = i;
                    choose(
                            g,
                            w,
                            block,
                            k,
                            width,
                            begin,
                            end,
                            positionDevice,
                            n,
                            group,
                            values,
                            weightsDevice,
                            format,
                            tensor,
                            tableDevice,
                            scales,
                            decodes,
                            groups);
                    chosen[g] = true;
                }
                launchSegment(
                        block,
                        errors,
                        factor,
                        decodes,
                        groupDevice,
                        orderDevice,
                        codes,
                        decoded,
                        n,
                        width,
                        k,
                        begin,
                        first,
                        width,
                        groups,
                        format,
                        segmentShared);
                if (end < k) {
                    // w[:, end:] -= errors @ U[begin:end, end:] (row-major), i.e. in column-major terms
                    // W^T[end:, :] -= U[begin:end, end:]^T E^T, with U column-major.
                    cuda.gemmFp32(
                            cublas,
                            Cuda.CUBLAS_OP_T,
                            Cuda.CUBLAS_OP_N,
                            k - end,
                            n,
                            width,
                            -1.0f,
                            factor.upper() + 4L * ((long) end * k + begin),
                            k,
                            errors,
                            width,
                            1.0f,
                            w + 4L * end,
                            k);
                }
            }
            if (decodedBf16 != 0) {
                cuda.launch(
                        floatToBf16,
                        blocks((long) n * k, 256),
                        1,
                        256,
                        0,
                        stream,
                        new Cuda.Arguments()
                                .pointer(decoded)
                                .pointer((long) n * k)
                                .pointer(decodedBf16));
            }
            cuda.synchronize(stream);
            return state(n, k, groups, format, tensor, codes, scales, arena);
        } finally {
            for (long address : new long[] {
                w,
                decoded,
                codes,
                decodes,
                scales,
                block,
                errors,
                values,
                orderDevice,
                positionDevice,
                groupDevice,
                weightsDevice
            }) gpu.freeAsync(address);
            if (tableDevice != 0) gpu.freeAsync(tableDevice);
        }
    }

    private void launchSegment(
            long block,
            long errors,
            Factor factor,
            long decodes,
            long groupDevice,
            long orderDevice,
            long codes,
            long decoded,
            int n,
            int width,
            int k,
            int begin,
            int first,
            int last,
            int groups,
            Format format,
            int shared) {
        cuda.launch(
                segment,
                blocks(n, 32),
                1,
                128,
                shared,
                stream,
                new Cuda.Arguments()
                        .pointer(block)
                        .pointer(errors)
                        .pointer(factor.upper())
                        .pointer(decodes)
                        .pointer(groupDevice + 4L * begin)
                        .pointer(orderDevice + 4L * begin)
                        .pointer(codes)
                        .pointer(decoded)
                        .integer(n)
                        .integer(width)
                        .pointer((long) k)
                        .pointer(1L)
                        .pointer((long) k)
                        .integer(begin)
                        .integer(first)
                        .integer(last)
                        .integer(groups)
                        .integer(format.nvfp4() ? 1 : 0)
                        .real(format.qmin())
                        .real(format.qmax()));
    }

    private void choose(
            int g,
            long w,
            long block,
            int k,
            int width,
            int begin,
            int end,
            long positionDevice,
            int n,
            int group,
            long values,
            long weightsDevice,
            Format format,
            TensorScales tensor,
            long tableDevice,
            long scales,
            long decodes,
            int groups) {
        cuda.launch(
                gatherGroup,
                blocks(n, 128),
                1,
                128,
                0,
                stream,
                new Cuda.Arguments()
                        .pointer(w)
                        .pointer((long) k)
                        .pointer(block)
                        .integer(width)
                        .integer(begin)
                        .integer(end)
                        .pointer(positionDevice + 4L * g * group)
                        .integer(group)
                        .integer(n)
                        .pointer(values));
        long weights = weightsDevice + 4L * g * group;
        if (format.nvfp4()) {
            cuda.launch(
                    nvfp4Choose,
                    blocks(n, 128),
                    1,
                    128,
                    0,
                    stream,
                    new Cuda.Arguments()
                            .pointer(values)
                            .integer(group)
                            .pointer(weights)
                            .integer(n)
                            .real(tensor.globalScale())
                            .pointer(tableDevice)
                            .integer(SEARCH_BELOW)
                            .pointer(scales + g)
                            .pointer(decodes + 4L * g)
                            .integer(groups));
        } else {
            cuda.launch(
                    groupedChoose,
                    blocks(n, 128),
                    1,
                    128,
                    0,
                    stream,
                    new Cuda.Arguments()
                            .pointer(values)
                            .integer(group)
                            .pointer(weights)
                            .integer(n)
                            .integer(group)
                            .integer(format.qmin())
                            .integer(format.qmax())
                            .pointer(scales + 2L * g)
                            .pointer(decodes + 4L * g)
                            .integer(groups));
        }
    }

    private QuantState state(
            int n, int k, int groups, Format format, TensorScales tensor, long codes, long scales, Arena arena) {
        MemorySegment wide = arena.allocate(2L * n * k);
        gpu.copyDeviceToHost(wide, codes, wide.byteSize());
        MemorySegment narrow = arena.allocate((long) n * k);
        for (long i = 0; i < (long) n * k; i++)
            narrow.set(ValueLayout.JAVA_BYTE, i, (byte) wide.getAtIndex(ValueLayout.JAVA_SHORT, i));
        MemorySegment hostScales = arena.allocate((format.nvfp4() ? 1L : 2L) * n * groups);
        gpu.copyDeviceToHost(hostScales, scales, hostScales.byteSize());
        return new QuantState(
                format.name(),
                format.nvfp4() ? format.layout() : "row-split-k128-v1",
                n,
                k,
                narrow,
                hostScales,
                format.nvfp4() ? tensor.globalScale() : 0.0f,
                format.sd4() ? tensor.table() : null);
    }

    private long upload(int[] values) {
        long address = gpu.allocate(4L * Math.max(1, values.length));
        gpu.copyHostToDevice(address, MemorySegment.ofArray(values), 4L * values.length);
        return address;
    }

    private long upload(float[] values) {
        long address = gpu.allocate(4L * Math.max(1, values.length));
        gpu.copyHostToDevice(address, MemorySegment.ofArray(values), 4L * values.length);
        return address;
    }

    /// Stream-ordered uploads for round(), freed with freeAsync on the same stream.
    private long uploadAsync(int[] values) {
        long address = gpu.allocateAsync(4L * Math.max(1, values.length));
        gpu.copyHostToDevice(address, MemorySegment.ofArray(values), 4L * values.length);
        return address;
    }

    private long uploadAsync(float[] values) {
        long address = gpu.allocateAsync(4L * Math.max(1, values.length));
        gpu.copyHostToDevice(address, MemorySegment.ofArray(values), 4L * values.length);
        return address;
    }

    private long uploadAsync(byte[] values) {
        long address = gpu.allocateAsync(Math.max(1, values.length));
        gpu.copyHostToDevice(address, MemorySegment.ofArray(values), values.length);
        return address;
    }

    @Override
    public void close() {}
}
