package io.euhedral_execution.inference.quantization.trellis;

import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import io.euhedral_execution.inference.quantization.gpu.Cuda;
import io.euhedral_execution.inference.quantization.gpu.Upload;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.SplittableRandom;
import java.util.stream.IntStream;

/// One linear quantized in exllamav3's trellis format (EXL3: QTIP's tail-biting trellis, mul1 codebook), entirely on
/// the device. Ported from exllamav3 1.6.1, `quantize_exl3` and the functions it calls; the random signs come from a
/// seeded `SplittableRandom` instead of torch's generator.
///
/// Portions ported from exllamav3 (MIT License, Copyright (c) 2025 Turboderp).
///
/// Weights are handled as `W [k = in, n = out]` in FP32. Steps, each a private method named after its exllamav3
/// counterpart: `finalizeHessian` (damp, rotate with the signs and the 128-point Hadamards), `blockLdl` (Cholesky
/// with damping retries, block-16 normalisation), `regularize` (output and input channel scales, rotations),
/// `globalScale` (sampled-tile scale search), `ldlq` (error feedback strip by strip through the Viterbi),
/// `refitScales` (two rounds of closed-form refit of the fp16 scales in the Hessian metric). Nothing synchronizes
/// inside the strip loop; the host reads back the diagonal mean, one Cholesky status per attempt, the tile energies
/// and two candidate grids of the scale search, and at the end the packed tensors and the error sums. Callers must have
/// the instance's stream bound on the thread (`Device.bindOwnStream`), because the uploads use the bound stream.
///
/// Peak device memory for `k` inputs and `n` outputs, on top of the caller's weight and Hessian: the regularized
/// weight, the decoded matrix and the feedback cache (`4 * k * n` bytes each) and the factor `L` (`4 * k * k`) while
/// rounding; afterwards the original weight (rebuilt from the BF16 input), the decoded matrix, `H W` and one product
/// (`4 * k * n` each) and the refit's system (`4 * k * k`); plus the Viterbi scratch ([#DEFAULT_VITERBI_SCRATCH_BYTES]
/// by default). Measured whole-device
/// peaks: 2.1 GB for `k = 5120`, `n = 17408` and 4.3 GB for `k = 17408`, `n = 5120` (1.4 GB of it the caller's).
public final class TrellisQuantizer implements AutoCloseable {

    /// Share of the mean Hessian diagonal added as damping, and (twice) per Cholesky retry.
    public static final double SIGMA_REG = 0.025;

    /// Default device memory for the Viterbi history; tiles run in chunks that fit.
    public static final long DEFAULT_VITERBI_SCRATCH_BYTES = 256L << 20;

    private static final int CHOLESKY_RETRIES = 10;
    private static final int STRIP_SPAN = 128;

    /// The packed trellis `[k / 16][n / 16][16 * bits]` int16 words, the fp16 bits of `suh` and `svh`, the BF16 `[n,
    /// k]`
    /// weight they reconstruct to (device address, owned by the caller), the proxy error `tr(E H Eᵀ) / tr(W H Wᵀ)`
    /// (the mean squared error of the regularized weight when uncalibrated), the global scale that was folded
    /// into `suh` and the number of refit rounds whose input-side factorization failed (those left the input scales
    /// alone).
    public record Result(
            short[] trellis,
            short[] suh,
            short[] svh,
            long decoded,
            double proxyError,
            float globalScale,
            int refitFailures) {}

    private final Cuda cuda;
    private final CudaGpuMemory gpu;
    private final long stream;
    private final TrellisKernels kernels;
    private final long cublas, cusolver;
    private final long scratch, small;
    private final List<Long> live = new ArrayList<>();

    private final long viterbiScratchBytes;

    public TrellisQuantizer(Cuda cuda, CudaGpuMemory gpu, long stream) {
        this(cuda, gpu, stream, DEFAULT_VITERBI_SCRATCH_BYTES);
    }

    /// `viterbiScratchBytes` is the device memory for the Viterbi history, fixed per instance.
    public TrellisQuantizer(Cuda cuda, CudaGpuMemory gpu, long stream, long viterbiScratchBytes) {
        this.viterbiScratchBytes = viterbiScratchBytes;
        this.cuda = cuda;
        this.gpu = gpu;
        this.stream = stream;
        this.kernels = new TrellisKernels(cuda, stream);
        this.cublas = cuda.cublasHandle(stream);
        this.cusolver = cuda.cusolverHandle(stream);
        this.scratch = gpu.allocate(viterbiScratchBytes);
        // 0: float mean, 1: float spare, 2-3: ints (Cholesky status, solve status), 4-11: four doubles (bytes 16-47),
        // 12: int refit failures (byte 48)
        this.small = gpu.allocate(64);
    }

    @Override
    public void close() {
        gpu.free(scratch);
        gpu.free(small);
    }

    /// Quantizes `weight` (BF16 `[n, k]`, `nn.Linear` layout, device) against `hessian` (FP32 `[k, k]` symmetric,
    /// normalised by the calibration row count and undamped, device; left unchanged), or without one when `hessian`
    /// is 0, `count` is 0, or its mean diagonal is non-finite or below 1e-20. `bits` is the rate (2..6, 2.5, 3.5),
    /// `seed` draws the sign vectors, `applyOutScales` the per-output-channel scales.
    public Result quantize(
            long weight, int n, int k, long hessian, long count, double bits, long seed, boolean applyOutScales) {
        if (n <= 0 || k <= 0 || n % 128 != 0 || k % 128 != 0)
            throw new IllegalArgumentException("shape [" + n + ", " + k + "] not a multiple of 128");
        int wordsPerTile = TrellisKernels.wordsPerTile(bits);
        int scratchTiles = TrellisKernels.scratchTiles(viterbiScratchBytes, bits);
        long kn = (long) k * n;
        int tiles = (k / 16) * (n / 16);
        try {
            SplittableRandom random = new SplittableRandom(seed);
            long suSigns = upload(signs(random, k)), svSigns = upload(signs(random, n));
            long original = allocate(4 * kn);
            launch("bf16_to_float_transposed", (k + 31) / 32, (n + 31) / 32, 256, weight, n, k, original);

            double diagonalMean = 0;
            boolean fallback = count == 0 || hessian == 0;
            if (!fallback) {
                launch("strided_mean", 1, 1, 1024, hessian, k, (long) k + 1, small);
                diagonalMean = readFloats(small, 1)[0];
                fallback = !Double.isFinite(diagonalMean) || diagonalMean < 1e-20;
            }
            long factor = fallback ? 0 : finalizeHessian(hessian, k, suSigns);

            long regularized = allocate(4 * kn);
            launch("copy_floats", elementwise(kn), 1, 256, original, regularized, kn);
            release(original);
            long su = allocate(4L * k), sv = allocate(4L * n);
            float scale = regularize(regularized, n, k, suSigns, svSigns, su, sv, applyOutScales, bits);

            long decoded = allocate(4 * kn), encoded = allocate(2 * kn);
            ldlq(regularized, factor, decoded, encoded, n, k, bits, scratchTiles);
            long words = allocate(2L * tiles * wordsPerTile);
            kernels.pack(encoded, tiles, bits, words);

            double proxy = Double.NaN;
            if (fallback) {
                long error = allocate(4 * kn);
                launch("fill_floats", 1, 1, 32, small + 16, 8L, 0f);
                launch("subtract_floats", elementwise(kn), 1, 256, regularized, decoded, error, kn);
                launch("sum_products", Math.min(1024, elementwise(kn)), 1, 256, error, error, kn, small + 16);
                proxy = readDoubles(small + 16, 1)[0] / kn;
            } else {
                release(factor);
            }
            release(regularized, encoded);

            // back to the original basis: Q = had_r(had_l(decoded) * su) * sv
            kernels.hadamardRows(decoded, k, n);
            launch("scale_floats", (n + 255) / 256, k, 256, decoded, k, n, su, 0, 0);
            kernels.hadamardColumns(decoded, k, n);
            launch("scale_floats", (n + 255) / 256, k, 256, decoded, k, n, sv, 1, 0);
            int refitFailures = 0;
            if (!fallback) {
                original = allocate(4 * kn);
                launch("bf16_to_float_transposed", (k + 31) / 32, (n + 31) / 32, 256, weight, n, k, original);
                double[] refit =
                        refitScales(original, decoded, hessian, (float) (SIGMA_REG * diagonalMean), su, sv, n, k);
                proxy = refit[0];
                refitFailures = (int) refit[1];
                release(original);
            }

            long suh = allocate(2L * k), svh = allocate(2L * n);
            launch("float_to_half", elementwise(k), 1, 256, su, (long) k, suh);
            launch("float_to_half", elementwise(n), 1, 256, sv, (long) n, svh);
            long rebuilt = gpu.allocate(2 * kn);
            try {
                kernels.reconstruct(words, suh, svh, k, n, bits, rebuilt, decoded);
                cuda.synchronize(stream);
                short[] trellis = readShorts(words, tiles * wordsPerTile);
                short[] suhHost = readShorts(suh, k), svhHost = readShorts(svh, n);
                return new Result(trellis, suhHost, svhHost, rebuilt, proxy, scale, refitFailures);
            } catch (RuntimeException | Error failure) {
                gpu.free(rebuilt);
                throw failure;
            }
        } finally {
            releaseAll();
        }
    }

    /// Damped, sign-flipped and rotated Hessian `H' = P S (H + d I) S P`, then its block-LDL factor. Returns the
    /// factor (row-major lower, zero diagonal blocks), a new device matrix.
    private long finalizeHessian(long hessian, int k, long suSigns) {
        long rotated = allocate(4L * k * k);
        launch("copy_floats", elementwise((long) k * k), 1, 256, hessian, rotated, (long) k * k);
        launch("add_diagonal", (k + 255) / 256, 1, 256, rotated, k, small, (float) SIGMA_REG);
        launch("scale_floats", (k + 255) / 256, k, 256, rotated, k, k, suSigns, 1, 0);
        kernels.hadamardColumns(rotated, k, k);
        launch("scale_floats", (k + 255) / 256, k, 256, rotated, k, k, suSigns, 0, 0);
        kernels.hadamardRows(rotated, k, k);
        blockLdl(rotated, k);
        return rotated;
    }

    /// In place: the Cholesky factor of the row-major `h` (retried up to ten times with `2 sigma mean(diag)` more
    /// damping), each 16-column block times the inverse of its diagonal block, diagonal blocks zeroed.
    private void blockLdl(long h, int k) {
        long diagonal = allocate(4L * k);
        // the upper triangle (what the factorization leaves alone) mirrors the lower, to restore it on a retry
        launch("triangle", (k + 31) / 32, (k + 7) / 8, 256, h, k, 0);
        launch("diagonal_io", (k + 255) / 256, 1, 256, h, k, diagonal, 1);
        int workspaceFloats = Math.max(1, cuda.solverWorkspace(cusolver, false, Cuda.FILL_UPPER, k, h));
        long workspace = allocate(4L * workspaceFloats);
        for (int attempt = 0; ; attempt++) {
            // the matrix is row-major: cuSOLVER's upper fill is the row-major lower triangle
            cuda.cholesky(cusolver, false, Cuda.FILL_UPPER, k, h, workspace, workspaceFloats, small + 8);
            cuda.synchronize(stream);
            if (readInts(small + 8, 1)[0] == 0) break;
            if (attempt == CHOLESKY_RETRIES)
                throw new IllegalStateException("Cholesky factorization failed after " + CHOLESKY_RETRIES + " retries");
            launch("triangle", (k + 31) / 32, (k + 7) / 8, 256, h, k, 1);
            launch("diagonal_io", (k + 255) / 256, 1, 256, h, k, diagonal, 0);
            launch("strided_mean", 1, 1, 1024, h, k, (long) k + 1, small);
            launch("add_diagonal", (k + 255) / 256, 1, 256, h, k, small, (float) (2 * SIGMA_REG));
            launch("diagonal_io", (k + 255) / 256, 1, 256, h, k, diagonal, 1);
        }
        launch("triangle", (k + 31) / 32, (k + 7) / 8, 256, h, k, 2);
        long inverse = allocate(4L * 256 * (k / 16));
        launch("invert_diagonal_blocks", k / 16, 1, 16, h, k, inverse);
        launch("right_multiply_blocks", (k + 127) / 128, k / 16, 128, h, k, inverse);
        release(diagonal, workspace, inverse);
    }

    /// Rotates `w` `[k, n]` into the distribution the trellis expects and fills `su`, `sv` with the scale vectors
    /// (signs folded in): output scales (column RMS, mean one), `W / sv`, Hadamard along n, input scales (row RMS),
    /// `W / su`, Hadamard along k, then the searched global scale `g`: `W *= g`, `su /= g`. Returns `g`.
    private float regularize(
            long w, int n, int k, long suSigns, long svSigns, long su, long sv, boolean applyOutScales, double bits) {
        long rms = allocate(4L * Math.max(n, k)), divisor = allocate(4L * n);
        launch("column_rms", (n + 31) / 32, 1, 256, w, k, n, rms);
        launch("strided_mean", 1, 1, 1024, rms, n, 1L, small);
        launch("output_scales", (n + 255) / 256, 1, 256, rms, small, n, svSigns, applyOutScales ? 1 : 0, divisor, sv);
        launch("scale_floats", (n + 255) / 256, k, 256, w, k, n, divisor, 1, 1);
        kernels.hadamardColumns(w, k, n);
        launch("row_rms", k, 1, 256, w, k, n, rms);
        launch("input_scales", (k + 255) / 256, 1, 256, rms, suSigns, k, su);
        launch("scale_floats", (n + 255) / 256, k, 256, w, k, n, su, 0, 1);
        kernels.hadamardRows(w, k, n);
        release(rms, divisor);
        float g = globalScale(w, n, k, bits);
        launch("scale_scalar", elementwise((long) k * n), 1, 256, w, (long) k * n, g, 0);
        launch("scale_scalar", elementwise(k), 1, 256, su, (long) k, g, 1);
        return g;
    }

    /// exllamav3's drift allowance for the feedback's growth of the weights, by rate.
    private static double drift(double bits) {
        if (bits == 2) return 1.018;
        if (bits == 2.5) return 1.009;
        if (bits == 3) return 1.004;
        return 1.0;
    }

    /// The global scale: quantization error (MSE of `quantize(x s) / s` against `x`) over a sample of tiles (a
    /// wrapped diagonal of width 3 plus the highest and lowest energy tiles), minimised over a coarse grid on every
    /// third sample, a fine grid around its minimum on all, and a parabolic refinement.
    private float globalScale(long w, int n, int k, double bits) {
        int tilesK = k / 16, tilesN = n / 16, width = 3;
        int diagonalLength = Math.max(tilesK, tilesN);
        int total = tilesK * tilesN;
        long energy = allocate(4L * total);
        launch("tile_mean_square", tilesN, tilesK, 256, w, n, energy);
        cuda.synchronize(stream);
        float[] tileEnergy = readFloats(energy, total);
        release(energy);
        int extreme = Math.min(Math.max(8, diagonalLength * width / 16), (total + 1) / 2);
        Integer[] order = IntStream.range(0, total).boxed().toArray(Integer[]::new);
        Arrays.sort(
                order, Comparator.comparingDouble((Integer i) -> tileEnergy[i]).thenComparingInt(i -> i));
        int samples = diagonalLength * width + 2 * extreme;
        int[] tileK = new int[samples], tileN = new int[samples];
        for (int i = 0; i < diagonalLength; i++)
            for (int j = 0; j < width; j++) {
                tileK[i * width + j] = i % tilesK;
                tileN[i * width + j] = (i + j) % tilesN;
            }
        int at = diagonalLength * width;
        for (int i = 0; i < extreme; i++) {
            int high = order[total - 1 - i], low = order[i];
            tileK[at + i] = high / tilesN;
            tileN[at + i] = high % tilesN;
            tileK[at + extreme + i] = low / tilesN;
            tileN[at + extreme + i] = low % tilesN;
        }
        long tileKDevice = upload(tileK), tileNDevice = upload(tileN);
        long sampled = allocate(1024L * samples);
        launch("gather_tiles", samples, 1, 256, w, n, tileKDevice, tileNDevice, (float) drift(bits), sampled);

        double[] coarse = new double[10];
        for (int i = 0; i < coarse.length; i++) coarse[i] = 0.1 + 0.2 * i;
        float[] coarseError = candidateErrors(sampled, samples, 3, coarse, bits);
        int center = 0;
        for (int i = 1; i < coarse.length; i++) if (coarseError[i] < coarseError[center]) center = i;
        double step = 0.075;
        double[] fine = new double[5];
        for (int i = 0; i < fine.length; i++) fine[i] = coarse[center] + step * (i - 2);
        float[] fineError = candidateErrors(sampled, samples, 1, fine, bits);
        int best = 0;
        for (int i = 1; i < fine.length; i++) if (fineError[i] < fineError[best]) best = i;
        double offset = 0;
        if (best > 0 && best < fine.length - 1) {
            double y0 = fineError[best - 1], y1 = fineError[best], y2 = fineError[best + 1];
            double denominator = y0 - 2 * y1 + y2;
            offset = denominator > 0 ? 0.5 * (y0 - y2) / denominator : 0;
            offset = Math.max(-0.5, Math.min(0.5, offset));
        }
        release(tileKDevice, tileNDevice, sampled);
        return (float) Math.max(fine[best] + offset * step, 0.01);
    }

    /// Mean squared error of quantizing every `stride`th of the `samples` tiles at each of `scales`.
    private float[] candidateErrors(long sampled, int samples, int stride, double[] scales, double bits) {
        int per = (samples + stride - 1) / stride, candidates = scales.length;
        int count = candidates * per;
        float[] floatScales = new float[candidates];
        for (int i = 0; i < candidates; i++) floatScales[i] = (float) scales[i];
        long scalesDevice = upload(floatScales);
        long scaled = allocate(1024L * count), quantized = allocate(1024L * count), indices = allocate(512L * count);
        long values = 256L * count;
        launch("candidate_tiles", elementwise(values), 1, 256, sampled, stride, per, scalesDevice, values, scaled);
        kernels.viterbi(
                scaled,
                count,
                bits,
                indices,
                quantized,
                scratch,
                TrellisKernels.scratchTiles(viterbiScratchBytes, bits));
        long error = allocate(4L * candidates);
        launch("candidate_mse", candidates, 1, 256, quantized, sampled, stride, per, scalesDevice, error);
        cuda.synchronize(stream);
        float[] result = readFloats(error, candidates);
        release(scalesDevice, scaled, quantized, indices, error);
        return result;
    }

    /// LDLQ over strips of 16 rows from the last up, in spans of 128 rows. A strip's tiles are its weights plus the
    /// feedback of the error of every later row (earlier spans through the cache, its own span directly), and the
    /// Viterbi rounds them. `factor` 0 skips the feedback (the uncalibrated path). Fills `decoded` (the rounded
    /// values `[k, n]`) and `encoded` (the states `[k / 16, n / 16, 256]`).
    private void ldlq(long w, long factor, long decoded, long encoded, int n, int k, double bits, int scratchTiles) {
        int tilesN = n / 16;
        long cache = factor == 0 ? 0 : allocate(4L * k * n);
        if (cache != 0) launch("fill_floats", elementwise((long) k * n), 1, 256, cache, (long) k * n, 0f);
        long strip = allocate(4L * n * 16), rounded = allocate(4L * n * 16);
        long error = factor == 0 ? 0 : allocate(4L * STRIP_SPAN * n);
        for (int j = k; j > 0; j -= STRIP_SPAN) {
            int i = j - STRIP_SPAN;
            for (int bj = STRIP_SPAN; bj > 0; bj -= 16) {
                int bi = bj - 16, row = i + bi, later = STRIP_SPAN - bj;
                if (factor != 0 && later > 0)
                    cuda.gemmFp32(
                            cublas,
                            Cuda.CUBLAS_OP_N,
                            Cuda.CUBLAS_OP_T,
                            n,
                            16,
                            later,
                            1f,
                            error + 4L * bj * n,
                            n,
                            factor + 4L * ((long) (i + bj) * k + row),
                            k,
                            1f,
                            cache + 4L * row * n,
                            n);
                launch("strip_gather_sum", tilesN, 1, 256, w, cache, n, row, strip);
                kernels.viterbi(
                        strip, tilesN, bits, encoded + 2L * (row / 16) * tilesN * 256, rounded, scratch, scratchTiles);
                kernels.fromTiles(rounded, k, n, row, decoded);
                if (factor != 0) launch("strip_error", (16 * n + 255) / 256, 1, 256, w, decoded, n, row, error, bi);
            }
            if (factor != 0 && i > 0)
                cuda.gemmFp32(
                        cublas,
                        Cuda.CUBLAS_OP_N,
                        Cuda.CUBLAS_OP_T,
                        n,
                        i,
                        STRIP_SPAN,
                        1f,
                        error,
                        n,
                        factor + 4L * i * k,
                        k,
                        1f,
                        cache,
                        n);
        }
        release(cache, strip, rounded, error);
    }

    /// Two rounds of closed-form refit of the scales to the rounded weight `q` (original basis, `[k, n]`, scaled in
    /// place) in the Hessian metric with the damping `d`: each output channel `c = qᵀ H w / qᵀ H q` (into `q` and
    /// `sv`), then the input scales solve `((Q Qᵀ) ∘ H) r = rowsum(Q ∘ H W)` (into `q` and `su`). A failed
    /// factorization leaves a round's input scales alone and is counted. Returns the proxy error
    /// `tr(E H Eᵀ) / tr(W H Wᵀ)` of the refitted weight with the undamped H, and that count.
    private double[] refitScales(long original, long q, long hessian, float damping, long su, long sv, int n, int k) {
        long kn = (long) k * n;
        long hw = allocate(4 * kn), product = allocate(4 * kn), system = allocate(4L * k * k);
        long numerator = allocate(4L * Math.max(n, k)), denominator = allocate(4L * n);
        launch("fill_floats", 1, 1, 32, small + 16, 9L, 0f);
        gemmH(original, hessian, hw, n, k);
        launch("sum_products", Math.min(1024, elementwise(kn)), 1, 256, original, hw, kn, small + 16);
        launch("axpy_floats", elementwise(kn), 1, 256, hw, original, kn, damping);
        int workspaceFloats = Math.max(1, cuda.solverWorkspace(cusolver, false, Cuda.FILL_UPPER, k, system));
        long workspace = allocate(4L * workspaceFloats);
        for (int round = 0; round < 2; round++) {
            gemmH(q, hessian, product, n, k);
            launch("column_dots", (n + 31) / 32, 1, 256, q, hw, 0f, k, n, numerator);
            launch("column_dots", (n + 31) / 32, 1, 256, q, product, damping, k, n, denominator);
            launch("column_ratio", (n + 255) / 256, 1, 256, numerator, denominator, n);
            launch("scale_floats", (n + 255) / 256, k, 256, q, k, n, numerator, 1, 0);
            launch("scale_floats", (n + 255) / 256, 1, 256, sv, 1, n, numerator, 1, 0);

            cuda.gemmFp32(cublas, Cuda.CUBLAS_OP_T, Cuda.CUBLAS_OP_N, k, k, n, 1f, q, n, q, n, 0f, system, k);
            launch("hessian_product", (k + 255) / 256, k, 256, system, hessian, k, damping);
            launch("strided_mean", 1, 1, 1024, system, k, (long) k + 1, small);
            launch("add_diagonal", (k + 255) / 256, 1, 256, system, k, small, 1e-6f);
            launch("row_dots", k, 1, 256, q, hw, k, n, numerator);
            cuda.cholesky(cusolver, false, Cuda.FILL_UPPER, k, system, workspace, workspaceFloats, small + 8);
            cuda.solveCholesky(cusolver, Cuda.FILL_UPPER, k, 1, system, numerator, small + 12);
            launch("clean_row_scales", (k + 255) / 256, 1, 256, numerator, small + 8, k, small + 48);
            launch("scale_floats", (n + 255) / 256, k, 256, q, k, n, numerator, 0, 0);
            launch("scale_floats", 1, k, 1, su, k, 1, numerator, 0, 0);
        }
        launch("subtract_floats", elementwise(kn), 1, 256, original, q, product, kn);
        gemmH(product, hessian, hw, n, k);
        launch("sum_products", Math.min(1024, elementwise(kn)), 1, 256, product, hw, kn, small + 24);
        cuda.synchronize(stream);
        double[] sums = readDoubles(small + 16, 2);
        int failures = readInts(small + 48, 1)[0];
        release(hw, product, system, numerator, denominator, workspace);
        return new double[] {sums[1] / Math.max(sums[0], 1e-30), failures};
    }

    /// `out [k, n] = H [k, k] x [k, n]`, row-major, FP32.
    private void gemmH(long x, long hessian, long out, int n, int k) {
        cuda.gemmFp32(cublas, Cuda.CUBLAS_OP_N, Cuda.CUBLAS_OP_N, n, k, k, 1f, x, n, hessian, k, 0f, out, n);
    }

    private static float[] signs(SplittableRandom random, int count) {
        float[] signs = new float[count];
        for (int i = 0; i < count; i++) signs[i] = random.nextBoolean() ? 1f : -1f;
        return signs;
    }

    // ---- device helpers ----

    private static int elementwise(long count) {
        return (int) Math.max(1, Math.min(1 << 20, (count + 255) / 256));
    }

    private void launch(String kernel, int gridX, int gridY, int block, Object... arguments) {
        Cuda.Arguments list = new Cuda.Arguments();
        for (Object argument : arguments) {
            if (argument instanceof Long address) list.pointer(address);
            else if (argument instanceof Integer value) list.integer(value);
            else if (argument instanceof Float value) list.real(value);
            else throw new IllegalArgumentException("argument " + argument);
        }
        cuda.launch(kernels.kernel(kernel), gridX, gridY, block, 0, stream, list);
    }

    private long allocate(long bytes) {
        long address = gpu.allocate(Math.max(bytes, 1));
        live.add(address);
        return address;
    }

    private void release(long... addresses) {
        for (long address : addresses) {
            if (address == 0) continue;
            if (live.remove(Long.valueOf(address))) gpu.free(address);
        }
    }

    private void releaseAll() {
        cuda.synchronize(stream);
        for (long address : live) gpu.free(address);
        live.clear();
    }

    private long upload(float[] values) {
        long address = allocate(4L * values.length);
        Upload.floats(gpu, address, values);
        return address;
    }

    private long upload(int[] values) {
        long address = allocate(4L * values.length);
        Upload.ints(gpu, address, values);
        return address;
    }

    private float[] readFloats(long address, int count) {
        cuda.synchronize(stream);
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment host = arena.allocate(4L * count);
            gpu.copyDeviceToHost(host, address, 4L * count);
            return host.toArray(ValueLayout.JAVA_FLOAT);
        }
    }

    private int[] readInts(long address, int count) {
        cuda.synchronize(stream);
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment host = arena.allocate(4L * count);
            gpu.copyDeviceToHost(host, address, 4L * count);
            return host.toArray(ValueLayout.JAVA_INT);
        }
    }

    private double[] readDoubles(long address, int count) {
        cuda.synchronize(stream);
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment host = arena.allocate(8L * count);
            gpu.copyDeviceToHost(host, address, 8L * count);
            return host.toArray(ValueLayout.JAVA_DOUBLE);
        }
    }

    private short[] readShorts(long address, int count) {
        cuda.synchronize(stream);
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment host = arena.allocate(2L * count);
            gpu.copyDeviceToHost(host, address, 2L * count);
            return host.toArray(ValueLayout.JAVA_SHORT);
        }
    }
}
