package io.euhedral_execution.inference.quantization.training;

import io.euhedral_execution.inference.quantization.gpu.Cuda;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/// The training kernels (`train.cu`) and FP32 products of one stream. Tensors are FP32 and row-major; a linear's weight
/// is [out, in], as the artifact stores it. Backward operations add into gradients (`+=`) unless their name says
/// otherwise.
public final class Ops {

    private static final int THREADS = 256;
    private static final List<String> KERNELS = List.of(
            "bf16_to_float",
            "axpby",
            "fill",
            "swiglu_forward",
            "swiglu_backward",
            "rms_norm_forward",
            "rms_norm_backward",
            "dequantize_grouped",
            "scale_gradient",
            "gdn_control_forward",
            "gdn_control_backward",
            "gdn_convolution_forward",
            "gdn_convolution_backward",
            "gdn_recurrence_forward",
            "gdn_recurrence_backward",
            "gdn_gather_query_key",
            "gdn_gated_norm_forward",
            "gdn_gated_norm_backward",
            "qk_norm_rope_forward",
            "qk_norm_rope_backward",
            "causal_softmax",
            "causal_softmax_backward",
            "gate_forward",
            "gate_backward",
            "squared_error",
            "adam");

    public final Cuda cuda;
    public final long stream, cublas;
    private final Map<String, Cuda.Function> functions = new HashMap<>();

    public Ops(Cuda cuda, long stream, long cublas) {
        this.cuda = cuda;
        this.stream = stream;
        this.cublas = cublas;
        List<Cuda.Function> compiled = cuda.compileResource("train.cu", KERNELS);
        for (int i = 0; i < KERNELS.size(); i++) functions.put(KERNELS.get(i), compiled.get(i));
        cuda.allowSharedMemory(functions.get("gdn_recurrence_forward"), 96 * 1024);
        cuda.allowSharedMemory(functions.get("gdn_recurrence_backward"), 96 * 1024);
    }

    private void elementwise(String kernel, long count, Cuda.Arguments arguments) {
        cuda.launch(functions.get(kernel), (int) ((count + THREADS - 1) / THREADS), 1, THREADS, 0, stream, arguments);
    }

    private void rowwise(String kernel, long blocks, int threads, int shared, Cuda.Arguments arguments) {
        cuda.launch(functions.get(kernel), Math.toIntExact(blocks), 1, threads, shared, stream, arguments);
    }

    // ---------------------------------------------------------------- products

    /// y [rows, out] = x [rows, in] w^T (+ y when `accumulate`).
    public void linear(long x, long w, long y, int rows, int in, int out, boolean accumulate) {
        cuda.gemmFp32(
                cublas, Cuda.CUBLAS_OP_T, Cuda.CUBLAS_OP_N, out, rows, in, 1, w, in, x, in, accumulate ? 1 : 0, y, out);
    }

    /// dx [rows, in] += dy [rows, out] w.
    public void linearInputGradient(long dy, long w, long dx, int rows, int in, int out) {
        cuda.gemmFp32(cublas, Cuda.CUBLAS_OP_N, Cuda.CUBLAS_OP_N, in, rows, out, 1, w, in, dy, out, 1, dx, in);
    }

    /// dw [out, in] = dy^T x (overwritten).
    public void linearWeightGradient(long dy, long x, long dw, int rows, int in, int out) {
        cuda.gemmFp32(cublas, Cuda.CUBLAS_OP_N, Cuda.CUBLAS_OP_T, in, out, rows, 1, x, in, dy, out, 0, dw, in);
    }

    /// Strided batched-free product for attention: c [m, n] = alpha a [m, k] b^T [k, n] (+ beta c), row-major, given
    /// leading dimensions.
    public void productTransposed(
            long a, int lda, long b, int ldb, long c, int ldc, int m, int n, int k, float alpha, float beta) {
        cuda.gemmFp32(cublas, Cuda.CUBLAS_OP_T, Cuda.CUBLAS_OP_N, n, m, k, alpha, b, ldb, a, lda, beta, c, ldc);
    }

    /// c [m, n] = alpha a [m, k] b [k, n] (+ beta c), row-major.
    public void product(
            long a, int lda, long b, int ldb, long c, int ldc, int m, int n, int k, float alpha, float beta) {
        cuda.gemmFp32(cublas, Cuda.CUBLAS_OP_N, Cuda.CUBLAS_OP_N, n, m, k, alpha, b, ldb, a, lda, beta, c, ldc);
    }

    /// c [m, n] = alpha a^T [m, k] b [k, n] (+ beta c) with a stored [k, m], row-major.
    public void productLeftTransposed(
            long a, int lda, long b, int ldb, long c, int ldc, int m, int n, int k, float alpha, float beta) {
        cuda.gemmFp32(cublas, Cuda.CUBLAS_OP_N, Cuda.CUBLAS_OP_T, n, m, k, alpha, b, ldb, a, lda, beta, c, ldc);
    }

    // ---------------------------------------------------------------- elementwise

    public void bf16ToFloat(long in, long count, long out) {
        elementwise(
                "bf16_to_float",
                count,
                new Cuda.Arguments().pointer(in).pointer(count).pointer(out));
    }

    /// y = a x + b y.
    public void axpby(long x, long y, long count, float a, float b) {
        elementwise(
                "axpby",
                count,
                new Cuda.Arguments()
                        .pointer(x)
                        .pointer(y)
                        .pointer(count)
                        .real(a)
                        .real(b));
    }

    public void fill(long x, long count, float value) {
        elementwise(
                "fill", count, new Cuda.Arguments().pointer(x).pointer(count).real(value));
    }

    public void zero(long x, long count) {
        fill(x, count, 0);
    }

    public void swigluForward(long gateUp, long h, int rows, int width) {
        elementwise(
                "swiglu_forward",
                (long) rows * width,
                new Cuda.Arguments().pointer(gateUp).pointer(h).integer(rows).integer(width));
    }

    public void swigluBackward(long gateUp, long dh, long dGateUp, int rows, int width) {
        elementwise(
                "swiglu_backward",
                (long) rows * width,
                new Cuda.Arguments()
                        .pointer(gateUp)
                        .pointer(dh)
                        .pointer(dGateUp)
                        .integer(rows)
                        .integer(width));
    }

    // ---------------------------------------------------------------- norms

    /// y = x inv (offset + w) per row; inverse [rows] keeps inv.
    public void rmsNormForward(long x, long w, long y, long inverse, int rows, int width, float eps, float offset) {
        rowwise(
                "rms_norm_forward",
                rows,
                THREADS,
                0,
                new Cuda.Arguments()
                        .pointer(x)
                        .pointer(w)
                        .pointer(y)
                        .pointer(inverse)
                        .integer(width)
                        .real(eps)
                        .real(offset));
    }

    public void rmsNormBackward(
            long x, long w, long inverse, long dy, long dx, long dw, int rows, int width, float offset) {
        rowwise(
                "rms_norm_backward",
                rows,
                THREADS,
                0,
                new Cuda.Arguments()
                        .pointer(x)
                        .pointer(w)
                        .pointer(inverse)
                        .pointer(dy)
                        .pointer(dx)
                        .pointer(dw)
                        .integer(width)
                        .real(offset));
    }

    // ---------------------------------------------------------------- grouped quantized weights

    public void dequantize(long codes, long scales, long w, long n, int k, int group) {
        elementwise(
                "dequantize_grouped",
                n * k,
                new Cuda.Arguments()
                        .pointer(codes)
                        .pointer(scales)
                        .pointer(w)
                        .pointer(n)
                        .integer(k)
                        .integer(group));
    }

    public void scaleGradient(long codes, long dw, long dscales, long n, int k, int group) {
        long groups = n * (k / group);
        cuda.launch(
                functions.get("scale_gradient"),
                (int) ((groups * 32 + THREADS - 1) / THREADS),
                1,
                THREADS,
                0,
                stream,
                new Cuda.Arguments()
                        .pointer(codes)
                        .pointer(dw)
                        .pointer(dscales)
                        .pointer(groups)
                        .integer(k)
                        .integer(group));
    }

    // ---------------------------------------------------------------- GDN

    public void gdnControlForward(long a, long b, long aLog, long dtBias, long alpha, long beta, int rows, int heads) {
        elementwise(
                "gdn_control_forward",
                (long) rows * heads,
                new Cuda.Arguments()
                        .pointer(a)
                        .pointer(b)
                        .pointer(aLog)
                        .pointer(dtBias)
                        .pointer(alpha)
                        .pointer(beta)
                        .integer(rows)
                        .integer(heads));
    }

    public void gdnControlBackward(
            long a,
            long aLog,
            long dtBias,
            long alpha,
            long beta,
            long dAlpha,
            long dBeta,
            long da,
            long db,
            int rows,
            int heads) {
        elementwise(
                "gdn_control_backward",
                (long) rows * heads,
                new Cuda.Arguments()
                        .pointer(a)
                        .pointer(aLog)
                        .pointer(dtBias)
                        .pointer(alpha)
                        .pointer(beta)
                        .pointer(dAlpha)
                        .pointer(dBeta)
                        .pointer(da)
                        .pointer(db)
                        .integer(rows)
                        .integer(heads));
    }

    public void gdnConvolutionForward(
            long qk, long vz, long w, long pre, long out, int rows, int qkWidth, int valueWidth, int kernel) {
        elementwise(
                "gdn_convolution_forward",
                (long) rows * (qkWidth + valueWidth),
                new Cuda.Arguments()
                        .pointer(qk)
                        .pointer(vz)
                        .pointer(w)
                        .pointer(pre)
                        .pointer(out)
                        .integer(rows)
                        .integer(qkWidth)
                        .integer(valueWidth)
                        .integer(kernel));
    }

    public void gdnConvolutionBackward(
            long pre, long dout, long w, long dqk, long dvz, int rows, int qkWidth, int valueWidth, int kernel) {
        elementwise(
                "gdn_convolution_backward",
                (long) rows * (qkWidth + valueWidth),
                new Cuda.Arguments()
                        .pointer(pre)
                        .pointer(dout)
                        .pointer(w)
                        .pointer(dqk)
                        .pointer(dvz)
                        .integer(rows)
                        .integer(qkWidth)
                        .integer(valueWidth)
                        .integer(kernel));
    }

    /// Floats of the checkpoints a recurrence of `rows` rows saves every `segment` rows.
    public static long checkpointFloats(int rows, int valueHeads, int dk, int dv, int segment) {
        return (long) valueHeads * (rows / segment + 1) * dv * dk;
    }

    public void gdnRecurrenceForward(
            long convolved,
            long alpha,
            long beta,
            long output,
            long checkpoints,
            int rows,
            int keyHeads,
            int valueHeads,
            int dk,
            int dv,
            int segment) {
        rowwise(
                "gdn_recurrence_forward",
                valueHeads,
                THREADS,
                4 * (dv * dk + 2 * dk + dv),
                new Cuda.Arguments()
                        .pointer(convolved)
                        .pointer(alpha)
                        .pointer(beta)
                        .pointer(output)
                        .pointer(checkpoints)
                        .integer(rows)
                        .integer(keyHeads)
                        .integer(valueHeads)
                        .integer(dk)
                        .integer(dv)
                        .integer(segment)
                        .real((float) (1.0 / Math.sqrt(dk))));
    }

    /// Floats of the per-segment states and per-value-head q/k gradients the recurrence's backward needs.
    public static long recurrenceScratchFloats(int rows, int valueHeads, int dk, int dv, int segment) {
        return (long) valueHeads * segment * dv * dk + (long) rows * valueHeads * 2 * dk;
    }

    /// Adds the recurrence's gradients to dConvolved (q, k and v columns) and writes dAlpha, dBeta.
    public void gdnRecurrenceBackward(
            long convolved,
            long alpha,
            long beta,
            long dOutput,
            long checkpoints,
            long scratch,
            long dConvolved,
            long dAlpha,
            long dBeta,
            int rows,
            int keyHeads,
            int valueHeads,
            int dk,
            int dv,
            int segment) {
        long states = scratch, dQK = scratch + 4L * valueHeads * segment * dv * dk;
        rowwise(
                "gdn_recurrence_backward",
                valueHeads,
                THREADS,
                4 * (dv * dk + 4 * dk + 4 * dv),
                new Cuda.Arguments()
                        .pointer(convolved)
                        .pointer(alpha)
                        .pointer(beta)
                        .pointer(dOutput)
                        .pointer(checkpoints)
                        .pointer(states)
                        .pointer(dQK)
                        .pointer(dConvolved)
                        .pointer(dAlpha)
                        .pointer(dBeta)
                        .integer(rows)
                        .integer(keyHeads)
                        .integer(valueHeads)
                        .integer(dk)
                        .integer(dv)
                        .integer(segment)
                        .real((float) (1.0 / Math.sqrt(dk))));
        elementwise(
                "gdn_gather_query_key",
                (long) rows * 2 * keyHeads * dk,
                new Cuda.Arguments()
                        .pointer(dQK)
                        .pointer(dConvolved)
                        .integer(rows)
                        .integer(keyHeads)
                        .integer(valueHeads)
                        .integer(dk)
                        .integer(dv));
    }

    public void gdnGatedNormForward(
            long o, long vz, long w, long y, long inverse, int rows, int heads, int dv, float eps) {
        rowwise(
                "gdn_gated_norm_forward",
                (long) rows * heads,
                Math.max(32, Math.min(dv, 128) / 32 * 32),
                0,
                new Cuda.Arguments()
                        .pointer(o)
                        .pointer(vz)
                        .pointer(w)
                        .pointer(y)
                        .pointer(inverse)
                        .integer(heads)
                        .integer(dv)
                        .real(eps));
    }

    public void gdnGatedNormBackward(
            long o, long vz, long w, long inverse, long dy, long dO, long dvz, long dw, int rows, int heads, int dv) {
        rowwise(
                "gdn_gated_norm_backward",
                (long) rows * heads,
                Math.max(32, Math.min(dv, 128) / 32 * 32),
                0,
                new Cuda.Arguments()
                        .pointer(o)
                        .pointer(vz)
                        .pointer(w)
                        .pointer(inverse)
                        .pointer(dy)
                        .pointer(dO)
                        .pointer(dvz)
                        .pointer(dw)
                        .integer(heads)
                        .integer(dv));
    }

    // ---------------------------------------------------------------- attention

    public void qkNormRopeForward(
            long qk,
            long qNorm,
            long kNorm,
            long out,
            long inverse,
            int rows,
            int qHeads,
            int kvHeads,
            int d,
            int rotary,
            long position0,
            double theta,
            float eps) {
        rowwise(
                "qk_norm_rope_forward",
                (long) rows * (qHeads + kvHeads),
                d,
                0,
                new Cuda.Arguments()
                        .pointer(qk)
                        .pointer(qNorm)
                        .pointer(kNorm)
                        .pointer(out)
                        .pointer(inverse)
                        .integer(qHeads)
                        .integer(kvHeads)
                        .integer(d)
                        .integer(rotary)
                        .pointer(position0)
                        .real64(theta)
                        .real(eps));
    }

    public void qkNormRopeBackward(
            long qk,
            long qNorm,
            long kNorm,
            long inverse,
            long dOut,
            long dqk,
            long dqNorm,
            long dkNorm,
            int rows,
            int qHeads,
            int kvHeads,
            int d,
            int rotary,
            long position0,
            double theta) {
        rowwise(
                "qk_norm_rope_backward",
                (long) rows * (qHeads + kvHeads),
                d,
                0,
                new Cuda.Arguments()
                        .pointer(qk)
                        .pointer(qNorm)
                        .pointer(kNorm)
                        .pointer(inverse)
                        .pointer(dOut)
                        .pointer(dqk)
                        .pointer(dqNorm)
                        .pointer(dkNorm)
                        .integer(qHeads)
                        .integer(kvHeads)
                        .integer(d)
                        .integer(rotary)
                        .pointer(position0)
                        .real64(theta));
    }

    /// Softmax in place of `heads` [rows, rows] score blocks, causal, after multiplying by `scale`.
    public void causalSoftmax(long scores, int heads, int rows, float scale) {
        rowwise(
                "causal_softmax",
                (long) heads * rows,
                THREADS,
                0,
                new Cuda.Arguments().pointer(scores).integer(rows).real(scale));
    }

    public void causalSoftmaxBackward(long probabilities, long dScores, int heads, int rows, float scale) {
        rowwise(
                "causal_softmax_backward",
                (long) heads * rows,
                THREADS,
                0,
                new Cuda.Arguments()
                        .pointer(probabilities)
                        .pointer(dScores)
                        .integer(rows)
                        .real(scale));
    }

    public void gateForward(long context, long gateValue, long out, int rows, int width, int rowWidth) {
        elementwise(
                "gate_forward",
                (long) rows * width,
                new Cuda.Arguments()
                        .pointer(context)
                        .pointer(gateValue)
                        .pointer(out)
                        .integer(rows)
                        .integer(width)
                        .integer(rowWidth));
    }

    public void gateBackward(
            long context,
            long gateValue,
            long dOut,
            long dContext,
            long dGateValue,
            int rows,
            int width,
            int rowWidth) {
        elementwise(
                "gate_backward",
                (long) rows * width,
                new Cuda.Arguments()
                        .pointer(context)
                        .pointer(gateValue)
                        .pointer(dOut)
                        .pointer(dContext)
                        .pointer(dGateValue)
                        .integer(rows)
                        .integer(width)
                        .integer(rowWidth));
    }

    // ---------------------------------------------------------------- loss and optimizer

    /// loss (one double, device) += sum (y - target)^2; dy = 2 (y - target) scale (when dy is not 0).
    public void squaredError(long y, long target, long count, float scale, long dy, long loss) {
        elementwise(
                "squared_error",
                count,
                new Cuda.Arguments()
                        .pointer(y)
                        .pointer(target)
                        .pointer(count)
                        .real(scale)
                        .pointer(dy)
                        .pointer(loss));
    }

    /// One AdamW step; zeroes the gradient.
    public void adam(
            long p,
            long g,
            long m,
            long v,
            long count,
            float lr,
            float beta1,
            float beta2,
            float eps,
            float decay,
            int step) {
        elementwise(
                "adam",
                count,
                new Cuda.Arguments()
                        .pointer(p)
                        .pointer(g)
                        .pointer(m)
                        .pointer(v)
                        .pointer(count)
                        .real(lr)
                        .real(beta1)
                        .real(beta2)
                        .real(eps)
                        .real(decay)
                        .integer(step));
    }
}
