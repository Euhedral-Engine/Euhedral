package io.euhedral_execution.inference.quantization.mtp;

import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import io.euhedral_execution.inference.quantization.gpu.Cuda;
import io.euhedral_execution.inference.quantization.training.Ops;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/// The MTP drafter (docs/MTP_CONTRACT.md §2) as a trainable FP32 model whose projections run NVFP4-rounded
/// (the artifact's rounding, [DrafterKernels#nvfp4Round]) with straight-through gradients into FP32 master
/// weights, unrolled over `depths` drafts of a window of rows.
///
/// Row r of a window is a lineage: its depth-1 row reads the base model's hidden at position a + r and the token at
/// a + r + 1, and drafts the token at a + r + 2; its depth-k row reads the depth-(k - 1) row's output and the token at
/// a + r + k, sits at position a + r + k - 1 and drafts the token at a + r + k + 1, as the decoder's recursion does.
/// Attention keys: the depth-1 keys and values of every position up to the row's lineage position (a sequence's
/// earlier windows included, from position `firstKey`), and the deeper keys and values of its own lineage, which is
/// what the drafter's cache holds when the decoder drafts.
final class Drafter implements AutoCloseable {

    /// Dimensions of the MTP layer.
    record Shape(
            int hidden, int intermediate, int heads, int kvHeads, int headDim, int rotaryDim, double theta, float eps) {
        int qkWidth() {
            return (heads + kvHeads) * headDim;
        }

        int mixerWidth() {
            return heads * headDim;
        }

        int kvWidth() {
            return kvHeads * headDim;
        }
    }

    /// A trainable tensor: FP32 values, gradient and Adam moments; a projection also its NVFP4 global scale's source,
    /// the bits of its largest magnitude (one unsigned).
    static final class Parameter {
        final String name;
        final int rows, columns;
        final long count;
        final long values, gradient, first, second, maximum;

        Parameter(CudaGpuMemory gpu, Ops ops, String name, int rows, int columns, boolean projection) {
            this.name = name;
            this.rows = rows;
            this.columns = columns;
            this.count = (long) rows * columns;
            this.values = gpu.allocate(4 * count);
            this.gradient = gpu.allocate(4 * count);
            this.first = gpu.allocate(4 * count);
            this.second = gpu.allocate(4 * count);
            this.maximum = projection ? gpu.allocate(4) : 0;
            ops.zero(gradient, count);
            ops.zero(first, count);
            ops.zero(second, count);
        }

        void free(CudaGpuMemory gpu) {
            for (long address : new long[] {values, gradient, first, second}) gpu.free(address);
            if (maximum != 0) gpu.free(maximum);
        }
    }

    static final String INPUT_PROJECTION = "mtp/input_projection";
    static final String QKGV = "mtp/layer/attention/query_key_gate_value";
    static final String OUTPUT = "mtp/layer/attention/output";
    static final String GATE_UP = "mtp/layer/mlp/gate_up";
    static final String DOWN = "mtp/layer/mlp/down";
    static final String EMBEDDING_NORM = "mtp/embedding_norm";
    static final String HIDDEN_NORM = "mtp/hidden_norm";
    static final String INPUT_NORM = "mtp/layer/input_norm";
    static final String POST_NORM = "mtp/layer/post_attention_norm";
    static final String QUERY_NORM = "mtp/layer/attention/query_norm";
    static final String KEY_NORM = "mtp/layer/attention/key_norm";
    static final String FINAL_NORM = "mtp/final_norm";
    static final List<String> PROJECTIONS = List.of(INPUT_PROJECTION, QKGV, OUTPUT, GATE_UP, DOWN);
    static final List<String> NORMS =
            List.of(EMBEDDING_NORM, HIDDEN_NORM, INPUT_NORM, POST_NORM, QUERY_NORM, KEY_NORM, FINAL_NORM);

    private final CudaGpuMemory gpu;
    private final Cuda cuda;
    private final long cublas;
    private final Ops ops;
    private final DrafterKernels kernels;
    final Shape shape;
    final int depths, rows, maxKeys, maxPositions;
    final Map<String, Parameter> parameters = new LinkedHashMap<>();
    /// Whether projections run NVFP4-rounded (training and evaluation of the exported layer) or as their FP32 values.
    boolean rounded = true;
    private final long weight;
    private final List<Map<String, Long>> depthBuffers = new ArrayList<>();
    private final Map<String, Long> shared = new HashMap<>();
    private final List<Long> owned = new ArrayList<>();
    /// The depth-1 keys and values of a sequence's positions [maxPositions, kvWidth], and their gradients.
    final long keys, values, keyGradients, valueGradients;

    Drafter(
            CudaGpuMemory gpu,
            Cuda cuda,
            long cublas,
            Ops ops,
            DrafterKernels kernels,
            Shape shape,
            int depths,
            int rows,
            int maxKeys,
            int maxPositions) {
        this.gpu = gpu;
        this.cuda = cuda;
        this.cublas = cublas;
        this.ops = ops;
        this.kernels = kernels;
        this.shape = shape;
        this.depths = depths;
        this.rows = rows;
        this.maxKeys = maxKeys;
        this.maxPositions = maxPositions;
        int h = shape.hidden, i = shape.intermediate, d = shape.headDim;
        add(INPUT_PROJECTION, h, 2 * h, true);
        add(QKGV, 2 * shape.qkWidth(), h, true);
        add(OUTPUT, h, shape.mixerWidth(), true);
        add(GATE_UP, 2 * i, h, true);
        add(DOWN, h, i, true);
        for (String norm : NORMS) add(norm, 1, norm.equals(QUERY_NORM) || norm.equals(KEY_NORM) ? d : h, false);
        long largest = 0;
        for (String name : PROJECTIONS) largest = Math.max(largest, parameters.get(name).count);
        weight = allocate(4 * largest);
        long kv = (long) maxPositions * shape.kvWidth();
        keys = allocate(4 * kv);
        values = allocate(4 * kv);
        keyGradients = allocate(4 * kv);
        valueGradients = allocate(4 * kv);
        for (int k = 0; k < depths; k++) depthBuffers.add(new HashMap<>());
    }

    private void add(String name, int rows, int columns, boolean projection) {
        parameters.put(name, new Parameter(gpu, ops, name, rows, columns, projection));
    }

    private long allocate(long bytes) {
        long address = gpu.allocate(Math.max(4, bytes));
        owned.add(address);
        return address;
    }

    /// A buffer of depth `k` (1-based), sized once.
    private long buffer(int k, String name, long floats) {
        return depthBuffers.get(k - 1).computeIfAbsent(name, ignored -> allocate(4 * floats));
    }

    private long shared(String name, long floats) {
        return shared.computeIfAbsent(name, ignored -> allocate(4 * floats));
    }

    // ---------------------------------------------------------------- weights

    /// Recomputes each projection's NVFP4 global-scale source after its values changed.
    void measure() {
        for (String name : PROJECTIONS) {
            Parameter p = parameters.get(name);
            ops.zero(p.maximum, 1);
            kernels.absoluteMaximum(p.values, p.count, p.maximum);
        }
    }

    /// The projection's weight as the forward pass uses it: NVFP4-rounded into the shared buffer, or its values.
    private long weight(String name) {
        Parameter p = parameters.get(name);
        if (!rounded) return p.values;
        kernels.nvfp4Round(p.values, weight, 0, 0, p.rows, p.columns, p.maximum);
        return weight;
    }

    /// dw [out, in] += dy^T x into rows [firstRow, firstRow + out) of the projection's gradient.
    private void weightGradient(String name, int firstRow, int out, long dy, long x, int count) {
        Parameter p = parameters.get(name);
        long dw = p.gradient + 4L * firstRow * p.columns;
        cuda.gemmFp32(
                cublas,
                Cuda.CUBLAS_OP_N,
                Cuda.CUBLAS_OP_T,
                p.columns,
                out,
                count,
                1,
                x,
                p.columns,
                dy,
                out,
                1,
                dw,
                p.columns);
    }

    // ---------------------------------------------------------------- forward

    /// Depth `k` (1-based) over the window at position `a` with `count` rows (at most `rows`): `hidden`
    /// [count, H] is the depth's input hidden (the base model's for depth 1, depth k - 1's output otherwise),
    /// `embedded` [count, H] the embedding of each row's input token. The causal keys run from position
    /// `firstKey`. Returns the output hidden (post final norm) [count, H].
    long forward(int k, long a, long firstKey, int count, long hidden, long embedded) {
        Shape s = shape;
        int h = s.hidden, d = s.headDim, width = s.qkWidth(), mixer = s.mixerWidth(), kvw = s.kvWidth();
        int group = s.heads / s.kvHeads, keyCount = Math.toIntExact(a + count - firstKey);
        if (keyCount > maxKeys) throw new IllegalArgumentException("a window reaches " + keyCount + " keys");
        if (a + count > maxPositions) throw new IllegalArgumentException("position " + (a + count) + " too far");
        long eN = buffer(k, "embedding_normed", (long) rows * h), invE = buffer(k, "embedding_inverse", rows);
        long hN = buffer(k, "hidden_normed", (long) rows * h), invH = buffer(k, "hidden_inverse", rows);
        ops.rmsNormForward(embedded, value(EMBEDDING_NORM), eN, invE, count, h, s.eps, 1);
        ops.rmsNormForward(hidden, value(HIDDEN_NORM), hN, invH, count, h, s.eps, 1);
        long cat = buffer(k, "cat", (long) rows * 2 * h);
        kernels.copyColumns(eN, h, cat, 2 * h, count, h, false);
        kernels.copyColumns(hN, h, cat + 4L * h, 2 * h, count, h, false);
        long c = buffer(k, "c", (long) rows * h);
        ops.linear(cat, weight(INPUT_PROJECTION), c, count, 2 * h, h, false);
        long n1 = buffer(k, "n1", (long) rows * h), inv1 = buffer(k, "inv1", rows);
        ops.rmsNormForward(c, value(INPUT_NORM), n1, inv1, count, h, s.eps, 1);
        long qk = buffer(k, "qk", (long) rows * width), gv = buffer(k, "gv", (long) rows * width);
        long w = weight(QKGV);
        ops.linear(n1, w, qk, count, h, width, false);
        ops.linear(n1, w + 4L * width * h, gv, count, h, width, false);
        long qkn = buffer(k, "qkn", (long) rows * width),
                qkInv = buffer(k, "qk_inverse", (long) rows * (s.heads + s.kvHeads));
        ops.qkNormRopeForward(
                qk,
                value(QUERY_NORM),
                value(KEY_NORM),
                qkn,
                qkInv,
                count,
                s.heads,
                s.kvHeads,
                d,
                s.rotaryDim,
                a + k - 1,
                s.theta,
                s.eps);
        if (k == 1) {
            kernels.copyColumns(qkn + 4L * mixer, width, keys + 4L * a * kvw, kvw, count, kvw, false);
            kernels.copyColumns(gv + 4L * mixer, width, values + 4L * a * kvw, kvw, count, kvw, false);
        }
        long scores = buffer(k, "probabilities", (long) s.heads * rows * maxKeys);
        long extras = buffer(k, "extra_probabilities", (long) s.heads * rows * Math.max(1, depths - 1));
        int stride = Math.max(1, depths - 1);
        for (int head = 0; head < s.heads; head++) {
            int kv = head / group;
            ops.productTransposed(
                    qkn + 4L * head * d,
                    width,
                    keys + 4L * (firstKey * kvw + kv * d),
                    kvw,
                    scores + 4L * head * count * keyCount,
                    keyCount,
                    count,
                    keyCount,
                    d,
                    1,
                    0);
        }
        for (int j = 2; j <= k; j++)
            kernels.lineageDot(
                    qkn,
                    width,
                    buffer(j, "qkn", 0) + 4L * mixer,
                    width,
                    s.heads,
                    group,
                    d,
                    count,
                    extras,
                    stride,
                    j - 2);
        int visible0 = Math.toIntExact(a - firstKey + 1);
        float scale = (float) (1 / Math.sqrt(d));
        kernels.softmax(scores, keyCount, visible0, extras, stride, k - 1, s.heads, count, scale);
        long context = buffer(k, "context", (long) rows * mixer);
        for (int head = 0; head < s.heads; head++) {
            int kv = head / group;
            ops.product(
                    scores + 4L * head * count * keyCount,
                    keyCount,
                    values + 4L * (firstKey * kvw + kv * d),
                    kvw,
                    context + 4L * head * d,
                    mixer,
                    count,
                    d,
                    keyCount,
                    1,
                    0);
        }
        for (int j = 2; j <= k; j++)
            kernels.lineageSpread(
                    extras,
                    stride,
                    j - 2,
                    buffer(j, "gv", 0) + 4L * mixer,
                    width,
                    context,
                    mixer,
                    s.heads,
                    group,
                    d,
                    count);
        long gated = buffer(k, "gated", (long) rows * mixer);
        ops.gateForward(context, gv, gated, count, mixer, width);
        long h1 = buffer(k, "h1", (long) rows * h);
        ops.copy(c, h1, (long) count * h);
        ops.linear(gated, weight(OUTPUT), h1, count, mixer, h, true);
        long n2 = buffer(k, "n2", (long) rows * h), inv2 = buffer(k, "inv2", rows);
        ops.rmsNormForward(h1, value(POST_NORM), n2, inv2, count, h, s.eps, 1);
        long gu = buffer(k, "gate_up", (long) rows * 2 * s.intermediate);
        ops.linear(n2, weight(GATE_UP), gu, count, h, 2 * s.intermediate, false);
        long act = buffer(k, "swiglu", (long) rows * s.intermediate);
        ops.swigluForward(gu, act, count, s.intermediate);
        long y = buffer(k, "y", (long) rows * h);
        ops.copy(h1, y, (long) count * h);
        ops.linear(act, weight(DOWN), y, count, s.intermediate, h, true);
        long m = buffer(k, "m", (long) rows * h), invF = buffer(k, "final_inverse", rows);
        ops.rmsNormForward(y, value(FINAL_NORM), m, invF, count, h, s.eps, 1);
        // Inputs kept for the backward pass.
        depthBuffers.get(k - 1).put("input_hidden", hidden);
        depthBuffers.get(k - 1).put("input_embedded", embedded);
        return m;
    }

    private long value(String name) {
        return parameters.get(name).values;
    }

    private long gradient(String name) {
        return parameters.get(name).gradient;
    }

    /// The gradient buffer of depth k's output hidden, which its loss and depth k + 1's input add into.
    long outputGradient(int k) {
        return buffer(k, "d_m", (long) rows * shape.hidden);
    }

    /// Zeroes the gradients a window's backward pass accumulates across depths: every depth's output, query-key
    /// and gate-value gradients and the window's key and value gradients.
    void startBackward(long firstKey, long a, int count) {
        int width = shape.qkWidth(), kvw = shape.kvWidth();
        for (int k = 1; k <= depths; k++) {
            ops.zero(outputGradient(k), (long) count * shape.hidden);
            ops.zero(buffer(k, "d_qkn", (long) rows * width), (long) count * width);
            ops.zero(buffer(k, "d_gv", (long) rows * width), (long) count * width);
        }
        long keyCount = a + count - firstKey;
        ops.zero(keyGradients + 4L * firstKey * kvw, keyCount * kvw);
        ops.zero(valueGradients + 4L * firstKey * kvw, keyCount * kvw);
    }

    // ---------------------------------------------------------------- backward

    /// Depth k's backward pass from its output gradient ([#outputGradient], complete once every deeper depth ran
    /// its backward): adds the parameters' gradients, the earlier depths' query-key and gate-value gradients, and
    /// its input hidden's gradient into depth k - 1's output gradient. Run from the deepest depth to depth 1, after
    /// the forward passes of all of them.
    void backward(int k, long a, long firstKey, int count) {
        Shape s = shape;
        int h = s.hidden, d = s.headDim, width = s.qkWidth(), mixer = s.mixerWidth(), kvw = s.kvWidth();
        int group = s.heads / s.kvHeads, keyCount = Math.toIntExact(a + count - firstKey);
        int stride = Math.max(1, depths - 1), inter = s.intermediate;
        long dm = outputGradient(k);
        long dy = shared("d_y", (long) rows * h);
        ops.zero(dy, (long) count * h);
        ops.rmsNormBackward(
                buffer(k, "y", 0),
                value(FINAL_NORM),
                buffer(k, "final_inverse", 0),
                dm,
                dy,
                gradient(FINAL_NORM),
                count,
                h,
                1);
        // MLP.
        long dh1 = shared("d_h1", (long) rows * h);
        ops.copy(dy, dh1, (long) count * h);
        long dAct = shared("d_swiglu", (long) rows * inter);
        ops.zero(dAct, (long) count * inter);
        ops.linearInputGradient(dy, weight(DOWN), dAct, count, inter, h);
        weightGradient(DOWN, 0, h, dy, buffer(k, "swiglu", 0), count);
        long dGu = shared("d_gate_up", (long) rows * 2 * inter);
        ops.zero(dGu, (long) count * 2 * inter);
        ops.swigluBackward(buffer(k, "gate_up", 0), dAct, dGu, count, inter);
        long dn2 = shared("d_n2", (long) rows * h);
        ops.zero(dn2, (long) count * h);
        ops.linearInputGradient(dGu, weight(GATE_UP), dn2, count, h, 2 * inter);
        weightGradient(GATE_UP, 0, 2 * inter, dGu, buffer(k, "n2", 0), count);
        ops.rmsNormBackward(
                buffer(k, "h1", 0), value(POST_NORM), buffer(k, "inv2", 0), dn2, dh1, gradient(POST_NORM), count, h, 1);
        // Attention output and gate.
        long dGated = shared("d_gated", (long) rows * mixer);
        ops.zero(dGated, (long) count * mixer);
        ops.linearInputGradient(dh1, weight(OUTPUT), dGated, count, mixer, h);
        weightGradient(OUTPUT, 0, h, dh1, buffer(k, "gated", 0), count);
        long dContext = shared("d_context", (long) rows * mixer);
        ops.zero(dContext, (long) count * mixer);
        long gv = buffer(k, "gv", 0),
                qkn = buffer(k, "qkn", 0),
                dQkn = buffer(k, "d_qkn", 0),
                dGv = buffer(k, "d_gv", 0);
        ops.gateBackward(buffer(k, "context", 0), gv, dGated, dContext, dGv, count, mixer, width);
        // Softmax attention: dP = dC V^T (causal and lineage), dV += P^T dC.
        long probabilities = buffer(k, "probabilities", 0), extras = buffer(k, "extra_probabilities", 0);
        long dScores = shared("d_scores", (long) s.heads * rows * maxKeys);
        long dExtras = shared("d_extras", (long) s.heads * rows * stride);
        for (int head = 0; head < s.heads; head++) {
            int kv = head / group;
            long p = probabilities + 4L * head * count * keyCount, dS = dScores + 4L * head * count * keyCount;
            ops.productTransposed(
                    dContext + 4L * head * d,
                    mixer,
                    values + 4L * (firstKey * kvw + kv * d),
                    kvw,
                    dS,
                    keyCount,
                    count,
                    keyCount,
                    d,
                    1,
                    0);
            ops.productLeftTransposed(
                    p,
                    keyCount,
                    dContext + 4L * head * d,
                    mixer,
                    valueGradients + 4L * (firstKey * kvw + kv * d),
                    kvw,
                    keyCount,
                    d,
                    count,
                    1,
                    1);
        }
        for (int j = 2; j <= k; j++) {
            kernels.lineageDot(
                    dContext,
                    mixer,
                    buffer(j, "gv", 0) + 4L * mixer,
                    width,
                    s.heads,
                    group,
                    d,
                    count,
                    dExtras,
                    stride,
                    j - 2);
            kernels.lineageGather(
                    extras,
                    stride,
                    j - 2,
                    dContext,
                    mixer,
                    buffer(j, "d_gv", 0) + 4L * mixer,
                    width,
                    s.heads,
                    group,
                    d,
                    count);
        }
        int visible0 = Math.toIntExact(a - firstKey + 1);
        float scale = (float) (1 / Math.sqrt(d));
        kernels.softmaxBackward(
                probabilities, dScores, keyCount, visible0, extras, dExtras, stride, k - 1, s.heads, count, scale);
        // dQ += dS K, dK += dS^T Q (causal and lineage).
        for (int head = 0; head < s.heads; head++) {
            int kv = head / group;
            long dS = dScores + 4L * head * count * keyCount;
            ops.product(
                    dS,
                    keyCount,
                    keys + 4L * (firstKey * kvw + kv * d),
                    kvw,
                    dQkn + 4L * head * d,
                    width,
                    count,
                    d,
                    keyCount,
                    1,
                    1);
            ops.productLeftTransposed(
                    dS,
                    keyCount,
                    qkn + 4L * head * d,
                    width,
                    keyGradients + 4L * (firstKey * kvw + kv * d),
                    kvw,
                    keyCount,
                    d,
                    count,
                    1,
                    1);
        }
        for (int j = 2; j <= k; j++) {
            kernels.lineageSpread(
                    dExtras,
                    stride,
                    j - 2,
                    buffer(j, "qkn", 0) + 4L * mixer,
                    width,
                    dQkn,
                    width,
                    s.heads,
                    group,
                    d,
                    count);
            kernels.lineageGather(
                    dExtras,
                    stride,
                    j - 2,
                    qkn,
                    width,
                    buffer(j, "d_qkn", 0) + 4L * mixer,
                    width,
                    s.heads,
                    group,
                    d,
                    count);
        }
        if (k == 1) {
            // The window's own depth-1 keys and values (every depth's causal part read them).
            kernels.copyColumns(keyGradients + 4L * a * kvw, kvw, dQkn + 4L * mixer, width, count, kvw, true);
            kernels.copyColumns(valueGradients + 4L * a * kvw, kvw, dGv + 4L * mixer, width, count, kvw, true);
        }
        long dQk = shared("d_qk", (long) rows * width);
        ops.zero(dQk, (long) count * width);
        ops.qkNormRopeBackward(
                buffer(k, "qk", 0),
                value(QUERY_NORM),
                value(KEY_NORM),
                buffer(k, "qk_inverse", 0),
                dQkn,
                dQk,
                gradient(QUERY_NORM),
                gradient(KEY_NORM),
                count,
                s.heads,
                s.kvHeads,
                d,
                s.rotaryDim,
                a + k - 1,
                s.theta);
        long n1 = buffer(k, "n1", 0), dn1 = shared("d_n1", (long) rows * h);
        ops.zero(dn1, (long) count * h);
        long w = weight(QKGV);
        ops.linearInputGradient(dQk, w, dn1, count, h, width);
        ops.linearInputGradient(dGv, w + 4L * width * h, dn1, count, h, width);
        weightGradient(QKGV, 0, width, dQk, n1, count);
        weightGradient(QKGV, width, width, dGv, n1, count);
        long dc = shared("d_c", (long) rows * h);
        ops.copy(dh1, dc, (long) count * h);
        ops.rmsNormBackward(
                buffer(k, "c", 0), value(INPUT_NORM), buffer(k, "inv1", 0), dn1, dc, gradient(INPUT_NORM), count, h, 1);
        // Input projection and the two input norms.
        long dCat = shared("d_cat", (long) rows * 2 * h);
        ops.zero(dCat, (long) count * 2 * h);
        ops.linearInputGradient(dc, weight(INPUT_PROJECTION), dCat, count, 2 * h, h);
        weightGradient(INPUT_PROJECTION, 0, h, dc, buffer(k, "cat", 0), count);
        long dPart = shared("d_part", (long) rows * h), discard = shared("d_discard", (long) rows * h);
        kernels.copyColumns(dCat, 2 * h, dPart, h, count, h, false);
        ops.rmsNormBackward(
                depthBuffers.get(k - 1).get("input_embedded"),
                value(EMBEDDING_NORM),
                buffer(k, "embedding_inverse", 0),
                dPart,
                discard,
                gradient(EMBEDDING_NORM),
                count,
                h,
                1);
        kernels.copyColumns(dCat + 4L * h, 2 * h, dPart, h, count, h, false);
        long dHidden = k > 1 ? outputGradient(k - 1) : discard;
        ops.rmsNormBackward(
                depthBuffers.get(k - 1).get("input_hidden"),
                value(HIDDEN_NORM),
                buffer(k, "hidden_inverse", 0),
                dPart,
                dHidden,
                gradient(HIDDEN_NORM),
                count,
                h,
                1);
    }

    /// One AdamW step of every parameter (zeroing the gradients), then the projections' global-scale sources.
    void step(float projectionRate, float normRate, int step) {
        for (Parameter p : parameters.values()) {
            float rate = p.maximum != 0 ? projectionRate : normRate;
            ops.adam(p.values, p.gradient, p.first, p.second, p.count, rate, 0.9f, 0.999f, 1e-8f, 0f, step);
        }
        measure();
    }

    @Override
    public void close() {
        for (long address : owned) gpu.free(address);
        owned.clear();
        for (Parameter p : parameters.values()) p.free(gpu);
    }
}
