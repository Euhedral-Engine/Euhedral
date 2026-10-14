package io.euhedral_execution.inference.quantization.training;

import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/// One Qwen3.8 decoder layer (GDN or full attention) whose quantized linears train their group scales and whose norms
/// train their weights, in FP32 on the training kernels ([Ops]): forward over one sequence keeping what the backward
/// needs, then backward from the output's gradient into the parameters' gradients and the input's.
///
/// A quantized linear keeps its codes (the GPTQ rounding) fixed: its weight is code * scale, so a scale's gradient is
/// the sum over its group of the weight gradient times the code. Everything else (the GDN a/b projections, A_log,
/// dt_bias, convolution) is frozen.
public final class TrainableLayer implements AutoCloseable {

    /// The layer's shapes.
    public record Shape(
            boolean attention,
            int hidden,
            int intermediate,
            int keyHeads,
            int valueHeads,
            int keyDim,
            int valueDim,
            int convKernel,
            int heads,
            int kvHeads,
            int headDim,
            int rotaryDim,
            double theta,
            float eps) {

        public int qkWidth() {
            return attention ? (heads + kvHeads) * headDim : 2 * keyHeads * keyDim;
        }

        public int valueWidth() {
            return valueHeads * valueDim;
        }

        public int convWidth() {
            return qkWidth() + valueWidth();
        }

        public int mixerWidth() {
            return attention ? heads * headDim : valueWidth();
        }
    }

    /// A trainable parameter: master values, gradient and Adam moments (FP32, device). A described parameter's
    /// addresses are bound from outside ([#bind]); only an allocated one is owned and freed with the layer.
    public static final class Parameter {
        public final String name;
        public final long count;
        public long values, gradient, first, second;
        private final boolean allocated;

        private Parameter(String name, long count) {
            this.name = name;
            this.count = count;
            this.allocated = false;
        }

        Parameter(CudaGpuMemory gpu, Ops ops, String name, long count) {
            this.name = name;
            this.count = count;
            this.allocated = true;
            this.values = gpu.allocate(4 * count);
            this.gradient = gpu.allocate(4 * count);
            this.first = gpu.allocate(4 * count);
            this.second = gpu.allocate(4 * count);
            ops.zero(gradient, count);
            ops.zero(first, count);
            ops.zero(second, count);
        }

        void free(CudaGpuMemory gpu) {
            if (allocated) for (long address : new long[] {values, gradient, first, second}) gpu.free(address);
        }
    }

    /// A grouped quantized linear [out, in]: int8 codes and trainable scales [out, in / group].
    public static final class QuantizedLinear {
        public final String role;
        public final int out, in, group;
        public long codes;
        public final Parameter scales;

        QuantizedLinear(String role, int out, int in, int group, long codes, Parameter scales) {
            this.role = role;
            this.out = out;
            this.in = in;
            this.group = group;
            this.codes = codes;
            this.scales = scales;
        }
    }

    private final CudaGpuMemory gpu;
    private final Ops ops;
    public final Shape shape;
    public final Map<String, QuantizedLinear> linears = new LinkedHashMap<>();
    /// Norm weights by role: input_norm, post_norm, and gdn_norm or query_norm and key_norm.
    public final Map<String, Parameter> norms = new LinkedHashMap<>();
    /// Frozen FP32 tensors (GDN): a_projection, b_projection [valueHeads, hidden], a_log, dt_bias [valueHeads],
    /// convolution [convKernel, convWidth].
    private final Map<String, Long> frozen = new LinkedHashMap<>();
    private final List<Long> owned = new ArrayList<>();
    private int rows;
    /// Activations of the last forward pass and gradient buffers, sized for `rows`.
    private final Map<String, Long> buffers = new LinkedHashMap<>();
    private long weight, weightGradient;

    public TrainableLayer(CudaGpuMemory gpu, Ops ops, Shape shape) {
        this.gpu = gpu;
        this.ops = ops;
        this.shape = shape;
    }

    // ---------------------------------------------------------------- parameters

    /// Adds a quantized linear from its int8 codes (device, [out, in]) and FP32 scales (device, copied).
    public void addLinear(String role, int out, int in, int group, long codes, long scales) {
        Parameter parameter = new Parameter(gpu, ops, role + "/scales", (long) out * (in / group));
        gpu.copyDeviceToDevice(parameter.values, scales, 4 * parameter.count);
        linears.put(role, new QuantizedLinear(role, out, in, group, codes, parameter));
    }

    /// Adds a trainable norm weight from FP32 values (device, copied).
    public void addNorm(String role, long values, int width) {
        Parameter parameter = new Parameter(gpu, ops, role, width);
        gpu.copyDeviceToDevice(parameter.values, values, 4L * width);
        norms.put(role, parameter);
    }

    /// Adds a frozen FP32 tensor (device, copied).
    public void addFrozen(String role, long values, long count) {
        long copy = gpu.allocate(4 * count);
        gpu.copyDeviceToDevice(copy, values, 4 * count);
        owned.add(copy);
        frozen.put(role, copy);
    }

    // ---------------------------------------------------------------- bound storage

    /// Declares a linear, norm or frozen tensor whose device storage [#bind] supplies.
    public void describeLinear(String role, int out, int in, int group) {
        linears.put(
                role,
                new QuantizedLinear(
                        role, out, in, group, 0, new Parameter(role + "/scales", (long) out * (in / group))));
    }

    public void describeNorm(String role, int width) {
        norms.put(role, new Parameter(role, width));
    }

    public void describeFrozen(String role) {
        frozen.put(role, 0L);
    }

    /// Where a layer's parameters are on the device: codes and scales of each linear, norms, frozen tensors, and the
    /// gradients (a layer's slices of the accumulated gradients) by parameter name.
    public record Binding(
            Map<String, Long> codes, Map<String, Long> values, Map<String, Long> gradients, Map<String, Long> frozen) {}

    /// Computes with the storage of `binding` from now on (its kernels read it when they run).
    public void bind(Binding binding) {
        for (QuantizedLinear linear : linears.values()) {
            linear.codes = binding.codes().get(linear.role);
            linear.scales.values = binding.values().get(linear.scales.name);
            linear.scales.gradient = binding.gradients().get(linear.scales.name);
        }
        for (Parameter norm : norms.values()) {
            norm.values = binding.values().get(norm.name);
            norm.gradient = binding.gradients().get(norm.name);
        }
        for (String role : List.copyOf(frozen.keySet()))
            frozen.put(role, binding.frozen().get(role));
    }

    /// Uses `weight` and `weightGradient` (FP32, each the largest linear) instead of allocating its own: layers that
    /// never run at once can share them.
    public void shareWeights(long weight, long weightGradient) {
        this.weight = weight;
        this.weightGradient = weightGradient;
    }

    // ---------------------------------------------------------------- reloadable storage

    /// Adds a quantized linear's device storage (codes and scales) to be filled by [#loadLinear]; the layer owns it.
    public void addLinearStorage(String role, int out, int in, int group) {
        long codes = gpu.allocate((long) out * in);
        owned.add(codes);
        linears.put(
                role,
                new QuantizedLinear(
                        role,
                        out,
                        in,
                        group,
                        codes,
                        new Parameter(gpu, ops, role + "/scales", (long) out * (in / group))));
    }

    public void addNormStorage(String role, int width) {
        norms.put(role, new Parameter(gpu, ops, role, width));
    }

    public void addFrozenStorage(String role, long count) {
        long address = gpu.allocate(4 * count);
        owned.add(address);
        frozen.put(role, address);
    }

    /// Fills a linear from host memory: int8 codes, and FP32 scales with their Adam moments; zeroes its gradient.
    public void loadLinear(
            String role,
            java.lang.foreign.MemorySegment codes,
            java.lang.foreign.MemorySegment scales,
            java.lang.foreign.MemorySegment first,
            java.lang.foreign.MemorySegment second) {
        QuantizedLinear linear = linears.get(role);
        gpu.copyHostToDevice(linear.codes, codes, (long) linear.out * linear.in);
        load(linear.scales, scales, first, second);
    }

    public void loadNorm(
            String role,
            java.lang.foreign.MemorySegment values,
            java.lang.foreign.MemorySegment first,
            java.lang.foreign.MemorySegment second) {
        load(norms.get(role), values, first, second);
    }

    public void loadFrozen(String role, java.lang.foreign.MemorySegment values) {
        gpu.copyHostToDevice(frozen.get(role), values, values.byteSize());
    }

    private void load(
            Parameter parameter,
            java.lang.foreign.MemorySegment values,
            java.lang.foreign.MemorySegment first,
            java.lang.foreign.MemorySegment second) {
        long bytes = 4 * parameter.count;
        gpu.copyHostToDevice(parameter.values, values, bytes);
        gpu.copyHostToDevice(parameter.first, first, bytes);
        gpu.copyHostToDevice(parameter.second, second, bytes);
        ops.zero(parameter.gradient, parameter.count);
    }

    /// Copies a parameter's values and Adam moments back to host memory.
    public void store(
            Parameter parameter,
            java.lang.foreign.MemorySegment values,
            java.lang.foreign.MemorySegment first,
            java.lang.foreign.MemorySegment second) {
        long bytes = 4 * parameter.count;
        gpu.copyDeviceToHost(values, parameter.values, bytes);
        gpu.copyDeviceToHost(first, parameter.first, bytes);
        gpu.copyDeviceToHost(second, parameter.second, bytes);
    }

    public List<Parameter> parameters() {
        List<Parameter> all = new ArrayList<>();
        for (QuantizedLinear linear : linears.values()) all.add(linear.scales);
        all.addAll(norms.values());
        return all;
    }

    // ---------------------------------------------------------------- buffers

    private long buffer(String name, long floats) {
        return buffers.computeIfAbsent(name, n -> {
            long address = gpu.allocate(4 * Math.max(1, floats));
            return address;
        });
    }

    /// Sizes the activation and gradient buffers for sequences of `rows` rows.
    public void prepare(int rows) {
        if (rows == this.rows) return;
        for (long address : buffers.values()) gpu.free(address);
        buffers.clear();
        this.rows = rows;
        long largest = 0;
        for (QuantizedLinear linear : linears.values()) largest = Math.max(largest, (long) linear.out * linear.in);
        if (weight == 0) {
            weight = gpu.allocate(4 * largest);
            weightGradient = gpu.allocate(4 * largest);
            owned.add(weight);
            owned.add(weightGradient);
        }
    }

    private long w(String role) {
        QuantizedLinear linear = linears.get(role);
        ops.dequantize(linear.codes, linear.scales.values, weight, linear.out, linear.in, linear.group);
        return weight;
    }

    /// y += x w(role)^T or y = x w^T.
    private void apply(String role, long x, long y, boolean accumulate) {
        QuantizedLinear linear = linears.get(role);
        ops.linear(x, w(role), y, rows, linear.in, linear.out, accumulate);
    }

    /// dx += dy w(role); the scales' gradient from dw = dy^T x.
    private void applyBackward(String role, long x, long dy, long dx) {
        QuantizedLinear linear = linears.get(role);
        ops.linearInputGradient(dy, w(role), dx, rows, linear.in, linear.out);
        ops.linearWeightGradient(dy, x, weightGradient, rows, linear.in, linear.out);
        ops.scaleGradient(linear.codes, weightGradient, linear.scales.gradient, linear.out, linear.in, linear.group);
    }

    // ---------------------------------------------------------------- forward

    /// y [rows, hidden] = the layer over x [rows, hidden] (FP32, device), positions from `position0`.
    public void forward(long x, long y, int rows, long position0) {
        prepare(rows);
        Shape s = shape;
        int h = s.hidden;
        long n1 = buffer("n1", (long) rows * h), inv1 = buffer("inv1", rows);
        ops.rmsNormForward(x, norms.get("input_norm").values, n1, inv1, rows, h, s.eps, 1);
        long mixed = buffer("mixed", (long) rows * h);
        if (s.attention) attentionForward(n1, mixed, position0);
        else gdnForward(n1, mixed);
        long h1 = buffer("h1", (long) rows * h);
        ops.copy(x, h1, (long) rows * h);
        ops.axpby(mixed, h1, (long) rows * h, 1, 1);
        long n2 = buffer("n2", (long) rows * h), inv2 = buffer("inv2", rows);
        ops.rmsNormForward(h1, norms.get("post_norm").values, n2, inv2, rows, h, s.eps, 1);
        long gu = buffer("gate_up", (long) rows * 2 * s.intermediate);
        apply("mlp/gate_up", n2, gu, false);
        long act = buffer("swiglu", (long) rows * s.intermediate);
        ops.swigluForward(gu, act, rows, s.intermediate);
        ops.copy(h1, y, (long) rows * h);
        apply("mlp/down", act, y, true);
    }

    private void gdnForward(long n1, long mixed) {
        Shape s = shape;
        int vh = s.valueHeads;
        long qk = buffer("qk", (long) rows * s.qkWidth()), vz = buffer("vz", (long) rows * 2 * s.valueWidth());
        apply("gdn/query_key", n1, qk, false);
        apply("gdn/value_z", n1, vz, false);
        long a = buffer("a", (long) rows * vh), b = buffer("b", (long) rows * vh);
        ops.linear(n1, frozen.get("a_projection"), a, rows, s.hidden, vh, false);
        ops.linear(n1, frozen.get("b_projection"), b, rows, s.hidden, vh, false);
        long alpha = buffer("alpha", (long) rows * vh), beta = buffer("beta", (long) rows * vh);
        ops.gdnControlForward(a, b, frozen.get("a_log"), frozen.get("dt_bias"), alpha, beta, rows, vh);
        long pre = buffer("conv_pre", (long) rows * s.convWidth()), conv = buffer("conv", (long) rows * s.convWidth());
        ops.gdnConvolutionForward(
                qk, vz, frozen.get("convolution"), pre, conv, rows, s.qkWidth(), s.valueWidth(), s.convKernel);
        long o = buffer("recurrent", (long) rows * s.valueWidth());
        long checkpoints = buffer("checkpoints", Ops.checkpointFloats(rows, vh, s.keyDim, s.valueDim, SEGMENT));
        ops.gdnRecurrenceForward(
                conv, alpha, beta, o, checkpoints, rows, s.keyHeads, vh, s.keyDim, s.valueDim, SEGMENT);
        long gated = buffer("gated", (long) rows * s.valueWidth()), inv = buffer("gated_inverse", (long) rows * vh);
        ops.gdnGatedNormForward(o, vz, norms.get("gdn_norm").values, gated, inv, rows, vh, s.valueDim, s.eps);
        apply("gdn/output", gated, mixed, false);
    }

    /// Rows per GDN recurrence checkpoint.
    static final int SEGMENT = 64;

    private void attentionForward(long n1, long mixed, long position0) {
        Shape s = shape;
        int d = s.headDim, width = s.qkWidth(), group = s.heads / s.kvHeads;
        long qk = buffer("qk", (long) rows * width), gv = buffer("gv", (long) rows * width);
        apply("attention/query_key", n1, qk, false);
        apply("attention/gate_value", n1, gv, false);
        long qkn = buffer("qkn", (long) rows * width), inv = buffer("qk_inverse", (long) rows * (s.heads + s.kvHeads));
        ops.qkNormRopeForward(
                qk,
                norms.get("query_norm").values,
                norms.get("key_norm").values,
                qkn,
                inv,
                rows,
                s.heads,
                s.kvHeads,
                d,
                s.rotaryDim,
                position0,
                s.theta,
                s.eps);
        long probabilities = buffer("probabilities", (long) s.heads * rows * rows);
        long context = buffer("context", (long) rows * s.heads * d);
        for (int head = 0; head < s.heads; head++) {
            int kv = head / group;
            long p = probabilities + 4L * head * rows * rows;
            ops.productTransposed(
                    qkn + 4L * head * d, width, qkn + 4L * (s.heads + kv) * d, width, p, rows, rows, rows, d, 1, 0);
            ops.causalSoftmax(p, 1, rows, (float) (1 / Math.sqrt(d)));
            ops.product(
                    p,
                    rows,
                    gv + 4L * (s.heads + kv) * d,
                    width,
                    context + 4L * head * d,
                    s.heads * d,
                    rows,
                    d,
                    rows,
                    1,
                    0);
        }
        long gated = buffer("gated", (long) rows * s.heads * d);
        ops.gateForward(context, gv, gated, rows, s.heads * d, width);
        apply("attention/output", gated, mixed, false);
    }

    // ---------------------------------------------------------------- backward

    /// From dy [rows, hidden] (the output's gradient): adds the parameters' gradients, and the input's into dx (when
    /// not 0). Needs the activations of the forward pass just before.
    public void backward(long x, long dy, long dx, long position0) {
        Shape s = shape;
        int h = s.hidden;
        long dh1 = buffer("d_h1", (long) rows * h);
        ops.copy(dy, dh1, (long) rows * h);
        long dAct = buffer("d_swiglu", (long) rows * s.intermediate);
        ops.zero(dAct, (long) rows * s.intermediate);
        applyBackward("mlp/down", buffer("swiglu", 0), dy, dAct);
        long dGu = buffer("d_gate_up", (long) rows * 2 * s.intermediate);
        ops.zero(dGu, (long) rows * 2 * s.intermediate);
        ops.swigluBackward(buffer("gate_up", 0), dAct, dGu, rows, s.intermediate);
        long dn2 = buffer("d_n2", (long) rows * h);
        ops.zero(dn2, (long) rows * h);
        applyBackward("mlp/gate_up", buffer("n2", 0), dGu, dn2);
        ops.rmsNormBackward(
                buffer("h1", 0),
                norms.get("post_norm").values,
                buffer("inv2", 0),
                dn2,
                dh1,
                norms.get("post_norm").gradient,
                rows,
                h,
                1);
        long dn1 = buffer("d_n1", (long) rows * h);
        ops.zero(dn1, (long) rows * h);
        if (s.attention) attentionBackward(dh1, dn1, position0);
        else gdnBackward(dh1, dn1);
        long dxTarget = dx != 0 ? dx : buffer("d_x", (long) rows * h);
        ops.copy(dh1, dxTarget, (long) rows * h);
        ops.rmsNormBackward(
                x,
                norms.get("input_norm").values,
                buffer("inv1", 0),
                dn1,
                dxTarget,
                norms.get("input_norm").gradient,
                rows,
                h,
                1);
    }

    private void gdnBackward(long dMixed, long dn1) {
        Shape s = shape;
        int vh = s.valueHeads;
        long n1 = buffer("n1", 0), vz = buffer("vz", 0);
        long dGated = buffer("d_gated", (long) rows * s.valueWidth());
        ops.zero(dGated, (long) rows * s.valueWidth());
        applyBackward("gdn/output", buffer("gated", 0), dMixed, dGated);
        long dO = buffer("d_recurrent", (long) rows * s.valueWidth());
        long dVz = buffer("d_vz", (long) rows * 2 * s.valueWidth());
        ops.zero(dO, (long) rows * s.valueWidth());
        ops.zero(dVz, (long) rows * 2 * s.valueWidth());
        ops.gdnGatedNormBackward(
                buffer("recurrent", 0),
                vz,
                norms.get("gdn_norm").values,
                buffer("gated_inverse", 0),
                dGated,
                dO,
                dVz,
                norms.get("gdn_norm").gradient,
                rows,
                vh,
                s.valueDim);
        long dConv = buffer("d_conv", (long) rows * s.convWidth());
        long dAlpha = buffer("d_alpha", (long) rows * vh), dBeta = buffer("d_beta", (long) rows * vh);
        ops.zero(dConv, (long) rows * s.convWidth());
        long scratch =
                buffer("recurrence_scratch", Ops.recurrenceScratchFloats(rows, vh, s.keyDim, s.valueDim, SEGMENT));
        ops.gdnRecurrenceBackward(
                buffer("conv", 0),
                buffer("alpha", 0),
                buffer("beta", 0),
                dO,
                buffer("checkpoints", 0),
                scratch,
                dConv,
                dAlpha,
                dBeta,
                rows,
                s.keyHeads,
                vh,
                s.keyDim,
                s.valueDim,
                SEGMENT);
        long dQk = buffer("d_qk", (long) rows * s.qkWidth());
        ops.zero(dQk, (long) rows * s.qkWidth());
        ops.gdnConvolutionBackward(
                buffer("conv_pre", 0),
                dConv,
                frozen.get("convolution"),
                dQk,
                dVz,
                rows,
                s.qkWidth(),
                s.valueWidth(),
                s.convKernel);
        long da = buffer("d_a", (long) rows * vh), db = buffer("d_b", (long) rows * vh);
        ops.zero(da, (long) rows * vh);
        ops.zero(db, (long) rows * vh);
        ops.gdnControlBackward(
                buffer("a", 0),
                frozen.get("a_log"),
                frozen.get("dt_bias"),
                buffer("alpha", 0),
                buffer("beta", 0),
                dAlpha,
                dBeta,
                da,
                db,
                rows,
                vh);
        ops.linearInputGradient(da, frozen.get("a_projection"), dn1, rows, s.hidden, vh);
        ops.linearInputGradient(db, frozen.get("b_projection"), dn1, rows, s.hidden, vh);
        applyBackward("gdn/query_key", n1, dQk, dn1);
        applyBackward("gdn/value_z", n1, dVz, dn1);
    }

    private void attentionBackward(long dMixed, long dn1, long position0) {
        Shape s = shape;
        int d = s.headDim, width = s.qkWidth(), group = s.heads / s.kvHeads;
        long gv = buffer("gv", 0), qkn = buffer("qkn", 0), context = buffer("context", 0);
        long probabilities = buffer("probabilities", 0);
        long dGated = buffer("d_gated", (long) rows * s.heads * d);
        ops.zero(dGated, (long) rows * s.heads * d);
        applyBackward("attention/output", buffer("gated", 0), dMixed, dGated);
        long dContext = buffer("d_context", (long) rows * s.heads * d), dGv = buffer("d_gv", (long) rows * width);
        ops.zero(dContext, (long) rows * s.heads * d);
        ops.zero(dGv, (long) rows * width);
        ops.gateBackward(context, gv, dGated, dContext, dGv, rows, s.heads * d, width);
        long dQkn = buffer("d_qkn", (long) rows * width), dScores = buffer("d_scores", (long) rows * rows);
        ops.zero(dQkn, (long) rows * width);
        for (int head = 0; head < s.heads; head++) {
            int kv = head / group;
            long p = probabilities + 4L * head * rows * rows;
            long q = qkn + 4L * head * d, k = qkn + 4L * (s.heads + kv) * d, v = gv + 4L * (s.heads + kv) * d;
            long dC = dContext + 4L * head * d;
            // dP = dC V^T, dV += P^T dC, dS = softmax'(dP), dQ = dS K, dK += dS^T Q.
            ops.productTransposed(dC, s.heads * d, v, width, dScores, rows, rows, rows, d, 1, 0);
            ops.productLeftTransposed(
                    p, rows, dC, s.heads * d, dGv + 4L * (s.heads + kv) * d, width, rows, d, rows, 1, 1);
            ops.causalSoftmaxBackward(p, dScores, 1, rows, (float) (1 / Math.sqrt(d)));
            ops.product(dScores, rows, k, width, dQkn + 4L * head * d, width, rows, d, rows, 1, 1);
            ops.productLeftTransposed(
                    dScores, rows, q, width, dQkn + 4L * (s.heads + kv) * d, width, rows, d, rows, 1, 1);
        }
        long dQk = buffer("d_qk", (long) rows * width);
        ops.zero(dQk, (long) rows * width);
        ops.qkNormRopeBackward(
                buffer("qk", 0),
                norms.get("query_norm").values,
                norms.get("key_norm").values,
                buffer("qk_inverse", 0),
                dQkn,
                dQk,
                norms.get("query_norm").gradient,
                norms.get("key_norm").gradient,
                rows,
                s.heads,
                s.kvHeads,
                d,
                s.rotaryDim,
                position0,
                s.theta);
        long n1 = buffer("n1", 0);
        applyBackward("attention/query_key", n1, dQk, dn1);
        applyBackward("attention/gate_value", n1, dGv, dn1);
    }

    @Override
    public void close() {
        for (long address : buffers.values()) gpu.free(address);
        buffers.clear();
        for (long address : owned) gpu.free(address);
        owned.clear();
        for (QuantizedLinear linear : linears.values()) linear.scales.free(gpu);
        for (Parameter parameter : norms.values()) parameter.free(gpu);
    }
}
