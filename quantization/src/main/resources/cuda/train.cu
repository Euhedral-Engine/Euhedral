// Block-wise training of a quantized decoder layer (quantization/README.md): forward and backward passes of the
// layer's operations in FP32, with the engine's semantics (native/src/gdn, native/src/attention, norm, elementwise):
// - RMSNorm, (1 + w) form for layer and q/k norms, w form for the GDN gated norm;
// - GDN: alpha = exp(-exp(A_log) softplus(a + dt_bias)), beta = sigmoid(b); a causal depthwise convolution with SiLU;
//   the gated delta rule S_t = alpha S_{t-1} + beta (v - alpha S_{t-1} k) k^T over L2-normalized keys, o = S q / sqrt(d);
//   the gated RMSNorm o * rsqrt(mean o^2) * w * silu(z) per value head;
// - attention: per-head RMSNorm and half-split RoPE on the first `rotary` dimensions, causal grouped-query softmax
//   attention, output gated by sigmoid(gate);
// - SwiGLU over [gate; up] rows.
// Rows are tokens of one sequence (row-major). Backward kernels accumulate (+=) into input gradients unless a name
// says otherwise, so a tensor used twice sums both contributions.

namespace {

constexpr int kThreads = 256;

__device__ __forceinline__ float silu(float x) { return x / (1.0f + expf(-x)); }
__device__ __forceinline__ float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ float silu_grad(float x) {
    float s = sigmoidf(x);
    return s * (1.0f + x * (1.0f - s));
}

// Block-wide sum over blockDim.x (a multiple of 32, at most 1024) threads; every thread gets the total.
__device__ float block_sum(float value, float* shared) {
    for (int offset = 16; offset > 0; offset >>= 1) value += __shfl_down_sync(0xffffffffu, value, offset);
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32, warps = blockDim.x / 32;
    __syncthreads();
    if (lane == 0) shared[warp] = value;
    __syncthreads();
    if (warp == 0) {
        value = lane < warps ? shared[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) value += __shfl_down_sync(0xffffffffu, value, offset);
        if (lane == 0) shared[0] = value;
    }
    __syncthreads();
    float total = shared[0];
    __syncthreads();
    return total;
}

}  // namespace

// ---------------------------------------------------------------------------------------------------------------
// Elementwise.

extern "C" __global__ void bf16_to_float(const unsigned short* in, long long count, float* out) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) out[i] = __uint_as_float((unsigned int)in[i] << 16);
}

extern "C" __global__ void float_to_bf16_rounded(const float* in, long long count, unsigned short* out) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    unsigned int bits = __float_as_uint(in[i]);
    out[i] = (unsigned short)((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

extern "C" __global__ void copy_floats(const float* x, float* y, long long count) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) y[i] = x[i];
}

// y = a * x + b * y.
extern "C" __global__ void axpby(const float* x, float* y, long long count, float a, float b) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) y[i] = a * x[i] + b * y[i];
}

extern "C" __global__ void fill(float* x, long long count, float value) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) x[i] = value;
}

// h[r, j] = silu(gate[r, j]) * up[r, j] with gateUp rows [gate (width); up (width)].
extern "C" __global__ void swiglu_forward(const float* gateUp, float* h, int rows, int width) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)rows * width) return;
    long long r = i / width, j = i % width;
    float gate = gateUp[r * 2 * width + j], up = gateUp[r * 2 * width + width + j];
    h[i] = silu(gate) * up;
}

extern "C" __global__ void swiglu_backward(const float* gateUp, const float* dh, float* dGateUp, int rows, int width) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)rows * width) return;
    long long r = i / width, j = i % width;
    float gate = gateUp[r * 2 * width + j], up = gateUp[r * 2 * width + width + j];
    dGateUp[r * 2 * width + j] += dh[i] * up * silu_grad(gate);
    dGateUp[r * 2 * width + width + j] += dh[i] * silu(gate);
}

// ---------------------------------------------------------------------------------------------------------------
// RMSNorm per row of `width`: y = x * inv * (offset + w), inv = rsqrt(mean(x^2) + eps). One block per row.

extern "C" __global__ void rms_norm_forward(const float* x, const float* w, float* y, float* inverse, int width,
                                            float eps, float offset) {
    __shared__ float shared[32];
    const float* row = x + (long long)blockIdx.x * width;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < width; i += blockDim.x) sum += row[i] * row[i];
    float inv = rsqrtf(block_sum(sum, shared) / width + eps);
    if (threadIdx.x == 0) inverse[blockIdx.x] = inv;
    for (int i = threadIdx.x; i < width; i += blockDim.x) y[(long long)blockIdx.x * width + i] = row[i] * inv * (offset + w[i]);
}

// dx += inv * (g - x * inv^2 * mean(g * x)) with g = dy * (offset + w); dw[i] += dy * x * inv (when dw is not null).
extern "C" __global__ void rms_norm_backward(const float* x, const float* w, const float* inverse, const float* dy,
                                             float* dx, float* dw, int width, float offset) {
    __shared__ float shared[32];
    long long base = (long long)blockIdx.x * width;
    float inv = inverse[blockIdx.x];
    float dot = 0.0f;
    for (int i = threadIdx.x; i < width; i += blockDim.x) dot += dy[base + i] * (offset + w[i]) * x[base + i];
    float mean = block_sum(dot, shared) / width;
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        float g = dy[base + i] * (offset + w[i]);
        dx[base + i] += inv * (g - x[base + i] * inv * inv * mean);
        if (dw != nullptr) atomicAdd(dw + i, dy[base + i] * x[base + i] * inv);
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Grouped quantized weights: w[n, k] = code[n, k] * scale[n, k / group].

extern "C" __global__ void dequantize_grouped(const signed char* codes, const float* scales, float* w, long long n,
                                              int k, int group) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * k) return;
    long long row = i / k;
    int column = (int)(i % k);
    w[i] = (float)codes[i] * scales[row * (k / group) + column / group];
}

// dscale[n, g] += sum over the group's columns of dw * code. One warp per group.
extern "C" __global__ void scale_gradient(const signed char* codes, const float* dw, float* dscales, long long groups,
                                          int k, int group) {
    long long g = ((long long)blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    if (g >= groups) return;
    long long row = g / (k / group);
    long long first = row * k + (g % (k / group)) * group;
    float sum = 0.0f;
    for (int j = lane; j < group; j += 32) sum += dw[first + j] * (float)codes[first + j];
    for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0) dscales[g] += sum;
}

// ---------------------------------------------------------------------------------------------------------------
// GDN control: alpha = exp(-exp(aLog[h]) * softplus(a + dtBias[h])), beta = sigmoid(b); a, b, alpha, beta [rows, heads].

extern "C" __global__ void gdn_control_forward(const float* a, const float* b, const float* aLog, const float* dtBias,
                                               float* alpha, float* beta, int rows, int heads) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)rows * heads) return;
    int h = (int)(i % heads);
    float shifted = a[i] + dtBias[h];
    float softplus = fmaxf(shifted, 0.0f) + log1pf(expf(-fabsf(shifted)));
    alpha[i] = expf(-expf(aLog[h]) * softplus);
    beta[i] = sigmoidf(b[i]);
}

extern "C" __global__ void gdn_control_backward(const float* a, const float* aLog, const float* dtBias,
                                                const float* alpha, const float* beta, const float* dAlpha,
                                                const float* dBeta, float* da, float* db, int rows, int heads) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)rows * heads) return;
    int h = (int)(i % heads);
    da[i] += dAlpha[i] * alpha[i] * -expf(aLog[h]) * sigmoidf(a[i] + dtBias[h]);
    db[i] += dBeta[i] * beta[i] * (1.0f - beta[i]);
}

// ---------------------------------------------------------------------------------------------------------------
// GDN convolution from zero state: out[t, c] = silu(sum_tap w[tap, c] * in[t - kernel + 1 + tap, c]) where in is
// [queryKey (qkWidth); value (the first valueWidth of each valueZ row)]; pre keeps the sums.

__device__ __forceinline__ float conv_input(const float* qk, const float* vz, long long t, int c, int qkWidth,
                                            int valueWidth) {
    return c < qkWidth ? qk[t * qkWidth + c] : vz[t * 2 * valueWidth + c - qkWidth];
}

extern "C" __global__ void gdn_convolution_forward(const float* qk, const float* vz, const float* w, float* pre,
                                                   float* out, int rows, int qkWidth, int valueWidth, int kernel) {
    int width = qkWidth + valueWidth;
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)rows * width) return;
    long long t = i / width;
    int c = (int)(i % width);
    float sum = 0.0f;
    for (int tap = 0; tap < kernel; tap++) {
        long long source = t - kernel + 1 + tap;
        if (source >= 0) sum += w[tap * width + c] * conv_input(qk, vz, source, c, qkWidth, valueWidth);
    }
    pre[i] = sum;
    out[i] = silu(sum);
}

// d(input)[s, c] += sum over taps of dpre[s + kernel - 1 - tap, c] * w[tap, c], dpre = dout * silu'(pre).
extern "C" __global__ void gdn_convolution_backward(const float* pre, const float* dout, const float* w, float* dqk,
                                                    float* dvz, int rows, int qkWidth, int valueWidth, int kernel) {
    int width = qkWidth + valueWidth;
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)rows * width) return;
    long long s = i / width;
    int c = (int)(i % width);
    float sum = 0.0f;
    for (int tap = 0; tap < kernel; tap++) {
        long long t = s + kernel - 1 - tap;
        if (t < rows) sum += dout[t * width + c] * silu_grad(pre[t * width + c]) * w[tap * width + c];
    }
    if (c < qkWidth) dqk[s * qkWidth + c] += sum;
    else dvz[s * 2 * valueWidth + c - qkWidth] += sum;
}

// ---------------------------------------------------------------------------------------------------------------
// GDN gated delta rule over one sequence from zero state. convolved rows: [q (keyHeads x dk); k (keyHeads x dk);
// v (valueHeads x dv)]; value head h reads key head h / (valueHeads / keyHeads). One block per value head holds the
// state S [dv, dk] in dynamic shared memory (dv dk + 2 dk + dv floats); blockDim.x is any multiple of 32. Every `segment` rows the state before the row is saved to checkpoints
// [valueHeads, rows / segment + 1, dv, dk] for the backward pass.

__device__ void load_normalized(const float* source, float* target, int dk, float* shared) {
    float sum = 0.0f;
    for (int j = threadIdx.x; j < dk; j += blockDim.x) sum += source[j] * source[j];
    float inv = rsqrtf(block_sum(sum, shared) + 1.0e-6f);
    for (int j = threadIdx.x; j < dk; j += blockDim.x) target[j] = source[j] * inv;
    __syncthreads();
}

extern "C" __global__ void gdn_recurrence_forward(const float* convolved, const float* alpha, const float* beta,
                                                  float* output, float* checkpoints, int rows, int keyHeads,
                                                  int valueHeads, int dk, int dv, int segment, float outputScale) {
    extern __shared__ float memory[];
    __shared__ float reduce[32];
    float* S = memory;
    float* q = S + dv * dk;
    float* k = q + dk;
    float* u = k + dk;
    int h = blockIdx.x, kh = h / (valueHeads / keyHeads);
    int qkWidth = 2 * keyHeads * dk, width = qkWidth + valueHeads * dv;
    int checkpointCount = rows / segment + 1;
    for (int i = threadIdx.x; i < dv * dk; i += blockDim.x) S[i] = 0.0f;
    __syncthreads();
    for (int t = 0; t < rows; t++) {
        if (t % segment == 0) {
            float* saved = checkpoints + ((long long)h * checkpointCount + t / segment) * dv * dk;
            for (int i = threadIdx.x; i < dv * dk; i += blockDim.x) saved[i] = S[i];
        }
        const float* row = convolved + (long long)t * width;
        load_normalized(row + kh * dk, q, dk, reduce);
        load_normalized(row + keyHeads * dk + kh * dk, k, dk, reduce);
        float a = alpha[(long long)t * valueHeads + h], b = beta[(long long)t * valueHeads + h];
        for (int c = threadIdx.x; c < dv; c += blockDim.x) {
            float dot = 0.0f;
            for (int j = 0; j < dk; j++) dot += S[c * dk + j] * k[j];
            u[c] = dot;
        }
        __syncthreads();
        for (int i = threadIdx.x; i < dv * dk; i += blockDim.x) {
            int c = i / dk, j = i % dk;
            float delta = b * (row[qkWidth + h * dv + c] - a * u[c]);
            S[i] = a * S[i] + delta * k[j];
        }
        __syncthreads();
        for (int c = threadIdx.x; c < dv; c += blockDim.x) {
            float dot = 0.0f;
            for (int j = 0; j < dk; j++) dot += S[c * dk + j] * q[j];
            output[(long long)t * valueHeads * dv + h * dv + c] = dot * outputScale;
        }
        __syncthreads();
    }
}

// Backward of gdn_recurrence_forward. dQK [rows, valueHeads, 2 dk] receives each value head's gradients of its raw
// (unnormalized) q and k, summed over a key head's value heads afterwards (gdn_gather_query_key); the value gradient is
// added to dConvolved's value columns; dAlpha, dBeta [rows, valueHeads] are written. The state's adjoint stays in
// dynamic shared memory (dv dk + 4 dk + 4 dv floats); `states` [valueHeads, segment, dv, dk] (global) takes the states
// before each row of a segment, recomputed from its checkpoint. S_t itself is never needed: the query's gradient
// s S_t^T dO is s (alpha S_{t-1}^T dO + k (delta . dO)).
extern "C" __global__ void gdn_recurrence_backward(const float* convolved, const float* alpha, const float* beta,
                                                   const float* dOutput, const float* checkpoints, float* states,
                                                   float* dQK, float* dConvolved, float* dAlpha, float* dBeta,
                                                   int rows, int keyHeads, int valueHeads, int dk, int dv, int segment,
                                                   float outputScale) {
    extern __shared__ float memory[];
    __shared__ float reduce[32];
    float* dS = memory;  // adjoint of the state after the current row
    float* q = dS + dv * dk;
    float* k = q + dk;
    float* dq = k + dk;
    float* dkey = dq + dk;
    float* u = dkey + dk;
    float* delta = u + dv;
    float* dDelta = delta + dv;
    float* dO = dDelta + dv;
    int h = blockIdx.x, kh = h / (valueHeads / keyHeads);
    int qkWidth = 2 * keyHeads * dk, width = qkWidth + valueHeads * dv;
    int checkpointCount = rows / segment + 1;
    long long area = (long long)dv * dk;
    float* segmentStates = states + (long long)h * segment * area;
    for (int i = threadIdx.x; i < area; i += blockDim.x) dS[i] = 0.0f;
    __syncthreads();
    for (int first = (rows - 1) / segment * segment; first >= 0; first -= segment) {
        int last = min(rows, first + segment);
        // The states before each row of the segment, from its checkpoint.
        const float* saved = checkpoints + ((long long)h * checkpointCount + first / segment) * area;
        for (int i = threadIdx.x; i < area; i += blockDim.x) segmentStates[i] = saved[i];
        __syncthreads();
        for (int t = first; t + 1 < last; t++) {
            const float* current = segmentStates + (long long)(t - first) * area;
            float* next = segmentStates + (long long)(t - first + 1) * area;
            const float* row = convolved + (long long)t * width;
            load_normalized(row + keyHeads * dk + kh * dk, k, dk, reduce);
            float a = alpha[(long long)t * valueHeads + h], b = beta[(long long)t * valueHeads + h];
            for (int c = threadIdx.x; c < dv; c += blockDim.x) {
                float dot = 0.0f;
                for (int j = 0; j < dk; j++) dot += current[c * dk + j] * k[j];
                u[c] = dot;
            }
            __syncthreads();
            for (int i = threadIdx.x; i < area; i += blockDim.x) {
                int c = i / dk, j = i % dk;
                next[i] = a * current[i] + b * (row[qkWidth + h * dv + c] - a * u[c]) * k[j];
            }
            __syncthreads();
        }
        for (int t = last - 1; t >= first; t--) {
            const float* row = convolved + (long long)t * width;
            const float* before = segmentStates + (long long)(t - first) * area;
            const float* rawQ = row + kh * dk;
            const float* rawK = row + keyHeads * dk + kh * dk;
            float a = alpha[(long long)t * valueHeads + h], b = beta[(long long)t * valueHeads + h];
            float sumQ = 0.0f, sumK = 0.0f;
            for (int j = threadIdx.x; j < dk; j += blockDim.x) {
                sumQ += rawQ[j] * rawQ[j];
                sumK += rawK[j] * rawK[j];
            }
            float normQ = sqrtf(block_sum(sumQ, reduce) + 1.0e-6f);
            float normK = sqrtf(block_sum(sumK, reduce) + 1.0e-6f);
            for (int j = threadIdx.x; j < dk; j += blockDim.x) {
                q[j] = rawQ[j] / normQ;
                k[j] = rawK[j] / normK;
            }
            for (int c = threadIdx.x; c < dv; c += blockDim.x) dO[c] = dOutput[(long long)t * valueHeads * dv + h * dv + c];
            __syncthreads();
            float partialDeltaO = 0.0f;
            for (int c = threadIdx.x; c < dv; c += blockDim.x) {
                float dot = 0.0f;
                for (int j = 0; j < dk; j++) dot += before[c * dk + j] * k[j];
                u[c] = dot;
                delta[c] = b * (row[qkWidth + h * dv + c] - a * dot);
                partialDeltaO += delta[c] * dO[c];
            }
            float deltaO = block_sum(partialDeltaO, reduce);
            // dS += s dO q^T: the adjoint of S_t.
            for (int i = threadIdx.x; i < area; i += blockDim.x) dS[i] += outputScale * dO[i / dk] * q[i % dk];
            __syncthreads();
            for (int j = threadIdx.x; j < dk; j += blockDim.x) {
                float sum = 0.0f, keySum = 0.0f;
                for (int c = 0; c < dv; c++) {
                    sum += before[c * dk + j] * dO[c];
                    keySum += dS[c * dk + j] * delta[c];
                }
                dq[j] = outputScale * (a * sum + k[j] * deltaO);
                dkey[j] = keySum;
            }
            for (int c = threadIdx.x; c < dv; c += blockDim.x) {
                float sum = 0.0f;
                for (int j = 0; j < dk; j++) sum += dS[c * dk + j] * k[j];
                dDelta[c] = sum;
            }
            float partialAlpha = 0.0f;
            for (int i = threadIdx.x; i < area; i += blockDim.x) partialAlpha += dS[i] * before[i];
            float dA = block_sum(partialAlpha, reduce);
            float partialBeta = 0.0f, partialU = 0.0f;
            for (int c = threadIdx.x; c < dv; c += blockDim.x) {
                partialBeta += dDelta[c] * (row[qkWidth + h * dv + c] - a * u[c]);
                partialU += dDelta[c] * u[c];
                dConvolved[(long long)t * width + qkWidth + h * dv + c] += b * dDelta[c];
            }
            float dB = block_sum(partialBeta, reduce);
            dA -= b * block_sum(partialU, reduce);
            // dk += S_{t-1}^T du with du = -a b dDelta; dS_{t-1} = a dS_t + du k^T.
            for (int j = threadIdx.x; j < dk; j += blockDim.x) {
                float sum = 0.0f;
                for (int c = 0; c < dv; c++) sum += dDelta[c] * before[c * dk + j];
                dkey[j] -= a * b * sum;
            }
            __syncthreads();
            for (int i = threadIdx.x; i < area; i += blockDim.x) dS[i] = a * dS[i] - a * b * dDelta[i / dk] * k[i % dk];
            float projectionQ = 0.0f, projectionK = 0.0f;
            for (int j = threadIdx.x; j < dk; j += blockDim.x) {
                projectionQ += dq[j] * q[j];
                projectionK += dkey[j] * k[j];
            }
            projectionQ = block_sum(projectionQ, reduce);
            projectionK = block_sum(projectionK, reduce);
            float* target = dQK + ((long long)t * valueHeads + h) * 2 * dk;
            for (int j = threadIdx.x; j < dk; j += blockDim.x) {
                target[j] = (dq[j] - q[j] * projectionQ) / normQ;
                target[dk + j] = (dkey[j] - k[j] * projectionK) / normK;
            }
            if (threadIdx.x == 0) {
                dAlpha[(long long)t * valueHeads + h] = dA;
                dBeta[(long long)t * valueHeads + h] = dB;
            }
            __syncthreads();
        }
    }
}

// Sums each key head's value heads' q and k gradients into dConvolved's q and k columns.
extern "C" __global__ void gdn_gather_query_key(const float* dQK, float* dConvolved, int rows, int keyHeads,
                                                int valueHeads, int dk, int dv) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    int qkWidth = 2 * keyHeads * dk, width = qkWidth + valueHeads * dv;
    if (i >= (long long)rows * qkWidth) return;
    long long t = i / qkWidth;
    int column = (int)(i % qkWidth);
    int part = column / (keyHeads * dk), kh = (column % (keyHeads * dk)) / dk, j = column % dk;
    int group = valueHeads / keyHeads;
    float sum = 0.0f;
    for (int g = 0; g < group; g++) sum += dQK[((t * valueHeads) + kh * group + g) * 2 * dk + part * dk + j];
    dConvolved[t * width + column] += sum;
}

// ---------------------------------------------------------------------------------------------------------------
// GDN gated norm per (row, value head): y = o * inv * w * silu(z), inv = rsqrt(mean(o^2) + eps); z is the second half of
// each valueZ row. One block per (row, head), blockDim.x = 32 * k.

extern "C" __global__ void gdn_gated_norm_forward(const float* o, const float* vz, const float* w, float* y,
                                                  float* inverse, int heads, int dv, float eps) {
    __shared__ float shared[32];
    long long row = blockIdx.x / heads;
    int h = blockIdx.x % heads, width = heads * dv;
    const float* x = o + row * width + h * dv;
    const float* z = vz + row * 2 * width + width + h * dv;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < dv; i += blockDim.x) sum += x[i] * x[i];
    float inv = rsqrtf(block_sum(sum, shared) / dv + eps);
    if (threadIdx.x == 0) inverse[blockIdx.x] = inv;
    for (int i = threadIdx.x; i < dv; i += blockDim.x) y[row * width + h * dv + i] = x[i] * inv * w[i] * silu(z[i]);
}

extern "C" __global__ void gdn_gated_norm_backward(const float* o, const float* vz, const float* w,
                                                   const float* inverse, const float* dy, float* dO, float* dvz,
                                                   float* dw, int heads, int dv) {
    __shared__ float shared[32];
    long long row = blockIdx.x / heads;
    int h = blockIdx.x % heads, width = heads * dv;
    const float* x = o + row * width + h * dv;
    const float* z = vz + row * 2 * width + width + h * dv;
    const float* g = dy + row * width + h * dv;
    float inv = inverse[blockIdx.x];
    float dot = 0.0f;
    for (int i = threadIdx.x; i < dv; i += blockDim.x) dot += g[i] * w[i] * silu(z[i]) * x[i];
    float mean = block_sum(dot, shared) / dv;
    for (int i = threadIdx.x; i < dv; i += blockDim.x) {
        float gi = g[i] * w[i] * silu(z[i]);
        dO[row * width + h * dv + i] += inv * (gi - x[i] * inv * inv * mean);
        dvz[row * 2 * width + width + h * dv + i] += g[i] * x[i] * inv * w[i] * silu_grad(z[i]);
        if (dw != nullptr) atomicAdd(dw + i, g[i] * x[i] * inv * silu(z[i]));
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Attention q/k norm and RoPE per (row, head) of qk rows [q (qHeads x d); k (kvHeads x d)]: n = x * inv * (1 + w)
// (query norm for q heads, key norm for k heads), then the first `rotary` dimensions rotate in halves at
// angle (position0 + row) * theta^(-2 f / rotary). One block per (row, head), blockDim.x = d.

__device__ __forceinline__ void rope_angle(long long position, int f, int rotary, double theta, float* cosine,
                                           float* sine) {
    double angle = (double)position * pow(theta, -(2.0 * f) / rotary);
    *cosine = (float)cos(angle);
    *sine = (float)sin(angle);
}

extern "C" __global__ void qk_norm_rope_forward(const float* qk, const float* qNorm, const float* kNorm, float* out,
                                                float* inverse, int qHeads, int kvHeads, int d, int rotary,
                                                long long position0, double theta, float eps) {
    __shared__ float shared[32];
    __shared__ float normed[512];
    int heads = qHeads + kvHeads;
    long long row = blockIdx.x / heads;
    int head = blockIdx.x % heads, i = threadIdx.x;
    const float* x = qk + row * heads * d + head * d;
    const float* w = head < qHeads ? qNorm : kNorm;
    float inv = rsqrtf(block_sum(x[i] * x[i], shared) / d + eps);
    if (i == 0) inverse[blockIdx.x] = inv;
    normed[i] = x[i] * inv * (1.0f + w[i]);
    __syncthreads();
    float result = normed[i];
    if (i < rotary) {
        int half = rotary / 2, pair = i < half ? i + half : i - half, f = i < half ? i : pair;
        float cosine, sine;
        rope_angle(position0 + row, f, rotary, theta, &cosine, &sine);
        result = i < half ? normed[i] * cosine - normed[pair] * sine : normed[i] * cosine + normed[pair] * sine;
    }
    out[row * heads * d + head * d + i] = result;
}

// The rotation's transpose, then the norm's backward; dw into dqNorm / dkNorm (nullable).
extern "C" __global__ void qk_norm_rope_backward(const float* qk, const float* qNorm, const float* kNorm,
                                                 const float* inverse, const float* dOut, float* dqk, float* dqNorm,
                                                 float* dkNorm, int qHeads, int kvHeads, int d, int rotary,
                                                 long long position0, double theta) {
    __shared__ float shared[32];
    __shared__ float dOutRow[512];
    int heads = qHeads + kvHeads;
    long long row = blockIdx.x / heads;
    int head = blockIdx.x % heads, i = threadIdx.x;
    long long base = row * heads * d + head * d;
    const float* x = qk + base;
    const float* w = head < qHeads ? qNorm : kNorm;
    float* dw = head < qHeads ? dqNorm : dkNorm;
    dOutRow[i] = dOut[base + i];
    __syncthreads();
    float dn = dOutRow[i];
    if (i < rotary) {
        int half = rotary / 2, pair = i < half ? i + half : i - half, f = i < half ? i : pair;
        float cosine, sine;
        rope_angle(position0 + row, f, rotary, theta, &cosine, &sine);
        // out[i] = n[i] c -+ n[pair] s, out[pair] = n[pair] c +- n[i] s  =>  dn[i] = dOut[i] c +- dOut[pair] s.
        dn = i < half ? dOutRow[i] * cosine + dOutRow[pair] * sine : dOutRow[i] * cosine - dOutRow[pair] * sine;
    }
    float inv = inverse[blockIdx.x];
    float g = dn * (1.0f + w[i]);
    float mean = block_sum(g * x[i], shared) / d;
    dqk[base + i] += inv * (g - x[i] * inv * inv * mean);
    if (dw != nullptr) atomicAdd(dw + i, dn * x[i] * inv);
}

// ---------------------------------------------------------------------------------------------------------------
// Causal attention pieces (the products run on cuBLAS): softmax of scores [heads, rows, rows] in place with the causal
// mask and `scale`; its backward dS = P * (dP - rowsum(dP * P)) * scale in place of dP. One block per (head, row).

extern "C" __global__ void causal_softmax(float* scores, int rows, float scale) {
    __shared__ float shared[32];
    long long r = blockIdx.x % rows;
    float* row = scores + (long long)blockIdx.x * rows;
    float maximum = -3.0e38f;
    for (int j = threadIdx.x; j <= r; j += blockDim.x) maximum = fmaxf(maximum, row[j] * scale);
    for (int offset = 16; offset > 0; offset >>= 1)
        maximum = fmaxf(maximum, __shfl_down_sync(0xffffffffu, maximum, offset));
    __syncthreads();
    if (threadIdx.x % 32 == 0) shared[threadIdx.x / 32] = maximum;
    __syncthreads();
    if (threadIdx.x == 0) {
        float m = shared[0];
        for (int w = 1; w < blockDim.x / 32; w++) m = fmaxf(m, shared[w]);
        shared[0] = m;
    }
    __syncthreads();
    maximum = shared[0];
    __syncthreads();
    float sum = 0.0f;
    for (int j = threadIdx.x; j <= r; j += blockDim.x) sum += expf(row[j] * scale - maximum);
    float total = block_sum(sum, shared);
    for (int j = threadIdx.x; j < rows; j += blockDim.x) row[j] = j <= r ? expf(row[j] * scale - maximum) / total : 0.0f;
}

extern "C" __global__ void causal_softmax_backward(const float* probabilities, float* dScores, int rows, float scale) {
    __shared__ float shared[32];
    long long r = blockIdx.x % rows;
    const float* p = probabilities + (long long)blockIdx.x * rows;
    float* d = dScores + (long long)blockIdx.x * rows;
    float dot = 0.0f;
    for (int j = threadIdx.x; j <= r; j += blockDim.x) dot += d[j] * p[j];
    float total = block_sum(dot, shared);
    for (int j = threadIdx.x; j < rows; j += blockDim.x) d[j] = j <= r ? p[j] * (d[j] - total) * scale : 0.0f;
}

// context[r, h d + i] (from the products, [rows, qHeads x d]) times sigmoid(gate), gate = the first qHeads x d of each
// gateValue row (width qHeads x d + kvHeads x d).
extern "C" __global__ void gate_forward(const float* context, const float* gateValue, float* out, int rows, int width,
                                        int rowWidth) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)rows * width) return;
    long long r = i / width;
    int c = (int)(i % width);
    out[i] = context[i] * sigmoidf(gateValue[r * rowWidth + c]);
}

extern "C" __global__ void gate_backward(const float* context, const float* gateValue, const float* dOut,
                                         float* dContext, float* dGateValue, int rows, int width, int rowWidth) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)rows * width) return;
    long long r = i / width;
    int c = (int)(i % width);
    float s = sigmoidf(gateValue[r * rowWidth + c]);
    dContext[i] += dOut[i] * s;
    dGateValue[r * rowWidth + c] += dOut[i] * context[i] * s * (1.0f - s);
}

// ---------------------------------------------------------------------------------------------------------------
// Loss and optimizer.

// loss[0] += sum (y - target)^2; dy = 2 (y - target) * scale.
extern "C" __global__ void squared_error(const float* y, const float* target, long long count, float scale, float* dy,
                                         double* loss) {
    __shared__ float shared[32];
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    float e = i < count ? y[i] - target[i] : 0.0f;
    if (i < count && dy != nullptr) dy[i] = 2.0f * e * scale;
    float total = block_sum(e * e, shared);
    if (threadIdx.x == 0) atomicAdd(loss, (double)total);
}

// AdamW step on p with gradient g and moments m, v (bias-corrected by step t).
extern "C" __global__ void adam(float* p, float* g, float* m, float* v, long long count, float lr, float beta1,
                                float beta2, float eps, float decay, int t) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float grad = g[i];
    m[i] = beta1 * m[i] + (1.0f - beta1) * grad;
    v[i] = beta2 * v[i] + (1.0f - beta2) * grad * grad;
    float mHat = m[i] / (1.0f - powf(beta1, t)), vHat = v[i] / (1.0f - powf(beta2, t));
    p[i] -= lr * (mHat / (sqrtf(vHat) + eps) + decay * p[i]);
    g[i] = 0.0f;
}

// ---------------------------------------------------------------------------------------------------------------
// Distillation to a teacher's top-k: the teacher's k most likely tokens of each row with their log-probabilities, and
// the student's KL(teacher top-k renormalized || student) with its gradient on the student's logits.

// For each row of `logits` [rows, vocabulary] (FP32): the `k` largest entries (k <= 64), in descending order, as
// token ids and log-probabilities under the full softmax. Each thread keeps the best 16 of its strided entries; the
// k best of those candidates (at most 4096) are chosen in shared memory. One block of 256 threads per row.
extern "C" __global__ void top_k_log_softmax(const float* logits, int vocabulary, int k, int* ids, float* logProbs) {
    constexpr int kLocal = 16;
    __shared__ float values[kThreads * kLocal];
    __shared__ int indices[kThreads * kLocal];
    __shared__ float shared[32];
    const float* row = logits + (long long)blockIdx.x * vocabulary;
    float best[kLocal];
    int bestIndex[kLocal];
    for (int i = 0; i < kLocal; i++) {
        best[i] = -3.0e38f;
        bestIndex[i] = -1;
    }
    float maximum = -3.0e38f;
    for (int j = threadIdx.x; j < vocabulary; j += kThreads) {
        float v = row[j];
        maximum = fmaxf(maximum, v);
        if (v > best[kLocal - 1]) {
            int position = kLocal - 1;
            while (position > 0 && best[position - 1] < v) {
                best[position] = best[position - 1];
                bestIndex[position] = bestIndex[position - 1];
                position--;
            }
            best[position] = v;
            bestIndex[position] = j;
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1) maximum = fmaxf(maximum, __shfl_down_sync(0xffffffffu, maximum, offset));
    if (threadIdx.x % 32 == 0) shared[threadIdx.x / 32] = maximum;
    __syncthreads();
    if (threadIdx.x == 0) {
        float m = shared[0];
        for (int w = 1; w < kThreads / 32; w++) m = fmaxf(m, shared[w]);
        shared[0] = m;
    }
    __syncthreads();
    maximum = shared[0];
    __syncthreads();
    float sum = 0.0f;
    for (int j = threadIdx.x; j < vocabulary; j += kThreads) sum += expf(row[j] - maximum);
    float logZ = maximum + logf(block_sum(sum, shared));
    for (int i = 0; i < kLocal; i++) {
        values[threadIdx.x * kLocal + i] = best[i];
        indices[threadIdx.x * kLocal + i] = bestIndex[i];
    }
    __syncthreads();
    // k rounds of a block-wide argmax over the candidates.
    for (int round = 0; round < k; round++) {
        float v = -3.0e38f;
        int at = -1;
        for (int c = threadIdx.x; c < kThreads * kLocal; c += kThreads)
            if (values[c] > v || (values[c] == v && indices[c] >= 0 && (at < 0 || indices[c] < indices[at]))) {
                v = values[c];
                at = c;
            }
        for (int offset = 16; offset > 0; offset >>= 1) {
            float ov = __shfl_down_sync(0xffffffffu, v, offset);
            int oa = __shfl_down_sync(0xffffffffu, at, offset);
            if (ov > v) { v = ov; at = oa; }
        }
        __shared__ float warpValue[32];
        __shared__ int warpAt[32];
        if (threadIdx.x % 32 == 0) {
            warpValue[threadIdx.x / 32] = v;
            warpAt[threadIdx.x / 32] = at;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            float bv = warpValue[0];
            int ba = warpAt[0];
            for (int w = 1; w < kThreads / 32; w++)
                if (warpValue[w] > bv) { bv = warpValue[w]; ba = warpAt[w]; }
            ids[(long long)blockIdx.x * k + round] = indices[ba];
            logProbs[(long long)blockIdx.x * k + round] = bv - logZ;
            values[ba] = -3.0e38f;
        }
        __syncthreads();
    }
}

// For each row of student logits [rows, vocabulary]: the KL from the teacher's top-k (ids, log-probabilities under its
// full softmax) plus one bucket holding the rest of its probability, p_rest = 1 - sum p_i, to the student's same
// partition, q_rest = the student's probability outside those k tokens:
//   loss[0] += sum_i p_i (log p_i - log q_i) + p_rest (log p_rest - log q_rest),
//   dLogits[j] = (q_j - p_j) * scale for j in the top-k, q_j (1 - p_rest / q_rest) * scale elsewhere.
// The teacher's tail is kept as one mass rather than dropped, so an uncertain teacher does not push the student to
// zero probability outside its top k. q_rest is summed over the tokens outside the top k directly (not 1 - sum q_i).
// One block per row; k <= 64.
extern "C" __global__ void top_k_distillation(const float* logits, int vocabulary, int k, const int* ids,
                                              const float* logProbs, float scale, float* dLogits, double* loss) {
    __shared__ float shared[32];
    __shared__ int sorted[64];
    __shared__ float teacher[64];
    __shared__ float teacherRest, restRatio;
    const float* row = logits + (long long)blockIdx.x * vocabulary;
    float* d = dLogits + (long long)blockIdx.x * vocabulary;
    const int* rowIds = ids + (long long)blockIdx.x * k;
    if (threadIdx.x == 0) {
        float total = 0.0f;
        for (int i = 0; i < k; i++) {
            teacher[i] = expf(logProbs[(long long)blockIdx.x * k + i]);
            total += teacher[i];
            int id = rowIds[i], at = i;
            while (at > 0 && sorted[at - 1] > id) {
                sorted[at] = sorted[at - 1];
                at--;
            }
            sorted[at] = id;
        }
        teacherRest = fmaxf(0.0f, 1.0f - total);
    }
    float maximum = -3.0e38f;
    for (int j = threadIdx.x; j < vocabulary; j += blockDim.x) maximum = fmaxf(maximum, row[j]);
    for (int offset = 16; offset > 0; offset >>= 1) maximum = fmaxf(maximum, __shfl_down_sync(0xffffffffu, maximum, offset));
    if (threadIdx.x % 32 == 0) shared[threadIdx.x / 32] = maximum;
    __syncthreads();
    if (threadIdx.x == 0) {
        float m = shared[0];
        for (int w = 1; w < blockDim.x / 32; w++) m = fmaxf(m, shared[w]);
        shared[0] = m;
    }
    __syncthreads();
    maximum = shared[0];
    __syncthreads();
    float sum = 0.0f, outside = 0.0f;
    for (int j = threadIdx.x; j < vocabulary; j += blockDim.x) {
        float e = expf(row[j] - maximum);
        sum += e;
        int low = 0, high = k;
        while (low < high) {
            int middle = (low + high) / 2;
            if (sorted[middle] < j) low = middle + 1;
            else high = middle;
        }
        if (low >= k || sorted[low] != j) outside += e;
    }
    float total = block_sum(sum, shared);
    float logZ = maximum + logf(total);
    float studentRest = block_sum(outside, shared) / total;
    if (threadIdx.x == 0) restRatio = studentRest > 0.0f ? teacherRest / studentRest : 0.0f;
    __syncthreads();
    for (int j = threadIdx.x; j < vocabulary; j += blockDim.x) d[j] = expf(row[j] - logZ) * (1.0f - restRatio) * scale;
    __syncthreads();
    if (threadIdx.x == 0) {
        double l = 0.0;
        for (int i = 0; i < k; i++) {
            float p = teacher[i];
            float q = expf(row[rowIds[i]] - logZ);
            // In the top k the gradient is q - p: undo the outside factor written above.
            d[rowIds[i]] = (q - p) * scale;
            if (p > 0.0f) l += (double)p * ((double)logf(p) - ((double)row[rowIds[i]] - logZ));
        }
        if (teacherRest > 0.0f && studentRest > 0.0f)
            l += (double)teacherRest * ((double)logf(teacherRest) - (double)logf(studentRest));
        atomicAdd(loss, l);
    }
}
