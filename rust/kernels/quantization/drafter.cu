// Kernels of the MTP drafter's training (quantization/README.md, `train-drafter`): NVFP4 rounding as the converter
// does it, the attention of drafts unrolled over depths, and the draft head's cross-entropy. FP32 and row-major,
// as train.cu; backward kernels accumulate (+=) into gradients unless a name says otherwise.
#include <cuda_fp16.h>

namespace {

constexpr int kThreads = 256;

// E2M1 magnitudes and E4M3 values (codes 0..126), ascending, as tools/euhedral_artifacts/nvfp4.py.
__constant__ float kE2m1[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

__device__ float e4m3_value(int code) {
    return code < 8 ? (float)(code & 7) / 8.0f * 0.015625f : (1.0f + (float)(code & 7) / 8.0f) * exp2f((float)((code >> 3) - 7));
}

// The nearest entry of an ascending table to a non-negative value, ties to the even index; values at or past the
// ends take the end entries (nvfp4.py round_to_table, numpy searchsorted side left).
template <class Table>
__device__ int round_to_table(float value, Table table, int count) {
    int low = 0, high = count;  // first index whose entry is >= value
    while (low < high) {
        int middle = (low + high) / 2;
        if (table(middle) < value) low = middle + 1; else high = middle;
    }
    int upper = min(max(low, 1), count - 1), lower = upper - 1;
    float below = value - table(lower), above = table(upper) - value;
    int index = (above < below || (above == below && upper % 2 == 0)) ? upper : lower;
    if (value >= table(count - 1)) index = count - 1;
    if (value <= table(0)) index = 0;
    return index;
}

struct E2m1Table { __device__ float operator()(int i) const { return kE2m1[i]; } };
struct E4m3Table { __device__ float operator()(int i) const { return e4m3_value(i); } };

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

__device__ float block_max(float value, float* shared) {
    for (int offset = 16; offset > 0; offset >>= 1) value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32, warps = blockDim.x / 32;
    __syncthreads();
    if (lane == 0) shared[warp] = value;
    __syncthreads();
    if (warp == 0) {
        value = lane < warps ? shared[lane] : -3.0e38f;
        for (int offset = 16; offset > 0; offset >>= 1) value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
        if (lane == 0) shared[0] = value;
    }
    __syncthreads();
    float total = shared[0];
    __syncthreads();
    return total;
}

__device__ unsigned short bf16_rounded(float value) {
    unsigned bits = __float_as_uint(value);
    if ((bits & 0x7fffffffu) > 0x7f800000u) return 0x7fc0u;
    return (unsigned short)((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

}  // namespace

// ---------------------------------------------------------------------------------------------------------------
// NVFP4 (tools/euhedral_artifacts/nvfp4.py, round to nearest): a tensor's global scale is its largest magnitude over
// 6 x 448; each block of 16 along a row takes the E4M3 scale nearest to its largest magnitude / 6 / global, and each
// value the E2M1 code nearest to |value| / (scale x global).

// out (one unsigned, zeroed first) = the bits of max |x| (non-negative floats order as their bits).
extern "C" __global__ void absolute_maximum(const float* x, long long count, unsigned* out) {
    __shared__ float shared[32];
    float m = 0.0f;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < count; i += (long long)gridDim.x * blockDim.x)
        m = fmaxf(m, fabsf(x[i]));
    m = block_max(m, shared);
    if (threadIdx.x == 0) atomicMax(out, __float_as_uint(m));
}

// One thread per block of 16: w [rows, k] (k a multiple of 16) rounded to NVFP4 under the global scale
// amax / 2688 (`amaxBits` from absolute_maximum). Writes the decoded values to `out` (may be w), and, when not null,
// the packed codes [rows, k / 2] (low nibble first) and E4M3 scale codes [rows, k / 16] of the artifact layout.
extern "C" __global__ void nvfp4_round(const float* w, float* out, unsigned char* packed, unsigned char* scales,
                                       long long blocks, const unsigned* amaxBits) {
    long long b = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (b >= blocks) return;
    float global = __uint_as_float(*amaxBits) / 2688.0f;
    const float* x = w + b * 16;
    float values[16], largest = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; i++) { values[i] = x[i]; largest = fmaxf(largest, fabsf(values[i])); }
    int scaleCode = 0;
    if (global > 0.0f) scaleCode = round_to_table(__fdiv_rn(__fdiv_rn(largest, 6.0f), global), E4m3Table(), 127);
    float decode = __fmul_rn(e4m3_value(scaleCode), global);
    unsigned char codes[16];
#pragma unroll
    for (int i = 0; i < 16; i++) {
        float scaled = decode > 0.0f ? __fdiv_rn(fabsf(values[i]), decode) : 0.0f;
        int magnitude = round_to_table(scaled, E2m1Table(), 8);
        codes[i] = (unsigned char)(magnitude | ((values[i] < 0.0f && magnitude > 0) ? 8 : 0));
        float decoded = __fmul_rn(kE2m1[magnitude] * ((codes[i] & 8) ? -1.0f : 1.0f), decode);
        out[b * 16 + i] = decoded;
    }
    if (packed != nullptr) {
#pragma unroll
        for (int i = 0; i < 8; i++) packed[b * 8 + i] = (unsigned char)(codes[2 * i] | (codes[2 * i + 1] << 4));
        scales[b] = (unsigned char)scaleCode;
    }
}

// An artifact's NVFP4 rows (row-split-k128-v1: packed codes [rows, kPad / 2], scale codes [rows, kPad / 16], FP32
// global scale) decoded to BF16 [rows, k].
extern "C" __global__ void nvfp4_decode_bf16(const unsigned char* packed, const unsigned char* scales, const float* global,
                                             long long rows, int kPad, int k, unsigned short* out) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= rows * k) return;
    long long row = i / k;
    int column = (int)(i % k);
    unsigned char byte = packed[row * (kPad / 2) + column / 2];
    int code = (column & 1) ? byte >> 4 : byte & 15;
    float scale = __fmul_rn(e4m3_value(scales[row * (kPad / 16) + column / 16]), *global);
    out[i] = bf16_rounded(__fmul_rn(kE2m1[code & 7] * ((code & 8) ? -1.0f : 1.0f), scale));
}

// ---------------------------------------------------------------------------------------------------------------
// Attention of unrolled drafts. A window's rows are lineages: row r drafts from the base model's hidden at position
// a + r, and its depth-k row sits at position a + r + k - 1. A depth-k query of lineage r attends the depth-1 keys of
// every position up to a + r (the causal part, keys [ldS columns], from the window's first context key) and the
// depth-2..k keys of its own lineage (the lineage part, `extras` [heads, rows, extraStride], slot j - 2 for depth j).

// Softmax in place over each (head, row): causal columns [0, visible0 + r) of `scores` [heads, rows, ldS] and the
// `extraCount` lineage slots of `extras`; masked causal columns become 0. One block per (head, row).
extern "C" __global__ void drafter_softmax(float* scores, int ldS, int visible0, float* extras, int extraStride,
                                           int extraCount, int rows, float scale) {
    __shared__ float shared[32];
    long long hr = blockIdx.x;
    int r = (int)(hr % rows), visible = visible0 + r;
    float* row = scores + hr * ldS;
    float* extra = extras + hr * extraStride;
    float maximum = -3.0e38f;
    for (int j = threadIdx.x; j < visible; j += blockDim.x) maximum = fmaxf(maximum, row[j] * scale);
    for (int j = threadIdx.x; j < extraCount; j += blockDim.x) maximum = fmaxf(maximum, extra[j] * scale);
    maximum = block_max(maximum, shared);
    float sum = 0.0f;
    for (int j = threadIdx.x; j < visible; j += blockDim.x) sum += expf(row[j] * scale - maximum);
    for (int j = threadIdx.x; j < extraCount; j += blockDim.x) sum += expf(extra[j] * scale - maximum);
    float total = block_sum(sum, shared);
    for (int j = threadIdx.x; j < ldS; j += blockDim.x) row[j] = j < visible ? expf(row[j] * scale - maximum) / total : 0.0f;
    for (int j = threadIdx.x; j < extraCount; j += blockDim.x) extra[j] = expf(extra[j] * scale - maximum) / total;
}

// dS = P (dP - sum(P dP)) scale, in place of dP, over the same columns.
extern "C" __global__ void drafter_softmax_backward(const float* probabilities, float* dScores, int ldS, int visible0,
                                                    const float* extraProbabilities, float* dExtras, int extraStride,
                                                    int extraCount, int rows, float scale) {
    __shared__ float shared[32];
    long long hr = blockIdx.x;
    int r = (int)(hr % rows), visible = visible0 + r;
    const float* p = probabilities + hr * ldS;
    float* d = dScores + hr * ldS;
    const float* pe = extraProbabilities + hr * extraStride;
    float* de = dExtras + hr * extraStride;
    float dot = 0.0f;
    for (int j = threadIdx.x; j < visible; j += blockDim.x) dot += d[j] * p[j];
    for (int j = threadIdx.x; j < extraCount; j += blockDim.x) dot += de[j] * pe[j];
    float total = block_sum(dot, shared);
    for (int j = threadIdx.x; j < ldS; j += blockDim.x) d[j] = j < visible ? p[j] * (d[j] - total) * scale : 0.0f;
    for (int j = threadIdx.x; j < extraCount; j += blockDim.x) de[j] = pe[j] * (de[j] - total) * scale;
}

// out[(h rows + r) stride + slot] (= or += when `accumulate`) = a[r, h d ..] . b[r, (h / group) d ..]: a lineage score
// (a = queries, b = one depth's keys) or its gradient (a = d context, b = values). One block of d threads per (r, h).
extern "C" __global__ void lineage_dot(const float* a, int lda, const float* b, int ldb, int heads, int group, int d,
                                       int rows, float* out, int stride, int slot) {
    __shared__ float shared[32];
    int r = blockIdx.x / heads, h = blockIdx.x % heads, i = threadIdx.x;
    float value = a[(long long)r * lda + h * d + i] * b[(long long)r * ldb + (h / group) * d + i];
    float total = block_sum(value, shared);
    if (i == 0) out[((long long)h * rows + r) * stride + slot] = total;
}

// y[r, h d + i] += c[(h rows + r) stride + slot] x[r, (h / group) d + i]: the lineage part of the context (x = values)
// or of the queries' gradient (x = keys). One block of d threads per (r, h).
extern "C" __global__ void lineage_spread(const float* c, int stride, int slot, const float* x, int ldx, float* y,
                                          int ldy, int heads, int group, int d, int rows) {
    int r = blockIdx.x / heads, h = blockIdx.x % heads, i = threadIdx.x;
    float coefficient = c[((long long)h * rows + r) * stride + slot];
    y[(long long)r * ldy + h * d + i] += coefficient * x[(long long)r * ldx + (h / group) * d + i];
}

// y[r, kv d + i] += sum over the group's heads h of c[(h rows + r) stride + slot] x[r, h d + i]: the lineage keys' or
// values' gradient (x = queries or d context). One block of d threads per (r, kv).
extern "C" __global__ void lineage_gather(const float* c, int stride, int slot, const float* x, int ldx, float* y,
                                          int ldy, int heads, int group, int d, int rows) {
    int kvHeads = heads / group, r = blockIdx.x / kvHeads, kv = blockIdx.x % kvHeads, i = threadIdx.x;
    float sum = 0.0f;
    for (int h = kv * group; h < (kv + 1) * group; h++)
        sum += c[((long long)h * rows + r) * stride + slot] * x[(long long)r * ldx + h * d + i];
    y[(long long)r * ldy + kv * d + i] += sum;
}

// y[r, yOffset + c] (= or +=) x[r, xOffset + c] for c < columns: copies or adds a column range between row-major
// buffers.
extern "C" __global__ void copy_columns(const float* x, int ldx, float* y, int ldy, int rows, int columns, int add) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= (long long)rows * columns) return;
    long long r = i / columns, c = i % columns;
    float v = x[r * ldx + c];
    if (add) y[r * ldy + c] += v; else y[r * ldy + c] = v;
}

// ---------------------------------------------------------------------------------------------------------------
// The draft head's cross-entropy, one block per row of logits [rows, vocabulary] (FP32): for a row with a label
// (>= 0), loss (double) += -log softmax[label] and the logits become d loss / d logits x scale = (softmax - onehot)
// scale; a row without one becomes 0. `predicted` gets each row's argmax (the lowest index among equal maxima, as
// the engine's device argmax) and `labelLogProbability` the label's log-probability (0 without a label).
extern "C" __global__ void draft_cross_entropy(float* logits, int vocabulary, const int* labels, float scale,
                                               double* loss, int* predicted, float* labelLogProbability) {
    __shared__ float shared[32];
    __shared__ float bestValue[kThreads];
    __shared__ int bestIndex[kThreads];
    long long row = blockIdx.x;
    float* x = logits + row * vocabulary;
    int label = labels[row];
    float value = -3.0e38f;
    int index = 0x7fffffff;
    for (int i = threadIdx.x; i < vocabulary; i += blockDim.x)
        if (x[i] > value) { value = x[i]; index = i; }
    bestValue[threadIdx.x] = value;
    bestIndex[threadIdx.x] = index;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            float other = bestValue[threadIdx.x + stride];
            int otherIndex = bestIndex[threadIdx.x + stride];
            if (other > bestValue[threadIdx.x] || (other == bestValue[threadIdx.x] && otherIndex < bestIndex[threadIdx.x])) {
                bestValue[threadIdx.x] = other;
                bestIndex[threadIdx.x] = otherIndex;
            }
        }
        __syncthreads();
    }
    float maximum = bestValue[0];
    if (threadIdx.x == 0) predicted[row] = bestIndex[0];
    float sum = 0.0f;
    for (int i = threadIdx.x; i < vocabulary; i += blockDim.x) sum += expf(x[i] - maximum);
    float total = block_sum(sum, shared);
    float logTotal = logf(total) + maximum;
    if (label < 0) {
        for (int i = threadIdx.x; i < vocabulary; i += blockDim.x) x[i] = 0.0f;
        if (threadIdx.x == 0) labelLogProbability[row] = 0.0f;
        return;
    }
    if (threadIdx.x == 0) {
        float logProbability = x[label] - logTotal;
        labelLogProbability[row] = logProbability;
        atomicAdd(loss, -(double)logProbability);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < vocabulary; i += blockDim.x)
        x[i] = (expf(x[i] - logTotal) - (i == label ? 1.0f : 0.0f)) * scale;
}

// out[i] = BF16 of x[i] (round to nearest even).
extern "C" __global__ void float_to_bf16(const float* x, long long count, unsigned short* out) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i < count) out[i] = bf16_rounded(x[i]);
}

// The token embedding's rows for `ids` from a Q3G64 row-split-k128-v1 table (3-bit codes, 24 bytes per group of 64,
// then FP16 scales at `scaleOffset`; pinned host memory read in place), each value bf16(code x scale) as the
// engine's euhedral_q3_embedding computes it, widened to FP32 [rows, width].
extern "C" __global__ void q3_embedding_rows(const unsigned char* table, const int* ids, int rows, int width,
                                             unsigned long long scaleOffset, float* out) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= (long long)rows * width) return;
    long long r = i / width;
    int c = (int)(i % width), groupsPerRow = ((width + 127) / 128) * 2;
    long long group = (long long)ids[r] * groupsPerRow + c / 64;
    int bit = (c % 64) * 3;
    const unsigned char* bytes = table + group * 24;
    unsigned word = bytes[bit / 8] | ((bit / 8 + 1 < 24 ? bytes[bit / 8 + 1] : 0u) << 8);
    int code = (int)((word >> (bit % 8)) & 7u);
    if (code >= 4) code -= 8;
    float scale = __half2float(__ushort_as_half(*reinterpret_cast<const unsigned short*>(table + scaleOffset + group * 2)));
    out[i] = __uint_as_float((unsigned)bf16_rounded((float)code * scale) << 16);
}
