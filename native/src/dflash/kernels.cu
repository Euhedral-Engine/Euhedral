// DFlash2 drafter (z-lab/dflash, docs/DFLASH2.md): the operators of its BF16 forward, each rounding where the
// published PyTorch model rounds, so the drafter tracks the reference at every intermediate.
//
// Torch evaluates the drafter in BF16 tensors: every elementwise operation rounds its result to BF16, and a linear
// accumulates in FP32 and rounds once. The kernels below keep those rounding points (two roundings in RMSNorm, one per
// product and sum in the dynamic convolution and RoPE, SiLU rounded before the up product); their FP32 reductions
// differ from torch's only in summation order.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include "common/pdl.cuh"

typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
typedef int int32_t;

namespace dflash {

__device__ __forceinline__ float bf(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

__device__ __forceinline__ __nv_bfloat16 rn(float value) {
    return __float2bfloat16_rn(value);
}

// BF16 round trip of an FP32 value.
__device__ __forceinline__ float r(float value) {
    return __bfloat162float(__float2bfloat16_rn(value));
}

__device__ __forceinline__ void mma_bf16(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
            "{%0,%1,%2,%3};\n"
            : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint4 load_row(const __nv_bfloat16* row, uint32_t k, bool valid) {
    if (!valid) return make_uint4(0, 0, 0, 0);
    return __ldg(reinterpret_cast<const uint4*>(row + k));
}

// y[m][n] = sum_k x[m][k] w[n][k]: BF16 operands on m16n8k16 tensor cores, FP32 accumulation, one BF16 rounding.
//
// A warp owns 8 output columns over the whole K; a CTA's four warps own 32. Per 32-wide K chunk a lane loads 16
// contiguous bytes of its column's weight row (k = 8q .. 8q + 7 for q = lane % 4) and the matching 16 bytes of its
// activation rows, and feeds them to two MMAs as logical k {2q, 2q+1, 2q+8, 2q+9} -> 8q + 4s + {0, 1, 2, 3}: the
// same permutation of K on both operands, so the product is the plain sum. Each output's accumulation sequence is
// fixed by K alone; `MT` (row tiles of 16 per CTA, which reuse each weight load) never changes a row's bits.
template <int MT, int UNROLL>
__device__ __forceinline__ void linear(
        const __nv_bfloat16* __restrict__ x,
        const __nv_bfloat16* __restrict__ w,
        __nv_bfloat16* __restrict__ y,
        uint32_t rows,
        uint32_t k,
        uint32_t n) {
    const uint32_t warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const uint32_t g = lane / 4, q = lane % 4;
    const uint32_t column0 = blockIdx.x * 32 + warp * 8;
    if (column0 >= n) return;
    const uint32_t rowBase = blockIdx.y * 16 * MT;
    const __nv_bfloat16* weightRow = w + static_cast<uint64_t>(column0 + g) * k;
    const __nv_bfloat16* rowPointer[MT][2];
    bool rowValid[MT][2];
#pragma unroll
    for (int t = 0; t < MT; t++) {
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const uint32_t row = rowBase + t * 16 + g + h * 8;
            rowValid[t][h] = row < rows;
            rowPointer[t][h] = x + static_cast<uint64_t>(rowValid[t][h] ? row : 0) * k;
        }
    }
    float acc[MT][4];
#pragma unroll
    for (int t = 0; t < MT; t++) acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.0f;
    uint32_t chunk = 0;
    for (; chunk + 32 * UNROLL <= k; chunk += 32 * UNROLL) {
        uint4 b[UNROLL];
        uint4 a[UNROLL][MT][2];
#pragma unroll
        for (int u = 0; u < UNROLL; u++) {
            const uint32_t at = chunk + u * 32 + q * 8;
            b[u] = __ldg(reinterpret_cast<const uint4*>(weightRow + at));
#pragma unroll
            for (int t = 0; t < MT; t++)
#pragma unroll
                for (int h = 0; h < 2; h++) a[u][t][h] = load_row(rowPointer[t][h], at, rowValid[t][h]);
        }
#pragma unroll
        for (int u = 0; u < UNROLL; u++) {
#pragma unroll
            for (int t = 0; t < MT; t++) {
                const uint32_t step0[4] = {a[u][t][0].x, a[u][t][1].x, a[u][t][0].y, a[u][t][1].y};
                mma_bf16(acc[t], step0, b[u].x, b[u].y);
                const uint32_t step1[4] = {a[u][t][0].z, a[u][t][1].z, a[u][t][0].w, a[u][t][1].w};
                mma_bf16(acc[t], step1, b[u].z, b[u].w);
            }
        }
    }
    for (; chunk < k; chunk += 32) {
        const uint32_t at = chunk + q * 8;
        const uint4 b = __ldg(reinterpret_cast<const uint4*>(weightRow + at));
#pragma unroll
        for (int t = 0; t < MT; t++) {
            const uint4 a0 = load_row(rowPointer[t][0], at, rowValid[t][0]);
            const uint4 a1 = load_row(rowPointer[t][1], at, rowValid[t][1]);
            const uint32_t step0[4] = {a0.x, a1.x, a0.y, a1.y};
            mma_bf16(acc[t], step0, b.x, b.y);
            const uint32_t step1[4] = {a0.z, a1.z, a0.w, a1.w};
            mma_bf16(acc[t], step1, b.z, b.w);
        }
    }
#pragma unroll
    for (int t = 0; t < MT; t++) {
#pragma unroll
        for (int h = 0; h < 2; h++) {
            if (!rowValid[t][h]) continue;
            const uint32_t row = rowBase + t * 16 + g + h * 8;
            __nv_bfloat162 pair = __floats2bfloat162_rn(acc[t][2 * h], acc[t][2 * h + 1]);
            *reinterpret_cast<__nv_bfloat162*>(y + static_cast<uint64_t>(row) * n + column0 + 2 * q) = pair;
        }
    }
}

// Sum of a 256-thread block in a fixed tree.
__device__ __forceinline__ float block_sum(float value, float* scratch) {
    for (int offset = 16; offset > 0; offset >>= 1) value += __shfl_xor_sync(0xffffffffu, value, offset);
    const uint32_t warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    __syncthreads();
    if (lane == 0) scratch[warp] = value;
    __syncthreads();
    float total = 0.0f;
    if (warp == 0) {
        total = lane < blockDim.x / 32 ? scratch[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) total += __shfl_xor_sync(0xffffffffu, total, offset);
        if (lane == 0) scratch[32] = total;
    }
    __syncthreads();
    return scratch[32];
}

// Qwen3RMSNorm of one 128-wide head held one element per thread of a 128-thread group: `weight * bf16(x / rms)`.
__device__ __forceinline__ float head_rms_norm(float value, float weight, float epsilon, float* scratch) {
    float square = value * value;
    for (int offset = 16; offset > 0; offset >>= 1) square += __shfl_xor_sync(0xffffffffu, square, offset);
    const uint32_t warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    __syncthreads();
    if (lane == 0) scratch[warp] = square;
    __syncthreads();
    const float total = scratch[0] + scratch[1] + scratch[2] + scratch[3];
    const float normalized = r(value * (1.0f / sqrtf(total / 128.0f + epsilon)));
    return r(weight * normalized);
}

// The rotary angle's cosine and sine of dimension `index` at `position` (Qwen3RotaryEmbedding, default type), BF16.
__device__ __forceinline__ void rope(uint64_t position, uint32_t index, uint32_t dim, float theta, float* c, float* s) {
    const uint32_t half = dim / 2;
    const uint32_t pair = index % half;
    const float inverse = 1.0f / powf(theta, static_cast<float>(2 * pair) / static_cast<float>(dim));
    const float angle = static_cast<float>(position) * inverse;
    *c = r(cosf(angle));
    *s = r(sinf(angle));
}

// One head's RoPE for one element (thread `index` of a 128-thread group); `others` holds the normalized head.
__device__ __forceinline__ float apply_rope(
        float value, const float* others, uint32_t index, uint32_t dim, uint64_t position, float theta) {
    float c, s;
    rope(position, index, dim, theta, &c, &s);
    const uint32_t half = dim / 2;
    const float rotated = index < half ? -others[index + half] : others[index - half];
    return r(r(value * c) + r(rotated * s));
}

}  // namespace dflash

using namespace dflash;

extern "C" __global__ __launch_bounds__(128) void euhedral_dflash_linear_bf16(
        const __nv_bfloat16* x, const __nv_bfloat16* w, __nv_bfloat16* y, uint32_t rows, uint32_t k, uint32_t n) {
    euhedral_pdl_begin();
    linear<1, 4>(x, w, y, rows, k, n);
}

// The same arithmetic per row as euhedral_dflash_linear_bf16, four row tiles per weight load (prefill contexts).
extern "C" __global__ __launch_bounds__(128) void euhedral_dflash_linear_rows_bf16(
        const __nv_bfloat16* x, const __nv_bfloat16* w, __nv_bfloat16* y, uint32_t rows, uint32_t k, uint32_t n) {
    euhedral_pdl_begin();
    linear<4, 2>(x, w, y, rows, k, n);
}

// Qwen3RMSNorm (plain weight): y = bf16(weight * bf16(x * rsqrt(mean(x^2) + eps))). One 256-thread CTA per row.
extern "C" __global__ __launch_bounds__(256) void euhedral_dflash_rms_norm_bf16(
        const __nv_bfloat16* x, const __nv_bfloat16* weight, __nv_bfloat16* y, uint32_t width, float epsilon) {
    euhedral_pdl_begin();
    __shared__ float scratch[33];
    const __nv_bfloat16* row = x + static_cast<uint64_t>(blockIdx.x) * width;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < width; i += blockDim.x) {
        const float value = bf(row[i]);
        sum = fmaf(value, value, sum);
    }
    const float inverse = 1.0f / sqrtf(block_sum(sum, scratch) / static_cast<float>(width) + epsilon);
    __nv_bfloat16* out = y + static_cast<uint64_t>(blockIdx.x) * width;
    for (uint32_t i = threadIdx.x; i < width; i += blockDim.x)
        out[i] = rn(bf(weight[i]) * r(bf(row[i]) * inverse));
}

// Grouped dynamic causal convolution over the rows of one block (GroupedDynamicCausalConv): for each tap `o`,
// out = bf16(out + bf16(base[o][c] * x[t - o][c])), then out = bf16(out + dynamic[t][part][o][c / group] * x[t - o][c]),
// with x[t - o] = 0 before the block's first row. `part` selects the prepare (0) or finish (1) kernel.
extern "C" __global__ __launch_bounds__(256) void euhedral_dflash_conv_bf16(
        const __nv_bfloat16* x,
        const __nv_bfloat16* dynamic,
        const __nv_bfloat16* base,
        __nv_bfloat16* y,
        uint32_t rows,
        uint32_t width,
        uint32_t group,
        uint32_t taps,
        uint32_t part) {
    euhedral_pdl_begin();
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= static_cast<uint64_t>(rows) * width) return;
    const uint32_t t = static_cast<uint32_t>(index / width), c = static_cast<uint32_t>(index % width);
    const uint32_t groups = width / group;
    const __nv_bfloat16* kernels = dynamic + static_cast<uint64_t>(t) * 2 * taps * groups + part * taps * groups;
    float out = 0.0f;
    for (uint32_t o = 0; o < taps; o++) {
        const float value = o <= t ? bf(x[static_cast<uint64_t>(t - o) * width + c]) : 0.0f;
        out = r(out + r(bf(base[(part * taps + o) * width + c]) * value));
        out = r(fmaf(bf(kernels[o * groups + c / group]), value, out));
    }
    y[index] = rn(out);
}

// The drafter's context keys and values of `rows` committed target rows at positions start .. start + rows - 1
// (`start` read from `position`): per KV head, k = RoPE(k_norm(k)); v as projected. Row p goes to ring slot
// p % window. `kv` rows hold the key heads, then the value heads. Grid (rows, KV heads), 128 threads (head dim).
extern "C" __global__ __launch_bounds__(128) void euhedral_dflash_context_kv_bf16(
        const __nv_bfloat16* kv,
        const __nv_bfloat16* keyNorm,
        __nv_bfloat16* ringKeys,
        __nv_bfloat16* ringValues,
        const uint64_t* position,
        uint32_t window,
        uint32_t keyValueHeads,
        float epsilon,
        float theta) {
    euhedral_pdl_begin();
    __shared__ float scratch[4];
    __shared__ float head[128];
    const uint32_t row = blockIdx.x, kvHead = blockIdx.y, i = threadIdx.x;
    const uint32_t width = keyValueHeads * 128;
    const uint64_t at = *position + row;
    const __nv_bfloat16* source = kv + static_cast<uint64_t>(row) * 2 * width + kvHead * 128;
    const float normalized = head_rms_norm(bf(source[i]), bf(keyNorm[i]), epsilon, scratch);
    head[i] = normalized;
    __syncthreads();
    const uint64_t slot = (at % window) * width + kvHead * 128 + i;
    ringKeys[slot] = rn(apply_rope(normalized, head, i, 128, at, theta));
    ringValues[slot] = source[width + i];
}

// The block's queries and keys: q = RoPE(q_norm(q)), k = RoPE(k_norm(k)) at positions start + row. Grid (rows,
// heads + KV heads), 128 threads. `kv` rows hold the key heads, then the value heads.
extern "C" __global__ __launch_bounds__(128) void euhedral_dflash_block_qk_bf16(
        const __nv_bfloat16* query,
        const __nv_bfloat16* kv,
        const __nv_bfloat16* queryNorm,
        const __nv_bfloat16* keyNorm,
        __nv_bfloat16* queryOut,
        __nv_bfloat16* keyOut,
        const uint64_t* position,
        uint32_t heads,
        uint32_t keyValueHeads,
        float epsilon,
        float theta) {
    euhedral_pdl_begin();
    __shared__ float scratch[4];
    __shared__ float head[128];
    const uint32_t row = blockIdx.x, i = threadIdx.x;
    const uint64_t at = *position + row;
    const bool isQuery = blockIdx.y < heads;
    const uint32_t h = isQuery ? blockIdx.y : blockIdx.y - heads;
    const float value = isQuery ? bf(query[static_cast<uint64_t>(row) * heads * 128 + h * 128 + i])
                                : bf(kv[static_cast<uint64_t>(row) * 2 * keyValueHeads * 128 + h * 128 + i]);
    const float normalized = head_rms_norm(value, bf(isQuery ? queryNorm[i] : keyNorm[i]), epsilon, scratch);
    head[i] = normalized;
    __syncthreads();
    const __nv_bfloat16 out = rn(apply_rope(normalized, head, i, 128, at, theta));
    if (isQuery)
        queryOut[static_cast<uint64_t>(row) * heads * 128 + h * 128 + i] = out;
    else
        keyOut[static_cast<uint64_t>(row) * keyValueHeads * 128 + h * 128 + i] = out;
}

// Sliding-window attention of the block's rows (non-causal within the block). Query row i at position start + i
// sees the context keys at positions max(0, start + i - window + 1) .. start - 1 (ring slots p % window) and every
// block key. Scores and softmax in FP32 with the 1/sqrt(128) scale, output rounded once. Grid (rows, KV heads), 256
// threads; the CTA serves the KV head's query heads.
constexpr uint32_t kMaxKeys = 2048 + 16;
constexpr uint32_t kGroup = 4;

extern "C" __global__ __launch_bounds__(256) void euhedral_dflash_attention_bf16(
        const __nv_bfloat16* query,
        const __nv_bfloat16* blockKeys,
        const __nv_bfloat16* kv,
        const __nv_bfloat16* ringKeys,
        const __nv_bfloat16* ringValues,
        __nv_bfloat16* out,
        const uint64_t* position,
        uint32_t rows,
        uint32_t window,
        uint32_t heads,
        uint32_t keyValueHeads) {
    euhedral_pdl_begin();
    extern __shared__ float shared[];
    float* scores = shared;                      // [kGroup][kMaxKeys]
    float* q = shared + kGroup * kMaxKeys;       // [kGroup][128]
    float* scratch = q + kGroup * 128;           // reductions
    const uint32_t row = blockIdx.x, kvHead = blockIdx.y;
    const uint32_t group = heads / keyValueHeads;
    const uint32_t width = keyValueHeads * 128;
    const uint64_t start = *position;
    const uint64_t at = start + row;
    const uint64_t first = at + 1 >= window ? at + 1 - window : 0;
    const uint32_t contextKeys = static_cast<uint32_t>(start > first ? start - first : 0);
    const uint32_t keys = contextKeys + rows;
    for (uint32_t index = threadIdx.x; index < group * 128; index += blockDim.x)
        q[index] = bf(query[static_cast<uint64_t>(row) * heads * 128 + (kvHead * group) * 128 + index]);
    __syncthreads();
    for (uint32_t key = threadIdx.x; key < keys; key += blockDim.x) {
        const __nv_bfloat16* k = key < contextKeys
                ? ringKeys + ((first + key) % window) * width + kvHead * 128
                : blockKeys + static_cast<uint64_t>(key - contextKeys) * width + kvHead * 128;
        float dot[kGroup] = {0.0f, 0.0f, 0.0f, 0.0f};
        uint4 row[16];
#pragma unroll
        for (uint32_t i = 0; i < 16; i++) row[i] = *reinterpret_cast<const uint4*>(k + 8 * i);
#pragma unroll
        for (uint32_t d = 0; d < 128; d += 8) {
            const uint4 packed = row[d / 8];
            const __nv_bfloat16* values = reinterpret_cast<const __nv_bfloat16*>(&packed);
#pragma unroll
            for (uint32_t e = 0; e < 8; e++) {
                const float kv_ = bf(values[e]);
#pragma unroll
                for (uint32_t h = 0; h < kGroup; h++)
                    if (h < group) dot[h] = fmaf(q[h * 128 + d + e], kv_, dot[h]);
            }
        }
#pragma unroll
        for (uint32_t h = 0; h < kGroup; h++)
            if (h < group) scores[h * kMaxKeys + key] = dot[h] * 0.08838834764831845f;
    }
    __syncthreads();
    // Softmax per query head: one warp per head.
    const uint32_t warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    if (warp < group) {
        float* s = scores + warp * kMaxKeys;
        float maximum = -CUDART_INF_F;
        for (uint32_t key = lane; key < keys; key += 32) maximum = fmaxf(maximum, s[key]);
        for (int offset = 16; offset > 0; offset >>= 1)
            maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, offset));
        float sum = 0.0f;
        for (uint32_t key = lane; key < keys; key += 32) {
            const float e = expf(s[key] - maximum);
            s[key] = e;
            sum += e;
        }
        for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, offset);
        if (lane == 0) scratch[warp] = 1.0f / sum;
    }
    __syncthreads();
    // Output: warp w takes keys w, w + 8, ...; lane l accumulates dimensions 4l .. 4l + 3 of every query head of the
    // group over its keys from one coalesced value row per key; the 8 warps' partial sums then add in a fixed order.
    float acc[kGroup][4];
#pragma unroll
    for (uint32_t h = 0; h < kGroup; h++) acc[h][0] = acc[h][1] = acc[h][2] = acc[h][3] = 0.0f;
    const uint32_t warps = blockDim.x / 32;
    // Keys go in batches of 8 per warp, every batch's value loads issued before its FMAs, so the loads overlap.
    constexpr uint32_t kBatch = 8;
    for (uint32_t base = warp * kBatch; base < keys; base += warps * kBatch) {
        uint2 packed[kBatch];
#pragma unroll
        for (uint32_t i = 0; i < kBatch; i++) {
            const uint32_t key = base + i;
            if (key >= keys) {
                packed[i] = make_uint2(0, 0);
                continue;
            }
            const __nv_bfloat16* v = key < contextKeys
                    ? ringValues + ((first + key) % window) * width + kvHead * 128
                    : kv + static_cast<uint64_t>(key - contextKeys) * 2 * width + width + kvHead * 128;
            packed[i] = *reinterpret_cast<const uint2*>(v + 4 * lane);
        }
#pragma unroll
        for (uint32_t i = 0; i < kBatch; i++) {
            const uint32_t key = base + i;
            if (key >= keys) break;
            const __nv_bfloat16* values = reinterpret_cast<const __nv_bfloat16*>(&packed[i]);
            float x[4];
#pragma unroll
            for (uint32_t e = 0; e < 4; e++) x[e] = bf(values[e]);
#pragma unroll
            for (uint32_t h = 0; h < kGroup; h++) {
                if (h >= group) continue;
                const float p = scores[h * kMaxKeys + key];
#pragma unroll
                for (uint32_t e = 0; e < 4; e++) acc[h][e] = fmaf(p, x[e], acc[h][e]);
            }
        }
    }
    __syncthreads();
    // Reuse the scores as the warps' partial sums: [warp][head][128].
    float* partial = scores;
#pragma unroll
    for (uint32_t h = 0; h < kGroup; h++)
#pragma unroll
        for (uint32_t e = 0; e < 4; e++) partial[(warp * kGroup + h) * 128 + 4 * lane + e] = acc[h][e];
    __syncthreads();
    for (uint32_t index = threadIdx.x; index < group * 128; index += blockDim.x) {
        const uint32_t h = index / 128, d = index % 128;
        float sum = 0.0f;
        for (uint32_t w = 0; w < warps; w++) sum += partial[(w * kGroup + h) * 128 + d];
        out[static_cast<uint64_t>(row) * heads * 128 + (kvHead * group + h) * 128 + d] = rn(sum * scratch[h]);
    }
}

// Qwen3MLP's activation: bf16(bf16(silu(gate)) * up); rows hold the gate, then the up projection.
extern "C" __global__ __launch_bounds__(256) void euhedral_dflash_swiglu_bf16(
        const __nv_bfloat16* gateUp, __nv_bfloat16* y, uint32_t rows, uint32_t intermediate) {
    euhedral_pdl_begin();
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= static_cast<uint64_t>(rows) * intermediate) return;
    const uint32_t row = static_cast<uint32_t>(index / intermediate), c = static_cast<uint32_t>(index % intermediate);
    const __nv_bfloat16* source = gateUp + static_cast<uint64_t>(row) * 2 * intermediate;
    const float gate = bf(source[c]);
    const float activated = r(gate / (1.0f + expf(-gate)));
    y[index] = rn(activated * bf(source[intermediate + c]));
}

// The 16 largest logits of each row, in descending order (equal values: lower token first; NaN never). Values stay
// BF16, indices are token ids. One 256-thread CTA per row.
constexpr int kTop = 16;

__device__ __forceinline__ bool before(float a, int32_t ia, float b, int32_t ib) {
    return a > b || (a == b && ia < ib);
}

// The 16 best of `count` (value, token) candidates held by this 256-thread CTA, merged in 16 rounds: each thread
// offers its best remaining candidate and the CTA takes the best offer. Writes them in order to `values` and
// `indices` (any float or BF16 sink through `store`).
__device__ __forceinline__ void top16_merge(float* candidateValue, int32_t* candidateIndex, int* head,
        float* bestValue, int32_t* bestIndex, int32_t* bestThread, float* outValues, int32_t* outIndices) {
    head[threadIdx.x] = 0;
    __syncthreads();
    const uint32_t warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    for (int out = 0; out < kTop; out++) {
        const int at = head[threadIdx.x];
        float value = at < kTop ? candidateValue[threadIdx.x * kTop + at] : -CUDART_INF_F;
        int32_t index = at < kTop ? candidateIndex[threadIdx.x * kTop + at] : 0x7fffffff;
        int32_t owner = static_cast<int32_t>(threadIdx.x);
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, value, offset);
            const int32_t oi = __shfl_xor_sync(0xffffffffu, index, offset);
            const int32_t oo = __shfl_xor_sync(0xffffffffu, owner, offset);
            if (before(ov, oi, value, index)) {
                value = ov;
                index = oi;
                owner = oo;
            }
        }
        if (lane == 0) {
            bestValue[warp] = value;
            bestIndex[warp] = index;
            bestThread[warp] = owner;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            int best = 0;
            for (int w = 1; w < 8; w++)
                if (before(bestValue[w], bestIndex[w], bestValue[best], bestIndex[best])) best = w;
            outValues[out] = bestValue[best];
            outIndices[out] = bestIndex[best];
            head[bestThread[best]]++;
        }
        __syncthreads();
    }
}

// Inserts (value, index) into a thread's sorted top-16.
__device__ __forceinline__ void top16_insert(float* v, int32_t* ix, float value, int32_t index) {
    if (!before(value, index, v[kTop - 1], ix[kTop - 1])) return;
#pragma unroll
    for (int i = 0; i < kTop; i++) {
        if (before(value, index, v[i], ix[i])) {
            const float tv = v[i];
            const int32_t ti = ix[i];
            v[i] = value;
            ix[i] = index;
            value = tv;
            index = ti;
        }
    }
}

// Pass 1: CTA (row, split) finds the top 16 of its slice of the row's logits (NaN never) into `partial`
// [row][split][16] values (FP32) and tokens. Grid (rows, splits), 256 threads.
extern "C" __global__ __launch_bounds__(256) void euhedral_dflash_topk_partial_bf16(
        const __nv_bfloat16* logits, uint32_t vocabulary, float* partialValues, int32_t* partialIndices) {
    euhedral_pdl_begin();
    __shared__ float candidateValue[256 * kTop];
    __shared__ int32_t candidateIndex[256 * kTop];
    __shared__ float bestValue[8];
    __shared__ int32_t bestIndex[8];
    __shared__ int32_t bestThread[8];
    __shared__ int head[256];
    const uint32_t splits = gridDim.y, split = blockIdx.y;
    const uint32_t span = (vocabulary + splits - 1) / splits;
    const uint32_t begin = split * span, end = min(vocabulary, begin + span);
    const __nv_bfloat16* row = logits + static_cast<uint64_t>(blockIdx.x) * vocabulary;
    float v[kTop];
    int32_t ix[kTop];
#pragma unroll
    for (int i = 0; i < kTop; i++) {
        v[i] = -CUDART_INF_F;
        ix[i] = 0x7fffffff;
    }
    for (uint32_t token = begin + threadIdx.x; token < end; token += blockDim.x) {
        const float value = bf(row[token]);
        if (value == value) top16_insert(v, ix, value, static_cast<int32_t>(token));
    }
#pragma unroll
    for (int i = 0; i < kTop; i++) {
        candidateValue[threadIdx.x * kTop + i] = v[i];
        candidateIndex[threadIdx.x * kTop + i] = ix[i];
    }
    const uint64_t out = (static_cast<uint64_t>(blockIdx.x) * splits + split) * kTop;
    top16_merge(candidateValue, candidateIndex, head, bestValue, bestIndex, bestThread, partialValues + out,
            partialIndices + out);
}

// Pass 2: the row's top 16 of its splits' candidates, in descending order (equal values: lower token first). Values
// return to BF16 exactly (they are BF16 logits). Grid rows, 256 threads.
extern "C" __global__ __launch_bounds__(256) void euhedral_dflash_topk_merge_bf16(
        const float* partialValues, const int32_t* partialIndices, uint32_t splits, __nv_bfloat16* values,
        int32_t* indices) {
    euhedral_pdl_begin();
    __shared__ float candidateValue[256 * kTop];
    __shared__ int32_t candidateIndex[256 * kTop];
    __shared__ float bestValue[8];
    __shared__ int32_t bestIndex[8];
    __shared__ int32_t bestThread[8];
    __shared__ int head[256];
    __shared__ float outValues[kTop];
    __shared__ int32_t outIndices[kTop];
    const uint64_t base = static_cast<uint64_t>(blockIdx.x) * splits * kTop;
    float v[kTop];
    int32_t ix[kTop];
#pragma unroll
    for (int i = 0; i < kTop; i++) {
        v[i] = -CUDART_INF_F;
        ix[i] = 0x7fffffff;
    }
    for (uint32_t i = threadIdx.x; i < splits * kTop; i += blockDim.x) top16_insert(v, ix, partialValues[base + i],
            partialIndices[base + i]);
#pragma unroll
    for (int i = 0; i < kTop; i++) {
        candidateValue[threadIdx.x * kTop + i] = v[i];
        candidateIndex[threadIdx.x * kTop + i] = ix[i];
    }
    top16_merge(candidateValue, candidateIndex, head, bestValue, bestIndex, bestThread, outValues, outIndices);
    if (threadIdx.x < kTop) {
        values[blockIdx.x * kTop + threadIdx.x] = rn(outValues[threadIdx.x]);
        indices[blockIdx.x * kTop + threadIdx.x] = outIndices[threadIdx.x];
    }
}

// The candidate selector at temperature 0 (CandidateSelector.select): from the anchor, position by position,
// score[k] = bf16(unary[k] + bf16(sum_r bf16(predecessor[prev][r] * hidden[r]) * successor[candidate k][r])) and the
// path takes the first highest score. `hidden` is the selector's projection of each proposal row; the codebooks may
// be read in place from mapped host memory. Writes the path's tokens and each step's scores. One CTA of 512
// threads: a warp per candidate.
extern "C" __global__ __launch_bounds__(512) void euhedral_dflash_select_bf16(
        const __nv_bfloat16* hidden,
        const __nv_bfloat16* values,
        const int32_t* indices,
        const __nv_bfloat16* predecessor,
        const __nv_bfloat16* successor,
        const int32_t* anchor,
        uint32_t positions,
        uint32_t rank,
        int32_t* tokens,
        float* scores) {
    euhedral_pdl_begin();
    __shared__ float product[1024];
    __shared__ float score[kTop];
    __shared__ int32_t previous;
    const uint32_t warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    if (threadIdx.x == 0) previous = *anchor;
    __syncthreads();
    for (uint32_t p = 0; p < positions; p++) {
        const __nv_bfloat16* pred = predecessor + static_cast<uint64_t>(previous) * rank;
        for (uint32_t i = threadIdx.x; i < rank; i += blockDim.x)
            product[i] = r(bf(pred[i]) * bf(hidden[static_cast<uint64_t>(p) * rank + i]));
        __syncthreads();
        if (warp < kTop) {
            const int32_t candidate = indices[p * kTop + warp];
            const __nv_bfloat16* succ = successor + static_cast<uint64_t>(candidate) * rank;
            float dot = 0.0f;
            for (uint32_t i = lane; i < rank; i += 32) dot = fmaf(product[i], bf(succ[i]), dot);
            for (int offset = 16; offset > 0; offset >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, offset);
            if (lane == 0) score[warp] = r(bf(values[p * kTop + warp]) + r(dot));
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            int best = 0;
            for (int k = 1; k < kTop; k++)
                if (score[k] > score[best]) best = k;
            for (int k = 0; k < kTop; k++) scores[p * kTop + k] = score[k];
            previous = indices[p * kTop + best];
            tokens[p] = previous;
        }
        __syncthreads();
    }
}
