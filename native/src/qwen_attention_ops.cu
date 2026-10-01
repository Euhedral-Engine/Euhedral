#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include "attention/nvfp4_kv.cuh"
#include "attention/nvfp4_attention.cuh"
#include "attention/nvfp4_prefill32.cuh"
#include "pdl.cuh"

// Four warp-owned rows per CTA. Page-table entries point to token-major pages;
// each head row has 128 packed-code bytes followed by 16 E4M3 scale bytes.
extern "C" __global__ __launch_bounds__(128) void euhedral_attention_kv_append_nvfp4(
        const __nv_bfloat16* queryKey, const __nv_bfloat16* gateValue,
        unsigned char* const* keyPages, unsigned char* const* valuePages,
        unsigned int rows, unsigned int queryWidth, unsigned int keyValueWidth,
        unsigned long long startPosition) {
    euhedral_pdl_begin();
    const unsigned int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const unsigned int heads = keyValueWidth / 256;
    const unsigned int rowHead = blockIdx.x * 4 + warp;
    if (rowHead >= rows * heads) return;
    const unsigned int row = rowHead / heads, head = rowHead % heads;
    const unsigned long long position = startPosition + row;
    const unsigned long long source = (unsigned long long)row * (queryWidth + keyValueWidth)
            + queryWidth + head * 256;
    const unsigned int offset = ((position % 256) * heads + head) * 144;
    unsigned char* key = keyPages[position / 256] + offset;
    unsigned char* value = valuePages[position / 256] + offset;
    __shared__ float scratch[4][256];
    nvfp4kv::quantize_row(queryKey + source, key, key + 128, scratch[warp], lane);
    nvfp4kv::quantize_row(gateValue + source, value, value + 128, scratch[warp], lane);
}


using uint32_t = unsigned int;
using uint64_t = unsigned long long;

__device__ __forceinline__ float attention_reduce_sum(float value, float* scratch) {
    scratch[threadIdx.x] = value;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride != 0; stride >>= 1) {
        if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        __syncthreads();
    }
    const float result = scratch[0];
    __syncthreads();
    return result;
}

// RoPE rotation with explicit rounding (no FMA contraction), shared by both QK norm layouts so they
// agree bit for bit whatever code the compiler generates around them.
static __device__ __forceinline__ float rope_rotate(float value, float paired, float cosine, float sine, bool firstHalf) {
    const float direct = __fmul_rn(value, cosine), cross = __fmul_rn(paired, sine);
    return firstHalf ? __fsub_rn(direct, cross) : __fadd_rn(direct, cross);
}

template<bool CACHE>
static __device__ __forceinline__ void qk_norm_rope(
        const __nv_bfloat16* queryKey,
        const __nv_bfloat16* queryNorm,
        const __nv_bfloat16* keyNorm,
        __nv_bfloat16* output,
        uint32_t rows,
        uint32_t queryHeads,
        uint32_t keyValueHeads,
        uint32_t headDim,
        uint32_t rotaryDim,
        uint64_t startPosition,
        float epsilon,
        double ropeTheta, __nv_bfloat16* keyCache) {
    const uint32_t rowHead = blockIdx.x;
    const uint32_t row = rowHead / (queryHeads + keyValueHeads);
    const uint32_t head = rowHead % (queryHeads + keyValueHeads);
    if (row >= rows || blockDim.x != headDim) return;

    const uint32_t lane = threadIdx.x;
    const uint32_t projectedWidth = (queryHeads + keyValueHeads) * headDim;
    const uint64_t inputOffset = static_cast<uint64_t>(row) * projectedWidth + static_cast<uint64_t>(head) * headDim;
    const uint64_t outputOffset = inputOffset;
    const bool isQuery = head < queryHeads;
    const __nv_bfloat16* norm = isQuery ? queryNorm : keyNorm;
    float value = __bfloat162float(queryKey[inputOffset + lane]);
    __shared__ float scratch[256];
    const float sum = attention_reduce_sum(value * value, scratch);
    const float inverse = rsqrtf(sum / static_cast<float>(headDim) + epsilon);
    value = value * inverse * (1.0f + __bfloat162float(norm[lane]));

    float result = value;
    if (lane < rotaryDim) {
        const uint32_t half = rotaryDim / 2;
        const uint32_t pair = lane < half ? lane + half : lane - half;
        const uint32_t frequencyIndex = lane < half ? lane : pair;
        const double exponent = (2.0 * static_cast<double>(frequencyIndex)) / static_cast<double>(rotaryDim);
        const double angle = static_cast<double>(startPosition + row) * pow(ropeTheta, -exponent);
        const float cosine = static_cast<float>(cos(angle));
        const float sine = static_cast<float>(sin(angle));
        const float paired = __bfloat162float(queryKey[inputOffset + pair]) * inverse
                * (1.0f + __bfloat162float(norm[pair]));
        result = rope_rotate(value, paired, cosine, sine, lane < half);
    }
    // The in-place query path cannot overwrite another warp's RoPE pair read.
    __syncthreads();
    if (CACHE && !isQuery)
        keyCache[(startPosition + row) * keyValueHeads * headDim
                + static_cast<uint64_t>(head - queryHeads) * headDim + lane] = __float2bfloat16_rn(result);
    else output[outputOffset + lane] = __float2bfloat16_rn(result);
}

// Row-owned variant for 256-dimension heads: one CTA per row, one warp per head (eight dimensions
// per lane). The RoPE cosines and sines, computed in FP64 as above, are shared by every head of the
// row instead of recomputed per element. The RMS sum reproduces the 256-thread tree above (strides
// 128, 64 and 32 in registers, then 16 .. 1 across lanes), so results match it bit for bit.
// Launch: grid = rows, 256 threads; requires headDim == 256.
template<bool CACHE>
static __device__ __forceinline__ void qk_norm_rope_rows(
        const __nv_bfloat16* queryKey, const __nv_bfloat16* queryNorm, const __nv_bfloat16* keyNorm,
        __nv_bfloat16* output, uint32_t rows, uint32_t queryHeads, uint32_t keyValueHeads,
        uint32_t headDim, uint32_t rotaryDim, uint64_t startPosition, float epsilon, double ropeTheta,
        __nv_bfloat16* keyCache) {
    const uint32_t row = blockIdx.x;
    if (row >= rows || headDim != 256u) return;
    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    const uint32_t heads = queryHeads + keyValueHeads, half = rotaryDim / 2;
    __shared__ float cosines[128], sines[128];
    for (uint32_t f = threadIdx.x; f < half; f += blockDim.x) {
        const double exponent = (2.0 * static_cast<double>(f)) / static_cast<double>(rotaryDim);
        const double angle = static_cast<double>(startPosition + row) * pow(ropeTheta, -exponent);
        cosines[f] = static_cast<float>(cos(angle));
        sines[f] = static_cast<float>(sin(angle));
    }
    __syncthreads();
    const uint64_t projectedWidth = static_cast<uint64_t>(heads) * 256u;
    for (uint32_t head = warp; head < heads; head += blockDim.x >> 5) {
        const uint64_t offset = row * projectedWidth + static_cast<uint64_t>(head) * 256u;
        const bool isQuery = head < queryHeads;
        const __nv_bfloat16* norm = isQuery ? queryNorm : keyNorm;
        float values[8], squares[8];
#pragma unroll
        for (int k = 0; k < 8; k++) {
            values[k] = __bfloat162float(queryKey[offset + lane + 32u * k]);
            squares[k] = values[k] * values[k];
        }
#pragma unroll
        for (int k = 0; k < 4; k++) squares[k] += squares[k + 4];
#pragma unroll
        for (int k = 0; k < 2; k++) squares[k] += squares[k + 2];
        float sum = squares[0] + squares[1];
#pragma unroll
        for (int stride = 16; stride; stride >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, stride);
        sum = __shfl_sync(0xffffffffu, sum, 0);
        const float inverse = rsqrtf(sum / 256.0f + epsilon);
        float results[8];
#pragma unroll
        for (int k = 0; k < 8; k++) {
            const uint32_t index = lane + 32u * k;
            const float value = values[k] * inverse * (1.0f + __bfloat162float(norm[index]));
            results[k] = value;
            if (index < rotaryDim) {
                const uint32_t pair = index < half ? index + half : index - half;
                const uint32_t frequency = index < half ? index : pair;
                const float paired = __bfloat162float(queryKey[offset + pair]) * inverse
                        * (1.0f + __bfloat162float(norm[pair]));
                results[k] = rope_rotate(value, paired, cosines[frequency], sines[frequency], index < half);
            }
        }
        // The in-place query path cannot overwrite a RoPE pair another lane still reads.
        __syncwarp();
#pragma unroll
        for (int k = 0; k < 8; k++) {
            const uint32_t index = lane + 32u * k;
            if (CACHE && !isQuery)
                keyCache[(startPosition + row) * keyValueHeads * 256u
                        + static_cast<uint64_t>(head - queryHeads) * 256u + index] = __float2bfloat16_rn(results[k]);
            else output[offset + index] = __float2bfloat16_rn(results[k]);
        }
    }
}

extern "C" __global__ __launch_bounds__(256) void euhedral_attention_qk_norm_rope_rows_bf16(
        const __nv_bfloat16* queryKey, const __nv_bfloat16* queryNorm, const __nv_bfloat16* keyNorm,
        __nv_bfloat16* output, uint32_t rows, uint32_t queryHeads, uint32_t keyValueHeads,
        uint32_t headDim, uint32_t rotaryDim, uint64_t startPosition, float epsilon, double ropeTheta) {
    euhedral_pdl_begin();
    qk_norm_rope_rows<false>(queryKey, queryNorm, keyNorm, output, rows, queryHeads, keyValueHeads,
            headDim, rotaryDim, startPosition, epsilon, ropeTheta, nullptr);
}

extern "C" __global__ __launch_bounds__(256) void euhedral_attention_qk_norm_cache_rows_bf16(
        const __nv_bfloat16* queryKey, const __nv_bfloat16* queryNorm, const __nv_bfloat16* keyNorm,
        __nv_bfloat16* output, uint32_t rows, uint32_t queryHeads, uint32_t keyValueHeads,
        uint32_t headDim, uint32_t rotaryDim, uint64_t startPosition, float epsilon, double ropeTheta,
        __nv_bfloat16* keyCache) {
    qk_norm_rope_rows<true>(queryKey, queryNorm, keyNorm, output, rows, queryHeads, keyValueHeads,
            headDim, rotaryDim, startPosition, epsilon, ropeTheta, keyCache);
}

extern "C" __global__ void euhedral_attention_qk_norm_rope_bf16(
        const __nv_bfloat16* queryKey, const __nv_bfloat16* queryNorm, const __nv_bfloat16* keyNorm,
        __nv_bfloat16* output, uint32_t rows, uint32_t queryHeads, uint32_t keyValueHeads,
        uint32_t headDim, uint32_t rotaryDim, uint64_t startPosition, float epsilon, double ropeTheta) {
    euhedral_pdl_begin();
    qk_norm_rope<false>(queryKey, queryNorm, keyNorm, output, rows, queryHeads, keyValueHeads,
            headDim, rotaryDim, startPosition, epsilon, ropeTheta, nullptr);
}

extern "C" __global__ void euhedral_attention_qk_norm_cache_bf16(
        const __nv_bfloat16* queryKey, const __nv_bfloat16* queryNorm, const __nv_bfloat16* keyNorm,
        __nv_bfloat16* output, uint32_t rows, uint32_t queryHeads, uint32_t keyValueHeads,
        uint32_t headDim, uint32_t rotaryDim, uint64_t startPosition, float epsilon, double ropeTheta,
        __nv_bfloat16* keyCache) {
    qk_norm_rope<true>(queryKey, queryNorm, keyNorm, output, rows, queryHeads, keyValueHeads,
            headDim, rotaryDim, startPosition, epsilon, ropeTheta, keyCache);
}

extern "C" __global__ void euhedral_attention_kv_append_bf16(
        const __nv_bfloat16* queryKey,
        const __nv_bfloat16* gateValue,
        __nv_bfloat16* keyCache,
        __nv_bfloat16* valueCache,
        uint32_t rows,
        uint32_t queryWidth,
        uint32_t keyValueWidth,
        uint64_t startPosition) {
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t count = static_cast<uint64_t>(rows) * keyValueWidth;
    if (index >= count) return;
    const uint32_t row = static_cast<uint32_t>(index / keyValueWidth);
    const uint32_t column = static_cast<uint32_t>(index % keyValueWidth);
    const uint64_t sourceOffset = static_cast<uint64_t>(row) * (queryWidth + keyValueWidth);
    const uint64_t destinationOffset = (startPosition + row) * keyValueWidth + column;
    keyCache[destinationOffset] = queryKey[sourceOffset + queryWidth + column];
    valueCache[destinationOffset] = gateValue[sourceOffset + queryWidth + column];
}

extern "C" __global__ void euhedral_attention_causal_bf16(
        const __nv_bfloat16* queryKey,
        const __nv_bfloat16* gateValue,
        const __nv_bfloat16* keyCache,
        const __nv_bfloat16* valueCache,
        __nv_bfloat16* output,
        uint32_t rows,
        uint32_t queryHeads,
        uint32_t keyValueHeads,
        uint32_t headDim,
        uint32_t cacheLength,
        uint64_t startPosition) {
    const uint32_t rowHead = blockIdx.x;
    const uint32_t row = rowHead / queryHeads;
    const uint32_t queryHead = rowHead % queryHeads;
    if (row >= rows || blockDim.x != headDim) return;

    const uint32_t lane = threadIdx.x;
    const uint32_t queryWidth = queryHeads * headDim;
    const uint32_t keyValueWidth = keyValueHeads * headDim;
    const uint32_t keyValueHead = queryHead / (queryHeads / keyValueHeads);
    const uint64_t queryOffset = static_cast<uint64_t>(row) * (queryWidth + keyValueWidth)
            + static_cast<uint64_t>(queryHead) * headDim;
    const uint64_t gateOffset = queryOffset;
    const uint64_t maxKey = startPosition + row + 1;
    if (maxKey > cacheLength) return;

    const float query = __bfloat162float(queryKey[queryOffset + lane]);
    float maximum = -3.402823466e+38F;
    float denominator = 0.0f;
    float accumulator = 0.0f;
    __shared__ float scratch[256];
    const float scale = rsqrtf(static_cast<float>(headDim));
    for (uint64_t keyIndex = 0; keyIndex < maxKey; keyIndex++) {
        const uint64_t keyOffset = keyIndex * keyValueWidth + static_cast<uint64_t>(keyValueHead) * headDim;
        const float key = __bfloat162float(keyCache[keyOffset + lane]);
        const float score = attention_reduce_sum(query * key, scratch) * scale;
        const float nextMaximum = fmaxf(maximum, score);
        const float previousScale = expf(maximum - nextMaximum);
        const float currentScale = expf(score - nextMaximum);
        denominator = denominator * previousScale + currentScale;
        const float value = __bfloat162float(valueCache[keyOffset + lane]);
        accumulator = accumulator * previousScale + value * currentScale;
        maximum = nextMaximum;
    }
    const float gate = __bfloat162float(gateValue[gateOffset + lane]);
    const float sigmoid = 1.0f / (1.0f + expf(-gate));
    output[static_cast<uint64_t>(row) * queryWidth + static_cast<uint64_t>(queryHead) * headDim + lane]
            = __float2bfloat16_rn((accumulator / denominator) * sigmoid);
}
