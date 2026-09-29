#include <cuda_bf16.h>
#include <cuda_runtime.h>

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
        result = lane < half ? value * cosine - paired * sine : value * cosine + paired * sine;
    }
    // The in-place query path cannot overwrite another warp's RoPE pair read.
    if (CACHE) __syncthreads();
    if (CACHE && !isQuery)
        keyCache[(startPosition + row) * keyValueHeads * headDim
                + static_cast<uint64_t>(head - queryHeads) * headDim + lane] = __float2bfloat16_rn(result);
    else output[outputOffset + lane] = __float2bfloat16_rn(result);
}

extern "C" __global__ void euhedral_attention_qk_norm_rope_bf16(
        const __nv_bfloat16* queryKey, const __nv_bfloat16* queryNorm, const __nv_bfloat16* keyNorm,
        __nv_bfloat16* output, uint32_t rows, uint32_t queryHeads, uint32_t keyValueHeads,
        uint32_t headDim, uint32_t rotaryDim, uint64_t startPosition, float epsilon, double ropeTheta) {
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
