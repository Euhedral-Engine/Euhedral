#pragma once
// Causal attention over BF16 K/V caches.
#include "reductions.cuh"
#include "common/pdl.cuh"

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
