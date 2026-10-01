#pragma once
// KV cache append: BF16 K/V rows into NVFP4 cache pages, and the exact BF16 cache append.
#include "nvfp4_kv.cuh"
#include "reductions.cuh"
#include "common/pdl.cuh"

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
