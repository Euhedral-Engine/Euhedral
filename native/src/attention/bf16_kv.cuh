#pragma once
// BF16 key/value cache rows for D256 attention heads.
//
// Storage contract: the page tables of the NVFP4 cache (256-token pages, token-major KV heads), with 512-byte rows:
// the 256 BF16 values of the head row as computed, unrotated. Attention reads them as FP32 (the per-head decode
// kernels) or widens them to FP16 tile rows (the tensor-core kernels): exact for every value from 2^-14 to 65504,
// rounded below it, and saturated to +-65504 above it, so the tiles stay finite.

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace bf16kv {

constexpr unsigned kRowBytes = 512;

__device__ __forceinline__ const unsigned char* cache_row(
        const unsigned char* const* pages, unsigned int token, unsigned int head, unsigned int heads) {
    return pages[token / 256] + ((token % 256) * heads + head) * kRowBytes;
}

__device__ __forceinline__ float element(const unsigned char* row, unsigned int d) {
    return __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(row)[d]);
}

// The low and high BF16 of a word, widened to FP32.
__device__ __forceinline__ float low(unsigned word) { return __uint_as_float(word << 16); }
__device__ __forceinline__ float high(unsigned word) { return __uint_as_float(word & 0xffff0000u); }

__device__ __forceinline__ float saturate(float value) { return fminf(fmaxf(value, -65504.0f), 65504.0f); }

// Two BF16 (a word, low first) as an FP16 pair, saturating.
__device__ __forceinline__ unsigned widen2(unsigned word) {
    const __half2 pair = __floats2half2_rn(saturate(low(word)), saturate(high(word)));
    return *reinterpret_cast<const unsigned*>(&pair);
}

// Eight BF16 (16 bytes) as four FP16 pairs.
__device__ __forceinline__ uint4 widen8(uint4 words) {
    return make_uint4(widen2(words.x), widen2(words.y), widen2(words.z), widen2(words.w));
}

// Writes 32 cached tokens of one KV head as FP16 rows of 256 values `stride` halves apart; tokens from `last` are
// zero. Each pass of the block writes 16 values (two 16-byte loads and stores) per thread.
__device__ __forceinline__ void stage_tokens32(__half* kv, unsigned stride, const unsigned char* const* pages,
        unsigned base, unsigned last, unsigned head, unsigned heads) {
    for (unsigned g = threadIdx.x; g < 32u * 16u; g += blockDim.x) {
        const unsigned token = base + g / 16u, group = g % 16u;
        uint4 lo = make_uint4(0, 0, 0, 0), hi = lo;
        if (token < last) {
            const uint4* row = reinterpret_cast<const uint4*>(cache_row(pages, token, head, heads)) + group * 2;
            lo = widen8(row[0]);
            hi = widen8(row[1]);
        }
        uint4* dst = reinterpret_cast<uint4*>(kv + (g / 16u) * stride + group * 16);
        dst[0] = lo;
        dst[1] = hi;
    }
}

}  // namespace bf16kv
