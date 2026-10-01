#pragma once
#include "nvfp4_kv.cuh"
#include <mma.h>

namespace nvfp4kv {
// Expands 32 cached tokens of one KV head into FP16 rows of 256 (exact). Each thread expands whole
// 16-element groups: one 8-byte code load, one scale byte and two 16-byte shared stores.
__device__ __forceinline__ void stage_tokens32(__half* kv, const unsigned char* const* pages, unsigned int base,
        unsigned int last, unsigned int head, unsigned int heads) {
    for (unsigned int g = threadIdx.x; g < 32u * kGroups; g += blockDim.x) {
        const unsigned int token = base + g / kGroups, group = g % kGroups;
        __half2 out[8];
        if (token < last) {
            const unsigned char* row = cache_row(pages, token, head, heads);
            const uint2 codes = *reinterpret_cast<const uint2*>(row + group * (kGroup / 2));
            const float scale = e4m3_decode(row[kCodeBytes + group]);
            __half2 (&lo)[4] = *reinterpret_cast<__half2 (*)[4]>(&out[0]);
            __half2 (&hi)[4] = *reinterpret_cast<__half2 (*)[4]>(&out[4]);
            dequantize8_f16(codes.x, scale, lo);
            dequantize8_f16(codes.y, scale, hi);
        } else {
#pragma unroll
            for (int i = 0; i < 8; i++) out[i] = __floats2half2_rn(0.0f, 0.0f);
        }
        uint4* dst = reinterpret_cast<uint4*>(kv + (g / kGroups) * kHeadDim + group * kGroup);
        dst[0] = *reinterpret_cast<const uint4*>(&out[0]);
        dst[1] = *reinterpret_cast<const uint4*>(&out[4]);
    }
}
}  // namespace nvfp4kv

// 32 query rows x 32 keys; K and V share the same staging storage. Accumulation
// and online softmax are FP32; rotated Q and probabilities are FP16 MMA operands.
// KV expansion into FP16 is exact. No quadratic score buffer is materialized.
// EXACT keeps one thread per query row for the softmax statistics, summing each row's 32
// probabilities in key order; otherwise four threads share a row and combine partial sums.
template<bool EXACT>
static __device__ __forceinline__ void attention_prefill32_nvfp4(
        const __nv_bfloat16* queryKey, const __nv_bfloat16* gateValue,
        const unsigned char* const* keyPages, const unsigned char* const* valuePages,
        __nv_bfloat16* output, unsigned int rows, unsigned int queryHeads, unsigned int keyHeads,
        unsigned int cacheLength, unsigned long long start) {
    using namespace nvcuda;
    const unsigned int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const unsigned int head = blockIdx.x % queryHeads, first = (blockIdx.x / queryHeads) * 32;
    const unsigned int kh = head / (queryHeads / keyHeads), width = (queryHeads + keyHeads) * 256;
    __shared__ __align__(32) __half q[32 * 256];
    __shared__ __align__(32) __half kv[32 * 256];
    __shared__ __align__(32) float scores[32 * 32];
    __shared__ __align__(32) __half probabilities[32 * 32];
    __shared__ float maximum[32], denominator[32], correction[32];
    for (unsigned int r = warp; r < 32; r += 4) {
        float values[8];
#pragma unroll
        for (int d = 0; d < 8; d++) values[d] = first + r < rows
                ? __bfloat162float(queryKey[(unsigned long long)(first + r) * width + head * 256 + lane + d * 32]) : 0;
        nvfp4kv::hadamard256(values, lane);
#pragma unroll
        for (int d = 0; d < 8; d++) q[r * 256 + lane + d * 32] = __float2half_rn(values[d]);
        if (lane == 0) { maximum[r] = -__int_as_float(0x7f800000); denominator[r] = 0; }
    }
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> out[8];
#pragma unroll
    for (int f = 0; f < 8; f++) wmma::fill_fragment(out[f], 0.0f);
    __syncthreads();
    const unsigned int last = min(cacheLength, (unsigned int)(start + min(first + 32, rows)));
    for (unsigned int base = 0; base < last; base += 32) {
        nvfp4kv::stage_tokens32(kv, keyPages, base, last, kh, keyHeads);
        __syncthreads();
        {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> score;
            wmma::fill_fragment(score, 0.0f);
            for (int d = 0; d < 256; d += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
                wmma::load_matrix_sync(a, q + (warp / 2) * 16 * 256 + d, 256);
                wmma::load_matrix_sync(b, kv + (warp % 2) * 16 * 256 + d, 256);
                wmma::mma_sync(score, a, b, score);
            }
            wmma::store_matrix_sync(scores + (warp / 2) * 16 * 32 + (warp % 2) * 16, score, 32, wmma::mem_row_major);
        }
        __syncthreads();
        if (!EXACT) {
            // Four threads per row, eight keys each; max is order independent, the sum is not.
            const unsigned int r = threadIdx.x >> 2, part = threadIdx.x & 3u;
            float next = maximum[r];
#pragma unroll
            for (unsigned int j = 0; j < 8; j++) {
                const unsigned int t = part * 8u + j;
                const bool visible = first + r < rows && base + t < last && base + t <= start + first + r;
                const float score = visible ? scores[r * 32 + t] * 0.0625f : -__int_as_float(0x7f800000);
                scores[r * 32 + t] = score;
                next = fmaxf(next, score);
            }
            next = fmaxf(next, __shfl_xor_sync(0xffffffffu, next, 1));
            next = fmaxf(next, __shfl_xor_sync(0xffffffffu, next, 2));
            const float previous = maximum[r];
            const float rescale = isfinite(previous) ? expf(previous - next) : 0.0f;
            float sum = 0.0f;
#pragma unroll
            for (unsigned int j = 0; j < 8; j++) {
                const unsigned int t = part * 8u + j;
                const float p = isfinite(scores[r * 32 + t]) ? expf(scores[r * 32 + t] - next) : 0.0f;
                probabilities[r * 32 + t] = __float2half_rn(p);
                sum += p;
            }
            sum += __shfl_xor_sync(0xffffffffu, sum, 1);
            sum += __shfl_xor_sync(0xffffffffu, sum, 2);
            __syncwarp();
            if (part == 0u) { maximum[r] = next; denominator[r] = denominator[r] * rescale + sum; correction[r] = rescale; }
        } else if (threadIdx.x < 32) {
            const unsigned int r = threadIdx.x;
            float next = maximum[r];
            for (unsigned int t = 0; t < 32; t++) {
                const bool visible = first + r < rows && base + t < last && base + t <= start + first + r;
                const float score = visible ? scores[r * 32 + t] * 0.0625f : -__int_as_float(0x7f800000);
                scores[r * 32 + t] = score;
                next = fmaxf(next, score);
            }
            const float rescale = isfinite(maximum[r]) ? expf(maximum[r] - next) : 0.0f;
            float sum = denominator[r] * rescale;
            for (unsigned int t = 0; t < 32; t++) {
                const float p = isfinite(scores[r * 32 + t]) ? expf(scores[r * 32 + t] - next) : 0.0f;
                probabilities[r * 32 + t] = __float2half_rn(p);
                sum += p;
            }
            maximum[r] = next; denominator[r] = sum; correction[r] = rescale;
        }
        __syncthreads();
        // Reuse the no-longer-live score tile as warp-private rescaling scratch.
        // Each warp owns eight column fragments of one 16-query row group.
#pragma unroll
        for (int f = 0; f < 8; f++) {
            float* tile = scores + warp * 256;
            wmma::store_matrix_sync(tile, out[f], 16, wmma::mem_row_major);
            __syncwarp();
            for (unsigned int i = lane; i < 256; i += 32)
                tile[i] *= correction[(warp / 2) * 16 + i / 16];
            __syncwarp();
            wmma::load_matrix_sync(out[f], tile, 16, wmma::mem_row_major);
            __syncwarp();
        }
        nvfp4kv::stage_tokens32(kv, valuePages, base, last, kh, keyHeads);
        __syncthreads();
#pragma unroll
        for (int f = 0; f < 8; f++) {
            for (int t = 0; t < 32; t += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b;
                wmma::load_matrix_sync(a, probabilities + (warp / 2) * 16 * 32 + t, 32);
                wmma::load_matrix_sync(b, kv + t * 256 + ((warp % 2) * 8 + f) * 16, 256);
                wmma::mma_sync(out[f], a, b, out[f]);
            }
        }
        __syncthreads();
    }
    // K/V staging becomes one 16-query FP32 output tile at final writeback.
    // Only this final rotation is serialized between the two row groups.
    float* accum = reinterpret_cast<float*>(kv);
    for (unsigned int group = 0; group < 2; group++) {
        if (warp / 2 == group) {
#pragma unroll
            for (int f = 0; f < 8; f++)
                wmma::store_matrix_sync(accum + ((warp % 2) * 8 + f) * 16, out[f], 256, wmma::mem_row_major);
        }
        __syncthreads();
        for (unsigned int r = warp; r < 16; r += 4) {
            const unsigned int row = first + group * 16 + r;
            if (row >= rows) continue;
            float values[8];
#pragma unroll
            for (int d = 0; d < 8; d++) values[d] = accum[r * 256 + lane + d * 32] / denominator[group * 16 + r];
            nvfp4kv::hadamard256(values, lane);
#pragma unroll
            for (int d = 0; d < 8; d++) {
                const unsigned int col = head * 256 + lane + d * 32;
                const float gate = __bfloat162float(gateValue[(unsigned long long)row * width + col]);
                output[(unsigned long long)row * queryHeads * 256 + col]
                        = __float2bfloat16_rn(values[d] / (1.0f + expf(-gate)));
            }
        }
        __syncthreads();
    }
}

extern "C" __global__ __launch_bounds__(128) void euhedral_attention_prefill32_nvfp4(
        const __nv_bfloat16* queryKey, const __nv_bfloat16* gateValue,
        const unsigned char* const* keyPages, const unsigned char* const* valuePages,
        __nv_bfloat16* output, unsigned int rows, unsigned int queryHeads, unsigned int keyHeads,
        unsigned int headDim, unsigned int cacheLength, unsigned long long start) {
    attention_prefill32_nvfp4<false>(queryKey, gateValue, keyPages, valuePages, output, rows, queryHeads,
            keyHeads, cacheLength, start);
}
extern "C" __global__ __launch_bounds__(128) void euhedral_attention_prefill32_nvfp4_exact(
        const __nv_bfloat16* queryKey, const __nv_bfloat16* gateValue,
        const unsigned char* const* keyPages, const unsigned char* const* valuePages,
        __nv_bfloat16* output, unsigned int rows, unsigned int queryHeads, unsigned int keyHeads,
        unsigned int headDim, unsigned int cacheLength, unsigned long long start) {
    attention_prefill32_nvfp4<true>(queryKey, gateValue, keyPages, valuePages, output, rows, queryHeads,
            keyHeads, cacheLength, start);
}
