#pragma once
#include "nvfp4_kv.cuh"
#include <mma.h>
#include "pdl.cuh"

namespace nvfp4kv {

__device__ __forceinline__ const unsigned char* cache_row(
        const unsigned char* const* pages, unsigned int token, unsigned int head, unsigned int heads) {
    return pages[token / 256] + ((token % 256) * heads + head) * 144;
}

__device__ __forceinline__ float element(const unsigned char* row, unsigned int d) {
    const unsigned int code = (row[d / 2] >> (4 * (d & 1))) & 15;
    return e2m1_decode(code) * e4m3_decode(row[128 + d / 16]);
}

// All lanes receive the sum. No CTA barrier in the per-key decode loop.
__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int mask = 16; mask; mask >>= 1) x += __shfl_xor_sync(0xffffffffu, x, mask);
    return x;
}

} // namespace nvfp4kv

// Differential-test and benchmark control; production dispatch uses the 32-row tile.
// 16 query rows x 32 keys; K and V share the same staging storage. Accumulation
// and online softmax are FP32; rotated Q and probabilities are FP16 MMA operands.
// KV expansion into FP16 is exact. No quadratic score buffer is materialized.
extern "C" __global__ __launch_bounds__(128) void euhedral_attention_prefill_nvfp4(
        const __nv_bfloat16* queryKey, const __nv_bfloat16* gateValue,
        const unsigned char* const* keyPages, const unsigned char* const* valuePages,
        __nv_bfloat16* output, unsigned int rows, unsigned int queryHeads, unsigned int keyHeads,
        unsigned int headDim, unsigned int cacheLength, unsigned long long start) {
    using namespace nvcuda;
    const unsigned int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const unsigned int head = blockIdx.x % queryHeads, first = (blockIdx.x / queryHeads) * 16;
    const unsigned int kh = head / (queryHeads / keyHeads), width = (queryHeads + keyHeads) * 256;
    __shared__ __align__(32) __half q[16 * 256];
    __shared__ __align__(32) __half kv[32 * 256];
    __shared__ __align__(32) float scores[16 * 32];
    __shared__ __align__(32) __half probabilities[16 * 32];
    __shared__ __align__(32) float accum[16 * 256];
    __shared__ float maximum[16], denominator[16], correction[16];
    for (unsigned int r = warp; r < 16; r += 4) {
        float values[8];
#pragma unroll
        for (int d = 0; d < 8; d++) values[d] = first + r < rows
                ? __bfloat162float(queryKey[(unsigned long long)(first + r) * width + head * 256 + lane + d * 32]) : 0;
        nvfp4kv::hadamard256(values, lane);
#pragma unroll
        for (int d = 0; d < 8; d++) q[r * 256 + lane + d * 32] = __float2half_rn(values[d]);
        if (lane == 0) { maximum[r] = -__int_as_float(0x7f800000); denominator[r] = 0; }
    }
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> out[4];
#pragma unroll
    for (int f = 0; f < 4; f++) wmma::fill_fragment(out[f], 0.0f);
    __syncthreads();
    const unsigned int last = min(cacheLength, (unsigned int)(start + min(first + 16, rows)));
    for (unsigned int base = 0; base < last; base += 32) {
        for (unsigned int i = threadIdx.x; i < 32 * 256; i += 128) {
            const unsigned int token = base + i / 256;
            kv[i] = __float2half_rn(token < last
                    ? nvfp4kv::element(nvfp4kv::cache_row(keyPages, token, kh, keyHeads), i % 256) : 0);
        }
        __syncthreads();
        if (warp < 2) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> score;
            wmma::fill_fragment(score, 0.0f);
            for (int d = 0; d < 256; d += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
                wmma::load_matrix_sync(a, q + d, 256);
                wmma::load_matrix_sync(b, kv + warp * 16 * 256 + d, 256);
                wmma::mma_sync(score, a, b, score);
            }
            wmma::store_matrix_sync(scores + warp * 16, score, 32, wmma::mem_row_major);
        }
        __syncthreads();
        if (threadIdx.x < 16) {
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
        // Give each warp four 16-column output fragments. The shared round trip
        // applies the per-query rescaling without depending on opaque WMMA lanes.
#pragma unroll
        for (int f = 0; f < 4; f++)
            wmma::store_matrix_sync(accum + (warp * 4 + f) * 16, out[f], 256, wmma::mem_row_major);
        __syncthreads();
        for (unsigned int i = threadIdx.x; i < 16 * 256; i += 128) accum[i] *= correction[i / 256];
        for (unsigned int i = threadIdx.x; i < 32 * 256; i += 128) {
            const unsigned int token = base + i / 256;
            kv[i] = __float2half_rn(token < last
                    ? nvfp4kv::element(nvfp4kv::cache_row(valuePages, token, kh, keyHeads), i % 256) : 0);
        }
        __syncthreads();
#pragma unroll
        for (int f = 0; f < 4; f++) {
            wmma::load_matrix_sync(out[f], accum + (warp * 4 + f) * 16, 256, wmma::mem_row_major);
            for (int t = 0; t < 32; t += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b;
                wmma::load_matrix_sync(a, probabilities + t, 32);
                wmma::load_matrix_sync(b, kv + t * 256 + (warp * 4 + f) * 16, 256);
                wmma::mma_sync(out[f], a, b, out[f]);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int f = 0; f < 4; f++)
        wmma::store_matrix_sync(accum + (warp * 4 + f) * 16, out[f], 256, wmma::mem_row_major);
    __syncthreads();
    for (unsigned int r = warp; r < 16; r += 4) {
        if (first + r >= rows) continue;
        float values[8];
#pragma unroll
        for (int d = 0; d < 8; d++) values[d] = accum[r * 256 + lane + d * 32] / denominator[r];
        nvfp4kv::hadamard256(values, lane);
#pragma unroll
        for (int d = 0; d < 8; d++) {
            const unsigned int col = head * 256 + lane + d * 32;
            const float gate = __bfloat162float(gateValue[(unsigned long long)(first + r) * width + col]);
            output[(unsigned long long)(first + r) * queryHeads * 256 + col]
                    = __float2bfloat16_rn(values[d] / (1.0f + expf(-gate)));
        }
    }
}

// Split the context across CTAs; each warp scans a disjoint subsequence of the
// split using warp shuffles only. Shared memory is used once to combine warps.
extern "C" __global__ __launch_bounds__(128) void euhedral_attention_decode_nvfp4(
        const __nv_bfloat16* queryKey, const __nv_bfloat16* gateValue,
        const unsigned char* const* keyPages, const unsigned char* const* valuePages,
        __nv_bfloat16* output, unsigned int rows, unsigned int queryHeads, unsigned int keyHeads,
        unsigned int headDim, unsigned int cacheLength, unsigned long long start,
        float* partial, unsigned int splits) {
    euhedral_pdl_begin();
    const unsigned int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const unsigned int head = blockIdx.x / splits, split = blockIdx.x % splits;
    const unsigned int kh = head / (queryHeads / keyHeads);
    const unsigned int span = (cacheLength + splits - 1) / splits;
    const unsigned int begin = split * span, end = min(cacheLength, begin + span);
    float q[8], acc[8] = {};
#pragma unroll
    for (int d = 0; d < 8; d++) q[d] = __bfloat162float(queryKey[head * 256 + lane + d * 32]);
    nvfp4kv::hadamard256(q, lane);
    float maximum = -__int_as_float(0x7f800000), denominator = 0;
    for (unsigned int token = begin + warp; token < end; token += 4) {
        const unsigned char* key = nvfp4kv::cache_row(keyPages, token, kh, keyHeads);
        const unsigned char* value = nvfp4kv::cache_row(valuePages, token, kh, keyHeads);
        float dot = 0;
#pragma unroll
        for (int d = 0; d < 8; d++) dot += q[d] * nvfp4kv::element(key, lane + d * 32);
        const float score = nvfp4kv::warp_sum(dot) * 0.0625f;
        const float next = fmaxf(maximum, score);
        const float a = expf(maximum - next), b = expf(score - next);
        denominator = denominator * a + b;
#pragma unroll
        for (int d = 0; d < 8; d++) acc[d] = acc[d] * a + nvfp4kv::element(value, lane + d * 32) * b;
        maximum = next;
    }
    __shared__ float staging[4][258];
#pragma unroll
    for (int d = 0; d < 8; d++) staging[warp][lane + d * 32] = acc[d];
    if (lane == 0) { staging[warp][256] = maximum; staging[warp][257] = denominator; }
    __syncthreads();
    if (warp == 0) {
        float maxAll = -__int_as_float(0x7f800000);
        for (int w = 0; w < 4; w++) maxAll = fmaxf(maxAll, staging[w][256]);
        float sum = 0, combined[8] = {};
        for (int w = 0; w < 4; w++) {
            const float scale = staging[w][257] > 0 ? expf(staging[w][256] - maxAll) : 0;
            sum += staging[w][257] * scale;
#pragma unroll
            for (int d = 0; d < 8; d++) combined[d] += staging[w][lane + d * 32] * scale;
        }
        float* destination = partial + (unsigned long long)blockIdx.x * 258;
#pragma unroll
        for (int d = 0; d < 8; d++) destination[lane + d * 32] = combined[d];
        if (lane == 0) { destination[256] = maxAll; destination[257] = sum; }
    }
}

extern "C" __global__ __launch_bounds__(128) void euhedral_attention_merge_nvfp4(
        const __nv_bfloat16* gateValue, __nv_bfloat16* output, const float* partial,
        unsigned int queryHeads, unsigned int keyHeads, unsigned int splits) {
    euhedral_pdl_begin();
    if (threadIdx.x >= 32) return;
    const unsigned int lane = threadIdx.x, head = blockIdx.x;
    const float* source = partial + (unsigned long long)head * splits * 258;
    float maximum = -__int_as_float(0x7f800000);
    for (unsigned int s = 0; s < splits; s++) maximum = fmaxf(maximum, source[s * 258 + 256]);
    float sum = 0, values[8] = {};
    for (unsigned int s = 0; s < splits; s++) {
        const float scale = source[s * 258 + 257] > 0 ? expf(source[s * 258 + 256] - maximum) : 0;
        sum += source[s * 258 + 257] * scale;
#pragma unroll
        for (int d = 0; d < 8; d++) values[d] += source[s * 258 + lane + d * 32] * scale;
    }
#pragma unroll
    for (int d = 0; d < 8; d++) values[d] /= sum;
    nvfp4kv::hadamard256(values, lane);
#pragma unroll
    for (int d = 0; d < 8; d++) {
        const unsigned int col = head * 256 + lane + d * 32;
        const float gate = __bfloat162float(gateValue[col]);
        output[col] = __float2bfloat16_rn(values[d] / (1.0f + expf(-gate)));
    }
}
