#pragma once
#include "nvfp4_kv.cuh"
#include <mma.h>

// Up to 16 grouped query heads x 32 keys per split; K and V share the same staging storage. Accumulation
// and online softmax are FP32; rotated Q and probabilities are FP16 MMA operands.
// KV expansion into FP16 is exact. No quadratic score buffer is materialized.
extern "C" __global__ __launch_bounds__(128) void euhedral_attention_decode_tc_nvfp4(
        const __nv_bfloat16* queryKey, const __nv_bfloat16* gateValue,
        const unsigned char* const* keyPages, const unsigned char* const* valuePages,
        __nv_bfloat16* output, unsigned int rows, unsigned int queryHeads, unsigned int keyHeads,
        unsigned int headDim, unsigned int cacheLength, unsigned long long start,
        float* partial, unsigned int splits) {
    using namespace nvcuda;
    const unsigned int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const unsigned int kh = blockIdx.x / splits, split = blockIdx.x % splits;
    const unsigned int ratio = queryHeads / keyHeads;
    const unsigned int span = (cacheLength + splits - 1) / splits;
    const unsigned int begin = split * span, last = min(cacheLength, begin + span);
    __shared__ __align__(32) __half q[16 * 256];
    __shared__ __align__(32) __half kv[32 * 256];
    __shared__ __align__(32) float scores[16 * 32];
    __shared__ __align__(32) __half probabilities[16 * 32];
    __shared__ __align__(32) float accum[16 * 256];
    __shared__ float maximum[16], denominator[16], correction[16];
    for (unsigned int r = warp; r < 16; r += 4) {
        float values[8];
#pragma unroll
        for (int d = 0; d < 8; d++) values[d] = r < ratio
                ? __bfloat162float(queryKey[(kh * ratio + r) * 256 + lane + d * 32]) : 0;
        nvfp4kv::hadamard256(values, lane);
#pragma unroll
        for (int d = 0; d < 8; d++) q[r * 256 + lane + d * 32] = __float2half_rn(values[d]);
        if (lane == 0) { maximum[r] = -__int_as_float(0x7f800000); denominator[r] = 0; }
    }
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> out[4];
#pragma unroll
    for (int f = 0; f < 4; f++) wmma::fill_fragment(out[f], 0.0f);
    __syncthreads();
    for (unsigned int base = begin; base < last; base += 32) {
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
                const bool visible = r < ratio && base + t < last;
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
    for (unsigned int r = warp; r < ratio; r += 4) {
        float* destination = partial + ((unsigned long long)(kh * ratio + r) * splits + split) * 258;
#pragma unroll
        for (int d = 0; d < 8; d++) destination[lane + d * 32] = accum[r * 256 + lane + d * 32];
        if (lane == 0) { destination[256] = maximum[r]; destination[257] = denominator[r]; }
    }
}
