#pragma once
#include "nvfp4_kv.cuh"
#include <mma.h>

// 32 query rows x 32 keys; K and V share the same staging storage. Accumulation
// and online softmax are FP32; rotated Q and probabilities are FP16 MMA operands.
// KV expansion into FP16 is exact. No quadratic score buffer is materialized.
extern "C" __global__ __launch_bounds__(128) void euhedral_attention_prefill32_nvfp4(
        const __nv_bfloat16* queryKey, const __nv_bfloat16* gateValue,
        const unsigned char* const* keyPages, const unsigned char* const* valuePages,
        __nv_bfloat16* output, unsigned int rows, unsigned int queryHeads, unsigned int keyHeads,
        unsigned int headDim, unsigned int cacheLength, unsigned long long start) {
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
        for (unsigned int i = threadIdx.x; i < 32 * 256; i += 128) {
            const unsigned int token = base + i / 256;
            kv[i] = __float2half_rn(token < last
                    ? nvfp4kv::element(nvfp4kv::cache_row(keyPages, token, kh, keyHeads), i % 256) : 0);
        }
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
        if (threadIdx.x < 32) {
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
        for (unsigned int i = threadIdx.x; i < 32 * 256; i += 128) {
            const unsigned int token = base + i / 256;
            kv[i] = __float2half_rn(token < last
                    ? nvfp4kv::element(nvfp4kv::cache_row(valuePages, token, kh, keyHeads), i % 256) : 0);
        }
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
