// Reference Q4/Q5 kernels: the pre-restructuring q45_linear_bf16.cu schedule
// (per-group packed loads, one row per decode CTA, 32 x 32 WMMA prefill) with
// renamed entry points. Tests require the restructured kernels to match it
// bitwise.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>

namespace q45_reference {
static __device__ __forceinline__ unsigned long long align256(unsigned long long n) {
    return (n + 255ull) & ~255ull;
}

template<int BITS>
static __device__ __forceinline__ unsigned int pair_codes(
        const unsigned char* weights, unsigned long long group, unsigned long long high_offset,
        unsigned int lane) {
    unsigned int pair = weights[group * 32ull + lane];
    if (BITS == 5) {
        unsigned int high = lane < 8 ? weights[high_offset + group * 8ull + lane] : 0;
        high = __shfl_sync(0xffffffffu, high, lane / 4);
        pair |= ((high >> ((lane % 4) * 2)) & 3u) << 8;
    }
    return pair;
}

template<int BITS>
static __device__ __forceinline__ int unpack_code(unsigned int pair, unsigned int index) {
    unsigned int value = (pair >> (index * 4)) & 15u;
    if (BITS == 5) value |= ((pair >> (8 + index)) & 1u) << 4;
    return (int)value - ((value & (1u << (BITS - 1))) ? (1 << BITS) : 0);
}

template<int BITS>
static __device__ __forceinline__ void decode(
        const __nv_bfloat16* input, const unsigned char* weights, __nv_bfloat16* output,
        unsigned int rows, unsigned int in_features, unsigned int out_features) {
    const unsigned int lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    const unsigned int tiles = out_features / 8u + (out_features % 8u != 0u);
    const unsigned int row = blockIdx.x / tiles;
    const unsigned int column = (blockIdx.x % tiles) * 8u + warp * 2u;
    if (row >= rows) return;
    const unsigned int groups = in_features / 64u;
    const unsigned long long code_bytes = (unsigned long long)out_features * groups * 32ull;
    const unsigned long long high_offset = align256(code_bytes);
    const unsigned long long scale_offset = high_offset + (BITS == 5 ? align256((unsigned long long)out_features * groups * 8ull) : 0);
    const __half* scales = (const __half*)(weights + scale_offset);
    float sums[2][4] = {};
    for (unsigned int base = 0; base < in_features; base += 128) {
        float x[4];
        #pragma unroll
        for (int p = 0; p < 4; p++)
            x[p] = __bfloat162float(input[(unsigned long long)row * in_features + base + lane + p * 32]);
        #pragma unroll
        for (int n = 0; n < 2; n++) {
            if (column + n >= out_features) continue;
            #pragma unroll
            for (int half = 0; half < 2; half++) {
                unsigned long long g = (unsigned long long)(column + n) * groups + base / 64u + half;
                unsigned int pair = pair_codes<BITS>(weights, g, high_offset, lane);
                float scale = __half2float(__shfl_sync(0xffffffffu, lane == 0 ? scales[g] : __float2half(0.0f), 0));
                #pragma unroll
                for (int p = 0; p < 2; p++) {
                    unsigned int packed = __shfl_sync(0xffffffffu, pair, lane / 2 + p * 16);
                    int code = unpack_code<BITS>(packed, lane & 1);
                    sums[n][half * 2 + p] += x[half * 2 + p] * (float)code * scale;
                }
            }
        }
    }
    #pragma unroll
    for (int n = 0; n < 2; n++) {
        float sum = (sums[n][0] + sums[n][2]) + (sums[n][1] + sums[n][3]);
        #pragma unroll
        for (int distance = 16; distance; distance >>= 1)
            sum += __shfl_down_sync(0xffffffffu, sum, distance);
        if (lane == 0 && column + n < out_features)
            output[(unsigned long long)row * out_features + column + n] = __float2bfloat16_rn(sum);
    }
}

template<int BITS>
static __device__ __forceinline__ void prefill(
        const __nv_bfloat16* input, const unsigned char* weights, __nv_bfloat16* output,
        unsigned int rows, unsigned int in_features, unsigned int out_features) {
    using namespace nvcuda;
    __shared__ __align__(32) __nv_bfloat16 a[32 * 64];
    __shared__ __align__(32) __nv_bfloat16 b_hi[32 * 64];
    __shared__ __align__(32) __nv_bfloat16 b_lo[32 * 64];
    __shared__ __align__(32) float result[32 * 32];
    unsigned int lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    unsigned int output_tiles = out_features / 32u + (out_features % 32u != 0u);
    unsigned int row_start = (blockIdx.x / output_tiles) * 32u, out_start = (blockIdx.x % output_tiles) * 32u;
    unsigned int groups = in_features / 64u;
    unsigned long long high_offset = align256((unsigned long long)out_features * groups * 32ull);
    unsigned long long scale_offset = high_offset + (BITS == 5 ? align256((unsigned long long)out_features * groups * 8ull) : 0);
    const __half* scales = (const __half*)(weights + scale_offset);
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);
    for (unsigned int base = 0; base < in_features; base += 64) {
        for (unsigned int i = threadIdx.x; i < 32 * 64; i += 128) {
            unsigned int r = row_start + i / 64, k = base + i % 64;
            a[i] = r < rows ? input[(unsigned long long)r * in_features + k] : __float2bfloat16(0.0f);
        }
        for (unsigned int col = warp; col < 32; col += 4) {
            unsigned int pairs = 0;
            float scale = 0.0f;
            if (out_start + col < out_features) {
                unsigned long long g = (unsigned long long)(out_start + col) * groups + base / 64u;
                pairs = pair_codes<BITS>(weights, g, high_offset, lane);
                scale = __half2float(__shfl_sync(0xffffffffu, lane == 0 ? scales[g] : __float2half(0.0f), 0));
            }
            #pragma unroll
            for (int p = 0; p < 2; p++) {
                int code = unpack_code<BITS>(pairs, p);
                float weight = (float)code * scale;
                unsigned int i = col * 64 + lane * 2 + p;
                b_hi[i] = __float2bfloat16(weight);
                b_lo[i] = __float2bfloat16(weight - __bfloat162float(b_hi[i]));
            }
        }
        __syncthreads();
        #pragma unroll
        for (unsigned int k = 0; k < 64; k += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> bf;
            wmma::load_matrix_sync(af, a + (warp / 2) * 16 * 64 + k, 64);
            wmma::load_matrix_sync(bf, b_hi + (warp % 2) * 16 * 64 + k, 64);
            wmma::mma_sync(acc, af, bf, acc);
            wmma::load_matrix_sync(bf, b_lo + (warp % 2) * 16 * 64 + k, 64);
            wmma::mma_sync(acc, af, bf, acc);
        }
        __syncthreads();
    }
    wmma::store_matrix_sync(result + (warp / 2) * 16 * 32 + (warp % 2) * 16, acc, 32, wmma::mem_row_major);
    __syncthreads();
    for (unsigned int i = threadIdx.x; i < 32 * 32; i += 128) {
        unsigned int r = row_start + i / 32, n = out_start + i % 32;
        if (r < rows && n < out_features) output[(unsigned long long)r * out_features + n] = __float2bfloat16_rn(result[i]);
    }
}
}  // namespace q45_reference

#define Q45_REFERENCE_KERNELS(B) \
extern "C" __global__ __launch_bounds__(128) void reference_q##B##_decode( \
        const __nv_bfloat16* input, const unsigned char* weights, __nv_bfloat16* output, \
        unsigned int rows, unsigned int in_features, unsigned int out_features) { \
    q45_reference::decode<B>(input, weights, output, rows, in_features, out_features); \
} \
extern "C" __global__ __launch_bounds__(128) void reference_q##B##_prefill( \
        const __nv_bfloat16* input, const unsigned char* weights, __nv_bfloat16* output, \
        unsigned int rows, unsigned int in_features, unsigned int out_features) { \
    q45_reference::prefill<B>(input, weights, output, rows, in_features, out_features); \
}
Q45_REFERENCE_KERNELS(4)
Q45_REFERENCE_KERNELS(5)
#undef Q45_REFERENCE_KERNELS
