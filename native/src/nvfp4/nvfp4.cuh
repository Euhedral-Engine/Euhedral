#pragma once
#include "q3/numeric.cuh"
#include "common/pdl.cuh"
#include "nvfp4/scale_table.cuh"

// NVFP4 weights (WeightFormat.NVFP4, row-split-k128-v1; Nvfp4Layout.java,
// docs/NVFP4_NATIVE.md). Rows of K values, K padded to
// 128: E2M1 codes two per byte with the even K in the low nibble, then a 256-aligned plane of one
// E4M3 scale per 16 values, then a 256-aligned FP32 global scale. A weight is
// e2m1(code) * e4m3(scale) * global.
//
// SD4 (row-split-k128-sd4-v1, docs/NVFP4_COMPRESSED.md) stores each block scale
// as a 4-bit index into the tensor's table of 16 E4M3 codes: the scale plane holds K/32 bytes per row,
// two indices per byte with the even block in the low nibble, and the table (16 bytes) precedes the
// global scale at the 256-aligned end of the plane. Kernels look the codes up and then run exactly
// as on the plain layout, so an SD4 tensor computes what its expansion to plain NVFP4 computes.
namespace nvfp4 {
static constexpr unsigned int kBlock = 16u;

static __host__ __device__ __forceinline__ unsigned long long align256(unsigned long long v) { return (v + 255ull) & ~255ull; }

template <bool kSd4>
struct LayoutT {
    const unsigned char* codes;
    const unsigned char* scales;
    float global;
    unsigned int row_bytes, row_scales;
    ScaleTable table;
    __device__ __forceinline__ LayoutT(const unsigned char* w, unsigned int in_features, unsigned int rows) {
        const unsigned long long k = (in_features + 127u) / 128u * 128u;
        row_bytes = (unsigned int)(k / 2u);
        row_scales = (unsigned int)(k / (kSd4 ? 2u * kBlock : kBlock));
        const unsigned long long scale_offset = align256((unsigned long long)rows * row_bytes);
        const unsigned long long global_offset = align256(scale_offset + (unsigned long long)rows * row_scales);
        codes = w;
        scales = w + scale_offset;
        if (kSd4) {
            table.codes = *reinterpret_cast<const uint4*>(w + global_offset);
            global = *reinterpret_cast<const float*>(w + global_offset + 16u);
        } else {
            global = *reinterpret_cast<const float*>(w + global_offset);
        }
    }
    // Scale-plane bits of blocks 2 pair and 2 pair + 1 of `row`, the first in the low bits: two E4M3
    // codes, or (SD4) two table indices.
    __device__ __forceinline__ unsigned int scale_bits(unsigned long long row, unsigned int pair) const {
        if (kSd4) return scales[row * row_scales + pair];
        return reinterpret_cast<const unsigned short*>(scales + row * row_scales)[pair];
    }
    // E4M3 codes of blocks 2 pair and 2 pair + 1 of `row`, the first in the low byte.
    __device__ __forceinline__ unsigned int scale_pair(unsigned long long row, unsigned int pair) const {
        if (kSd4) return table.lookup(scale_bits(row, pair)) & 0xffffu;
        return scale_bits(row, pair);
    }
};
using Layout = LayoutT<false>;

// E4M3 (no sign, never NaN in a valid tensor) is FP16 with the exponent bias of 7: shifting its bits
// into an FP16 and multiplying by 2^8 is exact, subnormals included.
static __device__ __forceinline__ float e4m3_to_float(unsigned int bits) {
    return __half2float(__ushort_as_half((unsigned short)((bits & 0x7fu) << 7))) * 256.0f;
}

// Block `half` (0 or 1) of a lane's scale bits (LayoutT::scale_bits) as a float: E4M3 decoded in
// registers, or (SD4) an index into the block's shared table of decoded scales.
template <bool kSd4>
static __device__ __forceinline__ float scale(const float* table, unsigned int bits, unsigned int half);

static __device__ __forceinline__ float e2m1_value(unsigned int code) {
    const unsigned int magnitude = code & 7u;
    const float value = magnitude < 4u ? 0.5f * (float)magnitude : magnitude == 4u ? 2.0f : magnitude == 5u ? 3.0f : magnitude == 6u ? 4.0f : 6.0f;
    return (code & 8u) ? -value : value;
}

template <bool kSd4>
static __device__ __forceinline__ float scale(const float* table, unsigned int bits, unsigned int half) {
    if (kSd4) return table[(bits >> (4u * half)) & 15u];
    return e4m3_to_float(half ? bits >> 8 : bits & 0xffu);
}

// ---------------------------------------------------------------------------------------------------
// Single-row decode: each lane owns 32 contiguous K values of a 1024-value slice (16 code bytes, one
// 16-byte load, and two block scales), each warp owns kRows rows and reuses its 32 activations for
// all of them. Code values come from a 16-entry shared table, one entry per bank. Each 16-value
// block forms an FP32 dot product scaled once by its block scale; the global scale is applied last.
// Requirements (checked by host dispatch): one row, in_features a multiple of 1024, out_features a
// multiple of 4 * kRows, 16-byte aligned input and weights.
static constexpr int kDecodeRows = 4;

template <bool kSd4 = false>
static __device__ __forceinline__ void decode(
        const unsigned short* input, const unsigned char* weights, unsigned short* output,
        unsigned int in_features, unsigned int out_features) {
    __shared__ float table[16];
    __shared__ float scale_table[16];  // SD4: the decoded scale table
    if (threadIdx.x < 16u) table[threadIdx.x] = e2m1_value(threadIdx.x);
    if (kSd4 && threadIdx.x < 16u)
        scale_table[threadIdx.x] = e4m3_to_float(LayoutT<true>(weights, in_features, out_features).table.lookup(threadIdx.x));
    __syncthreads();
    const unsigned int lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    const unsigned int first_row = (blockIdx.x * (blockDim.x >> 5) + warp) * kDecodeRows;
    const LayoutT<kSd4> w(weights, in_features, out_features);
    const unsigned int slices = in_features / 1024u;
    float sums[kDecodeRows] = {};
    euhedral_pdl_begin();
    for (unsigned int slice = 0; slice < slices; slice++) {
        uint4 codes[kDecodeRows];
        unsigned int scale_pair[kDecodeRows];
        #pragma unroll
        for (int r = 0; r < kDecodeRows; r++) {
            const unsigned long long row = first_row + r;
            codes[r] = reinterpret_cast<const uint4*>(w.codes + row * w.row_bytes)[slice * 32u + lane];
            scale_pair[r] = w.scale_bits(row, slice * 32u + lane);
        }
        float x[32];
        const uint4* activation = reinterpret_cast<const uint4*>(input + slice * 1024u + 32u * lane);
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            const uint4 v = activation[i];
            const unsigned int pairs[4] = {v.x, v.y, v.z, v.w};
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                x[i * 8 + j * 2] = __uint_as_float(pairs[j] << 16);
                x[i * 8 + j * 2 + 1] = __uint_as_float(pairs[j] & 0xffff0000u);
            }
        }
        #pragma unroll
        for (int r = 0; r < kDecodeRows; r++) {
            const unsigned int words[4] = {codes[r].x, codes[r].y, codes[r].z, codes[r].w};
            float block[2] = {0.0f, 0.0f};
            #pragma unroll
            for (int j = 0; j < 32; j++) {
                const float value = table[(words[j >> 3] >> ((j & 7) * 4)) & 15u];
                block[j >> 4] = fmaf(x[j], value, block[j >> 4]);
            }
            sums[r] = fmaf(block[0], scale<kSd4>(scale_table, scale_pair[r], 0), sums[r]);
            sums[r] = fmaf(block[1], scale<kSd4>(scale_table, scale_pair[r], 1), sums[r]);
        }
    }
    #pragma unroll
    for (int r = 0; r < kDecodeRows; r++) {
        #pragma unroll
        for (int distance = 16; distance; distance >>= 1) sums[r] += __shfl_xor_sync(0xffffffffu, sums[r], distance);
    }
    float mine = sums[0];
    #pragma unroll
    for (int r = 1; r < kDecodeRows; r++) mine = lane == (unsigned int)r ? sums[r] : mine;
    if (lane < (unsigned int)kDecodeRows) q3::write_bf16(output, 0, first_row + lane, out_features, mine * w.global);
}

// Multi-row decode for speculative verification: M activation rows against each weight row. Every
// token row repeats decode()'s exact FMA sequence (same blocks, same order, same warp reduction), so
// row t's output is bit for bit what decode() gives for that row alone; the weights stream once.
template <int M, bool kSd4 = false>
static __device__ __forceinline__ void decode_rows(
        const unsigned short* input, const unsigned char* weights, unsigned short* output,
        unsigned int in_features, unsigned int out_features) {
    __shared__ float table[16];
    __shared__ float scale_table[16];  // SD4: the decoded scale table
    if (threadIdx.x < 16u) table[threadIdx.x] = e2m1_value(threadIdx.x);
    if (kSd4 && threadIdx.x < 16u)
        scale_table[threadIdx.x] = e4m3_to_float(LayoutT<true>(weights, in_features, out_features).table.lookup(threadIdx.x));
    __syncthreads();
    const unsigned int lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    const unsigned int first_row = (blockIdx.x * (blockDim.x >> 5) + warp) * kDecodeRows;
    const LayoutT<kSd4> w(weights, in_features, out_features);
    const unsigned int slices = in_features / 1024u;
    float sums[M][kDecodeRows] = {};
    euhedral_pdl_begin();
    for (unsigned int slice = 0; slice < slices; slice++) {
        uint4 codes[kDecodeRows];
        unsigned int scale_pair[kDecodeRows];
        #pragma unroll
        for (int r = 0; r < kDecodeRows; r++) {
            const unsigned long long row = first_row + r;
            codes[r] = reinterpret_cast<const uint4*>(w.codes + row * w.row_bytes)[slice * 32u + lane];
            scale_pair[r] = w.scale_bits(row, slice * 32u + lane);
        }
        #pragma unroll
        for (int t = 0; t < M; t++) {
            float x[32];
            const uint4* activation = reinterpret_cast<const uint4*>(
                    input + (unsigned long long)t * in_features + slice * 1024u + 32u * lane);
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const uint4 v = activation[i];
                const unsigned int pairs[4] = {v.x, v.y, v.z, v.w};
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    x[i * 8 + j * 2] = __uint_as_float(pairs[j] << 16);
                    x[i * 8 + j * 2 + 1] = __uint_as_float(pairs[j] & 0xffff0000u);
                }
            }
            #pragma unroll
            for (int r = 0; r < kDecodeRows; r++) {
                const unsigned int words[4] = {codes[r].x, codes[r].y, codes[r].z, codes[r].w};
                float block[2] = {0.0f, 0.0f};
                #pragma unroll
                for (int j = 0; j < 32; j++) {
                    const float value = table[(words[j >> 3] >> ((j & 7) * 4)) & 15u];
                    block[j >> 4] = fmaf(x[j], value, block[j >> 4]);
                }
                sums[t][r] = fmaf(block[0], scale<kSd4>(scale_table, scale_pair[r], 0), sums[t][r]);
                sums[t][r] = fmaf(block[1], scale<kSd4>(scale_table, scale_pair[r], 1), sums[t][r]);
            }
        }
    }
    #pragma unroll
    for (int t = 0; t < M; t++) {
        #pragma unroll
        for (int r = 0; r < kDecodeRows; r++) {
            #pragma unroll
            for (int distance = 16; distance; distance >>= 1) sums[t][r] += __shfl_xor_sync(0xffffffffu, sums[t][r], distance);
        }
        float mine = sums[t][0];
        #pragma unroll
        for (int r = 1; r < kDecodeRows; r++) mine = lane == (unsigned int)r ? sums[t][r] : mine;
        if (lane < (unsigned int)kDecodeRows)
            q3::write_bf16(output + (unsigned long long)t * out_features, 0, first_row + lane, out_features, mine * w.global);
    }
}
}  // namespace nvfp4
