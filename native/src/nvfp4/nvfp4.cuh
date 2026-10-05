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
// Decode: 1 to 8 BF16 token rows on BF16 tensor cores (mma.m16n8k16, FP32 accumulation). A weight
// (e2m1 code times its E4M3 block scale) has at most 6 significant bits, so it is exact in BF16 and
// each product with a BF16 activation is exact in FP32; the tensor core sums 16 of them per step. Weight rows are the MMA's 16 M
// rows and token rows its 8 N columns, so a token row's result never depends on the other tokens:
// every row of an M-row call is bit for bit the one-row call on it, which speculative verification
// relies on. A CTA owns 16 * R output rows; its W warps split K into 128-value chunks (chunk c on
// warp c mod W) and are summed in warp order. Lane (g, q) loads 16 code bytes of rows g and g + 8
// and 32 activations of token g at K offset 32 q of a chunk; a 256-entry shared table turns a code
// byte into two BF16 values. The tile (R, W) is a function of the shape alone (host dispatch), so
// one-row and multi-row calls on a tensor always agree. Requirements (checked by host dispatch):
// in_features a multiple of 128, out_features a multiple of 16 * R, 16-byte aligned input and
// weights, blockDim.x = 32 * W.
template <int M, bool kSd4, int R, int W>
static __device__ __forceinline__ void decode_rows(
        const unsigned short* input, const unsigned char* weights, unsigned short* output,
        unsigned int in_features, unsigned int out_features) {
    __shared__ unsigned int pair_table[256];   // byte -> two E2M1 values as BF16x2 (low nibble in the low half)
    __shared__ float scale_table[16];
    __shared__ float partial[W][16 * R][8];
    for (unsigned int i = threadIdx.x; i < 256u; i += blockDim.x) {
        const __nv_bfloat162 v = __floats2bfloat162_rn(e2m1_value(i & 15u), e2m1_value(i >> 4));
        pair_table[i] = *reinterpret_cast<const unsigned int*>(&v);
    }
    if (kSd4 && threadIdx.x < 16u)
        scale_table[threadIdx.x] = e4m3_to_float(LayoutT<true>(weights, in_features, out_features).table.lookup(threadIdx.x));
    __syncthreads();
    const unsigned int lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    const unsigned int g = lane >> 2, q = lane & 3u;
    const unsigned int row0 = blockIdx.x * 16u * R;
    const LayoutT<kSd4> w(weights, in_features, out_features);
    const unsigned int chunks = in_features / 128u;
    float acc[R][4];
    #pragma unroll
    for (int r = 0; r < R; r++) acc[r][0] = acc[r][1] = acc[r][2] = acc[r][3] = 0.0f;
    euhedral_pdl_begin();
    const unsigned short* x = input + (unsigned long long)g * in_features;
    #pragma unroll 2
    for (unsigned int c = warp; c < chunks; c += (unsigned int)W) {
        const unsigned int k0 = c * 128u + 32u * q;
        uint4 xa[4];
        #pragma unroll
        for (int i = 0; i < 4; i++)
            xa[i] = (int)g < M ? reinterpret_cast<const uint4*>(x + k0)[i] : make_uint4(0, 0, 0, 0);
        const unsigned int xs[16] = {xa[0].x, xa[0].y, xa[0].z, xa[0].w, xa[1].x, xa[1].y, xa[1].z, xa[1].w,
                                     xa[2].x, xa[2].y, xa[2].z, xa[2].w, xa[3].x, xa[3].y, xa[3].z, xa[3].w};
        #pragma unroll
        for (int r = 0; r < R; r++) {
            const unsigned int rg = row0 + 16u * r + g, rh = rg + 8u;
            const uint4 cg = *reinterpret_cast<const uint4*>(w.codes + (unsigned long long)rg * w.row_bytes + k0 / 2u);
            const uint4 ch = *reinterpret_cast<const uint4*>(w.codes + (unsigned long long)rh * w.row_bytes + k0 / 2u);
            const unsigned int sg = w.scale_bits(rg, k0 / 32u), sh = w.scale_bits(rh, k0 / 32u);
            const unsigned int wg[4] = {cg.x, cg.y, cg.z, cg.w}, wh[4] = {ch.x, ch.y, ch.z, ch.w};
            #pragma unroll
            for (int half = 0; half < 2; half++) {
                const float fg = scale<kSd4>(scale_table, sg, half), fh = scale<kSd4>(scale_table, sh, half);
                const __nv_bfloat162 bg = __floats2bfloat162_rn(fg, fg), bh = __floats2bfloat162_rn(fh, fh);
                #pragma unroll
                for (int s = half * 4; s < half * 4 + 4; s++) {
                    const unsigned int word_g = wg[s >> 1], word_h = wh[s >> 1];
                    const unsigned int shift = (s & 1) * 16u;
                    unsigned int pg0 = pair_table[(word_g >> shift) & 0xffu], pg1 = pair_table[(word_g >> (shift + 8u)) & 0xffu];
                    unsigned int ph0 = pair_table[(word_h >> shift) & 0xffu], ph1 = pair_table[(word_h >> (shift + 8u)) & 0xffu];
                    __nv_bfloat162 vg0 = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&pg0), bg);
                    __nv_bfloat162 vg1 = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&pg1), bg);
                    __nv_bfloat162 vh0 = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&ph0), bh);
                    __nv_bfloat162 vh1 = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&ph1), bh);
                    const unsigned int a0 = *reinterpret_cast<unsigned int*>(&vg0), a1 = *reinterpret_cast<unsigned int*>(&vh0);
                    const unsigned int a2 = *reinterpret_cast<unsigned int*>(&vg1), a3 = *reinterpret_cast<unsigned int*>(&vh1);
                    asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                            "{%0,%1,%2,%3};\n"
                            : "+f"(acc[r][0]), "+f"(acc[r][1]), "+f"(acc[r][2]), "+f"(acc[r][3])
                            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(xs[2 * s]), "r"(xs[2 * s + 1]));
                }
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < R; r++) {
        partial[warp][16 * r + g][2 * q] = acc[r][0];
        partial[warp][16 * r + g][2 * q + 1] = acc[r][1];
        partial[warp][16 * r + g + 8][2 * q] = acc[r][2];
        partial[warp][16 * r + g + 8][2 * q + 1] = acc[r][3];
    }
    __syncthreads();
    for (unsigned int index = threadIdx.x; index < 16u * R * (unsigned int)M; index += blockDim.x) {
        const unsigned int r = index % (16u * R), t = index / (16u * R);
        float sum = partial[0][r][t];
        #pragma unroll
        for (int v = 1; v < W; v++) sum += partial[v][r][t];
        q3::write_bf16(output + (unsigned long long)t * out_features, 0, row0 + r, out_features, sum * w.global);
    }
}
}  // namespace nvfp4
