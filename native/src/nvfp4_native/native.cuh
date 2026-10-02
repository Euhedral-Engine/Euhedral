#pragma once
// Native Blackwell NVFP4 linear: both operands NVFP4, block-scaled FP4 tensor-core MMA.
//
// Compiled only for sm_120a/sm_120f (cuda_kernel_loader: arch-specific modules). The instruction is
//   mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3
// (SASS OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X). Contract, measured on an RTX 5070 Ti (docs/NVFP4_NATIVE.md):
// with g = lane / 4 and t = lane % 4,
//   A (16 x 64): a0 row g, K 8t..8t+7; a1 row g+8; a2, a3 the same rows at K + 32. Nibble j is K 8t+j.
//   B (64 x 8):  b0 column g, K 8t..8t+7; b1 at K + 32.
//   D (16 x 8):  d0, d1 row g, columns 2t, 2t+1; d2, d3 row g+8.
//   Scale A: row r < 8 from lane 4r + 2s, row r + 8 from lane 4r + 2s + 1 (selector s in 0..1).
//   Scale B: column n from lane 4n + s (selector s in 0..3). Byte b scales K 16b..16b+15.
// Scale bytes decode as E4M3 with the sign bit ignored, subnormals included.
//
// Operands are NVFP4 matrices with K padded to 128: rows of K/2 code bytes (even K in the low nibble)
// and rows of K/16 E4M3 scales, plus FP32 global scales (one per weight tensor; one per activation row).
// A weight tensor (Nvfp4Layout) is consumed in place; activations are quantized by quantize_rows.
// kSd4 weights (row-split-k128-sd4-v1, nvfp4/nvfp4.cuh) stage their 4-bit scale indices instead, and
// each B scale register is looked up in the tensor's table just before its MMA.
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include "common/pdl.cuh"
#include "nvfp4/scale_table.cuh"

namespace nvfp4n {

static __host__ __device__ __forceinline__ unsigned long long align256(unsigned long long v) { return (v + 255ull) & ~255ull; }

struct Matrix {
    const unsigned char* codes;
    const unsigned char* scales;
    unsigned int row_bytes, row_scales;
    nvfp4::ScaleTable table;  // SD4 weights only
};

// Weight tensor in the artifact's layout.
template <bool kSd4>
static __device__ __forceinline__ Matrix weights(const unsigned char* w, unsigned int k, unsigned int rows, float* global) {
    const unsigned long long padded = (k + 127u) / 128u * 128u;
    Matrix m;
    m.row_bytes = (unsigned int)(padded / 2u);
    m.row_scales = (unsigned int)(padded / (kSd4 ? 32u : 16u));
    const unsigned long long scale_offset = align256((unsigned long long)rows * m.row_bytes);
    m.codes = w;
    m.scales = w + scale_offset;
    const unsigned char* end = w + align256(scale_offset + (unsigned long long)rows * m.row_scales);
    if (kSd4) {
        m.table.codes = *reinterpret_cast<const uint4*>(end);
        end += 16;
    }
    *global = *reinterpret_cast<const float*>(end);
    return m;
}

// Four B scale codes from the staged scales of one row: E4M3 bytes, or (kSd4) two bytes of indices.
template <bool kSd4>
static __device__ __forceinline__ unsigned int b_scale_quad(const unsigned char* row_scales, unsigned int quad,
                                                            const Matrix& b) {
    if (kSd4) return b.table.lookup(*reinterpret_cast<const unsigned short*>(row_scales + 2u * quad));
    return *reinterpret_cast<const unsigned int*>(row_scales + 4u * quad);
}

// Activation buffer: `terms` planes of `rows` rows (term t at rows t * rows..), codes then scales, then
// one FP32 global per row shared by every term. Term 0 quantizes the activations; term 1, when present,
// quantizes the residual x - term0, so their sum carries about 1% error instead of about 10%.
struct ActivationLayout {
    unsigned long long scale_offset, global_offset, bytes;
    __host__ __device__ ActivationLayout(unsigned int rows, unsigned int k, unsigned int terms) {
        const unsigned long long padded = (k + 127u) / 128u * 128u, planes = (unsigned long long)rows * terms;
        scale_offset = align256(planes * (padded / 2u));
        global_offset = align256(scale_offset + planes * (padded / 16u));
        bytes = global_offset + 4ull * rows;
    }
};

// ---------------------------------------------------------------------------------------------------
// Activation quantization: one CTA per row. global = amax / (6 * 448), so every block scale fits E4M3;
// block scale = e4m3_rn(block amax / (6 * global)); code = e2m1_rn(x / (scale * global)), saturating.
// K must be a multiple of 128 (no padding is written).
static constexpr int kQuantizeThreads = 128;

static __device__ __forceinline__ float block_amax(const __nv_bfloat16* x) {
    const uint4 lo = *reinterpret_cast<const uint4*>(x), hi = *reinterpret_cast<const uint4*>(x + 8);
    const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&lo);
    const __nv_bfloat162* q = reinterpret_cast<const __nv_bfloat162*>(&hi);
    float m = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 a = __bfloat1622float2(p[i]), b = __bfloat1622float2(q[i]);
        m = fmaxf(m, fmaxf(fmaxf(fabsf(a.x), fabsf(a.y)), fmaxf(fabsf(b.x), fabsf(b.y))));
    }
    return m;
}

// Quantizes one 16-value block: writes its E4M3 scale, returns its 8 code bytes as two words, and
// leaves the dequantized values in `v` replaced by the residual.
static __device__ __forceinline__ uint2 quantize16(float (&v)[16], float global, float inverse_global,
                                                   unsigned char* scale_out) {
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) amax = fmaxf(amax, fabsf(v[i]));
    const __nv_fp8_storage_t scale = __nv_cvt_float_to_fp8(amax * (1.0f / 6.0f) * inverse_global, __NV_SATFINITE, __NV_E4M3);
    const float decoded = __half2float(__nv_cvt_fp8_to_halfraw(scale, __NV_E4M3));
    const float inverse = decoded > 0.0f ? inverse_global / decoded : 0.0f;
    const float step = decoded * global;
    unsigned int packed[2] = {0u, 0u};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const __nv_fp4x2_storage_t pair =
                __nv_cvt_float2_to_fp4x2(make_float2(v[2 * i] * inverse, v[2 * i + 1] * inverse), __NV_E2M1, cudaRoundNearest);
        packed[i >> 2] |= (unsigned int)pair << (8 * (i & 3));
        const __half2_raw back = __nv_cvt_fp4x2_to_halfraw2(pair, __NV_E2M1);
        const float2 q = __half22float2(*reinterpret_cast<const __half2*>(&back));
        v[2 * i] -= q.x * step;
        v[2 * i + 1] -= q.y * step;
    }
    *scale_out = scale;
    return make_uint2(packed[0], packed[1]);
}

template <int kTerms>
static __device__ __forceinline__ void quantize_rows(
        const __nv_bfloat16* input, unsigned char* output, unsigned int rows, unsigned int k) {
    const unsigned int row = blockIdx.x;
    const ActivationLayout layout(rows, k, kTerms);
    const unsigned int blocks = k / 16u;
    const __nv_bfloat16* x = input + (unsigned long long)row * k;
    euhedral_pdl_begin();
    float amax = 0.0f;
    for (unsigned int b = threadIdx.x; b < blocks; b += kQuantizeThreads) amax = fmaxf(amax, block_amax(x + 16u * b));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    __shared__ float warp_max[kQuantizeThreads / 32];
    if ((threadIdx.x & 31u) == 0) warp_max[threadIdx.x >> 5] = amax;
    __syncthreads();
    amax = 0.0f;
#pragma unroll
    for (int w = 0; w < kQuantizeThreads / 32; ++w) amax = fmaxf(amax, warp_max[w]);
    const float global = amax > 0.0f ? amax / (6.0f * 448.0f) : 1.0f;
    const float inverse_global = 1.0f / global;
    for (unsigned int b = threadIdx.x; b < blocks; b += kQuantizeThreads) {
        float v[16];
        const uint4 lo = *reinterpret_cast<const uint4*>(x + 16u * b), hi = *reinterpret_cast<const uint4*>(x + 16u * b + 8u);
        const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&lo);
        const __nv_bfloat162* q = reinterpret_cast<const __nv_bfloat162*>(&hi);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 a = __bfloat1622float2(p[i]), c = __bfloat1622float2(q[i]);
            v[2 * i] = a.x; v[2 * i + 1] = a.y; v[8 + 2 * i] = c.x; v[8 + 2 * i + 1] = c.y;
        }
#pragma unroll
        for (int t = 0; t < kTerms; ++t) {
            const unsigned long long plane_row = (unsigned long long)t * rows + row;
            const uint2 codes = quantize16(v, global, inverse_global, output + layout.scale_offset + plane_row * blocks + b);
            *reinterpret_cast<uint2*>(output + plane_row * (k / 2u) + 8u * b) = codes;
        }
    }
    if (threadIdx.x == 0) reinterpret_cast<float*>(output + layout.global_offset)[row] = global;
}

// ---------------------------------------------------------------------------------------------------
// Linear: output[M][N] (BF16) = activations[M][K] * weights[N][K]^T, both NVFP4.
// CTA tile 128 x 128, K tile 128 values (64 code bytes per row), a cp.async pipeline of Stages<kTerms>
// stages. 8 warps as 2 (M) x 4 (N); a warp owns 64 x 32: 4 m16 fragments by 4 n8 fragments, and issues
// kTerms MMAs per fragment and K step (one per activation term) into one FP32 accumulator.
// Shared code rows are padded to 80 bytes, so the eight 16-byte rows an ldmatrix phase reads fall in
// distinct bank groups.
static constexpr int kBM = 128, kBN = 128, kBK = 128, kThreads = 256;
static constexpr int kRowBytes = kBK / 2, kRowStride = kRowBytes + 16, kScaleBytes = kBK / 16;
static constexpr int kCodeTile = kBM * kRowStride;            // A and B code tiles are the same size
static constexpr int kScaleTile = kBM * kScaleBytes;
template <int kTerms> struct Pipeline {
    // Stage: A code tiles (one per term), the B code tile, A scale tiles, the B scale tile.
    static constexpr int kStageBytes = (kTerms + 1) * (kCodeTile + kScaleTile);
    static constexpr int kStages = kTerms == 1 ? 3 : 2;       // within the 99 KiB per-block limit
    static constexpr int kSharedBytes = kStages * kStageBytes;
};

static __device__ __forceinline__ unsigned int smem_address(const void* p) {
    return (unsigned int)__cvta_generic_to_shared(p);
}

static __device__ __forceinline__ void cp_async(void* dst, const void* src, int bytes, bool valid) {
    const int size = valid ? bytes : 0;
    if (bytes == 16)
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(smem_address(dst)), "l"(src), "r"(size));
    else if (bytes == 8)
        asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;" ::"r"(smem_address(dst)), "l"(src), "r"(size));
    else
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" ::"r"(smem_address(dst)), "l"(src), "r"(size));
}

static __device__ __forceinline__ void ldmatrix_x4(unsigned int (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_address(p)));
}

template <int SA, int SB>
static __device__ __forceinline__ void mma(float (&d)[4], const unsigned int (&a)[4], unsigned int b0, unsigned int b1,
                                           unsigned int sa, unsigned int sb) {
    asm volatile(
            "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0, %12}, %11, {0, %13};"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sa), "r"(sb), "n"(SA), "n"(SB));
}

// Weight row held in row `r` of a B tile starting at output column n0. A plain linear holds rows
// n0..n0+127. A paired gate/up tile (half = gate rows = up rows) holds 64 output columns: warp slice
// r / 32 holds gate rows n0 + 16 (r / 32) + 0..15, then the matching up rows (half + the same).
template <bool kPaired>
static __device__ __forceinline__ unsigned int b_row(unsigned int r, unsigned int n0, unsigned int cols, bool* valid) {
    if (!kPaired) {
        *valid = n0 + r < cols;
        return n0 + r;
    }
    const unsigned int half = cols / 2u, column = n0 + (r >> 5) * 16u + (r & 15u);
    *valid = column < half;
    return column + ((r >> 4) & 1u) * half;
}

// Issues the cp.async copies of K tile `tile` into `stage`. Rows past the matrix are zero-filled.
template <int kTerms, bool kPaired, bool kSd4>
static __device__ __forceinline__ void load_stage(
        unsigned char* stage, const Matrix& a, const Matrix& b, unsigned int m0, unsigned int n0,
        unsigned int rows, unsigned int cols, unsigned int tile) {
    unsigned char* b_codes = stage + kTerms * kCodeTile;
    unsigned char* scales = stage + (kTerms + 1) * kCodeTile;
    unsigned char* b_scales = scales + kTerms * kScaleTile;
    const unsigned int byte0 = tile * kRowBytes, scale0 = tile * kScaleBytes;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const unsigned int chunk = threadIdx.x + i * kThreads;     // 512 chunks of 16 bytes per operand
        const unsigned int r = chunk >> 2, c = (chunk & 3u) * 16u;
        const bool va = m0 + r < rows;
#pragma unroll
        for (int t = 0; t < kTerms; ++t)
            cp_async(stage + t * kCodeTile + r * kRowStride + c,
                     a.codes + ((unsigned long long)t * rows + (va ? m0 + r : 0)) * a.row_bytes + byte0 + c, 16, va);
        bool vb;
        const unsigned int br = b_row<kPaired>(r, n0, cols, &vb);
        cp_async(b_codes + r * kRowStride + c, b.codes + (unsigned long long)(vb ? br : 0) * b.row_bytes + byte0 + c, 16, vb);
    }
    const unsigned int r = threadIdx.x & 127u;
    if (threadIdx.x < 128u) {
        const bool v = m0 + r < rows;
#pragma unroll
        for (int t = 0; t < kTerms; ++t)
            cp_async(scales + t * kScaleTile + r * kScaleBytes,
                     a.scales + ((unsigned long long)t * rows + (v ? m0 + r : 0)) * a.row_scales + scale0, 8, v);
    } else {
        bool v;
        const unsigned int br = b_row<kPaired>(r, n0, cols, &v);
        constexpr int bytes = kSd4 ? kScaleBytes / 2 : kScaleBytes;
        cp_async(b_scales + r * kScaleBytes, b.scales + (unsigned long long)(v ? br : 0) * b.row_scales + (kSd4 ? tile * bytes : scale0),
                 bytes, v);
    }
}

// kPaired: `cols` weight rows are gate rows then up rows; output[M][cols / 2] = SwiGLU(gate, up), with
// gate and up rounded to BF16 first, as the BF16 regions do.
template <int kTerms, bool kPaired, bool kSd4 = false>
static __device__ __forceinline__ void linear(
        const unsigned char* activations, const unsigned char* weight_tensor, __nv_bfloat16* output,
        unsigned int rows, unsigned int k, unsigned int cols) {
    using P = Pipeline<kTerms>;
    extern __shared__ __align__(128) unsigned char shared[];
    const ActivationLayout act(rows, k, kTerms);
    const Matrix a{activations, activations + act.scale_offset, k / 2u, k / 16u};
    float weight_global;
    const Matrix b = weights<kSd4>(weight_tensor, k, cols, &weight_global);
    const float* row_globals = reinterpret_cast<const float*>(activations + act.global_offset);

    const unsigned int tile_cols = kPaired ? kBN / 2 : kBN;
    const unsigned int out_cols = kPaired ? cols / 2u : cols;
    const unsigned int tiles_n = (out_cols + tile_cols - 1) / tile_cols;
    const unsigned int m0 = (blockIdx.x / tiles_n) * kBM, n0 = (blockIdx.x % tiles_n) * tile_cols;
    const unsigned int lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    const unsigned int wm = (warp >> 2) * 64u, wn = (warp & 3u) * 32u;
    const unsigned int g = lane >> 2, t = lane & 3u;
    const unsigned int k_tiles = k / kBK;

    float acc[4][4][4] = {};
    euhedral_pdl_begin();
#pragma unroll
    for (int s = 0; s < P::kStages - 1; ++s) {
        if ((unsigned int)s < k_tiles) load_stage<kTerms, kPaired, kSd4>(shared + s * P::kStageBytes, a, b, m0, n0, rows, cols, s);
        asm volatile("cp.async.commit_group;");
    }
    for (unsigned int tile = 0; tile < k_tiles; ++tile) {
        asm volatile("cp.async.wait_group %0;" ::"n"(P::kStages - 2));
        __syncthreads();
        const unsigned int next = tile + P::kStages - 1;
        if (next < k_tiles) load_stage<kTerms, kPaired, kSd4>(shared + (next % P::kStages) * P::kStageBytes, a, b, m0, n0, rows, cols, next);
        asm volatile("cp.async.commit_group;");

        const unsigned char* stage = shared + (tile % P::kStages) * P::kStageBytes;
        const unsigned char* b_codes = stage + kTerms * kCodeTile;
        const unsigned char* a_scales = stage + (kTerms + 1) * kCodeTile;
        const unsigned char* b_scales = a_scales + kTerms * kScaleTile;
        // A scales: lane t serves row g + 8 (t & 1) for K step t >> 1 (selector 0 for step 0, 1 for
        // step 1). B scales: lane t serves column g for K step t (selectors 0 and 1).
        unsigned int sa[kTerms][4], sb[4];
#pragma unroll
        for (int term = 0; term < kTerms; ++term)
#pragma unroll
            for (int mf = 0; mf < 4; ++mf)
                sa[term][mf] = *reinterpret_cast<const unsigned int*>(
                        a_scales + term * kScaleTile + (wm + mf * 16u + g + 8u * (t & 1u)) * kScaleBytes + 4u * (t >> 1));
#pragma unroll
        for (int nf = 0; nf < 4; ++nf)
            sb[nf] = b_scale_quad<kSd4>(b_scales + (wn + nf * 8u + g) * kScaleBytes, t & 1u, b);
#pragma unroll
        for (int step = 0; step < 2; ++step) {
            const unsigned int mat = lane >> 3, r8 = lane & 7u;
            unsigned int bf[2][4];
#pragma unroll
            for (int np = 0; np < 2; ++np)
                ldmatrix_x4(bf[np], b_codes + (wn + np * 16u + r8 + 8u * (mat >> 1)) * kRowStride + 32u * step + 16u * (mat & 1u));
#pragma unroll
            for (int term = 0; term < kTerms; ++term) {
                unsigned int af[4][4];
#pragma unroll
                for (int mf = 0; mf < 4; ++mf)
                    ldmatrix_x4(af[mf], stage + term * kCodeTile + (wm + mf * 16u + r8 + 8u * (mat & 1u)) * kRowStride + 32u * step + 16u * (mat >> 1));
#pragma unroll
                for (int mf = 0; mf < 4; ++mf)
#pragma unroll
                    for (int nf = 0; nf < 4; ++nf) {
                        const unsigned int b0 = bf[nf >> 1][(nf & 1) * 2], b1 = bf[nf >> 1][(nf & 1) * 2 + 1];
                        if (step == 0) mma<0, 0>(acc[mf][nf], af[mf], b0, b1, sa[term][mf], sb[nf]);
                        else mma<1, 1>(acc[mf][nf], af[mf], b0, b1, sa[term][mf], sb[nf]);
                    }
            }
        }
    }
    asm volatile("cp.async.wait_group 0;");
    if (kPaired) {
        // Fragments 0 and 1 hold gate columns n0 + wn/2 + 0..15, fragments 2 and 3 the matching up columns.
#pragma unroll
        for (int mf = 0; mf < 4; ++mf) {
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const unsigned int row = m0 + wm + mf * 16u + g + 8u * half;
                if (row >= rows) continue;
                const float scale = row_globals[row] * weight_global;
#pragma unroll
                for (int nf = 0; nf < 2; ++nf) {
#pragma unroll
                    for (int i = 0; i < 2; ++i) {
                        const unsigned int col = n0 + wn / 2u + nf * 8u + 2u * t + i;
                        if (col >= out_cols) continue;
                        const float gate = __bfloat162float(__float2bfloat16_rn(acc[mf][nf][2 * half + i] * scale));
                        const float up = __bfloat162float(__float2bfloat16_rn(acc[mf][nf + 2][2 * half + i] * scale));
                        output[(unsigned long long)row * out_cols + col] = __float2bfloat16_rn((gate / (1.0f + expf(-gate))) * up);
                    }
                }
            }
        }
        return;
    }
#pragma unroll
    for (int mf = 0; mf < 4; ++mf) {
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const unsigned int row = m0 + wm + mf * 16u + g + 8u * half;
            if (row >= rows) continue;
            const float scale = row_globals[row] * weight_global;
#pragma unroll
            for (int nf = 0; nf < 4; ++nf) {
                const unsigned int col = n0 + wn + nf * 8u + 2u * t;
                if (col >= cols) continue;
                const float x = acc[mf][nf][2 * half] * scale, y = acc[mf][nf][2 * half + 1] * scale;
                __nv_bfloat16* out = output + (unsigned long long)row * cols + col;
                if (col + 1u < cols) *reinterpret_cast<__nv_bfloat162*>(out) = __floats2bfloat162_rn(x, y);
                else *out = __float2bfloat16_rn(x);
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------
// Skinny linear for decode-like row counts (M <= 16 * kMF): weight streaming, not MMA, is the cost.
// CTA: 4 warps over 64 output columns (one warp per 16 columns, two n8 fragments) and all M rows
// (kMF m16 fragments); K tile 256 values (128 code bytes per row); S::kStages-deep cp.async pipeline
// so each SM keeps several weight tiles in flight. gridDim.y splits K: with one split the CTA writes
// BF16; otherwise it stores its scaled FP32 partials in its own slice of `partials` (splits x rows x
// cols) and skinny_finish sums the slices in split order, so the result is deterministic and a row's
// output does not depend on the other rows (row-independent MMAs, per-row activation scales).
static constexpr int kSkinnyBK = 256, kSkinnyRowBytes = kSkinnyBK / 2, kSkinnyStride = kSkinnyRowBytes + 16;
static constexpr int kSkinnyScaleBytes = kSkinnyBK / 16, kSkinnyCols = 64, kSkinnyThreads = 128;
static constexpr int kSharedLimit = 101376;                   // sm_120 per-block opt-in
template <int kTerms, int kMF> struct Skinny {
    static constexpr int kRows = 16 * kMF;
    static constexpr int kStageBytes = (kTerms * kRows + kSkinnyCols) * (kSkinnyStride + kSkinnyScaleBytes);
    static constexpr int kStages = kSharedLimit / kStageBytes >= 4 ? 4 : kSharedLimit / kStageBytes;
    static constexpr int kSharedBytes = kStages * kStageBytes;
};

template <int kTerms, int kMF, bool kSd4>
static __device__ __forceinline__ void skinny_load(unsigned char* stage, const Matrix& a, const Matrix& b,
        unsigned int rows, unsigned int n0, unsigned int cols, unsigned int tile) {
    constexpr int R = Skinny<kTerms, kMF>::kRows;
    unsigned char* a_codes = stage;
    unsigned char* b_codes = stage + kTerms * R * kSkinnyStride;
    unsigned char* a_scales = b_codes + kSkinnyCols * kSkinnyStride;
    unsigned char* b_scales = a_scales + kTerms * R * kSkinnyScaleBytes;
    const unsigned int byte0 = tile * kSkinnyRowBytes, scale0 = tile * kSkinnyScaleBytes;
    // B: 64 rows x 8 chunks of 16 bytes = 512 chunks, 4 per thread; scales 64 x 16 bytes, one chunk each.
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const unsigned int chunk = threadIdx.x + i * kSkinnyThreads, r = chunk >> 3, c = (chunk & 7u) * 16u;
        const bool v = n0 + r < cols;
        cp_async(b_codes + r * kSkinnyStride + c, b.codes + (unsigned long long)(v ? n0 + r : 0) * b.row_bytes + byte0 + c, 16, v);
    }
    if (threadIdx.x < kSkinnyCols) {
        const unsigned int r = threadIdx.x;
        const bool v = n0 + r < cols;
        constexpr int bytes = kSd4 ? kSkinnyScaleBytes / 2 : kSkinnyScaleBytes;
        cp_async(b_scales + r * kSkinnyScaleBytes,
                 b.scales + (unsigned long long)(v ? n0 + r : 0) * b.row_scales + (kSd4 ? tile * bytes : scale0), bytes, v);
    }
    // A: kTerms * R rows x 8 chunks; scales kTerms * R chunks.
    for (unsigned int chunk = threadIdx.x; chunk < (unsigned int)(kTerms * R * 8); chunk += kSkinnyThreads) {
        const unsigned int pr = chunk >> 3, c = (chunk & 7u) * 16u, term = pr / R, r = pr % R;
        const bool v = r < rows;
        cp_async(a_codes + pr * kSkinnyStride + c, a.codes + ((unsigned long long)term * rows + (v ? r : 0)) * a.row_bytes + byte0 + c, 16, v);
    }
    for (unsigned int pr = threadIdx.x; pr < (unsigned int)(kTerms * R); pr += kSkinnyThreads) {
        const unsigned int term = pr / R, r = pr % R;
        const bool v = r < rows;
        cp_async(a_scales + pr * kSkinnyScaleBytes, a.scales + ((unsigned long long)term * rows + (v ? r : 0)) * a.row_scales + scale0, 16, v);
    }
}

template <int kTerms, int kMF, bool kSd4 = false>
static __device__ __forceinline__ void skinny_linear(
        const unsigned char* activations, const unsigned char* weight_tensor, __nv_bfloat16* output, float* partials,
        unsigned int rows, unsigned int k, unsigned int cols) {
    using S = Skinny<kTerms, kMF>;
    constexpr int R = S::kRows;
    extern __shared__ __align__(128) unsigned char shared[];
    const ActivationLayout act(rows, k, kTerms);
    const Matrix a{activations, activations + act.scale_offset, k / 2u, k / 16u};
    float weight_global;
    const Matrix b = weights<kSd4>(weight_tensor, k, cols, &weight_global);
    const float* row_globals = reinterpret_cast<const float*>(activations + act.global_offset);
    const unsigned int n0 = blockIdx.x * kSkinnyCols;
    const unsigned int lane = threadIdx.x & 31u, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3u;
    const unsigned int k_tiles = k / kSkinnyBK, splits = gridDim.y;
    const unsigned int first = (unsigned int)((unsigned long long)k_tiles * blockIdx.y / splits);
    const unsigned int last = (unsigned int)((unsigned long long)k_tiles * (blockIdx.y + 1) / splits);
    float acc[kMF][2][4] = {};
    euhedral_pdl_begin();
#pragma unroll
    for (int s = 0; s < S::kStages - 1; ++s) {
        if (first + s < last) skinny_load<kTerms, kMF, kSd4>(shared + s * S::kStageBytes, a, b, rows, n0, cols, first + s);
        asm volatile("cp.async.commit_group;");
    }
    for (unsigned int tile = first; tile < last; ++tile) {
        asm volatile("cp.async.wait_group %0;" ::"n"(S::kStages - 2));
        __syncthreads();
        const unsigned int next = tile + S::kStages - 1;
        if (next < last) skinny_load<kTerms, kMF, kSd4>(shared + ((next - first) % S::kStages) * S::kStageBytes, a, b, rows, n0, cols, next);
        asm volatile("cp.async.commit_group;");
        const unsigned char* stage = shared + ((tile - first) % S::kStages) * S::kStageBytes;
        const unsigned char* b_codes = stage + kTerms * R * kSkinnyStride;
        const unsigned char* a_scales = b_codes + kSkinnyCols * kSkinnyStride;
        const unsigned char* b_scales = a_scales + kTerms * R * kSkinnyScaleBytes;
        const unsigned int mat = lane >> 3, r8 = lane & 7u, wn = warp * 16u;
#pragma unroll
        for (int pair = 0; pair < 2; ++pair) {   // K steps 2 pair, 2 pair + 1 share scale registers
            // B scales for steps 2 pair (selector 0, lanes t = 0) and 2 pair + 1 (selector 1, lanes t = 1).
            unsigned int sb[2];
#pragma unroll
            for (int nf = 0; nf < 2; ++nf)
                sb[nf] = b_scale_quad<kSd4>(b_scales + (wn + nf * 8u + g) * kSkinnyScaleBytes, 2u * pair + (t & 1u), b);
            unsigned int sa[kTerms][kMF];
#pragma unroll
            for (int term = 0; term < kTerms; ++term)
#pragma unroll
                for (int mf = 0; mf < kMF; ++mf)
                    sa[term][mf] = *reinterpret_cast<const unsigned int*>(
                            a_scales + (term * R + mf * 16u + g + 8u * (t & 1u)) * kSkinnyScaleBytes + 8u * pair + 4u * (t >> 1));
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const unsigned int step = 2u * pair + half;
                unsigned int bf[4];
                ldmatrix_x4(bf, b_codes + (wn + r8 + 8u * (mat >> 1)) * kSkinnyStride + 32u * step + 16u * (mat & 1u));
#pragma unroll
                for (int term = 0; term < kTerms; ++term)
#pragma unroll
                    for (int mf = 0; mf < kMF; ++mf) {
                        unsigned int af[4];
                        ldmatrix_x4(af, stage + (term * R + mf * 16u + r8 + 8u * (mat & 1u)) * kSkinnyStride + 32u * step + 16u * (mat >> 1));
#pragma unroll
                        for (int nf = 0; nf < 2; ++nf) {
                            if (half == 0) mma<0, 0>(acc[mf][nf], af, bf[nf * 2], bf[nf * 2 + 1], sa[term][mf], sb[nf]);
                            else mma<1, 1>(acc[mf][nf], af, bf[nf * 2], bf[nf * 2 + 1], sa[term][mf], sb[nf]);
                        }
                    }
            }
        }
    }
    asm volatile("cp.async.wait_group 0;");
#pragma unroll
    for (int mf = 0; mf < kMF; ++mf)
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const unsigned int row = mf * 16u + g + 8u * half;
            if (row >= rows) continue;
            const float scale = row_globals[row] * weight_global;
#pragma unroll
            for (int nf = 0; nf < 2; ++nf) {
                const unsigned int col = n0 + warp * 16u + nf * 8u + 2u * t;
                if (col >= cols) continue;
                const float x = acc[mf][nf][2 * half] * scale, y = acc[mf][nf][2 * half + 1] * scale;
                if (splits == 1) {
                    __nv_bfloat16* out = output + (unsigned long long)row * cols + col;
                    if (col + 1u < cols) *reinterpret_cast<__nv_bfloat162*>(out) = __floats2bfloat162_rn(x, y);
                    else *out = __float2bfloat16_rn(x);
                } else {
                    float* p = partials + ((unsigned long long)blockIdx.y * rows + row) * cols + col;
                    p[0] = x;
                    if (col + 1u < cols) p[1] = y;
                }
            }
        }
}

}  // namespace nvfp4n
