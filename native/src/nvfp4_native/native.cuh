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
    unsigned int pad;         // activations: rows of a scale plane (a multiple of 128)
    nvfp4::ScaleTable table;  // SD4 weights only
};

// Weight tensor in the artifact's layout.
template <bool kSd4>
static __device__ __forceinline__ Matrix weights(const unsigned char* w, unsigned int k, unsigned int rows, float* global) {
    const unsigned long long padded = (k + 127u) / 128u * 128u;
    Matrix m;
    m.pad = 0;
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

// Activation buffer: `terms` planes of `rows` rows of codes, then the scales, then one FP32 global per row
// shared by every term. Term 0 quantizes the activations; term 1, when present, quantizes the residual
// x - term0, so their sum carries about 1% error instead of about 10%.
// The scales are tile-major: the 8 scale bytes of a row for K tile j (128 values) of term t are at
//   scales + ((j * terms + t) * pad + row) * 8,  pad = rows rounded up to 128,
// so the scales of 128 rows of one K tile and term are one contiguous KiB (one bulk copy).
struct ActivationLayout {
    unsigned long long scale_offset, global_offset, bytes;
    unsigned int pad;
    __host__ __device__ ActivationLayout(unsigned int rows, unsigned int k, unsigned int terms) {
        const unsigned long long padded = (k + 127u) / 128u * 128u, planes = (unsigned long long)rows * terms;
        pad = (rows + 127u) / 128u * 128u;
        scale_offset = align256(planes * (padded / 2u));
        global_offset = align256(scale_offset + (padded / 128u) * terms * pad * 8ull);
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
            unsigned char* scale_out = output + layout.scale_offset + (((unsigned long long)(b >> 3) * kTerms + t) * layout.pad + row) * 8u + (b & 7u);
            const uint2 codes = quantize16(v, global, inverse_global, scale_out);
            *reinterpret_cast<uint2*>(output + plane_row * (k / 2u) + 8u * b) = codes;
        }
    }
    if (threadIdx.x == 0) reinterpret_cast<float*>(output + layout.global_offset)[row] = global;
}

// ---------------------------------------------------------------------------------------------------
// Linear: output[M][N] (BF16) = activations[M][K] * weights[N][K]^T, both NVFP4.
//
// CTA tile 128 x 128 (a paired gate/up tile: 64 gate and 64 up weight rows, 64 outputs) and K tile 128 values.
// A producer warp (warp 8, one elected lane) streams the operands with TMA into a ring of kStages stages
// guarded by mbarriers; 8 consumer warps (2 along M x 4 along N, 64 x 32 each) run the MMAs without ever
// issuing a load: a warp issues kTerms MMAs per fragment and K step (one per activation term) into one FP32
// accumulator, and releases a stage with one mbarrier arrival per warp.
//   codes   tensor maps of the code planes, 64-byte rows (one K tile of a row), SWIZZLE_64B: chunk c of row r is
//           stored at chunk c ^ ((r >> 1) & 3), which makes every ldmatrix phase conflict-free.
//   A scales   one bulk copy per term of the tile-major activation scales (ActivationLayout).
//   B scales   a tensor map of the weight scales whose 16-byte boxes cover kScaleGroup K tiles (2 for E4M3 scales,
//              4 for the 4-bit SD4 indices); the group is fetched with the first tile that needs it and lives in
//              one of two slots.
// Every code and scale element a consumer reads is bit for bit the one the cp.async tile read, in the same
// order, so the results are unchanged.
struct alignas(64) TensorMap { unsigned long long opaque[16]; };

static constexpr int kBM = 128, kBN = 128, kBK = 128;
static constexpr int kTile = kBM * 64;                       // bytes of one operand's codes for one K tile
static constexpr int kAScale = kBM * 8;                      // activation scales of one tile and one term
static constexpr int kStages = 3;
static constexpr int kThreads = 288;                         // 8 consumer warps and the producer warp
static constexpr int kSlotBytes = kBN * 16;                  // B scales of one group of K tiles
template <int kTerms> struct Pipeline {
    static constexpr int kStageBytes = (kTerms + 1) * kTile + kTerms * kAScale;
    static constexpr int kSharedBytes = kStages * kStageBytes + 2 * kSlotBytes + 2 * kStages * 8 + 1024;
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

static __device__ __forceinline__ void mbar_init(unsigned long long* bar, unsigned int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_address(bar)), "r"(count));
}
static __device__ __forceinline__ void mbar_expect_tx(unsigned long long* bar, unsigned int bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_address(bar)), "r"(bytes) : "memory");
}
static __device__ __forceinline__ void mbar_arrive(unsigned long long* bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_address(bar)) : "memory");
}
static __device__ __forceinline__ void mbar_wait(unsigned long long* bar, unsigned int parity) {
    asm volatile(
            "{\n.reg .pred p;\nWAIT_%=:\n"
            "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
            "@!p bra WAIT_%=;\n}" ::"r"(smem_address(bar)), "r"(parity) : "memory");
}
static __device__ __forceinline__ void tma_2d(void* dst, const TensorMap* map, int c0, int c1, unsigned long long* bar) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
                 ::"r"(smem_address(dst)), "l"(map), "r"(c0), "r"(c1), "r"(smem_address(bar)) : "memory");
}
static __device__ __forceinline__ void tma_3d(void* dst, const TensorMap* map, int c0, int c1, int c2, unsigned long long* bar) {
    asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];"
                 ::"r"(smem_address(dst)), "l"(map), "r"(c0), "r"(c1), "r"(c2), "r"(smem_address(bar)) : "memory");
}
static __device__ __forceinline__ void bulk_copy(void* dst, const void* src, unsigned int bytes, unsigned long long* bar) {
    asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 ::"r"(smem_address(dst)), "l"(src), "r"(bytes), "r"(smem_address(bar)) : "memory");
}
// Byte offset of 16-byte chunk c16 of row r of a SWIZZLE_64B tile with 64-byte rows.
static __device__ __forceinline__ unsigned int swizzled(unsigned int r, unsigned int c16) {
    return r * 64u + ((c16 ^ ((r >> 1) & 3u)) << 4);
}

// kPaired: `cols` weight rows are gate rows then up rows; output[M][cols / 2] = SwiGLU(gate, up), with
// gate and up rounded to BF16 first, as the BF16 regions do. A paired tile holds 64 gate rows and the
// matching 64 up rows; its warps' fragments 0 and 1 are 16 gate columns, fragments 2 and 3 the same up columns.
// The maps: tm_a over the code planes (K / 2 bytes, rows, terms), tm_b over the weight codes (K / 2, cols), tm_bs
// over the weight scales (row_scales bytes, cols).
template <int kTerms, bool kPaired, bool kSd4 = false>
static __device__ __forceinline__ void linear(
        const TensorMap* tm_a, const TensorMap* tm_b, const TensorMap* tm_bs,
        const unsigned char* activations, const unsigned char* weight_tensor, __nv_bfloat16* output,
        unsigned int rows, unsigned int k, unsigned int cols) {
    using P = Pipeline<kTerms>;
    constexpr unsigned int kScaleGroup = kSd4 ? 4u : 2u;       // K tiles per 16-byte B scale box
    constexpr unsigned int kTileScaleBytes = kSd4 ? 4u : 8u;   // B scale bytes of one tile and row
    extern __shared__ __align__(1024) unsigned char smem_raw[];
    unsigned char* shared = reinterpret_cast<unsigned char*>((reinterpret_cast<unsigned long long>(smem_raw) + 1023ull) & ~1023ull);
    unsigned char* slots = shared + kStages * P::kStageBytes;
    unsigned long long* full = reinterpret_cast<unsigned long long*>(slots + 2 * kSlotBytes);
    unsigned long long* empty = full + kStages;

    const ActivationLayout act(rows, k, kTerms);
    const unsigned char* a_scales_global = activations + act.scale_offset;
    const float* row_globals = reinterpret_cast<const float*>(activations + act.global_offset);
    float weight_global;
    const Matrix b = weights<kSd4>(weight_tensor, k, cols, &weight_global);

    constexpr unsigned int tile_cols = kPaired ? kBN / 2 : kBN;
    const unsigned int out_cols = kPaired ? cols / 2u : cols;
    const unsigned int tiles_m = (rows + kBM - 1) / kBM;
    // M varies fastest: the CTAs running together share weight tiles.
    const unsigned int m0 = (blockIdx.x % tiles_m) * kBM, n0 = (blockIdx.x / tiles_m) * tile_cols;
    const unsigned int k_tiles = k / kBK;
    const unsigned int warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;

    if (threadIdx.x == 0) {
        for (int s = 0; s < kStages; ++s) {
            mbar_init(&full[s], 1);
            mbar_init(&empty[s], 8);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    euhedral_pdl_begin();

    if (warp == 8) {
        if (lane == 0) {
            const unsigned int half = cols / 2u;
            for (unsigned int tile = 0; tile < k_tiles; ++tile) {
                const unsigned int s = tile % kStages;
                if (tile >= kStages) mbar_wait(&empty[s], ((tile / kStages) - 1u) & 1u);
                unsigned char* stage = shared + s * P::kStageBytes;
                const bool first_of_group = tile % kScaleGroup == 0;
                mbar_expect_tx(&full[s], (kTerms + 1) * kTile + kTerms * kAScale + (first_of_group ? kSlotBytes : 0));
#pragma unroll
                for (int t = 0; t < kTerms; ++t) {
                    tma_3d(stage + t * kTile, tm_a, (int)(tile * 64u), (int)m0, t, &full[s]);
                    bulk_copy(stage + (kTerms + 1) * kTile + t * kAScale,
                              a_scales_global + (((unsigned long long)tile * kTerms + t) * act.pad + m0) * 8ull, kAScale, &full[s]);
                }
                unsigned char* b_dst = stage + kTerms * kTile;
                unsigned char* slot = slots + ((tile / kScaleGroup) & 1u) * kSlotBytes;
                if (kPaired) {
                    tma_2d(b_dst, tm_b, (int)(tile * 64u), (int)n0, &full[s]);
                    tma_2d(b_dst + kTile / 2, tm_b, (int)(tile * 64u), (int)(half + n0), &full[s]);
                    if (first_of_group) {
                        tma_2d(slot, tm_bs, (int)((tile / kScaleGroup) * 16u), (int)n0, &full[s]);
                        tma_2d(slot + kSlotBytes / 2, tm_bs, (int)((tile / kScaleGroup) * 16u), (int)(half + n0), &full[s]);
                    }
                } else {
                    tma_2d(b_dst, tm_b, (int)(tile * 64u), (int)n0, &full[s]);
                    if (first_of_group) tma_2d(slot, tm_bs, (int)((tile / kScaleGroup) * 16u), (int)n0, &full[s]);
                }
            }
        }
        return;
    }

    const unsigned int wm = (warp >> 2) * 64u, wn = (warp & 3u) * 32u;
    const unsigned int g = lane >> 2, t = lane & 3u, mat = lane >> 3, r8 = lane & 7u;
    // Tile row of B fragment pair np (16 rows) and of the scale row of fragment nf.
    unsigned int brow[2], bsrow[4];
#pragma unroll
    for (int np = 0; np < 2; ++np) brow[np] = kPaired ? (np == 0 ? wn / 2u : kBN / 2u + wn / 2u) : wn + np * 16u;
#pragma unroll
    for (int nf = 0; nf < 4; ++nf)
        bsrow[nf] = kPaired ? ((nf < 2 ? wn / 2u : kBN / 2u + wn / 2u) + (nf & 1) * 8u + g) : wn + nf * 8u + g;

    float acc[4][4][4] = {};
    for (unsigned int tile = 0; tile < k_tiles; ++tile) {
        const unsigned int s = tile % kStages;
        mbar_wait(&full[s], (tile / kStages) & 1u);
        const unsigned char* stage = shared + s * P::kStageBytes;
        const unsigned char* b_codes = stage + kTerms * kTile;
        const unsigned char* a_scales = stage + (kTerms + 1) * kTile;
        const unsigned char* b_scales = slots + ((tile / kScaleGroup) & 1u) * kSlotBytes + (tile % kScaleGroup) * kTileScaleBytes;
        // A scales: lane t serves row g + 8 (t & 1) for K step t >> 1 (selector 0 for step 0, 1 for step 1).
        // B scales: lane t serves column g for K step t (selectors 0 and 1).
        unsigned int sa[kTerms][4], sb[4];
#pragma unroll
        for (int term = 0; term < kTerms; ++term)
#pragma unroll
            for (int mf = 0; mf < 4; ++mf)
                sa[term][mf] = *reinterpret_cast<const unsigned int*>(
                        a_scales + term * kAScale + (wm + mf * 16u + g + 8u * (t & 1u)) * 8u + 4u * (t >> 1));
#pragma unroll
        for (int nf = 0; nf < 4; ++nf) sb[nf] = b_scale_quad<kSd4>(b_scales + bsrow[nf] * 16u, t & 1u, b);
#pragma unroll
        for (int step = 0; step < 2; ++step) {
            unsigned int bf[2][4];
#pragma unroll
            for (int np = 0; np < 2; ++np)
                ldmatrix_x4(bf[np], b_codes + swizzled(brow[np] + r8 + 8u * (mat >> 1), 2u * step + (mat & 1u)));
#pragma unroll
            for (int term = 0; term < kTerms; ++term) {
                unsigned int af[4][4];
#pragma unroll
                for (int mf = 0; mf < 4; ++mf)
                    ldmatrix_x4(af[mf], stage + term * kTile + swizzled(wm + mf * 16u + r8 + 8u * (mat & 1u), 2u * step + (mat >> 1)));
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
        __syncwarp();
        if (lane == 0) mbar_arrive(&empty[s]);
    }
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
    // The 16 scale bytes of a row for this 256-value tile are the two 8-byte pieces of K tiles 2 tile and 2 tile + 1.
    for (unsigned int pr = threadIdx.x; pr < (unsigned int)(kTerms * R); pr += kSkinnyThreads) {
        const unsigned int term = pr / R, r = pr % R;
        const bool v = r < rows;
#pragma unroll
        for (int half = 0; half < 2; ++half)
            cp_async(a_scales + pr * kSkinnyScaleBytes + 8 * half,
                     a.scales + (((unsigned long long)(2u * tile + half) * kTerms + term) * a.pad + (v ? r : 0)) * 8ull, 8, v);
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
    const Matrix a{activations, activations + act.scale_offset, k / 2u, k / 16u, act.pad};
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
