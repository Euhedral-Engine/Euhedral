#pragma once
#include "nvfp4_prefill_fa2.cuh"
// GQA tensor-core decode (relaxed, long contexts): one warp per (KV head, key split) serves the KV head's
// whole group of G <= 8 query heads, so each 16-key tile of the cache is expanded once into the warp's
// shared rows instead of once per query head. The query heads are the N = 8 columns of the mma tiles:
// scores S^T = K Q^T (K rows as A) and the output O^T = V^T P^T (V by transposed ldmatrix as A, P^T from
// shared memory as B), so the 256-dimension output takes 16 x 4 accumulator registers and both the queries
// and the probabilities enter as hi + lo FP16 parts (mma.sync m16n8k16, FP32 accumulation), with the
// accurate expf of the FP32 kernel: within about 1e-7 to 8e-5 relative rms of it. Writes per-(query head, split) partials in the format of
// euhedral_attention_merge_nvfp4.
namespace gqa_decode {
constexpr int D = 256, KT = 16, STRIDE = D + 8;
static __device__ __forceinline__ void stage16(__half* kv, const unsigned char* const* pages, unsigned base, unsigned end,
        unsigned head, unsigned heads, unsigned lane) {
    for (unsigned g = lane; g < KT * nvfp4kv::kGroups; g += 32) {
        const unsigned token = base + g / nvfp4kv::kGroups, group = g % nvfp4kv::kGroups;
        __half2 out[8];
        if (token < end) {
            const unsigned char* row = nvfp4kv::cache_row(pages, token, head, heads);
            const uint2 codes = *reinterpret_cast<const uint2*>(row + group * 8);
            const float scale = nvfp4kv::e4m3_decode(row[nvfp4kv::kCodeBytes + group]);
            __half2 (&lo)[4] = *reinterpret_cast<__half2 (*)[4]>(&out[0]);
            __half2 (&hi)[4] = *reinterpret_cast<__half2 (*)[4]>(&out[4]);
            nvfp4kv::dequantize8_f16(codes.x, scale, lo);
            nvfp4kv::dequantize8_f16(codes.y, scale, hi);
        } else {
            for (int i = 0; i < 8; i++) out[i] = __floats2half2_rn(0.0f, 0.0f);
        }
        uint4* dst = reinterpret_cast<uint4*>(kv + (g / nvfp4kv::kGroups) * STRIDE + group * 16);
        dst[0] = *reinterpret_cast<const uint4*>(&out[0]);
        dst[1] = *reinterpret_cast<const uint4*>(&out[4]);
    }
}
}  // namespace gqa_decode
namespace gqa_decode {
// One (KV head, split) block of the decode attention below; `block` is its index in the one-row grid.
static __device__ __forceinline__ void decode_block(unsigned block,
        const __nv_bfloat16* queryKey, const unsigned char* const* keyPages, const unsigned char* const* valuePages,
        unsigned int queryHeads, unsigned int keyHeads, unsigned int cacheLength, float* partial, unsigned int splits) {
    const unsigned lane = threadIdx.x, g = lane >> 2, tig = lane & 3u;
    const unsigned group = queryHeads / keyHeads;
    const unsigned kh = block / splits, split = block % splits;
    const unsigned span = (cacheLength + splits - 1) / splits;
    const unsigned begin = split * span, end = min(cacheLength, begin + span);
    __shared__ __align__(16) __half kbuf[KT * STRIDE], vbuf[KT * STRIDE];
    __shared__ __align__(16) __half pbuf[2][8 * 24];  // P^T source: [query][key], hi and lo, padded rows
    // Rotated queries (rows >= group zero) as hi + lo FP16 B fragments: qb[k][part][2].
    unsigned qh[16][2], ql[16][2];
    {
        __half* qhi = kbuf;              // 8 rows x STRIDE
        __half* qlo = kbuf + 8 * STRIDE;
        for (unsigned r = 0; r < 8; r++) {
            float values[8];
            for (int d = 0; d < 8; d++)
                values[d] = r < group ? __bfloat162float(queryKey[(kh * group + r) * D + lane + d * 32]) : 0.0f;
            if (r < group) nvfp4kv::hadamard256(values, lane);
            for (int d = 0; d < 8; d++) {
                const __half hi = __float2half_rn(values[d]);
                qhi[r * STRIDE + lane + d * 32] = hi;
                qlo[r * STRIDE + lane + d * 32] = __float2half_rn(values[d] - __half2float(hi));
            }
        }
        __syncwarp();
        for (int k = 0; k < 16; k += 2) {
            // x4: queries (lane & 7), dims 16 k + 8 ((lane >> 3) & 1) + 16 (lane >> 4): b0,b1 of step k and k + 1
            unsigned r[4];
            fa2::ldsm_x4(r, qhi + (lane & 7u) * STRIDE + k * 16 + ((lane >> 3) & 1u) * 8 + (lane >> 4) * 16);
            qh[k][0] = r[0]; qh[k][1] = r[1]; qh[k + 1][0] = r[2]; qh[k + 1][1] = r[3];
            fa2::ldsm_x4(r, qlo + (lane & 7u) * STRIDE + k * 16 + ((lane >> 3) & 1u) * 8 + (lane >> 4) * 16);
            ql[k][0] = r[0]; ql[k][1] = r[1]; ql[k + 1][0] = r[2]; ql[k + 1][1] = r[3];
        }
        __syncwarp();
    }
    float o[16][4];
    for (int j = 0; j < 16; j++) { o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.0f; }
    // This lane's queries are columns 2 tig and 2 tig + 1.
    float m[2] = {-__int_as_float(0x7f800000), -__int_as_float(0x7f800000)}, l[2] = {0.0f, 0.0f};
    for (unsigned base = begin; base < end; base += KT) {
        stage16(kbuf, keyPages, base, end, kh, keyHeads, lane);
        stage16(vbuf, valuePages, base, end, kh, keyHeads, lane);
        __syncwarp();
        float s[4] = {0.0f, 0.0f, 0.0f, 0.0f};  // (key g, q 2tig/+1), (key g + 8, q 2tig/+1)
        for (int k = 0; k < 16; k++) {
            unsigned a[4];
            fa2::ldsm_x4(a, kbuf + (lane & 15u) * STRIDE + k * 16 + (lane >> 4) * 8);
            fa2::mma16816(s, a, qh[k][0], qh[k][1]);
            fa2::mma16816(s, a, ql[k][0], ql[k][1]);
        }
        float mx[2], p[4];
        for (int c = 0; c < 2; c++) {
            const bool v0 = base + g < end, v1 = base + g + 8 < end;
            s[c] = v0 ? s[c] * 0.0625f : -__int_as_float(0x7f800000);
            s[2 + c] = v1 ? s[2 + c] * 0.0625f : -__int_as_float(0x7f800000);
            float x = fmaxf(fmaxf(s[c], s[2 + c]), m[c]);
            x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, 4));
            x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, 8));
            x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, 16));
            mx[c] = x;
            p[c] = isfinite(s[c]) ? expf(s[c] - x) : 0.0f;
            p[2 + c] = isfinite(s[2 + c]) ? expf(s[2 + c] - x) : 0.0f;
            float sum = p[c] + p[2 + c];
            sum += __shfl_xor_sync(0xffffffffu, sum, 4);
            sum += __shfl_xor_sync(0xffffffffu, sum, 8);
            sum += __shfl_xor_sync(0xffffffffu, sum, 16);
            const float scale = isfinite(m[c]) ? expf(m[c] - x) : 0.0f;
            l[c] = l[c] * scale + sum;
            m[c] = x;
            for (int j = 0; j < 16; j++) { o[j][c] *= scale; o[j][2 + c] *= scale; }
        }
        // P^T for the B operand: write P as [query][key] hi and lo, then ldmatrix x4 (hi: b0 b1, lo: b0 b1).
        for (int c = 0; c < 2; c++) {
            const unsigned q = 2 * tig + c;
            const __half h0 = __float2half_rn(p[c]), h1 = __float2half_rn(p[2 + c]);
            pbuf[0][q * 24 + g] = h0; pbuf[0][q * 24 + g + 8] = h1;
            pbuf[1][q * 24 + g] = __float2half_rn(p[c] - __half2float(h0));
            pbuf[1][q * 24 + g + 8] = __float2half_rn(p[2 + c] - __half2float(h1));
        }
        __syncwarp();
        unsigned pb[4];
        // matrices: hi keys 0-7, hi keys 8-15, lo keys 0-7, lo keys 8-15; rows = queries (lane & 7)
        fa2::ldsm_x4(pb, &pbuf[lane >> 4][(lane & 7u) * 24 + ((lane >> 3) & 1u) * 8]);
        for (int j = 0; j < 16; j++) {
            unsigned a[4];
            // A = V^T tile: dims 16 j .. +15 (rows), keys 0..15 (cols) from V [key][dim] via transpose
            fa2::ldsm_x4_t(a, vbuf + ((lane & 7u) + ((lane >> 4) << 3)) * STRIDE + j * 16 + ((lane >> 3) & 1u) * 8);
            fa2::mma16816(o[j], a, pb[0], pb[1]);
            fa2::mma16816(o[j], a, pb[2], pb[3]);
        }
        __syncwarp();
    }
    // Partials: query 2 tig + c, dims 16 j + g (+ 8).
    for (int c = 0; c < 2; c++) {
        const unsigned q = 2 * tig + c;
        if (q >= group) continue;
        float* destination = partial + ((unsigned long long)(kh * group + q) * splits + split) * 258;
        for (int j = 0; j < 16; j++) { destination[16 * j + g] = o[j][c]; destination[16 * j + g + 8] = o[j][2 + c]; }
        if (g == 0) { destination[256] = m[c]; destination[257] = l[c]; }
    }
}
}  // namespace gqa_decode

extern "C" __global__ __launch_bounds__(32) void euhedral_attention_decode_gqa_nvfp4(
        const __nv_bfloat16* queryKey, const __nv_bfloat16* gateValue,
        const unsigned char* const* keyPages, const unsigned char* const* valuePages,
        __nv_bfloat16* output, unsigned int rows, unsigned int queryHeads, unsigned int keyHeads,
        unsigned int headDim, unsigned int cacheLength, unsigned long long start,
        float* partial, unsigned int splits) {
    euhedral_pdl_begin();
    gqa_decode::decode_block(blockIdx.x, queryKey, keyPages, valuePages, queryHeads, keyHeads, cacheLength, partial, splits);
}

// Row-exact multi-row twin for speculative verification: blockIdx.y is the verified row, which attends
// the keys up to start + row exactly as the one-row launch above at that position would (same split
// count and block partition); rows below `from` keys are left to the per-head decode twin. Each row
// writes its own partials (rowStride floats apart). Grid: keyHeads * (largest split count) x rows.
extern "C" __global__ __launch_bounds__(32) void euhedral_attention_decode_gqa_nvfp4_rows(
        const __nv_bfloat16* queryKey, const unsigned char* const* keyPages, const unsigned char* const* valuePages,
        unsigned int queryHeads, unsigned int keyHeads, unsigned long long start, float* partial,
        unsigned long long rowStride, unsigned int from) {
    const unsigned row = blockIdx.y;
    const unsigned length = (unsigned)start + row + 1u;
    if (length < from) return;
    unsigned splits = (length + 31u) / 32u;
    if (splits > 64u) splits = 64u;
    if (blockIdx.x >= keyHeads * splits) return;
    euhedral_pdl_begin();
    gqa_decode::decode_block(blockIdx.x, queryKey + (unsigned long long)row * (queryHeads + keyHeads) * 256u, keyPages,
            valuePages, queryHeads, keyHeads, length, partial + row * rowStride, splits);
}
