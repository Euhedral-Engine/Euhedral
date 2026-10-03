#pragma once
#include "nvfp4_prefill_fa2.cuh"
#include "nvfp4_pipe.cuh"
// GQA tensor-core decode (relaxed, long contexts): one CTA per (KV head, key split) serves the KV head's
// whole group of G <= 8 query heads, so each 16-key tile of the cache is expanded once instead of once per
// query head. The query heads are the N = 8 columns of the mma tiles: scores S^T = K Q^T (K rows as A) and
// the output O^T = V^T P^T (V by transposed ldmatrix as A, P^T from shared memory as B), so the
// 256-dimension output takes 16 x 4 accumulator registers and both the queries and the probabilities enter
// as hi + lo FP16 parts (mma.sync m16n8k16, FP32 accumulation), with the accurate expf of the FP32 kernel:
// within about 1e-7 to 8e-5 relative rms of it. Writes per-(query head, split) partials in the format of
// euhedral_attention_merge_nvfp4.
//
// The CTA is three warps, so the 4 x 64 CTAs of a long context keep every SM scheduler busy instead of waiting on the
// dependency chain of a single warp each:
//   warp 0  scores: K tile by cp.async ring, A fragments straight from the raw codes, the QK mma chain;
//   warp 1  values: V tile by register prefetch, expanded to FP16 rows in shared memory;
//   warp 2  online softmax, P hi/lo, the PV mma chain, the partial write.
// Every value is computed by the same operations in the same order as in the single-warp kernel this
// replaced (reference_decode_gqa.cuh, the test control), so the partials are bit for bit identical: tiles
// reach the consumer in order, the mma chains are unchanged, and the exact-product lookup below equals the
// software dequantization for every legal cache scale. Tiles hand off through shared-memory mbarriers;
// named barriers would cap the SM at one resident CTA (nine ids used, sixteen per SM).
namespace gqa_decode {
using namespace nvfp4pipe;
constexpr int D = 256, KT = 16, STRIDE = D + 8, ROW = 144, TILE_BYTES = KT * ROW, kRing = 2;
// mbarrier slots: scores full [0, 2), scores empty [2, 4), V full [4, 6), V empty [6, 8); 32 arrivals each.
enum { kScoresFull = 0, kScoresEmpty = 2, kValuesFull = 4, kValuesEmpty = 6 };
// Starts the copy of the raw 144-byte rows of the K tile at `base` (nine 16-byte chunks per token).
static __device__ __forceinline__ void issue_keys(unsigned char* raw, const unsigned char* const* keyPages, unsigned base,
        unsigned end, unsigned head, unsigned heads, unsigned lane) {
#pragma unroll
    for (int j = 0; j < 5; j++) {
        const unsigned chunk = lane + 32u * j;
        if (chunk < KT * 9u) {
            const unsigned t = chunk / 9u, offset = chunk % 9u;
            if (base + t < end) cp_async16(raw + t * ROW + offset * 16u, nvfp4kv::cache_row(keyPages, base + t, head, heads) + offset * 16u);
        }
    }
}
// Loads this lane's eight (eight code bytes, scale byte) groups of the V tile at `base`; rows past `end` read as zero.
static __device__ __forceinline__ void load_values(uint2 (&codes)[8], unsigned (&scales)[8], const unsigned char* const* valuePages,
        unsigned base, unsigned end, unsigned head, unsigned heads, unsigned lane) {
#pragma unroll
    for (unsigned it = 0; it < 8; it++) {
        const unsigned token = base + 2 * it + (lane >> 4);
        const bool valid = token < end;
        const unsigned char* row = nvfp4kv::cache_row(valuePages, valid ? token : base, head, heads);
        const uint2 loaded = *reinterpret_cast<const uint2*>(row + (lane & 15u) * 8);
        codes[it] = valid ? loaded : make_uint2(0, 0);
        scales[it] = valid ? row[nvfp4kv::kCodeBytes + (lane & 15u)] : 0u;
    }
}
}  // namespace gqa_decode
namespace gqa_decode {
// One (KV head, split) block of the decode attention below; `block` is its index in the one-row grid. 96 threads.
static __device__ __forceinline__ void decode_block(unsigned block,
        const __nv_bfloat16* queryKey, const unsigned char* const* keyPages, const unsigned char* const* valuePages,
        unsigned int queryHeads, unsigned int keyHeads, unsigned int cacheLength, float* partial, unsigned int splits) {
    const unsigned warp = threadIdx.x >> 5, lane = threadIdx.x & 31u, g = lane >> 2, tig = lane & 3u;
    const unsigned group = queryHeads / keyHeads;
    const unsigned kh = block / splits, split = block % splits;
    const unsigned span = (cacheLength + splits - 1) / splits;
    const unsigned begin = split * span, end = min(cacheLength, begin + span);
    const unsigned tiles = begin < end ? (end - begin + KT - 1) / KT : 0;
    __shared__ __align__(16) __half vbuf[2][KT * STRIDE];            // expanded V tiles (warp 0 stages the queries here first)
    __shared__ __align__(16) unsigned char kraw[kRing][TILE_BYTES];  // raw K tiles
    __shared__ __align__(16) unsigned pairs[256];                    // byte of two E2M1 codes -> FP16 pair
    __shared__ __align__(16) float4 scores[2][32];                   // one lane's four scores per tile
    __shared__ __align__(16) __half pbuf[2][8 * 24];                 // P^T source: [query][key], hi and lo, padded rows
    __shared__ unsigned long long mbarriers[8];
    for (unsigned i = threadIdx.x; i < 256; i += 96) pairs[i] = e2m1_pair_bits(i);
    if (threadIdx.x == 0)
        for (int i = 0; i < 8; i++) mbarrier_init(&mbarriers[i], 32);
    // Rotated queries (rows >= group zero) as hi + lo FP16 B fragments: qb[k][part][2]. Warp 0 only.
    unsigned qh[16][2], ql[16][2];
    uint2 valueCodes[8];
    unsigned valueScales[8];
    if (warp == 0) {
#pragma unroll
        for (int s = 0; s < kRing - 1; s++) {
            if ((unsigned)s < tiles) issue_keys(kraw[s], keyPages, begin + s * KT, end, kh, keyHeads, lane);
            cp_async_commit();
        }
        __half* qhi = vbuf[0];  // 8 rows x STRIDE
        __half* qlo = vbuf[0] + 8 * STRIDE;
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
    } else if (warp == 1 && tiles > 0) {
        load_values(valueCodes, valueScales, valuePages, begin, end, kh, keyHeads, lane);
    }
    __syncthreads();
    if (warp == 0) {
        for (unsigned tile = 0; tile < tiles; tile++) {
            const unsigned base = begin + tile * KT, slot = tile & 1u;
            if (tile + kRing - 1 < tiles)
                issue_keys(kraw[(tile + kRing - 1) % kRing], keyPages, base + (kRing - 1) * KT, end, kh, keyHeads, lane);
            cp_async_commit();
            cp_async_wait<kRing - 1>();
            __syncwarp();
            // A = K tile (rows = keys g and g + 8; columns 2 tig, 2 tig + 1 and 8 + 2 tig, 9 + 2 tig of dims 16 k ..): the
            // codes are bytes tig and tig + 4 of the k-th group's eight code bytes. Rows past `end` hold stale bytes;
            // their scores are masked below, and an mma row never mixes with another.
            const unsigned char* row0 = kraw[tile % kRing] + g * ROW;
            const unsigned char* row1 = row0 + 8 * ROW;
            const uint4 scaleBytes0 = *reinterpret_cast<const uint4*>(row0 + 128);
            const uint4 scaleBytes1 = *reinterpret_cast<const uint4*>(row1 + 128);
            const unsigned scaleWords0[4] = {scaleBytes0.x, scaleBytes0.y, scaleBytes0.z, scaleBytes0.w};
            const unsigned scaleWords1[4] = {scaleBytes1.x, scaleBytes1.y, scaleBytes1.z, scaleBytes1.w};
            float s[4] = {0.0f, 0.0f, 0.0f, 0.0f};  // (key g, q 2tig/+1), (key g + 8, q 2tig/+1)
#pragma unroll
            for (int k = 0; k < 16; k++) {
                const uint2 codes0 = *reinterpret_cast<const uint2*>(row0 + k * 8);
                const uint2 codes1 = *reinterpret_cast<const uint2*>(row1 + k * 8);
                const unsigned scale0 = scale_pair((scaleWords0[k >> 2] >> (8 * (k & 3))) & 0xFFu);
                const unsigned scale1 = scale_pair((scaleWords1[k >> 2] >> (8 * (k & 3))) & 0xFFu);
                unsigned a[4];
                a[0] = half2_multiply(pairs[(codes0.x >> (8 * tig)) & 0xFFu], scale0);
                a[1] = half2_multiply(pairs[(codes1.x >> (8 * tig)) & 0xFFu], scale1);
                a[2] = half2_multiply(pairs[(codes0.y >> (8 * tig)) & 0xFFu], scale0);
                a[3] = half2_multiply(pairs[(codes1.y >> (8 * tig)) & 0xFFu], scale1);
                fa2::mma16816(s, a, qh[k][0], qh[k][1]);
                fa2::mma16816(s, a, ql[k][0], ql[k][1]);
            }
            if (tile >= 2) mbarrier_wait(&mbarriers[kScoresEmpty + slot], ((tile >> 1) - 1) & 1u);
            scores[slot][lane] = make_float4(s[0], s[1], s[2], s[3]);
            mbarrier_arrive(&mbarriers[kScoresFull + slot]);
            __syncwarp();
        }
    } else if (warp == 1) {
        for (unsigned tile = 0; tile < tiles; tile++) {
            const unsigned base = begin + tile * KT, slot = tile & 1u;
            uint2 nextCodes[8];
            unsigned nextScales[8];
            if (tile + 1 < tiles) load_values(nextCodes, nextScales, valuePages, base + KT, end, kh, keyHeads, lane);
            if (tile >= 2) mbarrier_wait(&mbarriers[kValuesEmpty + slot], ((tile >> 1) - 1) & 1u);
#pragma unroll
            for (unsigned it = 0; it < 8; it++) {
                const unsigned scale = scale_pair(valueScales[it]);
                const unsigned words[2] = {valueCodes[it].x, valueCodes[it].y};
                unsigned out[8];
#pragma unroll
                for (int h = 0; h < 2; h++)
#pragma unroll
                    for (int b = 0; b < 4; b++) out[h * 4 + b] = half2_multiply(pairs[(words[h] >> (8 * b)) & 0xFFu], scale);
                uint4* dst = reinterpret_cast<uint4*>(vbuf[slot] + (2 * it + (lane >> 4)) * STRIDE + (lane & 15u) * 16);
                dst[0] = make_uint4(out[0], out[1], out[2], out[3]);
                dst[1] = make_uint4(out[4], out[5], out[6], out[7]);
            }
            mbarrier_arrive(&mbarriers[kValuesFull + slot]);
            if (tile + 1 < tiles) {
#pragma unroll
                for (unsigned it = 0; it < 8; it++) { valueCodes[it] = nextCodes[it]; valueScales[it] = nextScales[it]; }
            }
        }
    } else {
        float o[16][4];
        for (int j = 0; j < 16; j++) { o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.0f; }
        // This lane's queries are columns 2 tig and 2 tig + 1.
        float m[2] = {-__int_as_float(0x7f800000), -__int_as_float(0x7f800000)}, l[2] = {0.0f, 0.0f};
        for (unsigned tile = 0; tile < tiles; tile++) {
            const unsigned base = begin + tile * KT, slot = tile & 1u;
            mbarrier_wait(&mbarriers[kScoresFull + slot], (tile >> 1) & 1u);
            const float4 tileScores = scores[slot][lane];
            if (tile + 2 < tiles) mbarrier_arrive(&mbarriers[kScoresEmpty + slot]);
            float s[4] = {tileScores.x, tileScores.y, tileScores.z, tileScores.w};
            float p[4];
            for (int c = 0; c < 2; c++) {
                const bool v0 = base + g < end, v1 = base + g + 8 < end;
                s[c] = v0 ? s[c] * 0.0625f : -__int_as_float(0x7f800000);
                s[2 + c] = v1 ? s[2 + c] * 0.0625f : -__int_as_float(0x7f800000);
                float x = fmaxf(fmaxf(s[c], s[2 + c]), m[c]);
                x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, 4));
                x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, 8));
                x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, 16));
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
            mbarrier_wait(&mbarriers[kValuesFull + slot], (tile >> 1) & 1u);
#pragma unroll
            for (int j = 0; j < 16; j++) {
                unsigned a[4];
                // A = V^T tile: dims 16 j .. +15 (rows), keys 0..15 (cols) from V [key][dim] via transpose
                fa2::ldsm_x4_t(a, vbuf[slot] + ((lane & 7u) + ((lane >> 4) << 3)) * STRIDE + j * 16 + ((lane >> 3) & 1u) * 8);
                fa2::mma16816(o[j], a, pb[0], pb[1]);
                fa2::mma16816(o[j], a, pb[2], pb[3]);
            }
            if (tile + 2 < tiles) mbarrier_arrive(&mbarriers[kValuesEmpty + slot]);
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
}
}  // namespace gqa_decode

extern "C" __global__ __launch_bounds__(96) void euhedral_attention_decode_gqa_nvfp4(
        const __nv_bfloat16* queryKey, const __nv_bfloat16* gateValue,
        const unsigned char* const* keyPages, const unsigned char* const* valuePages,
        __nv_bfloat16* output, unsigned int rows, unsigned int queryHeads, unsigned int keyHeads,
        unsigned int headDim, const unsigned long long* position, unsigned long long positionOffset,
        float* partial, unsigned int splits) {
    euhedral_pdl_begin();
    const unsigned int cacheLength = (unsigned int)(*position + positionOffset) + 1u;
    gqa_decode::decode_block(blockIdx.x, queryKey, keyPages, valuePages, queryHeads, keyHeads, cacheLength, partial, splits);
}

// Row-exact multi-row twin for speculative verification: blockIdx.y is the verified row, which attends
// the keys up to start + row exactly as the one-row launch above at that position would (same split
// count and block partition); rows below `from` keys are left to the per-head decode twin. Each row
// writes its own partials (rowStride floats apart). Grid: keyHeads * (largest split count) x rows.
extern "C" __global__ __launch_bounds__(96) void euhedral_attention_decode_gqa_nvfp4_rows(
        const __nv_bfloat16* queryKey, const unsigned char* const* keyPages, const unsigned char* const* valuePages,
        unsigned int queryHeads, unsigned int keyHeads, const unsigned long long* position, float* partial,
        unsigned long long rowStride, unsigned int from) {
    const unsigned row = blockIdx.y;
    const unsigned length = (unsigned)*position + row + 1u;
    if (length < from) return;
    unsigned splits = (length + 31u) / 32u;
    if (splits > 64u) splits = 64u;
    if (blockIdx.x >= keyHeads * splits) return;
    euhedral_pdl_begin();
    gqa_decode::decode_block(blockIdx.x, queryKey + (unsigned long long)row * (queryHeads + keyHeads) * 256u, keyPages,
            valuePages, queryHeads, keyHeads, length, partial + row * rowStride, splits);
}
