#pragma once
#include "nvfp4_decode_gqa.cuh"
// Fused verifier rows: one CTA per (KV head, key split) serves every verified row's query-head group at once, so each
// 16-key tile of the cache is loaded and expanded once for all rows instead of once per row. Column c of the NT mma
// tiles (8 columns each) is row first + c / G, query head c % G of the KV head's group. Each column is computed by the
// same operations in the same order as the one-row GQA kernel (gqa_decode::decode_block) at that row's length, so the
// partials are bit for bit those of the row twin: the rows served here share the one-row split partition (split count
// and span), so their tiles start at the same keys, and a column's keys past its row's end are masked; a tile wholly
// past a row's end leaves that row's state unchanged (scale exp(0) = 1, zero probabilities against finite values).
//
// Warps: NT score warps (one mma tile of queries each, the QK chain over the shared raw K tile), one loader warp (the
// raw K and V tiles by a cp.async ring of Stages<Kv>::kCount tiles, V expanded once to FP16 rows), NT consumer warps
// (online softmax, P hi/lo and the PV chain of one query tile over the shared V tile). Hand-offs through mbarriers.
namespace gqa_fused {
using namespace gqa_decode;
// Raw tiles in flight per CTA: enough K and V bytes to cover DRAM latency at two resident CTAs per SM. NVFP4 caches
// only: the BF16 cache keeps the row twin.
template <class Kv> struct Stages;
template <> struct Stages<Nvfp4Kv> { static constexpr int kCount = 4; };
// The FP16 rows of a raw V tile in shared memory (this lane's 16 values of token 2 it + (lane >> 4)); rows at or past
// `end` expand to zero, as the one-row kernel's register loads read them.
template <class Kv> struct RawValues;
template <> struct RawValues<Nvfp4Kv> {
    static __device__ __forceinline__ void expand(__half* vbuf, const unsigned char* raw, unsigned base, unsigned end,
            const unsigned* pairs, unsigned lane) {
#pragma unroll
        for (unsigned it = 0; it < 8; it++) {
            const unsigned token = 2 * it + (lane >> 4);
            const bool valid = base + token < end;
            const unsigned char* row = raw + token * ROW;
            const uint2 loaded = *reinterpret_cast<const uint2*>(row + (lane & 15u) * 8);
            const unsigned words[2] = {valid ? loaded.x : 0u, valid ? loaded.y : 0u};
            const unsigned scale = scale_pair(valid ? row[nvfp4kv::kCodeBytes + (lane & 15u)] : 0u);
            unsigned out[8];
#pragma unroll
            for (int h = 0; h < 2; h++)
#pragma unroll
                for (int b = 0; b < 4; b++) out[h * 4 + b] = half2_multiply(pairs[(words[h] >> (8 * b)) & 0xFFu], scale);
            uint4* dst = reinterpret_cast<uint4*>(vbuf + token * STRIDE + (lane & 15u) * 16);
            dst[0] = make_uint4(out[0], out[1], out[2], out[3]);
            dst[1] = make_uint4(out[4], out[5], out[6], out[7]);
        }
    }
};
// mbarriers (S = Stages<Kv>::kCount): K full [0, S), K empty [S, 2 S), V full 2 S + slot, V empty 2 S + 2 + slot, then
// per query tile n: scores full 2 S + 4 + 4 n + slot, scores empty 2 S + 6 + 4 n + slot.
template <int NT, class Kv> struct Layout {
    static constexpr int S = Stages<Kv>::kCount, R = 2;
    static constexpr unsigned kKeysFull = 0, kKeysEmpty = S, kValuesFull = 2 * S, kValuesEmpty = 2 * S + R,
            kScoreBarriers = 2 * S + 2 * R, kBarrierCount = 2 * S + 2 * R + 2 * R * NT;
    static constexpr unsigned kValues = 0;                                    // V tiles: R x KT x STRIDE halves
    static constexpr unsigned kScores = kValues + R * KT * STRIDE * 2;        // NT x R x 32 float4
    static constexpr unsigned kProbabilities = kScores + NT * R * 32 * 16;    // NT x (hi, lo) x 8 x 24 halves
    static constexpr unsigned kKeys = kProbabilities + NT * 2 * 8 * 24 * 2;   // S raw K tiles
    static constexpr unsigned kRawValues = kKeys + S * KT * Format<Kv>::kRow; // S raw V tiles
    static constexpr unsigned kPairs = kRawValues + S * KT * Format<Kv>::kRow;
    static constexpr unsigned kBarriers = kPairs + 256 * 4;
    static constexpr unsigned kBytes = kBarriers + 8 * kBarrierCount;
    // Query staging (prologue only): 8 x STRIDE halves per score warp, over the V tiles, scores and probabilities.
    static constexpr unsigned kStaging = 8 * STRIDE * 2;
    static_assert(NT * kStaging <= kKeys, "query staging overlaps the K ring");
};
template <int NT, class Kv>
static __device__ __forceinline__ void fused_block(unsigned char* shared, const __nv_bfloat16* queryKey,
        const unsigned char* const* keyPages, const unsigned char* const* valuePages, unsigned queryHeads, unsigned keyHeads,
        unsigned length, unsigned rows, unsigned first, float* partial, unsigned long long rowStride, unsigned splits) {
    typedef Layout<NT, Kv> L;
    const unsigned warp = threadIdx.x >> 5, lane = threadIdx.x & 31u, g = lane >> 2, tig = lane & 3u;
    const unsigned group = queryHeads / keyHeads, columns = (rows - first) * group;
    const unsigned kh = blockIdx.x / splits, split = blockIdx.x % splits;
    const unsigned last = length + rows - 1u, span = split_span(last, splits);
    const unsigned begin = split * span, end = min(last, begin + span);
    const unsigned tiles = begin < end ? (end - begin + KT - 1) / KT : 0;
    const unsigned keyRow = Format<Kv>::kRow;
    __half* vbuf = reinterpret_cast<__half*>(shared + L::kValues);
    float4* scores = reinterpret_cast<float4*>(shared + L::kScores);
    __half* pbuf = reinterpret_cast<__half*>(shared + L::kProbabilities);
    unsigned char* kraw = shared + L::kKeys;
    unsigned char* vraw = shared + L::kRawValues;
    const int S = L::S;
    const unsigned R = L::R;
    unsigned* pairs = reinterpret_cast<unsigned*>(shared + L::kPairs);
    unsigned long long* mbarriers = reinterpret_cast<unsigned long long*>(shared + L::kBarriers);
    for (unsigned i = threadIdx.x; i < 256; i += blockDim.x) pairs[i] = code_pair_bits(i);
    if (threadIdx.x == 0)
        for (unsigned i = 0; i < L::kBarrierCount; i++)
            mbarrier_init(&mbarriers[i], (i >= L::kKeysEmpty && i < L::kKeysEmpty + S)
                    || (i >= L::kValuesEmpty && i < L::kValuesEmpty + L::R) ? 32u * NT : 32u);
    unsigned qh[16][2], ql[16][2];
    if (warp < (unsigned)NT) {
        // This warp's eight query columns, rotated, as hi + lo FP16 B fragments (columns past the rows read as zero),
        // staged through shared memory one part at a time (the rotation is recomputed for the lo part).
        __half* staging = reinterpret_cast<__half*>(shared) + warp * 8 * STRIDE;
#pragma unroll
        for (int part = 0; part < 2; part++) {
            for (unsigned r = 0; r < 8; r++) {
                const unsigned column = warp * 8 + r;
                const bool valid = column < columns;
                const unsigned row = first + (valid ? column / group : 0), head = valid ? column % group : 0;
                const __nv_bfloat16* query = queryKey + (unsigned long long)row * (queryHeads + keyHeads) * D + (kh * group + head) * D;
                float v[8];
#pragma unroll
                for (int d = 0; d < 8; d++) v[d] = valid ? __bfloat162float(query[lane + d * 32]) : 0.0f;
                if (valid) Kv::rotate(v, lane);
#pragma unroll
                for (int d = 0; d < 8; d++) {
                    const __half hi = __float2half_rn(v[d]);
                    staging[r * STRIDE + lane + d * 32] = part == 0 ? hi : __float2half_rn(v[d] - __half2float(hi));
                }
            }
            __syncwarp();
#pragma unroll
            for (int k = 0; k < 16; k += 2) {
                unsigned x[4];
                fa2::ldsm_x4(x, staging + (lane & 7u) * STRIDE + k * 16 + ((lane >> 3) & 1u) * 8 + (lane >> 4) * 16);
                unsigned (&q)[16][2] = part == 0 ? qh : ql;
                q[k][0] = x[0]; q[k][1] = x[1]; q[k + 1][0] = x[2]; q[k + 1][1] = x[3];
            }
            __syncwarp();
        }
    } else if (warp == (unsigned)NT) {
        for (int s = 0; s < S - 1; s++) {
            if ((unsigned)s < tiles) {
                Format<Kv>::issue(kraw + s * KT * keyRow, keyPages, begin + s * KT, end, kh, keyHeads, lane);
                Format<Kv>::issue(vraw + s * KT * keyRow, valuePages, begin + s * KT, end, kh, keyHeads, lane);
            }
            cp_async_commit();
        }
    }
    __syncthreads();
    if (warp < (unsigned)NT) {
        const unsigned n = warp;
        for (unsigned tile = 0; tile < tiles; tile++) {
            const unsigned ring = tile % S, slot = tile % R;
            mbarrier_wait(&mbarriers[L::kKeysFull + ring], (tile / S) & 1u);
            const unsigned char* row0 = kraw + ring * KT * keyRow + g * keyRow;
            const typename Format<Kv>::LightKeys keys(row0, row0 + 8 * keyRow);
            float s[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
            for (int k = 0; k < 16; k++) {
                unsigned a[4];
                keys.fragment(k, tig, pairs, a);
                fa2::mma16816(s, a, qh[k][0], qh[k][1]);
                fa2::mma16816(s, a, ql[k][0], ql[k][1]);
            }
            if (tile + S < tiles) mbarrier_arrive(&mbarriers[L::kKeysEmpty + ring]);
            if (tile >= R) mbarrier_wait(&mbarriers[L::kScoreBarriers + 2 * R * n + R + slot], ((tile / R) - 1) & 1u);
            scores[(n * R + slot) * 32 + lane] = make_float4(s[0], s[1], s[2], s[3]);
            mbarrier_arrive(&mbarriers[L::kScoreBarriers + 2 * R * n + slot]);
            __syncwarp();
        }
    } else if (warp == (unsigned)NT) {
        for (unsigned tile = 0; tile < tiles; tile++) {
            // The raw V slot `ahead` reuses was expanded by this warp one iteration ago; the K slot waits for the
            // score warps.
            const unsigned ahead = tile + S - 1;
            if (ahead < tiles) {
                if (ahead >= (unsigned)S) mbarrier_wait(&mbarriers[L::kKeysEmpty + ahead % S], ((ahead / S) - 1) & 1u);
                Format<Kv>::issue(kraw + (ahead % S) * KT * keyRow, keyPages, begin + ahead * KT, end, kh, keyHeads, lane);
                Format<Kv>::issue(vraw + (ahead % S) * KT * keyRow, valuePages, begin + ahead * KT, end, kh, keyHeads, lane);
            }
            cp_async_commit();
            cp_async_wait<S - 1>();
            __syncwarp();
            mbarrier_arrive(&mbarriers[L::kKeysFull + tile % S]);
            const unsigned slot = tile % R;
            if (tile >= R) mbarrier_wait(&mbarriers[L::kValuesEmpty + slot], ((tile / R) - 1) & 1u);
            RawValues<Kv>::expand(vbuf + slot * KT * STRIDE, vraw + (tile % S) * KT * keyRow, begin + tile * KT, end, pairs, lane);
            mbarrier_arrive(&mbarriers[L::kValuesFull + slot]);
            __syncwarp();
        }
    } else {
        const unsigned n = warp - NT - 1;
        __half* phi = pbuf + n * 2 * 8 * 24;
        __half* plo = phi + 8 * 24;
        // This lane's columns 8 n + 2 tig + c: their rows' key ranges in this split (empty for columns past the rows).
        unsigned columnEnd[2];
#pragma unroll
        for (int c = 0; c < 2; c++) {
            const unsigned column = n * 8 + 2 * tig + c;
            columnEnd[c] = column < columns ? min(length + first + column / group, begin + span) : begin;
        }
        float o[16][4];
        #pragma unroll
        for (int j = 0; j < 16; j++) { o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.0f; }
        float m[2] = {-__int_as_float(0x7f800000), -__int_as_float(0x7f800000)}, l[2] = {0.0f, 0.0f};
        for (unsigned tile = 0; tile < tiles; tile++) {
            const unsigned base = begin + tile * KT, slot = tile % R;
            mbarrier_wait(&mbarriers[L::kScoreBarriers + 2 * R * n + slot], (tile / R) & 1u);
            const float4 tileScores = scores[(n * R + slot) * 32 + lane];
            if (tile + R < tiles) mbarrier_arrive(&mbarriers[L::kScoreBarriers + 2 * R * n + R + slot]);
            float s[4] = {tileScores.x, tileScores.y, tileScores.z, tileScores.w};
            float p[4];
            #pragma unroll
            for (int c = 0; c < 2; c++) {
                const bool v0 = base + g < columnEnd[c], v1 = base + g + 8 < columnEnd[c];
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
                #pragma unroll
                for (int j = 0; j < 16; j++) { o[j][c] *= scale; o[j][2 + c] *= scale; }
            }
            #pragma unroll
            for (int c = 0; c < 2; c++) {
                const unsigned q = 2 * tig + c;
                const __half h0 = __float2half_rn(p[c]), h1 = __float2half_rn(p[2 + c]);
                phi[q * 24 + g] = h0; phi[q * 24 + g + 8] = h1;
                plo[q * 24 + g] = __float2half_rn(p[c] - __half2float(h0));
                plo[q * 24 + g + 8] = __float2half_rn(p[2 + c] - __half2float(h1));
            }
            __syncwarp();
            unsigned pb[4];
            fa2::ldsm_x4(pb, (lane >> 4 ? plo : phi) + (lane & 7u) * 24 + ((lane >> 3) & 1u) * 8);
            mbarrier_wait(&mbarriers[L::kValuesFull + slot], (tile / R) & 1u);
            const __half* v = vbuf + slot * KT * STRIDE;
#pragma unroll
            #pragma unroll
            for (int j = 0; j < 16; j++) {
                unsigned a[4];
                fa2::ldsm_x4_t(a, v + ((lane & 7u) + ((lane >> 4) << 3)) * STRIDE + j * 16 + ((lane >> 3) & 1u) * 8);
                fa2::mma16816(o[j], a, pb[0], pb[1]);
                fa2::mma16816(o[j], a, pb[2], pb[3]);
            }
            if (tile + R < tiles) mbarrier_arrive(&mbarriers[L::kValuesEmpty + slot]);
            __syncwarp();
        }
        #pragma unroll
        // The block's KV head and split again from blockIdx (cheaper than keeping them live across the tile loop).
        const unsigned headSplit = (blockIdx.x / splits) * group * splits + blockIdx.x % splits;
#pragma unroll
        for (int c = 0; c < 2; c++) {
            const unsigned column = n * 8 + 2 * tig + c;
            if (column >= columns) continue;
            float* destination = partial + (first + column / group) * rowStride
                    + ((unsigned long long)(column % group) * splits + headSplit) * 258;
            #pragma unroll
            for (int j = 0; j < 16; j++) { destination[16 * j + g] = o[j][c]; destination[16 * j + g + 8] = o[j][2 + c]; }
            if (g == 0) { destination[256] = m[c]; destination[257] = l[c]; }
        }
    }
}
}  // namespace gqa_fused

// Fused verifier rows (above): grid keyHeads * (the last row's split count), (2 NT + 1) * 32 threads,
// gqa_fused::Layout<NT, Kv>::kBytes of dynamic shared memory. Serves rows first_fused_row(...) on; the row twin below
// takes the rest (its `fused` argument set).
#define EUHEDRAL_ATTENTION_DECODE_GQA_FUSED(name, Kv, NT)                                                         \
    extern "C" __global__ __launch_bounds__((2 * NT + 1) * 32, NT <= 4 ? 2 : 1) void name(const __nv_bfloat16* queryKey,           \
            const unsigned char* const* keyPages, const unsigned char* const* valuePages, unsigned int queryHeads, \
            unsigned int keyHeads, const unsigned long long* position, float* partial,                             \
            unsigned long long rowStride, unsigned int rows, unsigned int from) {                                  \
        extern __shared__ __align__(16) unsigned char shared[];                                                    \
        const unsigned length = (unsigned)*position + 1u;                                                          \
        const unsigned first = gqa_decode::first_fused_row(length, rows, from);                                     \
        if (first >= rows || (rows - first) * (queryHeads / keyHeads) > 8u * NT) return;                           \
        const unsigned splits = gqa_decode::split_count(length + rows - 1u);                                        \
        if (blockIdx.x >= keyHeads * splits) return;                                                               \
        euhedral_pdl_begin();                                                                                      \
        gqa_fused::fused_block<NT, Kv>(shared, queryKey, keyPages, valuePages, queryHeads, keyHeads, length, rows, \
                first, partial, rowStride, splits);                                                                \
    }
EUHEDRAL_ATTENTION_DECODE_GQA_FUSED(euhedral_attention_decode_gqa_nvfp4_fused2, Nvfp4Kv, 2)
EUHEDRAL_ATTENTION_DECODE_GQA_FUSED(euhedral_attention_decode_gqa_nvfp4_fused3, Nvfp4Kv, 3)
EUHEDRAL_ATTENTION_DECODE_GQA_FUSED(euhedral_attention_decode_gqa_nvfp4_fused4, Nvfp4Kv, 4)
EUHEDRAL_ATTENTION_DECODE_GQA_FUSED(euhedral_attention_decode_gqa_nvfp4_fused5, Nvfp4Kv, 5)
EUHEDRAL_ATTENTION_DECODE_GQA_FUSED(euhedral_attention_decode_gqa_nvfp4_fused6, Nvfp4Kv, 6)
