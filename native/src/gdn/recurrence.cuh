#pragma once
// GDN delta-rule recurrence: the exact warp-reduced leaf and the relaxed column-owned leaves.
#include "reductions.cuh"
/// Columns of one value head owned by a warp. Its lanes hold four state elements per column, so the
/// warp reuses one query/key load, one normalization and one alpha/beta pair across all columns.
#define QWEN_GDN_WARP_COLUMNS 8

/// One warp owns `QWEN_GDN_WARP_COLUMNS` contiguous value columns of one value head. Recurrent state
/// stays in registers across every row and is loaded and stored once. Grid: valueHeads * 16 warps,
/// launched as 32-thread CTAs. Reductions use `qwen_gdn_warp_sum_128`, so no barrier is needed.
extern "C" __global__ __launch_bounds__(32) void euhedral_gdn_recurrence_bf16(
        const __nv_bfloat16* convolved, const float* alpha, const float* beta,
        float* recurrentState, __nv_bfloat16* output, uint32_t rows,
        uint32_t keyHeads, uint32_t valueHeads, uint32_t keyHeadDim,
        uint32_t valueHeadDim, float outputScale) {
    euhedral_pdl_begin();
    constexpr uint32_t columns = QWEN_GDN_WARP_COLUMNS;
    if (blockDim.x != 32 || keyHeadDim != 128 || valueHeadDim != 128) return;
    const uint32_t tilesPerHead = valueHeadDim / columns;
    const uint32_t valueHead = blockIdx.x / tilesPerHead;
    const uint32_t column0 = (blockIdx.x % tilesPerHead) * columns;
    if (valueHead >= valueHeads) return;
    const uint32_t lane = threadIdx.x;
    const uint32_t queryKeyWidth = 2 * keyHeads * keyHeadDim;
    const uint32_t convolvedWidth = queryKeyWidth + valueHeads * valueHeadDim;
    const uint32_t keyHead = valueHead / (valueHeads / keyHeads);
    float state[columns][4];
#pragma unroll
    for (uint32_t c = 0; c < columns; c++) {
        const uint64_t base = (static_cast<uint64_t>(valueHead) * valueHeadDim + column0 + c) * keyHeadDim;
#pragma unroll
        for (uint32_t j = 0; j < 4; j++) state[c][j] = recurrentState[base + lane + 32 * j];
    }
    for (uint32_t row = 0; row < rows; row++) {
        const uint64_t rowOffset = static_cast<uint64_t>(row) * convolvedWidth;
        const uint64_t queryBase = rowOffset + keyHead * keyHeadDim + lane;
        const uint64_t keyBase = rowOffset + keyHeads * keyHeadDim + keyHead * keyHeadDim + lane;
        float query[4], key[4];
#pragma unroll
        for (uint32_t j = 0; j < 4; j++) {
            query[j] = __bfloat162float(convolved[queryBase + 32 * j]);
            key[j] = __bfloat162float(convolved[keyBase + 32 * j]);
        }
        const float keyInverse = rsqrtf(qwen_gdn_warp_sum_128(__fmul_rn(key[0], key[0]), __fmul_rn(key[1], key[1]),
                __fmul_rn(key[2], key[2]), __fmul_rn(key[3], key[3])) + 1.0e-6f);
        const float queryInverse = rsqrtf(qwen_gdn_warp_sum_128(__fmul_rn(query[0], query[0]),
                __fmul_rn(query[1], query[1]), __fmul_rn(query[2], query[2]), __fmul_rn(query[3], query[3])) + 1.0e-6f);
        float normalizedKey[4], normalizedQuery[4];
#pragma unroll
        for (uint32_t j = 0; j < 4; j++) {
            normalizedKey[j] = key[j] * keyInverse;
            normalizedQuery[j] = query[j] * queryInverse;
        }
        const float rowAlpha = alpha[static_cast<uint64_t>(row) * valueHeads + valueHead];
        const float rowBeta = beta[static_cast<uint64_t>(row) * valueHeads + valueHead];
        float stateKeyDot[columns], value[columns];
#pragma unroll
        for (uint32_t c = 0; c < columns; c++) {
            value[c] = __bfloat162float(convolved[rowOffset + queryKeyWidth + valueHead * valueHeadDim + column0 + c]);
            stateKeyDot[c] = qwen_gdn_warp_sum_128(__fmul_rn(state[c][0], normalizedKey[0]),
                    __fmul_rn(state[c][1], normalizedKey[1]), __fmul_rn(state[c][2], normalizedKey[2]),
                    __fmul_rn(state[c][3], normalizedKey[3]));
        }
        float outputSum[columns];
#pragma unroll
        for (uint32_t c = 0; c < columns; c++) {
            // Explicit FMA: the compiler already fuses this product-subtract; pinning it keeps the rounding independent of the driver.
            const float delta = rowBeta * __fmaf_rn(-rowAlpha, stateKeyDot[c], value[c]);
#pragma unroll
            for (uint32_t j = 0; j < 4; j++) state[c][j] = rowAlpha * state[c][j] + delta * normalizedKey[j];
            outputSum[c] = qwen_gdn_warp_sum_128(__fmul_rn(state[c][0], normalizedQuery[0]),
                    __fmul_rn(state[c][1], normalizedQuery[1]), __fmul_rn(state[c][2], normalizedQuery[2]),
                    __fmul_rn(state[c][3], normalizedQuery[3]));
        }
        // Lane c publishes column c; the select chain keeps the array in registers.
        float mine = outputSum[0];
#pragma unroll
        for (uint32_t c = 1; c < columns; c++) mine = lane == c ? outputSum[c] : mine;
        if (lane < columns) {
            output[static_cast<uint64_t>(row) * valueHeads * valueHeadDim + valueHead * valueHeadDim + column0 + lane]
                    = __float2bfloat16_rn(mine * outputScale);
        }
    }
#pragma unroll
    for (uint32_t c = 0; c < columns; c++) {
        const uint64_t base = (static_cast<uint64_t>(valueHead) * valueHeadDim + column0 + c) * keyHeadDim;
#pragma unroll
        for (uint32_t j = 0; j < 4; j++) recurrentState[base + lane + 32 * j] = state[c][j];
    }
}

// Relaxed recurrence with column-owned lanes: lane l owns value column (l % C) of the warp's C columns
// and the l / C-th of P = 32 / C contiguous key slices (128 / P keys) of that column's state row. Every
// key reduction is local FMAs over the slice plus log2(P) shuffles across the column's slice owners,
// instead of a 128-key tree across the whole warp per column. The key and query normalizations are
// applied to the reduced dot products (inverse * sum(state * key)) rather than to every element. Each
// row's inputs are loaded during the previous row, and each local reduction keeps four partial sums,
// so the latency-bound warps (about 11 per SM) overlap their loads and FMA chains.
// Grid: valueHeads * (128 / C) warps as 32-thread CTAs. Requires 128-dimension heads and 16-byte
// aligned rows. euhedral_gdn_recurrence_bf16 is the exact twin.
template<int C>
static __device__ __forceinline__ void gdn_recurrence_columns(
        const __nv_bfloat16* convolved, const float* alpha, const float* beta, float* recurrentState,
        __nv_bfloat16* output, uint32_t rows, uint32_t keyHeads, uint32_t valueHeads, float outputScale) {
    constexpr uint32_t kSlices = 32 / C, kKeys = 128 / kSlices, kVec = kKeys / 8;
    const uint32_t lane = threadIdx.x, column = lane % C, slice = lane / C;
    const uint32_t tilesPerHead = 128 / C;
    const uint32_t valueHead = blockIdx.x / tilesPerHead;
    if (valueHead >= valueHeads) return;
    const uint32_t valueColumn = (blockIdx.x % tilesPerHead) * C + column;
    const uint32_t queryKeyWidth = 2 * keyHeads * 128;
    const uint32_t convolvedWidth = queryKeyWidth + valueHeads * 128;
    const uint32_t keyHead = valueHead / (valueHeads / keyHeads);
    const uint64_t stateBase = (static_cast<uint64_t>(valueHead) * 128 + valueColumn) * 128 + slice * kKeys;
    float state[kKeys];
#pragma unroll
    for (uint32_t j = 0; j < kKeys; j += 4) {
        const float4 v = *reinterpret_cast<const float4*>(recurrentState + stateBase + j);
        state[j] = v.x; state[j + 1] = v.y; state[j + 2] = v.z; state[j + 3] = v.w;
    }
    uint4 nq[kVec], nk[kVec];
    __nv_bfloat16 nv = __float2bfloat16(0.0f);
    float na = 0.0f, nb = 0.0f;
    auto fetch = [&](uint32_t row) {
        const uint64_t rowOffset = static_cast<uint64_t>(row) * convolvedWidth;
        const uint4* q = reinterpret_cast<const uint4*>(convolved + rowOffset + keyHead * 128 + slice * kKeys);
        const uint4* k = reinterpret_cast<const uint4*>(convolved + rowOffset + keyHeads * 128 + keyHead * 128 + slice * kKeys);
#pragma unroll
        for (uint32_t i = 0; i < kVec; i++) { nq[i] = q[i]; nk[i] = k[i]; }
        nv = convolved[rowOffset + queryKeyWidth + valueHead * 128 + valueColumn];
        na = alpha[static_cast<uint64_t>(row) * valueHeads + valueHead];
        nb = beta[static_cast<uint64_t>(row) * valueHeads + valueHead];
    };
    if (rows > 0) fetch(0);
    for (uint32_t row = 0; row < rows; row++) {
        float query[kKeys], key[kKeys];
#pragma unroll
        for (uint32_t i = 0; i < kVec; i++) {
            const unsigned int qw[4] = {nq[i].x, nq[i].y, nq[i].z, nq[i].w}, kw[4] = {nk[i].x, nk[i].y, nk[i].z, nk[i].w};
#pragma unroll
            for (uint32_t h = 0; h < 4; h++) {
                query[8 * i + 2 * h] = __uint_as_float(qw[h] << 16);
                query[8 * i + 2 * h + 1] = __uint_as_float(qw[h] & 0xffff0000u);
                key[8 * i + 2 * h] = __uint_as_float(kw[h] << 16);
                key[8 * i + 2 * h + 1] = __uint_as_float(kw[h] & 0xffff0000u);
            }
        }
        const float value = __bfloat162float(nv), rowAlpha = na, rowBeta = nb;
        if (row + 1 < rows) fetch(row + 1);
        float ks[4] = {}, qs[4] = {}, sk[4] = {};
#pragma unroll
        for (uint32_t j = 0; j < kKeys; j++) {
            ks[j & 3] = fmaf(key[j], key[j], ks[j & 3]);
            qs[j & 3] = fmaf(query[j], query[j], qs[j & 3]);
            sk[j & 3] = fmaf(state[j], key[j], sk[j & 3]);
        }
        float keySquares = (ks[0] + ks[1]) + (ks[2] + ks[3]);
        float querySquares = (qs[0] + qs[1]) + (qs[2] + qs[3]);
        float stateKey = (sk[0] + sk[1]) + (sk[2] + sk[3]);
#pragma unroll
        for (uint32_t distance = C; distance < 32; distance <<= 1) {
            keySquares += __shfl_xor_sync(0xffffffffu, keySquares, distance);
            querySquares += __shfl_xor_sync(0xffffffffu, querySquares, distance);
            stateKey += __shfl_xor_sync(0xffffffffu, stateKey, distance);
        }
        const float keyInverse = rsqrtf(keySquares + 1.0e-6f);
        const float queryInverse = rsqrtf(querySquares + 1.0e-6f);
        const float delta = rowBeta * fmaf(-rowAlpha, stateKey * keyInverse, value);
        const float update = delta * keyInverse;
        float sq[4] = {};
#pragma unroll
        for (uint32_t j = 0; j < kKeys; j++) {
            state[j] = fmaf(rowAlpha, state[j], update * key[j]);
            sq[j & 3] = fmaf(state[j], query[j], sq[j & 3]);
        }
        float stateQuery = (sq[0] + sq[1]) + (sq[2] + sq[3]);
#pragma unroll
        for (uint32_t distance = C; distance < 32; distance <<= 1)
            stateQuery += __shfl_xor_sync(0xffffffffu, stateQuery, distance);
        if (slice == 0)
            output[static_cast<uint64_t>(row) * valueHeads * 128 + valueHead * 128 + valueColumn]
                    = __float2bfloat16_rn(stateQuery * queryInverse * outputScale);
    }
#pragma unroll
    for (uint32_t j = 0; j < kKeys; j += 4)
        *reinterpret_cast<float4*>(recurrentState + stateBase + j) = make_float4(state[j], state[j + 1], state[j + 2], state[j + 3]);
}
#define QWEN_GDN_RECURRENCE_COLUMNS(C) \
extern "C" __global__ __launch_bounds__(32) void euhedral_gdn_recurrence_c##C##_bf16( \
        const __nv_bfloat16* convolved, const float* alpha, const float* beta, float* recurrentState, \
        __nv_bfloat16* output, uint32_t rows, uint32_t keyHeads, uint32_t valueHeads, uint32_t keyHeadDim, \
        uint32_t valueHeadDim, float outputScale) { \
    euhedral_pdl_begin(); \
    if (blockDim.x != 32 || keyHeadDim != 128 || valueHeadDim != 128) return; \
    gdn_recurrence_columns<C>(convolved, alpha, beta, recurrentState, output, rows, keyHeads, valueHeads, outputScale); \
}
QWEN_GDN_RECURRENCE_COLUMNS(8)
QWEN_GDN_RECURRENCE_COLUMNS(4)
#undef QWEN_GDN_RECURRENCE_COLUMNS
