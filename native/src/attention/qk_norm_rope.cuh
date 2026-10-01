#pragma once
// Per-head Q/K RMSNorm and RoPE: rotated queries and keys for attention and the cache append.
#include "reductions.cuh"
#include "common/pdl.cuh"

// RoPE rotation with explicit rounding (no FMA contraction), shared by both QK norm layouts so they
// agree bit for bit whatever code the compiler generates around them.
static __device__ __forceinline__ float rope_rotate(float value, float paired, float cosine, float sine, bool firstHalf) {
    const float direct = __fmul_rn(value, cosine), cross = __fmul_rn(paired, sine);
    return firstHalf ? __fsub_rn(direct, cross) : __fadd_rn(direct, cross);
}

static __device__ __forceinline__ void qk_norm_rope(
        const __nv_bfloat16* queryKey,
        const __nv_bfloat16* queryNorm,
        const __nv_bfloat16* keyNorm,
        __nv_bfloat16* output,
        uint32_t rows,
        uint32_t queryHeads,
        uint32_t keyValueHeads,
        uint32_t headDim,
        uint32_t rotaryDim,
        uint64_t startPosition,
        float epsilon,
        double ropeTheta) {
    const uint32_t rowHead = blockIdx.x;
    const uint32_t row = rowHead / (queryHeads + keyValueHeads);
    const uint32_t head = rowHead % (queryHeads + keyValueHeads);
    if (row >= rows || blockDim.x != headDim) return;

    const uint32_t lane = threadIdx.x;
    const uint32_t projectedWidth = (queryHeads + keyValueHeads) * headDim;
    const uint64_t inputOffset = static_cast<uint64_t>(row) * projectedWidth + static_cast<uint64_t>(head) * headDim;
    const uint64_t outputOffset = inputOffset;
    const bool isQuery = head < queryHeads;
    const __nv_bfloat16* norm = isQuery ? queryNorm : keyNorm;
    float value = __bfloat162float(queryKey[inputOffset + lane]);
    __shared__ float scratch[256];
    const float sum = attention_reduce_sum(value * value, scratch);
    const float inverse = rsqrtf(sum / static_cast<float>(headDim) + epsilon);
    value = value * inverse * (1.0f + __bfloat162float(norm[lane]));

    float result = value;
    if (lane < rotaryDim) {
        const uint32_t half = rotaryDim / 2;
        const uint32_t pair = lane < half ? lane + half : lane - half;
        const uint32_t frequencyIndex = lane < half ? lane : pair;
        const double exponent = (2.0 * static_cast<double>(frequencyIndex)) / static_cast<double>(rotaryDim);
        const double angle = static_cast<double>(startPosition + row) * pow(ropeTheta, -exponent);
        const float cosine = static_cast<float>(cos(angle));
        const float sine = static_cast<float>(sin(angle));
        const float paired = __bfloat162float(queryKey[inputOffset + pair]) * inverse
                * (1.0f + __bfloat162float(norm[pair]));
        result = rope_rotate(value, paired, cosine, sine, lane < half);
    }
    // The in-place query path cannot overwrite another warp's RoPE pair read.
    __syncthreads();
    output[outputOffset + lane] = __float2bfloat16_rn(result);
}

// Row-owned variant for 256-dimension heads: one CTA per row, one warp per head (eight dimensions
// per lane). The RoPE cosines and sines, computed in FP64 as above, are shared by every head of the
// row instead of recomputed per element. The RMS sum reproduces the 256-thread tree above (strides
// 128, 64 and 32 in registers, then 16 .. 1 across lanes), so results match it bit for bit.
// Launch: grid = rows, 256 threads; requires headDim == 256.
static __device__ __forceinline__ void qk_norm_rope_rows(
        const __nv_bfloat16* queryKey, const __nv_bfloat16* queryNorm, const __nv_bfloat16* keyNorm,
        __nv_bfloat16* output, uint32_t rows, uint32_t queryHeads, uint32_t keyValueHeads,
        uint32_t headDim, uint32_t rotaryDim, uint64_t startPosition, float epsilon, double ropeTheta) {
    const uint32_t row = blockIdx.x;
    if (row >= rows || headDim != 256u) return;
    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    const uint32_t heads = queryHeads + keyValueHeads, half = rotaryDim / 2;
    __shared__ float cosines[128], sines[128];
    for (uint32_t f = threadIdx.x; f < half; f += blockDim.x) {
        const double exponent = (2.0 * static_cast<double>(f)) / static_cast<double>(rotaryDim);
        const double angle = static_cast<double>(startPosition + row) * pow(ropeTheta, -exponent);
        cosines[f] = static_cast<float>(cos(angle));
        sines[f] = static_cast<float>(sin(angle));
    }
    __syncthreads();
    const uint64_t projectedWidth = static_cast<uint64_t>(heads) * 256u;
    for (uint32_t head = warp; head < heads; head += blockDim.x >> 5) {
        const uint64_t offset = row * projectedWidth + static_cast<uint64_t>(head) * 256u;
        const bool isQuery = head < queryHeads;
        const __nv_bfloat16* norm = isQuery ? queryNorm : keyNorm;
        float values[8], squares[8];
#pragma unroll
        for (int k = 0; k < 8; k++) {
            values[k] = __bfloat162float(queryKey[offset + lane + 32u * k]);
            squares[k] = values[k] * values[k];
        }
#pragma unroll
        for (int k = 0; k < 4; k++) squares[k] += squares[k + 4];
#pragma unroll
        for (int k = 0; k < 2; k++) squares[k] += squares[k + 2];
        float sum = squares[0] + squares[1];
#pragma unroll
        for (int stride = 16; stride; stride >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, stride);
        sum = __shfl_sync(0xffffffffu, sum, 0);
        const float inverse = rsqrtf(sum / 256.0f + epsilon);
        float results[8];
#pragma unroll
        for (int k = 0; k < 8; k++) {
            const uint32_t index = lane + 32u * k;
            const float value = values[k] * inverse * (1.0f + __bfloat162float(norm[index]));
            results[k] = value;
            if (index < rotaryDim) {
                const uint32_t pair = index < half ? index + half : index - half;
                const uint32_t frequency = index < half ? index : pair;
                const float paired = __bfloat162float(queryKey[offset + pair]) * inverse
                        * (1.0f + __bfloat162float(norm[pair]));
                results[k] = rope_rotate(value, paired, cosines[frequency], sines[frequency], index < half);
            }
        }
        // The in-place query path cannot overwrite a RoPE pair another lane still reads.
        __syncwarp();
#pragma unroll
        for (int k = 0; k < 8; k++) {
            const uint32_t index = lane + 32u * k;
            output[offset + index] = __float2bfloat16_rn(results[k]);
        }
    }
}

extern "C" __global__ __launch_bounds__(256) void euhedral_attention_qk_norm_rope_rows_bf16(
        const __nv_bfloat16* queryKey, const __nv_bfloat16* queryNorm, const __nv_bfloat16* keyNorm,
        __nv_bfloat16* output, uint32_t rows, uint32_t queryHeads, uint32_t keyValueHeads,
        uint32_t headDim, uint32_t rotaryDim, uint64_t startPosition, float epsilon, double ropeTheta) {
    euhedral_pdl_begin();
    qk_norm_rope_rows(queryKey, queryNorm, keyNorm, output, rows, queryHeads, keyValueHeads,
            headDim, rotaryDim, startPosition, epsilon, ropeTheta);
}

extern "C" __global__ void euhedral_attention_qk_norm_rope_bf16(
        const __nv_bfloat16* queryKey, const __nv_bfloat16* queryNorm, const __nv_bfloat16* keyNorm,
        __nv_bfloat16* output, uint32_t rows, uint32_t queryHeads, uint32_t keyValueHeads,
        uint32_t headDim, uint32_t rotaryDim, uint64_t startPosition, float epsilon, double ropeTheta) {
    euhedral_pdl_begin();
    qk_norm_rope(queryKey, queryNorm, keyNorm, output, rows, queryHeads, keyValueHeads,
            headDim, rotaryDim, startPosition, epsilon, ropeTheta);
}

