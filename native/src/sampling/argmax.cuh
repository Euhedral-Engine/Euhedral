#pragma once
// Greedy token selection on the device. The host argmax keeps the lowest token ID among equal maxima
// and never selects NaN or negative infinity; the 64-bit key below orders logits the same way, so the
// largest key is the host's choice: the high word is the logit's order-preserving bit pattern (-0 is
// folded into +0, which compares equal on the host), the low word the inverted token ID. Key 0 marks a
// logit that cannot be selected, and a row of such logits yields 0.
#include "common/pdl.cuh"

static __device__ __forceinline__ unsigned long long argmax_key(unsigned int bits, unsigned int token) {
    unsigned int value = bits << 16;
    if (value == 0x80000000u) value = 0u;
    if ((value & 0x7fffffffu) > 0x7f800000u || value == 0xff800000u) return 0ull;
    unsigned int ordered = (value & 0x80000000u) ? ~value : value | 0x80000000u;
    return ((unsigned long long)ordered << 32) | (0xffffffffu - token);
}

static __device__ __forceinline__ unsigned long long argmax_max(unsigned long long a, unsigned long long b) {
    return a > b ? a : b;
}

// One 1024-thread CTA reads the row once, eight logits per 16-byte load where the row is aligned, and
// writes the winning key to `result`. A single CTA keeps the reduction order-free and deterministic; the
// row (248K logits, 0.5 MB) takes a few microseconds.
extern "C" __global__ __launch_bounds__(1024) void euhedral_argmax_bf16(
        const unsigned short* logits, unsigned int count, unsigned long long* result) {
    euhedral_pdl_begin();
    __shared__ unsigned long long warps[32];
    unsigned long long best = 0ull;
    unsigned int vectors = ((unsigned long long)logits & 15ull) == 0ull ? count / 8u : 0u;
    const uint4* rows = reinterpret_cast<const uint4*>(logits);
    for (unsigned int v = threadIdx.x; v < vectors; v += blockDim.x) {
        uint4 word = rows[v];
        unsigned int token = v * 8u;
        unsigned int parts[4] = {word.x, word.y, word.z, word.w};
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            best = argmax_max(best, argmax_key(parts[p] & 0xffffu, token + 2u * p));
            best = argmax_max(best, argmax_key(parts[p] >> 16, token + 2u * p + 1u));
        }
    }
    for (unsigned int token = vectors * 8u + threadIdx.x; token < count; token += blockDim.x)
        best = argmax_max(best, argmax_key(logits[token], token));
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        best = argmax_max(best, __shfl_xor_sync(0xffffffffu, best, offset));
    if ((threadIdx.x & 31u) == 0u) warps[threadIdx.x >> 5] = best;
    __syncthreads();
    if (threadIdx.x < 32u) {
        best = threadIdx.x < (blockDim.x >> 5) ? warps[threadIdx.x] : 0ull;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            best = argmax_max(best, __shfl_xor_sync(0xffffffffu, best, offset));
        if (threadIdx.x == 0u) *result = best;
    }
}

// Folds one logit into a thread's argmax key and its running softmax denominator `sum`, kept relative to `peak`
// (the largest selectable logit seen). Unselectable logits (NaN, negative infinity) add nothing.
static __device__ __forceinline__ void argmax_fold(
        unsigned int bits, unsigned int token, unsigned long long& best, float& peak, float& sum) {
    unsigned long long key = argmax_key(bits, token);
    if (key == 0ull) return;
    best = argmax_max(best, key);
    float value = __uint_as_float(bits << 16);
    if (value > peak) {
        sum = sum * __expf(peak - value) + 1.0f;
        peak = value;
    } else {
        sum += __expf(value - peak);
    }
}

static __device__ __forceinline__ void softmax_merge(float& peak, float& sum, float other_peak, float other_sum) {
    if (other_sum == 0.0f) return;
    if (sum == 0.0f) {
        peak = other_peak;
        sum = other_sum;
        return;
    }
    float top = fmaxf(peak, other_peak);
    sum = sum * __expf(peak - top) + other_sum * __expf(other_peak - top);
    peak = top;
}

// As euhedral_argmax_bf16, and the natural log of the winner's softmax probability over the row: `result` holds the
// 64-bit key, then that log-probability as a float (0 when no logit is selectable). The winner is the largest
// logit, so its log-probability is -log(sum over the row of exp(logit - max)). Positive infinity makes the sum
// infinite or NaN; such a row reports what the arithmetic gives. Each thread folds a fixed set of logits and the
// reduction runs in a fixed order, so the result is deterministic.
extern "C" __global__ __launch_bounds__(1024) void euhedral_argmax_logprob_bf16(
        const unsigned short* logits, unsigned int count, unsigned long long* result) {
    euhedral_pdl_begin();
    __shared__ unsigned long long warp_keys[32];
    __shared__ float warp_peaks[32];
    __shared__ float warp_sums[32];
    unsigned long long best = 0ull;
    float peak = __int_as_float(0xff800000), sum = 0.0f;
    unsigned int vectors = ((unsigned long long)logits & 15ull) == 0ull ? count / 8u : 0u;
    const uint4* rows = reinterpret_cast<const uint4*>(logits);
    for (unsigned int v = threadIdx.x; v < vectors; v += blockDim.x) {
        uint4 word = rows[v];
        unsigned int token = v * 8u;
        unsigned int parts[4] = {word.x, word.y, word.z, word.w};
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            argmax_fold(parts[p] & 0xffffu, token + 2u * p, best, peak, sum);
            argmax_fold(parts[p] >> 16, token + 2u * p + 1u, best, peak, sum);
        }
    }
    for (unsigned int token = vectors * 8u + threadIdx.x; token < count; token += blockDim.x)
        argmax_fold(logits[token], token, best, peak, sum);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        best = argmax_max(best, __shfl_xor_sync(0xffffffffu, best, offset));
        float other_peak = __shfl_xor_sync(0xffffffffu, peak, offset);
        float other_sum = __shfl_xor_sync(0xffffffffu, sum, offset);
        softmax_merge(peak, sum, other_peak, other_sum);
    }
    if ((threadIdx.x & 31u) == 0u) {
        warp_keys[threadIdx.x >> 5] = best;
        warp_peaks[threadIdx.x >> 5] = peak;
        warp_sums[threadIdx.x >> 5] = sum;
    }
    __syncthreads();
    if (threadIdx.x < 32u) {
        bool live = threadIdx.x < (blockDim.x >> 5);
        best = live ? warp_keys[threadIdx.x] : 0ull;
        peak = live ? warp_peaks[threadIdx.x] : __int_as_float(0xff800000);
        sum = live ? warp_sums[threadIdx.x] : 0.0f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            best = argmax_max(best, __shfl_xor_sync(0xffffffffu, best, offset));
            float other_peak = __shfl_xor_sync(0xffffffffu, peak, offset);
            float other_sum = __shfl_xor_sync(0xffffffffu, sum, offset);
            softmax_merge(peak, sum, other_peak, other_sum);
        }
        if (threadIdx.x == 0u) {
            result[0] = best;
            float log_probability = best == 0ull ? 0.0f : -logf(sum);
            result[1] = (unsigned long long)__float_as_uint(log_probability);
        }
    }
}
