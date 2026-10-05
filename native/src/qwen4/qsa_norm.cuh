#pragma once
#include "qwen4/common.cuh"

// Qwen Sparse Attention (QSA) front end: the per-head RMSNorm and partial RoPE of the attention and indexer
// vectors, the block-key pooling of the indexer and the raw-key tail it keeps between chunks
// (docs/FLASH_NEXT_QSA.md). Upstream computes these as BF16 tensor operations, so every step rounds to BF16.

namespace q4 {

// cos and sin of RoPE frequency `index` at `position`, as Qwen4ExpTextRotaryEmbedding builds them: inv_freq =
// 1 / theta^(2 index / rotary) in FP32, the phase position * inv_freq rounded to FP32, cos and sin rounded to BF16.
static __device__ __forceinline__ void rope_cos_sin(
        unsigned int position, unsigned int index, unsigned int rotary, float theta, float& cosine, float& sine) {
    const float power = (float)pow((double)theta, (double)(2u * index) / (double)rotary);
    const float angle = __fmul_rn((float)position, 1.0f / power);
    cosine = round_bf(cosf(angle));
    sine = round_bf(sinf(angle));
}

}  // namespace q4

// Per-head RMSNorm with one weight shared by every head, followed by RoPE on the first `rotary` values of each head.
// A warp owns one (row, head) vector of `width` values (a multiple of 32, at most 256): lane l holds values
// l + 32 r. With rotary = 64 the rotate_half partner of value l is value l + 32, held by the same lane, so the
// rotation needs no exchange. Strides are in elements; input and output may be the same memory.
//   normalized  = bf16((x * rsqrt(mean(x^2) + eps)) * (1 + w))
//   rotated     = bf16(bf16(x * cos) + bf16(rotate_half(x) * sin)),  position = position + row * position_step
//   grid ceil(rows * heads / 4), block 128. rotary = 0 only normalizes.
extern "C" __global__ __launch_bounds__(128) void euhedral_q4_head_norm_rope_bf16(
        const unsigned short* input, const unsigned short* __restrict__ weight, unsigned short* output,
        unsigned int rows, unsigned int heads, unsigned int width, unsigned int in_row_stride,
        unsigned int in_head_stride, unsigned int out_row_stride, unsigned int out_head_stride, unsigned int rotary,
        unsigned int position, unsigned int position_step, float epsilon, float theta) {
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int unit = blockIdx.x * 4u + (threadIdx.x >> 5);
    if (unit >= rows * heads) return;
    const unsigned int row = unit / heads, head = unit % heads;
    const unsigned short* x = input + (unsigned long long)row * in_row_stride + (unsigned long long)head * in_head_stride;
    unsigned short* y = output + (unsigned long long)row * out_row_stride + (unsigned long long)head * out_head_stride;
    float v[8];
    float sum = 0.0f;
#pragma unroll
    for (int r = 0; r < 8; r++) {
        v[r] = 0.0f;
        if (r * 32 < (int)width) {
            v[r] = q4::bf(x[lane + 32 * r]);
            sum = fmaf(v[r], v[r], sum);
        }
    }
    sum = q4::warp_sum(sum);
    const float inverse = rsqrtf(sum / (float)width + epsilon);
#pragma unroll
    for (int r = 0; r < 8; r++)
        if (r * 32 < (int)width) v[r] = q4::round_bf((v[r] * inverse) * (1.0f + q4::bf(weight[lane + 32 * r])));
    if (rotary == 64u) {
        float cosine, sine;
        q4::rope_cos_sin(position + row * position_step, lane, rotary, theta, cosine, sine);
        const float first = v[0], second = v[1];
        v[0] = q4::round_bf(q4::round_bf(first * cosine) + q4::round_bf(-second * sine));
        v[1] = q4::round_bf(q4::round_bf(second * cosine) + q4::round_bf(first * sine));
    }
#pragma unroll
    for (int r = 0; r < 8; r++)
        if (r * 32 < (int)width) y[lane + 32 * r] = q4::bfr(v[r]);
}

// Pools the raw indexer keys of the blocks completed by a chunk: block j (tokens 4j .. 4j + 3, j from start / 4) is
// bf16(mean in FP32 of its four raw keys), written to blocks[j]. A raw key is read from the chunk's rows (`raw`,
// row stride raw_stride, token start + row) or, for the tokens of the incomplete block before the chunk, from `tail`
// (token 4 (start / 4) + i at tail[i]). The result still has to be normalized and rotated
// (euhedral_q4_head_norm_rope_bf16 with position_step 4).
//   grid ceil(count / 4), block 128: a warp per block, lane l holds values l + 32 r.
extern "C" __global__ __launch_bounds__(128) void euhedral_q4_qsa_pool_keys_bf16(
        const unsigned short* __restrict__ raw, const unsigned short* __restrict__ tail,
        unsigned short* __restrict__ blocks, unsigned int count, unsigned int start, unsigned int raw_stride) {
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int unit = blockIdx.x * 4u + (threadIdx.x >> 5);
    if (unit >= count) return;
    const unsigned int first = start >> 2, block = first + unit;
#pragma unroll
    for (int r = 0; r < 4; r++) {
        float sum = 0.0f;
#pragma unroll
        for (unsigned int i = 0; i < 4u; i++) {
            const unsigned int token = 4u * block + i;
            const unsigned short* key = token < start ? tail + (unsigned long long)(token - 4u * first) * 128u
                                                      : raw + (unsigned long long)(token - start) * raw_stride;
            sum += q4::bf(key[lane + 32 * r]);
        }
        blocks[(unsigned long long)block * 128u + lane + 32 * r] = q4::bfr(sum * 0.25f);
    }
}

// The raw keys of the incomplete trailing block after a chunk of `rows` tokens at `start`: tokens
// 4 ((start + rows) / 4) .. start + rows - 1, at tail_out[0 ..]. They come from the previous tail or the chunk, so
// tail_out must not be tail_in. One block of 128 threads, thread d takes dimension d.
extern "C" __global__ __launch_bounds__(128) void euhedral_q4_qsa_tail_bf16(
        const unsigned short* __restrict__ raw, const unsigned short* __restrict__ tail_in,
        unsigned short* __restrict__ tail_out, unsigned int rows, unsigned int start, unsigned int raw_stride) {
    const unsigned int d = threadIdx.x;
    const unsigned int end = start + rows;
    const unsigned int first_token = end & ~3u, old_base = start & ~3u;
    for (unsigned int token = first_token; token < end; token++) {
        tail_out[(token - first_token) * 128u + d] =
                token < start ? tail_in[(token - old_base) * 128u + d] : raw[(unsigned long long)(token - start) * raw_stride + d];
    }
}
