#pragma once
#include "qwen4/common.cuh"

// Hyper-connection (gated residual) kernels. The residual state of a token is `streams` rows of `width` values
// stored one after another; Qwen4ExpTextGatedResidual normalizes each stream, mixes the streams down to one input
// for the block and injects the block's result back into every stream.

// Qwen4ExpTextRMSNorm with group_size: each of `groups` groups of `width` values of a row is normalized on its
// own, out = bf16((x * rsqrt(mean(x^2) + eps)) * (1 + weight)), the weight spanning all groups of a row.
//   grid rows * groups, block 256.
extern "C" __global__ __launch_bounds__(256) void euhedral_q4_grouped_rms_norm_bf16(
        const unsigned short* __restrict__ input, const unsigned short* __restrict__ weight,
        unsigned short* __restrict__ output, unsigned int rows, unsigned int groups, unsigned int width, float epsilon) {
    __shared__ float scratch[32];
    const unsigned int group = blockIdx.x % groups, row = blockIdx.x / groups;
    if (row >= rows) return;
    const unsigned long long base = ((unsigned long long)row * groups + group) * width;
    const unsigned short* x = input + base;
    float sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < width; i += blockDim.x) {
        const float v = q4::bf(x[i]);
        sum = fmaf(v, v, sum);
    }
    sum = q4::block_sum(sum, scratch);
    const float inverse = rsqrtf(sum / (float)width + epsilon);
    const unsigned short* w = weight + (unsigned long long)group * width;
    for (unsigned int i = threadIdx.x; i < width; i += blockDim.x)
        output[base + i] = q4::bfr((q4::bf(x[i]) * inverse) * (1.0f + q4::bf(w[i])));
}

// out = bf16(silu(bf16(in / divisor))): the low-rank input mix between its two projections.
extern "C" __global__ void euhedral_q4_scaled_silu_bf16(
        const unsigned short* __restrict__ input, unsigned short* __restrict__ output, unsigned int count, float divisor) {
    const unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const float scaled = q4::round_bf(q4::bf(input[i]) / divisor);
    output[i] = q4::bfr(q4::silu(scaled));
}

// mixed[r][d] = bf16(mean over streams s of bf16(bf16(sigmoid(up[r][s][d])) * normed[r][s][d])).
//   grid (ceil(width / 256), rows), block 256.
extern "C" __global__ __launch_bounds__(256) void euhedral_q4_hc_mix_bf16(
        const unsigned short* __restrict__ normed, const unsigned short* __restrict__ up,
        unsigned short* __restrict__ mixed, unsigned int rows, unsigned int streams, unsigned int width) {
    const unsigned int d = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int row = blockIdx.y;
    if (d >= width || row >= rows) return;
    float sum = 0.0f;
    for (unsigned int s = 0; s < streams; s++) {
        const unsigned long long at = ((unsigned long long)row * streams + s) * width + d;
        const float gate = q4::round_bf(q4::sigmoid(q4::bf(up[at])));
        sum += q4::round_bf(gate * q4::bf(normed[at]));
    }
    mixed[(unsigned long long)row * width + d] = q4::bfr(sum / (float)streams);
}

// out[r][s][d] = bf16(hyper[r][s][d] + bf16(block[r][d] * w[r][s])) with w = 2 * bf16(sigmoid(bf16(raw[r][s] / streams))).
//   grid (ceil(width / 256), rows), block 256; `raw` is the block_inject projection, [rows][streams].
extern "C" __global__ __launch_bounds__(256) void euhedral_q4_hc_inject_bf16(
        const unsigned short* __restrict__ hyper, const unsigned short* __restrict__ block,
        const unsigned short* __restrict__ raw, unsigned short* __restrict__ output,
        unsigned int rows, unsigned int streams, unsigned int width) {
    const unsigned int d = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int row = blockIdx.y;
    if (d >= width || row >= rows) return;
    const float value = q4::bf(block[(unsigned long long)row * width + d]);
    for (unsigned int s = 0; s < streams; s++) {
        const float weight = 2.0f * q4::round_bf(q4::sigmoid(q4::round_bf(q4::bf(raw[(unsigned long long)row * streams + s]) / (float)streams)));
        const unsigned long long at = ((unsigned long long)row * streams + s) * width + d;
        output[at] = q4::bfr(q4::bf(hyper[at]) + q4::round_bf(value * weight));
    }
}

// The embedding row repeated over `streams` streams: out[r][s][d] = in[r][d].
extern "C" __global__ void euhedral_q4_repeat_streams_bf16(
        const unsigned short* __restrict__ input, unsigned short* __restrict__ output,
        unsigned int rows, unsigned int streams, unsigned int width) {
    const unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long total = (unsigned long long)rows * streams * width;
    if (i >= total) return;
    const unsigned long long row = i / ((unsigned long long)streams * width);
    const unsigned long long d = i % width;
    output[i] = input[row * width + d];
}
