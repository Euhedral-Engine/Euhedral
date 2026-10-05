#pragma once
#include "qwen4/common.cuh"

// Gated DeltaNet operators of the Flash-Next layers around the shared delta-rule recurrence
// (euhedral_gdn_recurrence_*): the dense engine's recurrence reads the convolved [query | key | value] row and FP32
// decay/beta, which is the layout these kernels write. The convolution, the control and the output norm are written
// here because upstream's rounding, the fused QKV layout and the sigmoid output gate differ from the dense kernels.

// Causal depthwise convolution of the fused QKV projection, then SiLU, as Qwen4ExpTextGatedDeltaNet runs it:
//   out[t][c] = bf16(silu(bf16(sum_i w[c][i] * xpad[t + i][c]))),  xpad = (taps - 1) history rows ++ x.
// `weights` is [channels][taps]. The history is read, not written: [euhedral_q4_conv_history_bf16] updates it.
//   grid (ceil(channels / 256), rows), block 256.
extern "C" __global__ __launch_bounds__(256) void euhedral_q4_gdn_conv_bf16(
        const unsigned short* __restrict__ x, const unsigned short* __restrict__ history_rows,
        const unsigned short* __restrict__ weights, unsigned short* __restrict__ output,
        unsigned int rows, unsigned int channels, unsigned int taps) {
    const unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int t = blockIdx.y;
    if (c >= channels || t >= rows) return;
    const int history = (int)taps - 1;
    float sum = 0.0f;
    for (unsigned int i = 0; i < taps; i++) {
        const int source = (int)t - history + (int)i;  // row of x; negative reaches into the history
        const float value = source >= 0 ? q4::bf(x[(unsigned long long)source * channels + c])
                                        : q4::bf(history_rows[(unsigned long long)(history + source) * channels + c]);
        sum = fmaf(q4::bf(weights[(unsigned long long)c * taps + i]), value, sum);
    }
    output[(unsigned long long)t * channels + c] = q4::bfr(q4::silu(q4::round_bf(sum)));
}

// The recurrence's per-(row, value head) inputs from the BF16 a and b projections:
//   alpha = exp(g), g = -exp(A_log) * softplus(a + dt_bias)   (FP32; softplus passes values above 20 through)
//   beta  = bf16(sigmoid(b))                                  (a BF16 tensor upstream, widened to FP32)
//   grid ceil(rows * heads / 256), block 256.
extern "C" __global__ __launch_bounds__(256) void euhedral_q4_gdn_control_bf16(
        const unsigned short* __restrict__ a_projection, const unsigned short* __restrict__ b_projection,
        const unsigned short* __restrict__ a_log, const unsigned short* __restrict__ dt_bias,
        float* __restrict__ alpha, float* __restrict__ beta, unsigned int rows, unsigned int heads) {
    const unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= (unsigned long long)rows * heads) return;
    const unsigned int head = (unsigned int)(index % heads);
    const float shifted = q4::bf(a_projection[index]) + q4::bf(dt_bias[head]);
    const float softplus = shifted > 20.0f ? shifted : log1pf(expf(shifted));
    alpha[index] = expf(-expf(q4::bf(a_log[head])) * softplus);
    beta[index] = q4::round_bf(q4::sigmoid(q4::bf(b_projection[index])));
}

// Qwen4ExpTextRMSNormGated: norm before gate, with upstream's three roundings,
//   n = bf16(x * rsqrt(mean(x^2) + eps)),  w = bf16(weight * n),  out = bf16(w * act(z)),
// per head of `head_dim` values; `activation` is 0 for SiLU, 1 for sigmoid (the model's output gate).
//   grid rows * heads, block 128 (head_dim <= 128 per thread stride).
extern "C" __global__ __launch_bounds__(128) void euhedral_q4_gdn_gated_norm_bf16(
        const unsigned short* __restrict__ recurrent, const unsigned short* __restrict__ z,
        const unsigned short* __restrict__ weight, unsigned short* __restrict__ output,
        unsigned int rows, unsigned int heads, unsigned int head_dim, float epsilon, unsigned int activation) {
    __shared__ float scratch[32];
    const unsigned int row = blockIdx.x / heads;
    if (row >= rows) return;
    const unsigned long long base = (unsigned long long)blockIdx.x * head_dim;
    float sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        const float v = q4::bf(recurrent[base + i]);
        sum = fmaf(v, v, sum);
    }
    sum = q4::block_sum(sum, scratch);
    const float inverse = rsqrtf(sum / (float)head_dim + epsilon);
    for (unsigned int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        const float normalized = q4::round_bf(q4::bf(recurrent[base + i]) * inverse);
        const float weighted = q4::round_bf(q4::bf(weight[i]) * normalized);
        const float gate = q4::bf(z[base + i]);
        output[base + i] = q4::bfr(weighted * (activation == 0u ? q4::silu(gate) : q4::sigmoid(gate)));
    }
}
