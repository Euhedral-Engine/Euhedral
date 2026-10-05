#pragma once
#include "qwen4/common.cuh"

// Per-layer embedding (Qwen4ExpTextPLELayer) kernels.

// E4M3 block scale (sign ignored, subnormals included); 0x7f is NaN.
static __device__ __forceinline__ float q4_e4m3(unsigned int code) {
    const unsigned int exponent = (code >> 3) & 15u, mantissa = code & 7u;
    if (exponent == 15u && mantissa == 7u) return __uint_as_float(0x7fc00000u);
    if (exponent == 0u) return (float)mantissa * 0.001953125f;  // m / 8 * 2^-6
    return (1.0f + (float)mantissa * 0.125f) * __uint_as_float((exponent + 120u) << 23);  // 2^(e - 7)
}

static __device__ __forceinline__ float q4_e2m1(unsigned int nibble) {
    const float magnitude[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float value = magnitude[nibble & 7u];
    return (nibble & 8u) != 0u ? -value : value;
}

// Expands staged n-gram records into BF16 rows. A record is `record_bytes` (a multiple of 4) bytes: the row's
// width / 2 code bytes (even column in the low nibble), its width / 16 E4M3 block scales, zero padding, and the
// shard's FP32 global scale in the last four bytes. A value is bf16((e2m1 * sign) * (e4m3 * global)), the order of
// the engine's NVFP4 expansion. Rows land one after another, so records in (token, head) order give the
// concatenated [token][head * width] embedding of the n-gram heads.
//   grid records, block 256 (a thread per value, strided).
extern "C" __global__ __launch_bounds__(256) void euhedral_q4_ngram_expand_bf16(
        const unsigned char* __restrict__ records, unsigned short* __restrict__ output,
        unsigned int count, unsigned int width, unsigned int record_bytes) {
    const unsigned int record = blockIdx.x;
    if (record >= count) return;
    const unsigned char* base = records + (unsigned long long)record * record_bytes;
    const float global = *reinterpret_cast<const float*>(base + record_bytes - 4u);
    const unsigned char* codes = base;
    const unsigned char* scales = base + (width >> 1);
    for (unsigned int i = threadIdx.x; i < width; i += blockDim.x) {
        const unsigned int byte = codes[i >> 1];
        const unsigned int nibble = (i & 1u) != 0u ? byte >> 4 : byte & 15u;
        const float block = q4_e4m3(scales[i >> 4]) * global;
        output[(unsigned long long)record * width + i] = q4::bfr(q4_e2m1(nibble) * block);
    }
}

// The PLE gate. For every (row, stream) the dot product of the normalized key and the normalized query stream
// decides how much of the shared value the stream receives:
//   dot   = bf16(sum_d bf16(key[d] * query[d]))            (torch: the product, then the sum, each in BF16)
//   gate  = bf16(dot * (1 / sqrt(width)))
//   gate  = bf16(sqrt(bf16(max(|gate|, 1e-6)))) * sign(gate)
//   out[d]= bf16(bf16(sigmoid(gate)) * value[d])
//   grid rows * streams, block 256; key and query are [row][stream][width], value [row][width].
extern "C" __global__ __launch_bounds__(256) void euhedral_q4_ple_gate_bf16(
        const unsigned short* __restrict__ key, const unsigned short* __restrict__ query,
        const unsigned short* __restrict__ value, unsigned short* __restrict__ output,
        unsigned int rows, unsigned int streams, unsigned int width) {
    __shared__ float scratch[32];
    const unsigned int row = blockIdx.x / streams;
    if (row >= rows) return;
    const unsigned long long at = (unsigned long long)blockIdx.x * width;
    float sum = 0.0f;
    for (unsigned int d = threadIdx.x; d < width; d += blockDim.x)
        sum += q4::round_bf(q4::bf(key[at + d]) * q4::bf(query[at + d]));
    sum = q4::block_sum(sum, scratch);
    float gate = q4::round_bf(q4::round_bf(sum) * (1.0f / sqrtf((float)width)));
    const float magnitude = q4::round_bf(sqrtf(q4::round_bf(fmaxf(fabsf(gate), 1e-6f))));
    const float sign = gate > 0.0f ? 1.0f : (gate < 0.0f ? -1.0f : 0.0f);
    gate = magnitude * sign;
    const float scale = q4::round_bf(q4::sigmoid(gate));
    for (unsigned int d = threadIdx.x; d < width; d += blockDim.x)
        output[at + d] = q4::bfr(scale * q4::bf(value[(unsigned long long)row * width + d]));
}

// The PLE short convolution: a depthwise convolution of `taps` taps with `dilation` over the normalized gated
// values, preceded by (taps - 1) * dilation history rows, then
//   out[t][c] = bf16(gated[t][c] + bf16(silu(bf16(sum_i w[c][i] * x[t - history + dilation * i][c])))).
// `history_rows` holds the previous rows of x, oldest first, [history][channels] (zero at the start of a sequence).
// `weights` is [channels][taps]. The history is NOT updated here: [euhedral_q4_conv_history_bf16] does that.
//   grid (ceil(channels / 256), rows), block 256.
extern "C" __global__ __launch_bounds__(256) void euhedral_q4_ple_conv_bf16(
        const unsigned short* __restrict__ normed, const unsigned short* __restrict__ gated,
        const unsigned short* __restrict__ history_rows, const unsigned short* __restrict__ weights,
        unsigned short* __restrict__ output, unsigned int rows, unsigned int channels, unsigned int taps,
        unsigned int dilation) {
    const unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int t = blockIdx.y;
    if (c >= channels || t >= rows) return;
    const int history = (int)((taps - 1u) * dilation);
    float sum = 0.0f;
    for (unsigned int i = 0; i < taps; i++) {
        const int source = (int)t - history + (int)(dilation * i);  // row of x; negative reaches into the history
        const float x = source >= 0 ? q4::bf(normed[(unsigned long long)source * channels + c])
                                    : q4::bf(history_rows[(unsigned long long)(history + source) * channels + c]);
        sum = fmaf(q4::bf(weights[(unsigned long long)c * taps + i]), x, sum);
    }
    const float convolved = q4::round_bf(q4::silu(q4::round_bf(sum)));
    output[(unsigned long long)t * channels + c] = q4::bfr(q4::bf(gated[(unsigned long long)t * channels + c]) + convolved);
}

// The last `history` rows of (history rows ++ x), written back over the history. One thread per channel reads all
// it needs before it writes.
//   grid ceil(channels / 256), block 256.
extern "C" __global__ __launch_bounds__(256) void euhedral_q4_conv_history_bf16(
        const unsigned short* __restrict__ x, unsigned short* history_rows, unsigned int rows, unsigned int channels,
        unsigned int history) {
    const unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    // History is small (9 for the PLE convolution); keep it in registers. The new history is entries
    // rows .. rows + history - 1 of (old history ++ x).
    unsigned short kept[32];
    for (unsigned int i = 0; i < history; i++) {
        const unsigned long long j = (unsigned long long)rows + i;
        kept[i] = j >= history ? x[(j - history) * channels + c] : history_rows[j * channels + c];
    }
    for (unsigned int i = 0; i < history; i++) history_rows[(unsigned long long)i * channels + c] = kept[i];
}
