#pragma once
// GDN causal depthwise convolution over the projected Q/K/V rows, carrying the window state.
#include "reductions.cuh"
/// Rows are independent apart from a sliding window of the previous `kernelSize - 1` inputs, so
/// blockIdx.y splits them into blocks of `QWEN_GDN_CONV_ROWS` (at least the largest window): a block
/// seeds its window from the stored state (first block) or from the preceding input rows, which
/// hold the same values the window would. Each output keeps the original tap order. The new state
/// is the last window of inputs; the first block writes it after reading the old state, so no
/// other CTA touches a channel's state.
#define QWEN_GDN_CONV_ROWS 32u
extern "C" __global__ void euhedral_gdn_convolution_bf16(
        const __nv_bfloat16* queryKey, const __nv_bfloat16* valueZ,
        const __nv_bfloat16* convolutionWeights, __nv_bfloat16* convolutionState,
        __nv_bfloat16* output, uint32_t rows, uint32_t queryKeyWidth,
        uint32_t valueWidth, uint32_t convolutionWidth, uint32_t kernelSize) {
    euhedral_pdl_begin();
    const uint32_t channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= convolutionWidth) return;
    const uint32_t historyWidth = kernelSize - 1;
    const uint32_t first = blockIdx.y * QWEN_GDN_CONV_ROWS;
    if (first >= rows) return;
    const uint32_t last = min(rows, first + QWEN_GDN_CONV_ROWS);
    auto input = [&](uint32_t row) {
        return channel < queryKeyWidth
                ? __bfloat162float(queryKey[static_cast<uint64_t>(row) * queryKeyWidth + channel])
                : __bfloat162float(valueZ[static_cast<uint64_t>(row) * valueWidth * 2 + channel - queryKeyWidth]);
    };
    float history[31];
    for (uint32_t i = 0; i < historyWidth; i++) {
        history[i] = first == 0
                ? __bfloat162float(convolutionState[static_cast<uint64_t>(channel) * historyWidth + i])
                : input(first - historyWidth + i);
    }
    for (uint32_t row = first; row < last; row++) {
        float sum = 0.0f;
        for (uint32_t tap = 0; tap < kernelSize; tap++) {
            const float value = tap < historyWidth ? history[tap] : input(row);
            sum = fmaf(value, __bfloat162float(convolutionWeights[static_cast<uint64_t>(tap) * convolutionWidth + channel]), sum);
        }
        output[static_cast<uint64_t>(row) * convolutionWidth + channel] = __float2bfloat16_rn(qwen_gdn_silu(sum));
        if (historyWidth > 0) {
            for (uint32_t i = 0; i + 1 < historyWidth; i++) history[i] = history[i + 1];
            history[historyWidth - 1] = input(row);
        }
    }
    if (first != 0) return;
    // The first block owns the state. With one block its window already ends at the last row;
    // otherwise rows >= QWEN_GDN_CONV_ROWS > historyWidth and the window is the last inputs.
    for (uint32_t i = 0; i < historyWidth; i++) {
        const float value = last == rows ? history[i] : input(rows - historyWidth + i);
        convolutionState[static_cast<uint64_t>(channel) * historyWidth + i] = __float2bfloat16_rn(value);
    }
}
