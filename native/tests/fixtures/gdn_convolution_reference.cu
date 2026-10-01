// Frozen single-CTA-per-channel GDN convolution (before row blocks); test oracle only.
// It relies on qwen_gdn_silu from qwen_gdn_ops.cu, which the test includes first.
extern "C" __global__ void reference_gdn_convolution_bf16(
        const __nv_bfloat16* queryKey, const __nv_bfloat16* valueZ,
        const __nv_bfloat16* convolutionWeights, __nv_bfloat16* convolutionState,
        __nv_bfloat16* output, uint32_t rows, uint32_t queryKeyWidth,
        uint32_t valueWidth, uint32_t convolutionWidth, uint32_t kernelSize) {
    const uint32_t channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= convolutionWidth) return;
    const uint32_t historyWidth = kernelSize - 1;
    float history[31];
    for (uint32_t i = 0; i < historyWidth; i++) {
        history[i] = __bfloat162float(convolutionState[static_cast<uint64_t>(channel) * historyWidth + i]);
    }
    for (uint32_t row = 0; row < rows; row++) {
        float sum = 0.0f;
        for (uint32_t tap = 0; tap < kernelSize; tap++) {
            float value;
            if (tap < historyWidth) {
                value = history[tap];
            } else if (channel < queryKeyWidth) {
                value = __bfloat162float(queryKey[static_cast<uint64_t>(row) * queryKeyWidth + channel]);
            } else {
                value = __bfloat162float(valueZ[static_cast<uint64_t>(row) * valueWidth * 2 + channel - queryKeyWidth]);
            }
            sum = fmaf(value, __bfloat162float(convolutionWeights[static_cast<uint64_t>(tap) * convolutionWidth + channel]), sum);
        }
        output[static_cast<uint64_t>(row) * convolutionWidth + channel] = __float2bfloat16_rn(qwen_gdn_silu(sum));
        if (historyWidth > 0) {
            for (uint32_t i = 0; i + 1 < historyWidth; i++) history[i] = history[i + 1];
            const float current = channel < queryKeyWidth
                    ? __bfloat162float(queryKey[static_cast<uint64_t>(row) * queryKeyWidth + channel])
                    : __bfloat162float(valueZ[static_cast<uint64_t>(row) * valueWidth * 2 + channel - queryKeyWidth]);
            history[historyWidth - 1] = current;
        }
    }
    for (uint32_t i = 0; i < historyWidth; i++) {
        convolutionState[static_cast<uint64_t>(channel) * historyWidth + i] = __float2bfloat16_rn(history[i]);
    }
}
