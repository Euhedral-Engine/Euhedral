#pragma once
// GDN output: gated RMSNorm of the recurrent output by the Z projection.
#include "reductions.cuh"
extern "C" __global__ void euhedral_gdn_gated_rms_norm_bf16(
        const __nv_bfloat16* recurrent, const __nv_bfloat16* valueZ,
        const __nv_bfloat16* normWeight, __nv_bfloat16* output,
        uint32_t rows, uint32_t valueHeads, uint32_t headDim, float epsilon) {
    euhedral_pdl_begin();
    const uint32_t rowHead = blockIdx.x;
    const uint32_t row = rowHead / valueHeads;
    const uint32_t head = rowHead % valueHeads;
    if (row >= rows) return;
    const uint32_t lane = threadIdx.x;
    const uint32_t width = valueHeads * headDim;
    const uint64_t recurrentOffset = static_cast<uint64_t>(row) * width + static_cast<uint64_t>(head) * headDim;
    const uint64_t zOffset = static_cast<uint64_t>(row) * width * 2 + width + static_cast<uint64_t>(head) * headDim;
    float partial = 0.0f;
    for (uint32_t i = lane; i < headDim; i += blockDim.x) {
        const float value = __bfloat162float(recurrent[recurrentOffset + i]);
        partial = fmaf(value, value, partial);
    }
    __shared__ float scratch[128];
    const float sum = qwen_gdn_reduce_sum(partial, scratch);
    const float inverse = rsqrtf(sum / static_cast<float>(headDim) + epsilon);
    for (uint32_t i = lane; i < headDim; i += blockDim.x) {
        const float x = __bfloat162float(recurrent[recurrentOffset + i]);
        const float z = __bfloat162float(valueZ[zOffset + i]);
        output[recurrentOffset + i] = __float2bfloat16_rn(x * inverse * __bfloat162float(normWeight[i]) * qwen_gdn_silu(z));
    }
}
