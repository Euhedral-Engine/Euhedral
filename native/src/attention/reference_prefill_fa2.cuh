#pragma once
// Test-only control: the FlashAttention-2 prefill kernel as it was before the warp-specialized rewrite
// (nvfp4_prefill_fa2.cuh). The production kernel must equal it bit for bit for query-head groups up to 6. Not
// included by kernels.cu; tests include it after it.
extern "C" __global__ __launch_bounds__(256, 1) void euhedral_attention_prefill_fa2_nvfp4_reference(
        const __nv_bfloat16* queryKey, const __nv_bfloat16* gateValue,
        const unsigned char* const* keyPages, const unsigned char* const* valuePages,
        __nv_bfloat16* output, unsigned rows, unsigned queryHeads, unsigned keyHeads,
        unsigned headDim, unsigned cacheLength, unsigned long long start) {
    using namespace fa2;
    const unsigned lane = threadIdx.x & 31u, warp = threadIdx.x >> 5, g = lane >> 2, tig = lane & 3u;
    const unsigned group = queryHeads / keyHeads;
    const unsigned kh = blockIdx.x % keyHeads, first = (blockIdx.x / keyHeads) * 16;
    const unsigned head = kh * group + warp, width = (queryHeads + keyHeads) * D;
    __shared__ __align__(16) __half kv[2 * KT * STRIDE];
    __half* kbuf = kv;
    __half* vbuf = kv + KT * STRIDE;
    // Rotated queries: two warps at a time through qstage, then into A fragments (16 rows x 256 dims).
    unsigned qa[16][4];
    for (unsigned pass = 0; pass < (group + 1) / 2; pass++) {
        if (warp / 2 == pass) {
            __half* q = kv + (warp & 1u) * KT * STRIDE;
            for (unsigned r = 0; r < 16; r++) {
                float values[8];
                for (int d = 0; d < 8; d++) values[d] = first + r < rows
                        ? __bfloat162float(queryKey[(unsigned long long)(first + r) * width + head * D + lane + d * 32]) : 0.0f;
                nvfp4kv::hadamard256(values, lane);
                for (int d = 0; d < 8; d++) q[r * STRIDE + lane + d * 32] = __float2half_rn(values[d]);
            }
            __syncwarp();
            for (int k = 0; k < 16; k++) ldsm_x4(qa[k], q + (lane & 15u) * STRIDE + k * 16 + (lane >> 4) * 8);
        }
        __syncthreads();
    }
    float o[32][4];
    for (int j = 0; j < 32; j++) { o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.0f; }
    float m0 = -__int_as_float(0x7f800000), m1 = m0, l0 = 0.0f, l1 = 0.0f;
    const unsigned row0 = first + g, row1 = first + g + 8;
    const unsigned last = min(cacheLength, (unsigned)(start + min(first + 16, rows)));
    for (unsigned base = 0; base < last; base += KT) {
        stage(kbuf, keyPages, base, last, kh, keyHeads);
        stage(vbuf, valuePages, base, last, kh, keyHeads);
        __syncthreads();
        // S = Q K^T: 16 rows x 32 keys (4 n-tiles of 8 keys).
        float s[4][4];
        for (int n = 0; n < 4; n++) { s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.0f; }
        for (int k = 0; k < 16; k++) {
            for (int np = 0; np < 2; np++) {
                unsigned b[4];
                // keys 16 np + (lane & 7) + 8 (lane >> 4), dims 16 k + 8 ((lane >> 3) & 1)
                ldsm_x4(b, kbuf + (np * 16 + (lane & 7u) + ((lane >> 4) << 3)) * STRIDE + k * 16 + ((lane >> 3) & 1u) * 8);
                mma16816(s[2 * np], qa[k], b[0], b[1]);
                mma16816(s[2 * np + 1], qa[k], b[2], b[3]);
            }
        }
        // Mask, scale, online softmax per row (rows g and g + 8; the quad of tig lanes shares a row).
        float mx0 = m0, mx1 = m1;
        for (int n = 0; n < 4; n++) {
            for (int e = 0; e < 2; e++) {
                const unsigned key = base + n * 8 + 2 * tig + e;
                const bool v0 = row0 < rows && key < last && key <= start + row0;
                const bool v1 = row1 < rows && key < last && key <= start + row1;
                s[n][e] = v0 ? s[n][e] * 0.0625f : -__int_as_float(0x7f800000);
                s[n][2 + e] = v1 ? s[n][2 + e] * 0.0625f : -__int_as_float(0x7f800000);
                mx0 = fmaxf(mx0, s[n][e]); mx1 = fmaxf(mx1, s[n][2 + e]);
            }
        }
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1)); mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1)); mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
        const float c0 = isfinite(m0) ? __expf(m0 - mx0) : 0.0f, c1 = isfinite(m1) ? __expf(m1 - mx1) : 0.0f;
        float sum0 = 0.0f, sum1 = 0.0f;
        unsigned pa[2][4];  // P as A fragments: two k16 steps of 16 keys
        for (int n = 0; n < 4; n++) {
            const float p00 = isfinite(s[n][0]) ? __expf(s[n][0] - mx0) : 0.0f, p01 = isfinite(s[n][1]) ? __expf(s[n][1] - mx0) : 0.0f;
            const float p10 = isfinite(s[n][2]) ? __expf(s[n][2] - mx1) : 0.0f, p11 = isfinite(s[n][3]) ? __expf(s[n][3] - mx1) : 0.0f;
            sum0 += p00 + p01; sum1 += p10 + p11;
            pa[n / 2][(n & 1) * 2] = pack_half2(p00, p01);
            pa[n / 2][(n & 1) * 2 + 1] = pack_half2(p10, p11);
        }
        sum0 += __shfl_xor_sync(0xffffffffu, sum0, 1); sum0 += __shfl_xor_sync(0xffffffffu, sum0, 2);
        sum1 += __shfl_xor_sync(0xffffffffu, sum1, 1); sum1 += __shfl_xor_sync(0xffffffffu, sum1, 2);
        l0 = l0 * c0 + sum0; l1 = l1 * c1 + sum1; m0 = mx0; m1 = mx1;
        for (int j = 0; j < 32; j++) { o[j][0] *= c0; o[j][1] *= c0; o[j][2] *= c1; o[j][3] *= c1; }
        // O += P V: 2 k16 steps x 32 n-tiles of 8 dims; V^T fragments by transposed ldmatrix.
        for (int k = 0; k < 2; k++) {
            for (int jp = 0; jp < 16; jp++) {
                unsigned b[4];
                // keys 16 k + (lane & 7) + 8 ((lane >> 3) & 1), dims 16 jp + 8 (lane >> 4)
                ldsm_x4_t(b, vbuf + (k * 16 + (lane & 7u) + (((lane >> 3) & 1u) << 3)) * STRIDE + jp * 16 + (lane >> 4) * 8);
                mma16816(o[2 * jp], pa[k], b[0], b[1]);
                mma16816(o[2 * jp + 1], pa[k], b[2], b[3]);
            }
        }
        __syncthreads();
    }
    // Normalize, rotate back and gate: rows through shared memory, two warps at a time.
    const float inv0 = l0 > 0.0f ? 1.0f / l0 : 0.0f, inv1 = l1 > 0.0f ? 1.0f / l1 : 0.0f;
    float* rowsbuf = reinterpret_cast<float*>(kv);  // 16 x 256 FP32 = 16 KB per warp; kv holds two
    for (unsigned pass = 0; pass < (group + 1) / 2; pass++) {
        if (warp / 2 == pass) {
            float* buf = rowsbuf + (warp & 1u) * 16 * D;
            for (int j = 0; j < 32; j++) {
                const unsigned col = j * 8 + 2 * tig;
                buf[g * D + col] = o[j][0] * inv0; buf[g * D + col + 1] = o[j][1] * inv0;
                buf[(g + 8) * D + col] = o[j][2] * inv1; buf[(g + 8) * D + col + 1] = o[j][3] * inv1;
            }
            __syncwarp();
            for (unsigned r = 0; r < 16; r++) {
                const unsigned row = first + r;
                float values[8];
                for (int d = 0; d < 8; d++) values[d] = buf[r * D + lane + d * 32];
                nvfp4kv::hadamard256(values, lane);
                if (row < rows) {
                    for (int d = 0; d < 8; d++) {
                        const unsigned col = head * D + lane + d * 32;
                        const float gate = __bfloat162float(gateValue[(unsigned long long)row * width + col]);
                        output[(unsigned long long)row * queryHeads * D + col] = __float2bfloat16_rn(values[d] / (1.0f + expf(-gate)));
                    }
                }
            }
        }
        __syncthreads();
    }
}
