#pragma once
#include "nvfp4_kv.cuh"
#include "nvfp4_attention.cuh"
// FlashAttention-2 style NVFP4 prefill (relaxed). A CTA owns 16 query rows of one KV head's group of
// G query heads, one warp per query head (blockDim = 32 G, G <= 8). Each 32-key tile of the cache is
// expanded once into padded shared FP16 rows that the G warps share; the rotated queries (A fragments)
// and the 16 x 256 FP32 output stay in registers, scores and probabilities never leave registers, and
// the online softmax uses quad shuffles. mma.sync m16n8k16 FP16 with FP32 accumulation; only the
// accumulation order differs from euhedral_attention_prefill32_nvfp4 (and its _exact twin).
namespace fa2 {
constexpr int D = 256, KT = 32, STRIDE = D + 8;   // padded FP16 row stride (528 bytes)
static __device__ __forceinline__ void mma16816(float (&c)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
static __device__ __forceinline__ void ldsm_x4(unsigned (&r)[4], const void* p) {
    unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
static __device__ __forceinline__ void ldsm_x4_t(unsigned (&r)[4], const void* p) {
    unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
static __device__ __forceinline__ unsigned pack_half2(float lo, float hi) {
    __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<unsigned*>(&h);
}
// Expand 32 cached tokens of one KV head into padded FP16 rows (exact values).
static __device__ __forceinline__ void stage(__half* kv, const unsigned char* const* pages, unsigned base, unsigned last,
        unsigned head, unsigned heads) {
    for (unsigned g = threadIdx.x; g < KT * nvfp4kv::kGroups; g += blockDim.x) {
        const unsigned token = base + g / nvfp4kv::kGroups, group = g % nvfp4kv::kGroups;
        __half2 out[8];
        if (token < last) {
            const unsigned char* row = nvfp4kv::cache_row(pages, token, head, heads);
            const uint2 codes = *reinterpret_cast<const uint2*>(row + group * 8);
            const float scale = nvfp4kv::e4m3_decode(row[nvfp4kv::kCodeBytes + group]);
            __half2 (&lo)[4] = *reinterpret_cast<__half2 (*)[4]>(&out[0]);
            __half2 (&hi)[4] = *reinterpret_cast<__half2 (*)[4]>(&out[4]);
            nvfp4kv::dequantize8_f16(codes.x, scale, lo);
            nvfp4kv::dequantize8_f16(codes.y, scale, hi);
        } else {
            for (int i = 0; i < 8; i++) out[i] = __floats2half2_rn(0.0f, 0.0f);
        }
        uint4* dst = reinterpret_cast<uint4*>(kv + (g / nvfp4kv::kGroups) * STRIDE + group * 16);
        dst[0] = *reinterpret_cast<const uint4*>(&out[0]);
        dst[1] = *reinterpret_cast<const uint4*>(&out[4]);
    }
}
}  // namespace fa2

extern "C" __global__ __launch_bounds__(256, 1) void euhedral_attention_prefill_fa2_nvfp4(
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
