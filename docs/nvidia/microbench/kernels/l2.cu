// L2 residency probes: a "hot" buffer re-read between passes of a large streaming
// read, with and without cache-policy hints.
typedef unsigned long long u64;

__device__ __forceinline__ u64 policy_evict_last() {
    u64 p;
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p));
    return p;
}
__device__ __forceinline__ u64 policy_evict_first() {
    u64 p;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
    return p;
}

// hint: 0 plain, 1 evict_last (hot) / evict_first (stream) via createpolicy
extern "C" __global__ void hinted_read(const uint4 *__restrict__ p, size_t n, int hint, int hot,
                                       unsigned *out) {
    u64 pol = hot ? policy_evict_last() : policy_evict_first();
    size_t stride = (size_t)gridDim.x * blockDim.x;
    unsigned acc = 0;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i + 3 * stride < n; i += 4 * stride) {
        uint4 v[4];
#pragma unroll
        for (int u = 0; u < 4; u++) {
            const uint4 *a = p + i + u * stride;
            if (hint)
                asm volatile("ld.global.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;"
                             : "=r"(v[u].x), "=r"(v[u].y), "=r"(v[u].z), "=r"(v[u].w)
                             : "l"(a), "l"(pol));
            else
                asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                             : "=r"(v[u].x), "=r"(v[u].y), "=r"(v[u].z), "=r"(v[u].w)
                             : "l"(a));
        }
#pragma unroll
        for (int u = 0; u < 4; u++) acc ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
    }
    if (acc == 0x9e3779b9u) out[0] = acc;
}

// Reset persisting lines to normal (what cudaCtxResetPersistingL2Cache does per line).
extern "C" __global__ void fill_random(unsigned *p, size_t n, unsigned seed) {
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        unsigned x = (unsigned)i * 0x9e3779b9u ^ seed;
        x ^= x >> 16;
        x *= 0x7feb352du;
        x ^= x >> 15;
        p[i] = x;
    }
}

// Pattern generator for the compression probe. Word i of line L (32 words per 128 B line).
//  0 zeros | 1 constant 1.0f | 2 50% zero words | 3 90% zero words | 4 50% zero lines
//  5 bytes 0..15 (4-bit values in 8-bit containers) | 6 BF16 pairs, |x| in [1,2), random mantissa
//  7 FP32 values sharing exponent (random mantissa) | 8 small ints 0..255 as u32 | 9 random
__device__ __forceinline__ unsigned mix32(unsigned x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}
extern "C" __global__ void fill_pattern(unsigned *p, size_t n, int mode) {
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        unsigned r = mix32((unsigned)i * 0x9e3779b9u + 0x1234567u);
        unsigned v;
        switch (mode) {
        case 0: v = 0; break;
        case 1: v = 0x3f800000u; break;
        case 2: v = (r & 1) ? r : 0; break;
        case 3: v = (r % 10 == 0) ? r : 0; break;
        case 4: v = (mix32((unsigned)(i >> 5)) & 1) ? r : 0; break;
        case 5: v = r & 0x0f0f0f0fu; break;
        case 6: v = (r & 0x807f807fu) | 0x3f803f80u; break;
        case 7: v = (r & 0x807fffffu) | 0x3f800000u; break;
        case 8: v = r & 0xffu; break;
        default: v = r;
        }
        p[i] = v;
    }
}
