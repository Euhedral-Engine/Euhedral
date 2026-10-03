// Memory-system probes. Every kernel writes a clock/globaltimer stamp from block 0
// thread 0 so the host can convert time to SM cycles.
typedef unsigned long long u64;

__device__ __forceinline__ u64 gtimer() {
    u64 t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
#define STAMP_BEGIN(clk)                                                    \
    if (clk && blockIdx.x == 0 && threadIdx.x == 0) {                       \
        clk[0] = clock64();                                                 \
        clk[2] = gtimer();                                                  \
    }
#define STAMP_END(clk)                                                      \
    if (clk && blockIdx.x == 0 && threadIdx.x == 0) {                       \
        clk[1] = clock64();                                                 \
        clk[3] = gtimer();                                                  \
    }

// Streaming read, U independent 16-byte loads in flight per thread per step.
#define READ_V4(U)                                                                       \
    extern "C" __global__ void read_v4_u##U(const uint4 *__restrict__ p, size_t n,       \
                                            unsigned *out, u64 *clk) {                   \
        STAMP_BEGIN(clk);                                                                \
        size_t stride = (size_t)gridDim.x * blockDim.x;                                 \
        size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;                       \
        unsigned acc = 0;                                                                \
        for (; i + (U - 1) * stride < n; i += U * stride) {                             \
            uint4 v[U];                                                                  \
            _Pragma("unroll") for (int u = 0; u < U; u++) v[u] = p[i + u * stride];      \
            _Pragma("unroll") for (int u = 0; u < U; u++) acc ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w; \
        }                                                                                \
        for (; i < n; i += stride) {                                                     \
            uint4 v = p[i];                                                              \
            acc ^= v.x ^ v.y ^ v.z ^ v.w;                                                \
        }                                                                                \
        if (acc == 0x9e3779b9u) out[0] = acc;                                            \
        STAMP_END(clk);                                                                  \
    }
READ_V4(1)
READ_V4(2)
READ_V4(4)
READ_V4(8)

// Same with 32-bit loads (4x the LSU instructions per byte).
extern "C" __global__ void read_u32_u8(const unsigned *__restrict__ p, size_t n, unsigned *out,
                                       u64 *clk) {
    STAMP_BEGIN(clk);
    size_t stride = (size_t)gridDim.x * blockDim.x;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned acc = 0;
    for (; i + 7 * stride < n; i += 8 * stride) {
        unsigned v[8];
#pragma unroll
        for (int u = 0; u < 8; u++) v[u] = p[i + u * stride];
#pragma unroll
        for (int u = 0; u < 8; u++) acc ^= v[u];
    }
    if (acc == 0x9e3779b9u) out[0] = acc;
    STAMP_END(clk);
}

// 256-bit loads (ld.global.v8.u32). PTX 8.8+; probes whether sm_120 has LDG.256.
extern "C" __global__ void read_v8_u4(const uint4 *__restrict__ p, size_t n, unsigned *out,
                                      u64 *clk) {
    STAMP_BEGIN(clk);
    size_t n8 = n / 2; // 32-byte elements
    size_t stride = (size_t)gridDim.x * blockDim.x;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned acc = 0;
    const char *base = (const char *)p;
    for (; i + 3 * stride < n8; i += 4 * stride) {
        unsigned v[4][8];
#pragma unroll
        for (int u = 0; u < 4; u++) {
            const char *a = base + (i + u * stride) * 32;
            asm volatile("ld.global.nc.v8.u32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                         : "=r"(v[u][0]), "=r"(v[u][1]), "=r"(v[u][2]), "=r"(v[u][3]),
                           "=r"(v[u][4]), "=r"(v[u][5]), "=r"(v[u][6]), "=r"(v[u][7])
                         : "l"(a));
        }
#pragma unroll
        for (int u = 0; u < 4; u++)
#pragma unroll
            for (int k = 0; k < 8; k++) acc ^= v[u][k];
    }
    if (acc == 0x9e3779b9u) out[0] = acc;
    STAMP_END(clk);
}

// L2-resident re-read: each thread walks its slice `reps` times (every element read
// exactly once per rep; 4 independent loads in flight per thread).
__device__ __forceinline__ uint4 ldcg(const uint4 *a) {
    uint4 v;
    asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(a));
    return v;
}
extern "C" __global__ void reread_v4(const uint4 *__restrict__ p, size_t n, int reps, unsigned *out,
                                     u64 *clk) {
    STAMP_BEGIN(clk);
    size_t stride = (size_t)gridDim.x * blockDim.x;
    unsigned acc = 0;
    for (int r = 0; r < reps; r++) {
        for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += 4 * stride) {
            uint4 v[4];
#pragma unroll
            for (int u = 0; u < 4; u++) {
                size_t j = i + u * stride;
                v[u] = j < n ? ldcg(p + j) : make_uint4(0, 0, 0, 0);
            }
#pragma unroll
            for (int u = 0; u < 4; u++) acc ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
        }
    }
    if (acc == 0x9e3779b9u) out[0] = acc;
    STAMP_END(clk);
}

extern "C" __global__ void write_v4(uint4 *__restrict__ p, size_t n, unsigned seed, u64 *clk) {
    STAMP_BEGIN(clk);
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        p[i] = make_uint4(seed ^ (unsigned)i, seed, (unsigned)i, seed + 1);
    STAMP_END(clk);
}

extern "C" __global__ void copy_v4(const uint4 *__restrict__ s, uint4 *__restrict__ d, size_t n,
                                   u64 *clk) {
    STAMP_BEGIN(clk);
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) d[i] = s[i];
    STAMP_END(clk);
}

// Bulk (TMA-engine, non-tensor) copies global -> shared with an mbarrier ring.
// One elected thread issues the copies; the block only touches one word per
// thread per chunk, so this measures what the copy engine in the SM can pull.
__device__ __forceinline__ unsigned smem_u32(const void *p) {
    return (unsigned)__cvta_generic_to_shared(p);
}
extern "C" __global__ void read_bulk(const char *__restrict__ src, size_t bytes, unsigned chunk,
                                     unsigned stages, unsigned *out, u64 *clk) {
    extern __shared__ __align__(128) char smem[];
    __shared__ __align__(8) u64 bar[16];
    STAMP_BEGIN(clk);
    size_t nchunks = bytes / chunk;
    if (threadIdx.x == 0) {
        for (unsigned s = 0; s < stages; s++)
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(smem_u32(&bar[s])));
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (unsigned s = 0; s < stages; s++) {
            size_t c = blockIdx.x + (size_t)s * gridDim.x;
            if (c >= nchunks) break;
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(
                             smem_u32(&bar[s])),
                         "r"(chunk)
                         : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
                    smem_u32(smem + (size_t)s * chunk)),
                "l"(src + c * chunk), "r"(chunk), "r"(smem_u32(&bar[s]))
                : "memory");
        }
    }
    unsigned acc = 0;
    for (size_t k = 0;; k++) {
        size_t c = blockIdx.x + k * gridDim.x;
        if (c >= nchunks) break;
        unsigned s = k % stages;
        unsigned phase = (k / stages) & 1;
        unsigned done = 0;
        while (!done) {
            asm volatile(
                "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
                : "=r"(done)
                : "r"(smem_u32(&bar[s])), "r"(phase)
                : "memory");
        }
        acc ^= ((const unsigned *)(smem + (size_t)s * chunk))[threadIdx.x];
        __syncthreads();
        if (threadIdx.x == 0) {
            size_t cn = c + (size_t)stages * gridDim.x;
            if (cn < nchunks) {
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(
                                 smem_u32(&bar[s])),
                             "r"(chunk)
                             : "memory");
                asm volatile(
                    "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
                        smem_u32(smem + (size_t)s * chunk)),
                    "l"(src + cn * chunk), "r"(chunk), "r"(smem_u32(&bar[s]))
                    : "memory");
            }
        }
    }
    if (acc == 0x9e3779b9u) out[0] = acc;
    STAMP_END(clk);
}

// L2 prefetch without data return: cp.async.bulk.prefetch.L2 then a normal read.
extern "C" __global__ void prefetch_l2(const char *__restrict__ src, size_t bytes, unsigned chunk) {
    size_t nchunks = bytes / chunk;
    for (size_t c = (size_t)blockIdx.x * blockDim.x + threadIdx.x; c < nchunks;
         c += (size_t)gridDim.x * blockDim.x)
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(src + c * chunk), "r"(chunk)
                     : "memory");
}

// Dependent pointer chase. mode 0: ld.global.ca (L1 allowed), 1: ld.global.cg (L2).
extern "C" __global__ void chase(const unsigned *__restrict__ next, int steps, int mode,
                                 unsigned *out, u64 *cycles) {
    unsigned idx = 0;
    // warm the TLB/caches with one lap of 1/4 of the steps
    for (int i = 0; i < steps / 4; i++) {
        if (mode == 0)
            asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(idx) : "l"(next + idx));
        else
            asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(idx) : "l"(next + idx));
    }
    u64 t0 = clock64();
    for (int i = 0; i < steps; i++) {
        if (mode == 0)
            asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(idx) : "l"(next + idx));
        else
            asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(idx) : "l"(next + idx));
    }
    u64 t1 = clock64();
    out[0] = idx;
    cycles[0] = t1 - t0;
}

extern "C" __global__ void chase_smem(int steps, unsigned *out, u64 *cycles) {
    __shared__ unsigned s[4096];
    for (int i = threadIdx.x; i < 4096; i += blockDim.x) s[i] = (i * 1031 + 17) & 4095;
    __syncthreads();
    if (threadIdx.x) return;
    unsigned idx = 0;
    u64 t0 = clock64();
    for (int i = 0; i < steps; i++) idx = s[idx];
    u64 t1 = clock64();
    out[0] = idx;
    cycles[0] = t1 - t0;
}
