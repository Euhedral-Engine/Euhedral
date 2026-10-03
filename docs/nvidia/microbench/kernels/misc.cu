// Launch-overhead, graph, conditional-node and cluster/DSMEM probes.
#include <cuda_device_runtime_api.h>
typedef unsigned long long u64;

extern "C" __global__ void empty_k(int *p) {
    if (p && threadIdx.x == 1023) p[0] = 1;
}

extern "C" __global__ void empty_pdl(int *p) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    if (p && threadIdx.x == 1023) p[0] = 1;
}

// Body of a WHILE conditional node: count down on the device and stop the loop.
extern "C" __global__ void while_body(cudaGraphConditionalHandle h, int *counter) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int c = --counter[0];
        if (c <= 0) cudaGraphSetConditional(h, 0);
    }
}

// Shared-memory read bandwidth: local vs a peer CTA's shared memory in the cluster.
__device__ __forceinline__ unsigned cluster_rank() {
    unsigned r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
}
__device__ __forceinline__ unsigned cluster_n() {
    unsigned r;
    asm volatile("mov.u32 %0, %%cluster_nctarank;" : "=r"(r));
    return r;
}

extern "C" __global__ void smem_bw(int iters, int remote, unsigned *out, u64 *cycles) {
    __shared__ __align__(16) uint4 buf[2048]; // 32 KiB
    for (int i = threadIdx.x; i < 2048; i += blockDim.x)
        buf[i] = make_uint4(i, i * 3, i * 5, i * 7);
    asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" ::: "memory");
    unsigned base = (unsigned)__cvta_generic_to_shared(buf);
    unsigned addr = base;
    if (remote) {
        unsigned peer = cluster_rank() ^ 1u;
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(addr) : "r"(base), "r"(peer));
    }
    unsigned acc = 0;
    u64 t0 = clock64();
    for (int it = 0; it < iters; it++) {
#pragma unroll
        for (int u = 0; u < 8; u++) {
            unsigned off = ((threadIdx.x + u * blockDim.x) & 2047) * 16;
            unsigned a, b, c, d;
            if (remote)
                asm volatile("ld.shared::cluster.v4.u32 {%0,%1,%2,%3}, [%4];"
                             : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "r"(addr + off));
            else
                asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                             : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "r"(addr + off));
            acc ^= a ^ b ^ c ^ d;
        }
    }
    u64 t1 = clock64();
    asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" ::: "memory");
    out[blockIdx.x * blockDim.x + threadIdx.x] = acc;
    if (threadIdx.x == 0) cycles[blockIdx.x] = t1 - t0;
}
