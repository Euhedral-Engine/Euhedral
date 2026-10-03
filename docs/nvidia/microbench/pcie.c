// pcie: copy engines and host memory. H2D/D2H with cudaHostAlloc-style pinning vs a
// THP arena registered with cuMemHostRegister, 1 vs 2 streams, bidirectional,
// SM-driven zero-copy reads, zero-copy + DMA at once, and D2D copy engines vs SMs.
#define _GNU_SOURCE
#include "common.h"

#include <sys/mman.h>

#ifndef MADV_COLLAPSE
#define MADV_COLLAPSE 25
#endif

static CUmodule mod(void) {
    static CUmodule m;
    if (!m) m = load_module("mem.cu", "sm_120a", NULL, 0, 0);
    return m;
}

static void *thp_arena(size_t bytes) {
    void *p = NULL;
    if (posix_memalign(&p, 2u << 20, bytes)) exit(1);
    madvise(p, bytes, MADV_HUGEPAGE);
    memset(p, 1, bytes);
    madvise(p, bytes, MADV_COLLAPSE);
    return p;
}

static long anon_huge_kb(void) {
    FILE *f = fopen("/proc/self/smaps_rollup", "r");
    if (!f) return -1;
    char line[256];
    long kb = -1;
    while (fgets(line, sizeof line, f))
        if (!strncmp(line, "AnonHugePages:", 14)) kb = atol(line + 14);
    fclose(f);
    return kb;
}

static double copy_gbs(CUdeviceptr d, void *h, size_t bytes, size_t chunk, int nstreams, int dir,
                       CUstream *st) {
    CUevent a, b;
    CK(cuEventCreate(&a, 0));
    CK(cuEventCreate(&b, 0));
    double best = 1e30;
    for (int rep = 0; rep < 4; rep++) {
        CK(cuCtxSynchronize());
        CK(cuEventRecord(a, st[0]));
        for (int s = 1; s < nstreams; s++) CK(cuStreamWaitEvent(st[s], a, 0));
        size_t n = bytes / chunk;
        for (size_t i = 0; i < n; i++) {
            CUstream s = st[i % nstreams];
            if (dir == 0)
                CK(cuMemcpyHtoDAsync(d + i * chunk, (char *)h + i * chunk, chunk, s));
            else
                CK(cuMemcpyDtoHAsync((char *)h + i * chunk, d + i * chunk, chunk, s));
        }
        for (int s = 1; s < nstreams; s++) {
            CUevent e;
            CK(cuEventCreate(&e, CU_EVENT_DISABLE_TIMING));
            CK(cuEventRecord(e, st[s]));
            CK(cuStreamWaitEvent(st[0], e, 0));
            cuEventDestroy(e);
        }
        CK(cuEventRecord(b, st[0]));
        CK(cuEventSynchronize(b));
        float ms;
        CK(cuEventElapsedTime(&ms, a, b));
        if (ms < best) best = ms;
    }
    cuEventDestroy(a);
    cuEventDestroy(b);
    return bytes / (best * 1e-3) / 1e9;
}

int cmd_pcie(int argc, char **argv) {
    (void)argc;
    (void)argv;
    size_t bytes = (size_t)1 << 30;
    int ce;
    CK(cuDeviceGetAttribute(&ce, CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT, g_dev));
    printf("async (copy) engines reported: %d\n", ce);
    CUstream st[4];
    for (int i = 0; i < 4; i++) CK(cuStreamCreate(&st[i], CU_STREAM_NON_BLOCKING));
    CUdeviceptr d, d2;
    CK(cuMemAlloc(&d, bytes));
    CK(cuMemAlloc(&d2, bytes));

    // A: driver-pinned (cuMemHostAlloc), B: THP arena + cuMemHostRegister
    void *ha;
    CK(cuMemHostAlloc(&ha, bytes, CU_MEMHOSTALLOC_DEVICEMAP));
    memset(ha, 2, bytes);
    long before = anon_huge_kb();
    void *hb = thp_arena(bytes);
    long after = anon_huge_kb();
    CK(cuMemHostRegister(hb, bytes, CU_MEMHOSTREGISTER_DEVICEMAP));
    printf("THP arena: AnonHugePages grew by %ld MiB of %zu MiB\n", (after - before) / 1024, bytes >> 20);
    void *hc = malloc(bytes); // pageable
    memset(hc, 3, bytes);

    printf("\n## host->device and device->host copies (GB/s, best of 4, 1 GiB)\n");
    printf("%-22s %8s %8s %10s %10s\n", "host memory", "chunk", "streams", "H2D", "D2H");
    struct {
        const char *n;
        void *p;
    } hs[] = {{"cuMemHostAlloc", ha}, {"THP+cuMemHostRegister", hb}, {"pageable malloc", hc}};
    size_t chunks[] = {(size_t)2 << 20, (size_t)16 << 20, (size_t)64 << 20, bytes};
    for (int h = 0; h < 3; h++)
        for (int c = 0; c < 4; c++)
            for (int ns = 1; ns <= 2; ns++) {
                if (chunks[c] == bytes && ns == 2) continue;
                if (h == 2 && (ns == 2 || c < 2)) continue;
                double up = copy_gbs(d, hs[h].p, bytes, chunks[c], ns, 0, st);
                double dn = copy_gbs(d, hs[h].p, bytes, chunks[c], ns, 1, st);
                printf("%-22s %7zuM %8d %10.1f %10.1f\n", hs[h].n, chunks[c] >> 20, ns, up, dn);
            }

    // Bidirectional: H2D on stream 0 while D2H on stream 1
    {
        CUevent a, b, c2;
        CK(cuEventCreate(&a, 0));
        CK(cuEventCreate(&b, 0));
        CK(cuEventCreate(&c2, 0));
        CK(cuCtxSynchronize());
        CK(cuEventRecord(a, st[0]));
        CK(cuStreamWaitEvent(st[1], a, 0));
        CK(cuMemcpyHtoDAsync(d, hb, bytes / 2, st[0]));
        CK(cuMemcpyDtoHAsync((char *)hb + bytes / 2, d2, bytes / 2, st[1]));
        CK(cuEventRecord(c2, st[1]));
        CK(cuStreamWaitEvent(st[0], c2, 0));
        CK(cuEventRecord(b, st[0]));
        CK(cuEventSynchronize(b));
        float ms;
        CK(cuEventElapsedTime(&ms, a, b));
        printf("bidirectional 512 MiB each way (THP arena): %.1f GB/s total (%.1f per direction)\n",
               (double)bytes / (ms * 1e-3) / 1e9, 0.5 * bytes / (ms * 1e-3) / 1e9);
    }

    // SM-driven zero-copy read of host memory, alone and concurrently with a DMA copy
    {
        CUdeviceptr hdev;
        CK(cuMemHostGetDevicePointer(&hdev, hb, 0));
        CUdeviceptr out, clk;
        CK(cuMemAlloc(&out, 64));
        CK(cuMemAlloc(&clk, 64));
        size_t half = bytes / 2, n = half / 16;
        int bps[] = {1, 2, 4, 6};
        printf("\n## zero-copy kernel reads of the pinned THP arena (512 MiB)\n");
        for (int k = 0; k < 2; k++) {
            const char *kn = k ? "read_v4_u8" : "read_v4_u4";
            CUfunction f = get_fn(mod(), kn);
            for (int b = 0; b < 4; b++) {
                void *args[] = {&hdev, &n, &out, &clk};
                double mn, md;
                time_launch(f, L1D(g_sms * bps[b], 256, 0), args, 3, st[0], &mn, &md);
                printf("%-12s %d CTA/SM: %6.1f GB/s\n", kn, bps[b], half / (mn * 1e-3) / 1e9);
            }
        }
        CUfunction f = get_fn(mod(), "read_v4_u8");
        CUevent a, b, c2;
        CK(cuEventCreate(&a, 0));
        CK(cuEventCreate(&b, 0));
        CK(cuEventCreate(&c2, 0));
        for (int cps = 1; cps <= 2; cps++) {
            CK(cuCtxSynchronize());
            CK(cuEventRecord(a, st[0]));
            CK(cuStreamWaitEvent(st[1], a, 0));
            void *args[] = {&hdev, &n, &out, &clk};
            CK(cuLaunchKernel(f, g_sms * cps, 1, 1, 256, 1, 1, 0, st[0], args, NULL));
            CK(cuMemcpyHtoDAsync(d, (char *)hb + half, half, st[1]));
            CK(cuEventRecord(c2, st[1]));
            CK(cuStreamWaitEvent(st[0], c2, 0));
            CK(cuEventRecord(b, st[0]));
            CK(cuEventSynchronize(b));
            float ms;
            CK(cuEventElapsedTime(&ms, a, b));
            printf("zero-copy kernel %d CTA/SM (512 MiB) || DMA H2D (512 MiB): %.1f GB/s combined\n", cps,
                   2.0 * half / (ms * 1e-3) / 1e9);
        }
#if CUDA_VERSION >= 12080
        {
            // H2D through the batch API (always a copy engine)
            CUmemcpyAttributes at;
            memset(&at, 0, sizeof at);
            at.srcAccessOrder = CU_MEMCPY_SRC_ACCESS_ORDER_STREAM;
            at.flags = CU_MEMCPY_FLAG_PREFER_OVERLAP_WITH_COMPUTE;
            at.srcLocHint.type = CU_MEM_LOCATION_TYPE_HOST;
            at.dstLocHint.type = CU_MEM_LOCATION_TYPE_DEVICE;
            enum { NB = 16 };
            CUdeviceptr dsts[NB], srcs[NB];
            size_t sizes[NB], idx[1] = {0};
            for (int i = 0; i < NB; i++) {
                dsts[i] = d + (size_t)i * (bytes / NB);
                srcs[i] = (CUdeviceptr)((char *)hb + (size_t)i * (bytes / NB));
                sizes[i] = bytes / NB;
            }
            CK(cuEventRecord(a, st[0]));
            CUresult r = cuMemcpyBatchAsync(dsts, srcs, sizes, NB, &at, idx, 1, st[0]);
            CK(cuEventRecord(b, st[0]));
            CK(cuEventSynchronize(b));
            float ms;
            CK(cuEventElapsedTime(&ms, a, b));
            printf("cuMemcpyBatchAsync H2D 16 x 64 MiB: result %d, %.1f GB/s\n", (int)r, bytes / (ms * 1e-3) / 1e9);
        }
#endif
        cuMemFree(out);
        cuMemFree(clk);
    }

    // D2D: cuMemcpyDtoDAsync vs a copy kernel, and its effect on a concurrent kernel
    {
        CUevent a, b;
        CK(cuEventCreate(&a, 0));
        CK(cuEventCreate(&b, 0));
        CK(cuEventRecord(a, st[0]));
        CK(cuMemcpyDtoDAsync(d2, d, bytes, st[0]));
        CK(cuEventRecord(b, st[0]));
        CK(cuEventSynchronize(b));
        float ms;
        CK(cuEventElapsedTime(&ms, a, b));
        printf("\ncuMemcpyDtoDAsync 1 GiB: %.1f GB/s (read+write %.1f)\n", bytes / (ms * 1e-3) / 1e9,
               2.0 * bytes / (ms * 1e-3) / 1e9);
#if CUDA_VERSION >= 12080
        CUmemcpyAttributes at;
        memset(&at, 0, sizeof at);
        at.srcAccessOrder = CU_MEMCPY_SRC_ACCESS_ORDER_STREAM;
        at.flags = CU_MEMCPY_FLAG_PREFER_OVERLAP_WITH_COMPUTE;
        at.srcLocHint.type = CU_MEM_LOCATION_TYPE_DEVICE;
        at.dstLocHint.type = CU_MEM_LOCATION_TYPE_DEVICE;
        CUdeviceptr dsts[1] = {d2}, srcs[1] = {d};
        size_t sizes[1] = {bytes}, idx[1] = {0};
        CK(cuEventRecord(a, st[0]));
        CUresult r = cuMemcpyBatchAsync(dsts, srcs, sizes, 1, &at, idx, 1, st[0]);
        CK(cuEventRecord(b, st[0]));
        CK(cuEventSynchronize(b));
        CK(cuEventElapsedTime(&ms, a, b));
        printf("cuMemcpyBatchAsync D2D PREFER_OVERLAP_WITH_COMPUTE: result %d, %.1f GB/s\n", (int)r,
               bytes / (ms * 1e-3) / 1e9);
#endif
    }
    cuMemFreeHost(ha);
    cuMemHostUnregister(hb);
    free(hb);
    free(hc);
    cuMemFree(d);
    cuMemFree(d2);
    return 0;
}
