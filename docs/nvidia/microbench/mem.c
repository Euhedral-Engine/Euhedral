// mem: DRAM/L2 bandwidth, bytes-in-flight curve, TMA bulk copies, latency.
#include "common.h"

static CUmodule mod(void) {
    static CUmodule m;
    if (!m) m = load_module("mem.cu", "sm_120a", NULL, 0, 0);
    return m;
}

static void bw_curve(CUdeviceptr buf, size_t bytes, CUdeviceptr out, CUdeviceptr clk) {
    printf("\n## streaming read, %zu MiB (cold: buffer is %.0fx L2)\n", bytes >> 20,
           bytes / (48.0 * 1048576));
    printf("%-10s %6s %6s %10s %9s %8s %8s\n", "kernel", "blk/SM", "warps", "KB/SM fly", "GB/s",
           "MHz", "B/clk/SM");
    const char *names[] = {"read_v4_u1", "read_v4_u2", "read_v4_u4", "read_v4_u8"};
    int unr[] = {1, 2, 4, 8};
    int bps[] = {1, 2, 4, 6};
    size_t n = bytes / 16;
    for (int k = 0; k < 4; k++) {
        CUfunction f = get_fn(mod(), names[k]);
        for (int b = 0; b < 4; b++) {
            launch_t l = L1D(g_sms * bps[b], 256, 0);
            void *args[] = {&buf, &n, &out, &clk};
            double mn, md;
            time_launch(f, l, args, 7, 0, &mn, &md);
            double mhz = clk_mhz(clk);
            double gbs = bytes / (mn * 1e-3) / 1e9;
            double fly = bps[b] * 256.0 * 16 * unr[k] / 1024.0;
            printf("%-10s %6d %6d %10.0f %9.1f %8.0f %8.2f\n", names[k], bps[b], bps[b] * 8, fly,
                   gbs, mhz, gbs * 1e9 / (mhz * 1e6) / g_sms);
        }
    }
    {
        CUfunction f = get_fn(mod(), "read_u32_u8");
        size_t n32 = bytes / 4;
        void *args[] = {&buf, &n32, &out, &clk};
        for (int b = 2; b < 4; b++) {
            double mn, md;
            time_launch(f, L1D(g_sms * bps[b], 256, 0), args, 7, 0, &mn, &md);
            printf("%-10s %6d %6d %10.0f %9.1f %8.0f   (LDG.32 x8)\n", "read_u32", bps[b],
                   bps[b] * 8, bps[b] * 256.0 * 4 * 8 / 1024, bytes / (mn * 1e-3) / 1e9,
                   clk_mhz(clk));
        }
    }
    {
        CUfunction f = get_fn(mod(), "read_v8_u4");
        size_t n4 = bytes / 16;
        void *args[] = {&buf, &n4, &out, &clk};
        for (int b = 1; b < 4; b++) {
            double mn, md;
            time_launch(f, L1D(g_sms * bps[b], 256, 0), args, 7, 0, &mn, &md);
            printf("%-10s %6d %6d %10.0f %9.1f %8.0f   (LDG.256 x4)\n", "read_v8", bps[b],
                   bps[b] * 8, bps[b] * 256.0 * 32 * 4 / 1024, bytes / (mn * 1e-3) / 1e9,
                   clk_mhz(clk));
        }
    }
    {
        CUfunction w = get_fn(mod(), "write_v4");
        unsigned seed = 7;
        void *args[] = {&buf, &n, &seed, &clk};
        double mn, md;
        time_launch(w, L1D(g_sms * 4, 256, 0), args, 7, 0, &mn, &md);
        printf("%-10s %6d %6d %10s %9.1f %8.0f\n", "write_v4", 4, 32, "-", bytes / (mn * 1e-3) / 1e9,
               clk_mhz(clk));
        CUfunction c = get_fn(mod(), "copy_v4");
        size_t half = n / 2;
        CUdeviceptr dst = buf + half * 16;
        void *cargs[] = {&buf, &dst, &half, &clk};
        time_launch(c, L1D(g_sms * 4, 256, 0), cargs, 7, 0, &mn, &md);
        printf("%-10s %6d %6d %10s %9.1f %8.0f   (read+write bytes)\n", "copy_v4", 4, 32, "-",
               2.0 * half * 16 / (mn * 1e-3) / 1e9, clk_mhz(clk));
    }
}

static void bulk_curve(CUdeviceptr buf, size_t bytes, CUdeviceptr out, CUdeviceptr clk) {
    printf("\n## cp.async.bulk global->shared ring (one issuing thread per CTA), %zu MiB\n",
           bytes >> 20);
    printf("%8s %7s %6s %10s %9s %8s\n", "chunk", "stages", "CTA/SM", "KB/SM fly", "GB/s", "MHz");
    CUfunction f = get_fn(mod(), "read_bulk");
    unsigned chunks[] = {4096, 8192, 16384, 32768};
    unsigned stages[] = {2, 4, 8};
    for (int c = 0; c < 4; c++)
        for (int s = 0; s < 3; s++) {
            unsigned smem = chunks[c] * stages[s];
            if (smem > 99 * 1024) continue;
            int per_sm = (int)((100 * 1024) / (smem + 1024));
            if (per_sm > 4) per_sm = 4;
            if (per_sm < 1) per_sm = 1;
            void *args[] = {&buf, &bytes, &chunks[c], &stages[s], &out, &clk};
            double mn, md;
            time_launch(f, L1D(g_sms * per_sm, 128, smem), args, 5, 0, &mn, &md);
            printf("%8u %7u %6d %10.0f %9.1f %8.0f\n", chunks[c], stages[s], per_sm,
                   per_sm * smem / 1024.0, bytes / (mn * 1e-3) / 1e9, clk_mhz(clk));
        }
}

static void l2_sweep(CUdeviceptr buf, CUdeviceptr out, CUdeviceptr clk) {
    printf("\n## re-read bandwidth vs working set (ld.global.cg, 4 loads in flight/thread)\n");
    printf("%9s %9s %8s %10s\n", "MiB", "GB/s", "MHz", "B/clk/SM");
    CUfunction f = get_fn(mod(), "reread_v4");
    double sizes[] = {1, 2, 4, 8, 16, 24, 32, 40, 44, 48, 56, 64, 96, 128, 256};
    for (int i = 0; i < (int)(sizeof sizes / sizeof sizes[0]); i++) {
        size_t bytes = (size_t)(sizes[i] * 1048576);
        size_t n = bytes / 16;
        int reps = (int)(2048.0 / sizes[i]);
        if (reps < 2) reps = 2;
        void *args[] = {&buf, &n, &reps, &out, &clk};
        double mn, md;
        time_launch(f, L1D(g_sms * 6, 256, 0), args, 5, 0, &mn, &md);
        double gbs = (double)bytes * reps / (mn * 1e-3) / 1e9;
        double mhz = clk_mhz(clk);
        printf("%9.0f %9.1f %8.0f %10.2f\n", sizes[i], gbs, mhz, gbs * 1e3 / mhz / g_sms);
    }
}

static unsigned lcg(unsigned *s) { return *s = *s * 1664525u + 1013904223u; }

static void latency(CUdeviceptr buf, CUdeviceptr out) {
    printf("\n## dependent-load latency (random cycle over 128-byte lines)\n");
    printf("%10s %8s %8s %8s\n", "footprint", "mode", "cycles", "ns@clk");
    CUfunction f = get_fn(mod(), "chase");
    CUdeviceptr cyc;
    CK(cuMemAlloc(&cyc, 8));
    double kb[] = {16, 64, 96, 512, 2048, 3072, 4096, 6144, 8192, 16384, 24576, 40960, 65536, 262144, 1048576};
    for (int i = 0; i < (int)(sizeof kb / sizeof kb[0]); i++) {
        size_t bytes = (size_t)(kb[i] * 1024);
        size_t lines = bytes / 128;
        unsigned *h = calloc(bytes / 4, 4);
        unsigned *perm = malloc(lines * 4);
        for (size_t j = 0; j < lines; j++) perm[j] = (unsigned)j;
        unsigned seed = 12345;
        for (size_t j = lines - 1; j > 0; j--) {
            size_t r = lcg(&seed) % (j + 1);
            unsigned t = perm[j];
            perm[j] = perm[r];
            perm[r] = t;
        }
        for (size_t j = 0; j < lines; j++) h[perm[j] * 32] = perm[(j + 1) % lines] * 32;
        CK(cuMemcpyHtoD(buf, h, bytes));
        for (int mode = 0; mode < 2; mode++) {
            if (mode == 0 && kb[i] > 2048) continue;
            int steps = 20000;
            void *args[] = {&buf, &steps, &mode, &out, &cyc};
            CK(cuLaunchKernel(f, 1, 1, 1, 1, 1, 1, 0, 0, args, NULL));
            CK(cuCtxSynchronize());
            unsigned long long c;
            CK(cuMemcpyDtoH(&c, cyc, 8));
            printf("%8.0fKB %8s %8.0f\n", kb[i], mode ? "cg" : "ca", (double)c / steps);
        }
        free(h);
        free(perm);
    }

    // TLB reach: one 128-byte line per page (L2 footprint stays tiny), pages visited
    // in a random cycle. A latency step at P pages of size S means the TLB level
    // covering the walk holds about P*S bytes.
    printf("\n## TLB reach: one line per page, random page order (cg loads)\n");
    printf("%10s %8s %8s\n", "page", "pages", "cycles");
    size_t pgs[] = {(size_t)64 << 10, (size_t)2 << 20};
    for (int pi = 0; pi < 2; pi++) {
        int counts[] = {8, 16, 32, 64, 128, 256, 512, 1024};
        for (int ci = 0; ci < 8; ci++) {
            size_t page = pgs[pi], np = counts[ci], bytes = page * np;
            if (bytes > ((size_t)env_int("MB_MIB", 2048) << 20)) continue;
            unsigned *h = calloc(bytes / 4, 4);
            unsigned *perm = malloc(np * 4);
            for (size_t j = 0; j < np; j++) perm[j] = (unsigned)j;
            unsigned seed = 777;
            for (size_t j = np - 1; j > 0; j--) {
                size_t r = lcg(&seed) % (j + 1);
                unsigned t = perm[j];
                perm[j] = perm[r];
                perm[r] = t;
            }
            size_t wpp = page / 4;
            for (size_t j = 0; j < np; j++) h[perm[j] * wpp] = (unsigned)(perm[(j + 1) % np] * wpp);
            CK(cuMemcpyHtoD(buf, h, bytes));
            int steps = 20000, mode = 1;
            void *args[] = {&buf, &steps, &mode, &out, &cyc};
            CK(cuLaunchKernel(f, 1, 1, 1, 1, 1, 1, 0, 0, args, NULL));
            CK(cuCtxSynchronize());
            unsigned long long c;
            CK(cuMemcpyDtoH(&c, cyc, 8));
            printf("%9zuK %8zu %8.0f\n", page >> 10, np, (double)c / steps);
            free(h);
            free(perm);
        }
    }
    CUfunction fs = get_fn(mod(), "chase_smem");
    int steps = 20000;
    void *args[] = {&steps, &out, &cyc};
    CK(cuLaunchKernel(fs, 1, 1, 1, 32, 1, 1, 0, 0, args, NULL));
    CK(cuCtxSynchronize());
    unsigned long long c;
    CK(cuMemcpyDtoH(&c, cyc, 8));
    printf("%10s %8s %8.1f\n", "smem", "lds", (double)c / steps);
    cuMemFree(cyc);
}

int cmd_mem(int argc, char **argv) {
    const char *what = argc > 1 ? argv[1] : "all";
    size_t bytes = (size_t)env_int("MB_MIB", 2048) << 20;
    CUdeviceptr buf, out, clk;
    CK(cuMemAlloc(&buf, bytes));
    CK(cuMemAlloc(&out, 64));
    CK(cuMemAlloc(&clk, 64));
    CK(cuMemsetD32(buf, 0x01234567, bytes / 4));
    int all = !strcmp(what, "all");
    if (all || !strcmp(what, "bw")) bw_curve(buf, bytes, out, clk);
    if (all || !strcmp(what, "bulk")) bulk_curve(buf, bytes, out, clk);
    if (all || !strcmp(what, "l2")) l2_sweep(buf, out, clk);
    if (all || !strcmp(what, "lat")) latency(buf, out);
    cuMemFree(buf);
    cuMemFree(out);
    cuMemFree(clk);
    return 0;
}
