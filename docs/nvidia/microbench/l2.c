// l2: does persisting L2 (access policy window) work on this GeForce part, how do
// createpolicy hints compare, and is generic memory compression available?
#include "common.h"

static CUmodule mod(void) {
    static CUmodule m;
    if (!m) m = load_module("l2.cu", "sm_120a", NULL, 0, 0);
    return m;
}

static double run_mix(CUdeviceptr hot, size_t hot_b, CUdeviceptr strm, size_t strm_b, int hint,
                      CUstream s, int with_stream) {
    CUfunction f = get_fn(mod(), "hinted_read");
    CUdeviceptr out;
    CK(cuMemAlloc(&out, 64));
    CUevent a, b;
    CK(cuEventCreate(&a, 0));
    CK(cuEventCreate(&b, 0));
    size_t nh = hot_b / 16, ns = strm_b / 16;
    int one = 1, zero = 0;
    double t[16];
    int reps = 12;
    for (int r = 0; r < reps; r++) {
        if (with_stream) {
            void *sa[] = {&strm, &ns, &hint, &zero, &out};
            CK(cuLaunchKernel(f, g_sms * 4, 1, 1, 256, 1, 1, 0, s, sa, NULL));
        }
        void *ha[] = {&hot, &nh, &hint, &one, &out};
        CK(cuEventRecord(a, s));
        CK(cuLaunchKernel(f, g_sms * 4, 1, 1, 256, 1, 1, 0, s, ha, NULL));
        CK(cuEventRecord(b, s));
        CK(cuEventSynchronize(b));
        float ms;
        CK(cuEventElapsedTime(&ms, a, b));
        t[r] = ms;
    }
    // median of the last 10
    for (int i = 2; i < reps; i++)
        for (int j = i + 1; j < reps; j++)
            if (t[j] < t[i]) {
                double x = t[i];
                t[i] = t[j];
                t[j] = x;
            }
    double med = t[2 + (reps - 2) / 2];
    cuMemFree(out);
    cuEventDestroy(a);
    cuEventDestroy(b);
    return hot_b / (med * 1e-3) / 1e9;
}

static void persist(void) {
    int maxp, maxw, l2;
    CK(cuDeviceGetAttribute(&maxp, CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE, g_dev));
    CK(cuDeviceGetAttribute(&maxw, CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE, g_dev));
    CK(cuDeviceGetAttribute(&l2, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, g_dev));
    printf("L2 %d MiB, max persisting %.1f MiB, max window %.1f MiB\n", l2 >> 20, maxp / 1048576.0,
           maxw / 1048576.0);
    size_t strm_b = (size_t)1 << 30;
    CUdeviceptr strm;
    CK(cuMemAlloc(&strm, strm_b));
    CK(cuMemsetD32(strm, 5, strm_b / 4));
    CUstream s;
    CK(cuStreamCreate(&s, CU_STREAM_NON_BLOCKING));
    printf("\n## re-reading a hot buffer after each 1 GiB streaming pass (GB/s of the hot read)\n");
    printf("%8s %12s %12s %12s %14s %12s\n", "hot MiB", "alone(L2)", "plain", "createpolicy",
           "persist window", "window+hint");
    size_t hots[] = {8u << 20, 16u << 20, 24u << 20, 32u << 20};
    for (int h = 0; h < 4; h++) {
        size_t hot_b = hots[h];
        CUdeviceptr hot;
        CK(cuMemAlloc(&hot, hot_b));
        CK(cuMemsetD32(hot, 9, hot_b / 4));
        double alone = run_mix(hot, hot_b, strm, strm_b, 0, s, 0);
        double plain = run_mix(hot, hot_b, strm, strm_b, 0, s, 1);
        double hint = run_mix(hot, hot_b, strm, strm_b, 1, s, 1);
        double win = -1, winh = -1;
        if (maxp > 0) {
            CK(cuCtxSetLimit(CU_LIMIT_PERSISTING_L2_CACHE_SIZE, (size_t)maxp));
            CUstreamAttrValue v;
            memset(&v, 0, sizeof v);
            v.accessPolicyWindow.base_ptr = (void *)hot;
            v.accessPolicyWindow.num_bytes = hot_b < (size_t)maxw ? hot_b : (size_t)maxw;
            double ratio = (double)maxp / v.accessPolicyWindow.num_bytes;
            v.accessPolicyWindow.hitRatio = ratio > 1 ? 1.0f : (float)ratio;
            v.accessPolicyWindow.hitProp = CU_ACCESS_PROPERTY_PERSISTING;
            v.accessPolicyWindow.missProp = CU_ACCESS_PROPERTY_STREAMING;
            CK(cuStreamSetAttribute(s, CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW, &v));
            win = run_mix(hot, hot_b, strm, strm_b, 0, s, 1);
            winh = run_mix(hot, hot_b, strm, strm_b, 1, s, 1);
            memset(&v, 0, sizeof v);
            CK(cuStreamSetAttribute(s, CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW, &v));
            CK(cuCtxResetPersistingL2Cache());
            CK(cuCtxSetLimit(CU_LIMIT_PERSISTING_L2_CACHE_SIZE, 0));
        }
        printf("%8zu %12.1f %12.1f %12.1f %14.1f %12.1f\n", hot_b >> 20, alone, plain, hint, win, winh);
        cuMemFree(hot);
    }
    cuMemFree(strm);
    cuStreamDestroy(s);
}

static void compression(void) {
    int sup;
    CK(cuDeviceGetAttribute(&sup, CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED, g_dev));
    printf("\n## generic (compute data) compression supported: %d\n", sup);
    if (!sup) return;
    CUmemAllocationProp prop;
    memset(&prop, 0, sizeof prop);
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = g_dev;
    prop.allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_GENERIC;
    size_t gran;
    CK(cuMemGetAllocationGranularity(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
    size_t bytes = ((size_t)1 << 30);
    bytes = (bytes + gran - 1) / gran * gran;
    CUmemGenericAllocationHandle hdl;
    CUresult r = cuMemCreate(&hdl, bytes, &prop, 0);
    if (r != CUDA_SUCCESS) {
        printf("cuMemCreate(COMP_GENERIC) failed: %d\n", (int)r);
        return;
    }
    CUmemAllocationProp got;
    CK(cuMemGetAllocationPropertiesFromHandle(&got, hdl));
    printf("granted compressionType = %d (1 = generic)\n", (int)got.allocFlags.compressionType);
    CUdeviceptr p;
    CK(cuMemAddressReserve(&p, bytes, 0, 0, 0));
    CK(cuMemMap(p, bytes, 0, hdl, 0));
    CUmemAccessDesc ad;
    memset(&ad, 0, sizeof ad);
    ad.location = prop.location;
    ad.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CK(cuMemSetAccess(p, bytes, &ad, 1));
    CUdeviceptr q;
    CK(cuMemAlloc(&q, bytes));
    CUmodule mm = load_module("mem.cu", "sm_120a", NULL, 0, 0);
    CUfunction rd = get_fn(mm, "read_v4_u4");
    CUfunction wr = get_fn(mm, "write_v4");
    CUfunction fill = get_fn(mod(), "fill_random");
    CUdeviceptr out, clk;
    CK(cuMemAlloc(&out, 64));
    CK(cuMemAlloc(&clk, 64));
    size_t n = bytes / 16, n32 = bytes / 4;
    printf("%-14s %-8s %10s %10s\n", "allocation", "data", "read GB/s", "write GB/s");
    for (int alloc = 0; alloc < 2; alloc++) {
        CUdeviceptr b = alloc ? q : p;
        for (int data = 0; data < 3; data++) {
            double wgbs = 0;
            if (data == 0) CK(cuMemsetD32(b, 0, n32));
            if (data == 1) {
                unsigned seed = 77;
                void *fa[] = {&b, &n32, &seed};
                CK(cuLaunchKernel(fill, g_sms * 4, 1, 1, 256, 1, 1, 0, 0, fa, NULL));
            }
            if (data == 2) {
                unsigned seed = 0;
                void *wa[] = {&b, &n, &seed, &clk};
                double mn, md;
                time_launch(wr, L1D(g_sms * 4, 256, 0), wa, 3, 0, &mn, &md);
                wgbs = bytes / (mn * 1e-3) / 1e9;
            }
            CK(cuCtxSynchronize());
            void *args[] = {&b, &n, &out, &clk};
            double mn, md;
            time_launch(rd, L1D(g_sms * 4, 256, 0), args, 5, 0, &mn, &md);
            const char *dn[] = {"zeros", "random", "pattern"};
            printf("%-14s %-8s %10.1f %10s\n", alloc ? "cuMemAlloc" : "COMP_GENERIC", dn[data],
                   bytes / (mn * 1e-3) / 1e9, data == 2 ? "" : "-");
            if (data == 2) printf("%-14s %-8s %10s %10.1f\n", "", "(write)", "", wgbs);
        }
    }
    cuMemUnmap(p, bytes);
    cuMemAddressFree(p, bytes);
    cuMemRelease(hdl);
    cuMemFree(q);
}


static void compression_patterns(void) {
    int sup;
    CK(cuDeviceGetAttribute(&sup, CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED, g_dev));
    if (!sup) return;
    CUmemAllocationProp prop;
    memset(&prop, 0, sizeof prop);
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = g_dev;
    prop.allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_GENERIC;
    size_t gran;
    CK(cuMemGetAllocationGranularity(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
    size_t bytes = (((size_t)1 << 30) + gran - 1) / gran * gran;
    CUmemGenericAllocationHandle hdl;
    CK(cuMemCreate(&hdl, bytes, &prop, 0));
    CUdeviceptr p;
    CK(cuMemAddressReserve(&p, bytes, 0, 0, 0));
    CK(cuMemMap(p, bytes, 0, hdl, 0));
    CUmemAccessDesc ad;
    memset(&ad, 0, sizeof ad);
    ad.location = prop.location;
    ad.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CK(cuMemSetAccess(p, bytes, &ad, 1));
    CUmodule mm = load_module("mem.cu", "sm_120a", NULL, 0, 0);
    CUfunction rd = get_fn(mm, "read_v4_u4");
    CUfunction fill = get_fn(mod(), "fill_pattern");
    CUdeviceptr out, clk;
    CK(cuMemAlloc(&out, 64));
    CK(cuMemAlloc(&clk, 64));
    const char *names[] = {"zeros", "constant 1.0f", "50% zero words", "90% zero words", "50% zero 128B lines",
                           "bytes 0..15", "BF16 |x| in [1,2)", "FP32 same exponent", "u32 values 0..255", "random"};
    printf("\n## which data compresses? read GB/s from a COMP_GENERIC allocation (1 GiB, cold)\n");
    size_t n = bytes / 16, n32 = bytes / 4;
    for (int mode = 0; mode < 10; mode++) {
        void *fa[] = {&p, &n32, &mode};
        CK(cuLaunchKernel(fill, g_sms * 4, 1, 1, 256, 1, 1, 0, 0, fa, NULL));
        CK(cuCtxSynchronize());
        void *args[] = {&p, &n, &out, &clk};
        double mn, md;
        time_launch(rd, L1D(g_sms * 4, 256, 0), args, 5, 0, &mn, &md);
        printf("  %-22s %8.1f GB/s\n", names[mode], bytes / (mn * 1e-3) / 1e9);
    }
    cuMemUnmap(p, bytes);
    cuMemAddressFree(p, bytes);
    cuMemRelease(hdl);
    cuMemFree(out);
    cuMemFree(clk);
}

int cmd_l2(int argc, char **argv) {
    const char *what = argc > 1 ? argv[1] : "all";
    int all = !strcmp(what, "all");
    if (all || !strcmp(what, "persist")) persist();
    if (all || !strcmp(what, "compress")) {
        compression();
        compression_patterns();
    }
    return 0;
}
