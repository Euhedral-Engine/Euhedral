// misc: cluster limits and DSMEM bandwidth; kernel launch, PDL, graph and
// conditional-node (device-side loop) overheads.
#include "common.h"

static CUmodule mod(void) {
    static CUmodule m;
    if (!m) m = load_module("misc.cu", "sm_120a", NULL, 0, 0);
    return m;
}

static void clusters(void) {
    printf("## thread-block clusters\n");
    CUfunction f = get_fn(mod(), "smem_bw");
    for (int np = 0; np < 2; np++) {
        if (np) CK(cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, 1));
        CUlaunchConfig cfg;
        memset(&cfg, 0, sizeof cfg);
        cfg.gridDimX = g_sms * 2;
        cfg.gridDimY = cfg.gridDimZ = 1;
        cfg.blockDimX = 256;
        cfg.blockDimY = cfg.blockDimZ = 1;
        int maxc = 0;
        CUresult r = cuOccupancyMaxPotentialClusterSize(&maxc, f, &cfg);
        printf("non-portable allowed=%d: max potential cluster size = %d (r=%d)\n", np, maxc, (int)r);
        for (int cs = 2; cs <= 16; cs *= 2) {
            CUlaunchAttribute at;
            at.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
            at.value.clusterDim.x = cs;
            at.value.clusterDim.y = at.value.clusterDim.z = 1;
            cfg.attrs = &at;
            cfg.numAttrs = 1;
            cfg.gridDimX = 16 * 64;
            int nc = 0;
            r = cuOccupancyMaxActiveClusters(&nc, f, &cfg);
            printf("   cluster %2d: max active clusters %d (%d CTAs)  r=%d\n", cs, nc, nc * cs, (int)r);
        }
    }
    // DSMEM vs local shared memory bandwidth, cluster of 2
    CUdeviceptr out, cyc;
    CK(cuMemAlloc(&out, (size_t)g_sms * 8 * 256 * 4));
    CK(cuMemAlloc(&cyc, (size_t)g_sms * 8 * 8));
    for (int remote = 0; remote < 2; remote++) {
        CUlaunchAttribute at;
        at.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
        at.value.clusterDim.x = 2;
        at.value.clusterDim.y = at.value.clusterDim.z = 1;
        CUlaunchConfig cfg;
        memset(&cfg, 0, sizeof cfg);
        cfg.gridDimX = g_sms * 4;
        cfg.gridDimY = cfg.gridDimZ = 1;
        cfg.blockDimX = 256;
        cfg.blockDimY = cfg.blockDimZ = 1;
        cfg.attrs = &at;
        cfg.numAttrs = 1;
        int iters = 4096;
        void *args[] = {&iters, &remote, &out, &cyc};
        CUevent a, b;
        CK(cuEventCreate(&a, 0));
        CK(cuEventCreate(&b, 0));
        CK(cuLaunchKernelEx(&cfg, f, args, NULL));
        CK(cuEventRecord(a, 0));
        CK(cuLaunchKernelEx(&cfg, f, args, NULL));
        CK(cuEventRecord(b, 0));
        CK(cuEventSynchronize(b));
        float ms;
        CK(cuEventElapsedTime(&ms, a, b));
        unsigned long long c0;
        CK(cuMemcpyDtoH(&c0, cyc, 8));
        double bytes_cta = (double)iters * 8 * 256 * 16;
        printf("%s shared reads: %.1f bytes/clk per CTA (CTA 0: %.0f cycles), aggregate %.0f GB/s\n",
               remote ? "DSMEM (peer CTA)" : "local", bytes_cta / c0, (double)c0,
               bytes_cta * cfg.gridDimX / (ms * 1e-3) / 1e9);
    }
    cuMemFree(out);
    cuMemFree(cyc);
}

static double gpu_ms(CUevent a, CUevent b) {
    float ms;
    CK(cuEventSynchronize(b));
    CK(cuEventElapsedTime(&ms, a, b));
    return ms;
}

static void launches(void) {
    printf("\n## launch overheads (2000 back-to-back tiny kernels, 1 CTA x 128 threads)\n");
    const int N = 2000;
    CUfunction fe = get_fn(mod(), "empty_k"), fp = get_fn(mod(), "empty_pdl");
    CUstream s;
    CK(cuStreamCreate(&s, CU_STREAM_NON_BLOCKING));
    CUevent a, b;
    CK(cuEventCreate(&a, 0));
    CK(cuEventCreate(&b, 0));
    CUdeviceptr nullp = 0;
    void *args[] = {&nullp};
    for (int warm = 0; warm < 2; warm++) {
        // plain stream launches
        double t0 = now_s();
        CK(cuEventRecord(a, s));
        for (int i = 0; i < N; i++) CK(cuLaunchKernel(fe, 1, 1, 1, 128, 1, 1, 0, s, args, NULL));
        CK(cuEventRecord(b, s));
        double host = now_s() - t0;
        double g = gpu_ms(a, b);
        if (warm)
            printf("stream launch:     host %.2f us/launch, GPU %.2f us/kernel\n", host * 1e6 / N, g * 1e3 / N);
        // PDL launches
        CUlaunchAttribute at;
        at.id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
        at.value.programmaticStreamSerializationAllowed = 1;
        CUlaunchConfig cfg;
        memset(&cfg, 0, sizeof cfg);
        cfg.gridDimX = cfg.gridDimY = cfg.gridDimZ = 1;
        cfg.blockDimX = 128;
        cfg.blockDimY = cfg.blockDimZ = 1;
        cfg.hStream = s;
        cfg.attrs = &at;
        cfg.numAttrs = 1;
        t0 = now_s();
        CK(cuEventRecord(a, s));
        for (int i = 0; i < N; i++) CK(cuLaunchKernelEx(&cfg, fp, args, NULL));
        CK(cuEventRecord(b, s));
        host = now_s() - t0;
        g = gpu_ms(a, b);
        if (warm)
            printf("PDL launch:        host %.2f us/launch, GPU %.2f us/kernel\n", host * 1e6 / N, g * 1e3 / N);
        // graph of N kernels (captured), plain and PDL edges
        for (int pdl = 0; pdl < 2; pdl++) {
            CUgraph gr;
            CUgraphExec ge;
            CK(cuStreamBeginCapture(s, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL));
            for (int i = 0; i < N; i++) {
                if (pdl)
                    CK(cuLaunchKernelEx(&cfg, fp, args, NULL));
                else
                    CK(cuLaunchKernel(fe, 1, 1, 1, 128, 1, 1, 0, s, args, NULL));
            }
            CK(cuStreamEndCapture(s, &gr));
            CK(cuGraphInstantiate(&ge, gr, 0));
            CK(cuGraphUpload(ge, s));
            CK(cuGraphLaunch(ge, s));
            CK(cuStreamSynchronize(s));
            t0 = now_s();
            CK(cuEventRecord(a, s));
            CK(cuGraphLaunch(ge, s));
            CK(cuEventRecord(b, s));
            host = now_s() - t0;
            g = gpu_ms(a, b);
            if (warm)
                printf("graph (%s): host %.1f us/graph, GPU %.2f us/kernel node\n",
                       pdl ? "PDL edges" : "plain    ", host * 1e6, g * 1e3 / N);
            cuGraphExecDestroy(ge);
            cuGraphDestroy(gr);
        }
    }
    // WHILE conditional node: the loop runs on the device without host involvement
    {
        CUgraph gr;
        CK(cuGraphCreate(&gr, 0));
        CUgraphConditionalHandle h;
        CK(cuGraphConditionalHandleCreate(&h, gr, g_ctx, 1, CU_GRAPH_COND_ASSIGN_DEFAULT));
        CUgraphNodeParams cp;
        memset(&cp, 0, sizeof cp);
        cp.type = CU_GRAPH_NODE_TYPE_CONDITIONAL;
        cp.conditional.handle = h;
        cp.conditional.type = CU_GRAPH_COND_TYPE_WHILE;
        cp.conditional.size = 1;
        cp.conditional.ctx = g_ctx;
        CUgraphNode cn;
        CK(cuGraphAddNode(&cn, gr, NULL, NULL, 0, &cp));
        CUgraph body = cp.conditional.phGraph_out[0];
        CUdeviceptr counter;
        CK(cuMemAlloc(&counter, 4));
        CUfunction fw = get_fn(mod(), "while_body");
        void *wargs[] = {&h, &counter};
        CUDA_KERNEL_NODE_PARAMS kp;
        memset(&kp, 0, sizeof kp);
        kp.func = fw;
        kp.gridDimX = kp.gridDimY = kp.gridDimZ = 1;
        kp.blockDimX = 128;
        kp.blockDimY = kp.blockDimZ = 1;
        kp.kernelParams = wargs;
        CUgraphNode kn;
        CK(cuGraphAddKernelNode(&kn, body, NULL, 0, &kp));
        CUgraphExec ge;
        CK(cuGraphInstantiate(&ge, gr, 0));
        for (int rep = 0; rep < 2; rep++) {
            int n = N;
            CK(cuMemcpyHtoD(counter, &n, 4));
            CK(cuEventRecord(a, s));
            CK(cuGraphLaunch(ge, s));
            CK(cuEventRecord(b, s));
            double g = gpu_ms(a, b);
            int left;
            CK(cuMemcpyDtoH(&left, counter, 4));
            if (rep) printf("WHILE node: %d iterations, GPU %.2f us/iteration (counter left %d)\n", N, g * 1e3 / N, left);
        }
        cuGraphExecDestroy(ge);
        cuGraphDestroy(gr);
        cuMemFree(counter);
    }
    cuStreamDestroy(s);
}

int cmd_misc(int argc, char **argv) {
    const char *what = argc > 1 ? argv[1] : "all";
    int all = !strcmp(what, "all");
    if (all || !strcmp(what, "cluster")) clusters();
    if (all || !strcmp(what, "launch")) launches();
    return 0;
}
