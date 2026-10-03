// mma: tensor-core throughput per mma.sync kind, in dense-equivalent FLOP/s and
// FLOP per SM per clock; plus single-warp dependent latency.
#include "common.h"

#include <math.h>

static const struct {
    const char *v;
    double flops; // per warp-level mma, dense-equivalent for sparse
    const char *label;
} k_vars[] = {
    {"f16_f32", 4096, "m16n8k16 f16 -> f32 acc"},
    {"f16_f16", 4096, "m16n8k16 f16 -> f16 acc"},
    {"bf16_f32", 4096, "m16n8k16 bf16 -> f32"},
    {"tf32_f32", 2048, "m16n8k8 tf32 -> f32"},
    {"e4m3_f32", 8192, "m16n8k32 e4m3 -> f32 (sm_89 form)"},
    {"e4m3_f16", 8192, "m16n8k32 e4m3 -> f16 acc"},
    {"f8f6f4_e4m3_f32", 8192, "kind::f8f6f4 e4m3 -> f32"},
    {"f8f6f4_e4m3_f16", 8192, "kind::f8f6f4 e4m3 -> f16 acc"},
    {"f8f6f4_e3m2_f32", 8192, "kind::f8f6f4 e3m2 (FP6) -> f32"},
    {"f8f6f4_e2m1_f32", 8192, "kind::f8f6f4 e2m1 (FP4, 8-bit containers)"},
    {"f8f6f4_e2m1xe4m3_f32", 8192, "kind::f8f6f4 e2m1 x e4m3 (mixed)"},
    {"mxf8f6f4_e4m3", 8192, "kind::mxf8f6f4 e4m3, ue8m0 x1 (MXFP8)"},
    {"mxf8f6f4_e2m1", 8192, "kind::mxf8f6f4 e2m1, ue8m0 x1"},
    {"mxf4_2x", 16384, "kind::mxf4 m16n8k64, ue8m0 x2 (MXFP4)"},
    {"mxf4nvf4_4x", 16384, "kind::mxf4nvf4 m16n8k64, ue4m3 x4 (NVFP4)"},
    {"mxf4nvf4_2x_ue8m0", 16384, "kind::mxf4nvf4 m16n8k64, ue8m0 x2"},
    {"s8_s32", 8192, "m16n8k32 s8 -> s32 (IMMA)"},
    {"s4_s32", 16384, "m16n8k64 s4 -> s32"},
    {"b1_and", 65536, "m16n8k256 b1 and.popc"},
    {"sp_f16_f32", 8192, "2:4 sparse m16n8k32 f16 -> f32"},
    {"sp_e4m3_f32", 16384, "2:4 sparse m16n8k64 e4m3 -> f32"},
    {"sp_f8f6f4_e2m1", 16384, "2:4 sparse kind::f8f6f4 e2m1"},
    {"sp_mxf4_2x", 32768, "2:4 sparse kind::mxf4 m16n8k128"},
    {"sp_mxf4nvf4_4x", 32768, "2:4 sparse kind::mxf4nvf4 m16n8k128"},
};


// ---- accumulator precision (mma num) ------------------------------------------------
static unsigned f16b(int k) { return (unsigned)(k + 15) << 10; }
static unsigned bf16b(int k) { return (unsigned)(k + 127) << 7; }
static unsigned f32b(int k) { return (unsigned)(k + 127) << 23; }
static unsigned e4m3b(int k) { return (unsigned)(k + 7) << 3; }

typedef struct {
    const char *v;
    int kind; // 0 f16-like 16-bit inputs, 1 8-bit, 2 4-bit nvf4 (ue4m3 x4), 3 4-bit mxf4 (ue8m0 x2), 4 s8
    int f16acc, bf16;
} numv_t;

static void num_inputs(const numv_t *nv, int test, int e, unsigned in[7], double *expect_delta) {
    memset(in, 0, 7 * sizeof(unsigned));
    *expect_delta = 3.0;
    unsigned one16 = nv->bf16 ? bf16b(0) : f16b(0);
    switch (nv->kind) {
    case 0:
        in[0] = one16 | (one16 << 16);
        in[2] = one16 | (one16 << 16);
        if (test == 0) {
            in[1] = one16;
            in[3] = one16;
        } else {
            int ea = e / 2, eb = e - e / 2;
            in[1] = (nv->bf16 ? bf16b(ea) : f16b(ea)) | (one16 << 16);
            in[3] = (nv->bf16 ? bf16b(eb) : f16b(eb)) | (one16 << 16);
        }
        break;
    case 1:
        in[0] = 0x00383838u;
        in[2] = 0x00383838u;
        if (test == 1) {
            in[1] = e4m3b(e / 2);
            in[3] = e4m3b(e - e / 2);
        }
        in[5] = in[6] = 0x7f7f7f7fu;
        break;
    case 2:
        in[0] = 0x222u;
        in[2] = 0x222u;
        in[5] = in[6] = 0x38383838u;
        if (test == 1) {
            in[1] = 0x2u;
            in[3] = 0x2u;
            in[5] = (in[5] & ~0x00ff0000u) | (e4m3b(e / 2) << 16);
            in[6] = (in[6] & ~0x00ff0000u) | (e4m3b(e - e / 2) << 16);
        }
        break;
    case 3:
        in[0] = 0x222u;
        in[2] = 0x222u;
        in[5] = in[6] = 0x7f7f7f7fu;
        if (test == 1) {
            in[1] = 0x2u;
            in[3] = 0x2u;
            in[5] = (in[5] & ~0x0000ff00u) | ((unsigned)(127 + e) << 8);
        }
        break;
    case 4:
        in[0] = 0x010101u;
        in[2] = 0x010101u;
        break;
    }
    if (test == 0) {
        if (nv->kind == 4)
            in[4] = 1u << e;
        else
            in[4] = nv->f16acc ? f16b(e) : f32b(e);
    }
}

static int cmd_mma_num(void) {
    static const numv_t nvs[] = {
        {"f16_f32", 0, 0, 0},  {"bf16_f32", 0, 0, 1}, {"f16_f16", 0, 1, 0},   {"e4m3_f32", 1, 0, 0},
        {"e4m3_f16", 1, 1, 0}, {"mxf8f6f4_e4m3", 1, 0, 0}, {"mxf4nvf4_4x", 2, 0, 0}, {"mxf4_2x", 3, 0, 0},
        {"s8_s32", 4, 0, 0},
    };
    CUdeviceptr din, dout;
    CK(cuMemAlloc(&din, 64));
    CK(cuMemAlloc(&dout, 64));
    for (int test = 0; test < 2; test++) {
        printf(test == 0 ? "\n## test 1: D = C + 3 products of 1.0, C = 2^e; printed: D - C\n"
                         : "\n## test 2: D = one product 2^e + three products of 1.0 (C = 0); printed: D - 2^e\n");
        printf("%-15s", "kind \\ e");
        int es[] = {8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 23, 24, 25, 26};
        int ne = sizeof es / sizeof es[0];
        for (int i = 0; i < ne; i++) printf("%6d", es[i]);
        printf("\n");
        for (size_t v = 0; v < sizeof nvs / sizeof nvs[0]; v++) {
            const numv_t *nv = &nvs[v];
            char def[64];
            snprintf(def, sizeof def, "-DV_%s", nv->v);
            const char *opts[] = {def};
            CUmodule m = load_module("mma_num.cu", "sm_120a", opts, 1, 1);
            if (!m) continue;
            CUfunction f = get_fn(m, "k");
            printf("%-15s", nv->v);
            for (int i = 0; i < ne; i++) {
                int e = es[i];
                int maxe = nv->f16acc ? 15 : 30;
                if (test == 1) {
                    if (nv->kind == 0) maxe = nv->bf16 ? 60 : 29;
                    if (nv->kind == 1) maxe = 16;
                    if (nv->kind == 2) maxe = 16;
                    if (nv->kind == 3) maxe = 60;
                    if (nv->kind == 4) maxe = -1;
                    if (nv->f16acc) maxe = maxe < 15 ? maxe : 15;
                }
                if (e > maxe) {
                    printf("%6s", "-");
                    continue;
                }
                unsigned in[7];
                double ex;
                num_inputs(nv, test, e, in, &ex);
                CK(cuMemcpyHtoD(din, in, sizeof in));
                void *args[] = {&din, &dout};
                CK(cuLaunchKernel(f, 1, 1, 1, 32, 1, 1, 0, 0, args, NULL));
                float r;
                CK(cuMemcpyDtoH(&r, dout, 4));
                double big = ldexp(1.0, e);
                printf("%6g", (double)r - big);
            }
            printf("\n");
            cuModuleUnload(m);
        }
    }
    cuMemFree(din);
    cuMemFree(dout);
    return 0;
}

int cmd_mma(int argc, char **argv) {
    if (argc > 1 && !strcmp(argv[1], "num")) return cmd_mma_num();
    const char *only = argc > 1 ? argv[1] : NULL;
    int iters = env_int("MB_ITERS", 4096);
    CUdeviceptr out, clk;
    CK(cuMemAlloc(&out, (size_t)g_sms * 32 * 1024 * 4));
    CK(cuMemAlloc(&clk, 64));
    printf("%-22s %-44s %9s %8s %10s %8s %9s\n", "variant", "instruction", "TFLOPS", "MHz",
           "FLOP/clk/SM", "warps/SM", "lat(clk)");
    for (size_t i = 0; i < sizeof k_vars / sizeof k_vars[0]; i++) {
        if (only && !strstr(k_vars[i].v, only)) continue;
        char def[64];
        snprintf(def, sizeof def, "-DV_%s", k_vars[i].v);
        const char *opts[] = {def};
        CUmodule m = load_module("mma.cu", "sm_120a", opts, 1, 1);
        if (!m) {
            printf("%-22s %-44s %9s\n", k_vars[i].v, k_vars[i].label, "unsupported");
            continue;
        }
        CUfunction f = get_fn(m, "k");
        double best = 0, best_mhz = 0;
        int best_w = 0;
        int warps_per_sm[] = {8, 16, 32};
        for (int w = 0; w < 3; w++) {
            int threads = 128;
            int blocks = g_sms * warps_per_sm[w] / 4;
            void *args[] = {&iters, &out, &clk};
            double mn, md;
            time_launch(f, L1D(blocks, threads, 0), args, 5, 0, &mn, &md);
            double fl = (double)blocks * (threads / 32) * iters * 4 /*NACC*/ * k_vars[i].flops;
            double tf = fl / (mn * 1e-3) / 1e12;
            if (tf > best) {
                best = tf;
                best_mhz = clk_mhz(clk);
                best_w = warps_per_sm[w];
            }
        }
        // latency: one warp, one dependent chain
        double lat = 0;
        {
            char def2[80];
            snprintf(def2, sizeof def2, "-DV_%s", k_vars[i].v);
            const char *o2[] = {def2, "-DNACC=1"};
            CUmodule m1 = load_module("mma.cu", "sm_120a", o2, 2, 1);
            if (m1) {
                CUfunction f1 = get_fn(m1, "k");
                int it = 2048;
                void *args[] = {&it, &out, &clk};
                double mn, md;
                time_launch(f1, L1D(1, 32, 0), args, 3, 0, &mn, &md);
                unsigned long long h[4];
                CK(cuMemcpyDtoH(h, clk, sizeof h));
                lat = (double)(h[1] - h[0]) / it;
                cuModuleUnload(m1);
            }
        }
        printf("%-22s %-44s %9.1f %8.0f %10.0f %8d %9.1f\n", k_vars[i].v, k_vars[i].label, best,
               best_mhz, best * 1e12 / (best_mhz * 1e6) / g_sms, best_w, lat);
        fflush(stdout);
        cuModuleUnload(m);
    }
    cuMemFree(out);
    cuMemFree(clk);
    return 0;
}
