// alu: per-SM per-clock throughput of CUDA-core, SFU and conversion instructions.
#include "common.h"

static const struct {
    const char *v;
    double elems; // results per thread-instruction (packed ops count each lane element)
    double insts; // PTX instructions per OP (multi-instruction probes)
    const char *label;
} k_vars[] = {
    {"ffma", 1, 1, "fma.rn.f32"},
    {"ffma2", 2, 1, "fma.rn.f32x2 (packed FP32)"},
    {"fadd2", 2, 1, "add.rn.f32x2 (packed FP32)"},
    {"hfma2", 2, 1, "fma.rn.f16x2"},
    {"bfma2", 2, 1, "fma.rn.bf16x2"},
    {"fmax", 1, 1, "max.f32"},
    {"imad", 1, 1, "mad.lo.u32"},
    {"iadd", 2, 2, "2x add.u32 (Fibonacci pair)"},
    {"mix_ffma_lop3", 2, 2, "fma.rn.f32 + lop3.b32 (independent)"},
    {"mix_ffma_imad", 2, 2, "fma.rn.f32 + mad.lo.u32 (independent)"},
    {"mix_ffma_ex2", 5, 5, "4x fma.rn.f32 + ex2.approx (independent)"},
    {"lop3", 1, 1, "lop3.b32"},
    {"shf", 1, 1, "shf.l.wrap.b32"},
    {"prmt", 1, 1, "prmt.b32"},
    {"dp4a", 4, 1, "dp4a.u32.u32 (4 MACs)"},
    {"popc", 1, 2, "popc.b32 + xor"},
    {"i2f", 1, 1, "cvt.rn.f32.u32"},
    {"f2i", 1, 1, "cvt.rzi.u32.f32"},
    {"f2f16x2", 2, 1, "cvt.rn.f16x2.f32"},
    {"f2e4m3x2", 2, 2, "cvt.rn.satfinite.e4m3x2.f32 (+widen)"},
    {"e4m3x2_f16x2", 2, 1, "cvt.rn.f16x2.e4m3x2"},
    {"e2m1x2_f16x2", 8, 7, "4x cvt.rn.f16x2.e2m1x2 + 3 xor (8 FP4->FP16)"},
    {"f2e2m1x2", 2, 2, "cvt.rn.satfinite.e2m1x2.f32 (+widen)"},
    {"ex2", 1, 1, "ex2.approx.ftz.f32 (MUFU)"},
    {"rcp", 1, 2, "rcp.approx.ftz.f32 + fmul (MUFU)"},
    {"rsqrt", 1, 1, "rsqrt.approx.ftz.f32 (MUFU)"},
    {"lg2", 1, 1, "lg2.approx.ftz.f32 (MUFU)"},
    {"sin", 1, 1, "sin.approx.ftz.f32 (MUFU)"},
    {"tanh", 1, 1, "tanh.approx.f32 (MUFU)"},
    {"ex2_f16x2", 2, 1, "ex2.approx.f16x2"},
    {"ex2_bf16x2", 2, 1, "ex2.approx.ftz.bf16x2"},
    {"tanh_f16x2", 2, 1, "tanh.approx.f16x2"},
    {"shfl", 1, 1, "shfl.sync.bfly.b32"},
    {"redux", 1, 1, "redux.sync.add.u32"},
    {"dfma", 1, 1, "fma.rn.f64"},
};

int cmd_alu(int argc, char **argv) {
    const char *only = argc > 1 ? argv[1] : NULL;
    int iters = env_int("MB_ITERS", 8192);
    CUdeviceptr out, clk;
    CK(cuMemAlloc(&out, (size_t)g_sms * 2048 * 4));
    CK(cuMemAlloc(&clk, 64));
    int maxthr;
    CK(cuDeviceGetAttribute(&maxthr, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, g_dev));
    printf("%-14s %-46s %9s %8s %12s %12s\n", "variant", "instruction", "Gops/s", "MHz",
           "inst/clk/SM", "elem/clk/SM");
    for (size_t i = 0; i < sizeof k_vars / sizeof k_vars[0]; i++) {
        if (only && strcmp(k_vars[i].v, only)) continue;
        char def[64];
        snprintf(def, sizeof def, "-DV_%s", k_vars[i].v);
        const char *opts[] = {def};
        CUmodule m = load_module("alu.cu", "sm_120a", opts, 1, 1);
        if (!m) {
            printf("%-14s %-46s %9s\n", k_vars[i].v, k_vars[i].label, "unsupported");
            continue;
        }
        CUfunction f = get_fn(m, "k");
        int threads = 256, blocks = g_sms * (maxthr / threads);
        int it = iters;
        if (!strcmp(k_vars[i].v, "dfma")) it = iters / 16;
        void *args[] = {&it, &out, &clk};
        double mn, md;
        time_launch(f, L1D(blocks, threads, 0), args, 5, 0, &mn, &md);
        double mhz = clk_mhz(clk);
        double ops = (double)blocks * threads * it * 8; // CH chains
        double per_clk = ops / (mn * 1e-3) / (mhz * 1e6) / g_sms;
        printf("%-14s %-46s %9.1f %8.0f %12.1f %12.1f\n", k_vars[i].v, k_vars[i].label,
               ops / (mn * 1e-3) / 1e9, mhz, per_clk * k_vars[i].insts, per_clk * k_vars[i].elems);
        fflush(stdout);
        cuModuleUnload(m);
    }
    cuMemFree(out);
    cuMemFree(clk);
    return 0;
}
