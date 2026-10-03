#include "common.h"

#include <time.h>

CUdevice g_dev;
CUcontext g_ctx;
int g_sms;
const char *g_root = ".";

void bench_init(void) {
    CK(cuInit(0));
    CK(cuDeviceGet(&g_dev, 0));
    CK(cuDevicePrimaryCtxRetain(&g_ctx, g_dev));
    CK(cuCtxSetCurrent(g_ctx));
    CK(cuDeviceGetAttribute(&g_sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, g_dev));
    const char *r = getenv("MB_ROOT");
    if (r) g_root = r;
}

static char *slurp(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        exit(1);
    }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *b = malloc(n + 1);
    if (fread(b, 1, n, f) != (size_t)n) exit(1);
    b[n] = 0;
    fclose(f);
    return b;
}

CUmodule load_module(const char *name, const char *arch, const char **opts, int nopts,
                     int soft) {
    char path[1024];
    snprintf(path, sizeof path, "%s/kernels/%s", g_root, name);
    char *src = slurp(path);
    nvrtcProgram p;
    if (nvrtcCreateProgram(&p, src, name, 0, NULL, NULL) != NVRTC_SUCCESS) exit(1);
    const char *inc = getenv("CUDA_INCLUDE_DIR");
    char archopt[64], incopt[1024], cccl[1100];
    snprintf(archopt, sizeof archopt, "--gpu-architecture=%s", arch);
    const char *all[48];
    int n = 0;
    all[n++] = archopt;
    all[n++] = "-std=c++17";
    all[n++] = "-default-device";
    if (inc) {
        snprintf(incopt, sizeof incopt, "-I%s", inc);
        snprintf(cccl, sizeof cccl, "-I%s/cccl", inc);
        all[n++] = incopt;
        all[n++] = cccl;
    }
    for (int i = 0; i < nopts; i++) all[n++] = opts[i];
    nvrtcResult cr = nvrtcCompileProgram(p, n, all);
    size_t logn = 0;
    nvrtcGetProgramLogSize(p, &logn);
    if (cr != NVRTC_SUCCESS) {
        char *log = malloc(logn + 1);
        nvrtcGetProgramLog(p, log);
        fprintf(stderr, "NVRTC %s (%s) failed:\n%.4000s\n", name, arch, log);
        free(log);
        if (soft) return NULL;
        exit(1);
    }
    size_t sz = 0;
    char *img;
    int cubin = strncmp(arch, "sm_", 3) == 0;
    if (cubin) {
        nvrtcGetCUBINSize(p, &sz);
        img = malloc(sz);
        nvrtcGetCUBIN(p, img);
    } else {
        nvrtcGetPTXSize(p, &sz);
        img = malloc(sz);
        nvrtcGetPTX(p, img);
    }
    const char *dump = getenv("MB_DUMP");
    if (dump) {
        char out[1200];
        snprintf(out, sizeof out, "%s/%s.%s.%s", dump, name, arch, cubin ? "cubin" : "ptx");
        FILE *f = fopen(out, "wb");
        if (f) {
            fwrite(img, 1, sz, f);
            fclose(f);
        }
    }
    CUmodule m;
    CUresult lr = cuModuleLoadData(&m, img);
    if (lr != CUDA_SUCCESS) {
        const char *s = 0;
        cuGetErrorString(lr, &s);
        fprintf(stderr, "cuModuleLoadData %s: %s\n", name, s ? s : "?");
        if (soft) return NULL;
        exit(1);
    }
    free(img);
    free(src);
    nvrtcDestroyProgram(&p);
    return m;
}

CUfunction get_fn(CUmodule m, const char *name) {
    CUfunction f;
    CUresult r = cuModuleGetFunction(&f, m, name);
    if (r != CUDA_SUCCESS) {
        fprintf(stderr, "missing kernel %s\n", name);
        exit(1);
    }
    return f;
}

static int cmpd(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

void time_launch(CUfunction f, launch_t l, void **args, int reps, CUstream s,
                 double *min_ms, double *med_ms) {
    CUevent a, b;
    CK(cuEventCreate(&a, CU_EVENT_DEFAULT));
    CK(cuEventCreate(&b, CU_EVENT_DEFAULT));
    if (l.smem > 48 * 1024)
        CK(cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, l.smem));
    CK(cuLaunchKernel(f, l.gx, l.gy, l.gz, l.bx, l.by, l.bz, l.smem, s, args, NULL));
    CK(cuStreamSynchronize(s));
    double t[256];
    if (reps > 256) reps = 256;
    for (int i = 0; i < reps; i++) {
        CK(cuEventRecord(a, s));
        CK(cuLaunchKernel(f, l.gx, l.gy, l.gz, l.bx, l.by, l.bz, l.smem, s, args, NULL));
        CK(cuEventRecord(b, s));
        CK(cuEventSynchronize(b));
        float ms;
        CK(cuEventElapsedTime(&ms, a, b));
        t[i] = ms;
    }
    qsort(t, reps, sizeof(double), cmpd);
    if (min_ms) *min_ms = t[0];
    if (med_ms) *med_ms = t[reps / 2];
    cuEventDestroy(a);
    cuEventDestroy(b);
}

double clk_mhz(CUdeviceptr dev_clk) {
    unsigned long long h[4];
    CK(cuMemcpyDtoH(h, dev_clk, sizeof h));
    double cycles = (double)(h[1] - h[0]);
    double ns = (double)(h[3] - h[2]);
    return ns > 0 ? cycles / ns * 1000.0 : 0;
}

double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int env_int(const char *name, int dflt) {
    const char *v = getenv(name);
    return v ? atoi(v) : dflt;
}
