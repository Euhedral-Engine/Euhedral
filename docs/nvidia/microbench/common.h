// Shared host helpers for the RTX 5070 Ti microbenchmarks: driver API setup,
// NVRTC compilation of the kernels/ sources, and event timing.
#pragma once
#include <cuda.h>
#include <nvrtc.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CK(x)                                                                    \
    do {                                                                         \
        CUresult r_ = (x);                                                       \
        if (r_ != CUDA_SUCCESS) {                                                \
            const char *s_ = 0;                                                  \
            cuGetErrorString(r_, &s_);                                           \
            fprintf(stderr, "%s:%d %s -> %d %s\n", __FILE__, __LINE__, #x,       \
                    (int)r_, s_ ? s_ : "?");                                     \
            exit(1);                                                             \
        }                                                                        \
    } while (0)

extern CUdevice g_dev;
extern CUcontext g_ctx;
extern int g_sms;
extern const char *g_root; // directory holding kernels/

void bench_init(void);

// Compiles kernels/<name> with NVRTC. arch "sm_120a" yields a cubin, "compute_90"
// yields PTX for the driver JIT. Returns NULL (and prints the log) on failure when
// soft is set, so probes for optional instructions can report "unsupported".
CUmodule load_module(const char *name, const char *arch, const char **opts, int nopts,
                     int soft);
CUfunction get_fn(CUmodule m, const char *name);

typedef struct {
    unsigned gx, gy, gz, bx, by, bz, smem;
} launch_t;

static inline launch_t L1D(unsigned grid, unsigned block, unsigned smem) {
    launch_t l = {grid, 1, 1, block, 1, 1, smem};
    return l;
}

// Median and minimum of reps timed launches in milliseconds (one warmup launch).
void time_launch(CUfunction f, launch_t l, void **args, int reps, CUstream s,
                 double *min_ms, double *med_ms);

// SM clock seen by a kernel: dev_clk[0..1] = clock64 start/end, dev_clk[2..3] =
// globaltimer start/end, written by block 0 thread 0. Returns MHz.
double clk_mhz(CUdeviceptr dev_clk);

double now_s(void);
int env_int(const char *name, int dflt);
