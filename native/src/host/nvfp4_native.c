#include "cuda_kernel_loader.h"
#include <cuda_runtime_api.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <pthread.h>
#endif

/* Native Blackwell NVFP4 (native/src/nvfp4_native, docs/kernels/NVFP4_NATIVE.md): the BF16 activations are
 * quantized to two NVFP4 terms (the value and its quantized residual) into caller scratch, then multiplied
 * with block-scaled FP4 tensor-core MMA (OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X). The module is compiled for
 * this device's sm_12xa target and exists only there. The Rust engine launches the kernels itself; the library
 * answers whether the route runs on this device and how much scratch a linear needs. */
static uint64_t align256(uint64_t value) { return (value + 255u) & ~255ull; }

#define NATIVE_SHARED_BYTES 85040u  /* nvfp4n::Pipeline<2>::kSharedBytes */
#define SKINNY_MAX_ROWS 64u
#define SKINNY_COLUMNS 64u

/* CTAs the skinny kernel's K split aims for: two per streaming multiprocessor, from the first device queried and kept
 * after that (the library drives one device). A failed query is the caller's error and caches nothing. */
static int skinny_target_ctas(uint32_t* target_ctas) {
    static _Atomic uint32_t target = 0;
    uint32_t known = atomic_load_explicit(&target, memory_order_relaxed);
    if (known == 0u) {
        int32_t sms = 0;
        int status = euhedral_cuda_multiprocessor_count(&sms);
        if (status != EUHEDRAL_CUDA_SUCCESS) return status;
        if (sms <= 0) return EUHEDRAL_CUDA_INVALID_ARGUMENT;
        known = 2u * (uint32_t)sms;
        atomic_store_explicit(&target, known, memory_order_relaxed);
    }
    *target_ctas = known;
    return EUHEDRAL_CUDA_SUCCESS;
}

/* nvfp4n::Skinny<terms, rows / 16>::kSharedBytes: stages of (A rows + 64 weight rows) x 160 bytes. */
static uint32_t skinny_shared_bytes(uint32_t terms, uint32_t rows) {
    uint32_t stage = (terms * rows + SKINNY_COLUMNS) * 160u, stages = 101376u / stage;
    return (stages > 4u ? 4u : stages) * stage;
}

/* K splits for the skinny kernel: enough CTAs to keep every SM streaming weights. */
static uint32_t skinny_splits(uint32_t in_features, uint32_t out_features, uint32_t target_ctas) {
    uint32_t tiles = (out_features + SKINNY_COLUMNS - 1u) / SKINNY_COLUMNS, k_tiles = in_features / 256u;
    uint32_t splits = (target_ctas + tiles - 1u) / tiles, limit = k_tiles / 2u;
    if (splits > limit) splits = limit;
    return splits == 0u ? 1u : splits;
}

static int skinny_route(uint32_t rows, uint32_t in_features) {
    return rows <= SKINNY_MAX_ROWS && in_features % 256u == 0;
}
#ifdef _WIN32
static INIT_ONCE native_once = INIT_ONCE_STATIC_INIT;
#else
static pthread_once_t native_once = PTHREAD_ONCE_INIT;
#endif
static CUmodule native_module;
static CUfunction native_quantize, native_finish;
static CUfunction native_linear[2], native_gate_up[2];  /* [layout] */
static CUfunction native_skinny[2][3];  /* [layout][up to 16, 32 and 64 rows] */
static int native_status = EUHEDRAL_CUDA_ROUTE_UNAVAILABLE;
/* Activation terms: the residual term brings the error per linear to about 1%, the relaxed-numerics floor
 * of the drift harness (docs/kernels/NVFP4_NATIVE.md). */
#define NATIVE_TERMS 2u

static void initialize_native(void) {
    int status = euhedral_cuda_load_native_kernel((const void*)&native_once, "nvfp4_native/kernels.cu",
            "euhedral_nvfp4n_linear_128x128", &native_module, &native_linear[0]);
    if (status == EUHEDRAL_CUDA_SUCCESS
            && (cuModuleGetFunction(&native_quantize, native_module, "euhedral_nvfp4n_quantize_rows") != CUDA_SUCCESS
                    || cuModuleGetFunction(&native_finish, native_module, "euhedral_nvfp4n_skinny_finish") != CUDA_SUCCESS))
        status = EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
    static const char* skinny_names[3] = {"euhedral_nvfp4n_skinny_16", "euhedral_nvfp4n_skinny_32", "euhedral_nvfp4n_skinny_64"};
    for (int layout = 0; layout < 2 && status == EUHEDRAL_CUDA_SUCCESS; ++layout) {
        const char* suffix = layout ? "_sd4" : "";
        char name[64];
        snprintf(name, sizeof(name), "euhedral_nvfp4n_linear_128x128%s", suffix);
        if (cuModuleGetFunction(&native_linear[layout], native_module, name) != CUDA_SUCCESS) status = EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
        snprintf(name, sizeof(name), "euhedral_nvfp4n_gate_up_swiglu_128x64%s", suffix);
        if (status == EUHEDRAL_CUDA_SUCCESS && cuModuleGetFunction(&native_gate_up[layout], native_module, name) != CUDA_SUCCESS)
            status = EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
        for (int i = 0; i < 3 && status == EUHEDRAL_CUDA_SUCCESS; ++i) {
            snprintf(name, sizeof(name), "%s%s", skinny_names[i], suffix);
            if (cuModuleGetFunction(&native_skinny[layout][i], native_module, name) != CUDA_SUCCESS
                    || cuFuncSetAttribute(native_skinny[layout][i], CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                               (int)skinny_shared_bytes(NATIVE_TERMS, 16u << i)) != CUDA_SUCCESS)
                status = EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
        }
        if (status == EUHEDRAL_CUDA_SUCCESS
                && (cuFuncSetAttribute(native_linear[layout], CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                            (int)NATIVE_SHARED_BYTES) != CUDA_SUCCESS
                        || cuFuncSetAttribute(native_gate_up[layout], CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                   (int)NATIVE_SHARED_BYTES) != CUDA_SUCCESS))
            status = EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
    }
    native_status = status;
}

#ifdef _WIN32
static BOOL CALLBACK initialize_native_once(PINIT_ONCE state, PVOID parameter, PVOID* context) {
    (void)state; (void)parameter; (void)context;
    initialize_native();
    return TRUE;
}
#endif

static int ensure_native(void) {
    // The capability check and module load need this thread's device context.
    int status = euhedral_cuda_bind_thread_context();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
#ifdef _WIN32
    if (!InitOnceExecuteOnce(&native_once, initialize_native_once, NULL, NULL)) return EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
#else
    if (pthread_once(&native_once, initialize_native) != 0) return EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
#endif
    return native_status;
}

int euhedral_cuda_nvfp4_native_available(void) {
    return ensure_native() == EUHEDRAL_CUDA_SUCCESS;
}

/* Activations of `rows` rows of `in_features` values in the selected number of terms: codes, the tile-major
 * scales (rows rounded up to 128 per tile and term, 8 bytes per row) and one FP32 global per row
 * (nvfp4n::ActivationLayout). */
static uint64_t activation_bytes(uint32_t rows, uint32_t in_features) {
    uint64_t k = ((uint64_t)in_features + 127u) / 128u * 128u, planes = (uint64_t)rows * NATIVE_TERMS;
    uint64_t pad = ((uint64_t)rows + 127u) / 128u * 128u;
    uint64_t scales = align256(planes * k / 2u);
    return align256(scales + k / 128u * NATIVE_TERMS * pad * 8u) + 4ull * rows;
}

/* The scratch of one native linear whose skinny grid aims for `target_ctas`. */
static uint64_t native_scratch_bytes(uint32_t rows, uint32_t in_features, uint32_t out_features, uint32_t target_ctas) {
    uint64_t bytes = activation_bytes(rows, in_features);
    const uint32_t splits = skinny_splits(in_features, out_features, target_ctas);
    if (skinny_route(rows, in_features) && splits > 1u) bytes = align256(bytes) + 4ull * splits * rows * out_features;
    return bytes;
}

/* Scratch for one native linear: the activations, then (split-K skinny shapes) FP32 partials. UINT64_MAX when the
 * device's multiprocessor count cannot be read, which no scratch satisfies. */
uint64_t euhedral_cuda_nvfp4_native_scratch_bytes(uint32_t rows, uint32_t in_features, uint32_t out_features) {
    uint32_t target_ctas = 0;
    if (skinny_target_ctas(&target_ctas) != EUHEDRAL_CUDA_SUCCESS) return UINT64_MAX;
    return native_scratch_bytes(rows, in_features, out_features, target_ctas);
}
