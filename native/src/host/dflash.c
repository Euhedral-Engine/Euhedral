#include "cuda_kernel_loader.h"
#include "euhedral_cuda.h"
#include <cuda_runtime_api.h>
#include <math.h>
#include <stdint.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <pthread.h>
#endif

/* The DFlash2 drafter's operators (src/dflash/kernels.cu). */
#ifdef _WIN32
static INIT_ONCE once = INIT_ONCE_STATIC_INIT;
#else
static pthread_once_t once = PTHREAD_ONCE_INIT;
#endif
static CUmodule module;
static CUfunction linear, linear_rows, rms_norm, conv, context_kv, block_qk, attention, swiglu, topk, select_path;
static int init_status = EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;

/* Dynamic shared memory of the attention kernel: 4 query heads x (2048 + 16) scores, 4 x 128 queries, 8 sums. */
#define ATTENTION_SHARED_BYTES ((4u * (2048u + 16u) + 4u * 128u + 8u) * 4u)

static void initialize(void) {
    init_status = euhedral_cuda_load_kernel(
            (const void*)&once, "dflash/kernels.cu", "euhedral_dflash_linear_bf16", &module, &linear);
    if (init_status != EUHEDRAL_CUDA_SUCCESS) return;
    const char* const names[] = {"euhedral_dflash_linear_rows_bf16", "euhedral_dflash_rms_norm_bf16",
            "euhedral_dflash_conv_bf16", "euhedral_dflash_context_kv_bf16", "euhedral_dflash_block_qk_bf16",
            "euhedral_dflash_attention_bf16", "euhedral_dflash_swiglu_bf16", "euhedral_dflash_topk_bf16",
            "euhedral_dflash_select_bf16"};
    CUfunction* const functions[] = {&linear_rows, &rms_norm, &conv, &context_kv, &block_qk, &attention, &swiglu,
            &topk, &select_path};
    for (int index = 0; index < 9; index++) {
        CUresult status = cuModuleGetFunction(functions[index], module, names[index]);
        if (status != CUDA_SUCCESS) {
            init_status = (int)status;
            return;
        }
    }
    CUfunction all[] = {linear, linear_rows, rms_norm, conv, context_kv, block_qk, attention, swiglu, topk, select_path};
    for (int index = 0; index < 10; index++) euhedral_cuda_pdl_register(all[index]);
}
#ifdef _WIN32
static BOOL CALLBACK initialize_once(PINIT_ONCE state, PVOID parameter, PVOID* context) {
    (void)state; (void)parameter; (void)context;
    initialize();
    return TRUE;
}
#endif

static int ensure_initialized(void) {
    int status = euhedral_cuda_bind_thread_context();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
#ifdef _WIN32
    if (!InitOnceExecuteOnce(&once, initialize_once, NULL, NULL)) return EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
#else
    if (pthread_once(&once, initialize) != 0) return EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
#endif
    return init_status;
}

static int launch(CUfunction function, uint32_t gx, uint32_t gy, uint32_t block, uint32_t shared, void** parameters) {
    CUresult status = euhedral_launch_kernel(
            function, gx, gy, 1, block, 1, 1, shared, euhedral_cuda_submission_stream(), parameters, NULL);
    if (status != CUDA_SUCCESS) return (int)status;
    if (euhedral_cuda_submission_stream() != NULL) return EUHEDRAL_CUDA_SUCCESS;
    cudaError_t sync = cudaDeviceSynchronize();
    return sync == cudaSuccess ? EUHEDRAL_CUDA_SUCCESS : (int)sync;
}

#define PTR(name, value) CUdeviceptr name = (CUdeviceptr)(uintptr_t)(value)

int euhedral_cuda_dflash_linear_bf16(
        const void* input, const void* weights, void* output, uint32_t rows, uint32_t in_features, uint32_t out_features) {
    if (input == NULL || weights == NULL || output == NULL || rows == 0 || in_features == 0 || in_features % 32 != 0
            || out_features == 0 || out_features % 32 != 0)
        return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    int status = ensure_initialized();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    PTR(x, input);
    PTR(w, weights);
    PTR(y, output);
    void* parameters[] = {&x, &w, &y, &rows, &in_features, &out_features};
    /* Up to 16 rows (a draft block, a verification's context rows) stream the weights once per 16 rows; more rows
     * (prefill contexts) reuse each weight load over 64 rows. Both give each row the same bits. */
    if (rows <= 16) return launch(linear, out_features / 32, 1, 128, 0, parameters);
    return launch(linear_rows, out_features / 32, (rows + 63) / 64, 128, 0, parameters);
}

int euhedral_cuda_dflash_rms_norm_bf16(
        const void* input, const void* weight, void* output, uint32_t rows, uint32_t width, float epsilon) {
    if (input == NULL || weight == NULL || output == NULL || rows == 0 || width == 0 || !isfinite(epsilon))
        return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    int status = ensure_initialized();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    PTR(x, input);
    PTR(w, weight);
    PTR(y, output);
    void* parameters[] = {&x, &w, &y, &width, &epsilon};
    return launch(rms_norm, rows, 1, 256, 0, parameters);
}

int euhedral_cuda_dflash_conv_bf16(const void* input, const void* dynamic, const void* base, void* output,
        uint32_t rows, uint32_t width, uint32_t group, uint32_t taps, uint32_t part) {
    if (input == NULL || dynamic == NULL || base == NULL || output == NULL || rows == 0 || width == 0 || group == 0
            || width % group != 0 || taps == 0 || part > 1)
        return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    const uint64_t count = (uint64_t)rows * width;
    if (count > UINT32_MAX) return EUHEDRAL_CUDA_SIZE_OVERFLOW;
    int status = ensure_initialized();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    PTR(x, input);
    PTR(d, dynamic);
    PTR(b, base);
    PTR(y, output);
    void* parameters[] = {&x, &d, &b, &y, &rows, &width, &group, &taps, &part};
    return launch(conv, (uint32_t)((count + 255) / 256), 1, 256, 0, parameters);
}

int euhedral_cuda_dflash_context_kv_bf16(const void* kv, const void* key_norm, void* ring_keys, void* ring_values,
        uint32_t rows, const void* position, uint32_t window, uint32_t key_value_heads, uint32_t head_dim,
        float epsilon, float theta) {
    if (kv == NULL || key_norm == NULL || ring_keys == NULL || ring_values == NULL || position == NULL || rows == 0
            || window == 0 || key_value_heads == 0 || head_dim != 128)
        return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    int status = ensure_initialized();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    PTR(source, kv);
    PTR(norm, key_norm);
    PTR(keys, ring_keys);
    PTR(values, ring_values);
    PTR(at, position);
    void* parameters[] = {&source, &norm, &keys, &values, &at, &window, &key_value_heads, &epsilon, &theta};
    return launch(context_kv, rows, key_value_heads, 128, 0, parameters);
}

int euhedral_cuda_dflash_block_qk_bf16(const void* query, const void* kv, const void* query_norm,
        const void* key_norm, void* query_out, void* key_out, uint32_t rows, const void* position, uint32_t heads,
        uint32_t key_value_heads, uint32_t head_dim, float epsilon, float theta) {
    if (query == NULL || kv == NULL || query_norm == NULL || key_norm == NULL || query_out == NULL || key_out == NULL
            || position == NULL || rows == 0 || heads == 0 || key_value_heads == 0 || head_dim != 128)
        return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    int status = ensure_initialized();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    PTR(q, query);
    PTR(source, kv);
    PTR(qn, query_norm);
    PTR(kn, key_norm);
    PTR(qo, query_out);
    PTR(ko, key_out);
    PTR(at, position);
    void* parameters[] = {&q, &source, &qn, &kn, &qo, &ko, &at, &heads, &key_value_heads, &epsilon, &theta};
    return launch(block_qk, rows, heads + key_value_heads, 128, 0, parameters);
}

int euhedral_cuda_dflash_attention_bf16(const void* query, const void* block_keys, const void* kv,
        const void* ring_keys, const void* ring_values, void* output, uint32_t rows, const void* position,
        uint32_t window, uint32_t heads, uint32_t key_value_heads, uint32_t head_dim) {
    if (query == NULL || block_keys == NULL || kv == NULL || ring_keys == NULL || ring_values == NULL || output == NULL
            || position == NULL || rows == 0 || rows > 16 || window == 0 || window > 2048 || key_value_heads == 0
            || heads % key_value_heads != 0 || heads / key_value_heads > 4 || head_dim != 128)
        return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    int status = ensure_initialized();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    PTR(q, query);
    PTR(bk, block_keys);
    PTR(source, kv);
    PTR(rk, ring_keys);
    PTR(rv, ring_values);
    PTR(y, output);
    PTR(at, position);
    void* parameters[] = {&q, &bk, &source, &rk, &rv, &y, &at, &rows, &window, &heads, &key_value_heads};
    return launch(attention, rows, key_value_heads, 256, ATTENTION_SHARED_BYTES, parameters);
}

int euhedral_cuda_dflash_swiglu_bf16(const void* gate_up, void* output, uint32_t rows, uint32_t intermediate) {
    if (gate_up == NULL || output == NULL || rows == 0 || intermediate == 0) return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    const uint64_t count = (uint64_t)rows * intermediate;
    if (count > UINT32_MAX) return EUHEDRAL_CUDA_SIZE_OVERFLOW;
    int status = ensure_initialized();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    PTR(source, gate_up);
    PTR(y, output);
    void* parameters[] = {&source, &y, &rows, &intermediate};
    return launch(swiglu, (uint32_t)((count + 255) / 256), 1, 256, 0, parameters);
}

int euhedral_cuda_dflash_topk_bf16(
        const void* logits, uint32_t rows, uint32_t vocabulary, void* values, void* indices) {
    if (logits == NULL || values == NULL || indices == NULL || rows == 0 || vocabulary < 16)
        return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    int status = ensure_initialized();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    PTR(source, logits);
    PTR(v, values);
    PTR(i, indices);
    void* parameters[] = {&source, &vocabulary, &v, &i};
    return launch(topk, rows, 1, 256, 0, parameters);
}

int euhedral_cuda_dflash_select_bf16(const void* hidden, const void* values, const void* indices,
        const void* predecessor, const void* successor, const void* anchor, uint32_t positions, uint32_t rank,
        void* tokens, void* scores) {
    if (hidden == NULL || values == NULL || indices == NULL || predecessor == NULL || successor == NULL
            || anchor == NULL || tokens == NULL || scores == NULL || positions == 0 || rank == 0 || rank > 1024)
        return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    int status = ensure_initialized();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    PTR(h, hidden);
    PTR(v, values);
    PTR(i, indices);
    PTR(pred, predecessor);
    PTR(succ, successor);
    PTR(a, anchor);
    PTR(t, tokens);
    PTR(s, scores);
    void* parameters[] = {&h, &v, &i, &pred, &succ, &a, &positions, &rank, &t, &s};
    return launch(select_path, 1, 1, 512, 0, parameters);
}
