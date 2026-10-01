#ifndef _WIN32
#define _GNU_SOURCE
#endif

#include "euhedral_cuda.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <nvrtc.h>

#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#include <pthread.h>
#endif

#if !defined(CUDA_VERSION) || CUDA_VERSION < 13010
#error "Euhedral CUDA ABI requires CUDA toolkit 13.1 or newer"
#endif

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

#ifdef _WIN32
static INIT_ONCE q3_embedding_once = INIT_ONCE_STATIC_INIT;
#else
static pthread_once_t q3_embedding_once = PTHREAD_ONCE_INIT;
#endif
static CUmodule q3_embedding_module;
static CUfunction q3_embedding_function;
static int q3_embedding_init_status = EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;

static int native_module_path(char* path, size_t capacity) {
#ifdef _WIN32
    HMODULE module = NULL;
    DWORD flags = GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT;
    if (!GetModuleHandleExA(flags, (LPCSTR)(const void*)&q3_embedding_once, &module)) {
        return 0;
    }
    DWORD path_length = GetModuleFileNameA(module, path, (DWORD) capacity);
    if (path_length == 0 || path_length >= capacity) {
        return 0;
    }
    path[path_length] = '\0';
    return 1;
#else
    Dl_info module_info;
    if (dladdr(&q3_embedding_once, &module_info) == 0 || module_info.dli_fname == NULL) {
        return 0;
    }
    size_t path_length = strlen(module_info.dli_fname);
    if (path_length >= capacity) {
        return 0;
    }
    memcpy(path, module_info.dli_fname, path_length + 1u);
    return 1;
#endif
}

static char* load_q3_embedding_source(void) {
    char module_path[PATH_MAX];
    if (!native_module_path(module_path, sizeof(module_path))) {
        fprintf(stderr, "Euhedral CUDA Q3 embedding: cannot locate native library path\n");
        return NULL;
    }
    const char* last_separator = strrchr(module_path, '/');
    const char* windows_separator = strrchr(module_path, '\\');
    if (windows_separator != NULL && (last_separator == NULL || windows_separator > last_separator)) {
        last_separator = windows_separator;
    }
    if (last_separator == NULL) {
        fprintf(stderr, "Euhedral CUDA Q3 embedding: native library path has no directory\n");
        return NULL;
    }
    size_t directory_length = (size_t)(last_separator - module_path);

    char source_path[PATH_MAX];
    int path_length = snprintf(
            source_path,
            sizeof(source_path),
            "%.*s/../share/euhedral_cuda/embedding/kernels.cu",
            (int) directory_length,
            module_path);
    if (path_length < 0 || (size_t) path_length >= sizeof(source_path)) {
        fprintf(stderr, "Euhedral CUDA Q3 embedding: kernel source path is too long\n");
        return NULL;
    }

    FILE* source_file = fopen(source_path, "rb");
    if (source_file == NULL) {
        fprintf(stderr, "Euhedral CUDA Q3 embedding: cannot open %s: %s\n", source_path, strerror(errno));
        return NULL;
    }
    if (fseek(source_file, 0, SEEK_END) != 0) {
        fprintf(stderr, "Euhedral CUDA Q3 embedding: cannot seek kernel source %s\n", source_path);
        fclose(source_file);
        return NULL;
    }
    long source_length = ftell(source_file);
    if (source_length <= 0 || (uintmax_t) source_length >= SIZE_MAX) {
        fprintf(stderr, "Euhedral CUDA Q3 embedding: kernel source has an invalid size\n");
        fclose(source_file);
        return NULL;
    }
    rewind(source_file);

    char* source = (char*) malloc((size_t) source_length + 1u);
    if (source == NULL) {
        fprintf(stderr, "Euhedral CUDA Q3 embedding: unable to allocate kernel source\n");
        fclose(source_file);
        return NULL;
    }
    size_t bytes_read = fread(source, 1, (size_t) source_length, source_file);
    fclose(source_file);
    if (bytes_read != (size_t) source_length) {
        fprintf(stderr, "Euhedral CUDA Q3 embedding: short read from %s\n", source_path);
        free(source);
        return NULL;
    }
    source[source_length] = '\0';
    return source;
}

static void initialize_q3_embedding_kernel(void) {
    char* source = load_q3_embedding_source();
    if (source == NULL) {
        return;
    }

    nvrtcProgram program = NULL;
    nvrtcResult compiler_status = nvrtcCreateProgram(&program, source, "embedding/kernels.cu", 0, NULL, NULL);
    if (compiler_status != NVRTC_SUCCESS) {
        fprintf(stderr, "Euhedral CUDA Q3 embedding: NVRTC program creation failed: %s\n",
                nvrtcGetErrorString(compiler_status));
        free(source);
        return;
    }

    const char* options[] = {"--std=c++14", "--gpu-architecture=compute_90"};
    compiler_status = nvrtcCompileProgram(program, 2, options);
    if (compiler_status != NVRTC_SUCCESS) {
        size_t log_size = 0;
        (void) nvrtcGetProgramLogSize(program, &log_size);
        char* log = log_size == 0 ? NULL : (char*) malloc(log_size);
        if (log != NULL) {
            (void) nvrtcGetProgramLog(program, log);
            fprintf(stderr, "Euhedral CUDA Q3 embedding NVRTC compilation failed:\n%s\n", log);
            free(log);
        } else {
            fprintf(stderr, "Euhedral CUDA Q3 embedding: NVRTC compilation failed: %s\n",
                    nvrtcGetErrorString(compiler_status));
        }
        (void) nvrtcDestroyProgram(&program);
        free(source);
        return;
    }

    size_t ptx_size = 0;
    compiler_status = nvrtcGetPTXSize(program, &ptx_size);
    char* ptx = compiler_status == NVRTC_SUCCESS ? (char*) malloc(ptx_size) : NULL;
    if (compiler_status != NVRTC_SUCCESS || ptx == NULL) {
        fprintf(stderr, "Euhedral CUDA Q3 embedding: unable to allocate NVRTC PTX output\n");
        (void) nvrtcDestroyProgram(&program);
        free(source);
        return;
    }
    compiler_status = nvrtcGetPTX(program, ptx);
    (void) nvrtcDestroyProgram(&program);
    free(source);
    if (compiler_status != NVRTC_SUCCESS) {
        fprintf(stderr, "Euhedral CUDA Q3 embedding: NVRTC PTX retrieval failed: %s\n",
                nvrtcGetErrorString(compiler_status));
        free(ptx);
        return;
    }

    CUresult driver_status = cuInit(0);
    CUcontext context = NULL;
    if (driver_status == CUDA_SUCCESS) {
        driver_status = cuCtxGetCurrent(&context);
    }
    if (driver_status == CUDA_SUCCESS && context == NULL) {
        driver_status = CUDA_ERROR_INVALID_CONTEXT;
    }
    if (driver_status == CUDA_SUCCESS) {
        driver_status = cuModuleLoadData(&q3_embedding_module, ptx);
    }
    free(ptx);
    if (driver_status == CUDA_SUCCESS) {
        driver_status = cuModuleGetFunction(
                &q3_embedding_function, q3_embedding_module, "euhedral_q3_embedding");
    }
    if (driver_status != CUDA_SUCCESS) {
        const char* message = NULL;
        (void) cuGetErrorString(driver_status, &message);
        fprintf(stderr, "Euhedral CUDA Q3 embedding: driver initialization failed: %s\n",
                message == NULL ? "unknown CUDA driver error" : message);
        if (q3_embedding_module != NULL) {
            (void) cuModuleUnload(q3_embedding_module);
            q3_embedding_module = NULL;
        }
        return;
    }
    q3_embedding_init_status = EUHEDRAL_CUDA_SUCCESS;
}

#ifdef _WIN32
static BOOL CALLBACK initialize_q3_embedding_once(PINIT_ONCE once, PVOID parameter, PVOID* context) {
    (void) once;
    (void) parameter;
    (void) context;
    initialize_q3_embedding_kernel();
    return TRUE;
}
#endif

static int ensure_q3_embedding_kernel(void) {
#ifdef _WIN32
    if (!InitOnceExecuteOnce(&q3_embedding_once, initialize_q3_embedding_once, NULL, NULL)) {
        return EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
    }
#else
    if (pthread_once(&q3_embedding_once, initialize_q3_embedding_kernel) != 0) {
        return EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
    }
#endif
    return q3_embedding_init_status;
}

static int checked_multiply_u64(uint64_t left, uint64_t right, uint64_t* result) {
    if (right != 0 && left > UINT64_MAX / right) {
        return 0;
    }
    *result = left * right;
    return 1;
}

static int q3_embedding_geometry(
        uint32_t vocabulary_size,
        uint32_t hidden_size,
        uint64_t* scale_offset,
        uint64_t* expected_byte_size) {
    uint64_t padded_hidden = (((uint64_t) hidden_size + 127u) / 128u) * 128u;
    uint64_t groups_per_row = padded_hidden / 64u;
    uint64_t group_count = 0;
    uint64_t base_byte_size = 0;
    uint64_t scale_byte_size = 0;
    if (!checked_multiply_u64(vocabulary_size, groups_per_row, &group_count)
            || !checked_multiply_u64(group_count, 24u, &base_byte_size)
            || !checked_multiply_u64(group_count, 2u, &scale_byte_size)
            || base_byte_size > UINT64_MAX - 255u) {
        return 0;
    }
    *scale_offset = (base_byte_size + 255u) & ~255ull;
    if (*scale_offset > UINT64_MAX - scale_byte_size) {
        return 0;
    }
    *expected_byte_size = *scale_offset + scale_byte_size;
    return 1;
}

int euhedral_cuda_embed_q3(
        const int32_t* device_token_ids,
        const void* device_embedding_weights,
        void* device_hidden_state,
        uint32_t token_count,
        uint32_t vocabulary_size,
        uint32_t hidden_size,
        uint64_t embedding_byte_size) {
    if (device_token_ids == NULL || device_embedding_weights == NULL || device_hidden_state == NULL
            || token_count == 0 || vocabulary_size == 0 || hidden_size == 0 || hidden_size % 64u != 0) {
        return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    }

    uint64_t scale_offset = 0;
    uint64_t expected_byte_size = 0;
    if (!q3_embedding_geometry(vocabulary_size, hidden_size, &scale_offset, &expected_byte_size)) {
        return EUHEDRAL_CUDA_SIZE_OVERFLOW;
    }
    if (embedding_byte_size != expected_byte_size) {
        return EUHEDRAL_CUDA_FORMAT_MISMATCH;
    }

    uint64_t groups_per_token = hidden_size / 64u;
    uint64_t groups_per_block = (groups_per_token + 3u) / 4u;
    uint64_t grid_size = (uint64_t) token_count * groups_per_block;
    if (grid_size == 0 || grid_size > 2147483647u) {
        return EUHEDRAL_CUDA_SIZE_OVERFLOW;
    }

    int kernel_status = ensure_q3_embedding_kernel();
    if (kernel_status != EUHEDRAL_CUDA_SUCCESS) {
        return kernel_status;
    }

    CUdeviceptr token_ids = (CUdeviceptr) (uintptr_t) device_token_ids;
    CUdeviceptr weights = (CUdeviceptr) (uintptr_t) device_embedding_weights;
    CUdeviceptr output = (CUdeviceptr) (uintptr_t) device_hidden_state;
    unsigned int token_count_argument = token_count;
    unsigned int vocabulary_size_argument = vocabulary_size;
    unsigned int hidden_size_argument = hidden_size;
    unsigned long long scale_offset_argument = (unsigned long long) scale_offset;
    void* kernel_parameters[] = {
        &token_ids,
        &weights,
        &output,
        &token_count_argument,
        &vocabulary_size_argument,
        &hidden_size_argument,
        &scale_offset_argument,
    };

    CUresult status = cuLaunchKernel(
            q3_embedding_function,
            (unsigned int) grid_size,
            1,
            1,
            128,
            1,
            1,
            0,
            euhedral_cuda_submission_stream(),
            kernel_parameters,
            NULL);
    return status == CUDA_SUCCESS ? EUHEDRAL_CUDA_SUCCESS : (int) status;
}
