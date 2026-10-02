#ifndef _WIN32
#define _GNU_SOURCE
#endif
#include "cuda_kernel_loader.h"
#include <cuda_runtime_api.h>
#include <nvrtc.h>
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

int euhedral_cuda_bind_thread_context(void) {
    int device = 0;
    cudaError_t status = cudaGetDevice(&device);
    if (status == cudaSuccess) status = cudaSetDevice(device);
    return status == cudaSuccess ? EUHEDRAL_CUDA_SUCCESS : (int)status;
}

/* Compiles share/euhedral_cuda/<source_name> with `architecture` and loads `function_name`. A virtual
 * architecture (compute_XX) yields PTX for the driver to JIT; a real one (sm_XX[a]) yields a cubin. */
static int load_kernel(const void* anchor, const char* source_name, const char* function_name,
        CUmodule* module, CUfunction* function, const char* architecture) {
    char library_path[PATH_MAX];
#ifdef _WIN32
    HMODULE owner = NULL;
    DWORD flags = GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT;
    if (!GetModuleHandleExA(flags, (LPCSTR)anchor, &owner)) return EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
    DWORD copied = GetModuleFileNameA(owner, library_path, sizeof(library_path));
    if (copied == 0 || copied >= sizeof(library_path)) return EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
#else
    Dl_info info;
    if (dladdr(anchor, &info) == 0 || info.dli_fname == NULL) return EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
    size_t copied = strlen(info.dli_fname);
    if (copied >= sizeof(library_path)) return EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
    memcpy(library_path, info.dli_fname, copied + 1u);
#endif
    const char* slash = strrchr(library_path, '/');
    const char* backslash = strrchr(library_path, '\\');
    if (backslash != NULL && (slash == NULL || backslash > slash)) slash = backslash;
    if (slash == NULL) return EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
    char path[PATH_MAX];
    int length = snprintf(path, sizeof(path), "%.*s/../share/euhedral_cuda/%s",
            (int)(slash - library_path), library_path, source_name);
    if (length < 0 || (size_t)length >= sizeof(path)) return EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
    FILE* file = fopen(path, "rb");
    if (file == NULL) {
        fprintf(stderr, "Euhedral CUDA: cannot open %s: %s\n", path, strerror(errno));
        return EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
    }
    if (fseek(file, 0, SEEK_END) != 0) { fclose(file); return EUHEDRAL_CUDA_KERNEL_UNAVAILABLE; }
    long size = ftell(file);
    if (size <= 0) { fclose(file); return EUHEDRAL_CUDA_KERNEL_UNAVAILABLE; }
    rewind(file);
    char* source = malloc((size_t)size + 1u);
    if (source == NULL) { fclose(file); return EUHEDRAL_CUDA_KERNEL_UNAVAILABLE; }
    size_t read_size = fread(source, 1, (size_t)size, file);
    fclose(file);
    if (read_size != (size_t)size) { free(source); return EUHEDRAL_CUDA_KERNEL_UNAVAILABLE; }
    source[size] = '\0';

    nvrtcProgram program = NULL;
    nvrtcResult nv_status = nvrtcCreateProgram(&program, source, source_name, 0, NULL, NULL);
    free(source);
    if (nv_status != NVRTC_SUCCESS) return EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
    const char* include_directory = getenv("EUHEDRAL_CUDA_INCLUDE_DIR");
    char include_option[PATH_MAX + 16];
    char home_include[PATH_MAX];
    if (include_directory == NULL || include_directory[0] == '\0') {
        const char* cuda_home = getenv("CUDA_HOME");
        if (cuda_home == NULL || cuda_home[0] == '\0') cuda_home = getenv("CUDA_PATH");
        if (cuda_home != NULL && cuda_home[0] != '\0') {
            int home_length = snprintf(home_include, sizeof(home_include), "%s/include", cuda_home);
            if (home_length < 0 || (size_t)home_length >= sizeof(home_include)) {
                nvrtcDestroyProgram(&program);
                return EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
            }
            include_directory = home_include;
        } else {
            include_directory = "/usr/local/cuda/include";
        }
    }
    int include_length = snprintf(include_option, sizeof(include_option), "-I%s", include_directory);
    if (include_length < 0 || (size_t)include_length >= sizeof(include_option)) {
        nvrtcDestroyProgram(&program);
        return EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
    }
    char source_include_option[PATH_MAX + 16];
    int source_length = snprintf(source_include_option, sizeof(source_include_option), "-I%.*s",
            (int)(strrchr(path, '/') - path), path);
    if (source_length < 0 || (size_t)source_length >= sizeof(source_include_option)) {
        nvrtcDestroyProgram(&program);
        return EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
    }
    char root_include_option[PATH_MAX + 16];
    int root_length = snprintf(root_include_option, sizeof(root_include_option), "-I%.*s",
            (int)(strlen(path) - strlen(source_name) - 1u), path);
    if (root_length < 0 || (size_t)root_length >= sizeof(root_include_option)) {
        nvrtcDestroyProgram(&program);
        return EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
    }
    // Resolve packaged headers relative to this source and to the kernel tree's root (shared
    // headers such as common/pdl.cuh), not to the CUDA toolkit.
    char architecture_option[64];
    int architecture_length = snprintf(architecture_option, sizeof(architecture_option), "--gpu-architecture=%s", architecture);
    if (architecture_length < 0 || (size_t)architecture_length >= sizeof(architecture_option)) {
        nvrtcDestroyProgram(&program);
        return EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
    }
    const int real_architecture = strncmp(architecture, "sm_", 3) == 0;
    const char* options[] = {"--std=c++14", architecture_option, include_option, source_include_option,
            root_include_option};
    nv_status = nvrtcCompileProgram(program, 5, options);
    if (nv_status != NVRTC_SUCCESS) {
        size_t log_size = 0;
        nvrtcGetProgramLogSize(program, &log_size);
        if (log_size > 0) {
            char* log = malloc(log_size);
            if (log != NULL) { nvrtcGetProgramLog(program, log); fprintf(stderr, "%s\n", log); free(log); }
        }
        nvrtcDestroyProgram(&program);
        return EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
    }
    size_t ptx_size = 0;
    nv_status = real_architecture ? nvrtcGetCUBINSize(program, &ptx_size) : nvrtcGetPTXSize(program, &ptx_size);
    char* ptx = nv_status == NVRTC_SUCCESS ? malloc(ptx_size) : NULL;
    if (ptx == NULL || nv_status != NVRTC_SUCCESS
            || (real_architecture ? nvrtcGetCUBIN(program, ptx) : nvrtcGetPTX(program, ptx)) != NVRTC_SUCCESS) {
        free(ptx); nvrtcDestroyProgram(&program); return EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
    }
    nvrtcDestroyProgram(&program);
    CUresult status = cuInit(0);
    CUcontext context = NULL;
    if (status == CUDA_SUCCESS) status = cuCtxGetCurrent(&context);
    if (status == CUDA_SUCCESS && context == NULL) status = CUDA_ERROR_INVALID_CONTEXT;
    if (status == CUDA_SUCCESS) status = cuModuleLoadData(module, ptx);
    free(ptx);
    if (status == CUDA_SUCCESS) status = cuModuleGetFunction(function, *module, function_name);
    if (status != CUDA_SUCCESS && *module != NULL) {
        (void)cuModuleUnload(*module);
        *module = NULL;
    }
    return status == CUDA_SUCCESS ? EUHEDRAL_CUDA_SUCCESS : EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
}

int euhedral_cuda_load_kernel(const void* anchor, const char* source_name, const char* function_name,
        CUmodule* module, CUfunction* function) {
    return load_kernel(anchor, source_name, function_name, module, function, "compute_90");
}

int euhedral_cuda_native_architecture(char* architecture, size_t capacity) {
    CUdevice device;
    int major = 0, minor = 0;
    if (cuInit(0) != CUDA_SUCCESS || cuCtxGetDevice(&device) != CUDA_SUCCESS
            || cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device) != CUDA_SUCCESS
            || cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device) != CUDA_SUCCESS)
        return EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
    // Block-scaled FP4 mma.sync (OMMA.SF) is an arch-specific feature of the sm_12x family.
    if (major != 12) return EUHEDRAL_CUDA_ROUTE_UNAVAILABLE;
    int length = snprintf(architecture, capacity, "sm_%d%da", major, minor);
    return length > 0 && (size_t)length < capacity ? EUHEDRAL_CUDA_SUCCESS : EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
}

int euhedral_cuda_load_native_kernel(const void* anchor, const char* source_name, const char* function_name,
        CUmodule* module, CUfunction* function) {
    char architecture[16];
    int status = euhedral_cuda_native_architecture(architecture, sizeof(architecture));
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    return load_kernel(anchor, source_name, function_name, module, function, architecture);
}

#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

#define PDL_CAPACITY 64
static _Atomic(CUfunction) pdl_functions[PDL_CAPACITY];
static atomic_uint pdl_count;
static atomic_int pdl_mode = -1;
#ifdef _WIN32
static __declspec(thread) int pdl_selected;
#else
static _Thread_local int pdl_selected;
#endif

static atomic_int exact_numerics = -1;

int euhedral_cuda_exact_numerics(void) {
    int value = atomic_load(&exact_numerics);
    if (value < 0) {
        const char* exact = getenv("EUHEDRAL_EXACT");
        const char* q3 = getenv("EUHEDRAL_Q3_DECODE");
        int initial = (exact != NULL && strcmp(exact, "1") == 0) || (q3 != NULL && strcmp(q3, "EXACT") == 0);
        atomic_compare_exchange_strong(&exact_numerics, &value, initial);
        value = atomic_load(&exact_numerics);
    }
    return value;
}

int euhedral_cuda_select_exact_numerics(int exact) {
    int previous = euhedral_cuda_exact_numerics();
    atomic_store(&exact_numerics, exact != 0);
    return previous;
}

/* Decode chains select PDL for the calling thread; every other launch stays ordinary. */
void euhedral_cuda_pdl_select(int enabled) {
    pdl_selected = enabled != 0;
}

void euhedral_cuda_pdl_register(CUfunction function) {
    if (function == NULL) return;
    unsigned int slot = atomic_fetch_add(&pdl_count, 1u);
    if (slot < PDL_CAPACITY) atomic_store(&pdl_functions[slot], function);
}

static int pdl_enabled(void) {
    int mode = atomic_load(&pdl_mode);
    if (mode < 0) {
        const char* value = getenv("EUHEDRAL_PDL");
        mode = value == NULL || strcmp(value, "0") != 0;  /* EUHEDRAL_PDL=0 disables it everywhere */
        atomic_store(&pdl_mode, mode);
    }
    return mode;
}

static int pdl_registered(CUfunction function) {
    unsigned int count = atomic_load(&pdl_count);
    if (count > PDL_CAPACITY) count = PDL_CAPACITY;
    for (unsigned int index = 0; index < count; index++)
        if (atomic_load(&pdl_functions[index]) == function) return 1;
    return 0;
}

CUresult euhedral_launch_kernel(CUfunction function, unsigned int grid_x, unsigned int grid_y, unsigned int grid_z,
        unsigned int block_x, unsigned int block_y, unsigned int block_z, unsigned int shared_bytes,
        CUstream stream, void** parameters, void** extra) {
    if (stream == NULL || extra != NULL || !pdl_selected || !pdl_enabled() || !pdl_registered(function))
        return cuLaunchKernel(function, grid_x, grid_y, grid_z, block_x, block_y, block_z, shared_bytes, stream,
                parameters, extra);
    CUlaunchAttribute attribute;
    memset(&attribute, 0, sizeof(attribute));
    attribute.id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
    attribute.value.programmaticStreamSerializationAllowed = 1;
    CUlaunchConfig config;
    memset(&config, 0, sizeof(config));
    config.gridDimX = grid_x; config.gridDimY = grid_y; config.gridDimZ = grid_z;
    config.blockDimX = block_x; config.blockDimY = block_y; config.blockDimZ = block_z;
    config.sharedMemBytes = shared_bytes;
    config.hStream = stream;
    config.attrs = &attribute;
    config.numAttrs = 1;
    return cuLaunchKernelEx(&config, function, parameters, NULL);
}
