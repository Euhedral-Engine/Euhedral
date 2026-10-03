#include "submission.h"
#include "cuda_kernel_loader.h"

#include <stdint.h>
#include <string.h>

enum submission_mode { SUBMIT_ORDINARY, SUBMIT_RECORDING, SUBMIT_CHECKING };

#ifdef _WIN32
#define THREAD_LOCAL __declspec(thread)
#else
#define THREAD_LOCAL _Thread_local
#endif

static THREAD_LOCAL int mode;
static THREAD_LOCAL cudaStream_t shadow;
static THREAD_LOCAL int shadow_overlap;
static THREAD_LOCAL int shadow_failed;
static THREAD_LOCAL uint64_t hash;
static THREAD_LOCAL cudaStream_t sink;

/* Kernel parameter sizes, per thread: a direct-mapped cache of each function's parameter layout. */
#define PARAMETER_CACHE 256
#define MAX_PARAMETERS 48
struct parameter_layout {
    CUfunction function;
    unsigned int count;
    unsigned short sizes[MAX_PARAMETERS];
};
static THREAD_LOCAL struct parameter_layout layouts[PARAMETER_CACHE];

static const struct parameter_layout* layout_of(CUfunction function) {
    struct parameter_layout* entry = &layouts[((uintptr_t)function >> 4) % PARAMETER_CACHE];
    if (entry->function == function) return entry;
    entry->function = function;
    entry->count = 0;
    for (size_t index = 0; index < MAX_PARAMETERS; index++) {
        size_t offset = 0, size = 0;
        if (cuFuncGetParamInfo(function, index, &offset, &size) != CUDA_SUCCESS) break;
        entry->sizes[index] = (unsigned short)size;
        entry->count++;
    }
    return entry;
}

static void mix(uint64_t value) {
    uint64_t x = hash ^ (value + 0x9e3779b97f4a7c15ull + (hash << 6) + (hash >> 2));
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ull;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebull;
    hash = x ^ (x >> 31);
}

static void mix_bytes(const void* bytes, size_t count) {
    const unsigned char* data = (const unsigned char*)bytes;
    while (count >= 8) {
        uint64_t word;
        memcpy(&word, data, 8);
        mix(word);
        data += 8;
        count -= 8;
    }
    uint64_t tail = 0;
    memcpy(&tail, data, count);
    mix(tail ^ ((uint64_t)count << 56));
}

static void record_failure(int failed) {
    if (failed) shadow_failed = 1;
}

int euhedral_submission_checking(void) {
    return mode == SUBMIT_CHECKING;
}

int euhedral_submission_unrepeatable(void) {
    if (mode == SUBMIT_RECORDING) shadow_failed = 1;
    if (mode != SUBMIT_CHECKING) return 0;
    mix(0x0dd5eedull ^ hash);
    hash ^= 0x8000000000000000ull;
    return 1;
}

static CUresult launch(CUfunction function, unsigned int grid_x, unsigned int grid_y, unsigned int grid_z,
        unsigned int block_x, unsigned int block_y, unsigned int block_z, unsigned int shared_bytes, CUstream stream,
        void** parameters, int programmatic) {
    if (!programmatic)
        return cuLaunchKernel(function, grid_x, grid_y, grid_z, block_x, block_y, block_z, shared_bytes, stream,
                parameters, NULL);
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

CUresult euhedral_submit_kernel(CUfunction function, unsigned int grid_x, unsigned int grid_y, unsigned int grid_z,
        unsigned int block_x, unsigned int block_y, unsigned int block_z, unsigned int shared_bytes,
        CUstream stream, void** parameters, int programmatic, int overlappable) {
    if (mode != SUBMIT_ORDINARY) {
        const struct parameter_layout* layout = layout_of(function);
        mix((uint64_t)(uintptr_t)function);
        mix(((uint64_t)grid_x << 32) | grid_y);
        mix(((uint64_t)grid_z << 32) | block_x);
        mix(((uint64_t)block_y << 32) | block_z);
        mix(((uint64_t)shared_bytes << 8) | layout->count);
        for (unsigned int index = 0; index < layout->count; index++)
            mix_bytes(parameters[index], layout->sizes[index]);
        if (mode == SUBMIT_CHECKING) return CUDA_SUCCESS;
    }
    CUresult status = launch(function, grid_x, grid_y, grid_z, block_x, block_y, block_z, shared_bytes, stream,
            parameters, programmatic);
    if (mode == SUBMIT_RECORDING && status == CUDA_SUCCESS && !shadow_failed)
        record_failure(launch(function, grid_x, grid_y, grid_z, block_x, block_y, block_z, shared_bytes,
                (CUstream)shadow, parameters, programmatic || (shadow_overlap && overlappable)) != CUDA_SUCCESS);
    return status;
}

cudaError_t euhedral_submit_copy(void* destination, const void* source, size_t bytes, enum cudaMemcpyKind kind,
        cudaStream_t stream) {
    if (mode != SUBMIT_ORDINARY) {
        mix(0xc0b1ull);
        mix((uint64_t)(uintptr_t)destination);
        mix((uint64_t)(uintptr_t)source);
        mix(((uint64_t)bytes << 4) | (uint64_t)kind);
        if (mode == SUBMIT_CHECKING) return cudaSuccess;
    }
    cudaError_t status = cudaMemcpyAsync(destination, source, bytes, kind, stream);
    if (mode == SUBMIT_RECORDING && status == cudaSuccess && !shadow_failed)
        record_failure(cudaMemcpyAsync(destination, source, bytes, kind, shadow) != cudaSuccess);
    return status;
}

cudaError_t euhedral_submit_copy_2d(void* destination, size_t destination_pitch, const void* source,
        size_t source_pitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream) {
    if (mode != SUBMIT_ORDINARY) {
        mix(0xc0b2ull);
        mix((uint64_t)(uintptr_t)destination);
        mix((uint64_t)destination_pitch);
        mix((uint64_t)(uintptr_t)source);
        mix((uint64_t)source_pitch);
        mix((uint64_t)width);
        mix(((uint64_t)height << 4) | (uint64_t)kind);
        if (mode == SUBMIT_CHECKING) return cudaSuccess;
    }
    cudaError_t status = cudaMemcpy2DAsync(destination, destination_pitch, source, source_pitch, width, height, kind,
            stream);
    if (mode == SUBMIT_RECORDING && status == cudaSuccess && !shadow_failed)
        record_failure(cudaMemcpy2DAsync(destination, destination_pitch, source, source_pitch, width, height, kind,
                shadow) != cudaSuccess);
    return status;
}

cudaError_t euhedral_submit_memset(void* destination, int value, size_t bytes, cudaStream_t stream) {
    if (mode != SUBMIT_ORDINARY) {
        mix(0x5e7ull);
        mix((uint64_t)(uintptr_t)destination);
        mix(((uint64_t)bytes << 8) | (unsigned char)value);
        if (mode == SUBMIT_CHECKING) return cudaSuccess;
    }
    cudaError_t status = cudaMemsetAsync(destination, value, bytes, stream);
    if (mode == SUBMIT_RECORDING && status == cudaSuccess && !shadow_failed)
        record_failure(cudaMemsetAsync(destination, value, bytes, shadow) != cudaSuccess);
    return status;
}

void euhedral_cuda_submission_record(uint64_t shadow_stream, int overlap) {
    mode = SUBMIT_RECORDING;
    shadow = (cudaStream_t)(uintptr_t)shadow_stream;
    shadow_overlap = overlap != 0;
    shadow_failed = shadow_stream == 0;
    hash = 0;
}

void euhedral_cuda_submission_check(void) {
    mode = SUBMIT_CHECKING;
    hash = 0;
}

uint64_t euhedral_cuda_submission_finish(void) {
    uint64_t result = shadow_failed ? 0 : hash | 1u;
    mode = SUBMIT_ORDINARY;
    shadow = NULL;
    shadow_failed = 0;
    hash = 0;
    return result;
}

void euhedral_cuda_submission_mark_unrecordable(void) {
    if (mode == SUBMIT_RECORDING) shadow_failed = 1;
}

uint64_t euhedral_cuda_submission_check_begin(void) {
    if (euhedral_cuda_bind_thread_context() != EUHEDRAL_CUDA_SUCCESS) return 0;
    if (sink == NULL && cudaStreamCreateWithFlags(&sink, cudaStreamNonBlocking) != cudaSuccess) {
        sink = NULL;
        return 0;
    }
    if (cudaStreamBeginCapture(sink, cudaStreamCaptureModeRelaxed) != cudaSuccess) return 0;
    return (uint64_t)(uintptr_t)sink;
}

int euhedral_cuda_submission_check_end(void) {
    cudaGraph_t graph = NULL;
    if (sink == NULL) return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    cudaError_t status = cudaStreamEndCapture(sink, &graph);
    if (status != cudaSuccess) return (int)status;
    size_t nodes = 0;
    status = cudaGraphGetNodes(graph, NULL, &nodes);
    cudaGraphDestroy(graph);
    if (status != cudaSuccess) return (int)status;
    /* Every checked submission was only hashed: anything captured here bypassed the submission functions. */
    return nodes == 0 ? EUHEDRAL_CUDA_SUCCESS : EUHEDRAL_CUDA_ROUTE_UNAVAILABLE;
}

int euhedral_cuda_graph_capture_begin(uint64_t stream) {
    int status = euhedral_cuda_bind_thread_context();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    return (int)cudaStreamBeginCapture((cudaStream_t)(uintptr_t)stream, cudaStreamCaptureModeRelaxed);
}

/* Ends the capture on `stream` and instantiates it; *exec is 0 when nothing could be instantiated. A
 * capture is always ended, so the stream is usable afterwards even when the capture failed. */
int euhedral_cuda_graph_capture_end(uint64_t stream, uint64_t* exec) {
    *exec = 0;
    cudaGraph_t graph = NULL;
    cudaError_t status = cudaStreamEndCapture((cudaStream_t)(uintptr_t)stream, &graph);
    if (status != cudaSuccess) {
        if (graph != NULL) cudaGraphDestroy(graph);
        return (int)status;
    }
    cudaGraphExec_t instance = NULL;
    status = cudaGraphInstantiate(&instance, graph, 0);
    cudaGraphDestroy(graph);
    if (status != cudaSuccess) return (int)status;
    *exec = (uint64_t)(uintptr_t)instance;
    return EUHEDRAL_CUDA_SUCCESS;
}

int euhedral_cuda_graph_launch(uint64_t exec, uint64_t stream) {
    if (exec == 0 || stream == 0) return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    int status = euhedral_cuda_bind_thread_context();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    return (int)cudaGraphLaunch((cudaGraphExec_t)(uintptr_t)exec, (cudaStream_t)(uintptr_t)stream);
}

int euhedral_cuda_graph_destroy(uint64_t exec) {
    if (exec == 0) return EUHEDRAL_CUDA_SUCCESS;
    return (int)cudaGraphExecDestroy((cudaGraphExec_t)(uintptr_t)exec);
}
