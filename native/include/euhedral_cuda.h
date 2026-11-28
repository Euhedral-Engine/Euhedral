#ifndef EUHEDRAL_CUDA_H
#define EUHEDRAL_CUDA_H

#include <stdint.h>

#ifdef _WIN32
#define EUHEDRAL_CUDA_EXPORT __declspec(dllexport)
#else
#define EUHEDRAL_CUDA_EXPORT __attribute__((visibility("default")))
#endif

#define EUHEDRAL_CUDA_SUCCESS 0
#define EUHEDRAL_CUDA_INVALID_ARGUMENT (-1)
#define EUHEDRAL_CUDA_SIZE_OVERFLOW (-2)
#define EUHEDRAL_CUDA_FORMAT_MISMATCH (-3)
#define EUHEDRAL_CUDA_KERNEL_UNAVAILABLE (-4)
/* The requested specialized route does not apply (numerics, shape or alignment); use another. */
#define EUHEDRAL_CUDA_ROUTE_UNAVAILABLE (-5)

#ifdef __cplusplus
extern "C" {
#endif

/* The CUDA headers the kernels compile against with NVRTC. Kernels do not compile until it is set. Returns
 * EUHEDRAL_CUDA_INVALID_ARGUMENT for a null, empty or overlong path and cudaErrorMemoryAllocation when the path
 * cannot be copied. */
EUHEDRAL_CUDA_EXPORT int euhedral_cuda_set_include_directory(const char* path);
EUHEDRAL_CUDA_EXPORT void* euhedral_cuda_malloc(uint64_t byte_size);
EUHEDRAL_CUDA_EXPORT int euhedral_cuda_free(void* address);
/// Allocates and frees in stream order on the calling thread's selected stream, from the device's default memory pool.
EUHEDRAL_CUDA_EXPORT void* euhedral_cuda_malloc_async(uint64_t byte_size);
EUHEDRAL_CUDA_EXPORT int euhedral_cuda_free_async(void* address);
EUHEDRAL_CUDA_EXPORT void* euhedral_cuda_host_malloc(uint64_t byte_size);
EUHEDRAL_CUDA_EXPORT int euhedral_cuda_host_free(void* address);
/// Pinned, huge-page-backed host memory for weights that are staged to the device on use.
EUHEDRAL_CUDA_EXPORT void* euhedral_cuda_host_weights_malloc(uint64_t byte_size);
EUHEDRAL_CUDA_EXPORT int euhedral_cuda_host_weights_free(void* address);
/// The device address at which kernels read host weights in place (zero-copy, over PCIe).
EUHEDRAL_CUDA_EXPORT int euhedral_cuda_host_weights_device_pointer(const void* host_address, uint64_t* device_address);
EUHEDRAL_CUDA_EXPORT int euhedral_cuda_device_memory_info(uint64_t* free_byte_size, uint64_t* total_byte_size);
/// The streaming multiprocessors of the current device.
EUHEDRAL_CUDA_EXPORT int euhedral_cuda_multiprocessor_count(int32_t* count);
/// The compute capability (major, minor) of the current device.
EUHEDRAL_CUDA_EXPORT int euhedral_cuda_compute_capability(int32_t* major, int32_t* minor);

/// Both copies are synchronous: with a stream selected they are queued on it and finish before returning.
EUHEDRAL_CUDA_EXPORT int euhedral_cuda_copy_host_to_device(
        void* device_address,
        const void* host_address,
        uint64_t byte_size);
EUHEDRAL_CUDA_EXPORT int euhedral_cuda_copy_device_to_host(
        void* host_address,
        const void* device_address,
        uint64_t byte_size);

/* Selects exact numerics (nonzero) and returns the previous selection. The Rust engine mirrors the selection for
 * itself; the default is relaxed. */
EUHEDRAL_CUDA_EXPORT int euhedral_cuda_select_exact_numerics(int exact);

/// Whether the native Blackwell NVFP4 route (docs/kernels/NVFP4_NATIVE.md) runs on this device.
EUHEDRAL_CUDA_EXPORT int euhedral_cuda_nvfp4_native_available(void);
/* Scratch for one native NVFP4 linear or gate/up region: quantized activations, plus FP32 split-K partials for
 * decode-like row counts (2 to 64 rows run the skinny weight-streaming kernel). UINT64_MAX when the device's
 * multiprocessor count cannot be read. */
EUHEDRAL_CUDA_EXPORT uint64_t euhedral_cuda_nvfp4_native_scratch_bytes(
        uint32_t rows, uint32_t in_features, uint32_t out_features);
/// Whether the block-scaled FP8 prefill route of the Q3, Q4 and Q5 linears (docs/kernels/PREFILL_MX.md) runs on this device.
EUHEDRAL_CUDA_EXPORT int euhedral_cuda_q3_mx_available(void);

EUHEDRAL_CUDA_EXPORT int euhedral_cuda_synchronize(void);
/// Selects the stream the calling thread's later stream-ordered calls use; refused when one is selected already.
EUHEDRAL_CUDA_EXPORT int euhedral_cuda_stream_select(uint64_t stream);
EUHEDRAL_CUDA_EXPORT void euhedral_cuda_stream_clear(void);

/// The names of the Flash-Next (qwen4_exp) kernels (src/qwen4), or null past the last.
EUHEDRAL_CUDA_EXPORT const char* euhedral_cuda_qwen4_kernel_name(int index);

/// The shared linear kernels: the names of the table's entries, or null past the last.
EUHEDRAL_CUDA_EXPORT int euhedral_cuda_linear_kernel_count(void);
EUHEDRAL_CUDA_EXPORT const char* euhedral_cuda_linear_kernel_name(int index);
/// 1 when kernel `kernel` runs on this device, 0 when it does not (the kernels of an architecture-specific module are
/// unavailable elsewhere), else a negative status.
EUHEDRAL_CUDA_EXPORT int euhedral_cuda_linear_kernel_available(int32_t kernel);

#ifdef __cplusplus
}
#endif

#endif
