#ifndef _WIN32
// posix_memalign and madvise(MADV_HUGEPAGE) under -std=c11.
#define _GNU_SOURCE
#endif

#include "euhedral_cuda.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#if !defined(CUDART_VERSION) || CUDART_VERSION < 13010
#error "Euhedral CUDA requires CUDA headers 13.1 or newer"
#endif

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#ifndef _WIN32
#include <sys/mman.h>
#ifndef MADV_COLLAPSE
#define MADV_COLLAPSE 25
#endif
#endif

#if !defined(CUDA_VERSION) || CUDA_VERSION < 13010
#error "Euhedral CUDA ABI requires CUDA toolkit 13.1 or newer"
#endif

static _Thread_local cudaStream_t selected_stream;

int euhedral_cuda_synchronize(void) {
    cudaError_t status = cudaDeviceSynchronize();
    return status == cudaSuccess ? EUHEDRAL_CUDA_SUCCESS : (int)status;
}

void* euhedral_cuda_malloc(uint64_t byte_size) {
    if (byte_size == 0 || byte_size > SIZE_MAX) {
        return NULL;
    }

    void* address = NULL;
    if (cudaMalloc(&address, (size_t) byte_size) != cudaSuccess) {
        return NULL;
    }
    return address;
}

int euhedral_cuda_free(void* address) {
    if (address == NULL) {
        return EUHEDRAL_CUDA_SUCCESS;
    }

    cudaError_t status = cudaFree(address);
    return status == cudaSuccess ? EUHEDRAL_CUDA_SUCCESS : (int) status;
}

/* Allocates in stream order on the thread's selected stream (the legacy default stream when none is selected),
   from the device's default memory pool. Never synchronizes the host. */
void* euhedral_cuda_malloc_async(uint64_t byte_size) {
    if (byte_size == 0 || byte_size > SIZE_MAX) return NULL;
    void* address = NULL;
    if (cudaMallocAsync(&address, (size_t) byte_size, selected_stream) != cudaSuccess) return NULL;
    return address;
}

/* Frees in stream order on the thread's selected stream: the memory returns to the pool once the stream reaches
   this point, after every reader queued before it on that stream. Never synchronizes the host. */
int euhedral_cuda_free_async(void* address) {
    if (address == NULL) return EUHEDRAL_CUDA_SUCCESS;
    cudaError_t status = cudaFreeAsync(address, selected_stream);
    return status == cudaSuccess ? EUHEDRAL_CUDA_SUCCESS : (int) status;
}

void* euhedral_cuda_host_malloc(uint64_t byte_size) {
    if (byte_size == 0 || byte_size > SIZE_MAX) return NULL;
    void* address = NULL;
    if (cudaHostAlloc(&address, (size_t)byte_size, cudaHostAllocDefault) != cudaSuccess) return NULL;
    return address;
}

int euhedral_cuda_host_free(void* address) {
    if (address == NULL) return EUHEDRAL_CUDA_SUCCESS;
    cudaError_t status = cudaFreeHost(address);
    return status == cudaSuccess ? EUHEDRAL_CUDA_SUCCESS : (int)status;
}

/// Host memory that copy engines read for staged weights: 2 MiB-aligned, backed by transparent huge
/// pages where the platform offers them, and pinned with cudaHostRegister. Translated DMA under an
/// IOMMU runs at about 25 GB/s over 4 KiB pages and about 44 GB/s over 2 MiB pages. It is also mapped
/// into the device's address space, so kernels can read small gathers (the token embedding) in place.
#define EUHEDRAL_HOST_WEIGHT_ALIGNMENT ((size_t)2 << 20)

void* euhedral_cuda_host_weights_malloc(uint64_t byte_size) {
    if (byte_size == 0 || byte_size > SIZE_MAX - EUHEDRAL_HOST_WEIGHT_ALIGNMENT) return NULL;
    size_t size = ((size_t)byte_size + EUHEDRAL_HOST_WEIGHT_ALIGNMENT - 1) & ~(EUHEDRAL_HOST_WEIGHT_ALIGNMENT - 1);
#ifdef _WIN32
    void* address = _aligned_malloc(size, EUHEDRAL_HOST_WEIGHT_ALIGNMENT);
    if (address == NULL) return NULL;
    memset(address, 0, size);  // the arena contract is zeroed memory, as on the other branch
#else
    void* address = NULL;
    if (posix_memalign(&address, EUHEDRAL_HOST_WEIGHT_ALIGNMENT, size) != 0) return NULL;
    (void)madvise(address, size, MADV_HUGEPAGE);
    // Fault the pages in after the advice, so they are huge pages before pinning. Under memory pressure
    // a fault falls back to 4 KiB pages; MADV_COLLAPSE then reclaims and compacts synchronously.
    memset(address, 0, size);
    for (int attempt = 0; attempt < 3 && madvise(address, size, MADV_COLLAPSE) != 0; attempt++) {}
#endif
    // Portable: every device, not only the current one, can address the arena.
    if (cudaHostRegister(address, size, cudaHostRegisterMapped | cudaHostRegisterPortable) != cudaSuccess) {
#ifdef _WIN32
        _aligned_free(address);
#else
        free(address);
#endif
        return NULL;
    }
    return address;
}

int euhedral_cuda_host_weights_device_pointer(const void* host_address, uint64_t* device_address) {
    if (host_address == NULL || device_address == NULL) return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    void* device = NULL;
    cudaError_t status = cudaHostGetDevicePointer(&device, (void*)host_address, 0);
    if (status != cudaSuccess) return (int)status;
    *device_address = (uint64_t)(uintptr_t)device;
    return EUHEDRAL_CUDA_SUCCESS;
}

int euhedral_cuda_host_weights_free(void* address) {
    if (address == NULL) return EUHEDRAL_CUDA_SUCCESS;
    cudaError_t status = cudaHostUnregister(address);
    // Memory the driver still pins must not return to the allocator: a later allocation reusing the range could
    // not be registered, and the driver would keep translating its old pages. Keep it; the caller retries.
    if (status != cudaSuccess) {
        (void)cudaGetLastError();
        return (int)status;
    }
#ifdef _WIN32
    _aligned_free(address);
#else
    free(address);
#endif
    return EUHEDRAL_CUDA_SUCCESS;
}

int euhedral_cuda_device_memory_info(uint64_t* free_byte_size, uint64_t* total_byte_size) {
    if (free_byte_size == NULL || total_byte_size == NULL) {
        return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    }

    size_t free_bytes = 0;
    size_t total_bytes = 0;
    cudaError_t status = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (status != cudaSuccess) {
        return (int) status;
    }

    *free_byte_size = (uint64_t) free_bytes;
    *total_byte_size = (uint64_t) total_bytes;
    return EUHEDRAL_CUDA_SUCCESS;
}

int euhedral_cuda_multiprocessor_count(int32_t* count) {
    if (count == NULL) {
        return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    }
    int device = 0;
    cudaError_t status = cudaGetDevice(&device);
    if (status != cudaSuccess) {
        return (int) status;
    }
    int value = 0;
    status = cudaDeviceGetAttribute(&value, cudaDevAttrMultiProcessorCount, device);
    if (status != cudaSuccess) {
        return (int) status;
    }
    *count = (int32_t) value;
    return EUHEDRAL_CUDA_SUCCESS;
}

int euhedral_cuda_compute_capability(int32_t* major, int32_t* minor) {
    if (major == NULL || minor == NULL) return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    int device = 0;
    cudaError_t status = cudaGetDevice(&device);
    if (status != cudaSuccess) return (int) status;
    int high = 0;
    int low = 0;
    status = cudaDeviceGetAttribute(&high, cudaDevAttrComputeCapabilityMajor, device);
    if (status != cudaSuccess) return (int) status;
    status = cudaDeviceGetAttribute(&low, cudaDevAttrComputeCapabilityMinor, device);
    if (status != cudaSuccess) return (int) status;
    *major = (int32_t) high;
    *minor = (int32_t) low;
    return EUHEDRAL_CUDA_SUCCESS;
}

int euhedral_cuda_copy_host_to_device(
        void* device_address,
        const void* host_address,
        uint64_t byte_size) {
    if (byte_size == 0) {
        return EUHEDRAL_CUDA_SUCCESS;
    }
    if (device_address == NULL || host_address == NULL) {
        return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    }
    if (byte_size > SIZE_MAX) {
        return EUHEDRAL_CUDA_SIZE_OVERFLOW;
    }

    cudaStream_t stream = selected_stream;
    cudaError_t status = stream == NULL
            ? cudaMemcpy(device_address, host_address, (size_t)byte_size, cudaMemcpyHostToDevice)
            : cudaMemcpyAsync(device_address, host_address, (size_t)byte_size, cudaMemcpyHostToDevice, stream);
    // Callers may release pageable host memory on return (the KV page-table arena does), so an async
    // upload must finish here.
    if (status == cudaSuccess && stream != NULL) status = cudaStreamSynchronize(stream);
    return status == cudaSuccess ? EUHEDRAL_CUDA_SUCCESS : (int) status;
}

int euhedral_cuda_copy_device_to_host(
        void* host_address,
        const void* device_address,
        uint64_t byte_size) {
    if (byte_size == 0) {
        return EUHEDRAL_CUDA_SUCCESS;
    }
    if (host_address == NULL || device_address == NULL) {
        return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    }
    if (byte_size > SIZE_MAX) {
        return EUHEDRAL_CUDA_SIZE_OVERFLOW;
    }

    cudaStream_t stream = selected_stream;
    cudaError_t status = stream == NULL
            ? cudaMemcpy(host_address, device_address, (size_t)byte_size, cudaMemcpyDeviceToHost)
            : cudaMemcpyAsync(host_address, device_address, (size_t)byte_size, cudaMemcpyDeviceToHost, stream);
    if (status == cudaSuccess && stream != NULL) status = cudaStreamSynchronize(stream);
    return status == cudaSuccess ? EUHEDRAL_CUDA_SUCCESS : (int) status;
}

int euhedral_cuda_stream_select(uint64_t value) {
    if (value == 0 || selected_stream != NULL) return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    selected_stream = (cudaStream_t)(uintptr_t)value;
    return EUHEDRAL_CUDA_SUCCESS;
}

void euhedral_cuda_stream_clear(void) {
    selected_stream = NULL;
}
