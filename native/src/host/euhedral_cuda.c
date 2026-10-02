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

#ifdef _WIN32
static __declspec(thread) cudaStream_t selected_stream;
#else
static _Thread_local cudaStream_t selected_stream;
#endif

void* euhedral_cuda_submission_stream(void) {
    return selected_stream;
}

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
#else
    void* address = NULL;
    if (posix_memalign(&address, EUHEDRAL_HOST_WEIGHT_ALIGNMENT, size) != 0) return NULL;
    (void)madvise(address, size, MADV_HUGEPAGE);
    // Fault the pages in after the advice, so they are huge pages before pinning. Under memory pressure
    // a fault falls back to 4 KiB pages; MADV_COLLAPSE then reclaims and compacts synchronously.
    memset(address, 0, size);
    for (int attempt = 0; attempt < 3 && madvise(address, size, MADV_COLLAPSE) != 0; attempt++) {}
#endif
    if (cudaHostRegister(address, size, cudaHostRegisterMapped) != cudaSuccess) {
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
#ifdef _WIN32
    _aligned_free(address);
#else
    free(address);
#endif
    return status == cudaSuccess ? EUHEDRAL_CUDA_SUCCESS : (int)status;
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

    cudaStream_t stream = euhedral_cuda_submission_stream();
    cudaError_t status = stream == NULL
            ? cudaMemcpy(device_address, host_address, (size_t)byte_size, cudaMemcpyHostToDevice)
            : cudaMemcpyAsync(device_address, host_address, (size_t)byte_size, cudaMemcpyHostToDevice, stream);
    // Callers may release pageable host memory on return (the KV page-table arena does), so an async
    // upload must finish here.
    if (status == cudaSuccess && stream != NULL) status = cudaStreamSynchronize(stream);
    return status == cudaSuccess ? EUHEDRAL_CUDA_SUCCESS : (int) status;
}

int euhedral_cuda_copy_upload_to_device(
        void* device_address,
        const void* host_address,
        uint64_t byte_size) {
    if (device_address == NULL || host_address == NULL || byte_size == 0) return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    if (byte_size > SIZE_MAX) return EUHEDRAL_CUDA_SIZE_OVERFLOW;
    cudaStream_t stream = euhedral_cuda_submission_stream();
    // The source is pinned and retained through stream completion, so a queued copy needs no
    // synchronization; with no stream selected it completes before returning, like every operation.
    cudaError_t status = stream == NULL
            ? cudaMemcpy(device_address, host_address, (size_t)byte_size, cudaMemcpyHostToDevice)
            : cudaMemcpyAsync(device_address, host_address, (size_t)byte_size, cudaMemcpyHostToDevice, stream);
    return status == cudaSuccess ? EUHEDRAL_CUDA_SUCCESS : (int)status;
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

    cudaStream_t stream = euhedral_cuda_submission_stream();
    cudaError_t status = stream == NULL
            ? cudaMemcpy(host_address, device_address, (size_t)byte_size, cudaMemcpyDeviceToHost)
            : cudaMemcpyAsync(host_address, device_address, (size_t)byte_size, cudaMemcpyDeviceToHost, stream);
    if (status == cudaSuccess && stream != NULL) status = cudaStreamSynchronize(stream);
    return status == cudaSuccess ? EUHEDRAL_CUDA_SUCCESS : (int) status;
}

int euhedral_cuda_copy_device_to_readback(
        void* host_address,
        const void* device_address,
        uint64_t byte_size) {
    if (host_address == NULL || device_address == NULL || byte_size == 0) return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    if (byte_size > SIZE_MAX) return EUHEDRAL_CUDA_SIZE_OVERFLOW;
    cudaStream_t stream = euhedral_cuda_submission_stream();
    // The destination is pinned and owned until the stream's work retires, so a queued copy needs no
    // synchronization here; with no stream selected the copy completes before returning.
    cudaError_t status = stream == NULL
            ? cudaMemcpy(host_address, device_address, (size_t)byte_size, cudaMemcpyDeviceToHost)
            : cudaMemcpyAsync(host_address, device_address, (size_t)byte_size, cudaMemcpyDeviceToHost, stream);
    return status == cudaSuccess ? EUHEDRAL_CUDA_SUCCESS : (int)status;
}

int euhedral_cuda_copy_device_to_device(
        void* destination_address,
        const void* source_address,
        uint64_t byte_size) {
    if (byte_size == 0) return EUHEDRAL_CUDA_SUCCESS;
    if (destination_address == NULL || source_address == NULL) return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    if (byte_size > SIZE_MAX) return EUHEDRAL_CUDA_SIZE_OVERFLOW;
    cudaStream_t stream = euhedral_cuda_submission_stream();
    cudaError_t status = stream == NULL
            ? cudaMemcpy(destination_address, source_address, (size_t)byte_size, cudaMemcpyDeviceToDevice)
            : cudaMemcpyAsync(destination_address, source_address, (size_t)byte_size, cudaMemcpyDeviceToDevice, stream);
    // With a submission stream selected the copy is queued on it; the caller keeps both allocations
    // until that stream's work has retired.
    return status == cudaSuccess ? EUHEDRAL_CUDA_SUCCESS : (int)status;
}

int euhedral_cuda_copy_device_to_device_2d(
        void* destination_address,
        uint64_t destination_pitch,
        const void* source_address,
        uint64_t source_pitch,
        uint64_t row_bytes,
        uint64_t rows) {
    if (row_bytes == 0 || rows == 0) return EUHEDRAL_CUDA_SUCCESS;
    if (destination_address == NULL || source_address == NULL || destination_pitch < row_bytes || source_pitch < row_bytes)
        return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    if (destination_pitch > SIZE_MAX || source_pitch > SIZE_MAX || row_bytes > SIZE_MAX || rows > SIZE_MAX)
        return EUHEDRAL_CUDA_SIZE_OVERFLOW;
    cudaStream_t stream = euhedral_cuda_submission_stream();
    cudaError_t status = stream == NULL
            ? cudaMemcpy2D(destination_address, (size_t)destination_pitch, source_address, (size_t)source_pitch,
                    (size_t)row_bytes, (size_t)rows, cudaMemcpyDeviceToDevice)
            : cudaMemcpy2DAsync(destination_address, (size_t)destination_pitch, source_address, (size_t)source_pitch,
                    (size_t)row_bytes, (size_t)rows, cudaMemcpyDeviceToDevice, stream);
    return status == cudaSuccess ? EUHEDRAL_CUDA_SUCCESS : (int)status;
}

uint64_t euhedral_cuda_stream_create(void) {
    cudaStream_t stream = NULL;
    if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) return 0;
    return (uint64_t)(uintptr_t)stream;
}

int euhedral_cuda_stream_destroy(uint64_t value) {
    if (value == 0) return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    cudaError_t status = cudaStreamDestroy((cudaStream_t)(uintptr_t)value);
    return status == cudaSuccess ? EUHEDRAL_CUDA_SUCCESS : (int)status;
}

int euhedral_cuda_stream_synchronize(uint64_t value) {
    if (value == 0) return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    cudaError_t status = cudaStreamSynchronize((cudaStream_t)(uintptr_t)value);
    return status == cudaSuccess ? EUHEDRAL_CUDA_SUCCESS : (int)status;
}

int euhedral_cuda_stream_select(uint64_t value) {
    if (value == 0 || selected_stream != NULL) return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    selected_stream = (cudaStream_t)(uintptr_t)value;
    return EUHEDRAL_CUDA_SUCCESS;
}

void euhedral_cuda_stream_clear(void) {
    selected_stream = NULL;
}

uint64_t euhedral_cuda_completion_event_create(void) {
    cudaEvent_t event = NULL;
    if (cudaEventCreateWithFlags(&event, cudaEventDisableTiming) != cudaSuccess) return 0;
    return (uint64_t)(uintptr_t)event;
}

int euhedral_cuda_completion_event_record(uint64_t value, uint64_t stream) {
    if (value == 0 || stream == 0) return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    cudaError_t status = cudaEventRecord((cudaEvent_t)(uintptr_t)value, (cudaStream_t)(uintptr_t)stream);
    return status == cudaSuccess ? EUHEDRAL_CUDA_SUCCESS : (int)status;
}

int euhedral_cuda_stream_wait_event(uint64_t stream, uint64_t value) {
    if (stream == 0 || value == 0) return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    cudaError_t status = cudaStreamWaitEvent((cudaStream_t)(uintptr_t)stream, (cudaEvent_t)(uintptr_t)value, 0);
    return status == cudaSuccess ? EUHEDRAL_CUDA_SUCCESS : (int)status;
}

int euhedral_cuda_completion_event_query(uint64_t value) {
    if (value == 0) return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    cudaError_t status = cudaEventQuery((cudaEvent_t)(uintptr_t)value);
    if (status == cudaSuccess) return EUHEDRAL_CUDA_SUCCESS;
    if (status == cudaErrorNotReady) return -5;
    return (int)status;
}

int euhedral_cuda_completion_event_destroy(uint64_t value) {
    if (value == 0) return EUHEDRAL_CUDA_SUCCESS;
    cudaError_t status = cudaEventDestroy((cudaEvent_t)(uintptr_t)value);
    return status == cudaSuccess ? EUHEDRAL_CUDA_SUCCESS : (int)status;
}

struct completion_notification {
    void (*callback)(uint64_t, int);
    uint64_t token;
};

static void CUDART_CB notify_completion(cudaStream_t stream, cudaError_t status, void* user_data) {
    (void)stream;
    struct completion_notification* notification = (struct completion_notification*)user_data;
    void (*callback)(uint64_t, int) = notification->callback;
    uint64_t token = notification->token;
    free(notification);
    callback(token, (int)status);
}

int euhedral_cuda_completion_notify(uint64_t value, void (*callback)(uint64_t, int), uint64_t token) {
    if (value == 0 || callback == NULL || token == 0) return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    struct completion_notification* notification = malloc(sizeof(*notification));
    if (notification == NULL) return EUHEDRAL_CUDA_SIZE_OVERFLOW;
    notification->callback = callback;
    notification->token = token;
    cudaError_t status = cudaStreamAddCallback((cudaStream_t)(uintptr_t)value, notify_completion, notification, 0);
    if (status != cudaSuccess) free(notification);
    return status == cudaSuccess ? EUHEDRAL_CUDA_SUCCESS : (int)status;
}
