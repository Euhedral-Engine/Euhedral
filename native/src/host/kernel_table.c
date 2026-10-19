#include "kernel_table.h"
#include <cuda_runtime_api.h>
#include <stdlib.h>
#include <string.h>

static int anchor;

void euhedral_kernel_table_initialize(EuhedralKernelTable* table) {
    const int count = table->count;
    table->init_status = EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
    table->functions = calloc((size_t)count, sizeof(CUfunction));
    table->parameter_count = calloc((size_t)count, sizeof(int));
    table->parameter_size = calloc((size_t)count * EUHEDRAL_KERNEL_TABLE_MAX_PARAMETERS, sizeof(uint8_t));
    CUmodule* modules = calloc((size_t)count, sizeof(CUmodule));
    if (table->functions == NULL || table->parameter_count == NULL || table->parameter_size == NULL || modules == NULL) {
        free(modules);
        free(table->functions);
        free(table->parameter_count);
        free(table->parameter_size);
        table->functions = NULL;
        table->parameter_count = NULL;
        table->parameter_size = NULL;
        return;
    }
    int status = EUHEDRAL_CUDA_SUCCESS;
    for (int index = 0; index < count && status == EUHEDRAL_CUDA_SUCCESS; index++) {
        const EuhedralKernelEntry* entry = &table->entries[index];
        int first = index;
        for (int earlier = 0; earlier < index; earlier++) {
            if (strcmp(table->entries[earlier].module, entry->module) == 0) {
                first = earlier;
                break;
            }
        }
        if (first == index) {
            if (entry->architecture_specific) {
                status = euhedral_cuda_load_native_kernel(&anchor, entry->module, entry->name, &modules[index],
                        &table->functions[index]);
                // Not this device's architecture: the module's functions stay NULL.
                if (status == EUHEDRAL_CUDA_ROUTE_UNAVAILABLE) status = EUHEDRAL_CUDA_SUCCESS;
            } else {
                status = euhedral_cuda_load_kernel(&anchor, entry->module, entry->name, &modules[index],
                        &table->functions[index]);
            }
        } else if (modules[first] != NULL) {
            status = (int)cuModuleGetFunction(&table->functions[index], modules[first], entry->name);
        }
    }
    for (int index = 0; index < count && status == EUHEDRAL_CUDA_SUCCESS; index++) {
        const EuhedralKernelEntry* entry = &table->entries[index];
        if (table->functions[index] == NULL) continue;
        if (entry->max_dynamic_shared_bytes != 0) {
            status = (int)cuFuncSetAttribute(table->functions[index], CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    (int)entry->max_dynamic_shared_bytes);
            if (status != CUDA_SUCCESS) break;
        }
        int parameters = 0;
        for (; parameters < EUHEDRAL_KERNEL_TABLE_MAX_PARAMETERS; parameters++) {
            size_t offset = 0, size = 0;
            if (cuFuncGetParamInfo(table->functions[index], (size_t)parameters, &offset, &size) != CUDA_SUCCESS) break;
            if (size != 4 && size != 8) {
                status = EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
                break;
            }
            table->parameter_size[(size_t)index * EUHEDRAL_KERNEL_TABLE_MAX_PARAMETERS + (size_t)parameters] =
                    (uint8_t)size;
        }
        table->parameter_count[index] = parameters;
    }
    free(modules);
    table->init_status = status;
}

int euhedral_kernel_table_launch(EuhedralKernelTable* table, int (*ready)(void), int32_t kernel,
        uint32_t grid_x, uint32_t grid_y, uint32_t grid_z, uint32_t block_x, uint32_t block_y, uint32_t block_z,
        uint32_t shared_bytes, const uint64_t* words, const uint8_t* sizes, int32_t count) {
    if (kernel < 0 || kernel >= table->count || words == NULL || sizes == NULL || count < 0
            || count > EUHEDRAL_KERNEL_TABLE_MAX_PARAMETERS
            || grid_x == 0 || grid_y == 0 || grid_z == 0 || block_x == 0 || block_y == 0 || block_z == 0)
        return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    int context_status = euhedral_cuda_bind_thread_context();
    if (context_status != EUHEDRAL_CUDA_SUCCESS) return context_status;
    int init_status = ready();
    if (init_status != EUHEDRAL_CUDA_SUCCESS) return init_status;
    if (table->functions[kernel] == NULL) return EUHEDRAL_CUDA_ROUTE_UNAVAILABLE;
    if (count != table->parameter_count[kernel]) return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    const uint8_t* expected = &table->parameter_size[(size_t)kernel * EUHEDRAL_KERNEL_TABLE_MAX_PARAMETERS];
    void* parameters[EUHEDRAL_KERNEL_TABLE_MAX_PARAMETERS];
    uint32_t narrow[EUHEDRAL_KERNEL_TABLE_MAX_PARAMETERS];
    for (int index = 0; index < count; index++) {
        if (sizes[index] != expected[index]) return EUHEDRAL_CUDA_INVALID_ARGUMENT;
        if (sizes[index] == 8) {
            parameters[index] = (void*)&words[index];
        } else {
            narrow[index] = (uint32_t)words[index];
            parameters[index] = &narrow[index];
        }
    }
    CUresult status = euhedral_launch_kernel(table->functions[kernel], grid_x, grid_y, grid_z, block_x, block_y,
            block_z, shared_bytes, euhedral_cuda_submission_stream(), parameters, NULL);
    if (status != CUDA_SUCCESS) return (int)status;
    if (euhedral_cuda_submission_stream() != NULL) return EUHEDRAL_CUDA_SUCCESS;
    cudaError_t sync = cudaDeviceSynchronize();
    return sync == cudaSuccess ? EUHEDRAL_CUDA_SUCCESS : (int)sync;
}

int euhedral_kernel_table_available(EuhedralKernelTable* table, int (*ready)(void), int32_t kernel) {
    if (kernel < 0 || kernel >= table->count) return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    int context_status = euhedral_cuda_bind_thread_context();
    if (context_status != EUHEDRAL_CUDA_SUCCESS) return context_status;
    int init_status = ready();
    if (init_status != EUHEDRAL_CUDA_SUCCESS) return init_status;
    return table->functions[kernel] != NULL ? 1 : 0;
}
