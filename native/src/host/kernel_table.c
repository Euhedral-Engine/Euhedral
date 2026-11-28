#include "kernel_table.h"
#include <cuda_runtime_api.h>
#include <stdlib.h>
#include <string.h>

static int anchor;

void euhedral_kernel_table_initialize(EuhedralKernelTable* table) {
    const int count = table->count;
    table->init_status = EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
    table->functions = calloc((size_t)count, sizeof(CUfunction));
    CUmodule* modules = calloc((size_t)count, sizeof(CUmodule));
    if (table->functions == NULL || modules == NULL) {
        free(modules);
        free(table->functions);
        table->functions = NULL;
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
    }
    free(modules);
    table->init_status = status;
}

int euhedral_kernel_table_available(EuhedralKernelTable* table, int (*ready)(void), int32_t kernel) {
    if (kernel < 0 || kernel >= table->count) return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    int context_status = euhedral_cuda_bind_thread_context();
    if (context_status != EUHEDRAL_CUDA_SUCCESS) return context_status;
    int init_status = ready();
    if (init_status != EUHEDRAL_CUDA_SUCCESS) return init_status;
    return table->functions[kernel] != NULL ? 1 : 0;
}
