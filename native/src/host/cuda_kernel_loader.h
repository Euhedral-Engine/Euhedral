#ifndef EUHEDRAL_CUDA_KERNEL_LOADER_H
#define EUHEDRAL_CUDA_KERNEL_LOADER_H

#include "euhedral_cuda.h"
#include <cuda.h>

int euhedral_cuda_bind_thread_context(void);

int euhedral_cuda_load_kernel(
        const void* anchor,
        const char* source_name,
        const char* function_name,
        CUmodule* module,
        CUfunction* function);

/* Loads `function_name` from a module compiled for this device's arch-specific target (sm_XYa), for
 * kernels that use arch-specific instructions such as block-scaled FP4 MMA. Returns
 * EUHEDRAL_CUDA_ROUTE_UNAVAILABLE when the device is not an sm_12x GPU. */
int euhedral_cuda_load_native_kernel(
        const void* anchor,
        const char* source_name,
        const char* function_name,
        CUmodule* module,
        CUfunction* function);

/* Exact numerics: the selection the Rust engine mirrors for itself. Returns the previous selection. */
EUHEDRAL_CUDA_EXPORT int euhedral_cuda_select_exact_numerics(int exact);

#endif
