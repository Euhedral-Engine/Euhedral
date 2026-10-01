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

/* Programmatic dependent launch. Only kernels registered here call griddepcontrol.wait first,
 * so only they may be launched with the serialization attribute. Only threads that called euhedral_cuda_pdl_select(1)
 * launch with it; EUHEDRAL_PDL=0 disables it everywhere. */
void euhedral_cuda_pdl_register(CUfunction function);
EUHEDRAL_CUDA_EXPORT void euhedral_cuda_pdl_select(int enabled);
/* Exact numerics: nonzero routes every kernel whose FP32 accumulation order was relaxed back to the
 * exact kernels that reproduce the scalar references bit for bit (the numerical oracle). Initialized
 * from EUHEDRAL_EXACT=1 (or the older EUHEDRAL_Q3_DECODE=EXACT); settable for comparisons. */
int euhedral_cuda_exact_numerics(void);
EUHEDRAL_CUDA_EXPORT int euhedral_cuda_select_exact_numerics(int exact);
CUresult euhedral_launch_kernel(CUfunction function, unsigned int grid_x, unsigned int grid_y, unsigned int grid_z,
        unsigned int block_x, unsigned int block_y, unsigned int block_z, unsigned int shared_bytes,
        CUstream stream, void** parameters, void** extra);

#endif
