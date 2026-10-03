#ifndef EUHEDRAL_CUDA_SUBMISSION_H
#define EUHEDRAL_CUDA_SUBMISSION_H

#include "euhedral_cuda.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stddef.h>

/* Every launch, copy and memset a quantum submits goes through these functions, so a calling thread can
 * select how its submissions are handled:
 *
 * - ordinary: submitted to the stream;
 * - recording: submitted to the stream and again to the thread's shadow stream, which is being captured
 *   into a CUDA graph; the submission's description is hashed;
 * - checking: only hashed. The thread's selected stream is a sink under capture, so a submission that
 *   bypasses these functions is captured there and never runs.
 *
 * The hash covers each kernel's function, launch geometry, dynamic shared memory and parameter bytes, and
 * each copy's and memset's addresses and sizes. A replayed graph is valid for a quantum whose submissions
 * hash the same as the recorded quantum's. */

CUresult euhedral_submit_kernel(CUfunction function, unsigned int grid_x, unsigned int grid_y, unsigned int grid_z,
        unsigned int block_x, unsigned int block_y, unsigned int block_z, unsigned int shared_bytes,
        CUstream stream, void** parameters, int programmatic, int overlappable);
cudaError_t euhedral_submit_copy(void* destination, const void* source, size_t bytes, enum cudaMemcpyKind kind,
        cudaStream_t stream);
cudaError_t euhedral_submit_copy_2d(void* destination, size_t destination_pitch, const void* source,
        size_t source_pitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t euhedral_submit_memset(void* destination, int value, size_t bytes, cudaStream_t stream);

/* Whether the calling thread is checking: its submissions must not run. */
int euhedral_submission_checking(void);

/* Called by an operation a captured graph cannot repeat (a synchronous copy whose host memory may be released
 * on return). It ends the thread's recording and makes a checked quantum diverge. Returns whether the thread is
 * checking, in which case the operation must not run. */
int euhedral_submission_unrepeatable(void);

#endif
