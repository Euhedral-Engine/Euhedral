// The trellis prefill on block-scaled FP8 tensor cores: the module root the host compiles for sm_12xa and loads for
// the architecture-specific entries of the shared linear table (native/src/host/linear_kernels.c) that name
// trellis_fp8/kernels.cu. On other devices those entries are unavailable and the FP32-accumulating whole-K launches
// of trellis/kernels.cu run.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "trellis/codebook.cuh"
#include "trellis/decode.cuh"
#include "trellis/fp8.cuh"
