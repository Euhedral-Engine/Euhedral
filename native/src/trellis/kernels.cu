// Trellis (EXL3) linear kernels: the module root the host loads for the entries of the shared linear table
// (native/src/host/linear_kernels.c) that name trellis/kernels.cu. The decode math is trellis/codebook.cuh, which the
// quantization module compiles as well.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "trellis/codebook.cuh"
#include "trellis/decode.cuh"
#include "trellis/prepass.cuh"
#include "trellis/mma.cuh"
#include "trellis/epilogue.cuh"
#include "reference/trellis.cuh"
