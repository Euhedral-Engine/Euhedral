// Flash-Next (qwen4_exp) kernels: the module root the host loads for euhedral_cuda_qwen4_launch.
// The QSA cache keeps the E2M1 code grid (attention/nvfp4_kv.cuh): the midrise grid is measured on Qwen3.8 only.
#ifndef EUHEDRAL_KV_CODEC
#define EUHEDRAL_KV_CODEC 0
#endif
#include "qwen4/common.cuh"
#include "qwen4/linear.cuh"
#include "qwen4/hc.cuh"
#include "qwen4/ple.cuh"
#include "qwen4/gdn.cuh"
#include "qwen4/qsa_norm.cuh"
#include "qwen4/qsa_index.cuh"
#include "qwen4/qsa_attention.cuh"
#include "qwen4/moe.cuh"
#include "qwen4/experts.cuh"
