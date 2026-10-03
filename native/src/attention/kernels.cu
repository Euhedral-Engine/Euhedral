// NVRTC module root: the attention leaves. NVFP4 cache layout and decode/prefill attention come
// first; the QK norm/RoPE and KV append leaves follow.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include "nvfp4_kv.cuh"
#include "nvfp4_attention.cuh"
#include "nvfp4_prefill32.cuh"
#include "nvfp4_prefill_fa2.cuh"
#include "nvfp4_decode_gqa.cuh"
#include "common/pdl.cuh"
#include "kv_append.cuh"
#include "qk_norm_rope.cuh"
