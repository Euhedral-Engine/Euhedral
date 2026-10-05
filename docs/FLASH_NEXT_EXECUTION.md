# Flash-Next execution

Qwen3.8-Flash-Next (`qwen4_exp`) runs as a text model on the storage described in [FLASH_NEXT_RESIDENCY.md](FLASH_NEXT_RESIDENCY.md):
the planner decides where every object lives, the expert cache decides which routed experts are on the device, and the
operators below read whatever placement they are given. The behavioral oracle is the upstream `transformers` implementation
([FLASH_NEXT_REFERENCE.md](FLASH_NEXT_REFERENCE.md)); every operator is validated against fixtures captured from it.

```
token -> embedding row -> repeated over 4 residual streams
  -> 48 layers:  [PLE]  hyper-connection mix -> GDN | QSA -> inject -> hyper-connection mix -> MoE -> inject
  -> final hyper-connection mix -> output head -> logits
```

## Kernels and the launcher

Flash-Next's operators are many small kernels that differ in their arguments. They live in one NVRTC module
(`native/src/qwen4/kernels.cu`) behind one native entry point, `euhedral_cuda_qwen4_launch(kernel, grid, block, shared,
words, sizes, count)`: a table of kernel names in `native/src/host/qwen4_ops.c` and the enum `Qwen4Kernel` list the same
names in the same order (a test compares them). The launcher checks the number and width (4 or 8 bytes) of every argument
against the compiled kernel with `cuFuncGetParamInfo`, so a mismatch between a Java caller and the CUDA source fails the
call. Launches go through `euhedral_launch_kernel`, so they are hashed and recorded like every other submission. Adding a
kernel is a `.cuh`, an entry in the table and the enum, and a typed wrapper in `Qwen4Ops`.

## Numerics contract

Upstream runs BF16 tensors in PyTorch: every elementwise operation computes in FP32 and rounds its result to BF16, matmuls
accumulate in FP32 and round once. The Flash-Next kernels reproduce those roundings step by step (a gate is
`bf16(bf16(sigmoid(x)) * v)`, not `sigmoid(x) * v` rounded once), which is why most elementwise operators agree with the reference
bit for bit. Reductions (norms, dot products) accumulate in FP32 in a fixed tree, so they differ from PyTorch's order by a few
FP32 ulps, which is visible only where a result sits on a BF16 rounding boundary.

NVFP4 linears are the engine's: from nine rows the activations are quantized to two FP4 terms for the block-scaled tensor
cores, which makes prefill linears approximate; one to eight rows run BF16-activation kernels. The reference therefore has
two comparison modes. In *exact numerics* (`ExecutionGpu.selectExactNumerics`, the scalar references) the engine computes what
the reference computes, BF16 activations times exactly expanded weights, and agrees to a few BF16 roundings. In *production*
numerics the prefill linears add the activation quantization error. Tests run both and bound each separately.

| tensor family | exact numerics (relative RMS) | production numerics |
|---|---|---|
| hyper-connection norm, mix, injection | 0 to 2e-3 | same (BF16 kernels) |
| PLE rows and embedding | 0 (bit exact) | 0 |
| PLE projections, gate, convolution, output | 2e-5 to 7e-5 | 1e-3 to 3e-3 |

Relative RMS is `sqrt(mean((engine - reference)^2) / mean(reference^2))`; one BF16 rounding of a tensor is about 1.1e-3.

## Four-stream gated residual

The residual state of a token is four rows of 2560 values, `[stream][2560]`, started as the embedding repeated four times and
never collapsed into one. Before the attention block and before the MoE block `Qwen4HyperConnection.mix` normalizes each
stream (grouped RMSNorm with `1 + weight`), projects the 10,240 values to a rank-320 vector, applies `silu(x / 4)`, projects
back and gates the normalized streams with `sigmoid`; the block's input is the mean over streams. A third projection yields four
injection weights, `2 * sigmoid(x / 4)`; after the block, `inject` adds the block's result to every stream scaled by its weight.
The final mixer after layer 47 has no injection projection: it only mixes. A mix leaves the normalized streams and the raw
injection weights in its scratch for the injection that follows.

## Per-layer embedding and n-gram state

Layer 1 adds per-layer embeddings before its hyper-connection. The ids come from `Qwen4NgramIds`: for each token the 2-gram and
3-gram heads hash the token with its one or two predecessors (64-bit multiplies by the artifact's multipliers, XOR, modulus of
the head's prime vocabulary, plus the head's offset), a predecessor beyond an end-of-sequence token or the start of the
sequence reads as end-of-sequence. The two tokens before the next position are the state a sequence carries between prefill
chunks and decode steps. `NgramStore` gathers only the rows those ids name, as 96-byte records (the 90-byte row, padding, the
shard's global scale), through a pinned staging buffer; `euhedral_q4_ngram_expand_bf16` expands 16 records per token into the
2560-wide embedding. The table never reaches the device whole. The key and value projections are NVFP4; the gate is
`sigmoid(signed sqrt(key . query / sqrt(2560)))` per stream; the dilated depthwise convolution (4 taps, dilation 3) runs over
the normalized gated values with nine rows of history, which a sequence carries as `[9][10240]` BF16.
