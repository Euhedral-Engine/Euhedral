# Quantization tools

Calibration tools that run on the engine's CUDA kernels through the Euhedral lattice. They write pre-rounded codes
for the converter (`tools/convert_checkpoint.py --codes-from`); nothing here is part of the engine or the API.

## GPTQ calibration

```bash
./gradlew :quantization:run --args="calibrate --model /path/to/checkpoint --quantization nvfp4 --compressed \
    --tokens tokens.npy --draft-ids-from qwen3_8_27b_q3.edrl --out codes/ --act-order"
python3 tools/convert_checkpoint.py --model /path/to/checkpoint --quantization nvfp4 --compressed \
    --draft-ids-from qwen3_8_27b_q3.edrl --imatrix imatrix.gguf --codes-from codes/ --out qwen3_8_27b_nvfp4_compressed.edrl
```

`tokens.npy` holds int32 calibration sequences `[sequences, length]` (`tools/gptq/calibration.py` builds them). The
calibration runs the sequences through the BF16 checkpoint one decoder layer at a time:

1. the layer runs over every batch on the engine's kernels (norms, GDN convolution and recurrence, attention over the
   NVFP4 cache, SwiGLU; BF16 linears on cuBLAS), and each quantized linear's input rows accumulate into its Hessian
   (cuBLAS, BF16 products summed in FP32);
2. each object is rounded by GPTQ in the format and layout the recipe stores it (`Recipe`, the converter's rules):
   the upper Cholesky factor of the damped inverse Hessian comes from cuSOLVER, and the module's kernels (`gptq.cu`)
   round each run of columns between two scale choices and carry its error into the rest of the block, with the
   converter's weighted scale searches for Q3/Q4/Q5 G64, NVFP4 and NVFP4-SD4;
3. the decoded weights replace the layer's, and the sequences run through it again to give the next layer its inputs.

The LM head is then rounded under the final norm's output; the MTP layer on the rows the engine feeds it (the normed
hidden at p with the embedding of token p + 1); the draft head under the MTP output it reads. The token embedding is
left to the converter.

Concurrent work runs as frames on a lattice of six workers, placed on the CPUs the process may run on (`taskset` or
systemd's `AllowedCPUs` keeps CPUs out): the next layer is read from the checkpoint while the GPU runs the current one,
and a layer's objects round three at a time on their workers' own streams. The hidden states stay in pinned host memory and are checkpointed every 8 layers; a run with the same
settings resumes there.

Options: `--act-order` (columns by decreasing diag H), `--damp` (0.01), `--batch` (sequences per batch, 8),
`--sequences N`, `--layers N` (a short run), `--skip-mtp`, `--library` (the engine's native library).

## Scoring

```bash
./gradlew :quantization:run --args="score --artifact qwen3_8_27b_nvfp4_compressed.edrl --results results.txt \
    --reference swift_1_5_bf16.kld --name t1 --reference swift_t2.kld --name t2"
```

Teacher-forces every chunk of each llama.cpp reference (`llama-perplexity --kl-divergence-base`, docs/QUALITY.md) on
the artifact and appends a summary per reference to the results file: mean NLL, dNLL, KL(reference || artifact) over
the reference entries above its 16-nat floor, top-1 agreement with standard errors (chunk-level over several chunks),
the KL median, p95, p99 and maximum, and delta-p of the forced token. Each chunk of N tokens prefills N / 2 + 1 and
verifies the rest `--rows` (8) forced tokens at a time; each logits row is scored where the engine left it, by
`score.cu` reading the reference row a reader frame fetched into mapped host memory. `--concurrency` (3) chunks run at
once on their own sequences; `--host-mib` (2304) keeps that much of the weights in host memory.

## Libraries

`cudaMathLibraries` downloads cuBLAS 13.2, cuSOLVER 12.0 and nvJitLink 13.1 from NVIDIA's CUDA 13.1 redistributables
into this module's build directory (Linux x86-64); `run` and the CUDA test tasks put them on the library path. The
module binds them, the CUDA driver API and NVRTC with the FFM API, in the engine's CUDA context and on its streams.

## Tests

- `SafetensorsTest`, `RecipeTest`: CPU.
The fixtures below are frozen outputs of a PyTorch reference of the calibration (`tools/gptq`, last at commit
9ff02525).

- `Qwen38ObjectsTest`: the fused objects of two layers and the MTP layer equal the reference's byte for byte
  (`-Peuhedral.quantization.checkpoint=DIR`).
- `GptqCudaIntegrationTest`: one object in Q3, Q5, NVFP4 and NVFP4-SD4, with and without act-order, against the
  reference (fixture in the test resources): tensor scales exactly, codes by agreement and output error under H.
- `LayerForwardCudaIntegrationTest`: layers 0 and 3 against the reference's BF16 forward (fixture in
  `/mnt/shared/qwen38-quant/quality/fixtures/forward-fixture`, or `-Peuhedral.quantization.forward-fixture=DIR`).
- `ForwardPipelineCudaIntegrationTest`: the frame pipeline equals the sequential forward bit for bit.

```bash
./gradlew :quantization:cudaIntegrationTest --rerun -Peuhedral.quantization.checkpoint=/path/to/checkpoint \
    --tests '*GptqCudaIntegrationTest' --tests '*LayerForwardCudaIntegrationTest'
```
