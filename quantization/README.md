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

## Trellis calibration

```bash
./gradlew :quantization:run --args="calibrate --format trellis --bits 3.5 --model /path/to/checkpoint \
    --tokens tokens.npy --out exl3/"
```

Writes the checkpoint in exllamav3's EXL3 trellis format instead of GPTQ codes. `--bits B` is the average bits per
weight of the text linears: each linear gets a rate of 2 to 6 bits in half-bit steps (exllamav3's allocation), and the
rounding is the same layer-by-layer pass as above, with the layer's Hessians from the BF16 forward and each linear
rounded by the trellis quantizer (`trellis.cu`; ported from exllamav3 1.6.1) as a lattice frame on its own stream.

Output (`--out`, loadable by exllamav3): one `model.language_model.layers.<i>.safetensors` per decoder layer,
`mtp.safetensors` (the MTP linears at 4 bits, uncalibrated), `lm_head.safetensors` (F16), the token embedding,
`model.visual.safetensors` (vision tower copied unquantized), `model.safetensors.index.json`, `config.json` with the
`quantization_config` block and `quantization_config.json`.

Resource flags:

- `--rounding-lanes N`: linears rounded at once (default: sized from the device memory free at each layer, less the
  reserve and the decoded weights the layer keeps).
- `--viterbi-scratch-mib N` (256): each quantizer's Viterbi history; it must hold at least one tile at the largest rate
  of the run (about 8.1 MiB at 2.5 bits, 4 MiB at 2), otherwise the run is refused.
- `--memory-reserve-mib N` (1024): device memory the lane sizing leaves free.
- `--write-buffer-mib N` (16): the buffer each output file is copied through.
- `--batch`, `--sequences`, `--layers`, `--library` as above.

The GPTQ-only flags (`--skip-mtp`, `--act-order`, `--damp`, `--quantization`, `--compressed`, `--output-head`,
`--draft-ids-from`, `--nvfp4-above`, `--nvfp4-scale-max`) are rejected with `--format trellis`. `--bits` and every
allocated rate are validated before any GPU work.

Resuming: the hidden states are checkpointed every 8 layers into one `hidden.bin` (a `ResumeFile`, overwritten in
place) and a run with the same tokens, shape and `--bits` resumes after the last checkpoint; the layer files already
written are adopted rather than rounded again. The checkpoint is dropped only after the final files are written. The
MTP layer and the final files are written only when every layer runs (`--layers` below the model's layer count
stops after the layers, with no MTP file and no index).

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
`score.cu` reading the reference row a reader frame fetched into mapped host memory. `--batch` (4) chunks run together
on their own sequences, each step group of all of them in one batched verification (at most 32 rows).
`score` and `generate` keep weights in host memory only when the artifact, every sequence's KV and state and the
engine's reserve do not fit in the memory free on the GPU, the smallest selection that makes room, as the engine plans
it; `--host-mib N` sets the host-backed weights instead.

## MTP drafter corpus

```bash
./gradlew :quantization:run --args="generate --artifact qwen3_8_27b_nvfp4_compressed.edrl --tokenizer /path/to/checkpoint \
    --out corpus/ --prompts prompts.jsonl --max-tokens 4096 --batch 8"
```

Answers chat prompts (`{"id", "messages", "tools"?}` per line; `tools/gptq/calibration.py prompts ... --chat N`) with the
artifact's own greedy decode and keeps what the MTP drafter learns from: every token, and the post-final-norm hidden
of every row the base model ran, which is what seeds the drafter (docs/MTP_CONTRACT.md §2). Prompts are rendered by
the checkpoint's chat template with thinking on and prefilled in the engine's 512-row chunks, as the server does; then
up to `--batch` sequences (at most 32) decode together, one row each per batched verification, so every row is bit
for bit the artifact's own one-row decode. The corpus (`Corpus`) is `tokens.bin`, `hidden.bin` (BF16) and
`index.jsonl`; a stopped run resumes by prompt id. The run's bookkeeping is a lattice source that quanta post records
to. Each sequence holds about 336 MiB of device memory (144 MiB of it GDN state), so a larger `--batch` moves more
weights to host memory on a 16 GB card.

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
  `-Peuhedral.quantization.forward-fixture=DIR`).
- `ForwardPipelineCudaIntegrationTest`: the frame pipeline equals the sequential forward bit for bit.

```bash
./gradlew :quantization:cudaIntegrationTest --rerun -Peuhedral.quantization.checkpoint=/path/to/checkpoint \
    --tests '*GptqCudaIntegrationTest' --tests '*LayerForwardCudaIntegrationTest'
```
