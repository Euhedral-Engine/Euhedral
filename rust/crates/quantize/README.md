# Quantization tools

Every command below runs from the repository root as `cargo run --release --manifest-path rust/Cargo.toml -p euhedral-inference-quantize -- COMMAND [FLAGS]` (or `rust/target/release/quantization COMMAND [FLAGS]` once built). Run with no command to list them. The CUDA library defaults to `build/native/linux-x64/lib/libeuhedral_cuda.so` (build it with `(cd rust && cargo xtask native --product linux-x64)`); `--cuda-library FILE` (alias `--library`) names another. `--config YAML` takes the engine settings. Commands: `calibrate`, `score`, `generate`, `train-drafter`, `export-drafter`, `assemble`, `cost`, `allocate`, `reference` and `package-tokenizer`.

The quantization commands (`rust/crates/quantize`, binary `quantization`; the work is in `rust/crates/quantization`) run on the engine's CUDA kernels through the Euhedral lattice. They write pre-rounded codes
for the converter (`tools/convert_checkpoint.py --codes-from`); nothing here is part of the engine or the API.

## GPTQ calibration

```bash
cargo run --release --manifest-path rust/Cargo.toml -p euhedral-inference-quantize -- calibrate --model /path/to/checkpoint --quantization nvfp4 --compressed \
    --tokens tokens.npy --draft-ids-from qwen3_8_27b_q3.edrl --out codes/ --act-order
python3 tools/convert_checkpoint.py --model /path/to/checkpoint --quantization nvfp4 --compressed \
    --draft-ids-from qwen3_8_27b_q3.edrl --imatrix imatrix.gguf --codes-from codes/ --out qwen3_8_27b_nvfp4_compressed.edrl
```

`tokens.npy` holds int32 calibration sequences `[sequences, length]` (`tools/gptq/calibration.py` builds them). The
calibration runs the sequences through the BF16 checkpoint one decoder layer at a time:

1. the layer runs over every batch on the engine's kernels (norms, GDN convolution and recurrence, attention over the
   NVFP4 cache, SwiGLU; BF16 linears on cuBLAS), and each quantized linear's input rows accumulate into its Hessian
   (cuBLAS, BF16 products summed in FP32);
2. each object is rounded by GPTQ in the format and layout the recipe stores it (the recipe, the converter's rules):
   the upper Cholesky factor of the damped inverse Hessian comes from cuSOLVER, and the kernels in `rust/kernels/quantization/gptq.cu`
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
`--sequences N`, `--layers N` (a short run), `--skip-mtp`, `--cuda-library` (the engine's native library) and `--cuda-libraries DIR` (the CUDA math libraries, default `/usr/local/cuda/lib64` on Linux).

## Trellis calibration

```bash
cargo run --release --manifest-path rust/Cargo.toml -p euhedral-inference-quantize -- calibrate --quantization trellis --format trellis --bits 3.5 --model /path/to/checkpoint \
    --tokens tokens.npy --draft-ids-from artifact.edrl --out exl3/
```

Writes the checkpoint in exllamav3's EXL3 trellis format instead of GPTQ codes. `--bits B` is the average bits per
weight of the text linears: each linear gets a rate of 2 to 6 bits in half-bit steps (exllamav3's allocation), and the
rounding is the same layer-by-layer pass as above, with the layer's Hessians from the BF16 forward and each linear
rounded by the trellis quantizer (`rust/kernels/quantization/trellis.cu`; ported from exllamav3 1.6.1) as a lattice frame on its own stream.

Output (`--out`, loadable by exllamav3): one `model.language_model.layers.<i>.safetensors` per decoder layer,
`mtp.safetensors` (the MTP linears at 4 bits, uncalibrated), `lm_head.safetensors` (F16), the token embedding,
`model.visual.safetensors` (vision tower copied unquantized), `model.safetensors.index.json`, `config.json` with the
`quantization_config` block and `quantization_config.json`.

A run over the whole model also rounds the two heads by GPTQ, after the MTP linears and before the final files are
written: the LM head and the draft head (the shortlist of `--draft-ids-from`, which is required) in NVFP4 row-split
(`--output-head FORMAT` changes the LM head's format), both under the Hessian of the final norm's output over the
final hidden states. Their codes go to `exl3/heads/`, the `--codes-from` directory of the converter's
`--quantization trellis --trellis-from`. The resume point is dropped only after the heads are rounded and the final
files are written.

Resource flags:

- `--rounding-lanes N`: linears rounded at once (default: sized from the device memory free at each layer, less the
  reserve and the decoded weights the layer keeps).
- `--viterbi-scratch-mib N` (256): each quantizer's Viterbi history; it must hold at least one tile at the largest rate
  of the run (about 8.1 MiB at 2.5 bits, 4 MiB at 2), otherwise the run is refused.
- `--memory-reserve-mib N` (1024): device memory the lane sizing leaves free.
- `--write-buffer-mib N` (16): the buffer each output file is copied through.
- `--batch`, `--sequences`, `--layers`, `--cuda-library` as above.

`--format trellis` needs `--quantization trellis` (and `--quantization trellis` needs `--format trellis`) and `--bits`. It refuses `--skip-mtp`, because the checkpoint carries the MTP block, and `--compressed`, which the recipe rejects for trellis. The trellis options (`--bits`, `--viterbi-scratch-mib`, `--write-buffer-mib`) are refused without `--format trellis`. `--bits` and every
allocated rate are validated before any GPU work.

Resuming: the hidden states are checkpointed every 8 layers into one `hidden.bin` (overwritten in place) and a run with the same tokens, shape and `--bits` resumes after the last checkpoint; the layer files already
written are adopted rather than rounded again. The checkpoint is dropped only after the final files are written. The
MTP layer and the final files are written only when every layer runs (`--layers` below the model's layer count
stops after the layers, with no MTP file and no index).

## Scoring

```bash
cargo run --release --manifest-path rust/Cargo.toml -p euhedral-inference-quantize -- score --artifact qwen3_8_27b_nvfp4_compressed.edrl --results results.txt \
    --reference swift_1_5_bf16.kld --name t1 --reference swift_t2.kld --name t2
```

Teacher-forces every chunk of each llama.cpp reference (`llama-perplexity --kl-divergence-base`, docs/evaluation/QUALITY.md) on
the artifact and appends a summary per reference to the results file: mean NLL, dNLL, KL(reference || artifact) over
the reference entries above its 16-nat floor, top-1 agreement with standard errors (chunk-level over several chunks),
the KL median, p95, p99 and maximum, and delta-p of the forced token. Each chunk of N tokens prefills N / 2 + 1 and
verifies the rest `--rows` (8) forced tokens at a time; each logits row is scored where the engine left it, by
`rust/kernels/quantization/score.cu` reading the reference row a reader frame fetched into mapped host memory. `--batch` (4) chunks run together
on their own sequences, each step group of all of them in one batched verification (at most 32 rows).
`score` and `generate` keep weights in host memory only when the artifact, every sequence's KV and state and the
engine's reserve do not fit in the memory free on the GPU, the smallest selection that makes room, as the engine plans
it; `--host-mib N` sets the host-backed weights instead.

## MTP drafter corpus

```bash
cargo run --release --manifest-path rust/Cargo.toml -p euhedral-inference-quantize -- generate --artifact qwen3_8_27b_nvfp4_compressed.edrl --tokenizer /path/to/checkpoint \
    --out corpus/ --prompts prompts.jsonl --max-tokens 4096 --batch 8
```

Answers chat prompts (`{"id", "messages", "tools"?}` per line; `tools/gptq/calibration.py prompts ... --chat N`) with the
artifact's own greedy decode for the share `--greedy-fraction` of the prompts (default 0.25; the rest are sampled, and their labels are the greedy choices the sampled path passed over) and keeps what the MTP drafter learns from: every token, and the post-final-norm hidden
of every row the base model ran, which is what seeds the drafter (docs/speculation/MTP_CONTRACT.md §2). Prompts are rendered by
the checkpoint's chat template with thinking on and prefilled in the engine's 512-row chunks, as the server does; then
up to `--batch` sequences (at most 32) decode together, one row each per batched verification, so every row is bit
for bit the artifact's own one-row decode. The corpus is `tokens.bin`, `hidden.bin` (BF16) and
`index.jsonl`; a stopped run resumes by prompt id. The run's bookkeeping is a lattice source that quanta post records
to. Each sequence holds about 336 MiB of device memory (144 MiB of it GDN state), so a larger `--batch` moves more
weights to host memory on a 16 GB card. On NVFP4-Trellis-RCO, `--batch 8` answers at 258 tok/s with 1.1 GiB of weights in
host memory; `--batch 12` moves 3.0 GiB there and answers at 69 tok/s.

`tools/mtp_corpus_prompts.py` writes the drafter's prompts from training splits: UltraChat first turns and second
turns, Magicoder problems, Hermes function calls (the call itself, and the answer after the dataset's tool calls and
responses), coding-agent turns over pallets projects after the agent listed a package and read one file, Aya prompts
in five languages and GSM8K questions; none shares a user message with T4, the held-out set or the calibration set.
An assistant message of a prompt may carry OpenAI `tool_calls`, which the chat template renders.

## MTP drafter training

```bash
cargo run --release --manifest-path rust/Cargo.toml -p euhedral-inference-quantize -- train-drafter --corpus corpus/ --model /path/to/checkpoint \
    --artifact qwen3_8_27b_nvfp4_trellis_rco.edrl --out drafter/ --cuda-libraries /usr/local/cuda/lib64 \
    [--depths 4] [--rows 1024] [--keys 4096] [--tokens 16384] [--epochs 1] [--projection-rate 2e-5] \
    [--norm-rate 2e-5] [--warmup 10] [--holdout 50] [--evaluate-every 50]
cargo run --release --manifest-path rust/Cargo.toml -p euhedral-inference-quantize -- export-drafter --model /path/to/checkpoint \
    --artifact qwen3_8_27b_nvfp4_trellis_rco.edrl --out drafted.edrl [--parameters drafter/best.bin]
```

`train-drafter` trains every parameter of the MTP layer from the checkpoint's weights, in FP32 with Adam, on the
corpus (`rust/kernels/quantization/drafter.cu`). Its projections run NVFP4-rounded exactly as the export rounds
them, with straight-through gradients. The recursion is unrolled to `--depths` drafts as the decoder runs it: row r
of a window of `--rows` lineages reads the base model's hidden at position a + r, its depth-k row reads the depth-(k -
1) row's output and the token at a + r + k, sits at position a + r + k - 1 and attends the depth-1 keys of every position
up to a + r (from up to `--keys` positions back, earlier windows of the sequence included) and its own lineage's deeper
keys. Each depth's loss is the cross-entropy of the artifact's draft head (its NVFP4 shortlist, frozen) against the
base model's token, on answer tokens; token embeddings are the artifact's Q3 rows. An update follows every
`--tokens` trained lineages. Every `--holdout`-th sequence is held out: an evaluation counts, over held-out lineages
whose drafts all have targets, the leading drafts equal to the base model's tokens, the drafts a verification
accepts. `out/best.bin` keeps the best evaluation's parameters; a run resumes from `out/resume.bin`.

`train-drafter` saves each file to a sibling `.part` file and renames it over the target once whole, so a stop leaves
the old file or the new one (a leftover `.part` is overwritten by the next save). A start refuses, naming what differs,
when `out` holds a state of another corpus (a content digest, not a path), model, artifact or settings, a state that
was torn, or `best.bin` without a state; `--fresh` starts over and replaces them. `--poison-memory` fills memory
nothing has written with NaN, to show a read before a write, and `--fail-in-save N` fails the N-th state save halfway,
to prove that the saved state survives it. No time limit decides a run: it fails only when it finishes no window,
update, read or write over `--stall-seconds` (default 1800), and its lane is closed first.

`export-drafter` writes the base artifact with its MTP layer replaced: NVFP4 projections (round to nearest,
the same rounding, byte for byte the converter's `quantize_nvfp4_rows`) and BF16 norms, from a parameter file or, without
one, from the checkpoint's MTP weights.

## Other commands

`assemble`, `cost`, `allocate` and `reference` build and search mixed-format artifacts; `package-tokenizer` adds the tokenizer section ([OPERATIONS.md](../../../docs/OPERATIONS.md#artifacts-and-their-tokenizer)). The flags of each are in the usage text the tool prints, and the search is described in [the architecture](../../../docs/architecture/06-quantization.md).

## Tests

The tests of the quantization crate are in `rust/crates/quantization`: CPU tests run with `mise run rust-test`; the device tests (`kernels_gpu`, `generate_gpu`, `assemble_score_gpu`, `drafter_gpu`) are `#[ignore]` tests that need a GPU, the checkpoint and the fixtures, and run with `cargo test --release --manifest-path rust/Cargo.toml -p euhedral-inference-quantization --test <name> -- --ignored` ([testing](../../../docs/evaluation/TESTING.md)).
