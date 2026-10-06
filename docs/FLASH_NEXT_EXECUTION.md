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

## Gated DeltaNet

A GDN layer (36 of the 48) projects the mixed input to `qkv` (NVFP4, 10,240), `z` (6,144), and the BF16 `a` and `b` (48 each); a
causal depthwise convolution (4 taps) with SiLU runs over `qkv` and keeps three rows of history; the decay `exp(-exp(A_log) *
softplus(a + dt_bias))` and `beta = sigmoid(b)` feed the dense engine's delta-rule recurrence (FP32 state `[48][128][128]`, key
heads shared by three value heads); the output is RMS-normalized per head, gated by `sigmoid(z)` (the dense engine gates by SiLU)
and projected back. The convolution, control and gated norm are Flash-Next kernels because upstream rounds to BF16 at different
places and fuses the QKV projection. Rows run in order with the state carried, so any chunking gives one result.

## Sparse attention (QSA)

Every fourth layer (12 in all) is a sparse-attention layer: grouped-query attention (24 query heads, 2 KV heads of 256, partial
RoPE of 64 values, per-head Q/K RMSNorm, a sigmoid output gate) over a *selected* part of the history. An indexer chooses the
part: a BF16 projection gives four indexer queries and one raw key of 128 values per token; keys are pooled in blocks of four
tokens (mean of the raw keys, `k_layernorm`, RoPE at the block's first position), a query scores every complete block it sees
with `sum over heads of relu(q . key) / sqrt(128)`, and the top 512 blocks (2,048 tokens) plus the incomplete trailing block
are attended. While the history holds at most 512 blocks every block is selected and attention is dense over the visible
tokens; beyond that the selection is purely score-ranked, so a query whose own block is not among the best 512 does not attend
to itself (upstream's behavior, reproduced).

The sequence keeps the NVFP4 key/value pages (576 bytes per token) and one pooled BF16 key per block (64 bytes per token); the up
to three raw keys of an incomplete block are carried between chunks, so a block key is computed once, when it completes, and any
chunking of a sequence gives the same result. Selection scores a tile of rows at a time (at most 16 MiB of FP32 scores) and
selects with a digit search over the scores; ties go to the lower block. The attention kernel reads the selected blocks and the
tail directly from the pages: no dense matrix and no expanded token list. Arithmetic, tolerances and timings:
[FLASH_NEXT_QSA.md](FLASH_NEXT_QSA.md).

## Mixture of experts

```
router logits (BF16 [512]) -> softmax(FP32) -> top 10 (ties: lower expert) -> renormalize -> BF16 weights
shared expert (NVFP4 MLP, 640) * sigmoid(gate)                                          every token
routed experts (NVFP4 records in the expert cache), BF16 products, summed in ascending expert order
out = routed + gated shared
```

`Qwen4MoeLayer` runs one block. The router and the shared expert are queued on the layer's stream, followed by the copy of the
routing (ids and weights) to the host. That is the block's one host wait: which experts to bring in is known only on the host.

**Expert waves.** A chunk of 512 tokens names up to 512 distinct experts, more than a minimal cache (20 slots) holds, and the
expert records may not all be resident. `Qwen4ExpertWave` groups the (token, expert) pairs by expert, orders the experts by id
and cuts them into waves of at most `min(cache slots, 32)` experts. For each wave the layer leases its experts from the cache,
writes a descriptor (slot addresses, work items of up to 16 pairs, the pairs, per-row pair lists) into pinned memory, copies it
to the device, and runs three kernels over it: gate/up with SwiGLU, the down projection with the routing weights, and a
combine that adds each row's weighted results to the running routed sum in ascending expert order, one BF16 addition at a time
as upstream's `index_add_` does. No temporary memory scales with the 512 experts' weights: a wave needs its slots (already in the
cache) and `6,400 * pairs` bytes of activations.

**Lease lifetime.** The kernels of a wave read the slot addresses from the descriptor. After queueing them the layer records a
marker on its stream and closes every lease with a `StreamFence` on that marker. The cache's copy stream waits for the marker on
the device before it refills a slot, so the host never waits for the kernels, a lease is held only while a wave is being
submitted, and no device pointer into a slot is used after the lease. Experts that are hits cost nothing; misses are loaded by
the cache's own copy stream and the acquirer waits for that copy, not for any compute.

**Exactness.** The expert kernels read NVFP4 weights in place and take BF16 activations at every row count (no FP4 activation
quantization): the bits of a (token, expert) pair depend only on the token's own activations, never on how many other tokens
the expert saw, which wave it ran in or what was in the cache. `Qwen4MoeFixtureCudaIntegrationTest` runs layer 0 with the
20-slot minimum cache (every chunk evicts) and with 520 slots and requires identical bits.

Kernel contract and measurements of the expert kernels: [FLASH_NEXT_EXPERTS.md](FLASH_NEXT_EXPERTS.md).

## Execution plan, sequence state, prefill and decode

`Qwen4ExecutionPlan` describes how a loaded `Qwen4Model` runs one chunk of up to 512 tokens: the embedding gather (the table may be in
device memory or host-mapped), the repetition over four streams, 48 layers, then, when logits are wanted, the final mix of the
last row and the output head. Prefill chunks and decode steps are the same code over different row counts; the output of a step
is the logits row of its last token, offered to a `LogitsSink` that queues its copy behind the head. A chunk is a quantum of a
static stage graph that the lattice's runtime runs ([FLASH_NEXT_LATTICE.md](FLASH_NEXT_LATTICE.md)): the router's choice reaches the
host across a device-completion edge (the shared expert is a side branch that overlaps the host's planning), each wave's experts are
loaded by a stage that completes when they are resident, the wave's kernels follow, and the step ends with the single boundary the
runtime arms after the output head. Each graph owns a workspace (state, mixer, block output, hyper-connection, layer and MoE
scratch, about 260 MiB at 512 rows), allocated once with the plan.

**Sequence state** (`Qwen4Sequence`) is what one sequence carries between steps: the FP32 recurrent state and three rows of
convolution history of each of the 36 GDN layers (114 MiB, independent of length), the NVFP4 key/value pages and pooled indexer
keys of each of the 12 attention layers (640 bytes per token, pages reserved as the sequence grows), the per-layer embedding's
convolution history and n-gram context, and the position. Expert residency is not in it: it belongs to the model's cache, so any
number of sequences share it and none retains a slot. The residency plan's accounting (`Qwen4SequenceState`) is what a sequence
actually allocates (`aSequenceHoldsWhatThePlanReserved`). The KV chunk of a step is committed when the step has retired and discarded
when it failed, so a failed step leaves the sequence where it was; closing releases every buffer.

**Prefill** cuts a prompt into chunks of at most 512 tokens. A chunk's rows run through GDN in order (so the state is the one a
token-by-token run would reach), through attention with per-row selection, and through the MoE block in expert waves. Any
chunking gives the same result up to BF16 noise, because the engine's NVFP4 linears switch kernels at nine rows (activations are
quantized from nine rows on): the logits of one prompt differ by about 7% to 10% relative RMS between chunkings, the reference's own
chunk-to-chunk difference being 9%, with the same greedy token (`Qwen4InvarianceCudaIntegrationTest`). The expert cache is another
matter: the experts' kernels are row-exact and independent of what is resident, so a cold cache, a warm cache and the 20-slot
minimum cache with 10,019 evictions produce identical bits.

**Decode** is a one-row step from the prefilled state; the sampler is the engine's, outside the model.

**Dynamic boundaries.** The step runs uncaptured. For the performance work that follows, the parts that change from step to step
are: the experts each layer needs (a host decision after the routing readback, and the slot addresses of the leases it obtains),
the position (kernel arguments of the attention and RoPE kernels, and the block counts that size the selection), and the KV
pages the position reaches. The stable parts are the layer kernels' shapes and the addresses of every workspace buffer. A captured
decode step would therefore split at each MoE block into a graph of the layer's attention part and a graph of its expert part
whose slot addresses come from a device-side table the host fills; the cache's slot addresses are stable, so the table can be
written per layer without changing the graph, and fences stay device-ordered either way.

## Engine and API

A Flash-Next artifact is selected by the artifact alone: `InferenceEngine` recognises the `qwen4_exp` architecture, opens the
storage, planner, expert cache and `Qwen4ExecutionPlan` (`Qwen4Runtime`), and hands the API a `GenerationSession`. The session
interface is the part of generation the API needs (prefill chunks, decode steps, cancellation, the sampler's logits); the dense
model's session and `Qwen4GenerationSession` both implement it, and the API's request handling contains no model-specific
branches. A generation is a chain of continuations on the lattice, not a loop on a thread: each step's end samples, emits text and starts the
next step; the plan runs one step at a time and a step admitted meanwhile starts when the running one concludes. Cancellation stops the
running step at its next stage. The runtime owns no executor, pool or thread; prompt tokenization and the host work a request needs
are frames on the same lattice. The engine therefore needs worker CPUs like the dense model, and a Flash-Next deployment benefits
from several: the expert reads of one wave run on as many workers as the staging pool allows. Chat completions on the real artifact are covered by `Qwen4ChatCompletionsCudaIntegrationTest`.

## End-to-end agreement

`Qwen4GreedyAgreementCudaIntegrationTest` compares the engine's greedy continuation with the reference's, both reading the KV
cache through the NVFP4 codec (`tools/flash_next_reference/cli.py greedy`, fixture `core/src/test/resources/qwen4/greedy-reference.json`).
Teacher-forced (the engine fed the reference's tokens) and free-running (the engine fed its own), 72 of 72 positions agree, and the
engine's tokenisation of every prompt matches the reference tokenizer's. A position where the reference's top two logits are within
the BF16 noise of the comparison (1.0 logit) is not counted as a disagreement; none of the 72 needed it.

## Measured performance

Measured by `Qwen4PerformanceCudaIntegrationTest` (`EUHEDRAL_QWEN4_PERF=1`, report in `core/build/qwen4-performance.txt`) on a
16 GB GPU with the artifact on a file-backed expert store (52 MiB pinned host memory), a 5120-token context plan, 3547 of 24576
expert slots resident (9366 MiB), 4190 MiB of fixed weights, and the step uncaptured. The decode figures are at 64 and at 4096
tokens of context; 16K and 32K were not measured.

| Phase | Result |
| --- | --- |
| Prefill, 512 tokens (one chunk) | 5.3 s, 97 tokens/s |
| Prefill, 4096 tokens (8 chunks of 512) | 42.8 s, 96 tokens/s |
| Decode at 64 tokens of context | 99.8 ms/token, 10.0 tokens/s |
| Decode at 4096 tokens of context | 99.9 ms/token, 10.0 tokens/s |
| Device memory, allocated / peak | 13,704 MiB / 13,849 MiB |

Expert traffic. Prefill: every chunk of 512 rows touches 314 of 512 experts per layer on average (211 to 429 depending on the
layer), so the cache never hits (0.0%) and each token moves 29.4 expert misses, 81.5 MB host to device. Decode: 10 experts per layer per
token, 72.9% hit rate at 4096 tokens of context, 129.8 misses and 359.5 MB per token (75.6% and 324.9 MB at 64 tokens). The
hit rate rises with depth (33% in layer 0, 93% in layer 47). After the 4096-token prefill and both decodes the cache had served
29,206 hits, 533,081 misses and 529,534 evictions.

Component time, 1-row step at 4096 tokens of context (ms per step): embedding 0.07, per-layer embedding 0.35, gated residual
10.62, GDN 4.14, QSA 1.84, MoE 104.50 (of which routing readback 13.7 and expert acquisition 77.8), output head 1.63. A
512-row step: embedding 0.35, per-layer embedding 719.68, gated residual 115.74, GDN 25.51, QSA 20.07, MoE 4298.18, output head 1.62.
The per-layer embedding figure of a 512-row step is the n-gram record gather from the mapped file.

## Known bottlenecks

- **Expert transfers.** Every number above is bound by the PCIe copy of experts the cache does not hold: decode moves
  about 3.6 GB/s into 2.64 MiB slots (360 MB per 100 ms step) and prefill about 7.8 GB/s, with the lease wait serial per layer.
- **Serial routing.** Each layer reads its routing back, then acquires its experts; the next layer's attention does not start before
  that. About 91 ms of the 100 ms decode step is routing wait plus acquisition.
- **Prefill shape.** A 512-row chunk uses most experts of every layer, so the cache is a stream, not a cache; the cost per token is
  the transfer of the layer's experts divided by the chunk size.
- **Small launches.** The gated residual (four-stream mix and its low-rank gates) takes 10.6 ms of a decode step as many small
  launches; the step is uncaptured.
- **Per-layer embedding.** 720 ms of a 512-row prefill step is the host gather of n-gram records.
- **Planner workspace.** The planner reserves 130 MiB for the plan's workspace; the plan measures about 260 MiB at 512
  rows, so the plan's bound is too small by that much (the runtime reserve absorbs it).

## Next optimisation priorities

1. Pipelined expert acquisition: start the next wave's leases while the current wave computes, and prefetch a layer's likely
   experts from the previous layer's routing.
2. Larger prefill chunks, so the transfer of a layer's experts is shared by more tokens.
3. Capture a decode step as graphs split at the MoE blocks (see the dynamic boundaries above), with the slot table written per layer.
4. A replacement policy that uses the measured per-layer reuse (layer 47 reuses 93% of its experts, layer 0 33%).
5. Overlap the n-gram record gather with the previous chunk.
6. Fuse the gated residual's launches; tighten the planner's workspace bound with the measured figure.
7. Measure 16K and 32K contexts once 1 to 4 have landed.
