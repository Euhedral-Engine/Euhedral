# Flash-Next routed-expert kernels

The routed experts of Flash-Next (`qwen4_exp`: 512 experts per layer, 10 per token) run on NVFP4 kernels that read the expert
records where the expert cache holds them ([FLASH_NEXT_RESIDENCY.md](FLASH_NEXT_RESIDENCY.md)). The kernels take work items (an
expert and up to 16 of the pairs, token row and routing weight, routed to it) and write each pair's weighted output; a combine
kernel then adds every row's outputs into the layer's output rows in ascending expert order. The model launches the item kernels
once per expert, as soon as that expert is in the cache, and the combine once after every expert ran
([FLASH_NEXT_LATTICE.md](FLASH_NEXT_LATTICE.md)). The router, the cache and the leases belong to the caller; this document is the
contract between the two.

```
native/src/qwen4/experts.cuh        kernels (module native/src/qwen4/kernels.cu, table native/src/host/qwen4_ops.c)
core/.../qwen4/Qwen4ExpertRouting   host planner: pairs by expert, each expert's items, the descriptor the kernels read
core/.../qwen4/Qwen4ExpertOps       typed launches, record geometry, scratch layout
```

## Arithmetic

Upstream (`Qwen4ExpTextExperts.forward`), per expert, for the rows routed to it:

```
gate, up = chunk(linear(x, gate_up), 2)          BF16, one rounding per output
act      = bf16(bf16(silu(gate)) * up)
y        = linear(act, down)                      BF16
weighted = bf16(y * routing_weight)               routing weight BF16
out.index_add_(rows, weighted)                    BF16 add, one rounding per add, experts visited in ascending id
```

The kernels compute exactly this chain:

- **Weights stay NVFP4** (`row-split-k128-v1`, [NVFP4_NATIVE.md](NVFP4_NATIVE.md)): `gate_up` `[1280][2560]` at record offset 0,
  `down` `[2560][640]` at record offset 1,843,456, each a complete tensor with its own FP32 global scale. The offsets are kernel
  arguments. No BF16 copy of an expert exists.
- **Activations are BF16 for every row count.** There is no activation quantization at any size. A weight (E2M1 code times its E4M3
  block scale) is exact in BF16, so every product with a BF16 activation is exact in FP32; the tensor core (`mma.m16n8k16`,
  BF16 inputs, FP32 accumulate) sums them, and the tensor's global scale multiplies the FP32 sum once, before the single BF16
  rounding of the output. This is the arithmetic of the dense decode kernels.
- Weight rows are the MMA's 16 M rows and token rows its 8 N columns. The result of a column never depends on the other columns,
  and the order in which K is summed is a function of the tile alone (below), so **the result for a (token, expert) pair is the same
  bits whatever else is in the launch**: the other tokens of its expert, the other experts of the launch, the work item it falls
  in and the column it occupies. An expert that saw 1 token and one that saw 300 compute each pair identically.
- `silu` is `x / (1 + expf(-x))` in FP32 of the BF16 gate, rounded to BF16; the product with `up` is rounded again. `weighted`
  rounds `y` to BF16, multiplies by the BF16 routing weight in FP32 and rounds.
- **Routed sum.** A row's sum is a sequence of BF16 additions, `out = bf16(out + weighted)`, starting from zero, in ascending
  expert id (and, for an expert listed twice for one row, ascending top-k position). The combine kernel adds a row's
  contributions in that order, from the row list the planner wrote, after every expert's kernels ran; the experts themselves may
  run in any order. Adding to zero is exact, so the sum starts from zeros (`zeroFirst`).
- The padding id (`experts`, one past the last) is skipped, as upstream skips it.

## Kernels

Three kernels: gate_up and down over the items of one expert (or of several), then the combine over every row.

| kernel | grid | block | job |
|---|---|---|---|
| `euhedral_q4_expert_gate_up_swiglu_bf16` | `(inter / 32, items)` | 128 | `act[pair][640]` for every pair |
| `euhedral_q4_expert_down_bf16` | `(hidden / 128, items)` | 128 | `weighted[pair][2560]` for every pair |
| `euhedral_q4_expert_combine_bf16` | `(ceil(hidden / 2048), rows)` | 256 | `out[row][2560] += weighted` in order |

A **work item** is one expert and up to 16 of its pairs (consecutive in the pair list). The grid's second dimension is the item
list, so a decode step (10 experts, 1 token each) is 10 items and a 512-row chunk is about 1.5 items per expert. A CTA of the
gate_up kernel computes 32 act columns of one item (4 warps split K: chunk `c` of 128 values on warp `c mod 4`, summed in warp
order; each warp owns two gate fragments and the two matching up fragments of 16 rows); a CTA of the down kernel computes
128 output rows (four warps, each two 16-row fragments over the whole K, no cross-warp sum). Pairs 0..7 of an item are MMA
column tile 0 and pairs 8..15 tile 1, which share each decoded weight fragment; an item of 8 or fewer pairs runs tile 0 only.
Lanes without a token load nothing, so a decode item reads only its weights. Shared memory is 17.4 KB (gate_up) and 9.2 KB
(down), 168 and 104 registers, no local memory.

K order is `W = 4` warps for gate_up and one warp for down. The other tile dimensions (fragments per warp, warps along rows) do
not touch K order; if a variant of them is added it stays bit-exact, but changing `W` changes the bits of every result and must
not be selected per launch.

Limits: `hidden` and `inter` multiples of 128 (K of the two tensors), `inter` a multiple of 32 and `hidden` of 128 (tile rows),
`hidden` a multiple of 8 (combine). Item counts 1..16. Any number of experts and items per launch.

## Planning and the descriptor

`Qwen4ExpertRouting(experts, topK, maxRows)` preallocates everything; `plan(rows, topKIds, topKWeights)` (`[rows][topK]` row-major,
ids `int`, weights BF16 bits as `short`) groups the pairs by expert (stable: ascending row, then top-k position). The experts that
received pairs are the chunk's *active* experts, in ascending id order; the `i`-th has its work items contiguous
(`itemStart(i)`, `itemCount(i)`), and they name slot `i` of the descriptor's slot table. Neither `plan` nor `fill` allocates.

`fill(hostDescriptor)` writes the descriptor except the slot addresses; the caller copies it to the device once. For each active
expert, once the expert is held, the caller writes its record address with `setSlot(hostDescriptor, i, slotAddress)`, copies that
entry to the device, and calls `Qwen4ExpertOps.runExpert(gpu, geometry, plan, i, descriptorDevice, x, scratch)`. After every expert
ran, `Qwen4ExpertOps.combineExperts` adds the rows.

The descriptor is one block, little-endian, 16-byte aligned sections (offsets from the planner's accessors):

| section | content | bytes |
|---|---|---|
| slots | device address of each active expert's record (`setSlot`) | 8 E |
| items | slot index, first pair, pair count (1..16), 0 | 16 I |
| pairs | row (int32), BF16 routing weight (int32, low 16 bits) | 8 P |
| row offsets | `rows + 1` offsets into the row lists (uint32) | 4 (T + 1) |
| row pairs | per row, its pair indices in ascending expert order (uint32) | 4 P |

with `P = maxRows * topK`, `E = min(experts, P)`, `I = P / 16 + E`, `T = maxRows`. The pair list is the pairs of the active
experts one after another; an item is `(slot, first, count)` into it.

## What the caller must honor

- **Alignment.** Every device address the kernels read or write (descriptor sections, `x`, `out`, scratch) is 16-byte aligned; the
  expert slots are 4096-aligned (cache slots are) and both record offsets are multiples of 256. `x` and `out` are `[rows][hidden]`
  BF16, rows contiguous, `rows <= maxRows`.
- **Scratch.** `Qwen4ExpertRouting.scratchBytes(rows * topK)` bytes of device memory: `act` `[pairs][640]` at offset 0, then
  `weighted` `[pairs][2560]` at the next 16-byte boundary after `act`, where `pairs` is the chunk's pair count (6,400 bytes per
  pair: 64 KB for a decode token, 32.8 MB for the 5,120 pairs of a 512-row chunk). Every expert writes its own pairs' rows, so
  experts may run side by side on different streams.
- **Order.** The combine runs after every expert's kernels (on its stream, behind their markers). If a chunk has no pairs at all
  (every id is padding) there is nothing to combine and the caller zeroes `out`.
- **Descriptor reuse.** The host descriptor must not be rewritten before the copies it sources have retired: the next chunk's
  plan runs only after the router of that chunk retired, which follows this chunk's combine.
- **Leases.** An expert's kernels read its slot until they have run; record a fence after `runExpert` on its stream and close the
  lease with it.

## Validation

- `native/tests/test_qwen4_experts.py` (numpy venv): both projections against float64 over the exactly expanded weights, bit equality
  of a pair across work items of 16, 8, 5, 3 and 1 pairs and in reverse order, and the combine order and `zeroFirst`.
- `Qwen4ExpertRoutingTest` (CPU): every pair appears once, in its expert's contiguous items that name the expert's slot, each
  row's addition list ascends in expert id, padding ids and repeated experts.
- `Qwen4ExpertCudaIntegrationTest` (synthetic NVFP4 experts): `act` and `weighted` of every pair against a CPU reference written from
  upstream; the routed sum equals, bit for bit, the BF16 chain over the kernels' own `weighted` values; results are **bit-identical**
  for a 70-row chunk whichever order the experts run in and through 1 to 12 slots, in chunks of 7 and 1 rows and in one piece; a
  512-row skewed chunk (up to 347 pairs per expert) is bit-identical in either order and agrees with the CPU on three rows.
- `Qwen4ExpertFixtureCudaIntegrationTest`: real layer-0 experts, records read straight from the artifact into 16 device slots reused
  from expert to expert, against the upstream fixtures `layer_moe` (`c0` 1 row, `c1` 8 rows, `c2` 64 rows). The upstream
  product is an FP32 matmul over exactly expanded weights, the kernels accumulate in FP32 in another order, so values agree to BF16
  rounding:

| | values | bit-equal | within one step* | worst |
|---|---|---|---|---|
| `weighted`, all experts | 1,868,800 | 99.60% | 99.94% | 9 steps, 7.6e-3 of the row's largest |
| routed sum | 186,880 | 99.54% | 99.79% | 16 steps, 6.6e-3 of the row's largest |

\* of the values above 1/64 of their row's largest. A BF16 rounding of a gate or up sum that lands on the other side changes one
activation by a step, which moves every output of the down projection by a fraction of a step; the largest deviations are such
outputs close to zero in cancelling sums.

## Measured

RTX 5070 Ti (sm_120, 70 SMs, 16 GB), CUDA 13.1, grouped launches (several experts per launch, the benchmark's "waves"), GPU
time between events
(`native/tests/bench_qwen4_experts.py`). The weights of small chunks are replicated over distinct slots until each replay reads
more than 250 MB, so they come from DRAM. "GB/s" is the stored expert bytes (2,764,808 per expert: gate_up 1,843,204 and
down 921,604) over the whole time.

| chunk | experts | waves | gate_up | down | combine | total | GB/s |
|---|---|---|---|---|---|---|---|
| decode, 1 row x 10 experts | 10 | 1 | 31 us | 16 us | 3 us | 50 us | 550 |
| 2 rows | 20 | 1 | 57 us | 29 us | 3 us | 89 us | 618 |
| 4 rows | 40 | 1 | 96 us | 50 us | 3 us | 167 us | 663 |
| 64 rows, uniform | 367 | 6 (64 experts) | 958 us | 486 us | 15 us | 1.49 ms | 681 |
| 64 rows, skewed (<= 47 pairs/expert) | 251 | 4 | 669 us | 318 us | 12 us | 1.03 ms | 674 |
| 512 rows, uniform (~10 pairs/expert) | 512 | 8 | 1.49 ms | 757 us | 43 us | 2.41 ms | 586 |
| 512 rows, skewed (<= 347 pairs/expert) | 499 | 10 | 1.65 ms | 892 us | 47 us | 2.61 ms | 529 |
| 512 rows, uniform, waves of 20 experts | 512 | 26 | 1.66 ms | 847 us | 94 us | 2.75 ms | 515 |

Single launches of one wave, 64 experts at one item each: gate_up 156 us for 118 MB (756 GB/s), down 69 us for 59 MB (860 GB/s);
the same experts with 9 to 16 pairs (one two-tile item each) 187 us and 94 us; with 24 pairs 260 and 153 us; with 64 pairs (4 items
of 16) 506 and 321 us, where the decode of the weights into BF16 fragments and the MMAs, not DRAM, bound the kernels.

Memory: the kernels allocate nothing (no local memory). The device cost of a chunk is its slots, the descriptor and the scratch
(6,400 bytes per pair).

Tuning notes (one-wave launches, cold weights; us for 10 experts x 1 pair / 64 experts x 9 pairs / 64 experts x 64 pairs):
- gate_up, `W = 4`, fragments per warp x warp rows: 2 x 1 (chosen) 28 / 187 / 506; 1 x 2 29 / 198 / 670; 4 x 1 30 / 177 / 483;
  5 x 1 33 / 181 / 472. Larger tiles amortize the activation reads and win by up to 8% in prefill and lose up to 7% in decode.
  `W = 8` with one fragment per warp is about 10% faster at decode and 10% slower in prefill; it changes K order, so it would
  have to be the only choice.
- down, one warp per K: 2 fragments x 4 warp rows (chosen) 15.7 / 92 / 321; 1 x 8 14.8 / 88 / 381; 4 x 4 17.7 / 89 / 292; 5 x 4
  18.7 / 87 / 274.
- Two column tiles per item (16 pairs) against one (8): no change at decode, nothing at 9 to 16 pairs per expert (the activation
  reads, as many bytes as the weights, bound the kernel), 7% at 24 and 27% at 64 pairs per expert.

## Known gaps

- Gate_up reaches 60 to 70% of DRAM bandwidth when a launch has few items (decode and 2 to 4 rows); more bytes in flight per SM
  (a deeper K pipeline with `cp.async`) are not implemented.
- Compute-bound cases (an expert with 64 or more pairs) run at roughly 40% of the BF16 tensor-core peak, limited by decoding NVFP4 into
  BF16 fragments. Native FP4 MMA is not used: it needs FP4 activations, which would make a pair's result depend on the
  quantization and break the contract above.
- Only `row-split-k128-v1` plain NVFP4 experts are supported (no SD4 scale tables).
