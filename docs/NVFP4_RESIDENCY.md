# NVFP4 residency budget

The question was whether NVFP4 weights can run near Q3 speed with real VRAM headroom and MTP retained, by keeping NVFP4 as the
execution representation and streaming part of the weights from pinned host RAM ahead of use. The engine now makes this choice
itself: `ResidencyPlanner` keeps in host memory the fewest weights that let `maxContextTokens` of KV cache fit (see Host-backed
weights).

**Conclusion:** Single-sequence decode reads every weight every token, and the platform's host-to-device rate cannot stream the
useful offload (at least 1.8 GiB) within Q3's 16 ms token. It takes 44 ms even at the 44 GB/s that huge-page pinned memory
reaches; 4 KiB-page pinned memory under the IOMMU stalls at 25 GB/s.

**Staged NVFP4 can match non-MTP Q3 only with MTP accepting about 2.75 or more tokens per verification step**; MTP3 on NVFP4
accepts 2.84 tokens per verification ([MTP_SPECULATIVE.md](MTP_SPECULATIVE.md)). Prefill can hide the transfers.

Measured on 2026-10-01 on an RTX 5070 Ti (16 GB) with an i9-14900K.

## Memory budget

| Item | Size |
|---|---|
| Device total | 15.56 GiB (16,703,946,752 bytes) |
| Other processes (desktop) | about 0.46 GiB |
| CUDA context, NVRTC modules and engine after loading | about 0.75 GiB |
| Graph and workspace pools after the first run | +0.16 GiB (they persist) |
| KV (NVFP4 pages, 16 attention layers, 4 KV heads, D256, 144-byte rows) | 18 KiB per token: 0.56 GiB at 32K, 2.25 GiB at 128K |
| GDN state (48 layers, 48 x 128 x 128 FP32, plus convolution) | about 0.15 GiB per sequence |
| `q3` artifact payload (text, LM head, MTP layer, draft head) | 11.72 GiB |
| `nvfp4` artifact payload (Q3 embedding) | 14.52 GiB; about 0.19 GiB free after loading, so it cannot run fully resident |
| `nvfp4-compressed` artifact payload (SD4 scales) | 13.77 GiB |
| MTP layer + draft head, NVFP4 | 0.57 GiB |

The `nvfp4-compressed` artifact fully resident leaves about 0.4 GiB for KV and per-sequence state, roughly 15K tokens. Matching
Q3's headroom takes at least **1.8 GiB** of weights off the GPU (13.77 - 11.98 GiB, the artifact sizes at the time of the measurement), and more
once staging buffers are counted; all transfer times below use 1.8 GiB.

**Lossless compression of NVFP4 is limited.** The E2M1 codes carry 3.90 of their 4 bits, because the non-uniform FP4 grid already
equalizes code use. The block scales carry 3.63 of their 8. Only the scales compress usefully, saving 0.75 GiB
([NVFP4_COMPRESSED.md](NVFP4_COMPRESSED.md)).

## Platform

**Link: PCIe Gen 5 x16, about 63 GB/s theoretical per direction.** Under transfer load sysfs
reports 32 GT/s x16 for both the GPU (`01:00.0`) and its root port (`00:01.0`), and `nvidia-smi`
reports gen 5 current and max, x16. The link drops to 2.5 GT/s when idle.

**The GPU hangs off the CPU's own PCIe root port.** `nvidia-smi topo -m` shows CPU affinity 0-31 on
NUMA node 0, and the PCI tree shows no switch and no chipset path. The CPU-attached NVMe has its own
root port (`00:06.0`), so it does not split the GPU's lanes.

**IOMMU: Intel VT-d in translated mode.** `dmar0` is active, all 24 groups are `DMA-FQ`, and the
kernel command line has no `iommu=pt`.

**Host DRAM: about 52-55 GB/s read.** A STREAM-style test with 16-32 threads gives copy 47 GB/s,
triad 49-53 GB/s and read 52-55 GB/s.

All copies below use genuinely pinned memory (`cudaHostAlloc`, or `cudaHostRegister`), as raw
`cudaMemcpyAsync` timed with events, 1 GiB transfers:

| Pinned host memory | H2D | D2H |
|---|---|---|
| `cudaHostAlloc` (4 KiB pages; also write-combined and portable variants) | 25.0-25.4 GB/s | 22.5 GB/s |
| `malloc` + `cudaHostRegister`, 4 KiB pages | 24.8 GB/s | 21.7 GB/s |
| `malloc` + `cudaHostRegister`, 2 MiB transparent huge pages | **43.9-44.1 GB/s** | **56.4 GB/s** |

Two copy streams over huge-page buffers give the same 43.7 GB/s.

**What limits the 4 KiB-page copies is IOMMU translation.** Translated DMA over 4 KiB pages
throttles the copy engine at about 25 GB/s. Huge pages let the IOMMU map 2 MiB at a time, and D2H
then reaches the link's practical rate.

H2D stays at 44 GB/s. That is GPU-initiated reads of host memory, which run below D2H's posted
writes, plus any translation cost left. Booting with `iommu=pt` would separate the two.

**Staging buffers must be huge-page-backed pinned memory**, registered with `cudaHostRegister`.
`cudaHostAlloc` alone caps at 25 GB/s here.

## Bandwidth budget

**Host-to-device: 44 GB/s** with huge-page pinned memory (25 GB/s with 4 KiB pages). Transfer times
below use 44 GB/s.

Streaming cost of each family per decode token, from host RAM, in compressed bytes (4.26 bits per
weight):

| Family | Bytes per layer | PCIe time per layer (44 GB/s) | Whole family |
|---|---|---|---|
| FFN gate_up (x64) | 94.9 MB | 2.16 ms | 5.66 GiB |
| FFN down (x64) | 47.4 MB | 1.08 ms | 2.83 GiB |
| GDN value_z (x48) | 33.5 MB | 0.76 ms | 1.50 GiB |
| GDN query_key (x48) | 11.2 MB | 0.25 ms | 0.50 GiB |
| GDN output (x48) | 16.7 MB | 0.38 ms | 0.75 GiB |
| attention query_key (x16) | 19.5 MB | 0.44 ms | 0.29 GiB |
| attention gate_value (x16) | 19.5 MB | 0.44 ms | 0.29 GiB |
| attention output (x16) | 16.7 MB | 0.38 ms | 0.25 GiB |
| LM head (x1) | 676.8 MB | 15.38 ms | 0.63 GiB |

`q3` decode takes 16.0 ms per token (62.7 tok/s at 64 + 128), about 0.25 ms per layer for
every family together.

## Why decode cannot be hidden

**There is no reuse window.** Every token needs every layer, so a staged weight is read once and
evicted. The time between the earliest safe prefetch and its use does not matter; the bytes per
token do. With perfect overlap a token takes max(GPU time, PCIe time).

**Fully resident, NVFP4 reads more per token than Q3.** It reads 13.4 GiB of weights per token against Q3's 10.7 GiB. At the same
DRAM efficiency that is about 17.6 ms per token before any offload.

**Offload hides only up to the GPU time.** At 44 GB/s, 0.72 GiB per token stays under 17.6 ms
(0.4 GiB at 25 GB/s). The minimum useful offload, 1.8 GiB, is 44 ms per token: 2.7x Q3's token time.
Matching Q3 would need 120 GB/s, nearly twice PCIe 5.0 x16's theoretical rate.

**MTP does not change the ratio.** It divides streamed bytes and GPU work by the same number of
tokens accepted per verification step. Against non-MTP Q3 (16 ms per token), offloaded NVFP4 would
need about 2.75 accepted tokens per step to break even (4.8 at 25 GB/s).

## Prefill

A chunk reuses its weights across all its tokens. 1.8 GiB is 44 ms per chunk, against about 417 ms
of compute for a 512-token chunk on the former BF16-expansion path (about 204 ms at the native route's 2506 tok/s), so overlapped
transfers hide it.
Prefill-only staging is not useful alone: decode is the binding regime.

## Fully resident NVFP4

Executed objects only (no MTP layer or draft head in memory), the NVFP4 base model runs from the GPU. Measured 2026-10-01, one
session, warmup 1, 2 measured iterations, serving stopped: decode 48.1 tok/s with a 64-token prompt and 45.4 tok/s with an 8K
prompt, an engine allocation of 14.05 GiB and 0.62 GiB of device memory free.

**NVFP4 decode is bound by its weight bytes.** It reads about 25% more weight bytes per token than Q3, which decoded at 59.3
tok/s in the same session: 59.3 / 1.25 = 47.4 tok/s predicts the measured 48.1, so the NVFP4 GEMV streams at Q3's efficiency, and
no kernel work can close the gap at batch 1. Greedy generation answers "The capital of France is" with " Paris."

## Host-backed weights

When the planned weights do not fit, `HostWeightSelection` keeps layer projections in pinned host memory and stages them into a
device ring of `ResidencyPlanner.STAGING_SLOTS` = 4 slots on every use.

- **Planner:** the device budget is the resident weights, the staging ring (4 slots of the largest selected tensor), the KV pages
  of one sequence of `maxContextTokens` (`euhedral.inference.max-context-tokens`, default 32768), and a fixed 1280 MiB reserve for
  the per-sequence GDN state, the execution workspaces, the shared scratch and the kernel modules loaded after the model. The
  planner compares that need with the device's free memory at startup and grows the host-backed bytes in 64 MiB steps until it
  fits; if even the largest selection does not fit, startup fails and asks for a smaller maximum context. Nothing is host-backed
  when everything fits. There are no user options for host weight size, staging slots or residency.
- **Selection:** the token embedding first (it is a gather read in place over the bus, a few KiB per token). Then whole families
  smallest tensor first (GDN query_key, GDN output, attention output, ...); within the last family the layers are spread evenly, so
  the ring holds the smallest possible slot. The LM head is never host-backed, and the MTP layer and draft head never are
  (only `text/layers/N/...` projections).
- **Host memory:** one arena, allocated before any payload is read, 2 MiB-aligned, with `MADV_HUGEPAGE` and `MADV_COLLAPSE`, and
  pinned with `cudaHostRegister` (mapped, so the embedding is read in place).
- **Execution:** each plan view gets a `WEIGHT_TRANSFER` stage per use, run on a lane reserved for copies. A transfer depends only
  on the consumer that last read its slot. The consumer depends on the transfer and reads the slot. Ordering uses the graph's
  existing cross-lane markers, with no host synchronization.
- **Concurrency:** a quantum that stages weights holds the ring from admission until its last stage submitted. The next one's
  preparation awaits a marker recorded after the holder's lanes joined.
- **Prefetch:** a DFlash2 block, which leaves the transfer lane idle for its few milliseconds, also copies the decode view's
  first `STAGING_SLOTS` uses into slots 0 .. 3. A decode or verification quantum that is admitted while the ring holds them runs
  the decode view without those transfers (`ExecutionPlan.preloadedVariant`); the runtime marks the ring loaded, under its
  hold, only when every stage of the prefetching quantum ran, and any other staging quantum clears the mark
  ([DFLASH2.md](DFLASH2.md)).

Greedy tokens are identical to the resident run.

NVFP4, executed objects, 4 slots (warmup 1, 2 iterations; fully resident: 48.1 / 45.4 tok/s):

| Host-backed | Staged per token | Ring | Decode @64 | Decode @8K |
|---|---|---|---|---|
| 512 MiB (GDN query_key) | 0.54 GB | 47 MB | 45.4 tok/s | 42.5 tok/s |
| 1 GiB | 1.08 GB | 71 MB | 35.0 tok/s | 34.5 tok/s |
| 2 GiB | 2.15 GB | 83 MB | 19.2 tok/s | 19.3 tok/s |

**Up to the GPU's own token time the copies hide.** 0.54 GB is 12.4 ms of copying under a 20.8 ms token. Beyond that, decode runs
at the copy rate: 2.15 GB in 52 ms is 41 GB/s, against 43.7 GB/s for bare copies. Prefill is unaffected.

**Rejected:**
- **One allocation per tensor:** compaction under the loader's page-cache churn failed for a third of them, which fell back to
  4 KiB pages. Copies then split between 26 and 43 GB/s, about 30 GB/s overall, and 2 GiB decoded at 15.0 tok/s.
- **Deeper rings:** 8 and 16 slots gave 14.5 and 14.2 tok/s against 15.0 for 4 slots (per-tensor allocation). With the arena and
  DFlash2 speculation, 8 and 12 slots decoded 92.9 and 90.0 tok/s at 32K against 95.7 for 4: the extra slots are device memory
  that pushes more weights to the host, and the copies are bound by bandwidth, not by how far they run ahead.
- **The staging DAG without its copies** costs 2% (47.2 against 48.1 tok/s).

## Where it lives

- `tools/convert_checkpoint.py --quantization nvfp4`: the NVFP4 artifacts, quantized on the GPU by default (`tools/README.md`).
- Loading: `Nvfp4Layout`, a loader that accepts the vision-free inventory, and `Nvfp4CudaLoadIntegrationTest`.
- Execution: `native/src/nvfp4` (tensor-core decode, 1 to 8 rows) and `native/src/nvfp4_native` ([NVFP4_NATIVE.md](NVFP4_NATIVE.md)),
  `euhedral_cuda_linear_nvfp4_bf16`, `euhedral_cuda_linear_nvfp4_native_bf16` and
  `euhedral_cuda_nvfp4_native_gate_up_swiglu_bf16`, plan and frame dispatch for NVFP4 weights.
- Residency: `ArtifactProfile`, `ResidencyPlanner`, `HostWeightSelection`, `WeightStaging`, `WeightTransferFrame`, the transfer
  lane in `LanePool`, and `euhedral_cuda_host_weights_malloc`.
