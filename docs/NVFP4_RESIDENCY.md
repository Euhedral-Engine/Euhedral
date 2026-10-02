# NVFP4 residency budget

The question was whether NVFP4 weights can run near compact-Q3 speed with real VRAM headroom and
MTP retained, by keeping NVFP4 as the execution representation and streaming part of the weights
from pinned host RAM ahead of use.

**Conclusion:** Single-sequence decode reads every weight every token, and the platform's
host-to-device rate cannot stream the useful offload (at least 1.8 GiB) within Q3's 16 ms token. It
takes 44 ms even at the 44 GB/s that huge-page pinned memory reaches; 4 KiB-page pinned memory under
the IOMMU stalls at 25 GB/s.

**Staged NVFP4 can match non-MTP Q3 only with MTP accepting about 2.75 or more tokens per
verification step**, and Q3 would gain from MTP too. Prefill can hide the transfers.

Measured on 2026-10-01 on an RTX 5070 Ti (16 GB) with an i9-14900K.

## Memory budget

| Item | Size |
|---|---|
| Device total | 15.56 GiB (16,703,946,752 bytes) |
| Other processes (desktop) | about 0.46 GiB |
| Q3 run: used after loading 11.98 GiB of weights | 13.19 GiB, so about 0.75 GiB is CUDA context, NVRTC modules and engine |
| Graph and workspace pools after the first run | +0.16 GiB (they persist) |
| Q3 free for KV and per-sequence state | about 2.2 GiB, roughly 115K tokens of context |
| KV (NVFP4 pages, 16 attention layers, 4 KV heads, D256, 144-byte rows) | 18 KiB per token: 0.56 GiB at 32K, 2.25 GiB at 128K |
| GDN state (48 layers, 48 x 128 x 128 FP32, plus convolution) | about 0.15 GiB per sequence |
| NVFP4 artifact (text, LM head, MTP, draft head; Q3 embedding) | 14.52 GiB; about 0.19 GiB free after loading, so it cannot run |
| MTP layer + draft head, NVFP4 | 0.57 GiB |
| NVFP4 with block scales compressed (4-bit dictionary + escape, lossless) | about 13.77 GiB |

**Compressed NVFP4 with MTP leaves about 0.4 GiB** for KV and per-sequence state, roughly 15K
tokens. Matching Q3's headroom takes at least **1.8 GiB** of weights off the GPU (13.77 - 11.98), and
more once staging buffers are counted.

**Lossless compression of NVFP4 is limited.** The E2M1 codes carry 3.90 of their 4 bits, because the
non-uniform FP4 grid already equalizes code use. The block scales carry 3.63 of their 8. Only the
scales compress usefully, saving 0.75 GiB.

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

Compact-Q3 decode takes 16.0 ms per token (62.7 tok/s at 64 + 128), about 0.25 ms per layer for
every family together.

## Why decode cannot be hidden

**There is no reuse window.** Every token needs every layer, so a staged weight is read once and
evicted. The time between the earliest safe prefetch and its use does not matter; the bytes per
token do. With perfect overlap a token takes max(GPU time, PCIe time).

**Even fully resident, NVFP4 is behind Q3.** It reads 13.4 GiB of weights per token against Q3's
10.7 GiB. At the same DRAM efficiency that is about 17.6 ms per token, 10% slower than Q3, before
any offload.

**Offload hides only up to the GPU time.** At 44 GB/s, 0.72 GiB per token stays under 17.6 ms
(0.4 GiB at 25 GB/s). The minimum useful offload, 1.8 GiB, is 44 ms per token: 2.7x Q3's token time.
Matching Q3 would need 120 GB/s, nearly twice PCIe 5.0 x16's theoretical rate.

**MTP does not change the ratio.** It divides streamed bytes and GPU work by the same number of
tokens accepted per verification step. Against non-MTP Q3 (16 ms per token), offloaded NVFP4 would
need about 2.75 accepted tokens per step to break even (4.8 at 25 GB/s). Euhedral has no MTP decode
yet, so acceptance has not been measured.

## Prefill

A chunk reuses its weights across all its tokens. 1.8 GiB is 44 ms per chunk, against about 417 ms
of compute for a 512-token chunk, so overlapped transfers could hide it.

Prefill-only staging is not useful alone: decode is the binding regime.

## Fully resident NVFP4

With `weightResidency: EXECUTED` the vision tower, MTP layer and draft head stay in the artifact, and
the NVFP4 base model runs from the GPU. Measured 2026-10-01, one session per artifact, warmup 1,
2 measured iterations, serving stopped:

| Artifact | Decode, 64-token prompt | Decode, 8K prompt | Prefill 512 | TTFT at 8K | Engine allocation | Device free |
|---|---|---|---|---|---|---|
| Compact Q3 | 59.3 tok/s | 54.1 tok/s | 1154 tok/s | 7.66 s | 12.09 GiB | 2.23 GiB |
| NVFP4, executed objects | 48.1 tok/s (-19%) | 45.4 tok/s (-16%) | 1052 tok/s (-9%) | 8.34 s | 14.05 GiB | 0.62 GiB |

**NVFP4 is slower than compact Q3, by about its extra bytes.** Decode reads about 25% more weight
bytes per token. 59.3 / 1.25 = 47.4 tok/s predicts the measured 48.1, so the NVFP4 GEMV streams at
Q3's efficiency, and no kernel work can close the gap at batch 1. Prefill loses less because its
tiles are compute-bound.

Greedy generation answers "The capital of France is" with " Paris."
(`QwenNvfp4GenerationCudaIntegrationTest`).

## Host-backed weights

`hostWeightBytes` (benchmark `hostWeightMiB`) keeps layer projections in pinned host memory and stages
them into a device ring of `stagingSlots` slots on every use.

- **Selection:** whole families are taken smallest tensor first (GDN query_key, GDN output, attention
  output, ...). Within the last family the layers are spread evenly. The ring then holds the
  smallest possible slot.
- **Host memory:** one arena, allocated before any payload is read, 2 MiB-aligned, with
  `MADV_HUGEPAGE` and `MADV_COLLAPSE`, and pinned with `cudaHostRegister`.
- **Execution:** each plan view gets a `WEIGHT_TRANSFER` stage per use, run on a lane reserved for
  copies. A transfer depends only on the consumer that last read its slot. The consumer depends on
  the transfer and reads the slot. Ordering uses the graph's existing cross-lane markers, with no
  host synchronization.
- **Concurrency:** a quantum that stages weights holds the ring from admission until its last stage
  submitted. The next one's preparation awaits a marker recorded after the holder's lanes joined.

Greedy tokens are identical to the resident run (`QwenNvfp4GenerationCudaIntegrationTest`).

NVFP4, executed objects, 4 slots (warmup 1, 2 iterations; resident: 48.1 / 45.4 / 1052):

| Host-backed | Staged per token | Ring | Decode @64 | Decode @8K | Prefill 512 |
|---|---|---|---|---|---|
| 512 MiB (GDN query_key) | 0.54 GB | 47 MB | 45.4 tok/s | 42.5 tok/s | 1048 tok/s |
| 1 GiB | 1.08 GB | 71 MB | 35.0 tok/s | 34.5 tok/s | 1048 tok/s |
| 2 GiB | 2.15 GB | 83 MB | 19.2 tok/s | 19.3 tok/s | 1046 tok/s |

**Up to the GPU's own token time the copies hide.** 0.54 GB is 12.4 ms of copying under a 20.8 ms
token, and costs 6%. Beyond that, decode runs at the copy rate: 2.15 GB in 52 ms is 41 GB/s, against
43.7 GB/s for bare copies. Prefill is unaffected.

**Measured and not kept:**
- **One allocation per tensor:** compaction under the loader's page-cache churn failed for a third
  of them, which fell back to 4 KiB pages. Copies then split between 26 and 43 GB/s, about 30 GB/s
  overall, and 2 GiB decoded at 15.0 tok/s.
- **Deeper rings:** 8 and 16 slots gave 14.5 and 14.2 tok/s against 15.0 for 4 slots (per-tensor
  allocation).
- **The staging DAG without its copies** costs 2% (47.2 against 48.1 tok/s).

## What was built

- `tools/convert_qwen_safetensors_to_compact_edrl.py --profile nvfp4`: the NVFP4 artifact, quantized
  on the GPU by default.
- Loading: `Nvfp4Layout`, a loader that accepts the vision-free inventory, and
  `QwenNvfp4CudaLoadIntegrationTest`.
- Execution: `native/src/nvfp4` (decode GEMV, tile-engine producer for prefill, paired gate_up and
  SwiGLU), `euhedral_cuda_linear_nvfp4_bf16` and `euhedral_cuda_nvfp4_gate_up_swiglu_bf16`, plan and
  frame dispatch for NVFP4 weights, and `WeightResidency`.
- Host-backed weights: `HostWeightSelection`, `WeightStaging`, `WeightTransferFrame`, the transfer
  lane in `LanePool`, and `euhedral_cuda_host_weights_malloc`.
