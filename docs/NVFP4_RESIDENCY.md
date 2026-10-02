# NVFP4 residency budget

The question was whether NVFP4 weights can run near compact-Q3 speed with real VRAM headroom and
MTP retained, by keeping NVFP4 as the execution representation and streaming part of the weights
from pinned host RAM ahead of use.

**Conclusion: not on this platform.** Single-sequence decode reads every weight every token. The
measured host-to-device bandwidth (25 GB/s) cannot stream the offload that useful headroom needs
anywhere near Q3's token time. Prefill could hide it. Measured on 2026-10-01 on an RTX 5070 Ti
(16 GB) with an i9-14900K.

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

## Bandwidth budget

**Host-to-device: 25 GB/s.** Pinned `cudaMemcpyAsync` from default, write-combined and portable
memory, in 4-256 MiB transfers, with 1, 2 or 4 copy streams, alone or beside a memory-bound kernel,
all reach the same rate. The link trains at Gen 5 x16 under load, so the ceiling is on the host side.
Device-to-host: 22 GB/s.

Streaming cost of each family per decode token, from host RAM, in compressed bytes (4.26 bits per
weight):

| Family | Bytes per layer | PCIe time per layer | Whole family |
|---|---|---|---|
| FFN gate_up (x64) | 94.9 MB | 3.80 ms | 5.66 GiB |
| FFN down (x64) | 47.4 MB | 1.90 ms | 2.83 GiB |
| GDN value_z (x48) | 33.5 MB | 1.34 ms | 1.50 GiB |
| GDN query_key (x48) | 11.2 MB | 0.45 ms | 0.50 GiB |
| GDN output (x48) | 16.7 MB | 0.67 ms | 0.75 GiB |
| attention query_key (x16) | 19.5 MB | 0.78 ms | 0.29 GiB |
| attention gate_value (x16) | 19.5 MB | 0.78 ms | 0.29 GiB |
| attention output (x16) | 16.7 MB | 0.67 ms | 0.25 GiB |
| LM head (x1) | 676.8 MB | 27.07 ms | 0.63 GiB |

Compact-Q3 decode takes 16.0 ms per token (62.7 tok/s at 64 + 128), about 0.25 ms per layer for
every family together.

## Why decode cannot be hidden

**There is no reuse window.** Every token needs every layer, so a staged weight is read once and
evicted. The time between the earliest safe prefetch and its use does not matter; the bytes per
token do. With perfect overlap a token takes max(GPU time, PCIe time).

**Even fully resident, NVFP4 is behind Q3.** It reads 13.4 GiB of weights per token against Q3's
10.7 GiB. At the same DRAM efficiency that is about 17.6 ms per token, 10% slower than Q3, before
any offload.

**Offload hides only up to the GPU time.** 0.4 GiB per token stays under 17.6 ms, which frees
nothing useful. The minimum useful offload, 1.8 GiB, is 77 ms per token: 4.8x Q3's token time.
Matching Q3 would need 120 GB/s, nearly twice PCIe 5.0 x16's theoretical rate.

**MTP does not change the ratio.** It divides streamed bytes and GPU work by the same number of
tokens accepted per verification step. Against non-MTP Q3, offloaded NVFP4 would need about 4.8
accepted tokens per step to break even.

## Prefill

A chunk reuses its weights across all its tokens. 1.8 GiB is 77 ms per chunk, against about 417 ms
of compute for a 512-token chunk, so overlapped transfers could hide it.

Prefill-only staging is not useful alone: decode is the binding regime.

## What was built

- `tools/convert_qwen_safetensors_to_compact_edrl.py --profile nvfp4`: the NVFP4 artifact, quantized
  on the GPU by default.
- Loading: `Nvfp4Layout`, a loader that accepts the vision-free inventory, and
  `QwenNvfp4CudaLoadIntegrationTest`.
- No NVFP4 execution path was written; the budget above rules out the staged configuration first.
