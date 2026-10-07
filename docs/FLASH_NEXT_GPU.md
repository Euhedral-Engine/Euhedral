# Flash-Next GPU campaign

The storage and cache path reads at the drive's ceiling whenever experts are requested. This campaign is the GPU side:
idle time, launch overhead, and under-filled launches. RTX 5070 Ti, artifact on a Samsung 990 Pro, `RAM_CACHED` at
22,000 MiB, 32 workers. Decode is in tokens/s.

## Tools

- `/tmp/nsys.sh TAG MODE` profiles one window of the performance record with Nsight Systems (`EUHEDRAL_QWEN4_PERF_PROFILE`:
  `decode` (32 steps at 64 context), `decode4k` (32 steps after a 4096-token prefill), `prefill` (4096 tokens)); each
  window follows two idle seconds.
- `/tmp/nsys_analyze.py SQLITE STEPS`: kernel busy time, H2D, idle with and without a copy, gaps by length, launches and
  time per kernel. `/tmp/gaps.py SQLITE STEPS [US]`: idle gaps attributed to the kernels before and after them.
- `/tmp/qs.sh TAG prefill|decode [ENV..]`: one-minute screens.

## Baseline (main, `21180be`)

Profiled (the profiler slows the host side):

| window | per step | kernels busy | no kernel (copy only / nothing) | launches |
|---|---|---|---|---|
| decode at 64 context | 36.1 ms (27.7 tokens/s) | 18.8 ms | 3.2 / 14.1 ms | 2,639 |
| decode at 4096 context | 57.9 ms (17.3 tokens/s) | 19.3 ms | 8.0 / 30.5 ms | 2,663 |
| prefill of 4096 tokens | 8.59 s (476 tokens/s) | 3.59 s | 1.69 / 3.32 s | 44,376 |

Idle time by cause, decode at 64 context (per step): waiting for a missed expert's record between experts 9.75 ms (39
gaps of about 250 us); router, read-back and host plan before a layer's first expert 1.8 ms (35 gaps of 52 us); the
token boundary about 1.3 ms; small gaps around the routing, norms and the shared expert about 1.5 ms. At 4096 context
the expert waits are 29.6 ms of 57.9 ms.

Kernel time, decode at 64 context (per step): BF16 linear 518 launches 7.5 ms (mean 14.5 us); expert gate/up 480
launches 4.65 ms (9.7 us); expert down 480 launches 3.1 ms (6.5 us); NVFP4 decode GEMVs 302 launches 2.85 ms.

Kernel time, prefill of 4096 tokens: `nvfp4_reference` 48 launches 1.58 s (32.9 ms each, 44% of kernel time); BF16
linear 471 launches 1.01 s (2.1 ms each); expert kernels 42,448 launches 0.62 s; native NVFP4 linears 254 launches
0.075 s.

## The shared expert's down projection in prefill

`nvfp4_reference` was the shared expert's down projection: its input width, 640, is not a multiple of 256, which the
native kernels need for more than 64 rows (their tensor maps fetch the weight scales 16 bytes, 256 values, at a time,
and a row's 40 scale bytes are not a 16-byte stride), so every prefill chunk ran it on the scalar reference. Skipping it
entirely (an experiment, wrong output) bounded the gain: prefill 4096 485 to 580 tokens/s, 512 153 to 197.

The plan now pads each layer's shared down projection to 768 input columns on the device at startup (zero codes, zero
scales: every product they add is exactly 0; 53 MiB, reserved by the residency planner), and a prefill's SwiGLU writes
its activation in rows of 768 with zero padding (`euhedral_q4_swiglu_padded_bf16`), so the native kernels take it.
Decode keeps the unpadded weights and kernels. On the exact route the padded projection is bit for bit the unpadded one;
on the native route it is within 0.86% relative RMS of it (`Qwen4SharedDownPaddingCudaIntegrationTest`), and the MoE
fixtures' shared expert within 5e-5 to 1.5e-2 of upstream.

| scenario | before | after |
|---|---|---|
| prefill 4096 (tokens/s, 3 runs) | 484-485 | 590-594 |
| prefill 512 (tokens/s, median of 3) | 153 | 162 |
| decode cold 64 / 4096 (tokens/s) | 22.5 / 22.4 | 22.6 / 22.4 |
| decode warm 64 / 4096 (tokens/s) | 28.1 / 27.9 | 28.4 / 28.1 |

## BF16 linears on tensor cores in prefill

After the padding, the BF16 linears were half of a prefill's kernel time (1.01 s of 2.0 s): the hyper-connection's
down (10,240 to 320) and up (320 to 10,240) projections, two each per layer, at 3.6 and 5.9 ms per 4096-row launch
(about 7.5 and 4.5 TFLOPS on FP32 FMAs). `euhedral_q4_linear_tc_*` run them on m16n8k16 tensor cores (dflash's scheme:
K split in four for outputs up to 1536 wide, chosen by the shape alone; one or four row tiles of 16 per weight load,
which never change a row's bits). Their results differ from the FP32 kernel's in summation order only: 4e-6 to 8e-5
relative RMS on the model's shapes (`Qwen4LinearTensorCoreCudaIntegrationTest`).

They run for 9 rows and more. At one row an MMA uses one of its 16 rows, and the FP32 kernel is faster on most decode
shapes (profiled per launch: hyper-connection down 19.1 against 29.0 us, router 6.0 against 7.4 us, output head 1.49
against 1.73 ms; only the hyper-connection up projection was faster, 12.9 against 8.8 us). Exact numerics and
`EUHEDRAL_QWEN4_LINEAR_TC=0` keep the FP32 kernel everywhere.

| scenario | FP32 kernel | tensor cores from 9 rows |
|---|---|---|
| prefill 4096 (tokens/s) | 582 | 661-671 |
| prefill 512 (tokens/s) | 142-162 | 147-167 |
| decode cold 64 / 4096 (tokens/s) | 22.0 / 21.7 | 22.9 / 21.9 |
| decode warm 64 / 4096 (tokens/s) | 28.6 / 26.8 | 28.4 / 27.0 |

Not kept: tensor cores for decode rows too (prefill as above; decode cold 22.1 / 22.7, warm 27.2 / 27.8, kernel time
for the BF16 linears 7.5 to 9.4 ms per step).

## Decode's BF16 linears with K split across warps

One warp per output column left decode's narrow projections short of the device: the hyper-connection's down
projection (320 outputs) ran 40 CTAs, and the one- and four-output projections (the shared expert's gate, the
injection) ran one and four warps over K of 2560 and 10,240. `euhedral_q4_linear_split4/8_bf16` give each column 4 or 8
warps over contiguous slices of K, summed in slice order (deterministic), by the shape alone: 8 slices for fewer than
64 outputs or K of 8192 and more, 4 for K of 2048 and more. Summation-order noise only: 0 to 1.5e-6 relative RMS on the
model's shapes. The exact route and `EUHEDRAL_QWEN4_LINEAR_SPLIT=0` keep one warp per column.

Profiled, per decode step at 64 context: BF16 linears 7.5 to 4.8 ms (hyper-connection down 19.1 to 10.4 us per launch,
injection 13.2 to 3.0, shared gate 13.2 to 1.7, GDN gates 5.4 to 1.9, router 6.0 to 5.0); the step 36.4 to 34.1 ms.

| scenario (tokens/s) | split off (two runs) | split on (two runs) |
|---|---|---|
| decode cold 64 | 21.9 / 22.5 | 23.6 / 23.6 |
| decode cold 4096 | 22.3 / 22.8 | 22.8 / 23.8 |
| decode warm 64 | 28.3 / 28.7 | 30.2 / 30.8 |
| decode warm 4096 | 26.7 / 28.7 | 29.5 / 30.3 |

## Not kept: narrower expert CTAs in decode

A decode expert's kernels ran 20 CTAs per launch (32 act columns and 128 output rows per CTA). Narrower CTAs (16
columns and 32 rows, the same K split, so bit for bit the same outputs; verified against the wide kernels) cut the
expert kernels from 7.82 to 5.73 ms per step (gate/up 9.8 to 6.6 us, down 6.5 to 5.3 us per launch), but the step stayed
at 34.0 ms profiled and the screens were flat (decode warm 64 31.3 / 31.2 against 29.1 / 30.5 tokens/s, cold 64 23.3 /
24.0 against 23.5 / 22.7): the time moved to the waits for missed experts. In decode the experts' kernels are not on the
critical path; each layer waits for its slowest expert's record. Worth retrying once those waits shrink.

More lanes (`EUHEDRAL_QWEN4_LANES`): 2 / 4 / 8 lanes, decode warm 64 30.5 / 31.0 / 31.2, cold 64 23.6 / 24.0 / 23.6
tokens/s: within noise.

## Device memory left for the system

The plan filled the device: 84 MiB stayed free at the lowest point of a run. It now keeps 700 MiB for the rest of the
system beside the runtime's 1 GiB (`Qwen4ResidencyPlanner.SYSTEM_RESERVE_BYTES`, `EUHEDRAL_GPU_SYSTEM_RESERVE_MIB`):
774 MiB stayed free at the lowest point. The expert cache has 3,102 slots instead of about 3,385; decode cold 64 / 4096
23.0 / 22.7 tokens/s (23.6 / 23.6 before), warm 30.1 / 29.9 (30.0-30.5 / 29.9), prefill unchanged.

## Against main

Paired benchmark, 6 forks, control main (`21180be`), candidate `b672403` (before the system reserve):

| scenario | main | this branch | change | forks ahead |
|---|---|---|---|---|
| prefill 512 (tokens/s) | 139 | 148 | +6.5% | 6/6 |
| prefill 4096 (tokens/s) | 484.5 | 664 | +37.1% | 6/6 |
| decode cold 64 (tokens/s) | 22.4 | 23.6 | +5.3% | 6/6 |
| decode cold 4096 (tokens/s) | 22.1 | 23.6 | +7% | 6/6 |
| decode warm 64 (tokens/s) | 28.2 | 30.0 | +6.3% | 6/6 |
| decode warm 4096 (tokens/s) | 27.5 | 29.9 | +8.4% | 6/6 |
| disk median while prefilling (GB/s) | 4.53 | 6.23 | +37% | 6/6 |

Profiled per step, main against this branch: decode at 64 context 36.1 to 34.0 ms (kernels busy 18.8 to 16.2 ms,
nothing on the GPU 14.1 to 14.6 ms, copy only 3.2 to 3.3 ms, 2,639 launches both); decode at 4096 context 57.9 to 58.3
ms (kernels 19.3 to 16.7 ms; waiting for missed experts 29.6 to 30.7 ms); prefill of 4096 tokens 8.59 to 6.31 s
(kernels 3.59 to 1.21 s, 44,376 to 44,432 launches; the rest is waiting for expert records from the disk).

## What limits it now

- Prefill: the disk. Kernels run 1.21 s of 6.31 s; the disk reads at 7 GB/s whenever records are asked for and 37 GB
  per 4096 tokens at this tier size bounds a prefill at about 775 tokens/s.
- Decode: waiting for missed experts' records, 10 ms per step at 64 context and 31 ms at 4096 (each layer waits for its
  slowest expert). GPU-side changes that shortened kernels (narrower expert CTAs: 2.1 ms of kernel time) or removed
  a stage hop (the routing merged into the block stage) moved the time into those waits and left the step flat.
- Remaining software bubbles in decode, each a few percent: the router's round trip through the host (about 50 us per
  layer, 1.6 ms per step), the token boundary (about 1.3 ms), launch gaps in the dense chain (about 2.6 ms).
