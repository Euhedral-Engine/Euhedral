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
