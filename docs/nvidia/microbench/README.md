# nvbench: RTX 5070 Ti microbenchmarks

This is a small C harness that produced every measured number in `docs/nvidia/`.
- It drives the CUDA driver API directly.
- It compiles the kernels in `kernels/` at run time with NVRTC (for `sm_120a`, as cubins).
- It builds with the repo's pinned CUDA 13.1 headers and `zig cc`. No CUDA toolkit or nvcc is
  needed.
- The results from 2026-10-02 are in [RESULTS.md](RESULTS.md).

## Build and run

```bash
bash docs/nvidia/microbench/build.sh            # needs build/cuda-dev (from ./gradlew nativeBuild) and zig
export CUDA_INCLUDE_DIR=$PWD/build/cuda-dev/linux-x64/include
export LD_LIBRARY_PATH=$PWD/build/cuda-dev/linux-x64/runtime   # NVRTC builtins
export MB_ROOT=$PWD/docs/nvidia/microbench
docker stop euhedral-inference-serve             # it holds ~13 GB of the GPU
$MB_ROOT/nvbench attrs
$MB_ROOT/nvbench mem [bw|bulk|l2|lat]
$MB_ROOT/nvbench mma [num|<variant substring>]
$MB_ROOT/nvbench alu [<variant>]
$MB_ROOT/nvbench tex [pipes|precision|bc|gemv]
$MB_ROOT/nvbench pcie
$MB_ROOT/nvbench l2 [persist|compress]
$MB_ROOT/nvbench misc [cluster|launch]
docker start euhedral-inference-serve
```

## Environment variables

- **`MB_MIB`:** buffer size for `mem` (default 2048).
- **`MB_ITERS`:** loop count for `mma` and `alu`.
- **`MB_DUMP=/dir`:** writes every compiled cubin. Disassemble it with
  `nvdisasm -c` (the Euhedral-Execution venv's Triton ships one) to check SASS.
- **`MB_VERBOSE=1`:** prints a sample block in `tex precision`.

## Commands

| Command | Kernels | What it measures |
|---|---|---|
| `attrs` | none | Device attributes behind the feature questions: persisting L2, compression, clusters, green-context SM granularity, texture limits, watchdog |
| `mem bw` | `mem.cu` | Streaming read GB/s against bytes in flight per SM; LDG.32 / LDG.128 / LDG.256; write; copy |
| `mem bulk` | `mem.cu` (`read_bulk`) | `cp.async.bulk` + mbarrier ring with one issuing thread per CTA |
| `mem l2` | `mem.cu` (`reread_v4`) | Re-read bandwidth against working-set size (the L2 knee) |
| `mem lat` | `mem.cu` (`chase`) | Dependent-load latency against footprint, TLB reach, shared-memory latency |
| `mma` | `mma.cu` | Throughput and dependent latency of 24 `mma.sync` kinds. Each kind is its own NVRTC program, so an unsupported one reports "unsupported" |
| `mma num` | `mma_num.cu` | Accumulator width and rounding of each tensor path |
| `alu` | `alu.cu` | Per-SM per-clock throughput of CUDA-core, SFU and conversion instructions, and pipe-overlap mixes |
| `tex pipes` | `tex.cu` | LDG vs `tex1Dfetch` vs both; normalized 8-bit fetches |
| `tex precision` | `tex.cu` | BC4 decode error against the D3D10 formula; the exact hardware palette; gather order |
| `tex bc` | `tex.cu` | BC1/BC4/BC5/BC6H/BC7 point and gather decode throughput, cache-resident and DRAM-cold |
| `tex gemv` | `tex.cu` | GEMV with BC4 weights decoded by the texture units, cold (rotating copies) |
| `pcie` | `mem.cu` | H2D/D2H by host allocation type, chunk and streams; bidirectional; zero-copy; D2D on SMs vs copy engines |
| `l2 persist` | `l2.cu` | Hot-set retention through a 1 GiB stream: plain, `createpolicy` hints, persisting window |
| `l2 compress` | `l2.cu` | `CU_MEM_ALLOCATION_COMP_GENERIC` read and write bandwidth by data pattern |
| `misc cluster` | `misc.cu` | Cluster size limits, local vs DSMEM shared-memory bandwidth |
| `misc launch` | `misc.cu` | Stream, PDL and graph launch costs; `WHILE` conditional-node iteration cost |

## Measurement notes

**Clock-relative rates.** Every kernel stamps `clock64` and `%globaltimer` from block 0. Rates are
reported per SM per *measured* SM clock, which ran at about 2.86 GHz, well above the rated 2452 MHz.

**Cold vs warm data.**
- "Cold" sets exceed the 48 MiB L2 several times over: 1–2 GiB buffers, or rotating copies for the
  GEMV.
- The 16 MiB "reused" sets of `tex pipes` include L1 hits. Compare rows within that table, not
  against `mem l2`.

**ptxas folding.** ptxas optimizes inside `asm volatile`. Repeated constant adds and `rcp(rcp(x))`
fold away, so the `alu` probes use non-foldable chains. A first run that folded is excluded from
RESULTS.md.

**Desktop interference.** The desktop shares this GPU. Expect up to about 10% scatter. Best-of-N is
reported.

**Leftover probes.** `kernels/mma.cu` holds no probe that wasn't run. The mixed E4M3×E2M1
block-scaled and FP16-accumulate rates were planned but not measured.
