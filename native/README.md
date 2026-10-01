# Euhedral CUDA native ABI

This directory contains the plain C ABI used by the Java Foreign Function & Memory binding.

The minimum CUDA header/runtime ABI is 13.1.x. An installed newer CUDA 13.x toolkit is accepted
when its headers and target libraries are complete. The host must also provide the CUDA driver library and NVRTC runtime; the driver must
support the target GPU. This native layer does not install or manage NVIDIA drivers.

## Source layout

`src/host/` holds the C ABI compiled into the library. `euhedral_cuda.c` is limited to memory
allocation, memory queries, streams, markers and copies; `cuda_kernel_loader.c` loads an installed
CUDA module root and compiles it once per process with NVRTC for `compute_90`, and the CUDA driver
JITs that PTX for the active device. `q3_embedding.c`, `rms_norm_bf16.c`, `q3_linear_bf16.c` and
`qwen_layer_ops.c` own the operator entry points; the `*_policy.h` headers hold the shape-only
dispatch decisions their tests exercise without a GPU. These operators accept opaque device
addresses and know nothing about Euhedral frames.

Every other file under `src/` is CUDA source, installed unchanged at the same relative path under
`share/euhedral_cuda/` next to the library. Each domain folder's `kernels.cu` is the NVRTC module
root the host loads; its headers are that module's leaves and strategies:

| Folder | Module contents |
| --- | --- |
| `common/` | Shared device helpers (`pdl.cuh`: programmatic dependent launch). |
| `embedding/` | Q3 row-split embedding (`ROW_SPLIT_K128_V1`) to BF16 hidden states. |
| `norm/` | Standalone BF16 RMSNorm. |
| `elementwise/` | Residual add, residual RMSNorm and SwiGLU. |
| `linear/` | Generic quantized and BF16-to-FP32 linear fallbacks. |
| `q3/`, `q45/` | Q3 and Q4/Q5 weight formats: layout, numerics, primitives, decode and prefill strategies. |
| `gemm/` | The prefill tensor-core tile engines (`tiles.cuh`, `balanced.cuh`) and their weight producers (`formats.cuh`), shared by `q3/`, `q45/` and `ffn/`. |
| `ffn/` | Gate/up SwiGLU, down (unsplit, split-K and reduce) and the streamed FFN regions. |
| `gdn/` | Gated DeltaNet control, projections, convolution, recurrence and gated RMSNorm. |
| `attention/` | NVFP4 KV append, QK norm/RoPE, decode and prefill attention leaves. |
| `sampling/` | Greedy token selection on the device (argmax over the final logits row). |
| `experiments/` | Test-only research modules (Q3 cluster, fragment, hierarchical and pipeline kernels); no host dispatch loads them. |

Includes within a folder are relative; includes across folders name the path from the tree root
(`common/pdl.cuh`, `gemm/balanced.cuh`), which NVRTC resolves through the installed root.

`native-products.json` defines the two supported targets: `x86_64-linux-gnu` (`linux-x64`)
and `x86_64-windows-gnu` (`windows-x64`). There is no macOS CUDA product. The normal Gradle
`nativeBuild` and `:api:bootJar` paths build both products. `nativePackage` writes
`build/distributions/euhedral-cuda-native.zip` with the selected products. Zig installs each
product at `build/native/<product>/lib/<library>` and its runtime-compiled CUDA tree at
`build/native/<product>/share/euhedral_cuda/`; `nativeVerify` checks that the installed tree
matches `src/` exactly. The Spring Boot JAR build produces
the native ZIP alongside the application JAR, but does not embed native files: Java FFM loads
the library from a filesystem path, with the CUDA sources adjacent in the installed product layout.
The ZIP contains the native library and CUDA sources, not NVIDIA CUDA runtime libraries.
Deploy target-matching CUDA runtime and NVRTC (including builtins) separately, or use the
container image, which installs the pinned Linux CUDA user-space runtime libraries.

For ordinary builds Gradle uses the pinned CUDA 13.1 redistributables. Set `CUDA_HOME` or
`CUDA_PATH` to explicitly select an installed matching host toolkit instead; merely installing
one at `/usr/local/cuda` or under Program Files does not override the pinned default. A newer
host NVRTC can emit PTX the installed driver cannot JIT, even if its headers and libraries link.
The selected host toolkit must have headers at version 13.1 or higher,
the runtime/NVRTC link libraries, and the target driver stub/import library. When no suitable
host toolkit is selected, Gradle downloads the official NVIDIA 13.1 cudart, NVRTC, CUDA CRT,
and CCCL redistributable archives for each target, checks the SHA-256 values in the manifest,
and caches them under the Gradle user home. The Windows driver import is generated with the
pinned Zig `dlltool` from an explicit driver ABI export list; it does not link the host's
Linux driver stub or NVIDIA's Windows static loader. Downloads include no NVIDIA kernel driver.

Use Java 25, Gradle 9.6.1, and Zig 0.16.0 (the versions recorded in `.mise.toml`).
Gradle invokes Zig directly with argument lists (and honors `ZIG` when set); no shell is needed
by the host build. Build and verify both products with:

```text
./gradlew nativeVerify nativePackage :api:bootJar
```

Explicit per-target overrides take precedence over discovery and download:

```text
-Peuhedral.cuda.linux-x64.include-dir=/path/to/linux/cuda/include
-Peuhedral.cuda.linux-x64.library-dir=/path/to/linux/cuda/lib
-Peuhedral.cuda.windows-x64.include-dir=C:/path/to/windows/cuda/include
-Peuhedral.cuda.windows-x64.library-dir=C:/path/to/windows/cuda/lib/x64
```

Legacy `euhedral.cuda.include-dir` and `euhedral.cuda.library-dir` apply only to the matching
host target. Overrides must be paired, originate from one toolkit root, and provide matching
CUDA driver and runtime header versions (13.1+). Windows cross
builds still require Windows CUDA import libraries, never Linux `.so` files. Set
`-Peuhedral.cuda.force-download=true` to choose the pinned 13.1 archives over an installed
toolkit; `-Peuhedral.native.products=linux-x64` restricts packaging (not `nativeBuild`) to
the Linux product, useful for the container image. `nativeBuildLinuxX64` and
`nativeBuildWindowsX64` are individual tasks; only the matching product is executed in CUDA
integration tests. The integration task adds the resolved CUDA runtime DLL/SO directory to
the test process search path; a compatible host driver is still required.

Run the isolated CUDA integration suite with the same automatic CUDA resolution. The full compact-model test also
requires the compact Qwen EDRL artifact, its BF16 reference EDRL artifact, and sufficient free device
memory to keep the whole compact model resident:

```text
./gradlew cudaIntegrationTest
```

The runtime needs a compatible NVIDIA driver installed on the host (or injected by NVIDIA
Container Toolkit), along with target-matching CUDA user-space runtime/NVRTC libraries and
the CUDA headers used by NVRTC to compile the installed CUDA sources. Set
`EUHEDRAL_CUDA_INCLUDE_DIR` to their include directory. Driver stubs and development import
libraries are for linking only, not runtime deployment.

## Opt-in Q3 temporal fragment pipeline

`experiments/q3/pipeline_kernels.cu` is a separate C++17 NVRTC translation unit, not a production
Q3 dispatch mode. Its `euhedral_q3_pipeline_<rows>_<cm>x<cn>` entry points take the
same Q3 matrix arguments as the fragment-node kernels, followed by a nullable
`unsigned long long* observations`. Launch 256 threads per CTA, with a 2D grid
rounded up to the kernel's fixed `(cn, cm, 1)` cluster dimensions. Supported row
tiles are 32 and 64; compositions `(cm,cn)` are `(1,1)`, `(2,1)`, `(4,1)`, `(1,2)`,
`(1,4)` and `(2,2)`. Compilation needs the CUDA include directory and its `cccl`
subdirectory on the include path. The normal C++14 Q3 source and dispatch are unchanged.

Each A or B branch owns two execution-storage slots and separate ready, parent-ready
and borrower-release barriers per slot. Generation identity is branch/slot-local;
there is no CTA-wide current-generation stamp. Generations are consecutive from
zero, with `slot = generation % kSlots` and phase derived from the slot's reuse
count. The generation protocol does not encode a K extent. This first storage,
producer and MMA policy still uses K64; changing that extent also requires changing
those policy-specific layouts and loops, not merely selecting a different kernel name.

The first mapping uses four branch-producer warps and the existing four MMA warps.
Only the strategy assigns physical warp IDs. A single producer warp collectively
begins, accepts and publishes each branch generation; each registered descendant
warp acquires and releases it once in order. A parent owner aliases its local
execution slot. Remote producers copy each branch fragment once into their own slot,
release the parent borrow immediately after that copy, then publish locally to
sibling MMA descendants. Parent reuse counts both local MMA borrowers and remote
copy borrowers; child-slot reuse counts its local MMA borrowers. No extra full-tile
intermediate is introduced. Warp FP32 accumulators retain the existing K/high/low
numerical order and flow through CTA result storage to BF16 output.

Steady-state handoffs use scoped release arrivals and acquire waits on CUDA
`mbarrier` objects. Producer warps wait only for their branch slot's previous
borrowers; remote acceptance waits only on that parent branch's publication; MMA
warps wait only on their A and B branches. Warp joins connect all-lane reads/writes
to a leader's notification. There are no whole-CTA or whole-cluster barriers inside
the K loop. Relative to the single-slot fragment strategy, each generation removes
six explicit CTA barriers (one parent-production, one begin, two publish, two release)
and two cluster barriers (publication and retirement).

Two whole-cluster joins remain per launch: bootstrap makes initialized barriers and
DSM CTA lifetimes available, and the terminal join prevents any CTA exiting while a
peer still accesses its shared memory. The terminal join also publishes CTA result
writes before output writeback. Neither join scales with the number of generations.

A non-null observation buffer requires `grid.x * grid.y * ceil(width/64) * 44`
64-bit words. Per CTA/generation, four branches each record six words (production
start, acceptance complete, pre-publication timestamp, logical remote-fragment count,
generation, slot); four consumers each record five words (acquired, MMA complete,
pre-release timestamp, first A address, first B address). Timestamps use the device
global timer. These records are diagnostics, never synchronization state; publication
and release stamps precede the actual notifications. Timed launches pass null.
