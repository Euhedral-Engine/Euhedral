# Euhedral CUDA native ABI

This directory contains the plain C ABI used by the Java Foreign Function & Memory binding.

The engine requires an NVIDIA Blackwell GPU (compute capability 12.x); `CudaGpuMemory` fails to construct on any other device.

The minimum CUDA header/runtime ABI is 13.1.x. An installed newer CUDA 13.x toolkit is accepted
when its headers and target libraries are complete. The host must also provide the CUDA driver library and NVRTC runtime; the driver must
support the target GPU. This native layer does not install or manage NVIDIA drivers.

## Source layout

`src/host/` holds the C ABI compiled into the library. It accepts opaque device addresses and knows nothing about Euhedral frames:

| File | Entry points |
| --- | --- |
| `euhedral_cuda.c` | Memory allocation (device, pinned host, huge-page host weights), memory queries, streams, markers and completion events, copies. |
| `cuda_kernel_loader.c`, `.h` | Compiles an installed CUDA module root once per process with NVRTC and loads it, on a dedicated thread with a large stack (NVRTC overflows a worker thread's). Ordinary modules compile to the device's own cubin (`sm_<major><minor>`); the modules that use arch-specific instructions (`q3_mx`, `nvfp4_native`) compile for `sm_<major><minor>a`. No PTX is left for the driver to JIT. Also holds the thread-local selections: exact numerics, row-exact execution and programmatic dependent launch. |
| `q3_embedding.c` | Q3 row-split and P2E2 embedding gather to BF16 hidden states (rows may be read in place from mapped host memory). |
| `rms_norm_bf16.c` | Standalone BF16 RMSNorm. |
| `q3_linear_bf16.c` | Q3 decode (one row and the 2 to 8 row twins) and the P2E2 decode and expansion kernels. |
| `q45_linear.c` | Q4/Q5 decode (one row and the 2 to 8 row twins). |
| `nvfp4_linear.c` | NVFP4 decode (plain and SD4 scale-table layouts, 1 to 8 rows on BF16 tensor cores) and the native FP4 tensor-core route. |
| `dflash.c` | The DFlash2 drafter's operators (docs/DFLASH2.md). |
| `q3_mx.c` | The block-scaled MXFP8 route for Q3/Q4/Q5 linears and the paired gate/up SwiGLU region. |
| `reference.c`, `.h` | The scalar numerical references (Q3, Q4/Q5, NVFP4 plain and SD4): the oracle exact numerics select, and the fallback for shapes no kernel takes. |
| `qwen_layer_ops.c` | GDN control, convolution, recurrence and gated norm; residual add, residual RMSNorm and SwiGLU; NVFP4 KV append, QK norm/RoPE and attention; greedy argmax; the BF16-to-FP32 linear. |
| `qwen4_ops.c` | The Flash-Next (`qwen4_exp`) kernels behind one launcher: `qwen4_kernel_count`/`qwen4_kernel_name` and `qwen4_launch`, which checks every argument's count and width against the compiled kernel. |
| `decode_shapes.h`, `q3_p2e2_geometry.h` | Shape-only dispatch decisions and layout geometry checks (header-only). |

Quantized linears dispatch on the row count: 1 to 8 rows run the decode kernels (every row of the twins is bit for bit the one-row
result), 9 or more rows run the MXFP8 route for Q3/Q4/Q5, and NVFP4 linears from 2 rows run the native FP4 route; exact numerics and
any shape no kernel takes run the scalar reference. There are no environment variables or options that change this.

Every other file under `src/` is CUDA source, installed unchanged at the same relative path under
`share/euhedral_cuda/` next to the library. Each domain folder's `kernels.cu` is the NVRTC module
root the host loads; its headers are that module's leaves and strategies:

| Folder | Module contents |
| --- | --- |
| `common/` | Shared device helpers (`pdl.cuh`: programmatic dependent launch). |
| `embedding/` | Q3 row-split (`ROW_SPLIT_K128_V1`) and P2E2 embedding to BF16 hidden states. |
| `norm/` | Standalone BF16 RMSNorm. |
| `elementwise/` | Residual add, residual RMSNorm (including the one-row kernel) and SwiGLU. |
| `linear/` | BF16-to-FP32 linear. |
| `q3/`, `q45/` | Q3 and Q4/Q5 weight formats: layout, numerics, the one-row contiguous decode kernel and its 2 to 8 row twins; `q3/p2e2.cuh` holds the lossless P2E2 decode and expansion kernels. |
| `q3_mx/` | Prefill on block-scaled MXFP8 tensor cores for Q3, Q4 and Q5 linears and the paired gate/up SwiGLU region (`sm_12xa`). |
| `nvfp4/` | NVFP4 weights (plain and SD4 scale tables): scale-table helpers, the tensor-core decode kernels for 1 to 8 rows. |
| `nvfp4_native/` | Native FP4 tensor-core route: activation quantization, the 128x128 linear, the paired gate/up SwiGLU tile and the skinny kernels up to 64 rows (`sm_12xa`). |
| `reference/` | Scalar references (Q3, Q4/Q5, NVFP4 plain and SD4). |
| `gdn/` | Gated DeltaNet control, projections, convolution, recurrence and gated RMSNorm. |
| `attention/` | NVFP4 KV append, QK norm/RoPE, per-head and GQA decode attention with row twins, the 32-row prefill tile and the producer-warp FA2 prefill kernel, and the exact twins and single-warp controls used as test oracles. |
| `sampling/` | Greedy token selection on the device (argmax over the final logits row). |
| `qwen4/` | The Flash-Next (`qwen4_exp`) operators: BF16 linear, embedding gather, grouped RMSNorm, the hyper-connection mix and injection, the per-layer embedding (n-gram record expansion, gate, dilated convolution) and the rows of its convolution history, and (with the model's other blocks) the GDN control and gated norm, the router and the shared expert's SwiGLU. [docs/FLASH_NEXT_EXECUTION.md](../docs/FLASH_NEXT_EXECUTION.md). |
| `dflash/` | The DFlash2 drafter: BF16 tensor-core linear (rows independent of the row count), Qwen3 RMSNorm, grouped dynamic convolution, Q/K norm and RoPE, the context ring append, sliding-window block attention on tensor cores (key splits and a merge), SwiGLU, top-16 and the candidate selector, each rounding where the published PyTorch model rounds. |

Includes within a folder are relative; includes across folders name the path from the tree root
(`common/pdl.cuh`, `nvfp4/nvfp4.cuh`), which NVRTC resolves through the installed root.

The exported C API (`include/euhedral_cuda.h`) falls into these groups:
- memory and transfers: `euhedral_cuda_malloc`/`free`, `host_malloc`/`host_free`, `host_weights_malloc`/`free`/`device_pointer`,
  `device_memory_info`, the `copy_*` family, `zero_device_memory`, `synchronize`;
- streams, markers and completion: `stream_create`/`destroy`/`select`/`clear`/`synchronize`/`wait_event`, `completion_event_*`,
  `completion_notify`;
- selections: `select_exact_numerics`, `row_exact_select`, `pdl_select`;
- embedding: `embed_q3`, `embed_q3_p2e2`;
- norms and elementwise: `rms_norm_bf16`, `rms_norm_unit_offset_bf16`, `residual_add_bf16`, `residual_rms_norm_bf16`, `swiglu_bf16`;
- linears: `linear_bf16_to_float`, `linear_q3_bf16`, `linear_quantized_bf16` (Q4/Q5), `linear_nvfp4_bf16`, the P2E2 pair
  `linear_q3_p2e2_decode_bf16` and `q3_p2e2_expand`, the MXFP8 route (`q3_mx_available`, `linear_q3_mx_bf16`,
  `linear_q45_mx_bf16`, `q3_mx_gate_up_swiglu_bf16`, `q3_mx_scratch_bytes`, `q3_mx_select_split_rows`), the native FP4 route
  (`nvfp4_native_available`, `linear_nvfp4_native_bf16`, `nvfp4_native_gate_up_swiglu_bf16`, `nvfp4_native_scratch_bytes`) and
  `linear_q3_reference_bf16`;
- GDN: `gdn_control_fp32`, `gdn_project_control_fp32`, `gdn_convolution_bf16`, `gdn_recurrence_bf16`, `gdn_gated_rms_norm_bf16`;
- attention: `attention_kv_append_nvfp4`, `attention_qk_norm_rope_bf16`, `attention_causal_nvfp4`;
- sampling: `argmax_bf16`;
- Flash-Next: `qwen4_kernel_count`, `qwen4_kernel_name`, `qwen4_launch`.

## Tests

`tests/` holds the native Python/CUDA tests. They compile a source with the pinned NVRTC runtime and launch its kernels on the device,
and skip when the runtime or a GPU is unavailable; the pinned runtime is built by the Gradle native tasks. Several tests (attention, MXFP8, native FP4) need
NumPy. `gpu_harness.py` is the shared harness (NVRTC compilation, module loading, launches); the tests are:

| Test | Covers |
| --- | --- |
| `test_dflash.py` | The DFlash2 operators against NumPy oracles with the reference's rounding points: convolution, top-16 and selector bit for bit, the linear's rows identical at every row count. |
| `test_q3_kernels.py` | Q3 reference against FP64, contiguous decode against the reference, the 2 to 8 row twins bit for bit one-row. |
| `test_q45_kernels.py` | Q4/Q5 reference, contiguous decode, row twins, the capped grid. |
| `test_q3_conversion.py` | The device-level Q3 scale-conversion contract. |
| `test_q3_mx.py` | The MXFP8 route: activation quantizer, Q3/Q4/Q5 linears, split-K and the gate/up kernel against FP64 (`sm_12x`). |
| `test_q3_p2e2.py` | P2E2 decode, expansion and embedding bit for bit against the row-split routes. |
| `test_nvfp4.py` | NVFP4 decode at both tiles (rows bitwise one-row), the scalar reference, and the SD4 kernels against the plain kernels. |
| `test_nvfp4_native.py` | The native FP4 route: quantizer, linear, paired gate/up and skinny kernels against FP64, determinism, SD4 (`sm_12x`). |
| `test_attention_nvfp4.py` | NVFP4 KV format and attention against a mathematical oracle, the GQA decode kernel and the row twins bit for bit. |
| `test_gdn_convolution.py`, `test_gdn_recurrence.py` | GDN convolution and recurrence against the frozen reference kernels. |
| `test_qwen_regions.py` | Fused regions (control, residual norm, row-owned QK norm/RoPE) against the unfused numerical boundaries. |
| `test_sampling_argmax.py` | Device greedy selection against the host argmax. |
| `test_products.py` | The installed CUDA products against `native-products.json`. |

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

Use Java 25, Gradle 9.6.1, and Zig 0.16.0 (the versions recorded in `mise.toml`).
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

Build and test with the tasks of `mise.toml`:

```text
mise run native-build      # ./gradlew nativeBuildLinuxX64
mise run native-test       # python3 -m unittest discover -s native/tests -p 'test_*.py' (needs a GPU)
mise run native-verify     # ./gradlew nativeVerify: both products, installed tree equals src/
mise run cuda-test         # ./gradlew :core:cudaIntegrationTest :api:cudaIntegrationTest --rerun-tasks (reserve the GPU first)
mise run full-build        # ./gradlew nativeVerify nativePackage :api:bootJar build
```

The full-model CUDA integration tests need a `q3` artifact (`-Peuhedral.qwen.artifact`) and enough free device memory to keep it
resident; some also take an `nvfp4` artifact (`-Peuhedral.qwen.nvfp4-artifact`) or a `q3-compressed` artifact
(`-Peuhedral.qwen.q3-compressed-artifact`). Do not overlap the core and API suites on one GPU.

The runtime needs a compatible NVIDIA driver installed on the host (or injected by NVIDIA
Container Toolkit), along with target-matching CUDA user-space runtime/NVRTC libraries and
the CUDA headers used by NVRTC to compile the installed CUDA sources. Set
`EUHEDRAL_CUDA_INCLUDE_DIR` to their include directory. Driver stubs and development import
libraries are for linking only, not runtime deployment.
