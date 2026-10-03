# Raw microbenchmark results, RTX 5070 Ti

Measured on 2026-10-02 with the harness in this directory (see [README.md](README.md)). These are
the logs the documents in `docs/nvidia/` quote.

- **GPU:** RTX 5070 Ti (GB203, 70 SMs, 16 GB GDDR7), PCIe 5.0 x16, power limit raised to 350 W.
  `nvidia-smi`: driver, VBIOS, power limit, max SM and memory clock = `615.71.09, 98.03.58.00.43, 350.00 W, 3105 MHz, 14001 MHz`.
- **Software:** CUDA UMD 13.4. Kernels compiled by NVRTC 13.1 for `sm_120a` and loaded as cubins.
- **Host:** Intel i9-14900K, 64 GB, Linux 7.0. The IOMMU is in translated mode. The GNOME desktop
  runs on this GPU.
- **Conditions:** `euhedral-inference-serve` was stopped for these runs. Results are best-of-N event
  timings. Clocks are SM cycles from `clock64` over `%globaltimer` inside the kernel.

Notes on the logs:

- In the `alu` table, the `iadd` and `rcp` rows replace a first run in which ptxas folded the
  dependency chains. The three `mix_*` rows come from the same corrected run.
- The `pcie` bidirectional and combined lines are the corrected second run: both directions use the
  THP arena, and the zero-copy kernel runs at 1-2 CTAs/SM.
- The `mma num` run prints D - C. A value of 3 means the accumulator kept all three unit products.

## Device attributes (`nvbench attrs`)

```text
device            NVIDIA GeForce RTX 5070 Ti
total memory      15.56 GiB
driver API        13040
COMPUTE_CAPABILITY_MAJOR                                           12
COMPUTE_CAPABILITY_MINOR                                           0
MULTIPROCESSOR_COUNT                                               70
CLOCK_RATE                                                         2588000
MEMORY_CLOCK_RATE                                                  14001000
GLOBAL_MEMORY_BUS_WIDTH                                            256
L2_CACHE_SIZE                                                      50331648
MAX_PERSISTING_L2_CACHE_SIZE                                       31457280
MAX_ACCESS_POLICY_WINDOW_SIZE                                      134217728
MAX_THREADS_PER_BLOCK                                              1024
MAX_THREADS_PER_MULTIPROCESSOR                                     1536
MAX_BLOCKS_PER_MULTIPROCESSOR                                      24
MAX_REGISTERS_PER_BLOCK                                            65536
MAX_REGISTERS_PER_MULTIPROCESSOR                                   65536
MAX_SHARED_MEMORY_PER_BLOCK                                        49152
MAX_SHARED_MEMORY_PER_BLOCK_OPTIN                                  101376
MAX_SHARED_MEMORY_PER_MULTIPROCESSOR                               102400
RESERVED_SHARED_MEMORY_PER_BLOCK                                   1024
TOTAL_CONSTANT_MEMORY                                              65536
WARP_SIZE                                                          32
ASYNC_ENGINE_COUNT                                                 2
CONCURRENT_KERNELS                                                 1
GENERIC_COMPRESSION_SUPPORTED                                      1
CLUSTER_LAUNCH                                                     1
MEM_SYNC_DOMAIN_COUNT                                              4
GLOBAL_L1_CACHE_SUPPORTED                                          1
LOCAL_L1_CACHE_SUPPORTED                                           1
STREAM_PRIORITIES_SUPPORTED                                        1
MEMORY_POOLS_SUPPORTED                                             1
VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED                                1
HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED                        1
PAGEABLE_MEMORY_ACCESS                                             1
PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES                       0
CONCURRENT_MANAGED_ACCESS                                          1
DIRECT_MANAGED_MEM_ACCESS_FROM_HOST                                0
CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM                            1
HOST_NATIVE_ATOMIC_SUPPORTED                                       0
ONLY_PARTIAL_HOST_NATIVE_ATOMIC_SUPPORTED                          0
GPU_DIRECT_RDMA_SUPPORTED                                          0
DMA_BUF_SUPPORTED                                                  0
HOST_ALLOC_DMA_BUF_SUPPORTED                                       0
SPARSE_CUDA_ARRAY_SUPPORTED                                        1
DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED                              1
MAXIMUM_TEXTURE1D_LINEAR_WIDTH                                     268435456
MAXIMUM_TEXTURE2D_WIDTH                                            131072
MAXIMUM_TEXTURE2D_HEIGHT                                           65536
MAXIMUM_TEXTURE2D_LINEAR_WIDTH                                     131072
MAXIMUM_TEXTURE2D_LINEAR_PITCH                                     2097120
MAXIMUM_TEXTURE3D_WIDTH                                            16384
MAXIMUM_TEXTURE2D_LAYERED_LAYERS                                   2048
TEXTURE_ALIGNMENT                                                  512
TEXTURE_PITCH_ALIGNMENT                                            32
SURFACE_ALIGNMENT                                                  512
TIMELINE_SEMAPHORE_INTEROP_SUPPORTED                               1
TENSOR_MAP_ACCESS_SUPPORTED                                        1
IPC_EVENT_SUPPORTED                                                1
NUMA_CONFIG                                                        0
MPS_ENABLED                                                        0
HOST_NUMA_ID                                                       0
GPU_PCI_DEVICE_ID                                                  738529502
PCI_BUS_ID                                                         1
SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO                              64
COMPUTE_PREEMPTION_SUPPORTED                                       1
COOPERATIVE_LAUNCH                                                 1
KERNEL_EXEC_TIMEOUT                                                1
UNIFIED_ADDRESSING                                                 1
ECC_ENABLED                                                        0
green ctx SM resource: smCount=70 minSmPartitionSize=8 smCoscheduledAlignment=8
free now          14.82 GiB
```

## Streaming DRAM bandwidth vs bytes in flight (`nvbench mem bw`)

```text

## streaming read, 2048 MiB (cold: buffer is 43x L2)
kernel     blk/SM  warps  KB/SM fly      GB/s      MHz B/clk/SM
read_v4_u1      1      8          4     725.3     2878     3.60
read_v4_u1      2     16          8     847.4     2876     4.21
read_v4_u1      4     32         16     853.6     2878     4.24
read_v4_u1      6     48         24     851.5     2878     4.23
read_v4_u2      1      8          8     843.7     2878     4.19
read_v4_u2      2     16         16     853.6     2878     4.24
read_v4_u2      4     32         32     854.8     2879     4.24
read_v4_u2      6     48         48     850.1     2878     4.22
read_v4_u4      1      8         16     853.5     2879     4.24
read_v4_u4      2     16         32     855.4     2879     4.24
read_v4_u4      4     32         64     853.5     2880     4.23
read_v4_u4      6     48         96     850.8     2864     4.24
read_v4_u8      1      8         32     853.6     2864     4.26
read_v4_u8      2     16         64     852.9     2864     4.25
read_v4_u8      4     32        128     853.2     2865     4.25
read_v4_u8      6     48        192     850.5     2864     4.24
read_u32        4     32         32     851.2     2864   (LDG.32 x8)
read_u32        6     48         48     850.3     2863   (LDG.32 x8)
read_v8         2     16         64     853.6     2864   (LDG.256 x4)
read_v8         4     32        128     854.0     2864   (LDG.256 x4)
read_v8         6     48        192     848.1     2865   (LDG.256 x4)
write_v4        4     32          -     827.6     2860
copy_v4         4     32          -     761.8     2863   (read+write bytes)
```

## TMA bulk-copy ring (`nvbench mem bulk`)

```text

## cp.async.bulk global->shared ring (one issuing thread per CTA), 2048 MiB
   chunk  stages CTA/SM  KB/SM fly      GB/s      MHz
    4096       2      4         32     855.1     2873
    4096       4      4         64     855.0     2859
    4096       8      3         96     855.0     2863
    8192       2      4         64     854.9     2860
    8192       4      3         96     855.6     2860
    8192       8      1         64     852.9     2862
   16384       2      3         96     855.0     2861
   16384       4      1         64     853.7     2861
   32768       2      1         64     852.6     2863
```

## L2 capacity and bandwidth (`nvbench mem l2`)

```text

## re-read bandwidth vs working set (ld.global.cg, 4 loads in flight/thread)
      MiB      GB/s      MHz   B/clk/SM
        1    3071.2     2862      15.33
        2    3307.5     2870      16.46
        4    3463.9     2869      17.25
        8    3545.3     2871      17.64
       16    3545.9     2872      17.63
       24    3553.4     2857      17.77
       32    3564.7     2857      17.82
       40    3565.0     2858      17.82
       44    3561.3     2857      17.81
       48    3582.8     2857      17.91
       56    2009.8     2860      10.04
       64     993.1     2861       4.96
       96     863.5     2864       4.31
      128     852.9     2863       4.26
      256     850.1     2863       4.24
```

## Latency and TLB reach (`nvbench mem lat`)

```text

## dependent-load latency (random cycle over 128-byte lines)
 footprint     mode   cycles   ns@clk
      16KB       ca       45
      16KB       cg      356
      64KB       ca       45
      64KB       cg      356
      96KB       ca      149
      96KB       cg      355
     512KB       ca      356
     512KB       cg      355
    2048KB       ca      356
    2048KB       cg      356
    3072KB       cg      397
    4096KB       cg      501
    6144KB       cg      527
    8192KB       cg      537
   16384KB       cg      670
   24576KB       cg      698
   40960KB       cg      702
   65536KB       cg      728
  262144KB       cg      766
 1048576KB       cg      774

## TLB reach: one line per page, random page order (cg loads)
      page    pages   cycles
       64K        8      357
       64K       16      355
       64K       32      356
       64K       64      357
       64K      128      356
       64K      256      356
       64K      512      356
       64K     1024      355
     2048K        8      360
     2048K       16      357
     2048K       32      358
     2048K       64      356
     2048K      128      355
     2048K      256      360
     2048K      512      363
     2048K     1024      364
      smem      lds     34.0
```

## Tensor-core throughput per mma kind (`nvbench mma`)

```text
variant                instruction                                     TFLOPS      MHz FLOP/clk/SM warps/SM  lat(clk)
f16_f32                m16n8k16 f16 -> f32 acc                          102.2     2860        511       16      34.7
f16_f16                m16n8k16 f16 -> f16 acc                          204.4     2865       1019       32      29.2
bf16_f32               m16n8k16 bf16 -> f32                             102.5     2872        510       32      34.7
tf32_f32               m16n8k8 tf32 -> f32                               51.3     2857        256       32      34.7
e4m3_f32               m16n8k32 e4m3 -> f32 (sm_89 form)                204.9     2857       1025       32      34.7
e4m3_f16               m16n8k32 e4m3 -> f16 acc                         408.8     2866       2038       32      29.2
f8f6f4_e4m3_f32        kind::f8f6f4 e4m3 -> f32                         205.0     2869       1021       32      34.7
f8f6f4_e4m3_f16        kind::f8f6f4 e4m3 -> f16 acc                     407.9     2848       2046       32      29.2
f8f6f4_e3m2_f32        kind::f8f6f4 e3m2 (FP6) -> f32                   203.9     2852       1021       32      34.7
f8f6f4_e2m1_f32        kind::f8f6f4 e2m1 (FP4, 8-bit containers)        204.2     2858       1021       32      34.7
f8f6f4_e2m1xe4m3_f32   kind::f8f6f4 e2m1 x e4m3 (mixed)                 203.9     2853       1021       32      34.7
mxf8f6f4_e4m3          kind::mxf8f6f4 e4m3, ue8m0 x1 (MXFP8)            407.7     2856       2039       32      29.2
mxf8f6f4_e2m1          kind::mxf8f6f4 e2m1, ue8m0 x1                    205.0     2872       1020       32      34.7
mxf4_2x                kind::mxf4 m16n8k64, ue8m0 x2 (MXFP4)            815.4     2871       4058       32      29.2
mxf4nvf4_4x            kind::mxf4nvf4 m16n8k64, ue4m3 x4 (NVFP4)        815.4     2870       4059       32      29.2
mxf4nvf4_2x_ue8m0      kind::mxf4nvf4 m16n8k64, ue8m0 x2                815.5     2862       4070       32      29.2
s8_s32                 m16n8k32 s8 -> s32 (IMMA)                        407.8     2860       2037       32      27.2
s4_s32                 m16n8k64 s4 -> s32                                41.9     2845        210       32     398.7
b1_and                 m16n8k256 b1 and.popc                            251.5     2847       1262       16     383.7
sp_f16_f32             2:4 sparse m16n8k32 f16 -> f32                   203.9     2858       1019       32      34.7
sp_e4m3_f32            2:4 sparse m16n8k64 e4m3 -> f32                  407.8     2854       2041       32      34.7
sp_f8f6f4_e2m1         2:4 sparse kind::f8f6f4 e2m1                     408.3     2861       2039       32      34.7
sp_mxf4_2x             2:4 sparse kind::mxf4 m16n8k128                 1617.4     2851       8105       32      29.8
sp_mxf4nvf4_4x         2:4 sparse kind::mxf4nvf4 m16n8k128             1621.3     2848       8132       32      29.8
```

## Tensor-core accumulator precision (`nvbench mma num`)

```text
## test 1: D = C + 3 products of 1.0, C = 2^e; printed: D - C
kind \ e            8    10    11    12    13    14    15    16    18    20    22    23    24    25    26
f16_f32             3     3     3     3     3     3     3     3     3     3     3     3     2     0     0
bf16_f32            3     3     3     3     3     3     3     3     3     3     3     3     2     0     0
f16_f16             3     3     4     4     0     0     0     -     -     -     -     -     -     -     -
e4m3_f32            3     3     3     3     3     3     3     3     3     3     3     3     2     0     0
e4m3_f16            3     3     4     4     0     0     0     -     -     -     -     -     -     -     -
mxf8f6f4_e4m3       3     3     3     3     3     3     3     3     3     3     3     3     2     0     0
mxf4nvf4_4x         3     3     3     3     3     3     3     3     3     3     3     3     2     0     0
mxf4_2x             3     3     3     3     3     3     3     3     3     3     3     3     2     0     0
s8_s32              3     3     3     3     3     3     3     3     3     3     3     3     4     4     0
## test 2: D = one product 2^e + three products of 1.0 (C = 0); printed: D - 2^e
kind \ e            8    10    11    12    13    14    15    16    18    20    22    23    24    25    26
f16_f32             3     3     3     3     3     3     3     3     3     3     3     3     2     0     0
bf16_f32            3     3     3     3     3     3     3     3     3     3     3     3     2     0     0
f16_f16             3     3     4     4     0     0     0     -     -     -     -     -     -     -     -
e4m3_f32            3     3     3     3     3     3     3     3     -     -     -     -     -     -     -
e4m3_f16            3     3     4     4     0     0     0     -     -     -     -     -     -     -     -
mxf8f6f4_e4m3       3     3     3     3     3     3     3     3     -     -     -     -     -     -     -
mxf4nvf4_4x         3     3     3     3     3     3     3     3     -     -     -     -     -     -     -
mxf4_2x             3     3     3     3     3     3     3     3     3     3     3     3     2     0     0
s8_s32              -     -     -     -     -     -     -     -     -     -     -     -     -     -     -
```

## CUDA-core, SFU and conversion throughput (`nvbench alu`)

```text
variant        instruction                                       Gops/s      MHz  inst/clk/SM  elem/clk/SM
ffma           fma.rn.f32                                       24617.2     2867        122.7        122.7
ffma2          fma.rn.f32x2 (packed FP32)                       12525.7     2863         62.5        125.0
fadd2          add.rn.f32x2 (packed FP32)                       12644.3     2867         63.0        126.0
hfma2          fma.rn.f16x2                                     12653.0     2870         63.0        126.0
bfma2          fma.rn.bf16x2                                    12655.2     2873         62.9        125.8
fmax           max.f32                                          24906.8     2871        123.9        123.9
imad           mad.lo.u32                                       12645.1     2872         62.9         62.9
iadd           2x add.u32 (Fibonacci pair)                      12202.9     2855        122.1        122.1
mix_ffma_lop3  fma.rn.f32 + lop3.b32 (independent)              11817.8     2857        118.2        118.2
mix_ffma_imad  fma.rn.f32 + mad.lo.u32 (independent)            11219.9     2856        112.3        112.3
mix_ffma_ex2   4x fma.rn.f32 + ex2.approx (independent)          2712.3     2872         67.5         67.5
lop3           lop3.b32                                         12695.4     2874         63.1         63.1
shf            shf.l.wrap.b32                                   12693.2     2873         63.1         63.1
prmt           prmt.b32                                         12435.1     2858         62.2         62.2
dp4a           dp4a.u32.u32 (4 MACs)                            12600.2     2874         62.6        250.5
popc           popc.b32 + xor                                    3208.2     2875         31.9         15.9
i2f            cvt.rn.f32.u32                                   12790.5     2873         63.6         63.6
f2i            cvt.rzi.u32.f32                                   3213.8     2873         16.0         16.0
f2f16x2        cvt.rn.f16x2.f32                                 12798.7     2873         63.6        127.3
f2e4m3x2       cvt.rn.satfinite.e4m3x2.f32 (+widen)              6409.6     2873         63.7         63.7
e4m3x2_f16x2   cvt.rn.f16x2.e4m3x2                               7351.8     2873         36.6         73.1
e2m1x2_f16x2   4x cvt.rn.f16x2.e2m1x2 + 3 xor (8 FP4->FP16)      2139.6     2873         74.5         85.1
f2e2m1x2       cvt.rn.satfinite.e2m1x2.f32 (+widen)              6383.2     2857         63.8         63.8
ex2            ex2.approx.ftz.f32 (MUFU)                         3211.0     2873         16.0         16.0
rcp            rcp.approx.ftz.f32 + fmul (MUFU)                  3210.7     2876         31.9         15.9
rsqrt          rsqrt.approx.ftz.f32 (MUFU)                       3211.3     2879         15.9         15.9
lg2            lg2.approx.ftz.f32 (MUFU)                         3213.9     2874         16.0         16.0
sin            sin.approx.ftz.f32 (MUFU)                         3207.9     2877         15.9         15.9
tanh           tanh.approx.f32 (MUFU)                            3207.9     2872         16.0         16.0
ex2_f16x2      ex2.approx.f16x2                                  1607.8     2874          8.0         16.0
ex2_bf16x2     ex2.approx.ftz.bf16x2                             1606.9     2875          8.0         16.0
tanh_f16x2     tanh.approx.f16x2                                 1606.9     2874          8.0         16.0
shfl           shfl.sync.bfly.b32                                3205.8     2862         16.0         16.0
redux          redux.sync.add.u32                                3207.4     2873         16.0         16.0
dfma           fma.rn.f64                                         338.1     2873          1.7          1.7
```

## Texture path as a load pipe (`nvbench tex pipes`)

```text

## linear-memory loads: LDG vs TEX vs half/half (max 1D linear texels 268435456)
kernel     set           MiB       GB/s      MHz words/clk/SM
ldg_u32    L2-hot         16     5948.3     2866         7.41
tex_u32    L2-hot         16     5626.2     2872         7.00
mix_u32    L2-hot         16     5236.3     2864         6.53
ldg_v4     L2-hot         16     4958.5     2872         1.54
tex_v4     L2-hot         16     5104.1     2873         1.59
mix_v4     L2-hot         16     4762.9     2873         1.48
unorm8x4   L2-hot         16     3969.5     2866         4.95  (u8x4 -> 4 floats; values/clk/SM 19.8)
ldg_u32    DRAM         1024      852.8     2862         1.06
tex_u32    DRAM         1024      852.8     2862         1.06
mix_u32    DRAM         1024      854.0     2862         1.07
ldg_v4     DRAM         1024      855.3     2864         0.27
tex_v4     DRAM         1024      855.0     2862         0.27
mix_v4     DRAM         1024      854.9     2863         0.27
unorm8x4   DRAM         1024      851.3     2861         1.06  (u8x4 -> 4 floats; values/clk/SM 4.3)
```

## BC4 decode precision and hardware palette (`nvbench tex precision`)

```text

## BC4 decode precision vs exact D3D10 interpolation (64x64 texels, random blocks)
BC4u  max |hw-exact| = 0.0249 (6.35 LSB of 1/255), mean 0.00243; values on the 8-bit grid: 36.6%; gather order mismatches: 0
BC4s  max |hw-exact| = 0.0506 (6.42 LSB of 1/127), mean 0.00552; values on the 8-bit grid: 15.2%; gather order mismatches: 0

## BC4 hardware palette (index -> decoded value), endpoints at the range limits
UNORM r0=255 r1=0 (8-level)         1.00000  0.00000  0.85992  0.71984  0.56031  0.43969  0.28016  0.14008
  exact D3D10                       1.00000  0.00000  0.85714  0.71429  0.57143  0.42857  0.28571  0.14286
UNORM r0=0 r1=255 (6-level)         0.00000  1.00000  0.18677  0.37354  0.62646  0.81323  0.00000  1.00000
  exact D3D10                       0.00000  1.00000  0.20000  0.40000  0.60000  0.80000  0.00000  1.00000
UNORM r0=200 r1=40 (8-level)        0.78431  0.15686  0.69642  0.60853  0.50843  0.43275  0.33265  0.24475
  exact D3D10                       0.78431  0.15686  0.69468  0.60504  0.51541  0.42577  0.33613  0.24650
SNORM r0=127 r1=-127 (8-level)      1.00000 -1.00000  0.72094  0.44188  0.11625 -0.11625 -0.44188 -0.72094
  exact D3D10                       1.00000 -1.00000  0.71429  0.42857  0.14286 -0.14286 -0.42857 -0.71429
SNORM r0=-127 r1=127 (6-level)     -1.00000  1.00000 -0.62792 -0.25584  0.25584  0.62792 -1.00000  1.00000
  exact D3D10                      -1.00000  1.00000 -0.60000 -0.20000  0.20000  0.60000 -1.00000  1.00000
```

## Block-compressed decode throughput (`nvbench tex bc`)

```text

## BC decode throughput (point fetch and tld4 gather)
fmt    set          MiB fetch     Gtexel/s    Gvalues/s  GB/s(cmp)      MHz values/clk/SM
BC1    L2-hot         2 point1       950.7        950.7      475.3     2857         4.75
BC1    L2-hot         2 point2       949.0       1898.1      474.5     2849         9.52
BC1    L2-hot         2 point4       951.2       3804.8      475.6     2857        19.03
BC1    L2-hot         2 gather       821.8       3287.1     1643.5     2854        16.45
BC1    DRAM         512 point1       849.6        849.6      424.8     2860         4.24
BC1    DRAM         512 point2       849.8       1699.5      424.9     2845         8.53
BC1    DRAM         512 point4       848.6       3394.3      424.3     2842        17.06
BC1    DRAM         512 gather       402.1       1608.3      804.2     2858         8.04
BC4u   L2-hot         2 point1       949.9        949.9      475.0     2843         4.77
BC4u   L2-hot         2 gather       798.3       3193.2     1596.6     2839        16.07
BC4u   DRAM         512 point1       852.6        852.6      426.3     2861         4.26
BC4u   DRAM         512 gather       402.0       1607.9      803.9     2873         8.00
BC4s   L2-hot         2 point1       950.7        950.7      475.3     2845         4.77
BC4s   L2-hot         2 gather       803.2       3212.8     1606.4     2837        16.18
BC4s   DRAM         512 point1       855.3        855.3      427.6     2846         4.29
BC4s   DRAM         512 gather       400.7       1602.8      801.4     2858         8.01
BC5s   L2-hot         2 point1       478.6        478.6      478.6     2851         2.40
BC5s   L2-hot         2 point2       478.1        956.2      478.1     2846         4.80
BC5s   L2-hot         2 gather       443.0       1772.0     1772.0     2845         8.90
BC5s   DRAM        1024 point1       444.1        444.1      444.1     2838         2.24
BC5s   DRAM        1024 point2       443.8        887.5      443.8     2812         4.51
BC5s   DRAM        1024 gather       207.7        830.8      830.8     2785         4.26
BC6Hs  L2-hot         2 point1       474.6        474.6      474.6     2789         2.43
BC6Hs  L2-hot         2 point2       477.8        955.5      477.8     2790         4.89
BC6Hs  L2-hot         2 point4       478.0       1434.1      478.0     2852         7.18
BC6Hs  L2-hot         2 gather       469.6       1878.3     1878.3     2851         9.41
BC6Hs  DRAM        1024 point1       444.1        444.1      444.1     2866         2.21
BC6Hs  DRAM        1024 point2       446.0        892.0      446.0     2826         4.51
BC6Hs  DRAM        1024 point4       444.5       1333.4      444.5     2722         7.00
BC6Hs  DRAM        1024 gather       198.9        795.6      795.6     2704         4.20
BC7    L2-hot         2 point1       474.7        474.7      474.7     2844         2.38
BC7    L2-hot         2 point2       475.9        951.8      475.9     2845         4.78
BC7    L2-hot         2 point4       475.8       1903.3      475.8     2849         9.54
BC7    L2-hot         2 gather       453.3       1813.4     1813.4     2852         9.08
BC7    DRAM        1024 point1       443.7        443.7      443.7     2855         2.22
BC7    DRAM        1024 point2       444.1        888.2      444.1     2852         4.45
BC7    DRAM        1024 point4       444.9       1779.5      444.9     2757         9.22
BC7    DRAM        1024 gather       207.4        829.7      829.7     2729         4.34
```

## GEMV with BC4 weights decoded by the texture units (`nvbench tex gemv`)

```text

## GEMV y = W x, W 4 bits/weight, cold (rotating 6 copies)
kernel          K      N         us       GB/s      MHz
bc4_tld4     5120  17408       58.9      756.2     2848
int4_alu     5120  17408      463.0       96.3     2844
bc4_tld4    17408   5120       66.9      666.0     2853
int4_alu    17408   5120      491.1       90.7     2857
bc4_tld4     5120   5120       21.4      613.7     2870
int4_alu     5120   5120      145.6       90.0     2833
bc4_tld4    12288   5120       48.2      652.0     2855
int4_alu    12288   5120      347.0       90.6     2846
```

## Host transfers, copy engines, zero-copy (`nvbench pcie`)

```text
async (copy) engines reported: 2
THP arena: AnonHugePages grew by 1024 MiB of 1024 MiB

## host->device and device->host copies (GB/s, best of 4, 1 GiB)
host memory               chunk  streams        H2D        D2H
cuMemHostAlloc               2M        1       26.4       23.9
cuMemHostAlloc               2M        2       25.5       24.4
cuMemHostAlloc              16M        1       25.5       23.9
cuMemHostAlloc              16M        2       25.2       23.8
cuMemHostAlloc              64M        1       24.8       23.6
cuMemHostAlloc              64M        2       24.3       23.5
cuMemHostAlloc            1024M        1       25.1       25.1
THP+cuMemHostRegister        2M        1       43.4       55.7
THP+cuMemHostRegister        2M        2       43.2       55.6
THP+cuMemHostRegister       16M        1       43.9       56.8
THP+cuMemHostRegister       16M        2       43.6       56.8
THP+cuMemHostRegister       64M        1       43.5       57.1
THP+cuMemHostRegister       64M        2       43.4       57.0
THP+cuMemHostRegister     1024M        1       43.6       57.1
pageable malloc             64M        1       15.3       15.2
pageable malloc           1024M        1       15.5       15.4

## zero-copy kernel reads of the pinned THP arena (512 MiB)
read_v4_u4   1 CTA/SM:   40.3 GB/s
read_v4_u4   2 CTA/SM:   40.6 GB/s
read_v4_u4   4 CTA/SM:   15.0 GB/s
read_v4_u4   6 CTA/SM:   12.6 GB/s
read_v4_u8   1 CTA/SM:   40.8 GB/s
read_v4_u8   2 CTA/SM:   40.1 GB/s
read_v4_u8   4 CTA/SM:   18.5 GB/s
read_v4_u8   6 CTA/SM:   12.5 GB/s


bidirectional 512 MiB each way (THP arena): 54.1 GB/s total (27.0 per direction)
zero-copy kernel 1 CTA/SM (512 MiB) || DMA H2D (512 MiB): 37.6 GB/s combined
zero-copy kernel 2 CTA/SM (512 MiB) || DMA H2D (512 MiB): 39.2 GB/s combined
cuMemcpyBatchAsync H2D 16 x 64 MiB: result 0, 43.0 GB/s
cuMemcpyDtoDAsync 1 GiB: 387.1 GB/s (read+write 774.3)
cuMemcpyBatchAsync D2D PREFER_OVERLAP_WITH_COMPUTE: result 0, 133.8 GB/s
```

## Persisting L2, eviction hints, compressible memory (`nvbench l2`)

```text
L2 48 MiB, max persisting 30.0 MiB, max window 128.0 MiB

## re-reading a hot buffer after each 1 GiB streaming pass (GB/s of the hot read)
 hot MiB    alone(L2)        plain createpolicy persist window  window+hint
       8       1899.6        819.2       2048.0         2048.0       2048.0
      16       2621.4        910.2       2880.7         2730.7       2730.7
      24       2612.7        819.2       2561.7         3072.0       3072.0
      32       2865.0        819.2       2730.7         2730.7       2730.7

## generic (compute data) compression supported: 1
granted compressionType = 1 (1 = generic)
allocation     data      read GB/s write GB/s
COMP_GENERIC   zeros        4995.4          -
COMP_GENERIC   random        851.1          -
COMP_GENERIC   pattern       856.7           
               (write)                  790.8
cuMemAlloc     zeros         853.8          -
cuMemAlloc     random        853.8          -
cuMemAlloc     pattern       853.9           
               (write)                  828.1
## which data compresses? read GB/s from a COMP_GENERIC allocation (1 GiB, cold)
  zeros                    4999.2 GB/s
  constant 1.0f            3566.6 GB/s
  50% zero words            998.7 GB/s
  90% zero words           2993.5 GB/s
  50% zero 128B lines      1546.8 GB/s
  bytes 0..15               850.9 GB/s
  BF16 |x| in [1,2)         849.8 GB/s
  FP32 same exponent        850.1 GB/s
  u32 values 0..255        1690.9 GB/s
  random                    849.8 GB/s
```

## Clusters, DSMEM, launch and graph overheads (`nvbench misc`)

```text
## thread-block clusters
non-portable allowed=0: max potential cluster size = 8 (r=0)
   cluster  2: max active clusters 70 (140 CTAs)  r=0
   cluster  4: max active clusters 35 (140 CTAs)  r=0
   cluster  8: max active clusters 17 (136 CTAs)  r=0
   cluster 16: max active clusters 0 (0 CTAs)  r=912
non-portable allowed=1: max potential cluster size = 12 (r=0)
   cluster  2: max active clusters 70 (140 CTAs)  r=0
   cluster  4: max active clusters 35 (140 CTAs)  r=0
   cluster  8: max active clusters 17 (136 CTAs)  r=0
   cluster 16: max active clusters 0 (0 CTAs)  r=0
local shared reads: 51.7 bytes/clk per CTA (CTA 0: 2596238 cycles), aggregate 18879 GB/s
DSMEM (peer CTA) shared reads: 2.3 bytes/clk per CTA (CTA 0: 59135945 cycles), aggregate 706 GB/s

## launch overheads (2000 back-to-back tiny kernels, 1 CTA x 128 threads)
stream launch:     host 0.78 us/launch, GPU 0.78 us/kernel
PDL launch:        host 0.77 us/launch, GPU 0.77 us/kernel
graph (plain    ): host 3.5 us/graph, GPU 0.39 us/kernel node
graph (PDL edges): host 3.2 us/graph, GPU 0.30 us/kernel node
WHILE node: 2000 iterations, GPU 3.23 us/iteration (counter left 0)
```
