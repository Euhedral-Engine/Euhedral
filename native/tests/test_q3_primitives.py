"""Boundary tests for the Q3 module in native/src/q3/.

Each test includes the production module and appends a small probe kernel, so
probes exercise the same primitives as the shipped entry points. Tests skip
when the pinned NVRTC runtime or CUDA device is unavailable.
"""

import contextlib
import ctypes as C
import pathlib
import random
import struct
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE = ROOT / "native" / "src" / "q3" / "kernels.cu"
RUNTIME = ROOT / "build" / "cuda-dev" / "linux-x64" / "runtime"
INCLUDE = ROOT / "build" / "cuda-dev" / "linux-x64" / "include"
P, I = C.c_void_p, C.c_int

PROBES = r"""
// Lane-stripe ownership handoff: packed load -> stripe transfer -> decode.
// out[g * 64 + lane + 32 * s] = code for K offset lane + 32 * s in group g.
extern "C" __global__ void probe_stripe_codes(const unsigned char* bytes, int* out, unsigned int groups) {
    q3::Layout w(bytes, groups * 64u, 0);
    unsigned int lane = threadIdx.x;
    for (unsigned int g = 0; g < groups; g++) {
        unsigned int pairs = q3::load_packed_pair(w, g, lane);
        for (int s = 0; s < 2; s++)
            out[g * 64 + lane + 32 * s] = q3::decode_code(q3::stripe_pair(pairs, lane, s), lane & 1);
    }
}
// Load ownership: lane holds K offsets 2*lane and 2*lane+1.
extern "C" __global__ void probe_pair_codes(const unsigned char* bytes, int* out, unsigned int groups) {
    q3::Layout w(bytes, groups * 64u, 0);
    unsigned int lane = threadIdx.x;
    for (unsigned int g = 0; g < groups; g++) {
        unsigned int pairs = q3::load_packed_pair(w, g, lane);
        for (int p = 0; p < 2; p++) out[g * 64 + lane * 2 + p] = q3::decode_code(pairs, p);
    }
}
// Reference unpack, thread-local.
extern "C" __global__ void probe_reference_codes(const unsigned char* bytes, int* out, unsigned int groups) {
    for (unsigned int i = threadIdx.x; i < groups * 64; i += blockDim.x)
        out[i] = q3::load_code_at(bytes + (i / 64) * 24, i % 64);
}
// Scale application and hi/lo execution-tile staging for every code x scale.
extern "C" __global__ void probe_split_weights(
        const unsigned short* scales, float* exact, float* hi, float* lo, unsigned int count) {
    __shared__ __nv_bfloat16 h[8], l[8];
    for (unsigned int s = 0; s < count; s++) {
        if (threadIdx.x < 8) {
            float scale = q3::fp16_to_float(scales[s]);
            int code = (int)threadIdx.x - 4;
            float weight = q3::apply_scale(code, scale);
            q3::stage_split_weight(h, l, threadIdx.x, weight);
            exact[s * 8 + threadIdx.x] = weight;
            hi[s * 8 + threadIdx.x] = __bfloat162float(h[threadIdx.x]);
            lo[s * 8 + threadIdx.x] = __bfloat162float(l[threadIdx.x]);
        }
    }
}
// Pair staging vs the per-weight split: one warp covers all 64 6-bit
// code pairs in two passes. Output order: [pass][lane][p] of (hi, lo) bits.
extern "C" __global__ void probe_split_pairs(
        const unsigned short* scales, unsigned short* actual, unsigned short* expected, unsigned int count) {
    __shared__ __align__(4) __nv_bfloat16 h[64], l[64], rh[64], rl[64];
    if (threadIdx.x >= 32) return;  // one warp covers all 64 code pairs
    unsigned int lane = threadIdx.x;
    for (unsigned int s = 0; s < count; s++) {
        float scale = q3::fp16_to_float(scales[s]);
        for (unsigned int pass = 0; pass < 2; pass++) {
            unsigned int codes = lane + pass * 32;
            q3::stage_split_pair(h, l, lane * 2, codes, scale);
            for (int p = 0; p < 2; p++)
                q3::stage_split_weight(rh, rl, lane * 2 + p, q3::apply_scale(q3::decode_code(codes, p), scale));
            __syncwarp();
            unsigned long long o = ((unsigned long long)s * 2 + pass) * 128 + lane * 4;
            for (int p = 0; p < 2; p++) {
                actual[o + p * 2] = __bfloat16_as_ushort(h[lane * 2 + p]);
                actual[o + p * 2 + 1] = __bfloat16_as_ushort(l[lane * 2 + p]);
                expected[o + p * 2] = __bfloat16_as_ushort(rh[lane * 2 + p]);
                expected[o + p * 2 + 1] = __bfloat16_as_ushort(rl[lane * 2 + p]);
            }
            __syncwarp();
        }
    }
}
// Independent per-group decode control: the original G64 load schedule with
// the same lane ownership, stripe order, and FP32 accumulation order.
template<int R>
__device__ void probe_group_decode(
        const unsigned short* input, const unsigned char* weights, unsigned short* output,
        unsigned int rows, unsigned int in_features, unsigned int out_features,
        unsigned long long scale_offset) {
    constexpr int kWarpCols = 2, kStripes = 4;
    const unsigned int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const unsigned int output_tiles = (out_features + 7u) / 8u;
    const unsigned int first_out = (blockIdx.x % output_tiles) * 8 + warp * kWarpCols;
    const unsigned int first_row = (blockIdx.x / output_tiles) * R;
    const q3::Layout w(weights, in_features, scale_offset);
    float sums[R][kWarpCols][kStripes] = {};
    for (unsigned int k = 0; k < in_features; k += 2 * q3::kGroup) {
        float x[R][kStripes];
        q3::load_activation_stripes<R, kStripes>(x, input, rows, in_features, first_row, k, lane);
        for (int n = 0; n < kWarpCols; n++) {
            if (first_out + n < out_features) {
                for (int half = 0; half < 2; half++) {
                    unsigned long long g = w.group(first_out + n, k / q3::kGroup + half);
                    unsigned int pairs = q3::load_packed_pair(w, g, lane);
                    float scale = q3::load_group_scale(w, g, lane);
                    for (int p = 0; p < 2; p++) {
                        int code = q3::decode_code(q3::stripe_pair(pairs, lane, p), lane & 1);
                        for (int r = 0; r < R; r++)
                            sums[r][n][half * 2 + p] += q3::scaled_product(x[r][half * 2 + p], code, scale);
                    }
                }
            }
        }
    }
    for (int r = 0; r < R; r++)
        for (int n = 0; n < kWarpCols; n++) {
            float sum = q3::reduce_stripes(sums[r][n]);
            if (lane == 0 && first_row + r < rows && first_out + n < out_features)
                q3::write_bf16(output, first_row + r, first_out + n, out_features, sum);
        }
}
#define PROBE_GROUP_DECODE(R) \
extern "C" __global__ __launch_bounds__(128) void probe_group_decode_##R( \
        const unsigned short* input, const unsigned char* weights, unsigned short* output, \
        unsigned int rows, unsigned int in_features, unsigned int out_features, unsigned long long scale_offset) { \
    probe_group_decode<R>(input, weights, output, rows, in_features, out_features, scale_offset); \
}
PROBE_GROUP_DECODE(1)
PROBE_GROUP_DECODE(2)
PROBE_GROUP_DECODE(4)
// Activation staging: tile contents including zero-fill past rows / K.
extern "C" __global__ void probe_activation_tile(
        const unsigned short* input, float* out, unsigned int rows, unsigned int in_features,
        unsigned int row_start, unsigned int k_base) {
    __shared__ __nv_bfloat16 tile[32 * 64];
    q3::stage_activation_tile<32, 64, 128>(tile, input, rows, in_features, row_start, k_base, threadIdx.x);
    __syncthreads();
    for (unsigned int i = threadIdx.x; i < 32 * 64; i += 128) out[i] = __bfloat162float(tile[i]);
}
// Output writeback: only in-matrix elements of the tile are written.
extern "C" __global__ void probe_output_tile(
        unsigned short* output, unsigned int rows, unsigned int out_features,
        unsigned int row_start, unsigned int out_start) {
    __shared__ float result[32 * 32];
    for (unsigned int i = threadIdx.x; i < 32 * 32; i += 128) result[i] = (float)(i + 1);
    __syncthreads();
    q3::write_output_tile<32, 32, 128>(output, result, rows, out_features, row_start, out_start, threadIdx.x);
}
// Recomposed geometries: the same primitives under tiles no production route uses.
#define PROBE_TILED(NAME, TILE) \
extern "C" __global__ __launch_bounds__(128) void NAME( \
        const unsigned short* input, const unsigned char* weights, unsigned short* output, \
        unsigned int rows, unsigned int in_features, unsigned int out_features, unsigned long long scale_offset) { \
    using Tile = TILE; \
    __shared__ q3::PrefillShared<Tile> staging; \
    __shared__ __align__(32) __nv_bfloat16 b_hi[Tile::kCols * q3::kGroup]; \
    __shared__ __align__(32) __nv_bfloat16 b_lo[Tile::kCols * q3::kGroup]; \
    q3::tiled_prefill<Tile>(input, weights, output, rows, in_features, out_features, scale_offset, \
            staging, b_hi, b_lo); \
}
PROBE_TILED(probe_tiled_64x16, q3::WarpTile<4 COMMA 1 COMMA 1>)
PROBE_TILED(probe_tiled_16x64, q3::WarpTile<1 COMMA 4 COMMA 1>)
PROBE_TILED(probe_unpadded_32, q3::Prefill32)
PROBE_TILED(probe_unpadded_64, q3::Prefill64)
// Leaf probes: the same padded CTA AUTO staging with an explicit MMA leaf.
// DumpLeaf also copies each CTA's FP32 result tile, before BF16 conversion, to
// probe_accumulators[blockIdx.x] so leaves can be compared bit for bit.
__device__ float probe_accumulators[64 * 2048];
template<class Inner>
struct DumpLeaf {
    template<class Tile> struct Of {
        using Leaf = typename Inner::template Of<Tile>::Leaf;
        using Acc = typename Leaf::Acc;
        static __device__ __forceinline__ void fill(Acc& acc) { Leaf::fill(acc); }
        template<int K_TILE, int LDA, int LDB>
        static __device__ __forceinline__ void consume(Acc& acc, const __nv_bfloat16* a,
                const __nv_bfloat16* b_hi, const __nv_bfloat16* b_lo, unsigned int warp) {
            Leaf::template consume<K_TILE, LDA, LDB>(acc, a, b_hi, b_lo, warp);
        }
        static __device__ __forceinline__ void store(float* result, Acc& acc, unsigned int warp) {
            Leaf::store(result, acc, warp);
            __syncwarp();
            float* dump = probe_accumulators + (unsigned long long)blockIdx.x * Tile::kRows * Tile::kCols;
            for (int m = 0; m < Tile::kFrags; m++)
                for (unsigned int i = threadIdx.x & 31u; i < 256u; i += 32u) {
                    unsigned int at = (Tile::row(warp, m) + i / 16u) * Tile::kCols + Tile::col(warp) + i % 16u;
                    dump[at] = result[at];
                }
        }
    };
};
template<template<class> class L> struct LeafOf { template<class Tile> struct Of { using Leaf = L<Tile>; }; };
template<class Tile> using WmmaDump = DumpLeaf<LeafOf<q3::WmmaLeaf>>::Of<Tile>;
template<class Tile> using MmaDump = DumpLeaf<LeafOf<q3::MmaSyncLeaf>>::Of<Tile>;
template<class Tile> using PingPongDump = DumpLeaf<LeafOf<q3::MmaPingPongLeaf>>::Of<Tile>;
#define PROBE_LEAF(NAME, TILE, LEAF, AS, BS) \
extern "C" __global__ __launch_bounds__(128) void NAME( \
        const unsigned short* input, const unsigned char* weights, unsigned short* output, \
        unsigned int rows, unsigned int in_features, unsigned int out_features, unsigned long long scale_offset) { \
    using Tile = TILE; \
    __shared__ q3::PrefillShared<Tile, AS> staging; \
    __shared__ __align__(32) __nv_bfloat16 b_hi[Tile::kCols * BS]; \
    __shared__ __align__(32) __nv_bfloat16 b_lo[Tile::kCols * BS]; \
    q3::tiled_prefill<Tile, AS, BS, LEAF<Tile>>(input, weights, output, rows, in_features, out_features, \
            scale_offset, staging, b_hi, b_lo); \
}
PROBE_LEAF(probe_wmma_32, q3::Prefill32, WmmaDump, 80, 80)
PROBE_LEAF(probe_mma_32, q3::Prefill32, MmaDump, 80, 80)
PROBE_LEAF(probe_pingpong_32, q3::Prefill32, PingPongDump, 80, 80)
PROBE_LEAF(probe_wmma_64, q3::Prefill64, WmmaDump, 80, 64)
PROBE_LEAF(probe_mma_64, q3::Prefill64, MmaDump, 80, 64)
PROBE_LEAF(probe_pingpong_64, q3::Prefill64, PingPongDump, 80, 64)
PROBE_LEAF(probe_wmma_64x16, q3::WarpTile<4 COMMA 1 COMMA 1>, WmmaDump, 64, 64)
PROBE_LEAF(probe_mma_64x16, q3::WarpTile<4 COMMA 1 COMMA 1>, MmaDump, 64, 64)
PROBE_LEAF(probe_pingpong_64x16, q3::WarpTile<4 COMMA 1 COMMA 1>, PingPongDump, 64, 64)
PROBE_LEAF(probe_wmma_16x64, q3::WarpTile<1 COMMA 4 COMMA 1>, WmmaDump, 64, 64)
PROBE_LEAF(probe_mma_16x64, q3::WarpTile<1 COMMA 4 COMMA 1>, MmaDump, 64, 64)
PROBE_LEAF(probe_pingpong_16x64, q3::WarpTile<1 COMMA 4 COMMA 1>, PingPongDump, 64, 64)
// Independent sequential-staging control: the merged K schedule, with the
// same tile geometry, strides, union transitions, and WMMA accumulation order.
template<class Tile, int AS, int BS>
__device__ void probe_sequential_prefill(
        const unsigned short* input, const unsigned char* weights, unsigned short* output,
        unsigned int rows, unsigned int in_features, unsigned int out_features,
        unsigned long long scale_offset, q3::PrefillShared<Tile, AS>& staging,
        __nv_bfloat16* b_hi, __nv_bfloat16* b_lo) {
    constexpr int threads = Tile::kWarps * 32;
    unsigned int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    unsigned int output_tiles = (out_features + Tile::kCols - 1u) / Tile::kCols;
    unsigned int row_start = (blockIdx.x / output_tiles) * Tile::kRows;
    unsigned int out_start = (blockIdx.x % output_tiles) * Tile::kCols;
    q3::Layout w(weights, in_features, scale_offset);
    q3::Accumulators<Tile::kFrags> acc;
    acc.fill();
    using Shared = q3::PrefillShared<Tile, AS>;
    if (threadIdx.x == 0) new (static_cast<void*>(&staging)) Shared(typename Shared::ActivateA{});
    __syncthreads();
    __nv_bfloat16* a = staging.a.values;
    for (unsigned int base = 0; base < in_features; base += q3::kGroup) {
        q3::stage_activation_tile<Tile::kRows, q3::kGroup, threads, AS>(
                a, input, rows, in_features, row_start, base, threadIdx.x);
        q3::stage_weight_tile<Tile::kCols, Tile::kWarps, BS>(
                b_hi, b_lo, w, out_start, out_features, base, warp, lane);
        __syncthreads();
        q3::consume_mma_tile<Tile, q3::kGroup, AS, BS>(acc, a, b_hi, b_lo, warp);
        __syncthreads();
    }
    if (threadIdx.x == 0) new (static_cast<void*>(&staging)) Shared(typename Shared::ActivateResult{});
    __syncthreads();
    float* result = staging.result.values;
    q3::store_accumulators<Tile>(result, acc, warp);
    __syncthreads();
    q3::write_output_tile<Tile::kRows, Tile::kCols, threads>(
            output, result, rows, out_features, row_start, out_start, threadIdx.x);
}
#define PROBE_SEQUENTIAL(NAME, TILE, AS, BS) \
extern "C" __global__ __launch_bounds__(128) void NAME( \
        const unsigned short* input, const unsigned char* weights, unsigned short* output, \
        unsigned int rows, unsigned int in_features, unsigned int out_features, unsigned long long scale_offset) { \
    using Tile = TILE; \
    __shared__ q3::PrefillShared<Tile, AS> staging; \
    __shared__ __align__(32) __nv_bfloat16 b_hi[Tile::kCols * BS]; \
    __shared__ __align__(32) __nv_bfloat16 b_lo[Tile::kCols * BS]; \
    probe_sequential_prefill<Tile, AS, BS>(input, weights, output, rows, in_features, out_features, \
            scale_offset, staging, b_hi, b_lo); \
}
PROBE_SEQUENTIAL(probe_sequential_32, q3::Prefill32, 80, 80)
PROBE_SEQUENTIAL(probe_sequential_64, q3::Prefill64, 80, 64)
"""


def _load_libraries():
    if not (RUNTIME / "libnvrtc.so.13").is_file():
        return None, None, "pinned NVRTC runtime not built"
    try:
        # NVRTC resolves its builtins library through the dynamic loader.
        C.CDLL(str(RUNTIME / "libnvrtc-builtins.so.13.1"), mode=C.RTLD_GLOBAL)
        nvrtc = C.CDLL(str(RUNTIME / "libnvrtc.so.13"))
        cuda = C.CDLL("libcuda.so.1")
    except OSError as error:
        return None, None, str(error)
    return nvrtc, cuda, None


NVRTC, CUDA, SKIP_REASON = _load_libraries()


def _bind(lib, name, args):
    fn = getattr(lib, name)
    fn.argtypes, fn.restype = args, I
    return fn


def _check(status, what):
    if status:
        raise RuntimeError(f"{what} failed with status {status}")


class Gpu:
    """Owns one primary-context retain and one module for the test class."""

    def __init__(self, source, include_dir=None, cpp_std=14, architecture="compute_90"):
        nv, cu = NVRTC, CUDA
        self.create = _bind(nv, "nvrtcCreateProgram", [C.POINTER(P), C.c_char_p, C.c_char_p, I, P, P])
        self.compile = _bind(nv, "nvrtcCompileProgram", [P, I, C.POINTER(C.c_char_p)])
        self.log_size = _bind(nv, "nvrtcGetProgramLogSize", [P, C.POINTER(C.c_size_t)])
        self.get_log = _bind(nv, "nvrtcGetProgramLog", [P, P])
        self.ptx_size = _bind(nv, "nvrtcGetPTXSize", [P, C.POINTER(C.c_size_t)])
        self.get_ptx = _bind(nv, "nvrtcGetPTX", [P, P])
        self.cubin_size = _bind(nv, "nvrtcGetCUBINSize", [P, C.POINTER(C.c_size_t)])
        self.get_cubin = _bind(nv, "nvrtcGetCUBIN", [P, P])
        self.destroy = _bind(nv, "nvrtcDestroyProgram", [C.POINTER(P)])
        self.retain = _bind(cu, "cuDevicePrimaryCtxRetain", [C.POINTER(P), I])
        self.release = _bind(cu, "cuDevicePrimaryCtxRelease_v2", [I])
        self.set_current = _bind(cu, "cuCtxSetCurrent", [P])
        self.load = _bind(cu, "cuModuleLoadDataEx", [C.POINTER(P), P, C.c_uint, P, P])
        self.unload = _bind(cu, "cuModuleUnload", [P])
        self.function = _bind(cu, "cuModuleGetFunction", [C.POINTER(P), P, C.c_char_p])
        self.alloc = _bind(cu, "cuMemAlloc_v2", [C.POINTER(C.c_uint64), C.c_size_t])
        self.free = _bind(cu, "cuMemFree_v2", [C.c_uint64])
        self.htod = _bind(cu, "cuMemcpyHtoD_v2", [C.c_uint64, P, C.c_size_t])
        self.dtoh = _bind(cu, "cuMemcpyDtoH_v2", [P, C.c_uint64, C.c_size_t])
        self.memset = _bind(cu, "cuMemsetD8_v2", [C.c_uint64, C.c_ubyte, C.c_size_t])
        self.launch_kernel = _bind(cu, "cuLaunchKernel", [P, C.c_uint, C.c_uint, C.c_uint, C.c_uint, C.c_uint,
                                                          C.c_uint, C.c_uint, P, C.POINTER(P), P])
        self.sync = _bind(cu, "cuCtxSynchronize", [])
        self.set_attribute = _bind(cu, "cuFuncSetAttribute", [P, I, I])
        self.architecture = architecture
        count = I()
        if _bind(cu, "cuInit", [C.c_uint])(0) or _bind(cu, "cuDeviceGetCount", [C.POINTER(I)])(C.byref(count)) \
                or count.value == 0:
            raise unittest.SkipTest("no usable CUDA device")
        self.context = P()
        _check(self.retain(C.byref(self.context), 0), "cuDevicePrimaryCtxRetain")
        self.module = P()
        try:
            _check(self.set_current(self.context), "cuCtxSetCurrent")
            _check(self.load(C.byref(self.module), self._ptx(source, include_dir, cpp_std), 0, None, None), "cuModuleLoadDataEx")
        except BaseException:
            self.close()
            raise

    def _ptx(self, source, include_dir, cpp_std):
        program = P()
        _check(self.create(C.byref(program), source, b"q3_linear_bf16_probe.cu", 0, None, None), "nvrtcCreate")
        try:
            options = [f"--std=c++{cpp_std}".encode(), f"--gpu-architecture={self.architecture}".encode(), b"-I" + str(INCLUDE).encode(),
                       b"-DCOMMA=,", b"-I" + str(include_dir or (ROOT / "native/src")).encode(),
                       b"-I" + str(INCLUDE / "cccl").encode()]
            status = self.compile(program, len(options), (C.c_char_p * len(options))(*options))
            if status:
                size = C.c_size_t()
                self.log_size(program, C.byref(size))
                log = C.create_string_buffer(size.value)
                self.get_log(program, log)
                raise RuntimeError(log.value.decode())
            size = C.c_size_t()
            # A real architecture (sm_XY[a]) yields a cubin, a virtual one PTX for the driver to JIT.
            real = self.architecture.startswith("sm_")
            _check((self.cubin_size if real else self.ptx_size)(program, C.byref(size)), "nvrtcGet code size")
            ptx = C.create_string_buffer(size.value)
            _check((self.get_cubin if real else self.get_ptx)(program, ptx), "nvrtcGet code")
            return ptx
        finally:
            self.destroy(C.byref(program))

    def close(self):
        try:
            _check(self.set_current(self.context), "cuCtxSetCurrent for cleanup")
            if self.module.value:
                _check(self.unload(self.module), "cuModuleUnload")
                self.module = P()
        finally:
            try:
                _check(self.set_current(None), "cuCtxSetCurrent clear")
            finally:
                _check(self.release(0), "cuDevicePrimaryCtxRelease")

    def upload(self, data):
        ptr = self._allocate(len(data))
        try:
            if data:
                _check(self.htod(ptr, C.create_string_buffer(bytes(data), len(data)), len(data)), "cuMemcpyHtoD")
        except BaseException:
            self.free(ptr)
            raise
        return ptr

    def zeros(self, size, fill=0):
        ptr = self._allocate(size)
        try:
            _check(self.memset(ptr, fill, size), "cuMemset")
        except BaseException:
            self.free(ptr)
            raise
        return ptr

    def _allocate(self, size):
        out = C.c_uint64()
        _check(self.alloc(C.byref(out), max(size, 1)), "cuMemAlloc")
        return out.value

    def download(self, ptr, size):
        buffer = C.create_string_buffer(size)
        _check(self.dtoh(buffer, ptr, size), "cuMemcpyDtoH")
        return buffer.raw

    def launch(self, name, grid, arguments, synchronize=True, block=None, shared=0):
        function = P()
        _check(self.function(C.byref(function), self.module, name.encode()), name)
        if shared > 48 * 1024:  # CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
            _check(self.set_attribute(function, 8, shared), name + " shared size")
        params = (P * len(arguments))(*[C.cast(C.pointer(value), P) for value in arguments])
        grid_x, grid_y = grid if isinstance(grid, tuple) else (grid, 1)
        if block is None:
            block = 128 if name != "probe_stripe_codes" and name != "probe_pair_codes" else 32
        _check(self.launch_kernel(function, grid_x, grid_y, 1, block, 1, 1, shared, None, params, None), name)
        if synchronize:
            _check(self.sync(), name)


def reference_code(data, group, index):
    bit = index * 3
    offset = group * 24 + (bit >> 3)
    word = data[offset] | (data[offset + 1] << 8 if offset + 1 < len(data) else 0)
    code = (word >> (bit & 7)) & 7
    return code - 8 if code >= 4 else code


def fp16(bits):
    return struct.unpack("<e", struct.pack("<H", bits))[0]


def bf16_value(bits):
    return struct.unpack("<f", struct.pack("<I", bits << 16))[0]


def to_bf16(value):
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


@unittest.skipIf(NVRTC is None, f"CUDA probes unavailable: {SKIP_REASON}")
class Q3PrimitiveTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gpu = Gpu(b'#include "q3/kernels.cu"\n' + PROBES.encode())
        cls.rng = random.Random(0x0513)

    @classmethod
    def tearDownClass(cls):
        cls.gpu.close()

    def owned(self, stack, pointer):
        """Register a device allocation for release as soon as it exists."""
        stack.callback(self.gpu.free, pointer)
        return pointer

    def run_codes(self, kernel, data, groups):
        gpu = self.gpu
        with contextlib.ExitStack() as stack:
            source = self.owned(stack, gpu.upload(data))
            out = self.owned(stack, gpu.zeros(groups * 64 * 4))
            gpu.launch(kernel, 1, [C.c_uint64(source), C.c_uint64(out), C.c_uint(groups)])
            return list(struct.unpack(f"<{groups * 64}i", gpu.download(out, groups * 64 * 4)))

    def test_packed_load_transfers_every_code_to_its_documented_owner(self):
        # Arbitrary bytes are valid packed codes; include every 3-bit pattern at every
        # bit phase, plus a trailing pad so 32-bit warp loads stay in bounds.
        groups = 16
        data = bytes(self.rng.randrange(256) for _ in range(groups * 24)) + bytes(8)
        expected = [reference_code(data, g, i) for g in range(groups) for i in range(64)]
        self.assertEqual(self.run_codes("probe_reference_codes", data, groups), expected)
        self.assertEqual(self.run_codes("probe_pair_codes", data, groups), expected)
        self.assertEqual(self.run_codes("probe_stripe_codes", data, groups), expected)

    def test_scale_application_and_hi_lo_staging_are_exact(self):
        scales = [0x0000, 0x8000, 0x0001, 0x8001, 0x03FF, 0x0400, 0x3555, 0xB555, 0x3C00, 0x7BFF, 0xFBFF]
        scales += [self.rng.randrange(0x7C00) | (0x8000 if self.rng.random() < 0.5 else 0) for _ in range(64)]
        gpu = self.gpu
        count = len(scales)
        with contextlib.ExitStack() as stack:
            source = self.owned(stack, gpu.upload(struct.pack(f"<{count}H", *scales)))
            buffers = [self.owned(stack, gpu.zeros(count * 8 * 4)) for _ in range(3)]
            gpu.launch("probe_split_weights", 1, [C.c_uint64(source)] + [C.c_uint64(b) for b in buffers]
                       + [C.c_uint(count)])
            exact, hi, lo = [struct.unpack(f"<{count * 8}f", gpu.download(b, count * 8 * 4)) for b in buffers]
        for s, bits in enumerate(scales):
            for code in range(-4, 4):
                i = s * 8 + code + 4
                with self.subTest(scale=hex(bits), code=code):
                    self.assertEqual(exact[i], f32(code * fp16(bits)))
                    self.assertEqual(f32(hi[i] + lo[i]), exact[i])
                    self.assertEqual(hi[i] + lo[i], exact[i], "hi + lo must be exact without rounding")

    def test_pair_staging_matches_per_weight_split_bitwise(self):
        # Every 6-bit code pair, including zero, subnormal, max, infinite, and NaN scales.
        scales = [0x0000, 0x8000, 0x0001, 0x8001, 0x03FF, 0x0400, 0x3555, 0xB555, 0x3C00,
                  0x7BFF, 0xFBFF, 0x7C00, 0xFC00, 0x7E11, 0xFE11, 0x7C01]
        scales += [self.rng.randrange(0x10000) for _ in range(64)]
        gpu = self.gpu
        count, size = len(scales), len(scales) * 256 * 2
        with contextlib.ExitStack() as stack:
            source = self.owned(stack, gpu.upload(struct.pack(f"<{count}H", *scales)))
            actual = self.owned(stack, gpu.zeros(size, fill=0xA5))
            expected = self.owned(stack, gpu.zeros(size, fill=0x5A))
            gpu.launch("probe_split_pairs", 1, [C.c_uint64(source), C.c_uint64(actual),
                                                C.c_uint64(expected), C.c_uint(count)])
            got, want = gpu.download(actual, size), gpu.download(expected, size)
        self.assertEqual(got, want)

    def test_k128_decode_matches_per_group_decode_bitwise(self):
        gpu = self.gpu
        special = [0x3f80, 0xbf00, 0x7fc1, 0xffc3, 0x7f80, 0xff80, 0x0001, 0x8000]
        scales = [0x3555, 0xb555, 0x0001, 0x8000, 0x7bff, 0x7c00, 0x7e11, 0xfe11]
        for rows, width, outputs in [(1, 1, 1), (2, 65, 9), (4, 128, 8), (3, 200, 13),
                                     (4, 320, 37), (1, 5120, 16), (2, 1000, 7)]:
            groups = ((width + 127) // 128) * 2
            offset = (outputs * groups * 24 + 255) & ~255
            payload = bytearray(self.rng.randrange(256) for _ in range(offset + outputs * groups * 2))
            for i in range(outputs * groups):
                bits = (scales[i % len(scales)] if width < 320 else
                        self.rng.randrange(0x2c00, 0x3400) | (0x8000 if i & 1 else 0))
                struct.pack_into('<H', payload, offset + i * 2, bits)
            values = ([special[i % len(special)] for i in range(rows * width)] if width < 320 else
                      [to_bf16(self.rng.uniform(-2, 2)) for _ in range(rows * width)])
            with contextlib.ExitStack() as stack:
                x = self.owned(stack, gpu.upload(struct.pack(f'<{len(values)}H', *values)))
                w = self.owned(stack, gpu.upload(bytes(payload)))
                for tile in (1, 2, 4):
                    grid = ((rows + tile - 1) // tile) * ((outputs + 7) // 8)
                    got = {}
                    for kernel in (f'euhedral_q3_decode_{tile}', f'probe_group_decode_{tile}'):
                        y = self.owned(stack, gpu.zeros(rows * outputs * 2, fill=0xa5))
                        gpu.launch(kernel, grid, [C.c_uint64(x), C.c_uint64(w), C.c_uint64(y), C.c_uint(rows),
                                                  C.c_uint(width), C.c_uint(outputs), C.c_ulonglong(offset)])
                        got[kernel] = gpu.download(y, rows * outputs * 2)
                    with self.subTest(rows=rows, width=width, outputs=outputs, tile=tile):
                        actual = got[f'euhedral_q3_decode_{tile}']
                        self.assertNotIn(b'\xa5\xa5', [actual[i:i + 2] for i in range(0, len(actual), 2)])
                        self.assertEqual(actual, got[f'probe_group_decode_{tile}'])

    def test_explicit_mma_leaves_match_wmma_bitwise(self):
        # Every explicit leaf on every geometry against the WMMA leaf, comparing the
        # FP32 accumulators before BF16 conversion, plus the production kernels'
        # BF16 output against the WMMA reference.
        gpu = self.gpu
        get_global = _bind(CUDA, "cuModuleGetGlobal_v2", [C.POINTER(C.c_uint64), C.POINTER(C.c_size_t), P, C.c_char_p])
        dump, dump_size = C.c_uint64(), C.c_size_t()
        _check(get_global(C.byref(dump), C.byref(dump_size), gpu.module, b"probe_accumulators"), "probe_accumulators")
        special = [0x3f80, 0xbf00, 0x7fc1, 0xffc3, 0x7f80, 0xff80, 0x0001, 0x8000]
        scales = [0x3555, 0xb555, 0x0001, 0x8000, 0x7bff, 0x7c00, 0x7e11, 0xfe11]
        cases = [(1, 1, 1), (33, 65, 35), (65, 100, 9), (97, 192, 37), (64, 320, 64), (17, 1000, 70)]
        geometries = [('32', 32, 32), ('64', 64, 32), ('64x16', 64, 16), ('16x64', 16, 64)]
        for rows, width, outputs in cases:
            groups = ((width + 127) // 128) * 2
            offset = (outputs * groups * 24 + 255) & ~255
            payload = bytearray(self.rng.randrange(256) for _ in range(offset + outputs * groups * 2))
            nonfinite = width in (1, 65)
            for i in range(outputs * groups):
                bits = scales[i % len(scales)] if nonfinite else self.rng.randrange(0x2c00, 0x3400) | (i & 1) << 15
                struct.pack_into('<H', payload, offset + i * 2, bits)
            values = ([special[i % len(special)] for i in range(rows * width)] if nonfinite else
                      [to_bf16(self.rng.uniform(-2, 2)) for _ in range(rows * width)])
            with contextlib.ExitStack() as stack:
                x = self.owned(stack, gpu.upload(struct.pack(f'<{len(values)}H', *values)))
                w = self.owned(stack, gpu.upload(bytes(payload)))

                def run(name, grid, tile_elements):
                    y = self.owned(stack, gpu.zeros(rows * outputs * 2, fill=0xa5))
                    gpu.memset(dump, 0xa5, dump_size.value)
                    gpu.launch(name, grid, [C.c_uint64(x), C.c_uint64(w), C.c_uint64(y), C.c_uint(rows),
                                            C.c_uint(width), C.c_uint(outputs), C.c_ulonglong(offset)])
                    return gpu.download(y, rows * outputs * 2), gpu.download(dump, grid * tile_elements * 4)

                for suffix, tile_rows, tile_cols in geometries:
                    grid = ((rows + tile_rows - 1) // tile_rows) * ((outputs + tile_cols - 1) // tile_cols)
                    self.assertLessEqual(grid * tile_rows * tile_cols * 4, dump_size.value)
                    reference = run(f'probe_wmma_{suffix}', grid, tile_rows * tile_cols)
                    self.assertNotIn(b'\xa5\xa5\xa5\xa5', [reference[1][i:i + 4] for i in range(0, len(reference[1]), 4)])
                    for leaf in ('mma', 'pingpong'):
                        with self.subTest(rows=rows, width=width, outputs=outputs, geometry=suffix, leaf=leaf):
                            output, accumulators = run(f'probe_{leaf}_{suffix}', grid, tile_rows * tile_cols)
                            self.assertEqual(accumulators, reference[1])
                            self.assertEqual(output, reference[0])
                    kernels = {'32': ['euhedral_q3_prefill_exact'], '64': ['euhedral_q3_prefill_64_exact', 'euhedral_q3_prefill_64_wmma']}
                    for kernel in kernels.get(suffix, []):
                        with self.subTest(rows=rows, width=width, outputs=outputs, kernel=kernel):
                            y = self.owned(stack, gpu.zeros(rows * outputs * 2, fill=0xa5))
                            gpu.launch(kernel, grid, [C.c_uint64(x), C.c_uint64(w), C.c_uint64(y), C.c_uint(rows),
                                                      C.c_uint(width), C.c_uint(outputs), C.c_ulonglong(offset)])
                            self.assertEqual(gpu.download(y, rows * outputs * 2), reference[0])

    def test_activation_staging_zero_fills_outside_rows_and_k(self):
        gpu = self.gpu
        rows, width = 45, 100
        values = [to_bf16(self.rng.uniform(-4, 4)) for _ in range(rows * width)]
        with contextlib.ExitStack() as stack:
            source = self.owned(stack, gpu.upload(struct.pack(f"<{len(values)}H", *values)))
            out = self.owned(stack, gpu.zeros(32 * 64 * 4))
            for row_start, k_base in [(0, 0), (32, 64), (32, 0), (0, 64)]:
                gpu.launch("probe_activation_tile", 1, [C.c_uint64(source), C.c_uint64(out), C.c_uint(rows),
                                                         C.c_uint(width), C.c_uint(row_start), C.c_uint(k_base)])
                tile = struct.unpack("<2048f", gpu.download(out, 32 * 64 * 4))
                for i, value in enumerate(tile):
                    r, k = row_start + i // 64, k_base + i % 64
                    want = bf16_value(values[r * width + k]) if r < rows and k < width else 0.0
                    self.assertEqual(value, want, (row_start, k_base, r, k))

    def test_output_writeback_touches_only_in_matrix_elements(self):
        gpu = self.gpu
        rows, outputs = 45, 50
        for row_start, out_start in [(0, 0), (32, 32), (32, 0), (0, 32)]:
            with contextlib.ExitStack() as stack:
                out = self.owned(stack, gpu.zeros(rows * outputs * 2, fill=0xA5))
                gpu.launch("probe_output_tile", 1, [C.c_uint64(out), C.c_uint(rows), C.c_uint(outputs),
                                                     C.c_uint(row_start), C.c_uint(out_start)])
                written = struct.unpack(f"<{rows * outputs}H", gpu.download(out, rows * outputs * 2))
            for r in range(rows):
                for n in range(outputs):
                    inside = row_start <= r < row_start + 32 and out_start <= n < out_start + 32
                    want = to_bf16(float((r - row_start) * 32 + (n - out_start) + 1)) if inside else 0xA5A5
                    self.assertEqual(written[r * outputs + n], want, (row_start, out_start, r, n))

    def test_recomposed_tile_geometries_match_the_production_prefill(self):
        gpu = self.gpu
        for rows, width, outputs in [(33, 192, 35), (65, 65, 70), (17, 320, 9)]:
            groups = ((width + 127) // 128) * 2
            scale_offset = (outputs * groups * 24 + 255) & ~255
            payload = bytearray(self.rng.randrange(256) for _ in range(scale_offset + outputs * groups * 2))
            for i in range(outputs * groups):
                struct.pack_into("<H", payload, scale_offset + 2 * i, self.rng.randrange(0x2C00, 0x3400))
            values = [to_bf16(self.rng.uniform(-2, 2)) for _ in range(rows * width)]
            with contextlib.ExitStack() as stack:
                x = self.owned(stack, gpu.upload(struct.pack(f"<{len(values)}H", *values)))
                w = self.owned(stack, gpu.upload(bytes(payload)))
                results = {}
                for kernel, tile_rows, tile_cols in [("euhedral_q3_prefill_exact", 32, 32),
                                                      ("euhedral_q3_prefill_64_exact", 64, 32),
                                                      ("probe_unpadded_32", 32, 32),
                                                      ("probe_unpadded_64", 64, 32),
                                                      ("probe_tiled_64x16", 64, 16),
                                                      ("probe_tiled_16x64", 16, 64)]:
                    with contextlib.ExitStack() as launch_stack:
                        y = self.owned(launch_stack, gpu.zeros(rows * outputs * 2, fill=0xA5))
                        grid = ((rows + tile_rows - 1) // tile_rows) * ((outputs + tile_cols - 1) // tile_cols)
                        gpu.launch(kernel, grid, [C.c_uint64(x), C.c_uint64(w), C.c_uint64(y), C.c_uint(rows),
                                                  C.c_uint(width), C.c_uint(outputs), C.c_ulonglong(scale_offset)])
                        results[kernel] = gpu.download(y, rows * outputs * 2)
                with self.subTest(rows=rows, width=width, outputs=outputs):
                    self.assertNotIn(b"\xa5\xa5", [results["euhedral_q3_prefill_exact"][i:i + 2]
                                                   for i in range(0, rows * outputs * 2, 2)])
                    self.assertEqual(results["euhedral_q3_prefill_64_exact"], results["euhedral_q3_prefill_exact"])
                    self.assertEqual(results["probe_unpadded_32"], results["euhedral_q3_prefill_exact"])
                    self.assertEqual(results["probe_unpadded_64"], results["euhedral_q3_prefill_exact"])
                    self.assertEqual(results["probe_tiled_64x16"], results["euhedral_q3_prefill_exact"])
                    self.assertEqual(results["probe_tiled_16x64"], results["euhedral_q3_prefill_exact"])

    def test_prefetched_k_matches_sequential_staging_bitwise_at_boundaries(self):
        gpu = self.gpu
        special = [0x3f80, 0xbf00, 0x7fc1, 0xffc3, 0x7f80, 0xff80, 0x0001, 0x8000]
        scales = [0x3555, 0xb555, 0x0001, 0x7c00, 0x7e11, 0xfe11]
        for rows, width, outputs in [(1, 1, 9), (33, 65, 35), (65, 100, 9),
                                     (97, 192, 37), (31, 128, 32)]:
            groups = ((width + 127) // 128) * 2
            offset = (outputs * groups * 24 + 255) & ~255
            payload = bytearray(self.rng.randrange(256) for _ in range(offset + outputs * groups * 2))
            for i in range(outputs * groups):
                struct.pack_into('<H', payload, offset + i * 2, scales[i % len(scales)])
            values = ([special[i % len(special)] for i in range(rows * width)]
                      if width in (1, 65) else
                      [to_bf16(self.rng.uniform(-2, 2)) for _ in range(rows * width)])
            with contextlib.ExitStack() as stack:
                x = self.owned(stack, gpu.upload(struct.pack(f'<{len(values)}H', *values)))
                w = self.owned(stack, gpu.upload(payload))
                for tile, suffix in [(32, '32'), (64, '64')]:
                    expected = self.owned(stack, gpu.zeros(rows * outputs * 2, fill=0xa5))
                    actual = self.owned(stack, gpu.zeros(rows * outputs * 2, fill=0xa5))
                    grid = ((rows + tile - 1) // tile) * ((outputs + 31) // 32)
                    args = lambda y: [C.c_uint64(x), C.c_uint64(w), C.c_uint64(y), C.c_uint(rows),
                                      C.c_uint(width), C.c_uint(outputs), C.c_ulonglong(offset)]
                    gpu.launch('probe_sequential_' + suffix, grid, args(expected))
                    kernel = 'euhedral_q3_prefill' + ('_64' if tile == 64 else '') + '_exact'
                    gpu.launch(kernel, grid, args(actual))
                    with self.subTest(rows=rows, width=width, outputs=outputs, tile=tile):
                        self.assertEqual(gpu.download(actual, rows * outputs * 2),
                                         gpu.download(expected, rows * outputs * 2))

    def test_compact_b_kernel_matches_tile64_across_k_generation_counts(self):
        gpu = self.gpu
        rows, outputs = 64, 32
        for width in (32, 64, 96, 128):
            groups = ((width + 127) // 128) * 2
            scale_offset = (outputs * groups * 24 + 255) & ~255
            payload = bytearray(self.rng.randrange(256)
                                for _ in range(scale_offset + outputs * groups * 2))
            for index in range(outputs * groups):
                struct.pack_into('<H', payload, scale_offset + index * 2, 0x3555)
            values = [to_bf16(self.rng.uniform(-2, 2)) for _ in range(rows * width)]
            with contextlib.ExitStack() as stack:
                source = self.owned(stack, gpu.upload(struct.pack(f'<{len(values)}H', *values)))
                weights = self.owned(stack, gpu.upload(payload))
                expected = self.owned(stack, gpu.zeros(rows * outputs * 2, fill=0))
                actual = self.owned(stack, gpu.zeros(rows * outputs * 2, fill=0))
                grid = (rows + 63) // 64
                args = lambda destination: [C.c_uint64(source), C.c_uint64(weights),
                        C.c_uint64(destination), C.c_uint(rows), C.c_uint(width),
                        C.c_uint(outputs), C.c_ulonglong(scale_offset)]
                gpu.launch('probe_sequential_64', grid, args(expected))
                gpu.launch('euhedral_q3_prefill_64_k32_cb_exact', grid, args(actual))
                with self.subTest(width=width, generations=width // 32):
                    self.assertEqual(gpu.download(actual, rows * outputs * 2),
                                     gpu.download(expected, rows * outputs * 2))

    def test_prefetch_handles_two_byte_aligned_input_base(self):
        gpu = self.gpu
        rows, width, outputs = 33, 128, 35
        groups = ((width + 127) // 128) * 2
        offset = (outputs * groups * 24 + 255) & ~255
        payload = bytearray(self.rng.randrange(256) for _ in range(offset + outputs * groups * 2))
        for i in range(outputs * groups):
            struct.pack_into('<H', payload, offset + i * 2, 0x3555)
        values = [to_bf16(self.rng.uniform(-2, 2)) for _ in range(rows * width)]
        with contextlib.ExitStack() as stack:
            # A valid BF16 pointer need not be aligned for an ld.global.u32.
            source = self.owned(stack, gpu.upload(b'\x00\x00' + struct.pack(f'<{len(values)}H', *values)))
            w = self.owned(stack, gpu.upload(payload))
            for tile, suffix in [(32, '32'), (64, '64')]:
                expected = self.owned(stack, gpu.zeros(rows * outputs * 2, fill=0xa5))
                actual = self.owned(stack, gpu.zeros(rows * outputs * 2, fill=0xa5))
                grid = ((rows + tile - 1) // tile) * ((outputs + 31) // 32)
                args = lambda y: [C.c_uint64(source + 2), C.c_uint64(w), C.c_uint64(y), C.c_uint(rows),
                                  C.c_uint(width), C.c_uint(outputs), C.c_ulonglong(offset)]
                gpu.launch('probe_sequential_' + suffix, grid, args(expected))
                kernel = 'euhedral_q3_prefill' + ('_64' if tile == 64 else '') + '_exact'
                gpu.launch(kernel, grid, args(actual))
                with self.subTest(tile=tile):
                    self.assertEqual(gpu.download(actual, rows * outputs * 2),
                                     gpu.download(expected, rows * outputs * 2))


if __name__ == "__main__":
    unittest.main()
