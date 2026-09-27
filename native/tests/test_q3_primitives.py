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
SOURCE = ROOT / "native" / "src" / "q3_linear_bf16.cu"
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
    __shared__ __align__(32) __nv_bfloat16 a[Tile::kRows * q3::kGroup]; \
    __shared__ __align__(32) __nv_bfloat16 b_hi[Tile::kCols * q3::kGroup]; \
    __shared__ __align__(32) __nv_bfloat16 b_lo[Tile::kCols * q3::kGroup]; \
    __shared__ __align__(32) float result[Tile::kRows * Tile::kCols]; \
    q3::tiled_prefill<Tile>(input, weights, output, rows, in_features, out_features, scale_offset, \
            a, b_hi, b_lo, result); \
}
PROBE_TILED(probe_tiled_64x16, q3::WarpTile<4 COMMA 1 COMMA 1>)
PROBE_TILED(probe_tiled_16x64, q3::WarpTile<1 COMMA 4 COMMA 1>)
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

    def __init__(self, source, include_dir=None, cpp_std=14):
        nv, cu = NVRTC, CUDA
        self.create = _bind(nv, "nvrtcCreateProgram", [C.POINTER(P), C.c_char_p, C.c_char_p, I, P, P])
        self.compile = _bind(nv, "nvrtcCompileProgram", [P, I, C.POINTER(C.c_char_p)])
        self.log_size = _bind(nv, "nvrtcGetProgramLogSize", [P, C.POINTER(C.c_size_t)])
        self.get_log = _bind(nv, "nvrtcGetProgramLog", [P, P])
        self.ptx_size = _bind(nv, "nvrtcGetPTXSize", [P, C.POINTER(C.c_size_t)])
        self.get_ptx = _bind(nv, "nvrtcGetPTX", [P, P])
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
            options = [f"--std=c++{cpp_std}".encode(), b"--gpu-architecture=compute_90", b"-I" + str(INCLUDE).encode(),
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
            _check(self.ptx_size(program, C.byref(size)), "nvrtcGetPTXSize")
            ptx = C.create_string_buffer(size.value)
            _check(self.get_ptx(program, ptx), "nvrtcGetPTX")
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

    def launch(self, name, grid, arguments, synchronize=True):
        function = P()
        _check(self.function(C.byref(function), self.module, name.encode()), name)
        params = (P * len(arguments))(*[C.cast(C.pointer(value), P) for value in arguments])
        grid_x, grid_y = grid if isinstance(grid, tuple) else (grid, 1)
        _check(self.launch_kernel(function, grid_x, grid_y, 1, 128 if name != "probe_stripe_codes"
                                  and name != "probe_pair_codes" else 32, 1, 1, 0, None, params, None), name)
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
                for kernel, tile_rows, tile_cols in [("euhedral_q3_prefill", 32, 32),
                                                      ("euhedral_q3_prefill_64", 64, 32),
                                                      ("probe_tiled_64x16", 64, 16),
                                                      ("probe_tiled_16x64", 16, 64)]:
                    with contextlib.ExitStack() as launch_stack:
                        y = self.owned(launch_stack, gpu.zeros(rows * outputs * 2, fill=0xA5))
                        grid = ((rows + tile_rows - 1) // tile_rows) * ((outputs + tile_cols - 1) // tile_cols)
                        gpu.launch(kernel, grid, [C.c_uint64(x), C.c_uint64(w), C.c_uint64(y), C.c_uint(rows),
                                                  C.c_uint(width), C.c_uint(outputs), C.c_ulonglong(scale_offset)])
                        results[kernel] = gpu.download(y, rows * outputs * 2)
                with self.subTest(rows=rows, width=width, outputs=outputs):
                    self.assertNotIn(b"\xa5\xa5", [results["euhedral_q3_prefill"][i:i + 2]
                                                   for i in range(0, rows * outputs * 2, 2)])
                    self.assertEqual(results["euhedral_q3_prefill_64"], results["euhedral_q3_prefill"])
                    self.assertEqual(results["probe_tiled_64x16"], results["euhedral_q3_prefill"])
                    self.assertEqual(results["probe_tiled_16x64"], results["euhedral_q3_prefill"])


if __name__ == "__main__":
    unittest.main()
