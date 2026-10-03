"""Shared harness of the native kernel tests: compiles a source with the pinned NVRTC runtime, loads it on the
CUDA device and launches its kernels. Tests skip when the runtime or a device is unavailable."""

import contextlib
import ctypes as C
import pathlib
import random
import struct
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "build" / "cuda-dev" / "linux-x64" / "runtime"
INCLUDE = ROOT / "build" / "cuda-dev" / "linux-x64" / "include"
P, I = C.c_void_p, C.c_int

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
