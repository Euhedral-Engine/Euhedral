"""P2E2 Q3 tensors: the compressed routes must reproduce the row-split routes bit for bit.

The production q3 module and the embedding module are compiled with NVRTC. Tensors are built from
code matrices with tools/convert_compact_edrl_to_p2e2.py, the reference encoder. Each case covers
the paths a kernel takes on its own data: lanes with more than 16 BIG codes, slices whose payload
outruns the 32 prefetched words, rows without BIG codes, and realistic code distributions.
"""

import contextlib
import ctypes as C
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

try:
    import numpy as np
except ImportError:
    np = None

from test_q3_primitives import Gpu, NVRTC, SKIP_REASON

ROOT = Path(__file__).resolve().parents[2]
SENTINEL = 0xA5
UNAVAILABLE = None
if NVRTC is None:
    UNAVAILABLE = f"CUDA probes unavailable: {SKIP_REASON}"
elif np is None:
    UNAVAILABLE = "NumPy unavailable"
p2e2 = None
if np is not None:
    # The reference encoder (tools/convert_compact_edrl_to_p2e2.py) needs NumPy.
    spec = importlib.util.spec_from_file_location("p2e2_converter", ROOT / "tools/convert_compact_edrl_to_p2e2.py")
    p2e2 = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = p2e2
    spec.loader.exec_module(p2e2)

HOST_SOURCE = r'''#include <stdint.h>
#include <cuda.h>
#include "cuda_kernel_loader.h"
#include "euhedral_cuda.h"
static CUmodule injected_module;
static int exact;
void p2e2_test_set_module(uintptr_t handle) { injected_module = (CUmodule)handle; }
void p2e2_test_set_exact(int value) { exact = value; }
int euhedral_cuda_bind_thread_context(void) { return EUHEDRAL_CUDA_SUCCESS; }
int euhedral_cuda_load_kernel(const void* anchor, const char* source_name,
        const char* function_name, CUmodule* module, CUfunction* function) {
    (void)anchor; (void)source_name;
    *module = injected_module;
    return cuModuleGetFunction(function, injected_module, function_name) == CUDA_SUCCESS
            ? EUHEDRAL_CUDA_SUCCESS : EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
}
void* euhedral_cuda_submission_stream(void) { return NULL; }
void euhedral_cuda_pdl_register(CUfunction function) { (void)function; }
int euhedral_cuda_exact_numerics(void) { return exact; }
int euhedral_cuda_select_exact_numerics(int value) { int previous = exact; exact = value; return previous; }
CUresult euhedral_launch_kernel(CUfunction function, unsigned int gx, unsigned int gy, unsigned int gz,
        unsigned int bx, unsigned int by, unsigned int bz, unsigned int shared, CUstream stream,
        void** parameters, void** extra) {
    return cuLaunchKernel(function, gx, gy, gz, bx, by, bz, shared, stream, parameters, extra);
}
#include "q3_linear_bf16.c"
'''


def realistic(rng, rows, k):
    return rng.choice(np.arange(-3, 4, dtype=np.int8), size=(rows, k),
                      p=[0.02, 0.08, 0.23, 0.34, 0.23, 0.08, 0.02])


def row_split(codes, rng):
    rows, k = codes.shape
    groups = k // 64
    out = bytearray(p2e2.row_split_q3_size(rows, k))
    planes = p2e2.pack_q3(codes)
    out[:len(planes)] = planes
    scale_offset = p2e2.align_up(rows * groups * 24, 256)
    # Finite FP16 scales of both signs across normal and subnormal exponents.
    scales = rng.integers(0, 0x7C00, rows * groups, dtype=np.uint16) | (rng.integers(0, 2, rows * groups, dtype=np.uint16) << 15)
    out[scale_offset:scale_offset + rows * groups * 2] = scales.astype("<u2").tobytes()
    return bytes(out), scale_offset


def adversarial(rng, rows, k):
    """Rows that drive every decode path: dense BIG lanes, overflowing slices, empty rows."""
    codes = realistic(rng, rows, k)
    codes[0] = 0
    codes[1] = rng.choice(np.array([-3, -2, 2, 3], np.int8), size=k)
    if rows > 2:
        lane_dense = np.where(np.arange(k) % 32 < 17, 2, -1).astype(np.int8)
        codes[2] = lane_dense
    if rows > 3:
        codes[3, : k // 2] = 3
    return codes


@unittest.skipIf(UNAVAILABLE is not None, UNAVAILABLE or "")
class P2e2KernelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gpu = Gpu(b'#include "q3/kernels.cu"\n')
        cls.embedding = Gpu((ROOT / "native/src/embedding/kernels.cu").read_bytes())
        cls.rng = np.random.default_rng(0x9E2E2)

    @classmethod
    def tearDownClass(cls):
        cls.embedding.close()
        cls.gpu.close()

    def owned(self, stack, gpu, pointer):
        stack.callback(gpu.free, pointer)
        return pointer

    def decode_pair(self, codes):
        rows, k = codes.shape
        source, scale_offset = row_split(codes, self.rng)
        tensor = p2e2.encode(source, rows, k)
        x = (self.rng.standard_normal(k).astype(np.float32) * 3.0)
        xb = (x.view(np.uint32) >> 16).astype(np.uint16).tobytes()
        with contextlib.ExitStack() as stack:
            dx = self.owned(stack, self.gpu, self.gpu.upload(xb))
            dw = self.owned(stack, self.gpu, self.gpu.upload(source))
            dc = self.owned(stack, self.gpu, self.gpu.upload(tensor))
            y0 = self.owned(stack, self.gpu, self.gpu.zeros(rows * 2, fill=SENTINEL))
            y1 = self.owned(stack, self.gpu, self.gpu.zeros(rows * 2, fill=SENTINEL))
            self.gpu.launch("euhedral_q3_decode_contiguous", rows // 16,
                            [C.c_uint64(dx), C.c_uint64(dw), C.c_uint64(y0), C.c_uint(1), C.c_uint(k),
                             C.c_uint(rows), C.c_uint64(scale_offset)])
            self.gpu.launch("euhedral_q3_p2e2_decode", rows // 16,
                            [C.c_uint64(dx), C.c_uint64(dc), C.c_uint64(y1), C.c_uint(k), C.c_uint(rows)])
            return self.gpu.download(y0, rows * 2), self.gpu.download(y1, rows * 2)

    def test_decode_matches_contiguous_bitwise(self):
        for rows, k in ((16, 1024), (48, 3072), (32, 5120)):
            for name, codes in (("realistic", realistic(self.rng, rows, k)), ("adversarial", adversarial(self.rng, rows, k))):
                with self.subTest(rows=rows, k=k, codes=name):
                    expected, actual = self.decode_pair(codes)
                    self.assertNotIn(b"\xa5\xa5", expected[:2])
                    self.assertEqual(actual, expected)

    def test_expand_reproduces_the_row_split_tensor(self):
        for rows, k, name in ((5, 1024, "realistic"), (37, 2048, "adversarial"), (8, 6144, "realistic")):
            with self.subTest(rows=rows, k=k, codes=name):
                codes = realistic(self.rng, rows, k) if name == "realistic" else adversarial(self.rng, rows, k)
                source, _ = row_split(codes, self.rng)
                tensor = p2e2.encode(source, rows, k)
                with contextlib.ExitStack() as stack:
                    dc = self.owned(stack, self.gpu, self.gpu.upload(tensor))
                    out = self.owned(stack, self.gpu, self.gpu.zeros(len(source), fill=0))
                    self.gpu.launch("euhedral_q3_p2e2_expand", (rows + 3) // 4,
                                    [C.c_uint64(dc), C.c_uint64(out), C.c_uint(rows), C.c_uint(k), C.c_uint(0),
                                     C.c_uint(rows)])
                    self.assertEqual(self.gpu.download(out, len(source)), source)
                    # A row range expands to the row-split tensor of exactly those rows.
                    first, count = rows // 3, rows - rows // 3 - 1
                    part, _ = row_split(codes[first:first + count], self.rng)
                    groups = k // 64
                    part = bytearray(part)
                    scale_part = p2e2.align_up(count * groups * 24, 256)
                    scale_all = p2e2.align_up(rows * groups * 24, 256)
                    part[scale_part:scale_part + count * groups * 2] = source[scale_all + first * groups * 2:
                                                                              scale_all + (first + count) * groups * 2]
                    sub = self.owned(stack, self.gpu, self.gpu.zeros(len(part), fill=0))
                    self.gpu.launch("euhedral_q3_p2e2_expand", (count + 3) // 4,
                                    [C.c_uint64(dc), C.c_uint64(sub), C.c_uint(rows), C.c_uint(k), C.c_uint(first),
                                     C.c_uint(count)])
                    self.assertEqual(self.gpu.download(sub, len(part)), bytes(part))

    def test_embedding_matches_row_split_gather_bitwise(self):
        vocab, hidden = 41, 2048
        codes = adversarial(self.rng, vocab, hidden)
        source, scale_offset = row_split(codes, self.rng)
        tensor = p2e2.encode(source, vocab, hidden)
        ids = np.array([0, 1, 2, 3, 40, -1, 41, 7, 7, 19], dtype=np.int32)
        size = len(ids) * hidden * 2
        with contextlib.ExitStack() as stack:
            dids = self.owned(stack, self.embedding, self.embedding.upload(ids.tobytes()))
            dw = self.owned(stack, self.embedding, self.embedding.upload(source))
            dc = self.owned(stack, self.embedding, self.embedding.upload(tensor))
            y0 = self.owned(stack, self.embedding, self.embedding.zeros(size, fill=SENTINEL))
            y1 = self.owned(stack, self.embedding, self.embedding.zeros(size, fill=SENTINEL))
            self.embedding.launch("euhedral_q3_embedding", len(ids) * ((hidden // 64 + 3) // 4),
                                  [C.c_uint64(dids), C.c_uint64(dw), C.c_uint64(y0), C.c_uint(len(ids)),
                                   C.c_uint(vocab), C.c_uint(hidden), C.c_uint64(scale_offset)])
            self.embedding.launch("euhedral_q3_p2e2_embedding", (len(ids) + 3) // 4,
                                  [C.c_uint64(dids), C.c_uint64(dc), C.c_uint64(y1), C.c_uint(len(ids)),
                                   C.c_uint(vocab), C.c_uint(hidden)])
            expected, actual = self.embedding.download(y0, size), self.embedding.download(y1, size)
        self.assertEqual(actual, expected)
        row = hidden * 2
        self.assertEqual(expected[5 * row:7 * row], bytes([SENTINEL]) * 2 * row)


@unittest.skipIf(UNAVAILABLE is not None, UNAVAILABLE or "")
class P2e2HostTest(unittest.TestCase):
    """Host entry points: geometry checks, the route decision and the expansion destination."""

    @classmethod
    def setUpClass(cls):
        cls.gpu = Gpu(b'#include "q3/kernels.cu"\n')
        cls.directory = tempfile.TemporaryDirectory()
        harness = Path(cls.directory.name) / "p2e2_host.c"
        library = Path(cls.directory.name) / "libp2e2_host.so"
        harness.write_text(HOST_SOURCE)
        cuda_root = ROOT / "build/cuda-dev/linux-x64"
        libdir, runtime = cuda_root / "lib", cuda_root / "runtime"
        subprocess.run([shutil.which("gcc"), "-shared", "-fPIC", "-pthread", "-O2",
                        "-I", str(ROOT / "native/include"), "-I", str(ROOT / "native/src/host"),
                        "-I", str(cuda_root / "include"), "-L", str(libdir), f"-Wl,-rpath,{libdir}:{runtime}",
                        str(harness), "-lcuda", "-lcudart", "-o", str(library)], check=True, capture_output=True, text=True)
        cls.host = C.CDLL(str(library))
        cls.host.p2e2_test_set_module.argtypes = (C.c_size_t,)
        cls.host.p2e2_test_set_exact.argtypes = (C.c_int,)
        cls.host.p2e2_test_set_module(cls.gpu.module.value)
        cls.decode = cls.host.euhedral_cuda_linear_q3_p2e2_decode_bf16
        cls.decode.argtypes = (C.c_void_p, C.c_void_p, C.c_void_p, C.c_uint, C.c_uint, C.c_uint, C.c_uint64)
        cls.expand = cls.host.euhedral_cuda_q3_p2e2_expand
        cls.expand.argtypes = (C.c_void_p, C.c_uint64, C.c_uint, C.c_uint, C.c_uint, C.c_uint, C.c_void_p, C.c_uint64)
        cls.rng = np.random.default_rng(0x9E2E3)

    @classmethod
    def tearDownClass(cls):
        cls.gpu.close()
        cls.directory.cleanup()

    def setUp(self):
        self.host.p2e2_test_set_exact(0)

    def test_routes_and_geometry(self):
        rows, k = 32, 2048
        codes = realistic(self.rng, rows, k)
        source, _ = row_split(codes, self.rng)
        tensor = p2e2.encode(source, rows, k)
        with contextlib.ExitStack() as stack:
            dx = self.gpu.upload(bytes(2 * k)); stack.callback(self.gpu.free, dx)
            dc = self.gpu.upload(tensor); stack.callback(self.gpu.free, dc)
            y = self.gpu.zeros(rows * 2); stack.callback(self.gpu.free, y)
            out = self.gpu.zeros(len(source)); stack.callback(self.gpu.free, out)
            self.assertEqual(self.decode(dx, dc, y, 1, k, rows, len(tensor)), 0)
            self.assertEqual(self.decode(dx, dc, y, 2, k, rows, len(tensor)), -5)
            payload = p2e2.p2e2_offsets(rows, k)[2]
            full = payload + 4 * (rows * k // 16 + p2e2.PAYLOAD_PAD_WORDS)
            self.assertEqual(self.decode(dx, dc, y, 1, k, rows, len(tensor) - 2), -3)
            self.assertEqual(self.decode(dx, dc, y, 1, k, rows, full + 4), -3)
            self.assertEqual(self.decode(dx, dc, y, 1, k, rows, payload + 4 * p2e2.PAYLOAD_PAD_WORDS - 4), -3)
            self.assertEqual(self.decode(dx, dc, y, 1, k + 512, rows, len(tensor)), -3)
            self.host.p2e2_test_set_exact(1)
            self.assertEqual(self.decode(dx, dc, y, 1, k, rows, len(tensor)), -5)
            self.assertEqual(self.expand(dc, len(tensor), rows, k, 0, rows, out, len(source) - 1), -2)
            self.assertEqual(self.expand(dc, len(tensor), rows, k, rows - 1, 2, out, len(source)), -1)
            self.assertEqual(self.expand(dc, len(tensor), rows, k, 0, rows, out, len(source)), 0)
            self.assertEqual(self.gpu.download(out, len(source)), source)


if __name__ == "__main__":
    unittest.main()
