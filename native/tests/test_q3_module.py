"""Installed Q3 headers must compile from the product beside the unchanged entry TU."""
import ctypes as C
import hashlib
import os
from pathlib import Path
import random
import shutil
import struct
import subprocess
import tempfile
import unittest
from test_q3_primitives import Gpu, NVRTC

ROOT = Path(__file__).resolve().parents[2]
PRODUCT = ROOT / 'build/native/linux-x64/share/euhedral_cuda'
HEADERS = (
    'numeric.cuh', 'layout.cuh', 'primitives/packed_load.cuh',
    'primitives/activation.cuh', 'primitives/decode.cuh',
    'primitives/staging.cuh', 'primitives/mma.cuh', 'primitives/mma_leaf.cuh',
    'primitives/accumulation.cuh', 'primitives/writeback.cuh',
    'strategies/scalar.cuh', 'strategies/decode.cuh',
    'strategies/prefill.cuh', 'strategies/k32_prefill.cuh',
)

HOST_DISPATCH_SOURCE = r'''#include <stdint.h>
#include <cuda.h>
#include "cuda_kernel_loader.h"
#include "euhedral_cuda.h"
static CUmodule injected_module;
void q3_test_set_module(uintptr_t handle) { injected_module = (CUmodule)handle; }
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
int euhedral_cuda_exact_numerics(void) { return 0; }
int euhedral_cuda_select_exact_numerics(int exact) { (void)exact; return 0; }
CUresult euhedral_launch_kernel(CUfunction function, unsigned int gx, unsigned int gy, unsigned int gz,
        unsigned int bx, unsigned int by, unsigned int bz, unsigned int shared, CUstream stream,
        void** parameters, void** extra) {
    return cuLaunchKernel(function, gx, gy, gz, bx, by, bz, shared, stream, parameters, extra);
}
#include "q3_linear_bf16.c"
'''


def build_dispatch_host(directory, name):
    directory = Path(directory)
    harness = directory / f'{name}.c'
    library_path = directory / f'lib{name}.so'
    harness.write_text(HOST_DISPATCH_SOURCE)
    cuda_root = ROOT / 'build/cuda-dev/linux-x64'
    libdir, runtime = cuda_root / 'lib', cuda_root / 'runtime'
    command = [shutil.which('gcc'), '-shared', '-fPIC', '-pthread', '-O2',
               '-I', str(ROOT / 'native/include'), '-I', str(ROOT / 'native/src'),
               '-I', str(cuda_root / 'include'), '-L', str(libdir),
               f'-Wl,-rpath,{libdir}:{runtime}', str(harness), '-lcuda', '-lcudart',
               '-o', str(library_path)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    host = C.CDLL(str(library_path))
    host.q3_test_set_module.argtypes = (C.c_size_t,)
    host.q3_test_set_module.restype = None
    dispatch = host.euhedral_cuda_linear_q3_prefill_bf16
    dispatch.argtypes = (C.c_void_p, C.c_void_p, C.c_void_p, C.c_uint,
                         C.c_uint, C.c_uint, C.c_uint64)
    dispatch.restype = C.c_int
    return host, dispatch


class Q3ModuleTest(unittest.TestCase):
    def test_product_contains_header_graph(self):
        for relative in HEADERS:
            with self.subTest(header=relative):
                self.assertEqual((ROOT / 'native/src/q3' / relative).read_bytes(),
                                 (PRODUCT / 'q3' / relative).read_bytes())

    @unittest.skipIf(NVRTC is None, 'pinned NVRTC unavailable')
    def test_headers_can_be_included_directly(self):
        # Compile and load from the installed product, not the repo's source tree.
        source = b'''#include "q3/primitives/staging.cuh"
#include "q3/primitives/mma.cuh"
extern "C" __global__ void module_probe(unsigned short* x) {
'''
        source += b'''  x[0] = q3::float_to_bf16(q3::apply_scale(q3::decode_code(4,0), 1.0f));\n}\n'''
        gpu = Gpu(source, include_dir=PRODUCT)
        try:
            out = gpu.zeros(2)
            try:
                gpu.launch('module_probe', 1, [C.c_uint64(out)])
                self.assertEqual(gpu.download(out, 2), b'\x80\xc0')  # -4.0 BF16
            finally:
                gpu.free(out)
        finally:
            gpu.close()

    @unittest.skipIf(NVRTC is None, 'pinned NVRTC unavailable')
    def test_new_host_falls_back_with_pre_cb_module_and_uses_legacy_grid(self):
        gcc = shutil.which('gcc')
        if gcc is None:
            self.skipTest('gcc unavailable for host-dispatch harness')
        legacy_source = Path(__file__).parent / 'fixtures/q3-kernels-pre-cb.cu'
        self.assertEqual('219e408041d49f1d03d3ce0a8676f71ab09b2a391d173b1da3aef46d7824fc29',
                         hashlib.sha256(legacy_source.read_bytes()).hexdigest())
        gpu = Gpu(legacy_source.read_bytes(), include_dir=PRODUCT / 'q3')
        try:
            absent = C.c_void_p()
            status = gpu.function(C.byref(absent), gpu.module,
                                  b'euhedral_q3_prefill_64_k32_cb')
            self.assertNotEqual(0, status, 'legacy module unexpectedly exports candidate CB kernel')
            with tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR', str(ROOT / 'build'))) as temp:
                host, dispatch = build_dispatch_host(temp, 'legacy_dispatch')
                host.q3_test_set_module(gpu.module.value)

                rows, width, outputs = 256, 6144, 5120
                groups = ((width + 127) // 128 * 128) // 64
                base = outputs * groups * 24
                scale_offset = (base + 255) & ~255
                weights = bytearray(random.Random(7).randbytes(base))
                weights.extend(bytes(scale_offset - base + outputs * groups * 2))
                for index in range(outputs * groups):
                    struct.pack_into('<H', weights, scale_offset + index * 2, 0x3C00)
                input_data = b'\x80\x3f' * (rows * width)  # finite BF16 1.0
                x = gpu.upload(input_data)
                w = gpu.upload(weights)
                expected = gpu.zeros(rows * outputs * 2, 0xA5)
                actual = gpu.zeros(rows * outputs * 2, 0xA5)
                try:
                    grid = ((rows + 31) // 32) * ((outputs + 31) // 32)
                    gpu.launch('euhedral_q3_prefill', grid,
                               [C.c_uint64(x), C.c_uint64(w), C.c_uint64(expected),
                                C.c_uint(rows), C.c_uint(width), C.c_uint(outputs),
                                C.c_ulonglong(scale_offset)])
                    status = dispatch(C.c_void_p(x), C.c_void_p(w), C.c_void_p(actual),
                                      rows, width, outputs, len(weights))
                    self.assertEqual(0, status)
                    self.assertEqual(gpu.download(expected, rows * outputs * 2),
                                     gpu.download(actual, rows * outputs * 2))
                finally:
                    gpu.free(actual)
                    gpu.free(expected)
                    gpu.free(w)
                    gpu.free(x)
        finally:
            gpu.close()

    @unittest.skipIf(NVRTC is None, 'pinned NVRTC unavailable')
    def test_host_avoids_cb_for_two_byte_aligned_but_not_16_byte_aligned_input(self):
        if shutil.which('gcc') is None:
            self.skipTest('gcc unavailable for host-dispatch harness')
        legacy_source = Path(__file__).parent / 'fixtures/q3-kernels-pre-cb.cu'
        self.assertEqual('219e408041d49f1d03d3ce0a8676f71ab09b2a391d173b1da3aef46d7824fc29',
                         hashlib.sha256(legacy_source.read_bytes()).hexdigest())
        stub_cb = b'''\nextern "C" __global__ __launch_bounds__(128) void euhedral_q3_prefill_64_k32_cb(\n        const unsigned short*, const unsigned char*, unsigned short*, unsigned int,\n        unsigned int, unsigned int, unsigned long long) {}\n'''
        gpu = Gpu(legacy_source.read_bytes() + stub_cb, include_dir=PRODUCT / 'q3')
        try:
            candidate_symbol = C.c_void_p()
            self.assertEqual(0, gpu.function(C.byref(candidate_symbol), gpu.module,
                                             b'euhedral_q3_prefill_64_k32_cb'))
            with tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR', str(ROOT / 'build'))) as temp:
                host, dispatch = build_dispatch_host(temp, 'unaligned_dispatch')
                host.q3_test_set_module(gpu.module.value)
                rows, width, outputs = 256, 6144, 5120
                groups = width // 64
                base = outputs * groups * 24
                scale_offset = (base + 255) & ~255
                weights = bytearray(random.Random(11).randbytes(base))
                weights.extend(bytes(scale_offset - base + outputs * groups * 2))
                for index in range(outputs * groups):
                    struct.pack_into('<H', weights, scale_offset + index * 2, 0x3C00)
                input_data = b'\x00\x00' + b'\x80\x3f' * (rows * width)
                input_base = gpu.upload(input_data)
                x = input_base + 2
                self.assertEqual(2, x & 15)
                w = gpu.upload(weights)
                expected = gpu.zeros(rows * outputs * 2, 0xA5)
                actual = gpu.zeros(rows * outputs * 2, 0xA5)
                try:
                    grid32 = ((rows + 31) // 32) * ((outputs + 31) // 32)
                    gpu.launch('euhedral_q3_prefill', grid32,
                               [C.c_uint64(x), C.c_uint64(w), C.c_uint64(expected),
                                C.c_uint(rows), C.c_uint(width), C.c_uint(outputs),
                                C.c_ulonglong(scale_offset)])
                    status = dispatch(C.c_void_p(x), C.c_void_p(w), C.c_void_p(actual),
                                      rows, width, outputs, len(weights))
                    self.assertEqual(0, status)
                    self.assertEqual(gpu.download(expected, rows * outputs * 2),
                                     gpu.download(actual, rows * outputs * 2))
                finally:
                    gpu.free(actual)
                    gpu.free(expected)
                    gpu.free(w)
                    gpu.free(input_base)
        finally:
            gpu.close()


if __name__ == '__main__':
    unittest.main()
