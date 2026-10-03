"""Device-level Q3 scale-conversion contract and codegen gate."""
import contextlib
import ctypes as C
import struct
import unittest

from gpu_harness import Gpu, NVRTC

SOURCE = b'''
#include "q3/numeric.cuh"
extern "C" __global__ void probe_scale_conversion(const unsigned short* source, unsigned int* output) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    output[i] = __float_as_uint(q3::fp16_to_float(source[i]));
}
'''


def original_scale_bits(bits):
    exponent, mantissa = (bits >> 10) & 31, bits & 1023
    if exponent == 31 and mantissa:
        return ((bits & 0x8000) << 16) | 0x7f800000 | (mantissa << 13)
    value = struct.unpack('<e', struct.pack('<H', bits))[0]
    return struct.unpack('<I', struct.pack('<f', value))[0]


@unittest.skipIf(NVRTC is None, 'pinned NVRTC unavailable')
class Q3ScaleConversionTest(unittest.TestCase):
    def test_all_fp16_patterns_preserve_original_float_bits(self):
        with contextlib.ExitStack() as cleanup:
            gpu = Gpu(SOURCE)
            cleanup.callback(gpu.close)
            source = gpu.upload(struct.pack('<65536H', *range(65536)))
            cleanup.callback(gpu.free, source)
            result = gpu.zeros(65536 * 4, 0xa5)
            cleanup.callback(gpu.free, result)
            gpu.launch('probe_scale_conversion', 512, [C.c_uint64(source), C.c_uint64(result)])
            actual = struct.unpack('<65536I', gpu.download(result, 65536 * 4))
        mismatches = [(i, hex(original_scale_bits(i)), hex(b)) for i, b in enumerate(actual)
                      if original_scale_bits(i) != b]
        self.assertEqual([], mismatches[:12], f'{len(mismatches)} mismatches across all FP16 patterns')

    def test_converter_codegen_contains_native_half_conversion(self):
        with contextlib.ExitStack() as cleanup:
            gpu = Gpu(SOURCE)
            cleanup.callback(gpu.close)
            ptx = gpu._ptx(SOURCE, None, 14).value.decode()
        self.assertIn('cvt.f32.f16', ptx)
