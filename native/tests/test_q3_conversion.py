"""Device-level Q3 scale-conversion contract and codegen gate."""
import contextlib
import ctypes as C
import struct
import unittest

from test_q3_primitives import Gpu, NVRTC

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


STAGING_SOURCE = b'''
#include "q3/primitives/activation.cuh"
extern "C" __global__ void probe_stage(const unsigned short* source, unsigned short* output,
        unsigned int rows, unsigned int width, unsigned int row_start, unsigned int k_base) {
    __shared__ __align__(32) __nv_bfloat16 tile[32*64];
    q3::stage_activation_tile<32,64,128>(tile,source,rows,width,row_start+blockIdx.x*32,k_base,threadIdx.x);
    __syncthreads();
    for (unsigned i=threadIdx.x;i<32*64;i+=128)
        output[blockIdx.x*2048+i]=__bfloat16_as_ushort(tile[i]);
}
'''


@unittest.skipIf(NVRTC is None, 'pinned NVRTC unavailable')
class Q3StagingConversionTest(unittest.TestCase):
    def test_all_bf16_patterns_preserve_original_staging_bits(self):
        with contextlib.ExitStack() as cleanup:
            gpu = Gpu(STAGING_SOURCE)
            cleanup.callback(gpu.close)
            source = gpu.upload(struct.pack('<65536H', *range(65536)))
            cleanup.callback(gpu.free, source)
            output = gpu.zeros(65536 * 2, 0xa5)
            cleanup.callback(gpu.free, output)
            gpu.launch('probe_stage', 32, [C.c_uint64(source), C.c_uint64(output),
                                           C.c_uint(1024), C.c_uint(64), C.c_uint(0), C.c_uint(0)])
            actual = struct.unpack('<65536H', gpu.download(output, 65536 * 2))
        expected = tuple(0x7fff if i & 0x7f80 == 0x7f80 and i & 0x7f else i
                         for i in range(65536))
        self.assertEqual(expected, actual)

    def test_partial_rows_and_columns_still_zero_fill(self):
        with contextlib.ExitStack() as cleanup:
            gpu = Gpu(STAGING_SOURCE)
            cleanup.callback(gpu.close)
            source = gpu.upload(struct.pack('<4429H', *range(4429)))
            cleanup.callback(gpu.free, source)
            output = gpu.zeros(2048 * 2, 0xa5)
            cleanup.callback(gpu.free, output)
            gpu.launch('probe_stage', 1, [C.c_uint64(source), C.c_uint64(output),
                                          C.c_uint(43), C.c_uint(103), C.c_uint(32), C.c_uint(64)])
            actual = struct.unpack('<2048H', gpu.download(output, 2048 * 2))
        expected = tuple((32+i//64)*103+64+i%64 if 32+i//64 < 43 and 64+i%64 < 103 else 0
                         for i in range(2048))
        self.assertEqual(expected, actual)

    def test_odd_width_and_k_base_preserve_special_values(self):
        values = [0x7f81, 0xff81, 0x7fff, 0x7f80, 0xff80, 0x8000]
        source_bits = (values * ((43*103 + len(values)-1)//len(values)))[:43*103]
        for k_base in (1, 64):
            with self.subTest(k_base=k_base), contextlib.ExitStack() as cleanup:
                gpu = Gpu(STAGING_SOURCE)
                cleanup.callback(gpu.close)
                source = gpu.upload(struct.pack(f'<{len(source_bits)}H', *source_bits))
                cleanup.callback(gpu.free, source)
                output = gpu.zeros(2048 * 2, 0xa5)
                cleanup.callback(gpu.free, output)
                gpu.launch('probe_stage', 1, [C.c_uint64(source), C.c_uint64(output),
                                              C.c_uint(43), C.c_uint(103), C.c_uint(32), C.c_uint(k_base)])
                actual = struct.unpack('<2048H', gpu.download(output, 2048 * 2))
                expected = []
                for i in range(2048):
                    r, k = 32+i//64, k_base+i%64
                    bits = source_bits[r*103+k] if r < 43 and k < 103 else 0
                    expected.append(0x7fff if bits & 0x7f80 == 0x7f80 and bits & 0x7f else bits)
                self.assertEqual(tuple(expected), actual)


if __name__ == '__main__':
    unittest.main()
