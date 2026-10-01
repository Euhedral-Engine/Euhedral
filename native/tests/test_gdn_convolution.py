"""GDN causal convolution row blocks against the frozen one-thread-per-channel kernel."""
import ctypes as C
import pathlib
import random
import struct
import unittest

from test_q3_primitives import Gpu, NVRTC, SKIP_REASON

HERE = pathlib.Path(__file__).resolve().parent


def bf16(value):
    bits = struct.unpack('<I', struct.pack('<f', value))[0]
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF


@unittest.skipIf(NVRTC is None, str(SKIP_REASON))
class GdnConvolutionTest(unittest.TestCase):
    def test_row_blocks_match_the_sequential_kernel_with_carried_state(self):
        source = b'#include "gdn/kernels.cu"\n' + (HERE / 'fixtures/gdn_convolution_reference.cu').read_bytes()
        gpu = Gpu(source)
        try:
            rng = random.Random(1291)
            qk_width, value_width = 96, 64
            width = qk_width + value_width
            for kernel_size in (2, 4, 32):
                history = kernel_size - 1
                weights = gpu.upload(struct.pack(f'<{kernel_size * width}H',
                                                 *[bf16(rng.uniform(-1, 1)) for _ in range(kernel_size * width)]))
                initial = struct.pack(f'<{width * history}H', *[bf16(rng.uniform(-2, 2)) for _ in range(width * history)])
                states = [gpu.upload(initial), gpu.upload(initial)]
                try:
                    # Chained calls carry state: a single row, partial and whole blocks, many blocks.
                    for rows in (1, 2, 31, 32, 33, 64, 100, 512):
                        qk = gpu.upload(struct.pack(f'<{rows * qk_width}H', *[bf16(rng.uniform(-3, 3)) for _ in range(rows * qk_width)]))
                        vz = gpu.upload(struct.pack(f'<{rows * value_width * 2}H',
                                                    *[bf16(rng.uniform(-3, 3)) for _ in range(rows * value_width * 2)]))
                        outputs = []
                        try:
                            for name, grid, state in (('reference_gdn_convolution_bf16', ((width + 127) // 128, 1), states[0]),
                                                      ('euhedral_gdn_convolution_bf16', ((width + 127) // 128, (rows + 31) // 32), states[1])):
                                y = gpu.zeros(rows * width * 2, fill=0xA5)
                                try:
                                    gpu.launch(name, grid, [C.c_uint64(qk), C.c_uint64(vz), C.c_uint64(weights), C.c_uint64(state),
                                                            C.c_uint64(y), C.c_uint(rows), C.c_uint(qk_width), C.c_uint(value_width),
                                                            C.c_uint(width), C.c_uint(kernel_size)])
                                    outputs.append(gpu.download(y, rows * width * 2))
                                finally:
                                    gpu.free(y)
                            self.assertEqual(outputs[0], outputs[1], (kernel_size, rows, 'output'))
                            self.assertEqual(gpu.download(states[0], width * history * 2),
                                             gpu.download(states[1], width * history * 2), (kernel_size, rows, 'state'))
                        finally:
                            gpu.free(qk)
                            gpu.free(vz)
                finally:
                    for pointer in states + [weights]:
                        gpu.free(pointer)
        finally:
            gpu.close()


if __name__ == '__main__':
    unittest.main()
