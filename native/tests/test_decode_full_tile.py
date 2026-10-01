"""Full-tile decode route and boundary preservation against frozen schedules."""
import ctypes as C
import pathlib
import random
import struct
import unittest

from test_q3_primitives import Gpu, NVRTC, SKIP_REASON, PROBES

ROOT = pathlib.Path(__file__).resolve().parents[2]


class DecodePolicyTest(unittest.TestCase):
    def test_q45_has_an_explicit_full_tile_policy(self):
        source = (ROOT / 'native/src/q45/strategies/decode.cuh').read_text()
        self.assertIn('use_full_tile_decode', source)

    def test_q3_has_an_explicit_full_tile_policy(self):
        # Route observability is required: generic dispatch must not silently use
        # an optimized schedule for excluded rows or partial tiles.
        source = (ROOT / 'native/src/q3/strategies/decode.cuh').read_text()
        self.assertIn('use_full_tile_decode', source)


@unittest.skipIf(NVRTC is None, str(SKIP_REASON))
class FullTileDecodeTest(unittest.TestCase):
    def test_full_tile_policies_at_boundaries(self):
        source = b'''#include "q3/kernels.cu"
#include "q45/kernels.cu"
extern "C" __global__ void policy(unsigned int* out,unsigned int r,unsigned int k,unsigned int n){
 if(threadIdx.x==0){out[0]=q3::use_full_tile_decode(r,k,n);out[1]=q45::use_full_tile_decode(r,k,n);}}
'''
        gpu = Gpu(source)
        try:
            for rows, width, outputs, expected in [(1,128,8,1),(1,256,16,1),(0,128,8,0),
                    (2,128,8,0),(4,128,8,0),(1,0,8,0),(1,127,8,0),(1,129,8,0),
                    (1,128,0,0),(1,128,7,0),(1,128,9,0)]:
                ptr = gpu.zeros(8, fill=0xa5)
                try:
                    gpu.launch('policy',1,[C.c_uint64(ptr),C.c_uint(rows),C.c_uint(width),C.c_uint(outputs)])
                    self.assertEqual(struct.unpack('<II',gpu.download(ptr,8)),(expected,expected))
                finally:
                    gpu.free(ptr)
        finally:
            gpu.close()

    def test_q3_all_half_scale_patterns_match_group_reference(self):
        gpu = Gpu(b'#include "q3/kernels.cu"\n' + PROBES.encode())
        try:
            width, outputs = 128, 65536
            scale = outputs * 48
            codes = bytes((i*97+13)&255 for i in range(48))*outputs
            scales = b''.join(struct.pack('<HH',i,i) for i in range(outputs))
            w = gpu.upload(codes+scales)
            try:
                for value in (0x3f80,0x8001,0x7fc1):
                    x = gpu.upload(struct.pack('<128H',*([value]*128)))
                    actual = []
                    try:
                        for symbol in ('euhedral_q3_decode_1','probe_group_decode_1'):
                            y = gpu.zeros(outputs*2,fill=0xa5)
                            try:
                                gpu.launch(symbol,outputs//8,[C.c_uint64(x),C.c_uint64(w),C.c_uint64(y),
                                           C.c_uint(1),C.c_uint(width),C.c_uint(outputs),C.c_ulonglong(scale)])
                                actual.append(gpu.download(y,outputs*2))
                            finally:
                                gpu.free(y)
                        self.assertEqual(actual[0],actual[1])
                    finally:
                        gpu.free(x)
            finally:
                gpu.free(w)
        finally:
            gpu.close()

    def test_q3_full_and_partial_shapes_match_group_reference(self):
        gpu = Gpu(b'#include "q3/kernels.cu"\n' + PROBES.encode())
        try:
            rng = random.Random(731)
            for rows, width, outputs in [(1,128,8),(1,256,16),(1,129,9),(1,127,7),
                                          (2,128,8),(4,256,16),(1,5120,16)]:
                groups = ((width + 127)//128)*2
                scale = (outputs*groups*24+255)&~255
                payload = bytearray(rng.randrange(256) for _ in range(scale+outputs*groups*2))
                specials = [0,1,0x8000,0x8001,0x3c00,0x7bff,0x7c00,0x7e11,0xfe11]
                for mode in ['finite','special']:
                    for i in range(outputs*groups):
                        struct.pack_into('<H',payload,scale+i*2,
                                         specials[i%len(specials)] if mode=='special' else rng.randrange(0x2800,0x3800))
                    w=gpu.upload(payload)
                    for align in [0,2]:
                        values=[rng.randrange(0x3d00,0x4100) | (rng.randrange(2)<<15) for _ in range(rows*width)]
                        x=gpu.upload(bytes(align)+struct.pack(f'<{len(values)}H',*values))
                        try:
                            for tile in [1,2,4]:
                                result=[]
                                for symbol in [f'euhedral_q3_decode_{tile}',f'probe_group_decode_{tile}']:
                                    y=gpu.zeros(rows*outputs*2,fill=0xa5)
                                    try:
                                        gpu.launch(symbol,((rows+tile-1)//tile)*((outputs+7)//8),
                                                   [C.c_uint64(x+align),C.c_uint64(w),C.c_uint64(y),
                                                    C.c_uint(rows),C.c_uint(width),C.c_uint(outputs),C.c_ulonglong(scale)])
                                        result.append(gpu.download(y,rows*outputs*2))
                                    finally:gpu.free(y)
                                self.assertEqual(*result,(rows,width,outputs,mode,align,tile))
                        finally:gpu.free(x)
                    gpu.free(w)
        finally:gpu.close()

    def test_q3_wide_decode_matches_plain_decode_bitwise(self):
        # A partial single chunk (K 512), rows ending mid-chunk (5120), several stages of the double
        # slot (17408), the largest scale row (32768); one and many CTAs; finite and special scales;
        # activations with both signs and BF16 specials.
        gpu = Gpu(b'#include "q3/kernels.cu"\n')
        try:
            rng = random.Random(983)
            for width, outputs in [(512, 8), (1536, 16), (5120, 24), (6144, 5120), (17408, 1024), (32768, 8)]:
                groups = width // 64
                scale = (outputs * groups * 24 + 255) & ~255
                payload = bytearray(rng.randrange(256) for _ in range(scale + outputs * groups * 2))
                specials = [0, 1, 0x8000, 0x8001, 0x3c00, 0x7bff, 0x7c00, 0x7e11, 0xfe11]
                for mode in ['finite', 'special']:
                    for i in range(outputs * groups):
                        struct.pack_into('<H', payload, scale + i * 2,
                                         specials[i % len(specials)] if mode == 'special' else rng.randrange(0x2800, 0x3800))
                    w = gpu.upload(payload)
                    values = [rng.choice([0x7f80, 0xff80, 0x7fc1, 0x0001, 0x8000]) if rng.randrange(97) == 0
                              else rng.randrange(0x3d00, 0x4100) | (rng.randrange(2) << 15) for _ in range(width)]
                    x = gpu.upload(struct.pack(f'<{width}H', *values))
                    try:
                        result = []
                        for symbol in ['euhedral_q3_decode_1', 'euhedral_q3_decode_wide']:
                            y = gpu.zeros(outputs * 2, fill=0xa5)
                            try:
                                gpu.launch(symbol, outputs // 8, [C.c_uint64(x), C.c_uint64(w), C.c_uint64(y), C.c_uint(1),
                                                                  C.c_uint(width), C.c_uint(outputs), C.c_ulonglong(scale)])
                                result.append(gpu.download(y, outputs * 2))
                            finally:
                                gpu.free(y)
                        self.assertEqual(result[0], result[1], (width, outputs, mode))
                    finally:
                        gpu.free(x)
                        gpu.free(w)
        finally:
            gpu.close()


if __name__ == '__main__':
    unittest.main()
