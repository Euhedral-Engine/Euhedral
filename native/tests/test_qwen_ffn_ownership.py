"""FFN down continuation preserves ordinary K16 hi/lo FP32 state, including ragged rows."""
import contextlib
import ctypes as C
import random
import struct
import unittest
from test_q3_primitives import Gpu, NVRTC, CUDA, _check

SOURCE = '#include "qwen_ffn.cu"\n#include "q3/kernels.cu"\n#include "q45/kernels.cu"\n#include "qwen_attention_ops.cu"\nextern "C" __global__ void hold_consumer() {\n    unsigned long long begin=clock64();\n    while(clock64()-begin<1000000ull) {}\n}\nextern "C" __global__ __launch_bounds__(128) void reference_acc(\n const unsigned short* x,const unsigned char* w,unsigned short* y,unsigned int m,\n unsigned int k,unsigned int n,unsigned long long scale,float* accum) {\n __shared__ k32_probe::Shared storage;\n k32_probe::run<false,false,true,true>(x,w,y,m,k,n,scale,accum,storage);\n}\n'

@unittest.skipIf(NVRTC is None or CUDA is None, "CUDA/NVRTC unavailable")
class FfnOwnershipTest(unittest.TestCase):
    def test_wide_down_matches_ordinary_unrounded_state(self):
        with contextlib.ExitStack() as scope:
            gpu = Gpu(SOURCE.encode()); scope.callback(gpu.close)
            rng = random.Random(719)
            for rows in (64, 100, 129):
                for special in (False, True):
                    with self.subTest(rows=rows, special=special), contextlib.ExitStack() as case:
                        def upload(data):
                            v=gpu.upload(data);case.callback(gpu.free,v);return v
                        def alloc(size):
                            v=gpu.zeros(size,0xA5);case.callback(gpu.free,v);return v
                        p,u=C.c_uint64,C.c_uint
                        width,outputs=288,64
                        groups=outputs*((width+127)//128)*2;scale=(groups*24+255)&~255
                        weights=upload(rng.randbytes(groups*24)+bytes(scale-groups*24)+b'\x00\x28'*groups)
                        patterns=[0,1,0x8000,0x8001,0x3f81,0xbf80,0x3e01]
                        if special:patterns += [0x7f80,0xff80,0x7fc1,0xffc7,0x7f81]
                        raw=struct.pack('<'+'H'*(rows*width),*[rng.choice(patterns) for _ in range(rows*width)])
                        x=upload(raw);expected=alloc(rows*outputs*2);actual=alloc(rows*outputs*2)
                        dump_size=((rows+63)//64)*64*outputs*4
                        dump=alloc(dump_size);state=alloc(rows*outputs*4)
                        gpu.launch('reference_acc',((rows+63)//64)*(outputs//32),
                                   [p(x),p(weights),p(expected),u(rows),u(width),u(outputs),p(scale),p(dump)])
                        tiled=gpu.download(dump,dump_size)
                        oracle=b''.join(tiled[((r//64)*(outputs//32)+t)*64*32*4+(r%64)*32*4:((r//64)*(outputs//32)+t)*64*32*4+(r%64+1)*32*4] for r in range(rows) for t in range(outputs//32))
                        for chunk in (64,128):
                            for start in range(0,width,chunk):
                                count=min(chunk,width-start)
                                region=upload(b''.join(raw[(r*width+start)*2:(r*width+start+count)*2] for r in range(rows)))
                                tile=128
                                gpu.launch(f'stream_down_{tile}x64',((rows+tile-1)//tile)*(outputs//64),
                                           [p(region),p(weights),p(actual),u(rows),u(width),u(outputs),p(scale),p(state),u(start),u(count)])
                            self.assertEqual(oracle,gpu.download(state,rows*outputs*4),'FP32 boundary')
                            self.assertEqual(gpu.download(expected,rows*outputs*2),gpu.download(actual,rows*outputs*2),'BF16 boundary')
                        for tile in (64,128):
                            gpu.launch(f'euhedral_q3_ffn_down_{tile}x64',((rows+tile-1)//tile)*(outputs//64),
                                       [p(x),p(weights),p(actual),u(rows),u(width),u(outputs),p(scale)])
                            self.assertEqual(gpu.download(expected,rows*outputs*2),gpu.download(actual,rows*outputs*2),f'{tile}-row tile')

    def test_split_k_down_stays_within_one_bf16_step_of_the_unsplit_leaf(self):
        # Split-K reorders the FP32 accumulation; after one BF16 rounding every output stays within
        # one BF16 step of the unsplit leaf (or an absolute floor for cancellation near zero).
        with contextlib.ExitStack() as scope:
            gpu = Gpu(SOURCE.encode()); scope.callback(gpu.close)
            rng = random.Random(733)
            p, u = C.c_uint64, C.c_uint
            width, outputs, splits = 2048, 128, 4
            groups = outputs * (width // 64); scale = (groups * 24 + 255) & ~255
            weights = gpu.upload(rng.randbytes(groups * 24) + bytes(scale - groups * 24)
                                 + b''.join(struct.pack('<H', rng.randrange(0x1c00, 0x2400)) for _ in range(groups)))
            scope.callback(gpu.free, weights)
            for rows, tile in ((64, 64), (100, 128), (256, 128)):
                with self.subTest(rows=rows, tile=tile), contextlib.ExitStack() as case:
                    def alloc(size):
                        v = gpu.zeros(size, 0xA5); case.callback(gpu.free, v); return v
                    values = [rng.randrange(0x3c00, 0x4000) | (rng.randrange(2) << 15) for _ in range(rows * width)]
                    x = gpu.upload(struct.pack(f'<{rows * width}H', *values)); case.callback(gpu.free, x)
                    expected, actual = alloc(rows * outputs * 2), alloc(rows * outputs * 2)
                    partial = alloc(splits * rows * outputs * 4)
                    gpu.launch(f'euhedral_q3_ffn_down_{tile}x64', ((rows + tile - 1) // tile) * (outputs // 64),
                               [p(x), p(weights), p(expected), u(rows), u(width), u(outputs), p(scale)])
                    gpu.launch(f'euhedral_q3_ffn_down_split_{tile}x64', (((rows + tile - 1) // tile) * (outputs // 64), splits),
                               [p(x), p(weights), p(partial), u(rows), u(width), u(outputs), p(scale), u(splits)])
                    function = C.c_void_p()
                    self.assertEqual(0, gpu.function(C.byref(function), gpu.module, b'euhedral_q3_ffn_down_reduce'))
                    args = [p(partial), p(actual), u(rows * outputs), u(splits)]
                    params = (C.c_void_p * len(args))(*[C.cast(C.pointer(a), C.c_void_p) for a in args])
                    self.assertEqual(0, gpu.launch_kernel(function, (rows * outputs + 255) // 256, 1, 1, 256, 1, 1, 0, None, params, None))
                    self.assertEqual(0, gpu.sync())
                    a = struct.unpack(f'<{rows * outputs}H', gpu.download(expected, rows * outputs * 2))
                    b = struct.unpack(f'<{rows * outputs}H', gpu.download(actual, rows * outputs * 2))
                    f = lambda bits: struct.unpack('<f', struct.pack('<I', bits << 16))[0]
                    rms = (sum(f(v) ** 2 for v in a) / len(a)) ** 0.5
                    for i, (e, o) in enumerate(zip(a, b)):
                        self.assertLessEqual(abs(f(e) - f(o)), max(abs(f(e)) / 128, 1e-3 * rms), (rows, i, f(e), f(o)))
                    self.assertLess(sum(e != o for e, o in zip(a, b)), len(a) // 20)

if __name__=='__main__':unittest.main()
