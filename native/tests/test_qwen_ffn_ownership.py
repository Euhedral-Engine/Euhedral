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
            for rows in (64, 129):
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
                                tile=64 if rows==64 else 128
                                gpu.launch(f'stream_down_{tile}x64',((rows+tile-1)//tile)*(outputs//64),
                                           [p(region),p(weights),p(actual),u(rows),u(width),u(outputs),p(scale),p(state),u(start),u(count)])
                            self.assertEqual(oracle,gpu.download(state,rows*outputs*4),'FP32 boundary')
                            self.assertEqual(gpu.download(expected,rows*outputs*2),gpu.download(actual,rows*outputs*2),'BF16 boundary')
                        for tile in (64,128):
                            gpu.launch(f'euhedral_q3_ffn_down_{tile}x64',((rows+tile-1)//tile)*(outputs//64),
                                       [p(x),p(weights),p(actual),u(rows),u(width),u(outputs),p(scale)])
                            self.assertEqual(gpu.download(expected,rows*outputs*2),gpu.download(actual,rows*outputs*2),f'{tile}-row tile')

if __name__=='__main__':unittest.main()
