#pragma once
#include "../q3/strategies/k32_prefill.cuh"
namespace qwen_ffn_tiles {
using namespace k32_probe;
// P: B operand parts per weight. 2 stages BF16 hi and lo, whose sum is code * scale exactly (the
// exact kernels); 1 stages hi only, the BF16 rounding of code * scale, and runs half the MMAs.
template<int F,int N,int P=1>
struct alignas(32) Storage {
    __nv_bfloat16 a[2][2][F*16*32];
    __nv_bfloat16 b[2][2][P][N*32];
    alignas(8) unsigned long long ready[4][2],release[4][2];
};
template<int F>
static __device__ __forceinline__ void activation(__nv_bfloat16* a,const unsigned short* x,unsigned int rows,unsigned int width,unsigned int row0,unsigned int mb,unsigned int base,unsigned int lane){
    #pragma unroll
    for(unsigned int pass=0;pass<F*2;++pass){
        unsigned int lr=(lane>>2)+pass*8u,lk=(lane&3u)*8u;
        unsigned int row=row0+mb*16u+(lr&15u)+(lr/16u)*32u;
        unsigned int fields[4]={};
        if(row<rows){const unsigned short* src=x+(unsigned long long)row*width+base+lk;
            asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];":"=r"(fields[0]),"=r"(fields[1]),"=r"(fields[2]),"=r"(fields[3]):"l"(src):"memory");}
        #pragma unroll
        for(int j=0;j<4;++j){unsigned int lo=fields[j]&65535u,hi=fields[j]>>16;
            lo=(lo&32767u)>32640u?32767u:lo;hi=(hi&32767u)>32640u?32767u:hi;fields[j]=lo|(hi<<16);}
        unsigned int dst=static_cast<unsigned int>(__cvta_generic_to_shared(a+lr*32u+lk));
        asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};"::"r"(dst),"r"(fields[0]),"r"(fields[1]),"r"(fields[2]),"r"(fields[3]):"memory");
    }
}
}
