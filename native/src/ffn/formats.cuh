#pragma once
#include "../q3/strategies/k32_prefill.cuh"
#include "../q45/layout.cuh"
#include "../q45/primitives/staging.cuh"
// B-operand formats of the tile engine (down.cuh). Each stages one 16-column x K32 tile, column-major
// (col * 32 + k), from a compact register prefetch: lanes 8c..8c+7 own column c of each 4-column pass.
namespace qwen_ffn_tiles {
struct Q3B {
    using Layout = q3::Layout;
    using Compact = k32_probe::CompactB;
    static __device__ __forceinline__ Layout layout(const unsigned char* w,unsigned int width,unsigned int,unsigned long long scale){return Layout(w,width,scale);}
    template<int P>
    static __device__ __forceinline__ void produce(__nv_bfloat16* hi,__nv_bfloat16* lo,const Layout& w,unsigned int outputs,unsigned int col,unsigned int base,unsigned int lane){
        k32_probe::produce_b<false,P>(hi,lo,w,outputs,col,base,lane);}
    static __device__ __forceinline__ void prefetch(Compact& next,const Layout& w,unsigned int outputs,unsigned int col,unsigned int base,unsigned int lane){
        k32_probe::prefetch_compact_b(next,w,outputs,col,base,lane);}
    template<int P>
    static __device__ __forceinline__ void stage(__nv_bfloat16* hi,__nv_bfloat16* lo,const Compact& next,unsigned int lane){
        k32_probe::stage_compact_b<false,P>(hi,lo,next,lane);}
};
// Q4/Q5: a K32 half group is 4 code words (16 bytes) and, for Q5, one fifth-bit word. Sublanes 0-3
// hold the code words, sublane 4 the fifth-bit word and sublane 5 the FP16 scale, one register per pass.
template<int BITS>
struct Q45B {
    using Layout = q45::Layout<BITS>;
    struct Compact { unsigned int words[4]; };
    static __device__ __forceinline__ Layout layout(const unsigned char* w,unsigned int width,unsigned int outputs,unsigned long long){return Layout(w,width,outputs);}
    static __device__ __forceinline__ void prefetch(Compact& next,const Layout& w,unsigned int outputs,unsigned int first_col,unsigned int base,unsigned int lane){
        unsigned int sublane=lane&7u,half=(base>>5)&1u;
        #pragma unroll
        for(unsigned int j=0;j<4;++j){unsigned int col=first_col+(lane>>3)+4u*j,word=0;
            if(col<outputs){unsigned long long g=w.group(col,base/64u);
                if(sublane<4u)word=w.code_words(g)[half*4u+sublane];
                else if(BITS==5&&sublane==4u)word=w.high_words(g)[half];
                else if(sublane==5u)word=w.scales[g];}
            next.words[j]=word;}
    }
    template<int P>
    static __device__ __forceinline__ void stage(__nv_bfloat16* hi,__nv_bfloat16* lo,const Compact& next,unsigned int lane){
        unsigned int sublane=lane&7u;
        #pragma unroll
        for(unsigned int j=0;j<4;++j){unsigned int col=(lane>>3)+4u*j;
            float scale=q45::scale_to_float((unsigned short)__shfl_sync(0xffffffffu,next.words[j],5,8));
            unsigned int high=__shfl_sync(0xffffffffu,next.words[j],4,8);
            #pragma unroll
            for(unsigned int h=0;h<2;++h){
                unsigned int code=__shfl_sync(0xffffffffu,next.words[j],h*2u+(sublane>>2),8);
                unsigned int pair=(code>>((sublane&3u)*8u))&0xffu;
                if(BITS==5)pair|=((high>>(h*16u+sublane*2u))&3u)<<8;
                q45::stage_split_pair<BITS,P>(hi,lo,col*32u+h*16u+sublane*2u,pair,scale);}
        }
    }
    template<int P>
    static __device__ __forceinline__ void produce(__nv_bfloat16* hi,__nv_bfloat16* lo,const Layout& w,unsigned int outputs,unsigned int col,unsigned int base,unsigned int lane){
        Compact now;prefetch(now,w,outputs,col,base,lane);stage<P>(hi,lo,now,lane);}
};
}
