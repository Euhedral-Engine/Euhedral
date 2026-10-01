#pragma once
#include "tiles.cuh"
namespace qwen_ffn_paired {
using namespace qwen_ffn_tiles;
template<int F,int N,int P=1>
static __device__ __forceinline__ void run(const unsigned short* x,const unsigned char* w,unsigned short* y,unsigned int rows,unsigned int width,unsigned int outputs,unsigned long long scale,Storage<F,N,P>& s,unsigned int start,unsigned int count){
    unsigned int lane=threadIdx.x&31u,warp=threadIdx.x>>5,mb=warp>>1,fh=warp&1u;
    bool owns_a=warp==0u||warp==3u;unsigned int nb=warp==1u?0u:1u,branch=owns_a?mb:2u+nb;
    unsigned int tiles=count/N,row0=(blockIdx.x/tiles)*(32u*F),col0=start+(blockIdx.x%tiles)*N;
    q3::Layout layout(w,width,scale);float acc[F][2][N/16][4]={};
    if(threadIdx.x<8u){unsigned int b=threadIdx.x>>1,t=threadIdx.x&1u;init(&s.ready[b][t],32u);init(&s.release[b][t],b<2u?64u:128u);}
    __syncthreads();
    CompactB next[N/16];unsigned int generations=width/32u;
    if(generations){
        if(owns_a)activation<F>(s.a[0][mb],x,rows,width,row0,mb,0,lane);
        else {
            #pragma unroll
            for(int t=0;t<N/16;++t)produce_b<false,P>(s.b[0][nb][0]+t*16u*32u,s.b[0][nb][P-1]+t*16u*32u,layout,outputs,col0+t*16u+nb*(outputs/2u),0,lane);
        }
        arrive(&s.ready[branch][0]);
    }
    for(unsigned int gen=0;gen<generations;++gen){unsigned int slot=gen&1u;
        wait(&s.ready[mb][slot],(gen>>1)&1u);wait(&s.ready[2][slot],(gen>>1)&1u);wait(&s.ready[3][slot],(gen>>1)&1u);
        #pragma unroll
        for(unsigned int half=0;half<2;++half){
            unsigned int af[F][4],bf[2][2][N/8];
            #pragma unroll
            for(int m=0;m<F;++m)q3::ldmatrix_x4(af[m],s.a[slot][mb]+(m*16u+(lane&15u))*32u+half*16u+(lane>>4)*8u);
            unsigned int index=(fh*(N/2u)+(lane&7u)+(N==32?((lane>>4)<<3):0u))*32u+half*16u+((lane>>3)&1u)*8u;
            #pragma unroll
            for(int pair=0;pair<2;++pair)
                #pragma unroll
                for(int part=0;part<P;++part){
                    unsigned int address=static_cast<unsigned int>(__cvta_generic_to_shared(s.b[slot][pair][part]+index));
                    if(N==16)asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];":"=r"(bf[pair][part][0]),"=r"(bf[pair][part][1]):"r"(address));
                    else asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];":"=r"(bf[pair][part][0]),"=r"(bf[pair][part][1]),"=r"(bf[pair][part][2]),"=r"(bf[pair][part][3]):"r"(address));
                }
            if(half==1u){__syncwarp();arrive(&s.release[mb][slot]);arrive(&s.release[2][slot]);arrive(&s.release[3][slot]);}
            #pragma unroll
            for(int part=0;part<P;++part)
                #pragma unroll
                for(int m=0;m<F;++m)
                    #pragma unroll
                    for(int pair=0;pair<2;++pair)
                        #pragma unroll
                        for(int h=0;h<N/16;++h)q3::mma_16816(acc[m][pair][h],af[m],bf[pair][part][h*2],bf[pair][part][h*2+1]);
            if(!owns_a&&half==0u&&gen+1u<generations){
                #pragma unroll
                for(int t=0;t<N/16;++t)prefetch_compact_b(next[t],layout,outputs,col0+t*16u+nb*(outputs/2u),(gen+1u)*32u,lane);
            }
        }
        if(gen+1u<generations){unsigned int n=gen+1u;if(n>=2u)wait(&s.release[branch][n&1u],((n-2u)>>1)&1u);
            if(owns_a)activation<F>(s.a[n&1u][mb],x,rows,width,row0,mb,n*32u,lane);
            else {
                #pragma unroll
                for(int t=0;t<N/16;++t)stage_compact_b<false,P>(s.b[n&1u][nb][0]+t*16u*32u,s.b[n&1u][nb][P-1]+t*16u*32u,next[t],lane);
            }
            arrive(&s.ready[branch][n&1u]);
        }
    }
    __syncthreads();
    if(threadIdx.x<8u){unsigned int b=threadIdx.x>>1,t=threadIdx.x&1u;invalidate(&s.ready[b][t]);invalidate(&s.release[b][t]);}
    #pragma unroll
    for(int m=0;m<F;++m)
        #pragma unroll
        for(int h=0;h<N/16;++h)
            #pragma unroll
            for(int i=0;i<4;++i){
                unsigned int row=row0+mb*16u+m*32u+lane/4u+8u*(i>>1),col=col0+fh*(N/2u)+h*8u+2u*(lane%4u)+(i&1u);
                if(row<rows&&col<outputs/2u){float gate=q3::bf16_to_float(q3::float_to_bf16(acc[m][0][h][i])),up=q3::bf16_to_float(q3::float_to_bf16(acc[m][1][h][i]));
                    y[(unsigned long long)row*count+col-start]=__bfloat16_as_ushort(__float2bfloat16_rn((gate/(1.0f+expf(-gate)))*up));}
            }
}
}
// Relaxed (hi-only) leaves and their exact hi/lo twins, selected by host dispatch.
#define EUHEDRAL_Q3_GATE_UP(NAME, F, P) \
extern "C" __global__ __launch_bounds__(128) void NAME( \
 const unsigned short* x,const unsigned char* w,unsigned short* y,unsigned int m,unsigned int k,unsigned int n,unsigned long long scale) { \
 __shared__ qwen_ffn_tiles::Storage<F,32,P> s;qwen_ffn_paired::run<F,32,P>(x,w,y,m,k,n,scale,s,0,n/2u); }
EUHEDRAL_Q3_GATE_UP(euhedral_q3_gate_up_swiglu_64x32, 2, 1)
EUHEDRAL_Q3_GATE_UP(euhedral_q3_gate_up_swiglu_128x32, 4, 1)
EUHEDRAL_Q3_GATE_UP(euhedral_q3_gate_up_swiglu_64x32_exact, 2, 2)
EUHEDRAL_Q3_GATE_UP(euhedral_q3_gate_up_swiglu_128x32_exact, 4, 2)
#undef EUHEDRAL_Q3_GATE_UP
