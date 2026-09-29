#pragma once
#include "tiles.cuh"
namespace qwen_ffn_down {
using namespace k32_probe;
template<int F,int N>
static __device__ __forceinline__ void run(const unsigned short* x,const unsigned char* w,unsigned short* y,unsigned int rows,unsigned int width,unsigned int outputs,unsigned long long scale,float* state,unsigned int start,unsigned int count,qwen_ffn_tiles::Storage<F,N>& s){
    unsigned int lane=threadIdx.x&31u,warp=threadIdx.x>>5,mb=warp>>1,nb=warp&1u;
    bool owns_a=warp==0u||warp==3u;unsigned int branch=owns_a?mb:2u+nb;
    unsigned int tiles=(outputs+2u*N-1u)/(2u*N),row0=(blockIdx.x/tiles)*(32u*F),col0=(blockIdx.x%tiles)*(2u*N);
    q3::Layout layout(w,width,scale);float acc[F][N/8][4]={};
    if(start){
        #pragma unroll
        for(int m=0;m<F;++m)
            #pragma unroll
            for(int h=0;h<N/8;++h)
                #pragma unroll
                for(int i=0;i<4;++i){unsigned int row=row0+mb*16u+m*32u+lane/4u+8u*(i>>1),col=col0+nb*N+h*8u+2u*(lane%4u)+(i&1u);
                    if(row<rows&&col<outputs)acc[m][h][i]=state[(unsigned long long)row*outputs+col];}
    }
    if(threadIdx.x<8u){unsigned int b=threadIdx.x>>1,t=threadIdx.x&1u;init(&s.ready[b][t],32u);init(&s.release[b][t],64u);}
    __syncthreads();
    CompactB next[N/16];unsigned int generations=count/32u;
    if(generations){
        if(owns_a)qwen_ffn_tiles::activation<F>(s.a[0][mb],x,rows,count,row0,mb,0,lane);
        else {
            #pragma unroll
            for(int t=0;t<N/16;++t)produce_b(s.b[0][nb][0]+t*16u*32u,s.b[0][nb][1]+t*16u*32u,layout,outputs,col0+nb*N+t*16u,start,lane);
        }
        arrive(&s.ready[branch][0]);
    }
    for(unsigned int gen=0;gen<generations;++gen){unsigned int slot=gen&1u;
        wait(&s.ready[mb][slot],(gen>>1)&1u);wait(&s.ready[2u+nb][slot],(gen>>1)&1u);
        #pragma unroll
        for(unsigned int half=0;half<2;++half){
            unsigned int af[F][4],bf[2][N/16][4];
            #pragma unroll
            for(int m=0;m<F;++m)q3::ldmatrix_x4(af[m],s.a[slot][mb]+(m*16u+(lane&15u))*32u+half*16u+(lane>>4)*8u);
            #pragma unroll
            for(int t=0;t<N/16;++t){unsigned int index=(t*16u+(lane&7u)+((lane>>4)<<3))*32u+half*16u+((lane>>3)&1u)*8u;
                q3::ldmatrix_x4(bf[0][t],s.b[slot][nb][0]+index);q3::ldmatrix_x4(bf[1][t],s.b[slot][nb][1]+index);}
            if(half==1u){__syncwarp();arrive(&s.release[mb][slot]);arrive(&s.release[2u+nb][slot]);}
            #pragma unroll
            for(int part=0;part<2;++part)
                #pragma unroll
                for(int m=0;m<F;++m)
                    #pragma unroll
                    for(int h=0;h<N/8;++h)q3::mma_16816(acc[m][h],af[m],bf[part][h/2][(h&1)*2],bf[part][h/2][(h&1)*2+1]);
            if(!owns_a&&half==0u&&gen+1u<generations){
                #pragma unroll
                for(int t=0;t<N/16;++t)prefetch_compact_b(next[t],layout,outputs,col0+nb*N+t*16u,start+(gen+1u)*32u,lane);
            }
        }
        if(gen+1u<generations){unsigned int n=gen+1u;if(n>=2u)wait(&s.release[branch][n&1u],((n-2u)>>1)&1u);
            if(owns_a)qwen_ffn_tiles::activation<F>(s.a[n&1u][mb],x,rows,count,row0,mb,n*32u,lane);
            else {
                #pragma unroll
                for(int t=0;t<N/16;++t)stage_compact_b(s.b[n&1u][nb][0]+t*16u*32u,s.b[n&1u][nb][1]+t*16u*32u,next[t],lane);
            }
            arrive(&s.ready[branch][n&1u]);
        }
    }
    __syncthreads();
    if(threadIdx.x<8u){unsigned int b=threadIdx.x>>1,t=threadIdx.x&1u;invalidate(&s.ready[b][t]);invalidate(&s.release[b][t]);}
    #pragma unroll
    for(int m=0;m<F;++m)
        #pragma unroll
        for(int h=0;h<N/8;++h)
            #pragma unroll
            for(int i=0;i<4;++i){unsigned int row=row0+mb*16u+m*32u+lane/4u+8u*(i>>1),col=col0+nb*N+h*8u+2u*(lane%4u)+(i&1u);
                if(row<rows&&col<outputs){
                    if(state)state[(unsigned long long)row*outputs+col]=acc[m][h][i];
                    if(start+count==width)q3::write_bf16(y,row,col,outputs,acc[m][h][i]);
                }
            }
}
}
extern "C" __global__ __launch_bounds__(128) void euhedral_q3_ffn_down_128x64(
 const unsigned short* x,const unsigned char* w,unsigned short* y,unsigned int m,unsigned int k,unsigned int n,unsigned long long scale) {
 __shared__ qwen_ffn_tiles::Storage<4,32> s;qwen_ffn_down::run<4,32>(x,w,y,m,k,n,scale,nullptr,0,k,s);
}
