#pragma once
#include "tiles.cuh"
#include "formats.cuh"
#include "balanced.cuh"
namespace qwen_ffn_down {
using namespace k32_probe;
// SPLIT: the CTA accumulates K range [start, start + count) of the full activation (row stride
// width) from zero and writes FP32 partial sums to `state`; the reduction kernel adds the splits
// and rounds to BF16. Otherwise x is a region of `count` features continuing `state` (streamed FFN)
// or the whole K (start 0, count width). B selects the weight format (formats.cuh).
template<int F,int N,bool SPLIT=false,int P=1,class B=qwen_ffn_tiles::Q3B>
static __device__ __forceinline__ void run(const unsigned short* x,const unsigned char* w,unsigned short* y,unsigned int rows,unsigned int width,unsigned int outputs,unsigned long long scale,float* state,unsigned int start,unsigned int count,qwen_ffn_tiles::Storage<F,N,P>& s){
    unsigned int lane=threadIdx.x&31u,warp=threadIdx.x>>5,mb=warp>>1,nb=warp&1u;
    bool owns_a=warp==0u||warp==3u;unsigned int branch=owns_a?mb:2u+nb;
    unsigned int tiles=(outputs+2u*N-1u)/(2u*N),row0=(blockIdx.x/tiles)*(32u*F),col0=(blockIdx.x%tiles)*(2u*N);
    const typename B::Layout layout=B::layout(w,width,outputs,scale);float acc[F][N/8][4]={};
    if(!SPLIT&&start){
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
    typename B::Compact next[N/16];unsigned int generations=count/32u;
    if(generations){
        if(owns_a)qwen_ffn_tiles::activation<F>(s.a[0][mb],x,rows,SPLIT?width:count,row0,mb,SPLIT?start:0u,lane);
        else {
            #pragma unroll
            for(int t=0;t<N/16;++t)B::template produce<P>(s.b[0][nb][0]+t*16u*32u,s.b[0][nb][P-1]+t*16u*32u,layout,outputs,col0+nb*N+t*16u,start,lane);
        }
        arrive(&s.ready[branch][0]);
    }
    for(unsigned int gen=0;gen<generations;++gen){unsigned int slot=gen&1u;
        wait(&s.ready[mb][slot],(gen>>1)&1u);wait(&s.ready[2u+nb][slot],(gen>>1)&1u);
        #pragma unroll
        for(unsigned int half=0;half<2;++half){
            unsigned int af[F][4],bf[2][N/16][4];
            #pragma unroll
            for(int m=0;m<F;++m)q3::ldmatrix_x4(af[m],s.a[slot][mb]+b_index(m*16u+(lane&15u),half*16u+(lane>>4)*8u));
            #pragma unroll
            for(int t=0;t<N/16;++t){unsigned int index=b_index(t*16u+(lane&7u)+((lane>>4)<<3),half*16u+((lane>>3)&1u)*8u);
                q3::ldmatrix_x4(bf[0][t],s.b[slot][nb][0]+index);if(P>1)q3::ldmatrix_x4(bf[1][t],s.b[slot][nb][P-1]+index);}
            if(half==1u){__syncwarp();arrive(&s.release[mb][slot]);arrive(&s.release[2u+nb][slot]);}
            #pragma unroll
            for(int part=0;part<P;++part)
                #pragma unroll
                for(int m=0;m<F;++m)
                    #pragma unroll
                    for(int h=0;h<N/8;++h)q3::mma_16816(acc[m][h],af[m],bf[part][h/2][(h&1)*2],bf[part][h/2][(h&1)*2+1]);
            if(!owns_a&&half==0u&&gen+1u<generations){
                #pragma unroll
                for(int t=0;t<N/16;++t)B::prefetch(next[t],layout,outputs,col0+nb*N+t*16u,start+(gen+1u)*32u,lane);
            }
        }
        if(gen+1u<generations){unsigned int n=gen+1u;if(n>=2u)wait(&s.release[branch][n&1u],((n-2u)>>1)&1u);
            if(owns_a)qwen_ffn_tiles::activation<F>(s.a[n&1u][mb],x,rows,SPLIT?width:count,row0,mb,(SPLIT?start:0u)+n*32u,lane);
            else {
                #pragma unroll
                for(int t=0;t<N/16;++t)B::template stage<P>(s.b[n&1u][nb][0]+t*16u*32u,s.b[n&1u][nb][P-1]+t*16u*32u,next[t],lane);
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
                    if(SPLIT)state[(unsigned long long)row*outputs+col]=acc[m][h][i];
                    else{
                        if(state)state[(unsigned long long)row*outputs+col]=acc[m][h][i];
                        if(start+count==width)q3::write_bf16(y,row,col,outputs,acc[m][h][i]);
                    }
                }
            }
}
}
// Relaxed (hi-only) leaves and their exact hi/lo twins, selected by host dispatch.
#define EUHEDRAL_Q3_FFN_DOWN(NAME, F, P) \
extern "C" __global__ __launch_bounds__(128) void NAME( \
 const unsigned short* x,const unsigned char* w,unsigned short* y,unsigned int m,unsigned int k,unsigned int n,unsigned long long scale) { \
 __shared__ qwen_ffn_tiles::Storage<F,32,P> s;qwen_ffn_down::run<F,32,false,P>(x,w,y,m,k,n,scale,nullptr,0,k,s); }
EUHEDRAL_Q3_FFN_DOWN(euhedral_q3_ffn_down_64x64, 2, 1)
EUHEDRAL_Q3_FFN_DOWN(euhedral_q3_ffn_down_128x64, 4, 1)
EUHEDRAL_Q3_FFN_DOWN(euhedral_q3_ffn_down_64x64_exact, 2, 2)
EUHEDRAL_Q3_FFN_DOWN(euhedral_q3_ffn_down_128x64_exact, 4, 2)
#undef EUHEDRAL_Q3_FFN_DOWN
// Split-K down: blockIdx.y selects a 32-aligned K range; partial[split] (rows x n FP32) receives its
// sums. The 80-320 CTAs of the unsplit leaf leave a 64-512 row quantum in one or two partial waves.
#define EUHEDRAL_Q3_FFN_DOWN_SPLIT(NAME, F) \
extern "C" __global__ __launch_bounds__(128) void NAME( \
 const unsigned short* x,const unsigned char* w,float* partial,unsigned int m,unsigned int k,unsigned int n,unsigned long long scale,unsigned int splits) { \
 __shared__ qwen_ffn_tiles::Storage<F,32> s; \
 unsigned int per=(k/32u+splits-1u)/splits*32u,start=blockIdx.y*per; \
 if(start>=k)return; \
 qwen_ffn_down::run<F,32,true,1>(x,w,nullptr,m,k,n,scale,partial+(unsigned long long)blockIdx.y*m*n,start,min(per,k-start),s); }
EUHEDRAL_Q3_FFN_DOWN_SPLIT(euhedral_q3_ffn_down_split_64x64, 2)
#undef EUHEDRAL_Q3_FFN_DOWN_SPLIT
// The 128-row split leaf runs on the balanced engine (ffn/balanced.cuh): same K ranges and K16 order,
// bitwise equal partials. 512 rows, four splits 1753 -> 1455 us; 256 rows, three splits 727 us.
extern "C" __global__ __launch_bounds__(128) void euhedral_q3_ffn_down_split_128x64(
 const unsigned short* x,const unsigned char* w,float* partial,unsigned int m,unsigned int k,unsigned int n,unsigned long long scale,unsigned int splits) {
 __shared__ balanced::Storage s;
 unsigned int per=(k/32u+splits-1u)/splits*32u,start=blockIdx.y*per;
 if(start>=k)return;
 balanced::run<qwen_ffn_tiles::Q3B>(x,w,nullptr,m,k,n,scale,s,start,min(per,k-start),partial+(unsigned long long)blockIdx.y*m*n); }
// Adds the split partials of each output in split order and rounds once to BF16.
extern "C" __global__ __launch_bounds__(256) void euhedral_q3_ffn_down_reduce(
 const float* partial,unsigned short* y,unsigned int count,unsigned int splits) {
 unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i>=count)return;
 float sum=partial[i];
 for(unsigned int split=1;split<splits;split++)sum+=partial[(unsigned long long)split*count+i];
 y[i]=q3::float_to_bf16(sum);
}
