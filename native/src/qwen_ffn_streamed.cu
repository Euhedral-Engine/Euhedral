#include "ffn/gate_up.cuh"
#include "ffn/down.cuh"
// Separate entrypoints keep small-row register/shared budgets independent of the wide leaf.
#define FFN_REGION(F, GATE, DOWN) \
extern "C" __global__ __launch_bounds__(128) void GATE( \
 const unsigned short* x,const unsigned char* w,unsigned short* y,unsigned int m,unsigned int k,unsigned int n,unsigned long long scale,unsigned int start,unsigned int count) { \
 __shared__ qwen_ffn_tiles::Storage<F,32> s;qwen_ffn_paired::run<F,32>(x,w,y,m,k,n,scale,s,start,count); } \
extern "C" __global__ __launch_bounds__(128) void DOWN( \
 const unsigned short* x,const unsigned char* w,unsigned short* y,unsigned int m,unsigned int k,unsigned int n,unsigned long long scale,float* state,unsigned int start,unsigned int count) { \
 __shared__ qwen_ffn_tiles::Storage<F,32> s;qwen_ffn_down::run<F,32>(x,w,y,m,k,n,scale,state,start,count,s); }
FFN_REGION(4, stream_gate_up_128x32, stream_down_128x64)
