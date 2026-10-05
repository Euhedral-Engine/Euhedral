// Lane-interleaved static rANS for NVFP4 blocks and BF16 high bytes (format in docs/NVFP4_LOSSLESS.md).
//
// A segment is coded by L lanes; lane l owns blocks l, l + L, l + 2L, ... (the segment's last block repeats to fill
// the final row). Per block the decoder reads: the block's E4M3 scale, then for each of the 16 E2M1 codes its
// magnitude and, when the magnitude is nonzero (or the segment codes all signs), its sign as a 1-bit symbol.
//
// rANS state: 64 bits in [2^32, 2^48), 16 probability bits, 16-bit renormalisation words (at most one per symbol).
// A lane's stream is [renormalisation words in emission order][state low, middle, high]; the decoder reads it from
// the back. The encoder works backwards, so decode order is block order.
typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned char u8;

// Histograms for the tables, and two segment properties: the largest scale byte and whether any code is -0.
extern "C" __global__ void count_blocks(const u64* packed, const u8* scale, u32 B, u32 base_block, u32 bpe, u32 G,
                                        u32 shift, u32* cnt_scale, u32* cnt_mag, u32 CM, u32* stats) {
  extern __shared__ u32 hist[];
  for (u32 i = threadIdx.x; i < CM * 8; i += blockDim.x) hist[i] = 0;
  __syncthreads();
  u32 max_scale = 0, neg_zero = 0;
  for (u64 b = (u64)blockIdx.x * blockDim.x + threadIdx.x; b < B; b += (u64)gridDim.x * blockDim.x) {
    u64 pk = packed[b];
    u32 s = scale[b];
    max_scale = max(max_scale, s);
    u32 e = (u32)((base_block + b) / bpe);
    if (e >= G) e = G - 1;
    atomicAdd(&cnt_scale[e * 128 + (s & 127)], 1u);
    u32 seen = 0, base = ((s & 127) >> shift) * 32;
#pragma unroll
    for (int p = 0; p < 16; ++p) {
      u32 nib = (u32)(pk >> (4 * p)) & 15, mag = nib & 7;
      neg_zero |= (nib == 8);
      atomicAdd(&hist[(base + seen * 16 + p) * 8 + mag], 1u);
      seen |= (mag == 7);
    }
  }
  __syncthreads();
  for (u32 i = threadIdx.x; i < CM * 8; i += blockDim.x)
    if (hist[i]) atomicAdd(&cnt_mag[i], hist[i]);
  atomicMax(&stats[0], max_scale);
  if (neg_zero) atomicOr(&stats[1], 1u);
}

// Reads the lane's stream backwards; a corrupt stream that runs out reads zeros instead of leaving its buffer.
__device__ __forceinline__ u32 rd(const u16* wl, u32& pos) { return pos ? (u32)wl[--pos] : 0u; }

__device__ __forceinline__ void enc_sym(u64& x, u32 f, u32 c, u16* w, u32& pos, u32 L, u32 lane, u32 cap) {
  if (x >= ((u64)f << 32)) {
    if (pos < cap) w[(u64)pos * L + lane] = (u16)x;
    ++pos;
    x >>= 16;
  }
  u64 q = (u64)((double)x / (double)f);  // exact: x < 2^48 and f < 2^16
  x = (q << 16) + (x - q * f) + c;
}

__device__ __forceinline__ void enc_flush(u64 x, u16* w, u32& pos, u32 L, u32 lane, u32 cap, u32* lens, u32* flag) {
  if (pos < cap) w[(u64)pos * L + lane] = (u16)x;
  ++pos;
  if (pos < cap) w[(u64)pos * L + lane] = (u16)(x >> 16);
  ++pos;
  if (pos < cap) w[(u64)pos * L + lane] = (u16)(x >> 32);
  ++pos;
  lens[lane] = pos;
  if (pos > cap) atomicOr(flag, 1u);  // capacity too small: the host retries with a larger one
}

extern "C" __global__ void enc_blocks(const u64* packed, const u8* scale, u32 B, u32 base_block, u32 L, u32 J, u32 bpe,
                                      u32 G, u32 shift, u32 sign_all, u32 cap, const u16* freqS, const u16* cumS,
                                      const u16* freqM, const u16* cumM, u32 CM, u16* words, u32* lens, u32* flag) {
  extern __shared__ u16 sm[];
  u16* sf = sm;
  u16* scm = sm + CM * 8;
  for (u32 i = threadIdx.x; i < CM * 8; i += blockDim.x) {
    sf[i] = freqM[i];
    scm[i] = cumM[i];
  }
  __syncthreads();
  u32 lane = blockIdx.x * blockDim.x + threadIdx.x;
  if (lane >= L) return;
  u64 x = 1ull << 32;
  u32 pos = 0;
  for (int j = (int)J - 1; j >= 0; --j) {
    u64 b = (u64)j * L + lane;
    if (b >= B) b = B - 1;
    u64 pk = packed[b];
    u32 s = scale[b];
    u32 e = (u32)((base_block + b) / bpe);
    if (e >= G) e = G - 1;
    u32 ctx[16], seen = 0, base = (s >> shift) * 32;
#pragma unroll
    for (int p = 0; p < 16; ++p) {
      u32 mag = (u32)(pk >> (4 * p)) & 7;
      ctx[p] = base + seen * 16 + p;
      seen |= (mag == 7);
    }
#pragma unroll
    for (int p = 15; p >= 0; --p) {
      u32 nib = (u32)(pk >> (4 * p)) & 15, mag = nib & 7;
      if (mag || sign_all) {
        if (x >= (1ull << 47)) {
          if (pos < cap) words[(u64)pos * L + lane] = (u16)x;
          ++pos;
          x >>= 16;
        }
        x = ((x >> 15) << 16) + (x & 32767) + ((nib >> 3) ? 32768u : 0u);
      }
      u32 idx = ctx[p] * 8 + mag;
      enc_sym(x, sf[idx], scm[idx], words, pos, L, lane, cap);
    }
    enc_sym(x, freqS[e * 128 + s], cumS[e * 128 + s], words, pos, L, lane, cap);
  }
  enc_flush(x, words, pos, L, lane, cap, lens, flag);
}

extern "C" __global__ void dec_blocks(u64* packed, u8* scale, u32 B, u32 base_block, u32 L, u32 J, u32 bpe, u32 G,
                                      u32 shift, u32 sign_all, const u16* words, const u32* lens, const u64* offs,
                                      const u8* s2s, const u16* freqS, const u16* cumS, const u16* freqM,
                                      const u32* cumM, u32 CM) {
  extern __shared__ u32 smd[];
  u32* scm = smd;
  u16* sf = (u16*)(smd + CM * 8);
  for (u32 i = threadIdx.x; i < CM * 8; i += blockDim.x) {
    sf[i] = freqM[i];
    scm[i] = cumM[i];
  }
  __syncthreads();
  u32 lane = blockIdx.x * blockDim.x + threadIdx.x;
  if (lane >= L) return;
  u32 pos = lens[lane];
  const u16* wl = words + offs[lane];
  u64 x = ((u64)rd(wl, pos) << 32);
  x |= ((u64)rd(wl, pos) << 16);
  x |= (u64)rd(wl, pos);
  for (u32 j = 0; j < J; ++j) {
    u64 b = (u64)j * L + lane;
    u64 bc = b < B ? b : B - 1;
    u32 e = (u32)((base_block + bc) / bpe);
    if (e >= G) e = G - 1;
    u32 slot = (u32)(x & 0xFFFF);
    u32 s = s2s[(u64)e * 65536 + slot];
    u32 idx = e * 128 + s;
    x = (u64)freqS[idx] * (x >> 16) + slot - cumS[idx];
    if (x < (1ull << 32)) x = (x << 16) | rd(wl, pos);
    u64 pk = 0;
    u32 seen = 0, base = (s >> shift) * 32;
#pragma unroll
    for (int p = 0; p < 16; ++p) {
      u32 ctx = base + seen * 16 + p;
      slot = (u32)(x & 0xFFFF);
      const u32* cm = scm + ctx * 8;
      // the symbol is the number of cumulative bounds at or below the slot (bounds of absent symbols repeat)
      u32 sym = (slot >= cm[1]) + (slot >= cm[2]) + (slot >= cm[3]) + (slot >= cm[4]) + (slot >= cm[5]) +
                (slot >= cm[6]) + (slot >= cm[7]);
      x = (u64)sf[ctx * 8 + sym] * (x >> 16) + slot - cm[sym];
      if (x < (1ull << 32)) x = (x << 16) | rd(wl, pos);
      u32 nib = sym;
      if (sym || sign_all) {
        slot = (u32)(x & 0xFFFF);
        x = ((x >> 16) << 15) + (slot & 32767);
        if (x < (1ull << 32)) x = (x << 16) | rd(wl, pos);
        nib |= (slot >> 15) << 3;
      }
      pk |= (u64)nib << (4 * p);
      seen |= (sym == 7);
    }
    if (b < B) {
      packed[b] = pk;
      scale[b] = (u8)s;
    }
  }
}

// Concatenates the lanes' words (written lane-interleaved by the encoder) lane after lane.
extern "C" __global__ void compact_lanes(const u16* words, const u32* lens, const u64* offs, u32 L, u16* out) {
  u32 lane = blockIdx.x * blockDim.x + threadIdx.x;
  if (lane >= L) return;
  u32 n = lens[lane];
  u16* o = out + offs[lane];
  for (u32 i = 0; i < n; ++i) o[i] = words[(u64)i * L + lane];
}

extern "C" __global__ void enc_bytes(const u8* data, u32 n, u32 L, u32 J, u32 cap, const u16* freq, const u16* cum,
                                     u16* words, u32* lens, u32* flag) {
  u32 lane = blockIdx.x * blockDim.x + threadIdx.x;
  if (lane >= L) return;
  u64 x = 1ull << 32;
  u32 pos = 0;
  for (int j = (int)J - 1; j >= 0; --j) {
    u64 i = (u64)j * L + lane;
    if (i >= n) i = n - 1;
    u32 s = data[i];
    enc_sym(x, freq[s], cum[s], words, pos, L, lane, cap);
  }
  enc_flush(x, words, pos, L, lane, cap, lens, flag);
}

extern "C" __global__ void dec_bytes(u8* out, u32 n, u32 L, u32 J, const u16* words, const u32* lens, const u64* offs,
                                     const u8* s2s, const u16* freq, const u16* cum) {
  u32 lane = blockIdx.x * blockDim.x + threadIdx.x;
  if (lane >= L) return;
  u32 pos = lens[lane];
  const u16* wl = words + offs[lane];
  u64 x = ((u64)rd(wl, pos) << 32);
  x |= ((u64)rd(wl, pos) << 16);
  x |= (u64)rd(wl, pos);
  for (u32 j = 0; j < J; ++j) {
    u32 slot = (u32)(x & 0xFFFF);
    u32 s = s2s[slot];
    x = (u64)freq[s] * (x >> 16) + slot - cum[s];
    if (x < (1ull << 32)) x = (x << 16) | rd(wl, pos);
    u64 i = (u64)j * L + lane;
    if (i < n) out[i] = (u8)s;
  }
}
