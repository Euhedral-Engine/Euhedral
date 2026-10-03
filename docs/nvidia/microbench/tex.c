// tex: texture units as an extra load pipe and as a hardware dequantizer.
#include "common.h"

#include <math.h>

static CUmodule mod(void) {
    static CUmodule m;
    if (!m) m = load_module("tex.cu", "sm_120a", NULL, 0, 0);
    return m;
}

static CUtexObject lin_tex(CUdeviceptr p, size_t bytes, CUarray_format fmt, unsigned ch, int as_int) {
    CUDA_RESOURCE_DESC rd;
    memset(&rd, 0, sizeof rd);
    rd.resType = CU_RESOURCE_TYPE_LINEAR;
    rd.res.linear.devPtr = p;
    rd.res.linear.format = fmt;
    rd.res.linear.numChannels = ch;
    rd.res.linear.sizeInBytes = bytes;
    CUDA_TEXTURE_DESC td;
    memset(&td, 0, sizeof td);
    td.filterMode = CU_TR_FILTER_MODE_POINT;
    td.flags = as_int ? CU_TRSF_READ_AS_INTEGER : 0;
    CUtexObject t;
    CK(cuTexObjectCreate(&t, &rd, &td, NULL));
    return t;
}

static unsigned rng(unsigned *s) {
    *s ^= *s << 13;
    *s ^= *s >> 17;
    *s ^= *s << 5;
    return *s;
}

// ---- 1. TEX as an additional load pipe ---------------------------------------------
static void lin_pipes(void) {
    int maxw;
    CK(cuDeviceGetAttribute(&maxw, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH, g_dev));
    printf("\n## linear-memory loads: LDG vs TEX vs half/half (max 1D linear texels %d)\n", maxw);
    printf("%-10s %-8s %8s %10s %8s %12s\n", "kernel", "set", "MiB", "GB/s", "MHz", "words/clk/SM");
    CUdeviceptr out, clk;
    CK(cuMemAlloc(&out, 64));
    CK(cuMemAlloc(&clk, 64));
    size_t big = (size_t)1 << 30;
    CUdeviceptr buf;
    CK(cuMemAlloc(&buf, big));
    CK(cuMemsetD32(buf, 3, big / 4));
    struct {
        const char *set;
        size_t bytes;
        int reps;
    } sets[] = {{"L2-hot", 16u << 20, 64}, {"DRAM", big, 2}};
    for (int s = 0; s < 2; s++) {
        size_t bytes = sets[s].bytes;
        int reps = sets[s].reps;
        // 32-bit words
        size_t n32 = bytes / 4;
        if (n32 > (size_t)maxw) n32 = maxw;
        CUtexObject t32 = lin_tex(buf, n32 * 4, CU_AD_FORMAT_UNSIGNED_INT32, 1, 1);
        size_t n4 = bytes / 16;
        if (n4 > (size_t)maxw) n4 = maxw;
        CUtexObject t128 = lin_tex(buf, n4 * 16, CU_AD_FORMAT_UNSIGNED_INT32, 4, 1);
        struct {
            const char *k;
            int kind; // 0 ldg, 1 tex, 2 mix
            int wide;
        } ks[] = {{"ldg_u32", 0, 0}, {"tex_u32", 1, 0}, {"mix_u32", 2, 0},
                  {"ldg_v4", 0, 1},  {"tex_v4", 1, 1},  {"mix_v4", 2, 1}};
        for (int k = 0; k < 6; k++) {
            CUfunction f = get_fn(mod(), ks[k].k);
            size_t n = ks[k].wide ? n4 : n32;
            CUtexObject t = ks[k].wide ? t128 : t32;
            void *a_ldg[] = {&buf, &n, &reps, &out, &clk};
            void *a_tex[] = {&t, &n, &reps, &out, &clk};
            void *a_mix[] = {&buf, &t, &n, &reps, &out, &clk};
            void **args = ks[k].kind == 0 ? a_ldg : ks[k].kind == 1 ? a_tex : a_mix;
            double best = 1e30;
            int bps[] = {2, 4, 6};
            for (int b = 0; b < 3; b++) {
                double mn, md;
                time_launch(f, L1D(g_sms * bps[b], 256, 0), args, 5, 0, &mn, &md);
                if (mn < best) best = mn;
            }
            double mhz = clk_mhz(clk);
            double moved = (double)n * (ks[k].wide ? 16 : 4) * reps;
            double words = moved / (ks[k].wide ? 16 : 4);
            printf("%-10s %-8s %8zu %10.1f %8.0f %12.2f\n", ks[k].k, sets[s].set, bytes >> 20,
                   moved / (best * 1e-3) / 1e9, mhz, words / (best * 1e-3) / (mhz * 1e6) / g_sms);
        }
        cuTexObjectDestroy(t32);
        cuTexObjectDestroy(t128);
        // unorm8x4: 4 values per fetch converted to float by the TMU
        CUtexObject tu = lin_tex(buf, n4 * 16 / 4 * 4, CU_AD_FORMAT_UNSIGNED_INT8, 4, 0);
        size_t nt = n4 * 4 > (size_t)maxw ? (size_t)maxw : n4 * 4; // texels of 4 bytes
        CUfunction f = get_fn(mod(), "tex_unorm8x4");
        void *args[] = {&tu, &nt, &reps, &out, &clk};
        double mn, md;
        time_launch(f, L1D(g_sms * 6, 256, 0), args, 5, 0, &mn, &md);
        double mhz = clk_mhz(clk);
        printf("%-10s %-8s %8zu %10.1f %8.0f %12.2f  (u8x4 -> 4 floats; values/clk/SM %.1f)\n",
               "unorm8x4", sets[s].set, (nt * 4) >> 20, nt * 4.0 * reps / (mn * 1e-3) / 1e9, mhz,
               nt * (double)reps / (mn * 1e-3) / (mhz * 1e6) / g_sms,
               4 * nt * (double)reps / (mn * 1e-3) / (mhz * 1e6) / g_sms);
        cuTexObjectDestroy(tu);
    }
    cuMemFree(buf);
    cuMemFree(out);
    cuMemFree(clk);
}

// ---- 2. block-compressed textures --------------------------------------------------
typedef struct {
    const char *name;
    CUarray_format fmt;
    int chans, block_bytes;
} bcfmt_t;

static const bcfmt_t k_bc[] = {
    {"BC1", CU_AD_FORMAT_BC1_UNORM, 4, 8},   {"BC4u", CU_AD_FORMAT_BC4_UNORM, 1, 8},
    {"BC4s", CU_AD_FORMAT_BC4_SNORM, 1, 8},  {"BC5s", CU_AD_FORMAT_BC5_SNORM, 2, 16},
    {"BC6Hs", CU_AD_FORMAT_BC6H_SF16, 3, 16}, {"BC7", CU_AD_FORMAT_BC7_UNORM, 4, 16},
};

static CUarray bc_array(const bcfmt_t *f, int w, int h, const unsigned char *blocks) {
    CUDA_ARRAY_DESCRIPTOR d;
    memset(&d, 0, sizeof d);
    d.Width = w;
    d.Height = h;
    d.Format = f->fmt;
    d.NumChannels = f->chans;
    CUarray a;
    CK(cuArrayCreate(&a, &d));
    CUDA_MEMCPY2D c;
    memset(&c, 0, sizeof c);
    c.srcMemoryType = CU_MEMORYTYPE_HOST;
    c.srcHost = blocks;
    c.srcPitch = (size_t)(w / 4) * f->block_bytes;
    c.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    c.dstArray = a;
    c.WidthInBytes = (size_t)(w / 4) * f->block_bytes;
    c.Height = h / 4;
    CK(cuMemcpy2D(&c));
    return a;
}

static CUtexObject arr_tex(CUarray a) {
    CUDA_RESOURCE_DESC rd;
    memset(&rd, 0, sizeof rd);
    rd.resType = CU_RESOURCE_TYPE_ARRAY;
    rd.res.array.hArray = a;
    CUDA_TEXTURE_DESC td;
    memset(&td, 0, sizeof td);
    td.addressMode[0] = td.addressMode[1] = CU_TR_ADDRESS_MODE_CLAMP;
    td.filterMode = CU_TR_FILTER_MODE_POINT;
    CUtexObject t;
    CK(cuTexObjectCreate(&t, &rd, &td, NULL));
    return t;
}

// Reference BC4 decoder (D3D10 rules). Returns the exact rational value as float.
static void bc4_decode(const unsigned char *b, int snorm, float out[16]) {
    float r0, r1;
    if (snorm) {
        int a = (signed char)b[0], c = (signed char)b[1];
        if (a == -128) a = -127;
        if (c == -128) c = -127;
        r0 = a / 127.0f;
        r1 = c / 127.0f;
    } else {
        r0 = b[0] / 255.0f;
        r1 = b[1] / 255.0f;
    }
    int big = snorm ? ((signed char)b[0] > (signed char)b[1]) : (b[0] > b[1]);
    float pal[8];
    pal[0] = r0;
    pal[1] = r1;
    if (big) {
        for (int i = 1; i <= 6; i++) pal[i + 1] = ((7 - i) * r0 + i * r1) / 7.0f;
    } else {
        for (int i = 1; i <= 4; i++) pal[i + 1] = ((5 - i) * r0 + i * r1) / 5.0f;
        pal[6] = snorm ? -1.0f : 0.0f;
        pal[7] = 1.0f;
    }
    unsigned long long bits = 0;
    for (int i = 0; i < 6; i++) bits |= (unsigned long long)b[2 + i] << (8 * i);
    for (int t = 0; t < 16; t++) out[t] = pal[(bits >> (3 * t)) & 7];
}

static void bc_precision(void) {
    printf("\n## BC4 decode precision vs exact D3D10 interpolation (64x64 texels, random blocks)\n");
    int w = 64, h = 64;
    for (int sn = 0; sn < 2; sn++) {
        const bcfmt_t *f = &k_bc[1 + sn];
        size_t nb = (size_t)(w / 4) * (h / 4);
        unsigned char *blocks = malloc(nb * 8);
        unsigned s = 99 + sn;
        for (size_t i = 0; i < nb * 8; i++) blocks[i] = (unsigned char)rng(&s);
        CUarray a = bc_array(f, w, h, blocks);
        CUtexObject t = arr_tex(a);
        CUdeviceptr d, dg;
        CK(cuMemAlloc(&d, (size_t)w * h * 4));
        CK(cuMemAlloc(&dg, (size_t)w * h * 4));
        int ch = 1;
        void *args[] = {&t, &w, &h, &ch, &d};
        CK(cuLaunchKernel(get_fn(mod(), "bc_dump"), 1, h, 1, w, 1, 1, 0, 0, args, NULL));
        void *gargs[] = {&t, &w, &h, &dg};
        CK(cuLaunchKernel(get_fn(mod(), "bc_dump_gather"), 1, h / 2, 1, w / 2, 1, 1, 0, 0, gargs, NULL));
        float *hw = malloc((size_t)w * h * 4), *hg = malloc((size_t)w * h * 4);
        CK(cuMemcpyDtoH(hw, d, (size_t)w * h * 4));
        CK(cuMemcpyDtoH(hg, dg, (size_t)w * h * 4));
        double maxerr = 0, sumerr = 0;
        int exact8 = 0, n = 0, gather_mismatch = 0;
        for (int by = 0; by < h / 4; by++)
            for (int bx = 0; bx < w / 4; bx++) {
                float ref[16];
                bc4_decode(blocks + ((size_t)by * (w / 4) + bx) * 8, sn, ref);
                for (int t2 = 0; t2 < 16; t2++) {
                    int x = bx * 4 + (t2 & 3), y = by * 4 + (t2 >> 2);
                    float v = hw[(size_t)y * w + x];
                    double e = fabs((double)v - ref[t2]);
                    if (e > maxerr) maxerr = e;
                    sumerr += e;
                    n++;
                    double q = sn ? v * 127.0 : v * 255.0;
                    if (fabs(q - nearbyint(q)) < 1e-4) exact8++;
                }
            }
        // gather order check: w=(x0,y0) z=(x1,y0) x=(x0,y1) y=(x1,y1)
        for (int qy = 0; qy < h / 2; qy++)
            for (int qx = 0; qx < w / 2; qx++) {
                const float *g = hg + ((size_t)qy * (w / 2) + qx) * 4;
                float x0y0 = hw[(size_t)(2 * qy) * w + 2 * qx], x1y0 = hw[(size_t)(2 * qy) * w + 2 * qx + 1];
                float x0y1 = hw[(size_t)(2 * qy + 1) * w + 2 * qx], x1y1 = hw[(size_t)(2 * qy + 1) * w + 2 * qx + 1];
                if (g[3] != x0y0 || g[2] != x1y0 || g[0] != x0y1 || g[1] != x1y1) gather_mismatch++;
            }
        printf("%-5s max |hw-exact| = %.3g (%.2f LSB of 1/%d), mean %.3g; values on the 8-bit grid: %.1f%%; "
               "gather order mismatches: %d\n",
               f->name, maxerr, maxerr * (sn ? 127 : 255), sn ? 127 : 255, sumerr / n,
               100.0 * exact8 / n, gather_mismatch);
        if (getenv("MB_VERBOSE")) {
            float ref[16];
            bc4_decode(blocks, sn, ref);
            printf("  block0 bytes:");
            for (int i = 0; i < 8; i++) printf(" %02x", blocks[i]);
            printf("\n  ref:");
            for (int i = 0; i < 16; i++) printf(" %.5f", ref[i]);
            printf("\n  hw: ");
            for (int i = 0; i < 16; i++) printf(" %.5f", hw[(size_t)(i >> 2) * w + (i & 3)]);
            printf("\n");
        }
        free(hw);
        free(hg);
        free(blocks);
        cuMemFree(d);
        cuMemFree(dg);
        cuTexObjectDestroy(t);
        cuArrayDestroy(a);
    }
}


// Exact hardware palette: one 4x4 block whose texel t uses index t % 8, endpoints at
// the extremes, for both BC4 modes (r0 > r1: 8 levels; r0 <= r1: 6 levels + 0/1).
static void bc_palette(void) {
    printf("\n## BC4 hardware palette (index -> decoded value), endpoints at the range limits\n");
    struct {
        int fmt; // index into k_bc
        unsigned char r0, r1;
        const char *what;
    } cases[] = {{1, 255, 0, "UNORM r0=255 r1=0 (8-level)"},
                 {1, 0, 255, "UNORM r0=0 r1=255 (6-level)"},
                 {1, 200, 40, "UNORM r0=200 r1=40 (8-level)"},
                 {2, 127, 0x81, "SNORM r0=127 r1=-127 (8-level)"},
                 {2, 0x81, 127, "SNORM r0=-127 r1=127 (6-level)"}};
    for (int c = 0; c < 5; c++) {
        const bcfmt_t *f = &k_bc[cases[c].fmt];
        unsigned char blk[8] = {cases[c].r0, cases[c].r1, 0, 0, 0, 0, 0, 0};
        unsigned long long bits = 0;
        for (int t = 0; t < 16; t++) bits |= (unsigned long long)(t % 8) << (3 * t);
        for (int i = 0; i < 6; i++) blk[2 + i] = (unsigned char)(bits >> (8 * i));
        CUarray a = bc_array(f, 4, 4, blk);
        CUtexObject t = arr_tex(a);
        CUdeviceptr d;
        CK(cuMemAlloc(&d, 16 * 4));
        int w = 4, h = 4, ch = 1;
        void *args[] = {&t, &w, &h, &ch, &d};
        CK(cuLaunchKernel(get_fn(mod(), "bc_dump"), 1, h, 1, w, 1, 1, 0, 0, args, NULL));
        float v[16];
        CK(cuMemcpyDtoH(v, d, sizeof v));
        float ref[16];
        bc4_decode(blk, f->fmt == CU_AD_FORMAT_BC4_SNORM, ref);
        printf("%-34s", cases[c].what);
        for (int i = 0; i < 8; i++) printf(" %8.5f", v[i]);
        printf("\n%-34s", "  exact D3D10");
        for (int i = 0; i < 8; i++) printf(" %8.5f", ref[i]);
        printf("\n");
        cuMemFree(d);
        cuTexObjectDestroy(t);
        cuArrayDestroy(a);
    }
}

static void bc_throughput(void) {
    printf("\n## BC decode throughput (point fetch and tld4 gather)\n");
    printf("%-6s %-7s %8s %-7s %10s %12s %10s %8s %12s\n", "fmt", "set", "MiB", "fetch", "Gtexel/s",
           "Gvalues/s", "GB/s(cmp)", "MHz", "values/clk/SM");
    CUdeviceptr out, clk;
    CK(cuMemAlloc(&out, 64));
    CK(cuMemAlloc(&clk, 64));
    int maxw, maxh;
    CK(cuDeviceGetAttribute(&maxw, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH, g_dev));
    CK(cuDeviceGetAttribute(&maxh, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT, g_dev));
    for (size_t fi = 0; fi < sizeof k_bc / sizeof k_bc[0]; fi++) {
        const bcfmt_t *f = &k_bc[fi];
        for (int big = 0; big < 2; big++) {
            // hot: 2 MiB of compressed data; cold: 1 GiB of compressed data
            size_t cbytes = big ? ((size_t)1 << 30) : ((size_t)2 << 20);
            int w = 16384;
            int h = (int)(cbytes / ((size_t)(w / 4) * f->block_bytes) * 4);
            if (h > maxh) h = maxh;
            size_t nb = (size_t)(w / 4) * (h / 4);
            unsigned char *blocks = malloc(nb * f->block_bytes);
            unsigned s = 7 + (unsigned)fi;
            for (size_t i = 0; i < nb * f->block_bytes / 4; i++) ((unsigned *)blocks)[i] = rng(&s);
            if (f->fmt == CU_AD_FORMAT_BC7_UNORM)
                for (size_t i = 0; i < nb; i++) blocks[i * 16] |= 0x40; // mode 6: valid RGBA block
            CUarray a = bc_array(f, w, h, blocks);
            free(blocks);
            CUtexObject t = arr_tex(a);
            int reps = big ? 1 : 64;
            const char *ks[] = {"bc_point1", "bc_point2", "bc_point4", "bc_gather"};
            int vals[] = {1, 2, 4, 4};
            for (int k = 0; k < 4; k++) {
                if (k == 1 && f->chans < 2) continue;
                if (k == 2 && f->chans < 3) continue;
                CUfunction fn = get_fn(mod(), ks[k]);
                void *args[] = {&t, &w, &h, &reps, &out, &clk};
                double mn = 1e30, md;
                int bps[] = {2, 4, 6};
                for (int b = 0; b < 3; b++) {
                    double m1;
                    time_launch(fn, L1D(g_sms * bps[b], 256, 0), args, 3, 0, &m1, &md);
                    if (m1 < mn) mn = m1;
                }
                double mhz = clk_mhz(clk);
                double texels = (double)w * h * reps; // every texel visited once per rep
                double fetches = k == 3 ? texels / 4 : texels;
                double values = k == 3 ? texels : texels * (k == 0 ? 1 : k == 1 ? 2 : 4);
                if (k == 3) values = texels; // gather returns one channel of each texel
                if (k == 1) values = texels * 2;
                if (k == 2) values = texels * (f->chans >= 4 ? 4 : 3);
                printf("%-6s %-7s %8zu %-7s %10.1f %12.1f %10.1f %8.0f %12.2f\n", f->name,
                       big ? "DRAM" : "L2-hot", (nb * f->block_bytes) >> 20, ks[k] + 3,
                       fetches / (mn * 1e-3) / 1e9, values / (mn * 1e-3) / 1e9,
                       (double)nb * f->block_bytes * reps / (mn * 1e-3) / 1e9, mhz,
                       values / (mn * 1e-3) / (mhz * 1e6) / g_sms);
                (void)vals;
            }
            cuTexObjectDestroy(t);
            cuArrayDestroy(a);
        }
    }
    cuMemFree(out);
    cuMemFree(clk);
}

// ---- 3. GEMV with the texture unit as the 4-bit dequantizer -------------------------
static void bc_gemv(void) {
    printf("\n## GEMV y = W x, W 4 bits/weight, cold (rotating %d copies)\n", 6);
    int shapes[][2] = {{5120, 17408}, {17408, 5120}, {5120, 5120}, {12288, 5120}};
    CUdeviceptr out, clk;
    CK(cuMemAlloc(&out, 64));
    CK(cuMemAlloc(&clk, 64));
    printf("%-10s %6s %6s %10s %10s %8s\n", "kernel", "K", "N", "us", "GB/s", "MHz");
    for (int si = 0; si < 4; si++) {
        int K = shapes[si][0], N = shapes[si][1];
        size_t wbytes = (size_t)K * N / 2;
        int copies = (int)((size_t)400 * 1048576 / wbytes) + 2;
        if (copies > 24) copies = 24;
        CUdeviceptr x, y;
        CK(cuMemAlloc(&x, (size_t)K * 4));
        CK(cuMemAlloc(&y, (size_t)N * 4));
        CK(cuMemsetD32(x, 0x3c000000, K));
        // BC4 SNORM textures: width K, height N
        CUarray *arr = calloc(copies, sizeof(CUarray));
        CUtexObject *tex = calloc(copies, sizeof(CUtexObject));
        size_t nb = (size_t)(K / 4) * (N / 4);
        unsigned char *blocks = malloc(nb * 8);
        unsigned s = 5;
        for (size_t i = 0; i < nb * 2; i++) ((unsigned *)blocks)[i] = rng(&s);
        for (int c = 0; c < copies; c++) {
            arr[c] = bc_array(&k_bc[2], K, N, blocks);
            tex[c] = arr_tex(arr[c]);
        }
        free(blocks);
        CUdeviceptr *lin = calloc(copies, sizeof(CUdeviceptr));
        for (int c = 0; c < copies; c++) {
            CK(cuMemAlloc(&lin[c], wbytes));
            CK(cuMemsetD32(lin[c], 0x89abcdef, wbytes / 4));
        }
        float scale = 0.01f;
        CUfunction fb = get_fn(mod(), "bc4_gemv2"), fi = get_fn(mod(), "int4_gemv");
        CUevent e0, e1;
        CK(cuEventCreate(&e0, 0));
        CK(cuEventCreate(&e1, 0));
        for (int kind = 0; kind < 2; kind++) {
            int warps = kind == 0 ? (N + 1) / 2 : N;
            unsigned blocks_n = (unsigned)((warps * 32 + 255) / 256);
            double best = 1e30;
            for (int rep = 0; rep < 3; rep++) {
                CK(cuEventRecord(e0, 0));
                for (int c = 0; c < copies; c++) {
                    if (kind == 0) {
                        void *args[] = {&tex[c], &x, &K, &N, &scale, &y, &clk};
                        CK(cuLaunchKernel(fb, blocks_n, 1, 1, 256, 1, 1, 0, 0, args, NULL));
                    } else {
                        void *args[] = {&lin[c], &x, &K, &N, &scale, &y, &clk};
                        CK(cuLaunchKernel(fi, blocks_n, 1, 1, 256, 1, 1, 0, 0, args, NULL));
                    }
                }
                CK(cuEventRecord(e1, 0));
                CK(cuEventSynchronize(e1));
                float ms;
                CK(cuEventElapsedTime(&ms, e0, e1));
                if (ms < best) best = ms;
            }
            double us = best * 1e3 / copies;
            printf("%-10s %6d %6d %10.1f %10.1f %8.0f\n", kind == 0 ? "bc4_tld4" : "int4_alu", K, N,
                   us, wbytes / (us * 1e-6) / 1e9, clk_mhz(clk));
        }
        for (int c = 0; c < copies; c++) {
            cuTexObjectDestroy(tex[c]);
            cuArrayDestroy(arr[c]);
            cuMemFree(lin[c]);
        }
        free(arr);
        free(tex);
        free(lin);
        cuMemFree(x);
        cuMemFree(y);
        cuEventDestroy(e0);
        cuEventDestroy(e1);
    }
    cuMemFree(out);
    cuMemFree(clk);
}

int cmd_tex(int argc, char **argv) {
    const char *what = argc > 1 ? argv[1] : "all";
    int all = !strcmp(what, "all");
    if (all || !strcmp(what, "pipes")) lin_pipes();
    if (all || !strcmp(what, "precision")) {
        bc_precision();
        bc_palette();
    }
    if (all || !strcmp(what, "bc")) bc_throughput();
    if (all || !strcmp(what, "gemv")) bc_gemv();
    return 0;
}
