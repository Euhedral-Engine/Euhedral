#include "kernel_table.h"

/* The shared linear kernel table: the weight formats' own launches behind one launcher (kernel_table.c). Entries name
 * their module root, so the table may span modules. The order of `kernels` is
 * the ABI: artifact.format.LinearKernel on the Java side lists the same names in the same order and a test compares
 * them. */
static const EuhedralKernelEntry kernels[] = {
    {"euhedral_trellis_prepass", 0, "trellis/kernels.cu"},
    {"euhedral_trellis_mma", 0, "trellis/kernels.cu"},
    {"euhedral_trellis_mma_rows", 0, "trellis/kernels.cu"},
    {"euhedral_trellis_mma_rows64", 0, "trellis/kernels.cu"},
    {"euhedral_trellis_epilogue", 0, "trellis/kernels.cu"},
    {"euhedral_trellis_decode_bf16", 0, "trellis/kernels.cu"},
    {"euhedral_trellis_epilogue_bf16", 0, "trellis/kernels.cu"},
    /* One CTA per output element holds the rotated input (4 bytes per input feature) in dynamic shared memory: room
     * for 17408 features, the widest linear input, rounded to 1 KiB. It lives in the trellis module so the table
     * loads one module for all of its entries. */
    {"euhedral_trellis_reference", 4 * 17408, "trellis/kernels.cu"},
    /* Block-scaled FP8 tensor cores (sm_12xa): elsewhere unavailable, and the FP32-accumulating launches run. */
    {"euhedral_trellis_mma_rows_fp8", 0, "trellis_fp8/kernels.cu", 1},
    {"euhedral_trellis_mma_rows64_fp8", 0, "trellis_fp8/kernels.cu", 1},
};

static EuhedralKernelTable table = EUHEDRAL_KERNEL_TABLE_INIT(kernels);
EUHEDRAL_KERNEL_TABLE_READY(table_ready, table)

EUHEDRAL_CUDA_EXPORT int euhedral_cuda_linear_kernel_count(void) {
    return table.count;
}

EUHEDRAL_CUDA_EXPORT const char* euhedral_cuda_linear_kernel_name(int index) {
    return index >= 0 && index < table.count ? kernels[index].name : NULL;
}

EUHEDRAL_CUDA_EXPORT int euhedral_cuda_linear_kernel_available(int32_t kernel) {
    return euhedral_kernel_table_available(&table, table_ready, kernel);
}

EUHEDRAL_CUDA_EXPORT int euhedral_cuda_linear_launch(int32_t kernel, uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
        uint32_t block_x, uint32_t block_y, uint32_t block_z, uint32_t shared_bytes,
        const uint64_t* words, const uint8_t* sizes, int32_t count) {
    return euhedral_kernel_table_launch(&table, table_ready, kernel, grid_x, grid_y, grid_z, block_x, block_y, block_z,
            shared_bytes, words, sizes, count);
}
