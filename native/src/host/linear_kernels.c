#include "kernel_table.h"

/* The shared linear kernel table: the weight formats' own launches behind one launcher (kernel_table.c). Entries name
 * their module root, so the table spans modules (trellis/kernels.cu, reference/kernels.cu). The order of `kernels` is
 * the ABI: artifact.format.LinearKernel on the Java side lists the same names in the same order and a test compares
 * them. */
static const EuhedralKernelEntry kernels[] = {
    {"euhedral_trellis_probe", 0, "trellis/kernels.cu"},
};

static EuhedralKernelTable table = EUHEDRAL_KERNEL_TABLE_INIT(kernels);
EUHEDRAL_KERNEL_TABLE_READY(table_ready, table)

EUHEDRAL_CUDA_EXPORT int euhedral_cuda_linear_kernel_count(void) {
    return table.count;
}

EUHEDRAL_CUDA_EXPORT const char* euhedral_cuda_linear_kernel_name(int index) {
    return index >= 0 && index < table.count ? kernels[index].name : NULL;
}

EUHEDRAL_CUDA_EXPORT int euhedral_cuda_linear_launch(int32_t kernel, uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
        uint32_t block_x, uint32_t block_y, uint32_t block_z, uint32_t shared_bytes,
        const uint64_t* words, const uint8_t* sizes, int32_t count) {
    return euhedral_kernel_table_launch(&table, table_ready, kernel, grid_x, grid_y, grid_z, block_x, block_y, block_z,
            shared_bytes, words, sizes, count);
}
