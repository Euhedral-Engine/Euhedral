// Trellis (EXL3) linear kernels: the module root the host loads for the entries of the shared linear table
// (native/src/host/linear_kernels.c) that name trellis/kernels.cu.

// A launch of the table writes this value, which a test reads back.
extern "C" __global__ void euhedral_trellis_probe(int* out) {
    if (blockIdx.x == 0 && threadIdx.x == 0) *out = 0x7e11150;
}
