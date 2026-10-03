#ifndef EUHEDRAL_REFERENCE_H
#define EUHEDRAL_REFERENCE_H

#include <stdint.h>

/* Scalar numerical references (native/src/reference/kernels.cu), launched when exact numerics are selected
 * and for shapes no production kernel takes. They check the layout size of `weights_byte_size` themselves
 * where the layout is size-addressed and return the EUHEDRAL_CUDA_* statuses. */
int euhedral_reference_q3(const void* input, const void* weights, void* output, uint32_t rows,
        uint32_t in_features, uint32_t out_features, uint64_t weights_byte_size);
int euhedral_reference_q45(const void* input, const void* weights, void* output, uint32_t rows,
        uint32_t in_features, uint32_t out_features, uint32_t bits);
/* `sd4` selects the scale-table layout. */
int euhedral_reference_nvfp4(const void* input, const void* weights, void* output, uint32_t rows,
        uint32_t in_features, uint32_t out_features, int sd4);

#endif
