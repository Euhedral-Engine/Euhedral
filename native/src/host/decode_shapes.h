#ifndef EUHEDRAL_DECODE_SHAPES_H
#define EUHEDRAL_DECODE_SHAPES_H

#include <stdint.h>

/* Shapes and operand alignment the contiguous decode kernels accept (native/src/q3/contiguous.cuh,
 * native/src/q45/contiguous.cuh): K in whole 1024-value slices, whole 16-row (Q3) or 8-row (Q4, Q5) CTAs. */
static inline int euhedral_q3_decode_shape(uint32_t in_features, uint32_t out_features) {
    return in_features != 0 && in_features % 1024u == 0 && out_features != 0 && out_features % 16u == 0;
}
static inline int euhedral_q45_decode_shape(uint32_t in_features, uint32_t out_features) {
    return in_features != 0 && in_features % 1024u == 0 && out_features != 0 && out_features % 8u == 0;
}

/* Most rows a decode kernel takes in one launch: the single-row kernel and its 2 to 8 row twins. */
#define EUHEDRAL_DECODE_MAX_ROWS 8u

#endif
