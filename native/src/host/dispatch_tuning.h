#ifndef EUHEDRAL_DISPATCH_TUNING_H
#define EUHEDRAL_DISPATCH_TUNING_H

#include <stdint.h>

/* Host dispatch values that were measured on one card and are settings (kernels.rms-norm-decode-slices and
 * kernels.fused-verify-from). The defaults here are the measured values; euhedral_cuda_set_dispatch replaces them. */

/* Slices of the output pass of a one-row RMS norm. */
uint32_t euhedral_dispatch_rms_norm_slices(void);
/* Fewest keys at which the fused verifier kernel serves `rows` verified rows (UINT32_MAX: never). */
uint32_t euhedral_dispatch_fused_from(uint32_t rows);

#endif
