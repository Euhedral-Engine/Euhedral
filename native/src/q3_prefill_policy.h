#ifndef EUHEDRAL_Q3_PREFILL_POLICY_H
#define EUHEDRAL_Q3_PREFILL_POLICY_H

#include <stdint.h>

// The explicit 64-row entry point remains independent of AUTO's shape policy.
static inline int euhedral_q3_wide_prefill(int mode, uint32_t rows,
        uint32_t in_features, uint32_t out_features) {
    if (mode == 3) return 1;
    if (mode != 2) return 0;
    // Smaller mixer batches win at the operator level but have inconsistent
    // full-model gains; route only repeatably faster prefill chunks to tile64.
    if (in_features == 6144 && out_features == 5120) return rows >= 512;
    return rows >= 64 && ((in_features == 5120 && out_features == 34816)
            || (in_features == 17408 && out_features == 5120));
}

#endif
