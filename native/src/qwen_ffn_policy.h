#ifndef EUHEDRAL_QWEN_FFN_POLICY_H
#define EUHEDRAL_QWEN_FFN_POLICY_H
#include <stdint.h>
// These are measured shape leaves, not tunable runtime schedules.
static inline uint32_t euhedral_ffn_gate_tile_rows(uint32_t rows, uint32_t width, uint32_t outputs) {
    if (width != 5120u || outputs != 34816u) return 0u;
    if (rows == 64u) return 64u;
    return rows == 256u || rows == 512u || rows == 1024u ? 128u : 0u;
}
static inline int euhedral_ffn_down_wide(uint32_t rows, uint32_t width, uint32_t outputs) {
    return width == 17408u && outputs == 5120u && (rows == 256u || rows == 512u || rows == 1024u);
}
static inline int euhedral_ffn_streamed_rows(uint32_t rows) {
    return rows == 64u || rows == 1024u;
}
#endif
