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
// Down row tile: 0 keeps the generic Q3 prefill route. Measured on the RTX 5070 Ti, the 64-row tile
// (five 64-row bands, 400 CTAs) beats the 128-row tile by 11-20% and the generic kernel by 20-25% for
// 257-320 rows, where three or four 128-row bands leave the second wave nearly empty. It loses at
// exact 256/512/1024 rows and above 320 rows, where decoding each B tile twice as often costs more
// than the tail it removes.
static inline uint32_t euhedral_ffn_down_tile_rows(uint32_t rows, uint32_t width, uint32_t outputs) {
    if (width != 17408u || outputs != 5120u) return 0u;
    if (rows > 256u && rows <= 320u) return 64u;
    return euhedral_ffn_down_wide(rows, width, outputs) ? 128u : 0u;
}
// Split-K down: four K splits for 64-512 row quanta, whose unsplit down runs 80-320 CTAs in one or
// two partial waves. Measured with cold weights, whole FFN per layer: 64 rows 1558 -> 1253 us
// (1400 us for the streamed regions), 128 rows 2222 -> 1816 us, 256 rows 3785 -> 3570 us; the
// down alone at 512 rows 2480 -> 2342 us. Splitting gate/up measured flat or slower.
static inline uint32_t euhedral_ffn_down_splits(uint32_t rows, uint32_t width, uint32_t outputs) {
    return width == 17408u && outputs == 5120u && rows >= 64u && rows <= 512u ? 4u : 0u;
}
// 64-row quanta use full-width gate/up and split-K down instead of streamed regions.
static inline int euhedral_ffn_streamed_rows(uint32_t rows) {
    return rows == 1024u;
}
#endif
