package io.euhedral_execution.inference.core.model.qwen38;

import io.euhedral_execution.inference.core.settings.KernelSettings;
import io.euhedral_execution.inference.core.state.KvFormat;
import java.util.List;

/// What the host dispatch of multi-row decode attention (`attention_causal` in
/// `native/src/host/qwen_layer_ops.c`) chooses from a quantum's rows and positions: the split counts that size
/// its launches, and whether the fused verifier kernel serves the rows. A captured graph replays the launches
/// recorded for one choice, so the capture key holds all of it; the values here repeat the C code's, and the fused
/// table is the one the native library was given.
final class AttentionLaunchGeometry {

    /// Keys from which the rows use the GQA kernels, for query-head groups up to [#MAX_GROUP]
    /// (`from` in `attention_causal`).
    static final long GQA_FROM = 2048;

    /// The largest query-head group the GQA kernels serve.
    static final int MAX_GROUP = 8;

    /// The most mma tiles of 8 query columns the fused kernel is built for (`tiles <= 6u` in `attention_causal`).
    static final int MAX_FUSED_TILES = 6;

    private AttentionLaunchGeometry() {}

    /// Fewest keys at which `rows` verified rows use the fused kernel under `table`, which lists them by rows
    /// (`kernels.fused-verify-from`): a row count outside the table never fuses, nor does {@link
    /// KernelSettings#NEVER}.
    static long fusedFrom(List<Long> table, int rows) {
        if (rows < 0 || rows >= table.size()) return Long.MAX_VALUE;
        long keys = table.get(rows);
        return keys == KernelSettings.NEVER ? Long.MAX_VALUE : keys;
    }

    /// Whether the fused verifier kernel serves `rows` rows of query-head `group` over `format` caches whose last
    /// row has `last` keys: the fused choice in `attention_causal`, which exists for NVFP4 caches only.
    static boolean fused(List<Long> table, int group, KvFormat format, int rows, long last) {
        if (format != KvFormat.NVFP4 || rows <= 1 || group <= 0 || group > MAX_GROUP) return false;
        int tiles = Math.max(2, (rows * group + 7) / 8);
        return tiles <= MAX_FUSED_TILES && last >= GQA_FROM && last >= fusedFrom(table, rows);
    }

    /// The split counts that size decode attention's launches, and the fused choice, of `rows` rows from
    /// `startPosition`: rows below 2048 keys split into 48-key spans, rows from 2048 keys (groups up to 8) into
    /// 32-key spans, at most 64 each.
    static long of(List<Long> table, int group, KvFormat format, long startPosition, int rows) {
        long first = startPosition + 1, last = startPosition + rows;
        long from = group <= MAX_GROUP ? GQA_FROM : Long.MAX_VALUE;
        long below = first < from ? Math.min(64, (Math.min(last, from - 1) + 47) / 48) : 0;
        long above = last >= from ? Math.min(64, (last + 31) / 32) : 0;
        return (fused(table, group, format, rows, last) ? 1L << 16 : 0L) | below << 8 | above;
    }
}
