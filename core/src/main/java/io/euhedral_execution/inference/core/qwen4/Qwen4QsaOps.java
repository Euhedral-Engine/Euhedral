package io.euhedral_execution.inference.core.qwen4;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.gpu.KernelArguments;
import io.euhedral_execution.inference.core.gpu.Qwen4Kernel;

/// Typed launches of the QSA (Qwen Sparse Attention) kernels of native/src/qwen4/qsa_*.cuh
/// (docs/FLASH_NEXT_QSA.md): the geometry each kernel is written for and the argument checks the
/// native launcher cannot make. Every method queues on the GPU's selected stream and none allocates
/// on the hot path.
public final class Qwen4QsaOps {

    /// Tokens per indexer block.
    public static final int BLOCK_TOKENS = 4;
    /// Values of an indexer key or query head.
    public static final int INDEX_WIDTH = 128;
    /// Indexer query heads.
    public static final int INDEX_HEADS = 4;
    /// Rows a scores launch covers per CTA row group.
    static final int SCORE_ROWS_PER_CTA = 32;
    /// Blocks a scores launch covers per CTA.
    static final int SCORE_BLOCKS_PER_CTA = 64;
    /// Dynamic shared bytes of an attention CTA, one unit of three warps (q4qsa::kUnitSharedBytes).
    static final int ATTENTION_SHARED_BYTES = 28_224;
    /// Floats of one (row, head, split) partial: 256 values, the running maximum and the running
    /// sum.
    static final int PARTIAL_FLOATS = 258;

    private static final ThreadLocal<KernelArguments> ARGUMENTS = ThreadLocal.withInitial(KernelArguments::new);

    private Qwen4QsaOps() {}

    private static KernelArguments arguments() {
        return ARGUMENTS.get().clear();
    }

    /// Per-head RMSNorm with the one weight of `width` values shared by every head, then (rotary
    /// 64) RoPE on the first 64 values of each head at position `position + row * positionStep`;
    /// BF16 throughout like the upstream tensor operations. Strides are in elements. Input and
    /// output may be the same memory.
    public static void headNormRope(
            ExecutionGpu gpu,
            long input,
            long weight,
            long output,
            int rows,
            int heads,
            int width,
            int inRowStride,
            int inHeadStride,
            int outRowStride,
            int outHeadStride,
            int rotary,
            int position,
            int positionStep,
            float epsilon,
            float theta) {
        requirePositive(rows, heads, width);
        if (width % 32 != 0 || width > 256) throw new IllegalArgumentException("head width must be 32..256 by 32");
        if (rotary != 0 && rotary != 64) throw new IllegalArgumentException("the kernel rotates 64 values or none");
        if (rotary != 0 && width < 64)
            throw new IllegalArgumentException("rotary 64 needs a head of 64 values or more");
        if (position < 0 || positionStep < 0 || (long) position + (long) rows * positionStep >= 1 << 24)
            throw new IllegalArgumentException("positions must stay below 2^24");
        requireAligned(4, input, output, weight);
        gpu.launchTableKernel(
                Qwen4Kernel.HEAD_NORM_ROPE_BF16,
                ceilDiv(Math.multiplyExact(rows, heads), 4),
                1,
                1,
                128,
                1,
                1,
                0,
                arguments()
                        .pointer(input)
                        .pointer(weight)
                        .pointer(output)
                        .int32(rows)
                        .int32(heads)
                        .int32(width)
                        .int32(inRowStride)
                        .int32(inHeadStride)
                        .int32(outRowStride)
                        .int32(outHeadStride)
                        .int32(rotary)
                        .int32(position)
                        .int32(positionStep)
                        .float32(epsilon)
                        .float32(theta));
    }

    /// Pools the blocks a chunk of `rows` tokens at `start` completes: `(start + rows) / 4 - start
    /// / 4` blocks, each the BF16 mean of its four raw keys (the chunk's rows at `raw`, row stride
    /// `rawStride`, or the `tail` of the previous chunks), written to `blocks[start / 4 ..]` still
    /// to be normalized and rotated.
    public static void poolKeys(
            ExecutionGpu gpu, long raw, long tail, long blocks, int rows, int start, int rawStride) {
        int count = completedBlocks(start, rows);
        if (count == 0) return;
        requireAligned(4, raw, tail, blocks);
        gpu.launchTableKernel(
                Qwen4Kernel.QSA_POOL_KEYS_BF16,
                ceilDiv(count, 4),
                1,
                1,
                128,
                1,
                1,
                0,
                arguments()
                        .pointer(raw)
                        .pointer(tail)
                        .pointer(blocks)
                        .int32(count)
                        .int32(start)
                        .int32(rawStride));
    }

    /// Blocks completed by `rows` tokens appended at `start`.
    public static int completedBlocks(int start, int rows) {
        return (start + rows) / BLOCK_TOKENS - start / BLOCK_TOKENS;
    }

    /// Writes the raw keys of the incomplete trailing block after the chunk to `tailOut` (not
    /// `tailIn`).
    public static void tail(ExecutionGpu gpu, long raw, long tailIn, long tailOut, int rows, int start, int rawStride) {
        requirePositive(rows);
        requireAligned(4, raw, tailIn, tailOut);
        gpu.launchTableKernel(
                Qwen4Kernel.QSA_TAIL_BF16,
                1,
                1,
                1,
                128,
                1,
                1,
                0,
                arguments()
                        .pointer(raw)
                        .pointer(tailIn)
                        .pointer(tailOut)
                        .int32(rows)
                        .int32(start)
                        .int32(rawStride));
    }

    /// Scores of chunk rows `rowBegin ..< rowBegin + tileRows` against the first `blocksTotal`
    /// block keys, into `scores[tileRows][scoreStride]` (entry (r, j) only for the blocks row
    /// `rowBegin + r` sees).
    public static void scores(
            ExecutionGpu gpu,
            long queries,
            long keys,
            long scores,
            int rowBegin,
            int tileRows,
            int start,
            int queryRowStride,
            int scoreStride,
            int blocksTotal) {
        requirePositive(tileRows, blocksTotal);
        if (queryRowStride % 2 != 0) throw new IllegalArgumentException("queries are read in pairs");
        requireAligned(4, queries, keys);
        gpu.launchTableKernel(
                Qwen4Kernel.QSA_SCORES,
                ceilDiv(blocksTotal, SCORE_BLOCKS_PER_CTA),
                ceilDiv(tileRows, SCORE_ROWS_PER_CTA),
                1,
                256,
                1,
                1,
                0,
                arguments()
                        .pointer(queries)
                        .pointer(keys)
                        .pointer(scores)
                        .int32(rowBegin)
                        .int32(tileRows)
                        .int32(start)
                        .int32(queryRowStride)
                        .int32(scoreStride)
                        .int32(blocksTotal));
    }

    /// Selects the blocks of chunk rows `rowBegin ..< rowBegin + tileRows`: `ids[row][0 ..<
    /// counts[row]]` ascending.
    public static void select(
            ExecutionGpu gpu,
            long scores,
            long ids,
            long counts,
            int rowBegin,
            int tileRows,
            int start,
            int scoreStride,
            int budget) {
        requirePositive(tileRows, budget);
        gpu.launchTableKernel(
                Qwen4Kernel.QSA_SELECT,
                tileRows,
                1,
                1,
                1024,
                1,
                1,
                0,
                arguments()
                        .pointer(scores)
                        .pointer(ids)
                        .pointer(counts)
                        .int32(rowBegin)
                        .int32(start)
                        .int32(scoreStride)
                        .int32(budget));
    }

    /// Sparse attention of `rows` query rows at `start ..` over their selected blocks (`ids == 0`:
    /// every block, the dense causal prefix) and tail tokens; see
    /// native/src/qwen4/qsa_attention.cuh. With `splits == 1` the finished rows go to `gated` (and
    /// `core` unless 0); with more, partials go to `partial` and [#merge] finishes them.
    public static void attention(
            ExecutionGpu gpu,
            long queries,
            long gate,
            long keyPages,
            long valuePages,
            long ids,
            long counts,
            long partial,
            long core,
            long gated,
            int rows,
            int queryHeads,
            int keyHeads,
            int start,
            int splits,
            int budget,
            int queryRowStride,
            int queryHeadStride,
            int gateRowStride,
            int gateHeadStride) {
        requirePositive(rows, queryHeads, keyHeads, splits);
        if (queryHeads % keyHeads != 0 || queryHeads / keyHeads > 16)
            throw new IllegalArgumentException("a KV head's group of query heads must have at most 16 heads");
        if (splits > 1 && partial == 0) throw new IllegalArgumentException("split attention needs a partial buffer");
        if ((ids == 0) != (counts == 0)) throw new IllegalArgumentException("ids and counts go together");
        long units = (long) rows * keyHeads * splits;
        gpu.launchTableKernel(
                Qwen4Kernel.QSA_ATTENTION,
                Math.toIntExact(units),
                1,
                1,
                96,
                1,
                1,
                ATTENTION_SHARED_BYTES,
                arguments()
                        .pointer(queries)
                        .pointer(gate)
                        .pointer(keyPages)
                        .pointer(valuePages)
                        .pointer(ids)
                        .pointer(counts)
                        .pointer(partial)
                        .pointer(core)
                        .pointer(gated)
                        .int32(rows)
                        .int32(queryHeads)
                        .int32(keyHeads)
                        .int32(start)
                        .int32(splits)
                        .int32(budget)
                        .int32(queryRowStride)
                        .int32(queryHeadStride)
                        .int32(gateRowStride)
                        .int32(gateHeadStride));
    }

    /// Combines the key splits of [#attention] and writes `core` (unless 0) and `gated`.
    public static void merge(
            ExecutionGpu gpu,
            long partial,
            long gate,
            long core,
            long gated,
            int rows,
            int queryHeads,
            int splits,
            int gateRowStride,
            int gateHeadStride) {
        requirePositive(rows, queryHeads, splits);
        gpu.launchTableKernel(
                Qwen4Kernel.QSA_MERGE,
                ceilDiv(Math.multiplyExact(rows, queryHeads), 4),
                1,
                1,
                128,
                1,
                1,
                0,
                arguments()
                        .pointer(partial)
                        .pointer(gate)
                        .pointer(core)
                        .pointer(gated)
                        .int32(rows)
                        .int32(queryHeads)
                        .int32(splits)
                        .int32(gateRowStride)
                        .int32(gateHeadStride));
    }

    /// Quantizes `rows` rows of BF16 keys and values (`keyHeads` heads of 256) into the cache pages
    /// at positions `start ..`; `keyPages` and `valuePages` are the device page tables of the
    /// sequence's KV state.
    public static void kvAppend(
            ExecutionGpu gpu,
            long keys,
            long values,
            long keyPages,
            long valuePages,
            int rows,
            int keyHeads,
            int keyRowStride,
            int valueRowStride,
            int start) {
        requirePositive(rows, keyHeads);
        requireAligned(16, keys, values);
        gpu.launchTableKernel(
                Qwen4Kernel.QSA_KV_APPEND,
                ceilDiv(Math.multiplyExact(rows, keyHeads), 4),
                1,
                1,
                128,
                1,
                1,
                0,
                arguments()
                        .pointer(keys)
                        .pointer(values)
                        .pointer(keyPages)
                        .pointer(valuePages)
                        .int32(rows)
                        .int32(keyHeads)
                        .int32(keyRowStride)
                        .int32(valueRowStride)
                        .int32(start));
    }

    static int ceilDiv(int value, int divisor) {
        return (value + divisor - 1) / divisor;
    }

    static long ceilDiv(long value, long divisor) {
        return (value + divisor - 1) / divisor;
    }

    private static void requirePositive(int... values) {
        for (int value : values) if (value <= 0) throw new IllegalArgumentException("extents must be positive");
    }

    private static void requireAligned(int alignment, long... addresses) {
        for (long address : addresses) {
            if (address == 0 || address % alignment != 0)
                throw new IllegalArgumentException("address " + address + " is not " + alignment + "-byte aligned");
        }
    }
}
