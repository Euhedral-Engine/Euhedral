package io.euhedral_execution.inference.core.qwen4;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.model_loader.qwen4.Qwen4Config;

/// The attention block of a sparse-attention layer of Flash-Next (Qwen4ExpTextAttention with its
/// QSA indexer): the mixed block input of a chunk of rows in, the output projection's result out,
/// before the hyper-connection injects it into the residual streams.
///
/// ```
/// x [rows, hidden]
///   q_proj (NVFP4) -> per head [q | gate] -> q_norm, RoPE(64 of 256)          k_proj -> k_norm, RoPE     v_proj
///   index_qk_proj (BF16) -> 4 heads x 128 [q] + 128 [raw key]                          |
///      q: q_layernorm, RoPE(64 of 128)    raw key: pooled by 4, k_layernorm, RoPE at the block's first position
///   k, v -> NVFP4 cache pages                                                          |
///   scores(row, block) = sum_h relu(q_h . blockKey) / sqrt(128); the top 512 blocks of each row (ids ascending)
///   attention over the selected blocks' tokens + the incomplete tail block (softmax scale 1/16)
///   gated = attention * sigmoid(gate);   out = o_proj(gated)
/// ```
///
/// Any chunking of a sequence gives the same result (within the arithmetic noise of the linears):
/// the pooled block keys are computed once, when a block completes, and the raw keys of an
/// incomplete block are carried in the [Qwen4QsaState]. See docs/FLASH_NEXT_QSA.md for the
/// arithmetic and the selection rule.
public final class Qwen4QsaLayer {

    /// Dimensions of the layer. `maxTokens` bounds the sequence a state may hold and the scores
    /// scratch.
    public record Config(
            int hidden,
            int queryHeads,
            int keyHeads,
            int headDim,
            int rotaryDim,
            float ropeTheta,
            float epsilon,
            int indexHeads,
            int indexDim,
            int blockTokens,
            int budgetTokens,
            int maxTokens) {

        /// The layer of a `qwen4_exp` model for sequences of up to `maxTokens`.
        public static Config of(Qwen4Config config, int maxTokens) {
            Qwen4Config.Attention attention = config.attention();
            Qwen4Config.Qsa qsa = config.qsa();
            return new Config(
                    config.text().hiddenSize(),
                    attention.numHeads(),
                    attention.numKvHeads(),
                    attention.headDim(),
                    (int) (attention.headDim() * attention.partialRotaryFactor()),
                    (float) attention.ropeTheta(),
                    (float) config.text().rmsNormEpsilon(),
                    qsa.numHeads(),
                    qsa.headDim(),
                    qsa.compressRatio(),
                    qsa.indexerBudget(),
                    maxTokens);
        }

        int budgetBlocks() {
            return this.budgetTokens / this.blockTokens;
        }

        int queryWidth() {
            return this.queryHeads * this.headDim;
        }

        int keyValueWidth() {
            return this.keyHeads * this.headDim;
        }

        int indexProjectionWidth() {
            return (this.indexHeads + 1) * this.indexDim;
        }
    }

    /// The layer's weights. The projections are NVFP4; the norms and the indexer projection are BF16 (`indexProj` is
    /// `[(heads + 1) * 128][hidden]`, the norm weights are shared by all heads).
    public record Weights(
            Qwen4Weight qProj,
            Qwen4Weight kProj,
            Qwen4Weight vProj,
            Qwen4Weight oProj,
            Qwen4Weight qNorm,
            Qwen4Weight kNorm,
            Qwen4Weight indexProj,
            Qwen4Weight indexQNorm,
            Qwen4Weight indexKNorm) {}

    /// The device buffers one call uses ([#scratch]).
    public record Scratch(
            long qProj,
            long kProj,
            long kNormed,
            long vProj,
            long indexProj,
            long scores,
            long ids,
            long counts,
            long partial,
            long gated,
            long end) {}

    /// Units (row, KV head, key split) a decode-sized launch aims for.
    private static final int TARGET_UNITS = 512;
    /// Keys a split holds at least.
    private static final int MIN_SPLIT_KEYS = 32;
    private static final int MAX_SPLITS = 64;
    private static final long ALIGNMENT = 256;
    /// Default size of the scores scratch: rows are scored in tiles that fit it.
    public static final long DEFAULT_SCORE_BYTES = 16L << 20;

    private final Config config;
    private final long scoreBytes;

    public Qwen4QsaLayer(Config config) {
        this(config, DEFAULT_SCORE_BYTES);
    }

    /// `scoreBytes` bounds the FP32 scores scratch (a tile of rows times the blocks they see); it
    /// is raised to hold at least one row of a sequence of `maxTokens`.
    public Qwen4QsaLayer(Config config, long scoreBytes) {
        this.config = config;
        if (config.blockTokens() != Qwen4QsaOps.BLOCK_TOKENS
                || config.indexHeads() != Qwen4QsaOps.INDEX_HEADS
                || config.indexDim() != Qwen4QsaOps.INDEX_WIDTH)
            throw new IllegalArgumentException("the kernels are written for 4 indexer heads of 128 over blocks of 4");
        if (config.budgetTokens() % config.blockTokens() != 0)
            throw new IllegalArgumentException("the budget must be whole blocks");
        if (config.headDim() != 256 || config.rotaryDim() != 64)
            throw new IllegalArgumentException("the kernels are written for heads of 256 with 64 rotary values");
        if (config.queryHeads() % config.keyHeads() != 0 || config.queryHeads() / config.keyHeads() > 16)
            throw new IllegalArgumentException("a group of query heads must have at most 16 heads");
        if (config.maxTokens() <= 0 || config.maxTokens() >= 1 << 24)
            throw new IllegalArgumentException("maxTokens out of range");
        long oneRow = (long) scoreStride(Qwen4QsaState.maxBlocks(config.maxTokens())) * Float.BYTES;
        this.scoreBytes = align(Math.max(scoreBytes, oneRow));
    }

    public Config config() {
        return this.config;
    }

    private static int scoreStride(int blocks) {
        return (blocks + 31) & ~31;
    }

    private static long align(long bytes) {
        return (bytes + ALIGNMENT - 1) / ALIGNMENT * ALIGNMENT;
    }

    /// Key splits for a launch of `rows` rows whose longest key list has `maxKeys` keys.
    int splitsFor(int rows, int maxKeys) {
        int cap = Math.clamp(TARGET_UNITS / (rows * this.config.keyHeads()), 1, MAX_SPLITS);
        return Math.clamp(maxKeys / MIN_SPLIT_KEYS, 1, cap);
    }

    /// Bytes of split partials a call of up to `rows` rows can need: the largest over the sizes
    /// that split.
    private long partialBytes(int rows) {
        long bytes = 0;
        for (int r = 1; r <= rows && splitsCap(r) > 1; r++)
            bytes = Math.max(
                    bytes,
                    (long) r * this.config.queryHeads() * splitsCap(r) * Qwen4QsaOps.PARTIAL_FLOATS * Float.BYTES);
        return bytes;
    }

    private int splitsCap(int rows) {
        return Math.clamp(TARGET_UNITS / (rows * this.config.keyHeads()), 1, MAX_SPLITS);
    }

    /// Device bytes of the scratch for calls of up to `rows` rows.
    public long scratchBytes(int rows) {
        return scratch(0, rows).end();
    }

    /// The buffers laid out from `base` (256-byte aligned) for calls of up to `rows` rows.
    public Scratch scratch(long base, int rows) {
        if (rows <= 0) throw new IllegalArgumentException("rows must be positive");
        Config c = this.config;
        long at = base;
        long qProj = at;
        at += align((long) rows * c.queryWidth() * 2 * Short.BYTES);
        long kProj = at;
        at += align((long) rows * c.keyValueWidth() * Short.BYTES);
        long kNormed = at;
        at += align((long) rows * c.keyValueWidth() * Short.BYTES);
        long vProj = at;
        at += align((long) rows * c.keyValueWidth() * Short.BYTES);
        long indexProj = at;
        at += align((long) rows * c.indexProjectionWidth() * Short.BYTES);
        long scores = at;
        at += this.scoreBytes;
        long ids = at;
        at += align((long) rows * c.budgetBlocks() * Integer.BYTES);
        long counts = at;
        at += align((long) rows * Integer.BYTES);
        long partial = at;
        at += align(partialBytes(rows));
        long gated = at;
        at += align((long) rows * c.queryWidth() * Short.BYTES);
        return new Scratch(qProj, kProj, kNormed, vProj, indexProj, scores, ids, counts, partial, gated, at);
    }

    /// Runs the attention block for `rows` rows of `input` (`[rows][hidden]` BF16) at the state's
    /// committed length, writing `output` (`[rows][hidden]` BF16). The state's pages are reserved
    /// and the chunk marked submitted; the caller commits it once the work retired, or discards it.
    /// `coreOutput`, when not 0, receives upstream's attention output before the gate
    /// (`[rows][queryHeads * headDim]`), a diagnostic.
    public void run(
            ExecutionGpu gpu,
            Weights w,
            Qwen4QsaState state,
            long input,
            int rows,
            long output,
            Scratch s,
            long coreOutput) {
        int start = state.beginChunk(rows);
        project(gpu, w, input, rows, s);
        normAndRope(gpu, w, rows, start, s);
        appendKeyValues(gpu, state, rows, start, s);
        indexKeys(gpu, w, state, rows, start, s);
        boolean selecting = select(gpu, state, rows, start, s);
        attend(gpu, state, rows, start, s, selecting, coreOutput);
        Qwen4Weight o = w.oProj();
        gpu.linearNvfp4Bf16(s.gated(), o.address(), output, rows, config.queryWidth(), config.hidden(), o.bytes());
        state.submitted();
    }

    /// The query/gate, key and value projections and the indexer projection of `rows` rows.
    void project(ExecutionGpu gpu, Weights w, long input, int rows, Scratch s) {
        Config c = this.config;
        gpu.linearNvfp4Bf16(
                input,
                w.qProj().address(),
                s.qProj(),
                rows,
                c.hidden(),
                2 * c.queryWidth(),
                w.qProj().bytes());
        gpu.linearNvfp4Bf16(
                input,
                w.kProj().address(),
                s.kProj(),
                rows,
                c.hidden(),
                c.keyValueWidth(),
                w.kProj().bytes());
        gpu.linearNvfp4Bf16(
                input,
                w.vProj().address(),
                s.vProj(),
                rows,
                c.hidden(),
                c.keyValueWidth(),
                w.vProj().bytes());
        Qwen4Ops.linearBf16(
                gpu, input, w.indexProj().address(), s.indexProj(), rows, c.hidden(), c.indexProjectionWidth());
    }

    /// q_norm and RoPE in place on the query half of every head's [q | gate] (the gate stays where
    /// it is), k_norm and RoPE into `kNormed`, q_layernorm and RoPE in place on the indexer
    /// queries.
    void normAndRope(ExecutionGpu gpu, Weights w, int rows, int start, Scratch s) {
        Config c = this.config;
        int qWidth = 2 * c.queryWidth(), head = 2 * c.headDim(), kvWidth = c.keyValueWidth();
        int indexWidth = c.indexProjectionWidth();
        Qwen4QsaOps.headNormRope(
                gpu,
                s.qProj(),
                w.qNorm().address(),
                s.qProj(),
                rows,
                c.queryHeads(),
                c.headDim(),
                qWidth,
                head,
                qWidth,
                head,
                c.rotaryDim(),
                start,
                1,
                c.epsilon(),
                c.ropeTheta());
        Qwen4QsaOps.headNormRope(
                gpu,
                s.kProj(),
                w.kNorm().address(),
                s.kNormed(),
                rows,
                c.keyHeads(),
                c.headDim(),
                kvWidth,
                c.headDim(),
                kvWidth,
                c.headDim(),
                c.rotaryDim(),
                start,
                1,
                c.epsilon(),
                c.ropeTheta());
        Qwen4QsaOps.headNormRope(
                gpu,
                s.indexProj(),
                w.indexQNorm().address(),
                s.indexProj(),
                rows,
                c.indexHeads(),
                c.indexDim(),
                indexWidth,
                c.indexDim(),
                indexWidth,
                c.indexDim(),
                c.rotaryDim(),
                start,
                1,
                c.epsilon(),
                c.ropeTheta());
    }

    /// Quantizes the chunk's keys and values into the cache pages.
    void appendKeyValues(ExecutionGpu gpu, Qwen4QsaState state, int rows, int start, Scratch s) {
        int kvWidth = this.config.keyValueWidth();
        Qwen4QsaOps.kvAppend(
                gpu,
                s.kNormed(),
                s.vProj(),
                state.keyPages(),
                state.valuePages(),
                rows,
                this.config.keyHeads(),
                kvWidth,
                kvWidth,
                start);
    }

    /// Pools the blocks the chunk completes (the raw key is the last 128 values of an indexer row),
    /// normalizes and rotates them into the block keys, and leaves the raw keys of the incomplete
    /// block as the new tail.
    void indexKeys(ExecutionGpu gpu, Weights w, Qwen4QsaState state, int rows, int start, Scratch s) {
        Config c = this.config;
        int indexWidth = c.indexProjectionWidth();
        long rawKeys = s.indexProj() + (long) c.indexHeads() * c.indexDim() * Short.BYTES;
        Qwen4QsaOps.poolKeys(gpu, rawKeys, state.tailIn(), state.blockKeys(), rows, start, indexWidth);
        int completed = Qwen4QsaOps.completedBlocks(start, rows);
        if (completed > 0) {
            int first = start / Qwen4QsaOps.BLOCK_TOKENS;
            long fresh = state.blockKeys() + (long) first * c.indexDim() * Short.BYTES;
            Qwen4QsaOps.headNormRope(
                    gpu,
                    fresh,
                    w.indexKNorm().address(),
                    fresh,
                    completed,
                    1,
                    c.indexDim(),
                    c.indexDim(),
                    c.indexDim(),
                    c.indexDim(),
                    c.indexDim(),
                    c.rotaryDim(),
                    first * Qwen4QsaOps.BLOCK_TOKENS,
                    Qwen4QsaOps.BLOCK_TOKENS,
                    c.epsilon(),
                    c.ropeTheta());
        }
        Qwen4QsaOps.tail(gpu, rawKeys, state.tailIn(), state.tailOut(), rows, start, indexWidth);
    }

    /// Whether any row of a chunk at `start` chooses among more blocks than the budget: only then
    /// do the scores and the selection (`ids`, `counts` of the scratch) exist.
    boolean selects(int rows, int start) {
        return start + rows - 1 >= this.config.budgetTokens() + Qwen4QsaOps.BLOCK_TOKENS - 1;
    }

    /// Rows scored at a time: as many as the scores scratch holds for `selectingRows` rows seeing
    /// the blocks of a sequence that ends at position `end`.
    int scoreTileRows(int selectingRows, int end) {
        int stride = scoreStride(end / Qwen4QsaOps.BLOCK_TOKENS);
        return (int) Math.max(1, Math.min(selectingRows, this.scoreBytes / Float.BYTES / stride));
    }

    /// Scores and selects the blocks of every row. Row p keeps every block while it sees at most
    /// `budget` of them, so the rows before the first one that chooses just list their blocks.
    /// Returns whether `ids` and `counts` were written ([#selects]).
    boolean select(ExecutionGpu gpu, Qwen4QsaState state, int rows, int start, Scratch s) {
        if (!selects(rows, start)) return false;
        Config c = this.config;
        int budget = c.budgetBlocks();
        int indexWidth = c.indexProjectionWidth();
        int firstSelecting = Math.max(0, c.budgetTokens() + Qwen4QsaOps.BLOCK_TOKENS - 1 - start);
        if (firstSelecting > 0)
            Qwen4QsaOps.select(gpu, s.scores(), s.ids(), s.counts(), 0, firstSelecting, start, 0, budget);
        int stride = scoreStride((start + rows) / Qwen4QsaOps.BLOCK_TOKENS);
        int tile = scoreTileRows(rows - firstSelecting, start + rows);
        for (int begin = firstSelecting; begin < rows; begin += tile) {
            int count = Math.min(tile, rows - begin);
            int blocks = (start + begin + count) / Qwen4QsaOps.BLOCK_TOKENS;
            Qwen4QsaOps.scores(
                    gpu, s.indexProj(), state.blockKeys(), s.scores(), begin, count, start, indexWidth, stride, blocks);
            Qwen4QsaOps.select(gpu, s.scores(), s.ids(), s.counts(), begin, count, start, stride, budget);
        }
        return true;
    }

    /// Attention of every row over its selected blocks and tail, with the sigmoid gate: `gated` of
    /// the scratch (and `coreOutput` unless 0).
    void attend(
            ExecutionGpu gpu, Qwen4QsaState state, int rows, int start, Scratch s, boolean selecting, long coreOutput) {
        Config c = this.config;
        int qWidth = 2 * c.queryWidth(), head = 2 * c.headDim();
        int maxKeys = Math.min(c.budgetTokens() + Qwen4QsaOps.BLOCK_TOKENS - 1, start + rows);
        int splits = splitsFor(rows, maxKeys);
        long gate = s.qProj() + (long) c.headDim() * Short.BYTES;
        Qwen4QsaOps.attention(
                gpu,
                s.qProj(),
                gate,
                state.keyPages(),
                state.valuePages(),
                selecting ? s.ids() : 0,
                selecting ? s.counts() : 0,
                splits > 1 ? s.partial() : 0,
                coreOutput,
                s.gated(),
                rows,
                c.queryHeads(),
                c.keyHeads(),
                start,
                splits,
                c.budgetBlocks(),
                qWidth,
                head,
                qWidth,
                head);
        if (splits > 1)
            Qwen4QsaOps.merge(
                    gpu, s.partial(), gate, coreOutput, s.gated(), rows, c.queryHeads(), splits, qWidth, head);
    }
}
