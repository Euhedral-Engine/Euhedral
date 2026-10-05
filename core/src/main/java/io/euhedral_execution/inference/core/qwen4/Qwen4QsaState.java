package io.euhedral_execution.inference.core.qwen4;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.scheduling.AttentionKvState;
import java.util.Objects;

/// What one QSA layer keeps of one sequence: the NVFP4 key/value pages (an [AttentionKvState], reserved
/// lazily page
/// by page), the pooled indexer key of every complete block of four tokens (BF16, 128 values, normalized
/// and rotated)
/// and the raw indexer keys of the up to three tokens of the incomplete trailing block.
///
/// A chunk follows the KV state's protocol: [#beginChunk] reserves the pages and names the start
/// position, the
/// layer's kernels run on the owning quantum's stream, [#submitted] marks the rows readable by later
/// stages of the
/// quantum, and [#commit] publishes them once the quantum retired ([#discard] drops them instead). The
/// raw tail
/// is double buffered: a chunk reads the committed tail and writes the other one, which commit makes the
/// committed
/// tail, so a discarded chunk leaves the committed state intact. Block keys and cache rows at or past the
/// committed length are overwritten by the next chunk, so they need no undoing.
///
/// Bytes per token: 576 of KV in whole 256-token pages, and 64 of indexer keys (256 per block of four
/// tokens).
public final class Qwen4QsaState implements AutoCloseable {

    /// Raw keys kept of the incomplete block.
    public static final int TAIL_SLOTS = Qwen4QsaOps.BLOCK_TOKENS - 1;

    private static final long KEY_BYTES = (long) Qwen4QsaOps.INDEX_WIDTH * Short.BYTES;
    private static final long TAIL_BYTES = TAIL_SLOTS * KEY_BYTES;

    private final ExecutionGpu gpu;
    private final AttentionKvState kv;
    private final int maxTokens;
    private final long blockKeys;
    private final long tails;
    private int liveTail;
    private int pendingRows;
    private boolean closed;

    /// A state for sequences of up to `maxTokens` positions of a layer with `keyValueWidth` K (and V)
    /// values per
    /// token. The indexer block keys are allocated whole; the KV pages as the sequence grows.
    public Qwen4QsaState(ExecutionGpu gpu, int keyValueWidth, int maxTokens) {
        this.gpu = Objects.requireNonNull(gpu, "gpu");
        if (maxTokens <= 0) throw new IllegalArgumentException("maxTokens must be positive");
        this.maxTokens = maxTokens;
        this.kv = new AttentionKvState(gpu, keyValueWidth);
        long keys = 0, pair = 0;
        try {
            keys = gpu.allocate(Math.multiplyExact((long) maxBlocks(maxTokens), KEY_BYTES));
            pair = gpu.allocate(2 * TAIL_BYTES);
        } catch (RuntimeException | Error failure) {
            if (keys != 0) gpu.free(keys);
            this.kv.close();
            throw failure;
        }
        this.blockKeys = keys;
        this.tails = pair;
    }

    /// Blocks of a sequence of `tokens` positions.
    public static int maxBlocks(int tokens) {
        return Math.max(1, tokens / Qwen4QsaOps.BLOCK_TOKENS);
    }

    /// Device bytes the indexer keeps for a sequence of `tokens` positions (the KV pages are in
    /// `io.euhedral_execution.inference.core.model_loader.qwen4.Qwen4SequenceState.kvBytes`).
    public static long indexerBytes(int tokens) {
        return (long) maxBlocks(tokens) * KEY_BYTES + 2 * TAIL_BYTES;
    }

    /// Committed positions.
    public int length() {
        return this.kv.length();
    }

    public int maxTokens() {
        return this.maxTokens;
    }

    /// Reserves the cache pages for `rows` more positions and returns the first one (the committed
    /// length).
    public int beginChunk(int rows) {
        ensureOpen();
        if (rows <= 0) throw new IllegalArgumentException("a chunk needs rows");
        if (this.pendingRows != 0)
            throw new IllegalStateException("the previous chunk was neither committed nor discarded");
        int start = this.kv.length();
        if ((long) start + rows > this.maxTokens)
            throw new IllegalArgumentException("the sequence would exceed " + this.maxTokens + " tokens");
        this.kv.prepareAppend(start, rows);
        this.pendingRows = rows;
        return start;
    }

    /// Records that the chunk's cache writes are queued, so later stages of the quantum may read them.
    public void submitted() {
        ensureOpen();
        if (this.pendingRows == 0) throw new IllegalStateException("no chunk in flight");
        this.kv.appendSubmitted(this.pendingRows);
    }

    /// Publishes the chunk; the quantum's device work has retired.
    public void commit() {
        ensureOpen();
        if (this.pendingRows == 0) throw new IllegalStateException("no chunk in flight");
        this.kv.commitSubmitted();
        this.liveTail ^= 1;
        this.pendingRows = 0;
    }

    /// Drops the chunk; the quantum's device work has retired.
    public void discard() {
        if (this.closed) return;
        this.kv.discardSubmitted();
        this.pendingRows = 0;
    }

    /// Forgets the sequence: the next chunk starts at position 0.
    public void reset() {
        ensureOpen();
        if (this.pendingRows != 0) throw new IllegalStateException("a chunk is in flight");
        this.kv.truncate(0);
        this.liveTail = 0;
    }

    /// Device table of the K pages' addresses.
    public long keyPages() {
        return this.kv.keyCacheAddress();
    }

    /// Device table of the V pages' addresses.
    public long valuePages() {
        return this.kv.valueCacheAddress();
    }

    /// The pooled block keys, `[block][128]` BF16.
    public long blockKeys() {
        return this.blockKeys;
    }

    /// The committed raw tail, `[3][128]` BF16: token `4 (length / 4) + i` at slot `i`.
    public long tailIn() {
        return this.tails + this.liveTail * TAIL_BYTES;
    }

    /// Where the pending chunk writes the tail it leaves.
    public long tailOut() {
        return this.tails + (this.liveTail ^ 1) * TAIL_BYTES;
    }

    /// The underlying KV state (page count and lengths).
    public AttentionKvState kv() {
        return this.kv;
    }

    @Override
    public void close() {
        if (this.closed) return;
        Throwable failure = null;
        try {
            this.kv.close();
        } catch (Throwable error) {
            failure = error;
        }
        try {
            this.gpu.free(this.blockKeys);
            this.gpu.free(this.tails);
            this.closed = true;
        } catch (Throwable error) {
            if (failure == null) failure = error;
            else failure.addSuppressed(error);
        }
        if (failure instanceof RuntimeException runtime) throw runtime;
        if (failure instanceof Error error) throw error;
    }

    private void ensureOpen() {
        if (this.closed) throw new IllegalStateException("QSA state is closed");
    }
}
