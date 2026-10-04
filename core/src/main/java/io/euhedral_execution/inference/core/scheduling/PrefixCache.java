package io.euhedral_execution.inference.core.scheduling;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.model_loader.config.QwenConfig;
import io.euhedral_execution.inference.core.prefix.HostExtents;
import io.euhedral_execution.inference.core.prefix.PrefixNode;
import io.euhedral_execution.inference.core.prefix.PrefixTree;
import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Supplier;

/// Sequence state kept across requests, in one pinned host arena: checkpoints of the KV pages and GDN state
/// that prefill produced, found again by the token prefix they cover.
///
/// A checkpoint sits on the prefill chunk grid, because prefill state depends on how the prompt was split
/// into chunks: a sequence restored at a multiple of [#CHUNK_TOKENS] prefills the rest in the chunks a cold
/// run would use, and ends up with the state a cold run has. Checkpoints are taken every `intervalTokens`
/// and at the last chunk boundary before a prompt ends (see [#wantsCheckpoint]).
///
/// Capture and restore move the state in frames of at most [#PIECE_BYTES] of copies each, so no worker holds
/// a long copy. A capture reserves its bytes first (evicting the least recently used nodes), copies, then
/// publishes; a failed copy gives the bytes back. A restore pins the chain it reads until the copies ran.
public final class PrefixCache implements AutoCloseable {
    /// The prefill chunk the cache's checkpoints are aligned to; sessions that use the cache prefill in it.
    public static final int CHUNK_TOKENS = QwenGenerationSession.DEFAULT_PREFILL_CHUNK_TOKENS;

    /// The most bytes of copies one frame runs.
    static final long PIECE_BYTES = 16L << 20;

    /// Runs a piece of host work as one frame and completes with its result.
    public interface Frames {
        <T> CompletableFuture<T> run(Supplier<T> work);
    }

    /// A pinned match: its chain stays in the cache until [#release].
    public record Hit(PrefixTree.Match match) {
        public int position() {
            return this.match.position();
        }

        /// Where later captures of the same sequence attach: the deepest matched node.
        public PrefixNode cursor() {
            return this.match.leaf();
        }
    }

    public record Stats(
            long lookups,
            long hits,
            long reusedTokens,
            long captured,
            long skipped,
            long failed,
            long evictions,
            long usedBytes,
            long totalBytes,
            int nodes,
            long captureNanos,
            long restores,
            long restoreNanos) {}

    private final ExecutionGpu gpu;
    private final PrefixLayout layout;
    private final HostExtents extents;
    private final PrefixTree tree;
    private final MemorySegment arena;
    private final Runnable release;
    private final int intervalTokens;
    private final int hidden;
    private final AtomicLong lookups = new AtomicLong();
    private final AtomicLong captureNanos = new AtomicLong();
    private final AtomicLong restores = new AtomicLong();
    private final AtomicLong restoreNanos = new AtomicLong();
    private final AtomicLong hits = new AtomicLong();
    private final AtomicLong reusedTokens = new AtomicLong();
    private final AtomicLong captured = new AtomicLong();
    private final AtomicLong skipped = new AtomicLong();
    private final AtomicLong failed = new AtomicLong();
    private volatile boolean closed;

    /// Pins `bytes` of host memory as the arena of a new cache. With `mtp` the sequences it serves carry the MTP
    /// layer's cache, which checkpoints of speculative prompts keep. Throws when the memory cannot be pinned.
    public static PrefixCache create(ExecutionGpu gpu, QwenConfig config, boolean mtp, long bytes, int intervalTokens) {
        if (bytes <= 0) throw new IllegalArgumentException("bytes must be positive");
        long address = gpu.allocateHostWeights(bytes);
        MemorySegment arena = MemorySegment.ofAddress(address).reinterpret(bytes);
        return new PrefixCache(gpu, config, mtp, arena, () -> gpu.freeHostWeights(address), intervalTokens);
    }

    PrefixCache(
            ExecutionGpu gpu,
            QwenConfig config,
            boolean mtp,
            MemorySegment arena,
            Runnable release,
            int intervalTokens) {
        if (intervalTokens <= 0 || intervalTokens % CHUNK_TOKENS != 0)
            throw new IllegalArgumentException(
                    "the checkpoint interval must be a positive multiple of " + CHUNK_TOKENS);
        this.gpu = Objects.requireNonNull(gpu, "gpu");
        this.layout = PrefixLayout.of(config, mtp);
        this.hidden = config.hiddenSize();
        this.arena = Objects.requireNonNull(arena, "arena");
        this.release = Objects.requireNonNull(release, "release");
        this.extents = new HostExtents(arena.byteSize());
        this.tree = new PrefixTree(this.extents);
        this.intervalTokens = intervalTokens;
    }

    /// Whether checkpoints can hold the MTP cache, so that speculative prompts can restore from them.
    public boolean supportsMtp() {
        return this.layout.mtpLayer() >= 0;
    }

    public int intervalTokens() {
        return this.intervalTokens;
    }

    public PrefixNode root() {
        return this.tree.root();
    }

    /// Whether to store the state after the prefill chunk that ended at `end` of a prompt of `promptLength`
    /// tokens: on the checkpoint interval, and at the last chunk boundary before the prompt ends, where the
    /// next chunk is the final one. The latter is what lets a follow-up that extends the prompt, or repeats
    /// it, skip all but the last chunk.
    public boolean wantsCheckpoint(int end, int promptLength) {
        if (end % this.intervalTokens == 0) return true;
        return end < promptLength && promptLength - end <= CHUNK_TOKENS;
    }

    /// The longest stored prefix of `prompt` that leaves a token to prefill, pinned; null on a miss.
    public Hit lookup(int[] prompt, boolean needsMtp) {
        this.lookups.incrementAndGet();
        PrefixTree.Match match = this.tree.lookup(prompt, needsMtp);
        if (match == null) return null;
        this.hits.incrementAndGet();
        this.reusedTokens.addAndGet(match.position());
        return new Hit(match);
    }

    public void release(Hit hit) {
        this.tree.release(hit.match());
    }

    /// Saves the state of `sequence` at `position` as a node under `parent`, whose tokens are `tokens`, and
    /// completes with the node the next capture of the sequence attaches to: the new or existing node, or
    /// `parent` when the sequence does not hold `position` rows, the cache is full, or a copy failed. Never
    /// fails.
    public CompletableFuture<PrefixNode> capture(
            Frames frames, QwenSequenceState sequence, PrefixNode parent, int[] tokens, int position) {
        return capture(frames, sequence, parent, tokens, position, 0);
    }

    /// As [#capture(Frames, QwenSequenceState, PrefixNode, int[], int)], also keeping the MTP state of a
    /// speculative prompt: the MTP cache's rows below `position - 1` and the base hidden row at
    /// `seedRowAddress`, that of position `position - 1`. Without MTP state in the cache, in the parent chain,
    /// or in the sequence (its MTP cache lags), the node is stored without it.
    public CompletableFuture<PrefixNode> capture(
            Frames frames,
            QwenSequenceState sequence,
            PrefixNode parent,
            int[] tokens,
            int position,
            long seedRowAddress) {
        if (this.closed) return CompletableFuture.completedFuture(parent);
        // The copies read the sequence's buffers between quanta. Holding its lease for them keeps a cancellation
        // from releasing those buffers mid-copy: cancel() then only flags the sequence, and the release after the
        // copies frees them. A sequence already cancelled, terminal or executing is not captured.
        long at = sequence.currentTokenPosition();
        QwenSequenceState.ExecutionLease lease;
        try {
            lease = sequence.claimExecution(at);
        } catch (RuntimeException cancelledOrBusy) {
            return CompletableFuture.completedFuture(parent);
        }
        CompletableFuture<PrefixNode> captured;
        long started = System.nanoTime();
        try {
            captured = captureHeld(frames, sequence, parent, tokens, position, seedRowAddress);
        } catch (RuntimeException | Error failure) {
            sequence.releaseExecution(lease, at);
            throw failure;
        }
        return captured.whenComplete((node, failure) -> {
            sequence.releaseExecution(lease, at);
            if (node != parent) this.captureNanos.addAndGet(System.nanoTime() - started);
        });
    }

    private CompletableFuture<PrefixNode> captureHeld(
            Frames frames,
            QwenSequenceState sequence,
            PrefixNode parent,
            int[] tokens,
            int position,
            long seedRowAddress) {
        if (!(sequence.recurrentState() instanceof GdnSequenceStates gdn)
                || !(sequence.kvCacheState() instanceof AttentionSequenceStates attention)
                || attention.forLayer(this.layout.kvLayers()[0]).length() < position)
            return CompletableFuture.completedFuture(parent);
        boolean mtp = seedRowAddress != 0
                && supportsMtp()
                && (parent == this.tree.root() || parent.hasMtp())
                && attention.forLayer(this.layout.mtpLayer()).length() >= position - 1;
        PrefixNode existing = this.tree.find(parent, tokens, position, mtp);
        // A node with MTP state holds everything a plain one does, and plain lookups accept it.
        if (existing == null && !mtp) existing = this.tree.find(parent, tokens, position, true);
        if (existing != null) return CompletableFuture.completedFuture(existing);
        PrefixNode node;
        try {
            node = this.tree.reserve(
                    parent, tokens, position, mtp, this.layout.extentBytes(parent.position(), position, mtp));
        } catch (RuntimeException invalid) {
            this.failed.incrementAndGet();
            return CompletableFuture.completedFuture(parent);
        }
        if (node == null) {
            this.skipped.incrementAndGet();
            return CompletableFuture.completedFuture(parent);
        }
        PrefixNode reserved = node;
        List<PrefixLayout.Copy> copies = this.layout.captureCopies(reserved, gdn, attention, seedRowAddress);
        return runCopies(frames, copies, false).handle((done, failure) -> {
            if (failure != null) {
                this.tree.abort(reserved);
                this.failed.incrementAndGet();
                return parent;
            }
            this.tree.publish(reserved);
            this.captured.incrementAndGet();
            return reserved;
        });
    }

    public CompletableFuture<Boolean> restore(
            Frames frames, QwenExecutionPlan plan, QwenSequenceState sequence, Hit hit) {
        return restore(frames, plan, sequence, hit, false);
    }

    /// Gives `sequence`, a fresh one, the state at `hit`'s position: allocates it, loads the chain's KV pages
    /// and the last node's GDN state, and publishes the position. With `mtp` (the hit came from
    /// `lookup(prompt, true)`) the MTP cache is restored too, to `position - 1` rows, and the last node's
    /// hidden row is left in the sequence's draft seed buffer for the speculative decoder to pair with the
    /// next prompt token. Completes with true when restored, and with false when the sequence was cancelled
    /// meanwhile; it fails on any other error, leaving the sequence failed for the caller to close.
    public CompletableFuture<Boolean> restore(
            Frames frames, QwenExecutionPlan plan, QwenSequenceState sequence, Hit hit, boolean mtp) {
        int position = hit.position();
        if (this.closed) return CompletableFuture.failedFuture(new IllegalStateException("the prefix cache is closed"));
        if (cancelled(sequence)) return CompletableFuture.completedFuture(false);
        QwenSequenceState.ExecutionLease lease;
        List<PrefixLayout.Copy> copies;
        try {
            lease = sequence.claimExecution(0);
        } catch (RuntimeException failure) {
            // A cancellation that landed after the check above is the same outcome, not an error.
            return cancelled(sequence)
                    ? CompletableFuture.completedFuture(false)
                    : CompletableFuture.failedFuture(failure);
        }
        try {
            QwenExecutionContext.attachSequenceState(plan, sequence, lease, this.gpu);
            var attention = (AttentionSequenceStates) sequence.kvCacheState();
            var gdn = (GdnSequenceStates) sequence.recurrentState();
            for (int layer : this.layout.kvLayers()) attention.forLayer(layer).prepareAppend(0, position);
            long seedRow = 0;
            if (mtp) {
                attention.forLayer(this.layout.mtpLayer()).prepareAppend(0, position - 1);
                seedRow = attention.draftSeedRows(1, this.hidden);
            }
            copies = this.layout.restoreCopies(hit.match().chain(), gdn, attention, seedRow);
        } catch (RuntimeException | Error failure) {
            sequence.markFailed(lease, failure);
            return CompletableFuture.failedFuture(failure);
        }
        long started = System.nanoTime();
        return runCopies(frames, copies, true).handle((done, failure) -> {
            if (failure != null) {
                sequence.markFailed(lease, failure);
                throw new CompletionException(failure);
            }
            this.restores.incrementAndGet();
            this.restoreNanos.addAndGet(System.nanoTime() - started);
            try {
                var attention = (AttentionSequenceStates) sequence.kvCacheState();
                for (int layer : this.layout.kvLayers()) {
                    AttentionKvState state = attention.forLayer(layer);
                    state.appendSubmitted(position);
                    state.commitSubmitted();
                }
                if (mtp) {
                    AttentionKvState state = attention.forLayer(this.layout.mtpLayer());
                    state.appendSubmitted(position - 1);
                    state.commitSubmitted();
                }
                return !sequence.releaseExecutionAndCheckCancellation(lease, position);
            } catch (RuntimeException | Error publication) {
                sequence.markFailed(lease, publication);
                throw new CompletionException(publication);
            }
        });
    }

    private static boolean cancelled(QwenSequenceState sequence) {
        return sequence.cancellationRequested()
                || sequence.terminalState() == QwenSequenceState.TerminalState.CANCELLED;
    }

    public Stats stats() {
        return new Stats(
                this.lookups.get(),
                this.hits.get(),
                this.reusedTokens.get(),
                this.captured.get(),
                this.skipped.get(),
                this.failed.get(),
                this.tree.evictions(),
                this.extents.usedBytes(),
                this.extents.totalBytes(),
                this.tree.size(),
                this.captureNanos.get(),
                this.restores.get(),
                this.restoreNanos.get());
    }

    @Override
    public synchronized void close() {
        if (this.closed) return;
        this.closed = true;
        this.release.run();
    }

    /// Runs `copies` in frames of at most [#PIECE_BYTES], one after another. `toDevice` selects the direction.
    private CompletableFuture<Void> runCopies(Frames frames, List<PrefixLayout.Copy> copies, boolean toDevice) {
        CompletableFuture<Void> done = CompletableFuture.completedFuture(null);
        int from = 0;
        while (from < copies.size()) {
            int start = from;
            long bytes = 0;
            while (from < copies.size()
                    && (from == start || bytes + copies.get(from).bytes() <= PIECE_BYTES))
                bytes += copies.get(from++).bytes();
            List<PrefixLayout.Copy> piece = copies.subList(start, from);
            done = done.thenCompose(ignored -> frames.run(() -> {
                // The arena is freed by close(): a frame that runs after it must not touch it.
                if (this.closed) throw new IllegalStateException("the prefix cache is closed");
                for (PrefixLayout.Copy copy : piece) {
                    MemorySegment host = this.arena.asSlice(copy.hostOffset(), copy.bytes());
                    if (toDevice) this.gpu.copyHostToDevice(copy.deviceAddress(), host, copy.bytes());
                    else this.gpu.copyDeviceToHost(host, copy.deviceAddress(), copy.bytes());
                }
                return null;
            }));
        }
        return done;
    }
}
