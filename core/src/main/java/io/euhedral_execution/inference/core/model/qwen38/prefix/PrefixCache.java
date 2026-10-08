package io.euhedral_execution.inference.core.model.qwen38.prefix;

import io.euhedral_execution.core.frames.AbstractFrame;
import io.euhedral_execution.hashing.HasherApi;
import io.euhedral_execution.inference.core.InferenceConfig;
import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.gpu.GpuStream;
import io.euhedral_execution.inference.core.model.qwen38.AttentionStates;
import io.euhedral_execution.inference.core.model.qwen38.ExecutionPlan;
import io.euhedral_execution.inference.core.model.qwen38.GdnStates;
import io.euhedral_execution.inference.core.model.qwen38.Quantum;
import io.euhedral_execution.inference.core.model.qwen38.Qwen38Config;
import io.euhedral_execution.inference.core.model.qwen38.Sequence;
import io.euhedral_execution.inference.core.model.qwen38.speculative.SpeculativeCheckpoint;
import io.euhedral_execution.inference.core.prefix.HostExtents;
import io.euhedral_execution.inference.core.prefix.PrefixCacheStats;
import io.euhedral_execution.inference.core.prefix.PrefixNode;
import io.euhedral_execution.inference.core.prefix.PrefixTree;
import io.euhedral_execution.inference.core.runtime.graph.FrameLake;
import io.euhedral_execution.inference.core.state.AttentionKvState;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicLong;

/// Sequence state kept across requests, in one pinned host arena: checkpoints of the KV pages and GDN state
/// that prefill produced, found again by the token prefix they cover.
///
/// A checkpoint sits on the prefill chunk grid, because prefill state depends on how the prompt was split
/// into chunks: a sequence restored at a multiple of [InferenceConfig#PREFILL_CHUNK_TOKENS] prefills the rest in the
/// chunks a cold run would use, and ends up with the state a cold run has. Checkpoints are taken every
/// `intervalTokens` and at the last chunk boundary before a prompt ends (see [#wantsCheckpoint]).
///
/// The tree belongs to the cache's owner: every lookup, reservation, publication, abort and release is a frame
/// whose `idHash` is [#HASH] and which stays ordered on it, so they run one at a time and the tree needs no lock.
/// Within a call each frame is published by the one before it, which is the only order the cache relies on.
/// The copies are queued asynchronously on the cache's own stream, in pieces of at most [#PIECE_BYTES]
/// ([PrefixCopies]); their device completion is a frame. A capture reserves its bytes first (evicting the least
/// recently used nodes), copies, then publishes; a failed copy gives the bytes back. A restore pins the chain it
/// reads until the copies ran. Each call ends by telling its [Steps] what happened and throwing the frame it was
/// given.
public final class PrefixCache implements AutoCloseable {

    /// The cache owner's `idHash`: its tree frames are built with it and stay ordered; its copies carry it too, as
    /// the frames it owns, but run anywhere.
    public static final long HASH = HasherApi.mix(0x0b_9f1c_ac4eL);

    /// The most bytes of copies one frame queues.
    static final long PIECE_BYTES = 16L << 20;

    /// A pinned match: its chain stays in the cache until the restore that reads it released it.
    public record Hit(PrefixTree.Match match) {
        public int position() {
            return this.match.position();
        }

        /// Where later captures of the same sequence attach: the deepest matched node.
        public PrefixNode cursor() {
            return this.match.leaf();
        }
    }

    /// What a session hears from its prefix cache, each on a frame of the cache's owner or of its copies, before
    /// the session's next frame is thrown.
    public interface Steps {
        /// The lookup's pinned match, or null on a miss.
        default void found(Hit hit) {}

        /// Whether the restore gave the sequence the hit's state (false: the sequence was cancelled meanwhile), or
        /// the failure that left the sequence failed.
        default void restored(boolean restored, Throwable failure) {}

        /// The node the sequence's next capture attaches to: the new or existing node, or the parent when nothing
        /// was stored. `failure` is an unexpected error; an ordinary miss is not one.
        default void captured(PrefixNode node, Throwable failure) {}
    }

    private final ExecutionGpu gpu;
    private final FrameLake lake;
    private final PrefixLayout layout;
    private final HostExtents extents;
    private final PrefixTree tree;
    private final MemorySegment arena;
    private final Runnable release;
    private final GpuStream stream;
    private final int intervalTokens;
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

    /// Pins `bytes` of host memory as the arena of a new cache whose frames go to `lake`. Throws when the memory
    /// cannot be pinned.
    public static PrefixCache create(
            ExecutionGpu gpu, FrameLake lake, Qwen38Config config, long bytes, int intervalTokens) {
        if (bytes <= 0) throw new IllegalArgumentException("bytes must be positive");
        long address = gpu.allocateHostWeights(bytes);
        MemorySegment arena = MemorySegment.ofAddress(address).reinterpret(bytes);
        try {
            return new PrefixCache(gpu, lake, config, arena, () -> gpu.freeHostWeights(address), intervalTokens);
        } catch (RuntimeException | Error failure) {
            gpu.freeHostWeights(address);
            throw failure;
        }
    }

    public PrefixCache(
            ExecutionGpu gpu,
            FrameLake lake,
            Qwen38Config config,
            MemorySegment arena,
            Runnable release,
            int intervalTokens) {
        if (intervalTokens <= 0 || intervalTokens % InferenceConfig.PREFILL_CHUNK_TOKENS != 0)
            throw new IllegalArgumentException(
                    "the checkpoint interval must be a positive multiple of " + InferenceConfig.PREFILL_CHUNK_TOKENS);
        this.gpu = Objects.requireNonNull(gpu, "gpu");
        this.lake = Objects.requireNonNull(lake, "lake");
        this.layout = PrefixLayout.of(config);
        this.arena = Objects.requireNonNull(arena, "arena");
        this.release = Objects.requireNonNull(release, "release");
        this.extents = new HostExtents(arena.byteSize());
        this.tree = new PrefixTree(this.extents);
        this.intervalTokens = intervalTokens;
        // Lifecycle: the stream lives as long as the cache.
        this.stream = gpu.openStream();
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
        return end < promptLength && promptLength - end <= InferenceConfig.PREFILL_CHUNK_TOKENS;
    }

    // ---------------------------------------------------------------- lookup

    /// Finds the longest stored prefix of `prompt` that leaves a token to prefill, through nodes that hold
    /// `speculation`'s state (any node when null), and pins it: an owner frame tells `steps` (null on a miss), then
    /// throws `next`.
    public void lookup(int[] prompt, String speculation, Steps steps, AbstractFrame next) {
        onOwner(steps, next, () -> {
            if (this.closed) {
                steps.found(null);
                return;
            }
            this.lookups.incrementAndGet();
            PrefixTree.Match match = this.tree.lookup(prompt, speculation);
            if (match == null) {
                steps.found(null);
                return;
            }
            this.hits.incrementAndGet();
            this.reusedTokens.addAndGet(match.position());
            steps.found(new Hit(match));
        });
    }

    // ---------------------------------------------------------------- restore

    /// Gives `sequence`, a fresh one, the state at `hit`'s position: allocates it, loads the chain's KV pages
    /// and the last node's GDN state, and publishes the position. With `speculative` (the hit came from a lookup
    /// with its kind) that strategy's state is restored too. Then an owner frame releases the hit, tells `steps`
    /// whether the sequence was restored (false when it was cancelled meanwhile) or the failure that left it
    /// failed, and throws `next`.
    public void restore(
            ExecutionPlan plan,
            Sequence sequence,
            Hit hit,
            SpeculativeCheckpoint speculative,
            Steps steps,
            AbstractFrame next) {
        int position = hit.position();
        if (speculative != null)
            for (PrefixNode node : hit.match().chain())
                if (!speculative.kind().equals(node.speculation())) {
                    released(
                            hit,
                            false,
                            new IllegalArgumentException("the hit does not hold " + speculative.kind() + " state"),
                            steps,
                            next);
                    return;
                }
        if (this.closed) {
            released(hit, false, new IllegalStateException("the prefix cache is closed"), steps, next);
            return;
        }
        if (cancelled(sequence)) {
            released(hit, false, null, steps, next);
            return;
        }
        var work = new CopyWork(sequence, 0);
        try {
            work.admit(position);
        } catch (RuntimeException failure) {
            // A cancellation that landed after the check above is the same outcome, not an error.
            released(hit, false, cancelled(sequence) ? null : failure, steps, next);
            return;
        }
        List<PrefixLayout.Copy> copies = new ArrayList<>();
        try {
            // On the cache's stream, so the allocations and uploads the state takes are ordered before the copies.
            this.stream.submit(
                    () -> {
                        Quantum.attachSequenceState(plan, sequence, this.gpu);
                        var attention = (AttentionStates) sequence.kvCacheState();
                        var gdn = (GdnStates) sequence.recurrentState();
                        for (int layer : this.layout.kvLayers())
                            attention.forLayer(layer).prepareAppend(0, position);
                        copies.addAll(this.layout.restoreCopies(hit.match().chain(), gdn, attention));
                        if (speculative != null)
                            copies.addAll(speculative.restoreCopies(
                                    hit.match().chain(), this.layout::speculativeOffset, sequence));
                    },
                    false);
        } catch (RuntimeException | Error failure) {
            work.complete(blocked -> {
                abandon(sequence, failure);
                released(hit, false, failure, steps, next);
            });
            return;
        }
        long started = System.nanoTime();
        PrefixCopies.start(
                this,
                copies,
                true,
                copyFailure -> work.complete(blocked -> {
                    Throwable failure = copyFailure != null ? copyFailure : blocked;
                    if (failure != null) {
                        abandon(sequence, failure);
                        released(hit, false, failure, steps, next);
                        return;
                    }
                    this.restores.incrementAndGet();
                    this.restoreNanos.addAndGet(System.nanoTime() - started);
                    try {
                        var attention = (AttentionStates) sequence.kvCacheState();
                        for (int layer : this.layout.kvLayers()) {
                            AttentionKvState state = attention.forLayer(layer);
                            state.appendSubmitted(position);
                            state.commitSubmitted();
                        }
                        if (speculative != null) speculative.restored(sequence, position);
                    } catch (RuntimeException | Error publication) {
                        abandon(sequence, publication);
                        released(hit, false, publication, steps, next);
                        return;
                    }
                    // A cancellation requested meanwhile ends the sequence instead of publishing the position.
                    if (sequence.cancellationRequested()) {
                        sequence.abandon();
                        released(hit, false, null, steps, next);
                        return;
                    }
                    sequence.commit(position);
                    released(hit, true, null, steps, next);
                }));
    }

    /// Unpins `hit` on the owner, then tells `steps` the restore's outcome and throws `next`.
    private void released(Hit hit, boolean restored, Throwable failure, Steps steps, AbstractFrame next) {
        onOwner(steps, next, () -> {
            this.tree.release(hit.match());
            steps.restored(restored, failure);
        });
    }

    /// Fails the sequence and settles the restore's work without it.
    private static void abandon(Sequence sequence, Throwable failure) {
        sequence.fail(failure);
        sequence.abandon();
    }

    private static boolean cancelled(Sequence sequence) {
        return sequence.cancellationRequested() || sequence.terminalState() == Sequence.TerminalState.CANCELLED;
    }

    // ---------------------------------------------------------------- capture

    /// Saves the state of `sequence` at `position` as a node under `parent`, whose tokens are `tokens`, also keeping
    /// `speculative`'s state when the parent chain and the sequence hold it. Then tells `steps` the node the next
    /// capture of the sequence attaches to: the new or existing node, or `parent` when the sequence does not hold
    /// `position` rows, the cache is full, or a copy failed; and throws `next`.
    public void capture(
            Sequence sequence,
            PrefixNode parent,
            int[] tokens,
            int position,
            SpeculativeCheckpoint speculative,
            Steps steps,
            AbstractFrame next) {
        // The copies read the sequence's buffers between quanta, as work on the sequence: they conclude in its
        // admission order, and the buffers close only when the session completes the sequence. A sequence already
        // cancelled, terminal or executing is not captured.
        if (this.closed || sequence.inFlight()) {
            captured(parent, null, steps, next);
            return;
        }
        long at = sequence.committedFrontier();
        var work = new CopyWork(sequence, at);
        try {
            work.admit(at);
        } catch (RuntimeException cancelledOrBusy) {
            captured(parent, null, steps, next);
            return;
        }
        long started = System.nanoTime();
        onOwner(() -> reserve(work, sequence, parent, tokens, position, speculative, steps, next, started));
    }

    /// The owner's part of a capture: finds or reserves the node and queues its copies; the capture ends in a later
    /// frame, or here when there is nothing to copy.
    private void reserve(
            CopyWork work,
            Sequence sequence,
            PrefixNode parent,
            int[] tokens,
            int position,
            SpeculativeCheckpoint speculative,
            Steps steps,
            AbstractFrame next,
            long started) {
        PrefixNode reserved = null;
        List<PrefixLayout.Copy> copies;
        try {
            if (this.closed
                    || !(sequence.recurrentState() instanceof GdnStates gdn)
                    || !(sequence.kvCacheState() instanceof AttentionStates attention)
                    || attention.forLayer(this.layout.kvLayers()[0]).length() < position) {
                settle(work, sequence, () -> captured(parent, null, steps, next));
                return;
            }
            SpeculativeCheckpoint kept = speculative != null
                            && (parent == this.tree.root() || speculative.kind().equals(parent.speculation()))
                            && speculative.holds(sequence, position)
                    ? speculative
                    : null;
            String kind = kept == null ? null : kept.kind();
            PrefixNode existing = this.tree.find(parent, tokens, position, kind);
            // A node with speculative state holds everything a plain one does, and plain lookups accept it.
            if (existing == null && kind == null) existing = this.tree.findAny(parent, tokens, position);
            if (existing != null) {
                PrefixNode found = existing;
                settle(work, sequence, () -> captured(found, null, steps, next));
                return;
            }
            long bytes = this.layout.extentBytes(parent.position(), position)
                    + (kept == null ? 0 : kept.extentBytes(parent.position(), position));
            try {
                reserved = this.tree.reserve(parent, tokens, position, kind, bytes);
            } catch (RuntimeException invalid) {
                this.failed.incrementAndGet();
                settle(work, sequence, () -> captured(parent, null, steps, next));
                return;
            }
            if (reserved == null) {
                this.skipped.incrementAndGet();
                settle(work, sequence, () -> captured(parent, null, steps, next));
                return;
            }
            copies = new ArrayList<>(this.layout.captureCopies(reserved, gdn, attention));
            if (kept != null)
                copies.addAll(kept.captureCopies(reserved, this.layout.speculativeOffset(reserved), sequence));
        } catch (RuntimeException | Error failure) {
            if (reserved != null) this.tree.abort(reserved);
            this.failed.incrementAndGet();
            settle(work, sequence, () -> captured(parent, failure, steps, next));
            return;
        }
        PrefixNode node = reserved;
        PrefixCopies.start(
                this,
                copies,
                false,
                failure -> settle(
                        work,
                        sequence,
                        () -> onOwner(steps, next, () -> {
                            if (failure != null) {
                                this.tree.abort(node);
                                this.failed.incrementAndGet();
                                steps.captured(parent, null);
                                return;
                            }
                            this.tree.publish(node);
                            this.captured.incrementAndGet();
                            this.captureNanos.addAndGet(System.nanoTime() - started);
                            steps.captured(node, null);
                        })));
    }

    /// Concludes a capture's work in the sequence's admission order, then runs `then`. A capture leaves the
    /// frontiers where they are; it commits unless something before it failed the sequence.
    private static void settle(CopyWork work, Sequence sequence, Runnable then) {
        long at = work.admittedAt();
        work.complete(blocked -> {
            try {
                if (blocked != null) sequence.abandon();
                else sequence.commit(at);
            } finally {
                then.run();
            }
        });
    }

    /// Tells `steps` where the sequence's next capture attaches and throws `next`.
    private void captured(PrefixNode node, Throwable failure, Steps steps, AbstractFrame next) {
        try {
            steps.captured(node, failure);
        } finally {
            this.lake.publishOrRun(next);
        }
    }

    // ---------------------------------------------------------------- the owner

    /// Runs `tell` on the owner, then throws `next`.
    private void onOwner(Steps steps, AbstractFrame next, Runnable tell) {
        onOwner(() -> {
            try {
                tell.run();
            } finally {
                this.lake.publishOrRun(next);
            }
        });
    }

    private void onOwner(Runnable work) {
        this.lake.publishOrRun(new Owned(work));
    }

    /// One operation on the tree: ordered on [#HASH], so the tree is confined to these frames.
    static final class Owned extends AbstractFrame {
        private final Runnable work;

        Owned(Runnable work) {
            super(HASH);
            this.work = work;
        }

        private boolean ran;

        @Override
        public void execute() {
            if (this.ran) return;
            this.ran = true;
            this.work.run();
        }

        /// An owner frame the lattice rejected still runs, on the rejecting thread, so its call is told and goes
        /// on: rejection happens only while the lattice shuts down, when nothing else runs on the owner.
        @Override
        public void doFinallyWithError(Throwable rejection) {
            execute();
        }
    }

    // ---------------------------------------------------------------- what the copies read

    ExecutionGpu gpu() {
        return this.gpu;
    }

    FrameLake lake() {
        return this.lake;
    }

    GpuStream stream() {
        return this.stream;
    }

    long arenaAddress() {
        return this.arena.address();
    }

    boolean isClosed() {
        return this.closed;
    }

    public PrefixCacheStats stats() {
        return new PrefixCacheStats(
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

    /// Lifecycle: frees the arena once; frames that run after it copy nothing.
    @Override
    public synchronized void close() {
        if (this.closed) return;
        this.closed = true;
        try {
            this.stream.close();
        } finally {
            this.release.run();
        }
    }
}
