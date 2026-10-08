package io.euhedral_execution.inference.core.model.qwen38.prefix;

import io.euhedral_execution.core.frames.AbstractFrame;
import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.gpu.GpuStream;
import io.euhedral_execution.inference.core.runtime.graph.FrameLake;
import io.euhedral_execution.inference.core.runtime.graph.FrameSeeds;
import java.util.List;
import java.util.function.Consumer;

/// One piece of a capture's or a restore's copies: at most [PrefixCache#PIECE_BYTES] queued on the cache's stream,
/// asynchronously, so no worker holds a long copy. The cache owns it (its `idHash` is [PrefixCache#HASH]) but it
/// runs on any worker. A piece queues its copies and throws the next piece; the last one arms a device-completion
/// boundary, whose driver callback throws the frame that confirms it and completes the copies.
///
/// The pieces queue in list order on one stream: a restore's list overwrites, in chain order, the page a node shares
/// with its parent. A piece that fails to queue stops the rest, and the copies complete with its failure once what
/// was queued before it retired, so nothing still writes the memory the failure releases.
final class PrefixCopies extends AbstractFrame {

    private final PrefixCache cache;
    private final List<PrefixLayout.Copy> copies;
    private final int from;
    private final boolean toDevice;
    private final Consumer<Throwable> done;
    /// Whether an earlier piece queued copies.
    private final boolean queued;
    private boolean ran;

    private PrefixCopies(
            PrefixCache cache,
            List<PrefixLayout.Copy> copies,
            int from,
            boolean toDevice,
            Consumer<Throwable> done,
            boolean queued) {
        super(PrefixCache.HASH);
        randomizeHash(FrameSeeds.forHostWork().next());
        this.cache = cache;
        this.copies = copies;
        this.from = from;
        this.toDevice = toDevice;
        this.done = done;
        this.queued = queued;
    }

    /// Runs `copies` (`toDevice`: host to device, else device to host) and calls `done` with null once all of them
    /// completed on the device, or with the failure that stopped them.
    static void start(PrefixCache cache, List<PrefixLayout.Copy> copies, boolean toDevice, Consumer<Throwable> done) {
        if (copies.isEmpty()) {
            done.accept(null);
            return;
        }
        cache.lake().publishOrRun(new PrefixCopies(cache, copies, 0, toDevice, done, false));
    }

    @Override
    public void execute() {
        if (this.ran) return;
        this.ran = true;
        int end = this.from;
        long bytes = 0;
        while (end < this.copies.size()
                && (end == this.from || bytes + this.copies.get(end).bytes() <= PrefixCache.PIECE_BYTES))
            bytes += this.copies.get(end++).bytes();
        boolean queuing = false;
        try {
            // The arena is freed by close(): a piece that runs after it must not touch it.
            if (this.cache.isClosed()) throw new IllegalStateException("the prefix cache is closed");
            queuing = true;
            queue(this.from, end);
        } catch (RuntimeException | Error failure) {
            // Copies this piece queued before the failure still write their memory: wait for them too.
            retire(this.queued || queuing, failure);
            return;
        }
        if (end < this.copies.size())
            this.cache
                    .lake()
                    .publishOrRun(new PrefixCopies(this.cache, this.copies, end, this.toDevice, this.done, true));
        else retire(true, null);
    }

    /// A piece the lattice rejected still runs, so its copies complete.
    @Override
    public void doFinallyWithError(Throwable rejection) {
        execute();
    }

    private void queue(int start, int end) {
        ExecutionGpu gpu = this.cache.gpu();
        long arena = this.cache.arenaAddress();
        this.cache.stream()
                .submit(
                        () -> {
                            for (int i = start; i < end; i++) {
                                PrefixLayout.Copy copy = this.copies.get(i);
                                long host = arena + copy.hostOffset();
                                if (this.toDevice) gpu.copyHostToDeviceAsync(copy.deviceAddress(), host, copy.bytes());
                                else gpu.copyDeviceToHostAsync(host, copy.deviceAddress(), copy.bytes());
                            }
                        },
                        false);
    }

    /// Completes the copies once what was queued retired: at once when nothing was.
    private void retire(boolean wait, Throwable failure) {
        if (!wait) {
            this.done.accept(failure);
            return;
        }
        GpuStream stream = this.cache.stream();
        FrameLake lake = this.cache.lake();
        try {
            stream.notifyRetired((ticket, driverThread) -> {
                var retired = new Retired(stream, ticket, failure, this.done);
                // A driver callback may only enqueue; an ordinary thread runs a refused completion itself.
                if (driverThread) lake.publishFromCallback(retired);
                else lake.publishOrRun(retired);
            });
        } catch (RuntimeException | Error armFailure) {
            if (failure != null) armFailure.addSuppressed(failure);
            stream.recover(armFailure);
            this.done.accept(armFailure);
        }
    }

    /// The copies' device-completion boundary, confirmed on a worker.
    static final class Retired extends AbstractFrame {
        private final GpuStream stream;
        private final long ticket;
        private final Throwable failure;
        private final Consumer<Throwable> done;

        Retired(GpuStream stream, long ticket, Throwable failure, Consumer<Throwable> done) {
            super(PrefixCache.HASH);
            randomizeHash(FrameSeeds.forHostWork().next());
            this.stream = stream;
            this.ticket = ticket;
            this.failure = failure;
            this.done = done;
        }

        private boolean ran;

        @Override
        public void execute() {
            if (this.ran) return;
            this.ran = true;
            Throwable device = this.stream.confirmRetired(this.ticket);
            this.done.accept(this.failure != null ? this.failure : device);
        }

        /// A boundary the lattice rejected is still confirmed, so the copies complete.
        @Override
        public void doFinallyWithError(Throwable rejection) {
            execute();
        }
    }
}
