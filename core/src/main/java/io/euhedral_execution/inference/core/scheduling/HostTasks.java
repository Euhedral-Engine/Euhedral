package io.euhedral_execution.inference.core.scheduling;

import io.euhedral_execution.core.frames.AbstractFrame;
import io.euhedral_execution.core.generics.LatticeTerminal;
import io.euhedral_execution.inference.core.host.HostFrames;
import io.euhedral_execution.inference.core.scheduling.graph.FrameLake;
import io.euhedral_execution.inference.core.scheduling.graph.FrameSeeds;
import io.euhedral_execution.inference.core.scheduling.graph.InferenceLake;
import io.euhedral_execution.inference.core.tokenizer.QwenTokenizer;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Supplier;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/// Host work that is not a quantum's device stage, as frames on the lattice's workers: prompt
/// tokenization, the prefix cache's pieces, the work a caller hands over, and everything a model's
/// runtime describes through [HostFrames]. Both inference runtimes (the dense graphs' and
/// Flash-Next's) share it, so there is one way a host task becomes a frame, one lake it is thrown
/// into, and one shutdown.
///
/// Tasks are published into a [FrameLake] (the runtime's pool of ready work). [#close] closes
/// admission of this host work and returns once every admitted task finished; a task that hands on
/// to another publishes the next before it ends, so the host work cannot drain between them. A host
/// work that built its own lake completes it too.
public final class HostTasks implements HostFrames, AutoCloseable {

    private static final Logger LOG = LoggerFactory.getLogger(HostTasks.class);
    private static final int CLOSED = Integer.MIN_VALUE;
    /// Sinks and producer partitions of a lake that host work builds for itself.
    private static final int OWN_SINKS = 2;
    private static final int OWN_PARTITIONS = 2;

    private final FrameLake lake;
    private final InferenceLake ownedLake;
    private final long seedBase = FrameSeeds.forHostWork().next();
    private final AtomicLong nextSeed = new AtomicLong();
    private final AtomicInteger accepted = new AtomicInteger();
    private final CompletableFuture<Void> drained = new CompletableFuture<>();

    /// Host work in a lake of its own, attached to `lattice`.
    public HostTasks(LatticeTerminal lattice) {
        this.ownedLake = new InferenceLake(Objects.requireNonNull(lattice, "lattice"), OWN_SINKS, OWN_PARTITIONS);
        this.lake = this.ownedLake;
    }

    /// Host work in `lake`, which another owner closes.
    public HostTasks(FrameLake lake) {
        this.ownedLake = null;
        this.lake = Objects.requireNonNull(lake, "lake");
    }

    /// Tokenizes `text` on the lattice's workers (PromptTokenization), with the BOS/EOS tokens of
    /// tokenizer_config.json when `modelSpecialTokens`. The future completes on a worker.
    public CompletableFuture<int[]> tokenize(QwenTokenizer tokenizer, String text, boolean modelSpecialTokens) {
        admit();
        return PromptTokenization.start(tokenizer, text, modelSpecialTokens, this.lake::publish, this::terminated);
    }

    /// Runs `work` as one frame on the lattice's workers; the future completes on that worker.
    public <T> CompletableFuture<T> onWorker(Supplier<T> work) {
        Objects.requireNonNull(work, "work");
        CompletableFuture<T> result = new CompletableFuture<>();
        run(new HostFrames.Task() {
            @Override
            public void run() {
                result.complete(work.get());
            }

            @Override
            public void failed(Throwable cause) {
                result.completeExceptionally(cause);
            }
        });
        return result;
    }

    /// Host work for the prefix cache: each piece runs as one frame on the lattice's workers.
    public PrefixCache.Frames prefixFrames() {
        return new PrefixCache.Frames() {
            @Override
            public <T> CompletableFuture<T> run(Supplier<T> work) {
                return onWorker(work);
            }
        };
    }

    @Override
    public void run(Task task) {
        publish(task, false);
    }

    @Override
    public void runFromCallback(Task task) {
        publish(task, true);
    }

    private void publish(Task task, boolean fromCallback) {
        Objects.requireNonNull(task, "task");
        admit();
        try {
            TaskFrame frame = new TaskFrame(task, this, this.seedBase + this.nextSeed.getAndIncrement());
            if (fromCallback) this.lake.publishFromCallback(frame);
            else this.lake.publish(frame);
        } catch (RuntimeException | Error failure) {
            terminated();
            throw failure;
        }
    }

    /// Admission is refused once the host work closed and nothing is left running; while work
    /// drains, the tasks it publishes may continue.
    private void admit() {
        while (true) {
            int count = this.accepted.get();
            boolean draining = count < 0;
            int running = count & Integer.MAX_VALUE;
            if (draining && running == 0) throw new IllegalStateException("host tasks are closed");
            if (running == Integer.MAX_VALUE) throw new IllegalStateException("too many active host tasks");
            if (this.accepted.compareAndSet(count, count + 1)) break;
        }
        try {
            if (this.accepted.get() < 0) this.lake.admitDuringDrain();
            else this.lake.admit();
        } catch (RuntimeException | Error refused) {
            terminated();
            throw refused;
        }
    }

    /// One admitted task ended.
    private void terminated() {
        try {
            this.lake.terminated();
        } finally {
            if (this.accepted.decrementAndGet() == CLOSED) this.drained.complete(null);
        }
    }

    /// Tasks admitted and not yet finished.
    public int activeTasks() {
        return this.accepted.get() & Integer.MAX_VALUE;
    }

    /// Whether the lake this host work publishes into is still attached to the lattice.
    public boolean isAttached() {
        return this.ownedLake != null ? this.ownedLake.isAttached() : this.activeTasks() > 0 || !this.drained.isDone();
    }

    /// Closes admission, waits until every admitted task finished (and, for host work that built
    /// its own lake, until the lattice detached it).
    @Override
    public void close() {
        int old = this.accepted.getAndUpdate(value -> value | CLOSED);
        if ((old & Integer.MAX_VALUE) == 0) this.drained.complete(null);
        if (old < 0) {
            this.drained.join();
            return;
        }
        this.drained.join();
        if (this.ownedLake != null) {
            this.ownedLake.completeGracefully();
            this.ownedLake.awaitTermination();
        }
    }

    /// One task as a frame. A task never throws into the lattice: an exception ends the task
    /// through [HostFrames.Task#failed].
    private static final class TaskFrame extends AbstractFrame {
        private final Task task;
        private final HostTasks owner;
        private final AtomicBoolean finished = new AtomicBoolean();

        TaskFrame(Task task, HostTasks owner, long seed) {
            super(FrameSeeds.ID_HASH);
            randomizeHash(seed);
            this.task = task;
            this.owner = owner;
        }

        @Override
        public void execute() {
            try {
                this.task.run();
            } catch (Throwable failure) {
                report(failure);
            }
        }

        @Override
        public void doFinally() {
            end();
        }

        @Override
        public void doFinallyWithError(Throwable rejection) {
            report(new IllegalStateException("the lattice rejected a host task", rejection));
            end();
        }

        private void report(Throwable failure) {
            try {
                this.task.failed(failure);
            } catch (Throwable reportFailure) {
                LOG.error("a host task could not report its failure", reportFailure);
                if (failure != reportFailure) LOG.error("the failure it could not report", failure);
            }
        }

        private void end() {
            if (this.finished.compareAndSet(false, true)) this.owner.terminated();
        }
    }
}
