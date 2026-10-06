package io.euhedral_execution.inference.core.scheduling;

import io.euhedral_execution.core.frames.AbstractFrame;
import io.euhedral_execution.core.generics.LatticeTerminal;
import io.euhedral_execution.inference.core.host.HostFrames;
import io.euhedral_execution.inference.core.scheduling.graph.FrameSeeds;
import io.euhedral_execution.inference.core.scheduling.graph.QwenExecutionSource;
import io.euhedral_execution.inference.core.tokenizer.QwenTokenizer;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Supplier;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/// Host work that is not a quantum's device stage, as frames on the lattice's workers: prompt tokenization, the
/// prefix cache's pieces, the work a caller hands over, and everything a model's runtime describes through
/// [HostFrames]. Both inference runtimes (the dense graphs' and Flash-Next's) share it, so there is one way a
/// host task becomes a frame, one source it is published through, and one shutdown.
///
/// The source attaches to the lattice on first use and stays attached until [#close], which closes admission and
/// returns once every admitted task finished. Every task is an admitted unit of that source: a task that hands
/// on to another publishes the next before it ends, so the source cannot complete between them.
public final class HostTasks implements HostFrames, AutoCloseable {

    private static final Logger LOG = LoggerFactory.getLogger(HostTasks.class);

    private final LatticeTerminal lattice;
    private final Object lock = new Object();
    private final long seedBase = FrameSeeds.forHostWork().next();
    private final AtomicLong nextSeed = new AtomicLong();
    private QwenExecutionSource source;
    private boolean closed;

    public HostTasks(LatticeTerminal lattice) {
        this.lattice = Objects.requireNonNull(lattice, "lattice");
    }

    /// Tokenizes `text` on the lattice's workers (PromptTokenization), with the BOS/EOS tokens of
    /// tokenizer_config.json when `modelSpecialTokens`. The future completes on a worker.
    public CompletableFuture<int[]> tokenize(QwenTokenizer tokenizer, String text, boolean modelSpecialTokens) {
        QwenExecutionSource tasks = source();
        admit(tasks);
        return PromptTokenization.start(tokenizer, text, modelSpecialTokens, tasks::publish, tasks::terminated);
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
        QwenExecutionSource tasks = source();
        admit(tasks);
        try {
            TaskFrame frame = new TaskFrame(task, tasks, this.seedBase + this.nextSeed.getAndIncrement());
            if (fromCallback) tasks.publishFromCallback(frame);
            else tasks.publish(frame);
        } catch (RuntimeException | Error failure) {
            tasks.terminated();
            throw failure;
        }
    }

    /// Admission is refused once the host work closed and nothing is left running; while work drains, the tasks it
    /// publishes may continue.
    private void admit(QwenExecutionSource tasks) {
        boolean draining;
        synchronized (this.lock) {
            draining = this.closed;
        }
        if (draining) tasks.admitDuringDrain();
        else tasks.admit();
    }

    private QwenExecutionSource source() {
        QwenExecutionSource tasks;
        synchronized (this.lock) {
            tasks = this.source;
            if (tasks != null) return tasks;
            if (this.closed) throw new IllegalStateException("host tasks are closed");
            tasks = new QwenExecutionSource();
            this.source = tasks;
        }
        // Attached once; a close from here on completes it.
        this.lattice.addUpstream(tasks);
        return tasks;
    }

    /// Tasks admitted and not yet finished.
    public int activeTasks() {
        QwenExecutionSource tasks;
        synchronized (this.lock) {
            tasks = this.source;
        }
        return tasks == null ? 0 : tasks.activeGraphs();
    }

    /// Whether the task source is still attached to the lattice.
    public boolean isAttached() {
        QwenExecutionSource tasks;
        synchronized (this.lock) {
            tasks = this.source;
        }
        return tasks != null && tasks.isAttached();
    }

    /// Closes admission, waits until every admitted task finished and the lattice detached the source.
    @Override
    public void close() {
        QwenExecutionSource tasks;
        synchronized (this.lock) {
            if (this.closed) return;
            this.closed = true;
            tasks = this.source;
        }
        if (tasks == null) return;
        tasks.completeGracefully();
        tasks.awaitTermination();
    }

    /// One task as a frame. A task never throws into the lattice: an exception ends the task through
    /// [HostFrames.Task#failed].
    private static final class TaskFrame extends AbstractFrame {
        private final Task task;
        private final QwenExecutionSource source;
        private final AtomicBoolean finished = new AtomicBoolean();

        TaskFrame(Task task, QwenExecutionSource source, long seed) {
            super(FrameSeeds.ID_HASH);
            randomizeHash(seed);
            this.task = task;
            this.source = source;
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
            if (this.finished.compareAndSet(false, true)) this.source.terminated();
        }
    }
}
