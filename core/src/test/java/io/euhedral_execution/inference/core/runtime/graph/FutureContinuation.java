package io.euhedral_execution.inference.core.runtime.graph;

import io.euhedral_execution.core.frames.AbstractFrame;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.function.Function;

/// Where a test that waits on its own thread meets a quantum's continuation: completes a future with what `read`
/// takes from the concluded quantum.
public final class FutureContinuation<T> extends AbstractFrame implements AbstractQuantum.Continuation {
    private final Function<AbstractQuantum, T> read;
    private final CompletableFuture<T> future = new CompletableFuture<>();
    private AbstractQuantum quantum;

    public FutureContinuation(Function<AbstractQuantum, T> read) {
        super(FrameSeeds.ID_HASH);
        randomizeHash(FrameSeeds.forHostWork().next());
        this.read = Objects.requireNonNull(read, "read");
    }

    public CompletableFuture<T> future() {
        return this.future;
    }

    @Override
    public void concluded(AbstractQuantum quantum) {
        this.quantum = quantum;
    }

    @Override
    public void execute() {
        try {
            this.future.complete(this.read.apply(this.quantum));
        } catch (Throwable failure) {
            this.future.completeExceptionally(failure);
        }
    }

    @Override
    public void doFinallyWithError(Throwable rejection) {
        execute();
    }
}
