package io.euhedral_execution.inference.core.model.qwen38;

import java.util.List;
import java.util.concurrent.CompletionException;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.locks.LockSupport;
import java.util.function.Consumer;

/// Decoded text on its way from the generating workers to the thread that called the blocking `generate`. Workers
/// append without waiting and wake that thread; only it parks. Construct it on that thread.
final class Emission {
    private static final Object END = new Object();
    private final ConcurrentLinkedQueue<Object> queue = new ConcurrentLinkedQueue<>();
    private final Thread caller = Thread.currentThread();
    private List<Integer> tokens;
    private Throwable failure;

    void text(String text) {
        if (text.isEmpty()) return;
        this.queue.add(text);
        LockSupport.unpark(this.caller);
    }

    /// The generation ended; the queue publishes the result to the caller.
    void end(List<Integer> tokens, Throwable failure) {
        this.tokens = tokens;
        this.failure = failure;
        this.queue.add(END);
        LockSupport.unpark(this.caller);
    }

    /// Hands every text to `output` until the generation ended, then returns its tokens. When `output` throws or the
    /// calling thread is interrupted, `cancel` stops the generation, which is still awaited.
    List<Integer> drain(Consumer<String> output, Runnable cancel) throws InterruptedException, ExecutionException {
        Throwable outputFailure = null;
        boolean interrupted = false;
        while (true) {
            Object item = this.queue.poll();
            if (item == null) {
                LockSupport.park(this);
                if (Thread.interrupted()) {
                    if (!interrupted) cancel.run();
                    interrupted = true;
                }
                continue;
            }
            if (item == END) break;
            if (outputFailure != null || interrupted) continue;
            try {
                output.accept((String) item);
            } catch (RuntimeException | Error failure) {
                outputFailure = failure;
                cancel.run();
            }
        }
        if (interrupted) {
            Thread.currentThread().interrupt();
            throw new InterruptedException("generation was interrupted");
        }
        if (outputFailure instanceof RuntimeException runtimeFailure) throw runtimeFailure;
        if (outputFailure instanceof Error error) throw error;
        if (this.failure == null) return this.tokens;
        Throwable cause = this.failure instanceof CompletionException wrapped && wrapped.getCause() != null
                ? wrapped.getCause()
                : this.failure;
        if (cause instanceof ExecutionException executionFailure) throw executionFailure;
        if (cause instanceof RuntimeException runtimeFailure) throw runtimeFailure;
        if (cause instanceof Error error) throw error;
        throw new ExecutionException(cause);
    }
}
