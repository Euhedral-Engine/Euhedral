package io.euhedral_execution.inference.core.model_loader.qwen4.expert;

import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Function;

/// An [ExpertTransfer] with no GPU: it copies a record into "device" memory (native memory handed out by a
/// [HostBackedGpu]) on its own threads, after an injectable delay and gate, and can fail on demand. It counts
/// everything the cache asks of it, and detects two overlapping transfers into one slot.
final class SyntheticExpertTransfer implements ExpertTransfer {

    /// One transfer as the cache requested it.
    record Request(long deviceAddress, long bytes, DeviceFence waitFor, int sequence) {}

    private final ExecutorService threads;
    private final ConcurrentLinkedQueue<Request> requests = new ConcurrentLinkedQueue<>();
    private final Set<Long> active = ConcurrentHashMap.newKeySet();
    private final AtomicInteger started = new AtomicInteger();
    private final AtomicInteger completed = new AtomicInteger();
    private final AtomicInteger failed = new AtomicInteger();
    private final AtomicInteger overlaps = new AtomicInteger();
    private final AtomicInteger closes = new AtomicInteger();
    private final AtomicLong bytes = new AtomicLong();
    private volatile long delayNanos;
    private volatile Semaphore gate;
    private volatile Function<Request, Throwable> failure = request -> null;
    private volatile RuntimeException startFailure;
    private volatile boolean completeInline;

    SyntheticExpertTransfer() {
        this.threads = Executors.newCachedThreadPool(runnable -> {
            Thread thread = new Thread(runnable, "synthetic-transfer");
            thread.setDaemon(true);
            return thread;
        });
    }

    /// Every transfer waits this long before it copies.
    void delay(long nanos) {
        this.delayNanos = nanos;
    }

    /// Every transfer takes one permit before it copies: with no permits released, transfers stay in flight.
    void gate(Semaphore gate) {
        this.gate = gate;
    }

    /// Decides per transfer whether it fails (non-null) after writing half of the record.
    void failWhen(Function<Request, Throwable> failure) {
        this.failure = failure;
    }

    /// `start` throws this, with the record closed, instead of transferring (null to stop).
    void failStart(RuntimeException failure) {
        this.startFailure = failure;
    }

    /// Complete on the calling thread before `start` returns (only without delay and gate).
    void completeInline(boolean inline) {
        this.completeInline = inline;
    }

    @Override
    public void start(HostRecord record, long deviceAddress, DeviceFence waitFor, Completion done)
            throws InterruptedException {
        RuntimeException refusal = this.startFailure;
        if (refusal != null) {
            record.close();
            throw refusal;
        }
        Request request = new Request(deviceAddress, record.byteSize(), waitFor, this.started.incrementAndGet());
        this.requests.add(request);
        if (!this.active.add(deviceAddress)) this.overlaps.incrementAndGet();
        if (this.completeInline) transfer(request, record, done);
        else this.threads.execute(() -> transfer(request, record, done));
    }

    private void transfer(Request request, HostRecord record, Completion done) {
        Throwable outcome = null;
        try {
            long delay = this.delayNanos;
            if (delay > 0) TimeUnit.NANOSECONDS.sleep(delay);
            Semaphore held = this.gate;
            if (held != null && !held.tryAcquire(20, TimeUnit.SECONDS)) throw new AssertionError("gate never opened");
            outcome = this.failure.apply(request);
            MemorySegment destination =
                    MemorySegment.ofAddress(request.deviceAddress()).reinterpret(request.bytes());
            MemorySegment source = record.segment();
            if (outcome == null) destination.copyFrom(source);
            else {
                // A failed copy leaves a torn slot behind.
                long half = request.bytes() / 2;
                destination.asSlice(0, half).copyFrom(source.asSlice(0, half));
            }
        } catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
            outcome = interrupted;
        } catch (Throwable broken) {
            outcome = broken;
        } finally {
            record.close();
            this.active.remove(request.deviceAddress());
        }
        if (outcome == null) {
            this.completed.incrementAndGet();
            this.bytes.addAndGet(request.bytes());
        } else this.failed.incrementAndGet();
        done.complete(outcome);
    }

    int started() {
        return this.started.get();
    }

    /// Transfers that completed successfully.
    int completed() {
        return this.completed.get();
    }

    int failed() {
        return this.failed.get();
    }

    /// Transfers started into a slot another transfer was still writing.
    int overlaps() {
        return this.overlaps.get();
    }

    int closes() {
        return this.closes.get();
    }

    long bytes() {
        return this.bytes.get();
    }

    List<Request> requests() {
        return List.copyOf(this.requests);
    }

    /// Waits until `count` transfers have started.
    boolean awaitStarted(int count, long timeoutMillis) throws InterruptedException {
        long deadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(timeoutMillis);
        while (this.started.get() < count) {
            if (System.nanoTime() > deadline) return false;
            Thread.sleep(1);
        }
        return true;
    }

    @Override
    public void close() {
        this.closes.incrementAndGet();
        this.threads.shutdown();
        try {
            if (!this.threads.awaitTermination(30, TimeUnit.SECONDS))
                throw new IllegalStateException("synthetic transfers are still running");
        } catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
        }
    }
}
