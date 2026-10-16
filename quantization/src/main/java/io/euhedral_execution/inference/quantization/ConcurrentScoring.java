package io.euhedral_execution.inference.quantization;

import io.euhedral_execution.inference.core.artifact.TensorDescriptor;
import io.euhedral_execution.inference.core.artifact.TensorHandle;
import io.euhedral_execution.inference.core.gpu.GpuMemory;
import io.euhedral_execution.inference.core.model.qwen38.ArtifactProfile;
import io.euhedral_execution.inference.core.model.qwen38.Execution;
import io.euhedral_execution.inference.core.model.qwen38.Qwen38Model;
import io.euhedral_execution.inference.core.model.qwen38.artifact.Artifact;
import io.euhedral_execution.inference.core.model.qwen38.loader.ObjectLoader;
import io.euhedral_execution.inference.core.model.qwen38.loader.WeightLoader;
import io.euhedral_execution.inference.quantization.mixed.HostPool;
import io.euhedral_execution.inference.quantization.mixed.Search;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

/// Scores as many candidates at once as the device's memory holds, over one [WeightCache] of the objects they
/// share: an object is read and uploaded once, whichever candidates use it. Each candidate keeps a model, an
/// execution (its own lane pool, graph pool and workspace), sequences and score buffers of its own; the runtime has
/// no process-wide state they would share (each [Execution] makes its own lake and runtime; the only statics are
/// atomic counters).
///
/// A candidate waits in submission order until [Admission] says its additional device bytes fit (its objects not in
/// the cache, its sequences' state and KV and its workspace). Weights are never host-backed for a candidate that
/// runs beside another; one that does not fit the device alone runs by itself, as [Scorer#score] would plan it.
///
/// Host work is frames on a pool of its own (reading and uploading one object, freeing one, working out what a
/// candidate needs); the decisions are taken by whichever thread last changed something, one at a time through an
/// ownership flag, never a lock. The execution of a candidate is started and closed by the thread that waits in
/// [#service], never on a lattice worker, which a close would block.
public final class ConcurrentScoring implements Search.Scoring, AutoCloseable {

    /// What a candidate needs, worked out in a frame: its objects with their descriptors, the batch of chunks it
    /// forces together, whether it fits the device entirely alone, and the bytes it adds beyond its weights.
    private record Plan(
            Map<WeightCache.Key, TensorDescriptor> descriptors, Admission.Need need, int batch, boolean fitsAlone) {}

    private static final class Entry {
        final Path path;
        final Artifact artifact;
        final Runnable started;
        final CompletableFuture<Scorer.Result> result = new CompletableFuture<>();
        CompletableFuture<Plan> plan;
        Qwen38Model model;
        Execution runtime;
        boolean cached;
        Scorer.Result score;
        Throwable failure;

        Entry(Path path, Artifact artifact, Runnable started) {
            this.path = path;
            this.artifact = artifact;
            this.started = started;
        }
    }

    private final Scorer scorer;
    private final List<Path> references;
    private final int rows, requestedBatch, maxConcurrent, context, chunks;
    private final long marginBytes;
    private final HostPool frames;
    private final WeightCache cache;
    private final ObjectLoader fromCache;

    private final ConcurrentLinkedQueue<Entry> inbox = new ConcurrentLinkedQueue<>();
    private final ConcurrentLinkedQueue<Entry> ended = new ConcurrentLinkedQueue<>();
    /// Work for the thread that waits.
    private final ConcurrentLinkedQueue<Runnable> mainWork = new ConcurrentLinkedQueue<>();
    private final AtomicReference<CompletableFuture<Void>> attention = new AtomicReference<>(new CompletableFuture<>());
    private final AtomicBoolean dirty = new AtomicBoolean(), owner = new AtomicBoolean();
    private final AtomicInteger live = new AtomicInteger();
    private volatile boolean closed;
    private final AtomicInteger peakRunning = new AtomicInteger();

    // Confined to whoever holds `owner`.
    private final ArrayDeque<Entry> waiting = new ArrayDeque<>();
    private int running;
    private long runningOtherBytes;
    private boolean exclusiveRunning;

    /// A scoring of `references` on `scorer`: `rows` rows per verification and `batch` chunks together (0 or less
    /// sizes it to the device, which leaves no room for a second candidate: give a batch to score several at once),
    /// at most `maxConcurrent` candidates at once (0: only memory limits), `marginBytes` left free, and `sinks`
    /// sinks for the host frames.
    public ConcurrentScoring(
            Scorer scorer, List<Path> references, int rows, int batch, int maxConcurrent, long marginBytes, int sinks)
            throws IOException {
        if (rows < 1 || rows > Scorer.MAX_ROWS)
            throw new IllegalArgumentException("--rows must be at least 1 and at most " + Scorer.MAX_ROWS);
        if (maxConcurrent < 0) throw new IllegalArgumentException("--max-concurrent must not be negative");
        if (marginBytes < 0) throw new IllegalArgumentException("--concurrent-margin-mib must not be negative");
        this.scorer = scorer;
        this.references = List.copyOf(references);
        this.rows = rows;
        this.requestedBatch = batch;
        this.maxConcurrent = maxConcurrent;
        this.marginBytes = marginBytes;
        int[] dimensions = Scorer.dimensions(references);
        this.context = dimensions[0];
        this.chunks = dimensions[1];
        this.frames = HostPool.startAttachingOnDemand(sinks);
        var device = scorer.device();
        this.cache = new WeightCache(new WeightCache.Backend() {
            @Override
            public TensorHandle load(Path file, TensorDescriptor descriptor) throws IOException {
                device.cuda.bindThread(); // the context only: a selected stream would stay on the worker and refuse its
                // next submission
                return ObjectLoader.OWNED.load(file, descriptor, device.gpu);
            }

            @Override
            public void free(TensorHandle handle) {
                device.cuda.bindThread(); // the context only: a selected stream would stay on the worker and refuse its
                // next submission
                device.gpu.free(handle.deviceAddress());
            }
        });
        this.fromCache = new ObjectLoader() {
            @Override
            public TensorHandle load(Path path, TensorDescriptor descriptor, GpuMemory memory) {
                return cache.handle(WeightCache.Key.of(path, descriptor));
            }

            @Override
            public boolean borrowed(TensorHandle handle) {
                return true;
            }
        };
    }

    @Override
    public Scorer.Result score(Path artifactPath, Artifact artifact) throws Exception {
        try {
            return submit(artifactPath, artifact, () -> {}).get();
        } catch (ExecutionException failure) {
            if (failure.getCause() instanceof Exception exception) throw exception;
            throw failure;
        }
    }

    /// The most candidates that were running at once.
    int peakRunning() {
        return peakRunning.get();
    }

    @Override
    public boolean concurrent() {
        return true;
    }

    @Override
    public CompletableFuture<Scorer.Result> submit(Path artifactPath, Artifact artifact, Runnable started) {
        if (closed) return CompletableFuture.failedFuture(new IllegalStateException("the scoring is closed"));
        Entry entry = new Entry(artifactPath, artifact, started);
        live.incrementAndGet();
        entry.plan = frames.submit(() -> plan(artifactPath, artifact));
        inbox.add(entry);
        entry.plan.whenComplete((plan, failure) -> pump());
        pump();
        return entry.result;
    }

    @Override
    public void service() {
        attention.set(new CompletableFuture<>());
        for (Runnable work; (work = mainWork.poll()) != null; ) work.run();
    }

    @Override
    public CompletableFuture<Void> attention() {
        return attention.get();
    }

    private void toMain(Runnable work) {
        mainWork.add(work);
        attention.get().complete(null);
    }

    // ------------------------------------------------------------------ what a candidate needs

    private Plan plan(Path path, Artifact artifact) {
        ArtifactProfile.Speculation none = ArtifactProfile.Speculation.NONE;
        Map<WeightCache.Key, TensorDescriptor> descriptors = new LinkedHashMap<>();
        Map<WeightCache.Key, Long> bytes = new LinkedHashMap<>();
        for (TensorDescriptor descriptor : artifact.tensors()) {
            if (!WeightLoader.uploads(descriptor.name(), none)) continue;
            WeightCache.Key key = WeightCache.Key.of(artifact.pathOf(path, descriptor), descriptor);
            descriptors.put(key, descriptor);
            bytes.put(key, descriptor.byteSize());
        }
        int batch = requestedBatch <= 0
                ? scorer.autoBatch(artifact, Residency.PLANNED, rows, context, chunks)
                : Scorer.checkedBatch(requestedBatch, rows, Scorer.WORKSPACE_ROWS);
        boolean fits = Residency.fitsResident(artifact, none, scorer.freeAtOpen(), batch, context);
        long other = Residency.otherBytes(artifact, none, batch, context);
        return new Plan(descriptors, new Admission.Need(bytes, other), batch, fits);
    }

    // ------------------------------------------------------------------ admission

    /// Runs the scheduling pass, by whoever finds work and no one doing it; a change that arrives while it runs
    /// is seen by its loop.
    private void pump() {
        dirty.set(true);
        while (dirty.get() && owner.compareAndSet(false, true)) {
            try {
                dirty.set(false);
                schedule();
            } catch (Throwable failure) {
                // A pass that cannot run leaves its candidates to fail rather than wait for ever.
                failWaiting(failure);
            } finally {
                owner.set(false);
            }
        }
    }

    private void schedule() {
        for (Entry entry; (entry = inbox.poll()) != null; ) waiting.add(entry);
        for (Entry entry; (entry = ended.poll()) != null; ) {
            running--;
            if (entry.cached) runningOtherBytes -= entry.plan.join().need().otherBytes();
            else exclusiveRunning = false;
        }
        for (Entry head; (head = waiting.peek()) != null; ) {
            if (closed) {
                waiting.poll();
                fail(head, new IllegalStateException("the scoring was closed before the candidate started"));
                continue;
            }
            if (!head.plan.isDone()) return;
            if (head.plan.isCompletedExceptionally()) {
                waiting.poll();
                fail(head, planFailure(head));
                continue;
            }
            Plan plan = head.plan.join();
            if (exclusiveRunning) return;
            boolean alone = running == 0;
            // A candidate that does not fit the device alone runs by itself; one that does waits for the memory
            // the frees in flight will return before it starts alone, so its fit is judged on a clear device.
            if (!plan.fitsAlone() && !alone) return;
            if (alone && cache.freesInFlight() > 0) return;
            long additional = plan.need().additional(cache::contains);
            long available = Admission.available(
                    scorer.device().gpu.deviceMemoryInfo().freeBytes(),
                    scorer.freeAtOpen(),
                    cache.residentBytes(),
                    runningOtherBytes,
                    marginBytes);
            if (plan.fitsAlone() && !Admission.admits(running, maxConcurrent, additional, available)) return;
            waiting.poll();
            start(head, plan);
        }
    }

    private static Throwable planFailure(Entry entry) {
        try {
            entry.plan.join();
            return new IllegalStateException("no plan");
        } catch (RuntimeException failure) {
            return failure.getCause() == null ? failure : failure.getCause();
        }
    }

    private void failWaiting(Throwable failure) {
        for (Entry entry; (entry = waiting.poll()) != null; ) fail(entry, failure);
    }

    private void fail(Entry entry, Throwable failure) {
        live.decrementAndGet();
        entry.result.completeExceptionally(failure);
        attention.get().complete(null);
    }

    private void start(Entry entry, Plan plan) {
        running++;
        peakRunning.accumulateAndGet(running, Math::max);
        entry.cached = plan.fitsAlone();
        entry.started.run();
        if (!entry.cached) {
            exclusiveRunning = true;
            toMain(() -> launch(entry, plan));
            return;
        }
        runningOtherBytes += plan.need().otherBytes();
        List<CompletableFuture<TensorHandle>> loads = new ArrayList<>();
        for (Map.Entry<WeightCache.Key, TensorDescriptor> object :
                plan.descriptors().entrySet()) loads.add(cache.acquire(object.getKey(), object.getValue(), frames));
        System.out.printf(
                Locale.ROOT,
                "concurrent scoring: %d running, %d MiB of objects on the device, %d MiB added%n",
                running,
                cache.residentBytes() >> 20,
                plan.need().additional(key -> false) >> 20);
        CompletableFuture.allOf(loads.toArray(new CompletableFuture<?>[0])).whenComplete((done, failure) -> {
            if (failure != null) entry.failure = failure;
            toMain(() -> launch(entry, plan));
        });
    }

    // ------------------------------------------------------------------ on the waiting thread

    /// Loads the model (every object is in the cache, or, for a candidate alone that does not fit, in host memory as
    /// the planner puts it) and starts forcing; the end is queued for the waiting thread as well.
    private void launch(Entry entry, Plan plan) {
        if (entry.failure != null) {
            finish(entry, plan);
            return;
        }
        try {
            var device = scorer.device();
            device.bindThread();
            ArtifactProfile.Speculation none = ArtifactProfile.Speculation.NONE;
            entry.model = entry.cached
                    ? Qwen38Model.load(entry.path, entry.artifact, device.gpu, none, Set.of(), Set.of(), fromCache)
                    : Qwen38Model.load(
                            entry.path,
                            entry.artifact,
                            device.gpu,
                            none,
                            Residency.hostBacked(
                                    entry.artifact,
                                    none,
                                    scorer.freeAtOpen(),
                                    Residency.PLANNED,
                                    plan.batch(),
                                    context));
            Scorer.Forced forced = scorer.force(
                    entry.model, entry.path, entry.artifact, references, rows, plan.batch(), (index, scored) -> {});
            entry.runtime = forced.runtime();
            forced.scored().whenComplete((scored, failure) -> {
                if (failure != null) entry.failure = failure;
                else entry.score = Scorer.result(scored);
                toMain(() -> finish(entry, plan));
            });
        } catch (Throwable failure) {
            entry.failure = failure;
            finish(entry, plan);
        }
    }

    /// Closes the candidate's execution and model, gives its objects back to the cache and ends it.
    private void finish(Entry entry, Plan plan) {
        try {
            if (entry.runtime != null) entry.runtime.close();
        } catch (Throwable failure) {
            if (entry.failure == null) entry.failure = failure;
        }
        try {
            if (entry.model != null) entry.model.close();
        } catch (Throwable failure) {
            if (entry.failure == null) entry.failure = failure;
        }
        List<CompletableFuture<Void>> freed = new ArrayList<>();
        if (entry.cached) for (WeightCache.Key key : plan.descriptors().keySet()) freed.add(cache.release(key, frames));
        CompletableFuture.allOf(freed.toArray(new CompletableFuture<?>[0])).whenComplete((done, failure) -> {
            Throwable problem = entry.failure != null ? entry.failure : failure;
            ended.add(entry);
            live.decrementAndGet();
            if (problem != null) entry.result.completeExceptionally(unwrap(problem));
            else entry.result.complete(entry.score);
            attention.get().complete(null);
            pump();
        });
    }

    private static Throwable unwrap(Throwable failure) {
        return failure instanceof CompletionException && failure.getCause() != null ? failure.getCause() : failure;
    }

    /// Waits for the running candidates to end (the waiting ones fail) and closes the host pool. Their executions
    /// cannot be stopped part-way, so this takes as long as the slowest of them has left.
    @Override
    public void close() {
        closed = true;
        pump();
        while (live.get() > 0) {
            service();
            try {
                attention.get().get(100, TimeUnit.MILLISECONDS);
            } catch (TimeoutException | ExecutionException ignored) {
                // Look again.
            } catch (InterruptedException interrupted) {
                Thread.currentThread().interrupt();
                break;
            }
            pump();
        }
        frames.close();
    }
}
