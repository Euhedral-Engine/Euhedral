package io.euhedral_execution.inference.core.model.qwen4;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.core.config.LatticeConfig;
import io.euhedral_execution.core.control_plane.ControlPlaneLattice;
import io.euhedral_execution.core.control_plane.ControlPlaneShard;
import io.euhedral_execution.core.impl.BaseCloneableObject;
import io.euhedral_execution.core.impl.DefaultExecutor;
import io.euhedral_execution.hardware_utils.topology.SystemInfo;
import io.euhedral_execution.inference.core.runtime.HostFrames;
import io.euhedral_execution.inference.core.runtime.HostTasks;
import java.time.Duration;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.api.parallel.Execution;
import org.junit.jupiter.api.parallel.ExecutionMode;

/// The architecture of Flash-Next host execution, on the real lattice: host work is frames on
/// lattice-attached sources, independent pieces run on several workers, waiting on an asynchronous dependency
/// parks a continuation and not a worker, a continuation published from a CUDA driver callback runs on a
/// worker, and closing drains and detaches.
@Execution(ExecutionMode.SAME_THREAD)
class LatticeHostTest {

    private static ControlPlaneLattice lattice(BitSet cpus) {
        var workers = new BaseCloneableObject(new DefaultExecutor());
        var shard = ControlPlaneShard.createBaseShard("Qwen4HostTestShard", workers);
        return ControlPlaneLattice.getOrCreate(
                new LatticeConfig("Qwen4HostTestLattice", cpus, Duration.ofSeconds(10), shard));
    }

    private static BitSet twoWorkerCpus() {
        BitSet cpus = new BitSet();
        BitSet cores = new BitSet();
        var physical = SystemInfo.getPCpuSet();
        int selected = 0;
        for (int cpu = physical.nextSetBit(0); cpu >= 0 && selected < 2; cpu = physical.nextSetBit(cpu + 1)) {
            var info = SystemInfo.getCpuInfo(cpu);
            if (info == null || cores.get(info.core())) continue;
            cores.set(info.core());
            cpus.set(cpu);
            selected++;
        }
        return cpus;
    }

    private static HostFrames.Task task(Runnable work, CompletableFuture<?> failed) {
        return HostFrames.of(work, failed::completeExceptionally);
    }

    @Test
    @Timeout(value = 60, unit = TimeUnit.SECONDS)
    void hostWorkRunsAsFramesOnALatticeAttachedSourceAndCloseDetaches() throws Exception {
        BitSet cpus = twoWorkerCpus();
        assumeTrue(cpus.cardinality() == 2);
        var lattice = lattice(cpus);
        try {
            lattice.start();
            HostTasks tasks = new HostTasks(lattice);
            assertFalse(tasks.isAttached(), "nothing attaches before the first task");
            String caller = Thread.currentThread().getName();
            String ran = tasks.onWorker(() -> Thread.currentThread().getName()).get(10, TimeUnit.SECONDS);
            assertNotEquals(caller, ran, "the work ran on a lattice worker");
            assertTrue(tasks.isAttached());
            tasks.close();
            assertFalse(tasks.isAttached(), "close detached the source");
            assertEquals(0, tasks.activeTasks());
            assertThrows(IllegalStateException.class, () -> tasks.run(HostFrames.of(() -> {}, f -> {})));
        } finally {
            lattice.close();
        }
    }

    @Test
    @Timeout(value = 60, unit = TimeUnit.SECONDS)
    void independentHostWorkRunsOnSeveralWorkersAtOnce() throws Exception {
        BitSet cpus = twoWorkerCpus();
        assumeTrue(cpus.cardinality() == 2);
        var lattice = lattice(cpus);
        try {
            lattice.start();
            HostTasks tasks = new HostTasks(lattice);
            Set<String> workers = ConcurrentHashMap.newKeySet();
            AtomicInteger running = new AtomicInteger();
            AtomicInteger peak = new AtomicInteger();
            List<CompletableFuture<Void>> done = new ArrayList<>();
            for (int i = 0; i < 24; i++)
                done.add(tasks.onWorker(() -> {
                    workers.add(Long.toString(Thread.currentThread().threadId()));
                    peak.accumulateAndGet(running.incrementAndGet(), Math::max);
                    try {
                        Thread.sleep(25);
                    } catch (InterruptedException interrupted) {
                        Thread.currentThread().interrupt();
                    }
                    running.decrementAndGet();
                    return null;
                }));
            for (var d : done) d.get(20, TimeUnit.SECONDS);
            assertTrue(workers.size() >= 2, "independent frames used " + workers);
            assertTrue(peak.get() >= 2, "frames overlapped: " + peak.get());
            tasks.close();
        } finally {
            lattice.close();
        }
    }

    @Test
    @Timeout(value = 60, unit = TimeUnit.SECONDS)
    void waitingOnAnAsynchronousDependencyOccupiesNoWorker() throws Exception {
        BitSet cpus = twoWorkerCpus();
        assumeTrue(cpus.cardinality() == 2);
        var lattice = lattice(cpus);
        try {
            lattice.start();
            HostTasks tasks = new HostTasks(lattice);
            // Four frames depend on completions that have not happened: each has an incoming edge per outstanding
            // operation, and the arrival that completes the set publishes it. Until then they exist only as a count:
            // more of them than the lattice has workers, and none is running or queued.
            int dependents = 4;
            AtomicInteger arrivals = new AtomicInteger(dependents * 3);
            CountDownLatch resumed = new CountDownLatch(dependents);
            // Unrelated host work still runs promptly on the two workers: the dependents hold none.
            List<CompletableFuture<Integer>> others = new ArrayList<>();
            for (int i = 0; i < 40; i++) {
                int value = i;
                others.add(tasks.onWorker(() -> value));
            }
            for (int i = 0; i < 40; i++) assertEquals(i, others.get(i).get(5, TimeUnit.SECONDS));
            assertEquals(dependents, resumed.getCount(), "the dependents have not run");
            // The operations complete, from a thread that only enqueues (a driver callback): the last arrival publishes
            // the dependents, and they run as frames.
            Thread completions = new Thread(() -> {
                for (int i = 0; i < dependents * 3; i++) {
                    if (arrivals.decrementAndGet() == 0)
                        for (int d = 0; d < dependents; d++)
                            tasks.runFromCallback(HostFrames.of(resumed::countDown, failure -> {}));
                }
            });
            completions.start();
            completions.join();
            assertTrue(resumed.await(10, TimeUnit.SECONDS), "every dependent ran once its last arrival came");
            for (Thread thread : Thread.getAllStackTraces().keySet()) {
                String name = thread.getName();
                assertFalse(name.contains("flash-next"), "no model-specific thread: " + name);
                assertFalse(name.contains("expert-transfer-completion"), name);
                assertFalse(name.contains("qwen4-expert-acquire"), name);
            }
            tasks.close();
        } finally {
            lattice.close();
        }
    }

    @Test
    @Timeout(value = 60, unit = TimeUnit.SECONDS)
    void aCompletionPublishedFromADriverCallbackThreadRunsOnAWorker() throws Exception {
        BitSet cpus = twoWorkerCpus();
        assumeTrue(cpus.cardinality() == 2);
        var lattice = lattice(cpus);
        try {
            lattice.start();
            HostTasks tasks = new HostTasks(lattice);
            CompletableFuture<String> ran = new CompletableFuture<>();
            Thread driver = new Thread(
                    () -> tasks.runFromCallback(
                            task(() -> ran.complete(Thread.currentThread().getName()), ran)),
                    "fake-cuda-driver");
            driver.start();
            driver.join();
            String worker = ran.get(10, TimeUnit.SECONDS);
            assertNotEquals("fake-cuda-driver", worker, "the callback only enqueued; a worker ran the frame");
            tasks.close();
        } finally {
            lattice.close();
        }
    }

    @Test
    @Timeout(value = 60, unit = TimeUnit.SECONDS)
    void closeWaitsForTasksThatPublishFurtherTasksAndLeavesNothingAttached() throws Exception {
        BitSet cpus = twoWorkerCpus();
        assumeTrue(cpus.cardinality() == 2);
        var lattice = lattice(cpus);
        try {
            lattice.start();
            HostTasks tasks = new HostTasks(lattice);
            AtomicInteger ran = new AtomicInteger();
            CompletableFuture<Void> failure = new CompletableFuture<>();
            // A chain of 200 continuations, each published by the one before it.
            HostFrames.Task[] link = new HostFrames.Task[1];
            AtomicInteger depth = new AtomicInteger();
            link[0] = task(
                    () -> {
                        ran.incrementAndGet();
                        if (depth.incrementAndGet() < 200) tasks.run(link[0]);
                    },
                    failure);
            tasks.run(link[0]);
            tasks.close();
            assertEquals(200, ran.get(), "close returned only after the whole chain ran");
            assertFalse(tasks.isAttached());
            assertEquals(0, tasks.activeTasks());
            assertFalse(failure.isCompletedExceptionally());
        } finally {
            lattice.close();
        }
    }
}
