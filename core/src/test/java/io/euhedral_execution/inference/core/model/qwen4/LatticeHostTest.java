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
import io.euhedral_execution.inference.core.runtime.LatticeTesting;
import io.euhedral_execution.inference.core.runtime.SteppingLattice;
import java.time.Duration;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
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
        assumeTrue(!cpus.isEmpty());
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
            LatticeTesting.closeAndAwait(lattice);
        }
    }

    /// Independent host tasks are all runnable at once, each on its own route, so Euhedral may spread them over its
    /// workers. Stepped on the test thread: no core count is assumed.
    @Test
    @Timeout(value = 60, unit = TimeUnit.SECONDS)
    void independentHostWorkIsRunnableTogetherOnDistinctRoutes() throws Exception {
        var lattice = new SteppingLattice();
        HostTasks tasks = new HostTasks(lattice);
        List<CompletableFuture<long[]>> done = new ArrayList<>();
        for (int i = 0; i < 24; i++) done.add(tasks.onWorker(SteppingLattice::running));
        lattice.run(lattice.ready());
        Set<Long> routes = new HashSet<>();
        for (var task : done) {
            assertTrue(task.isDone(), "every independent task ran in the first wave");
            long[] ran = task.get();
            assertEquals(1, ran[0]);
            routes.add(ran[1]);
        }
        assertEquals(24, routes.size(), "independent tasks share routes");
        lattice.closeWhileStepping(tasks::close);
    }

    @Test
    @Timeout(value = 60, unit = TimeUnit.SECONDS)
    void waitingOnAnAsynchronousDependencyOccupiesNoWorker() throws Exception {
        BitSet cpus = twoWorkerCpus();
        assumeTrue(!cpus.isEmpty());
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
            LatticeTesting.closeAndAwait(lattice);
        }
    }

    @Test
    @Timeout(value = 60, unit = TimeUnit.SECONDS)
    void aCompletionPublishedFromADriverCallbackThreadRunsOnAWorker() throws Exception {
        BitSet cpus = twoWorkerCpus();
        assumeTrue(!cpus.isEmpty());
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
            LatticeTesting.closeAndAwait(lattice);
        }
    }

    @Test
    @Timeout(value = 60, unit = TimeUnit.SECONDS)
    void closeWaitsForTasksThatPublishFurtherTasksAndLeavesNothingAttached() throws Exception {
        BitSet cpus = twoWorkerCpus();
        assumeTrue(!cpus.isEmpty());
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
            LatticeTesting.closeAndAwait(lattice);
        }
    }
}
