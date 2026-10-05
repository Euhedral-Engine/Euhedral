package io.euhedral_execution.inference.core.model_loader.qwen4.expert;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.model_loader.qwen4.expert.CacheRig.StoreKind;
import io.euhedral_execution.inference.core.model_loader.qwen4.expert.CacheRig.TransferKind;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.LockSupport;
import java.util.stream.Stream;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

/// Thousands of acquisitions over many access patterns, from many threads, with every lease's bytes checked
/// against the file when it is obtained and again just before it is closed (a slot refilled under an open lease
/// would show), and the cache's bookkeeping checked while the traces run and after.
class ExpertCacheStressTest {
    private static final int THREADS = 8;

    @TempDir
    Path directory;

    private ExpertFixture fixture;
    private ExecutorService pool;
    private final List<int[]> allKeys = new ArrayList<>();

    @BeforeEach
    void setUp() throws IOException {
        this.fixture = ExpertFixture.standard(this.directory, 41);
        this.pool = Executors.newFixedThreadPool(THREADS + 2);
        for (int bank = 0; bank < this.fixture.banks.length; bank++)
            for (int expert = 0; expert < this.fixture.expertCount(bank); expert++)
                this.allKeys.add(new int[] {bank, expert});
    }

    @AfterEach
    void tearDown() {
        this.pool.shutdownNow();
    }

    static Stream<Arguments> combos() {
        return Stream.of(
                Arguments.of(StoreKind.ARENA, TransferKind.SYNTHETIC),
                Arguments.of(StoreKind.FILE, TransferKind.GPU_ASYNC),
                Arguments.of(StoreKind.ARENA, TransferKind.GPU_INLINE));
    }

    /// Chooses the next `(bank, expert)` for a thread.
    @FunctionalInterface
    private interface Trace {
        int[] next(Random random, int thread, int step);
    }

    private int[] uniform(Random random) {
        return this.allKeys.get(random.nextInt(this.allKeys.size()));
    }

    /// Runs `threads` workers of `steps` acquisitions each and returns the total.
    private int run(CacheRig rig, int threads, int steps, long seed, Trace trace) throws Exception {
        AtomicBoolean running = new AtomicBoolean(true);
        AtomicInteger acquired = new AtomicInteger();
        Future<?> monitor = this.pool.submit(() -> {
            while (running.get()) {
                rig.cache.checkInvariants();
                assertTrue(rig.cache.openLeaseCount() <= threads);
                LockSupport.parkNanos(1_000_000);
            }
            return null;
        });
        List<Future<?>> workers = new ArrayList<>();
        for (int t = 0; t < threads; t++) {
            int thread = t;
            workers.add(this.pool.submit(() -> {
                Random random = new Random(seed * 31 + thread);
                for (int step = 0; step < steps; step++) {
                    int[] key = trace.next(random, thread, step);
                    try (ExpertLease lease = rig.cache.acquire(key[0], key[1])) {
                        acquired.incrementAndGet();
                        rig.assertLeaseBytes(lease);
                        if (random.nextInt(6) == 0) LockSupport.parkNanos(20_000);
                        else if (random.nextInt(4) == 0) Thread.yield();
                        assertTrue(lease.isValid());
                        rig.assertLeaseBytes(lease);
                    }
                }
                return null;
            }));
        }
        try {
            for (Future<?> worker : workers) worker.get(120, TimeUnit.SECONDS);
        } finally {
            running.set(false);
            monitor.get(30, TimeUnit.SECONDS);
        }
        rig.cache.checkQuiescent();
        assertEquals(0, rig.cache.openLeaseCount());
        assertEquals(threads * steps, acquired.get());
        ExpertCacheStats.Snapshot stats = rig.cache.stats().snapshot();
        assertEquals(threads * steps, stats.requests(), "every request was a hit, a miss or a joined transfer");
        assertEquals(0, stats.failedTransfers());
        assertTrue(stats.peakSlotsInUse() <= rig.cache.slotCount());
        return acquired.get();
    }

    @ParameterizedTest
    @MethodSource("combos")
    void repeatedHotExperts(StoreKind store, TransferKind transfer) throws Exception {
        try (CacheRig rig = new CacheRig(this.fixture, store, transfer, 6)) {
            int[][] hot = {{0, 2}, {2, 9}, {4, 4}};
            run(
                    rig,
                    THREADS,
                    400,
                    1,
                    (random, thread, step) -> random.nextInt(10) < 9 ? hot[random.nextInt(3)] : uniform(random));
            ExpertCacheStats.Snapshot stats = rig.cache.stats().snapshot();
            assertTrue(stats.hits() > stats.misses() * 5, "hot experts stay resident: " + stats);
        }
    }

    @ParameterizedTest
    @MethodSource("combos")
    void uniformRandomOverAllExperts(StoreKind store, TransferKind transfer) throws Exception {
        try (CacheRig rig = new CacheRig(this.fixture, store, transfer, 12)) {
            run(rig, THREADS, 400, 2, (random, thread, step) -> uniform(random));
            assertTrue(rig.cache.stats().snapshot().evictions() > 0);
        }
    }

    @ParameterizedTest
    @MethodSource("combos")
    void layerLocalHotSets(StoreKind store, TransferKind transfer) throws Exception {
        try (CacheRig rig = new CacheRig(this.fixture, store, transfer, 9)) {
            // Each thread works through the layers in order, using a few experts of each many times, like the
            // routed experts of consecutive tokens.
            run(rig, THREADS, 400, 3, (random, thread, step) -> {
                int bank = (step / 20 + thread) % this.fixture.banks.length;
                int expert = (bank * 7 + random.nextInt(4) * 3) % this.fixture.expertCount(bank);
                return new int[] {bank, expert};
            });
            assertTrue(rig.cache.stats().snapshot().hits() > 0);
        }
    }

    @ParameterizedTest
    @MethodSource("combos")
    void thrashingWithTheWorkingSetOneLargerThanTheCache(StoreKind store, TransferKind transfer) throws Exception {
        try (CacheRig rig = new CacheRig(this.fixture, store, transfer, 4)) {
            // One thread, cyclic over slots + 1 experts: least recently used evicts exactly the next one needed.
            int steps = 1000;
            run(rig, 1, steps, 4, (random, thread, step) -> this.allKeys.get(step % 5));
            ExpertCacheStats.Snapshot stats = rig.cache.stats().snapshot();
            assertEquals(steps, stats.misses());
            assertEquals(0, stats.hits());
            assertEquals(steps - 4, stats.evictions());
        }
        try (CacheRig rig = new CacheRig(this.fixture, store, transfer, 4)) {
            // Many threads over the same working set: no exact counts, but every byte and invariant holds.
            run(rig, THREADS, 250, 5, (random, thread, step) -> this.allKeys.get((step + thread) % 5));
        }
    }

    @ParameterizedTest
    @MethodSource("combos")
    void fullEvictionCycles(StoreKind store, TransferKind transfer) throws Exception {
        try (CacheRig rig = new CacheRig(this.fixture, store, transfer, 4)) {
            // A sweep through all 40 experts, over and over, on four slots: every access evicts.
            int steps = 40 * 25;
            run(rig, 1, steps, 6, (random, thread, step) -> this.allKeys.get(step % this.allKeys.size()));
            ExpertCacheStats.Snapshot stats = rig.cache.stats().snapshot();
            assertEquals(steps, stats.misses());
            assertEquals(steps - 4, stats.evictions());
        }
        try (CacheRig rig = new CacheRig(this.fixture, store, transfer, 5)) {
            run(
                    rig,
                    THREADS,
                    250,
                    7,
                    (random, thread, step) -> this.allKeys.get((step * 3 + thread * 5) % this.allKeys.size()));
            assertTrue(rig.cache.stats().snapshot().evictions() > 0);
        }
    }

    @ParameterizedTest
    @MethodSource("combos")
    void smallCacheWithAsManySlotsAsConcurrentLeases(StoreKind store, TransferKind transfer) throws Exception {
        try (CacheRig rig = new CacheRig(this.fixture, store, transfer, THREADS)) {
            run(rig, THREADS, 400, 8, (random, thread, step) -> uniform(random));
            // No thread ever waited for a slot another thread was holding for good: leases are short.
            assertTrue(rig.cache.stats().snapshot().peakSlotsInUse() <= THREADS);
        }
    }

    @ParameterizedTest
    @MethodSource("combos")
    void largeCacheHoldsEverythingSoEachExpertLoadsOnce(StoreKind store, TransferKind transfer) throws Exception {
        try (CacheRig rig = new CacheRig(this.fixture, store, transfer, this.allKeys.size())) {
            run(rig, THREADS, 400, 9, (random, thread, step) -> uniform(random));
            ExpertCacheStats.Snapshot stats = rig.cache.stats().snapshot();
            assertEquals(0, stats.evictions());
            assertTrue(stats.misses() <= this.allKeys.size());
            assertEquals(stats.misses(), rig.transfers(), "one transfer per distinct expert");
            long distinct = this.allKeys.stream()
                    .filter(k -> rig.cache.isResident(k[0], k[1]))
                    .count();
            assertEquals(stats.misses(), distinct);
        }
    }

    @ParameterizedTest
    @MethodSource("combos")
    void sameExpertBurstsCoalesceIntoOneTransfer(StoreKind store, TransferKind transfer) throws Exception {
        try (CacheRig rig = new CacheRig(this.fixture, store, transfer, 3)) {
            int rounds = 120;
            CyclicBarrier start = new CyclicBarrier(THREADS);
            CyclicBarrier held = new CyclicBarrier(THREADS);
            CyclicBarrier finish = new CyclicBarrier(THREADS);
            List<Future<?>> workers = new ArrayList<>();
            AtomicInteger distinctAddresses = new AtomicInteger();
            long[] addressOfRound = new long[THREADS];
            for (int t = 0; t < THREADS; t++) {
                int thread = t;
                workers.add(this.pool.submit(() -> {
                    for (int round = 0; round < rounds; round++) {
                        int[] key = this.allKeys.get((round * 7) % this.allKeys.size());
                        start.await(30, TimeUnit.SECONDS);
                        try (ExpertLease lease = rig.cache.acquire(key[0], key[1])) {
                            rig.assertLeaseBytes(lease);
                            addressOfRound[thread] = lease.deviceAddress();
                            held.await(30, TimeUnit.SECONDS); // all eight leases are open at once
                            if (thread == 0) {
                                for (long address : addressOfRound) assertEquals(addressOfRound[0], address);
                                distinctAddresses.incrementAndGet();
                            }
                            rig.assertLeaseBytes(lease);
                        }
                        finish.await(30, TimeUnit.SECONDS);
                    }
                    return null;
                }));
            }
            for (Future<?> worker : workers) worker.get(120, TimeUnit.SECONDS);
            ExpertCacheStats.Snapshot stats = rig.cache.stats().snapshot();
            assertEquals(rounds, distinctAddresses.get());
            assertTrue(stats.misses() <= rounds, "at most one transfer per round: " + stats);
            assertEquals((long) rounds * THREADS, stats.requests());
            assertEquals(stats.misses(), rig.transfers());
            assertTrue(stats.coalescedRequests() + stats.hits() >= rounds * (THREADS - 1L));
            rig.cache.checkQuiescent();
        }
    }

    @Test
    void failuresCancellationsAndTimeoutsUnderLoadNeverLeakOwnership() throws Exception {
        try (CacheRig rig = new CacheRig(this.fixture, StoreKind.ARENA, TransferKind.SYNTHETIC, 6)) {
            Random faults = new Random(99);
            rig.synthetic.failWhen(request -> {
                synchronized (faults) {
                    return faults.nextInt(25) == 0 ? new IllegalStateException("injected transfer failure") : null;
                }
            });
            rig.synthetic.delay(20_000);
            AtomicInteger succeeded = new AtomicInteger();
            AtomicInteger failed = new AtomicInteger();
            AtomicInteger timedOut = new AtomicInteger();
            AtomicInteger interrupted = new AtomicInteger();
            AtomicBoolean running = new AtomicBoolean(true);
            Thread[] threads = new Thread[THREADS];
            CyclicBarrier ready = new CyclicBarrier(THREADS + 1);
            Future<?> canceller = this.pool.submit(() -> {
                ready.await(30, TimeUnit.SECONDS);
                Random random = new Random(5);
                while (running.get()) {
                    Thread target = threads[random.nextInt(THREADS)];
                    if (target != null) target.interrupt();
                    LockSupport.parkNanos(150_000);
                }
                return null;
            });
            List<Future<?>> workers = new ArrayList<>();
            for (int t = 0; t < THREADS; t++) {
                int thread = t;
                workers.add(this.pool.submit(() -> {
                    threads[thread] = Thread.currentThread();
                    ready.await(30, TimeUnit.SECONDS);
                    Random random = new Random(1000 + thread);
                    for (int step = 0; step < 600; step++) {
                        int[] key = uniform(random);
                        try (ExpertLease lease = random.nextInt(3) == 0
                                ? rig.cache.acquire(key[0], key[1], random.nextInt(400_000))
                                : rig.cache.acquire(key[0], key[1])) {
                            rig.assertLeaseBytes(lease);
                            succeeded.incrementAndGet();
                        } catch (ExpertTransferException failure) {
                            failed.incrementAndGet();
                        } catch (TimeoutException timeout) {
                            timedOut.incrementAndGet();
                        } catch (InterruptedException interrupt) {
                            interrupted.incrementAndGet();
                        }
                    }
                    return null;
                }));
            }
            try {
                for (Future<?> worker : workers) worker.get(120, TimeUnit.SECONDS);
            } finally {
                running.set(false);
                canceller.get(30, TimeUnit.SECONDS);
            }
            Thread.interrupted();
            // Late transfers may still be completing: wait for the cache to settle.
            long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(10);
            while (true) {
                try {
                    rig.cache.checkQuiescent();
                    break;
                } catch (IllegalStateException loading) {
                    if (System.nanoTime() > deadline) throw loading;
                    Thread.sleep(2);
                }
            }
            assertEquals(0, rig.cache.openLeaseCount());
            assertTrue(succeeded.get() > 0 && failed.get() > 0, "succeeded " + succeeded + ", failed " + failed);
            assertTrue(
                    timedOut.get() + interrupted.get() > 0, "timed out " + timedOut + ", interrupted " + interrupted);
            // The cache is fully usable afterwards, and nothing is pinned.
            rig.synthetic.failWhen(request -> null);
            rig.synthetic.delay(0);
            for (int[] key : this.allKeys) {
                try (ExpertLease lease = rig.cache.acquire(key[0], key[1])) {
                    rig.assertLeaseBytes(lease);
                }
            }
            rig.cache.checkQuiescent();
            assertEquals(0, rig.synthetic.overlaps(), "no two transfers ever wrote one slot at once");
        }
    }

    @Test
    void anInterruptAtAnyMomentNeverLosesAFileStoreSlotOrTheDirectory() throws Exception {
        // File-backed staging and an asynchronous stream: interrupts land in the store, the transfer and the wait.
        try (CacheRig rig =
                new CacheRig(this.fixture, StoreKind.FILE, TransferKind.GPU_ASYNC, 4, 2, System::nanoTime)) {
            AtomicBoolean running = new AtomicBoolean(true);
            Thread[] threads = new Thread[4];
            CyclicBarrier ready = new CyclicBarrier(5);
            Future<?> canceller = this.pool.submit(() -> {
                ready.await(30, TimeUnit.SECONDS);
                Random random = new Random(7);
                while (running.get()) {
                    Thread target = threads[random.nextInt(4)];
                    if (target != null) target.interrupt();
                    LockSupport.parkNanos(100_000);
                }
                return null;
            });
            List<Future<?>> workers = new ArrayList<>();
            AtomicInteger completed = new AtomicInteger();
            for (int t = 0; t < 4; t++) {
                int thread = t;
                workers.add(this.pool.submit(() -> {
                    threads[thread] = Thread.currentThread();
                    ready.await(30, TimeUnit.SECONDS);
                    Random random = new Random(2000 + thread);
                    for (int step = 0; step < 500; step++) {
                        int[] key = uniform(random);
                        try (ExpertLease lease = rig.cache.acquire(key[0], key[1])) {
                            rig.assertLeaseBytes(lease);
                            completed.incrementAndGet();
                        } catch (InterruptedException interrupt) {
                            // cancelled: nothing may remain claimed
                        }
                    }
                    return null;
                }));
            }
            try {
                for (Future<?> worker : workers) worker.get(120, TimeUnit.SECONDS);
            } finally {
                running.set(false);
                canceller.get(30, TimeUnit.SECONDS);
            }
            long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(10);
            while (true) {
                try {
                    rig.cache.checkQuiescent();
                    break;
                } catch (IllegalStateException loading) {
                    if (System.nanoTime() > deadline) throw loading;
                    Thread.sleep(2);
                }
            }
            assertTrue(completed.get() > 0);
            assertEquals(2, ((FileExpertStore) rig.store).freeSlots(), "every staging slot came back");
            assertEquals(0, rig.cache.stats().snapshot().failedTransfers());
            for (int[] key : this.allKeys) {
                try (ExpertLease lease = rig.cache.acquire(key[0], key[1])) {
                    rig.assertLeaseBytes(lease);
                }
            }
        }
    }

    @Test
    void twentyThousandAcquisitionsKeepMemoryFlat() throws Exception {
        try (CacheRig rig = new CacheRig(this.fixture, StoreKind.ARENA, TransferKind.GPU_INLINE, 7)) {
            long hostBytes = rig.gpu.liveHostBytes();
            int total = run(rig, THREADS, 2500, 10, (random, thread, step) -> {
                int bank = (step / 31 + thread) % this.fixture.banks.length;
                return random.nextInt(3) == 0
                        ? uniform(random)
                        : new int[] {bank, random.nextInt(this.fixture.expertCount(bank))};
            });
            assertTrue(total >= 20_000);
            assertEquals(List.of(7 * this.fixture.slotBytes()), rig.gpu.deviceAllocationSizes());
            assertEquals(1, rig.gpu.liveDeviceAllocations());
            assertEquals(hostBytes, rig.gpu.liveHostBytes(), "host memory did not grow either");
        }
    }
}
