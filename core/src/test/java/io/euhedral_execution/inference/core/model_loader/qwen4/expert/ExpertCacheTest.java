package io.euhedral_execution.inference.core.model_loader.qwen4.expert;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.model_loader.qwen4.expert.CacheRig.StoreKind;
import io.euhedral_execution.inference.core.model_loader.qwen4.expert.CacheRig.TransferKind;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.BooleanSupplier;
import java.util.stream.Stream;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

class ExpertCacheTest {
    // Fixture keys: bank 0 has 8 experts, bank 1 has 3, bank 2 has 16, bank 3 has 1, bank 4 has 12.

    @TempDir
    Path directory;

    private ExpertFixture fixture;
    private ExecutorService pool;

    @BeforeEach
    void setUp() throws IOException {
        this.fixture = ExpertFixture.standard(this.directory, 31);
        this.pool = Executors.newCachedThreadPool();
    }

    @AfterEach
    void tearDown() {
        this.pool.shutdownNow();
    }

    static Stream<Arguments> combos() {
        List<Arguments> all = new ArrayList<>();
        for (StoreKind store : StoreKind.values())
            for (TransferKind transfer : TransferKind.values()) all.add(Arguments.of(store, transfer));
        return all.stream();
    }

    static Stream<Arguments> holdableCombos() {
        return combos().filter(arguments -> arguments.get()[1] != TransferKind.GPU_INLINE);
    }

    private CacheRig rig(StoreKind store, TransferKind transfer, int slots) throws IOException {
        return new CacheRig(this.fixture, store, transfer, slots);
    }

    private static void await(BooleanSupplier condition) throws InterruptedException {
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(10);
        while (!condition.getAsBoolean()) {
            if (System.nanoTime() > deadline) throw new AssertionError("condition never became true");
            Thread.sleep(1);
        }
    }

    private Future<ExpertLease> acquireAsync(CacheRig rig, int bank, int expert) {
        return this.pool.submit(() -> rig.cache.acquire(bank, expert));
    }

    // -------------------------------------------------------------- hits and misses

    @ParameterizedTest
    @MethodSource("combos")
    void aMissTransfersTheRecordAndAHitDoesNot(StoreKind store, TransferKind transfer) throws Exception {
        try (CacheRig rig = rig(store, transfer, 4)) {
            long address;
            try (ExpertLease lease = rig.cache.acquire(1, 2)) {
                rig.assertLeaseBytes(lease);
                assertEquals(this.fixture.banks[1].recordBytes(2), lease.byteSize());
                assertTrue(lease.isValid());
                address = lease.deviceAddress();
                assertTrue(address >= rig.cache.slabAddress()
                        && address + lease.byteSize() <= rig.cache.slabAddress() + rig.cache.capacityBytes());
            }
            assertEquals(1, rig.transfers());
            ExpertCacheStats.Snapshot afterMiss = rig.cache.stats().snapshot();
            assertEquals(1, afterMiss.misses());
            assertEquals(0, afterMiss.hits());
            assertEquals(this.fixture.banks[1].recordBytes(2), afterMiss.transferBytes());

            try (ExpertLease hit = rig.cache.acquire(1, 2)) {
                assertEquals(address, hit.deviceAddress(), "a hit returns the resident address");
                rig.assertLeaseBytes(hit);
            }
            assertEquals(1, rig.transfers(), "a hit transfers nothing");
            ExpertCacheStats.Snapshot snapshot = rig.cache.stats().snapshot();
            assertEquals(1, snapshot.hits());
            assertEquals(1, snapshot.misses());
            assertEquals(2, snapshot.requests());
            rig.cache.checkQuiescent();
        }
    }

    @ParameterizedTest
    @MethodSource("combos")
    void everyExpertLoadsCorrectlyThroughASmallCache(StoreKind store, TransferKind transfer) throws Exception {
        try (CacheRig rig = rig(store, transfer, 4)) {
            int count = 0;
            for (int bank = 0; bank < this.fixture.banks.length; bank++) {
                for (int expert = 0; expert < this.fixture.expertCount(bank); expert++) {
                    try (ExpertLease lease = rig.cache.acquire(bank, expert)) {
                        assertEquals(bank, lease.bank());
                        assertEquals(expert, lease.expert());
                        rig.assertLeaseBytes(lease);
                    }
                    count++;
                }
            }
            ExpertCacheStats.Snapshot stats = rig.cache.stats().snapshot();
            assertEquals(count, stats.misses());
            assertEquals(count - 4, stats.evictions());
            assertEquals(0, stats.failedTransfers());
            assertEquals(4, rig.cache.evictableSlots());
            rig.cache.checkQuiescent();
        }
    }

    @ParameterizedTest
    @MethodSource("combos")
    void leasesExposeProjectionAddressesAndRefuseUseAfterClose(StoreKind store, TransferKind transfer)
            throws Exception {
        try (CacheRig rig = rig(store, transfer, 2)) {
            ExpertLease lease = rig.cache.acquire(0, 5);
            assertEquals(lease.deviceAddress(), lease.projectionAddress("gate_up"));
            assertEquals(lease.deviceAddress() + 4096, lease.projectionAddress("down"));
            assertEquals(1024, lease.projection("down").byteSize());
            assertThrows(IllegalArgumentException.class, () -> lease.projectionAddress("up"));
            assertSame(this.fixture.banks[0].name(), lease.expertBank().name());
            lease.close();
            assertFalse(lease.isValid());
            assertThrows(IllegalStateException.class, lease::deviceAddress);
            assertThrows(IllegalStateException.class, () -> lease.projectionAddress("down"));
        }
    }

    @ParameterizedTest
    @MethodSource("combos")
    void closingALeaseTwiceReleasesOnePin(StoreKind store, TransferKind transfer) throws Exception {
        try (CacheRig rig = rig(store, transfer, 1)) {
            ExpertLease first = rig.cache.acquire(0, 0);
            ExpertLease second = rig.cache.acquire(0, 0);
            assertEquals(first.deviceAddress(), second.deviceAddress());
            assertEquals(2, rig.cache.openLeaseCount());
            first.close();
            first.close();
            first.close();
            assertEquals(1, rig.cache.openLeaseCount());
            // The one slot is still pinned by the second lease.
            assertThrows(TimeoutException.class, () -> rig.cache.acquire(0, 1, TimeUnit.MILLISECONDS.toNanos(50)));
            second.close();
            try (ExpertLease other = rig.cache.acquire(0, 1)) {
                rig.assertLeaseBytes(other);
            }
            rig.cache.checkQuiescent();
        }
    }

    @ParameterizedTest
    @MethodSource("combos")
    void tryAcquireOnlyHitsAndNeverStartsATransfer(StoreKind store, TransferKind transfer) throws Exception {
        try (CacheRig rig = rig(store, transfer, 3)) {
            assertEquals(Optional.empty(), rig.cache.tryAcquire(0, 0));
            assertEquals(0, rig.cache.stats().snapshot().requests(), "a refused attempt is not a request");
            rig.cache.acquire(0, 0).close();
            Optional<ExpertLease> hit = rig.cache.tryAcquire(0, 0);
            assertTrue(hit.isPresent());
            rig.assertLeaseBytes(hit.get());
            assertEquals(1, rig.transfers());
            hit.get().close();
            assertEquals(Optional.empty(), rig.cache.tryAcquire(0, 1));
            assertEquals(1, rig.transfers());
        }
    }

    @ParameterizedTest
    @MethodSource("holdableCombos")
    void tryAcquireDoesNotWaitForAnExpertThatIsLoading(StoreKind store, TransferKind transfer) throws Exception {
        try (CacheRig rig = rig(store, transfer, 3)) {
            rig.hold();
            Future<ExpertLease> loading = acquireAsync(rig, 2, 7);
            await(() -> rig.cache.stats().snapshot().misses() == 1);
            assertEquals(Optional.empty(), rig.cache.tryAcquire(2, 7));
            rig.releaseHeld();
            try (ExpertLease lease = loading.get(10, TimeUnit.SECONDS)) {
                rig.assertLeaseBytes(lease);
            }
            assertTrue(rig.cache.tryAcquire(2, 7).isPresent());
        }
    }

    // -------------------------------------------------------------- coalescing and pinning

    @ParameterizedTest
    @MethodSource("holdableCombos")
    void concurrentMissesForOneExpertShareOneTransfer(StoreKind store, TransferKind transfer) throws Exception {
        try (CacheRig rig = rig(store, transfer, 2)) {
            rig.hold();
            List<Future<ExpertLease>> futures = new ArrayList<>();
            for (int i = 0; i < 8; i++) futures.add(acquireAsync(rig, 2, 9));
            await(() -> {
                ExpertCacheStats.Snapshot stats = rig.cache.stats().snapshot();
                return stats.misses() == 1 && stats.coalescedRequests() == 7;
            });
            assertEquals(0, rig.transfers(), "nothing completed while held");
            rig.releaseHeld();
            List<ExpertLease> leases = new ArrayList<>();
            for (Future<ExpertLease> future : futures) leases.add(future.get(10, TimeUnit.SECONDS));
            assertEquals(1, rig.transfers(), "eight requests, one transfer");
            Set<Long> addresses = new HashSet<>();
            for (ExpertLease lease : leases) {
                addresses.add(lease.deviceAddress());
                rig.assertLeaseBytes(lease);
            }
            assertEquals(1, addresses.size(), "every waiter leases the same slot");
            assertEquals(8, rig.cache.openLeaseCount());
            rig.cache.checkInvariants();

            // Eight pins: the slot stays unevictable until the last lease closes.
            for (int i = 0; i < 7; i++) {
                leases.get(i).close();
                assertEquals(0, rig.cache.evictableSlots());
            }
            // One slot remains for other experts while the pinned one is held.
            try (ExpertLease other = rig.cache.acquire(0, 0)) {
                rig.assertLeaseBytes(other);
            }
            leases.get(7).close();
            assertEquals(2, rig.cache.evictableSlots());
            rig.cache.checkQuiescent();
            ExpertCacheStats.Snapshot stats = rig.cache.stats().snapshot();
            assertEquals(7, stats.coalescedRequests());
        }
    }

    @ParameterizedTest
    @MethodSource("combos")
    void aLeasedExpertIsNeverEvictedWhileOneSlotCyclesTheRest(StoreKind store, TransferKind transfer) throws Exception {
        try (CacheRig rig = rig(store, transfer, 4)) {
            List<ExpertLease> held = new ArrayList<>();
            held.add(rig.cache.acquire(0, 1));
            held.add(rig.cache.acquire(2, 3));
            held.add(rig.cache.acquire(4, 11));
            long[] addresses =
                    held.stream().mapToLong(ExpertLease::deviceAddress).toArray();
            // slotCount == leased + 1: every other expert must go through the single free slot.
            for (int round = 0; round < 3; round++) {
                for (int bank = 0; bank < this.fixture.banks.length; bank++) {
                    for (int expert = 0; expert < this.fixture.expertCount(bank); expert++) {
                        if (bank == 0 && expert == 1 || bank == 2 && expert == 3 || bank == 4 && expert == 11) continue;
                        try (ExpertLease lease = rig.cache.acquire(bank, expert)) {
                            rig.assertLeaseBytes(lease);
                            assertFalse(java.util.Arrays.stream(addresses).anyMatch(a -> a == lease.deviceAddress()));
                        }
                    }
                }
                for (int i = 0; i < held.size(); i++) {
                    assertTrue(held.get(i).isValid());
                    assertEquals(addresses[i], held.get(i).deviceAddress());
                    rig.assertLeaseBytes(held.get(i));
                }
            }
            for (ExpertLease lease : held) lease.close();
            rig.cache.checkQuiescent();
        }
    }

    @ParameterizedTest
    @MethodSource("holdableCombos")
    void aLoadingSlotIsNeverEvictedOrReused(StoreKind store, TransferKind transfer) throws Exception {
        try (CacheRig rig = rig(store, transfer, 2)) {
            rig.hold();
            Future<ExpertLease> first = acquireAsync(rig, 0, 0);
            Future<ExpertLease> second = acquireAsync(rig, 0, 1);
            await(() -> rig.cache.stats().snapshot().misses() == 2);
            // Both slots are loading: a third expert has nowhere to go.
            assertThrows(TimeoutException.class, () -> rig.cache.acquire(0, 2, TimeUnit.MILLISECONDS.toNanos(100)));
            assertEquals(0, rig.cache.stats().snapshot().evictions());
            Future<ExpertLease> third = acquireAsync(rig, 0, 2);
            Thread.sleep(50);
            assertFalse(third.isDone());
            rig.releaseHeld();
            ExpertLease a = first.get(10, TimeUnit.SECONDS);
            ExpertLease b = second.get(10, TimeUnit.SECONDS);
            rig.assertLeaseBytes(a);
            rig.assertLeaseBytes(b);
            assertFalse(third.isDone(), "both slots are leased");
            a.close();
            ExpertLease c = third.get(10, TimeUnit.SECONDS);
            rig.assertLeaseBytes(c);
            rig.assertLeaseBytes(b);
            b.close();
            c.close();
            assertTrue(rig.cache.stats().snapshot().slotWaitNanos() > 0);
            assertEquals(2, rig.cache.stats().snapshot().peakSlotsInUse());
            rig.cache.checkQuiescent();
        }
    }

    // -------------------------------------------------------------- eviction order

    @ParameterizedTest
    @MethodSource("combos")
    void evictionTakesTheLeastRecentlyUsedUnpinnedSlot(StoreKind store, TransferKind transfer) throws Exception {
        try (CacheRig rig = rig(store, transfer, 3)) {
            ExpertCache cache = rig.cache;
            for (int expert = 0; expert < 3; expert++) cache.acquire(0, expert).close(); // order: 0 1 2
            cache.acquire(0, 0).close(); // order: 1 2 0
            assertTrue(cache.isResident(0, 0) && cache.isResident(0, 1) && cache.isResident(0, 2));

            cache.acquire(0, 3).close(); // evicts 1; order: 2 0 3
            assertFalse(cache.isResident(0, 1));
            assertTrue(cache.isResident(0, 0) && cache.isResident(0, 2) && cache.isResident(0, 3));

            cache.acquire(0, 4).close(); // evicts 2; order: 0 3 4
            assertFalse(cache.isResident(0, 2));
            assertTrue(cache.isResident(0, 0) && cache.isResident(0, 3) && cache.isResident(0, 4));

            // A pinned slot is skipped even when it is the least recently used.
            ExpertLease pinned = cache.acquire(0, 0);
            cache.acquire(0, 5).close(); // 0 is pinned: evicts 3; order: 4 5 (0 pinned)
            assertFalse(cache.isResident(0, 3));
            assertTrue(cache.isResident(0, 0) && cache.isResident(0, 4) && cache.isResident(0, 5));
            pinned.close(); // 0 is now the most recently used: order 4 5 0
            cache.acquire(0, 6).close(); // evicts 4
            assertFalse(cache.isResident(0, 4));
            cache.acquire(0, 7).close(); // evicts 5
            assertFalse(cache.isResident(0, 5));
            assertTrue(cache.isResident(0, 0), "released last, so evicted last");

            ExpertCacheStats.Snapshot stats = cache.stats().snapshot();
            assertEquals(5, stats.evictions());
            assertEquals(8, stats.misses());
            cache.checkQuiescent();
        }
    }

    // -------------------------------------------------------------- failures

    @ParameterizedTest
    @MethodSource("combos")
    void aFailedTransferResetsTheSlotAndLaterRequestsRetryCleanly(StoreKind store, TransferKind transfer)
            throws Exception {
        try (CacheRig rig = rig(store, transfer, 2)) {
            rig.failNextTransfers(1);
            ExpertTransferException failure =
                    assertThrows(ExpertTransferException.class, () -> rig.cache.acquire(2, 4));
            assertNotEquals(null, failure.getCause());
            assertFalse(rig.cache.isResident(2, 4));
            assertEquals(0, rig.cache.evictableSlots());
            assertEquals(0, rig.cache.openLeaseCount());
            assertEquals(1, rig.cache.stats().snapshot().failedTransfers());
            rig.cache.checkQuiescent();
            // The torn slot was not poisoned: the retry rewrites it.
            try (ExpertLease lease = rig.cache.acquire(2, 4)) {
                rig.assertLeaseBytes(lease);
            }
            try (ExpertLease lease = rig.cache.acquire(2, 5)) {
                rig.assertLeaseBytes(lease);
            }
            rig.cache.checkQuiescent();
        }
    }

    @ParameterizedTest
    @MethodSource("holdableCombos")
    void everyWaiterOfAFailedTransferReceivesTheFailure(StoreKind store, TransferKind transfer) throws Exception {
        try (CacheRig rig = rig(store, transfer, 2)) {
            rig.hold();
            rig.failNextTransfers(1);
            List<Future<ExpertLease>> futures = new ArrayList<>();
            for (int i = 0; i < 5; i++) futures.add(acquireAsync(rig, 3, 0));
            await(() -> rig.cache.stats().snapshot().coalescedRequests() == 4);
            rig.releaseHeld();
            for (Future<ExpertLease> future : futures) {
                ExecutionException failure =
                        assertThrows(ExecutionException.class, () -> future.get(10, TimeUnit.SECONDS));
                assertInstanceOf(ExpertTransferException.class, failure.getCause());
            }
            assertEquals(1, rig.cache.stats().snapshot().failedTransfers());
            rig.cache.checkQuiescent();
            try (ExpertLease lease = rig.cache.acquire(3, 0)) {
                rig.assertLeaseBytes(lease);
            }
        }
    }

    @Test
    void anExceptionFromStartFailsTheLoadWithoutLeakingStagingOrSlots() throws Exception {
        try (CacheRig rig =
                new CacheRig(this.fixture, StoreKind.FILE, TransferKind.SYNTHETIC, 2, 2, System::nanoTime)) {
            rig.synthetic.failStart(new IllegalStateException("start refused"));
            for (int attempt = 0; attempt < 5; attempt++) {
                ExpertTransferException failure =
                        assertThrows(ExpertTransferException.class, () -> rig.cache.acquire(0, 0));
                assertEquals("start refused", failure.getCause().getMessage());
            }
            assertEquals(2, ((FileExpertStore) rig.store).freeSlots(), "the refused records were closed");
            assertEquals(0, rig.cache.evictableSlots());
            rig.cache.checkQuiescent();
            rig.synthetic.failStart(null);
            try (ExpertLease lease = rig.cache.acquire(0, 0)) {
                rig.assertLeaseBytes(lease);
            }
        }
    }

    @Test
    void aStoreReadFailureFailsTheLoadAndNothingElse() throws Exception {
        HostBackedGpu gpu = new HostBackedGpu();
        RecordSource flaky = new RecordSource() {
            private final FileRecordSource real = real();
            private final AtomicInteger failures = new AtomicInteger(2);

            private FileRecordSource real() {
                try {
                    return new FileRecordSource(ExpertCacheTest.this.fixture.file, ExpertCacheTest.this.fixture.banks);
                } catch (IOException failure) {
                    throw new IllegalStateException(failure);
                }
            }

            @Override
            public void read(ExpertBank bank, int expert, java.lang.foreign.MemorySegment destination)
                    throws IOException, InterruptedException {
                if (bank.name().equals("layer0") && expert == 5 && this.failures.getAndDecrement() > 0)
                    throw new IOException("disk fault");
                this.real.read(bank, expert, destination);
            }

            @Override
            public long bytesRead() {
                return this.real.bytesRead();
            }

            @Override
            public void close() {
                this.real.close();
            }
        };
        FileExpertStore store = new FileExpertStore(gpu, flaky, this.fixture.banks, 2);
        SyntheticExpertTransfer transfer = new SyntheticExpertTransfer();
        try (ExpertCache cache = new ExpertCache(store, transfer, gpu, 3, this.fixture.slotBytes())) {
            for (int i = 0; i < 2; i++) {
                ExpertTransferException failure =
                        assertThrows(ExpertTransferException.class, () -> cache.acquire(0, 5));
                assertInstanceOf(IOException.class, failure.getCause());
            }
            try (ExpertLease other = cache.acquire(0, 4);
                    ExpertLease retry = cache.acquire(0, 5)) {
                assertArrayEquals(this.fixture.record(0, 4), gpu.readDevice(other.deviceAddress(), other.byteSize()));
                assertArrayEquals(this.fixture.record(0, 5), gpu.readDevice(retry.deviceAddress(), retry.byteSize()));
            }
            assertEquals(2, store.freeSlots());
            cache.checkQuiescent();
        }
        gpu.assertAllReleased();
    }

    // -------------------------------------------------------------- cancellation

    @ParameterizedTest
    @MethodSource("holdableCombos")
    void anInterruptedWaiterGivesUpOnlyItsOwnPin(StoreKind store, TransferKind transfer) throws Exception {
        try (CacheRig rig = rig(store, transfer, 2)) {
            rig.hold();
            AtomicBoolean interruptedSeen = new AtomicBoolean();
            CountDownLatch joined = new CountDownLatch(1);
            Future<ExpertLease> loader = acquireAsync(rig, 1, 1);
            await(() -> rig.cache.stats().snapshot().misses() == 1);
            Thread waiter = new Thread(() -> {
                joined.countDown();
                try {
                    rig.cache.acquire(1, 1);
                } catch (InterruptedException interrupted) {
                    interruptedSeen.set(true);
                }
            });
            waiter.start();
            assertTrue(joined.await(5, TimeUnit.SECONDS));
            await(() -> rig.cache.stats().snapshot().coalescedRequests() == 1);
            waiter.interrupt();
            waiter.join(10_000);
            assertTrue(interruptedSeen.get());
            rig.releaseHeld();
            ExpertLease lease = loader.get(10, TimeUnit.SECONDS);
            rig.assertLeaseBytes(lease);
            assertEquals(1, rig.cache.openLeaseCount());
            lease.close();
            assertEquals(1, rig.cache.evictableSlots(), "exactly the loader's pin was released, once");
            rig.cache.checkQuiescent();
        }
    }

    @ParameterizedTest
    @MethodSource("holdableCombos")
    void whenTheLastWaiterLeavesTheTransferStillCompletesAndTheExpertIsResident(StoreKind store, TransferKind transfer)
            throws Exception {
        try (CacheRig rig = rig(store, transfer, 2)) {
            rig.hold();
            AtomicBoolean interruptedSeen = new AtomicBoolean();
            Thread only = new Thread(() -> {
                try {
                    rig.cache.acquire(4, 7);
                } catch (InterruptedException interrupted) {
                    interruptedSeen.set(true);
                }
            });
            only.start();
            await(() -> rig.cache.stats().snapshot().misses() == 1);
            only.interrupt();
            only.join(10_000);
            assertTrue(interruptedSeen.get());
            assertEquals(0, rig.cache.openLeaseCount());
            assertFalse(rig.cache.isResident(4, 7), "still loading");
            rig.releaseHeld();
            await(() -> rig.cache.isResident(4, 7));
            await(() -> rig.cache.evictableSlots() == 1);
            rig.cache.checkQuiescent();
            // The data is valid: the next request is a hit and moves no bytes.
            int transfers = rig.transfers();
            try (ExpertLease lease = rig.cache.acquire(4, 7)) {
                rig.assertLeaseBytes(lease);
            }
            assertEquals(transfers, rig.transfers());
            assertEquals(1, rig.cache.stats().snapshot().hits());
        }
    }

    @ParameterizedTest
    @MethodSource("holdableCombos")
    void aTimedOutWaiterLeavesTheTransferRunningAndTheCacheUsable(StoreKind store, TransferKind transfer)
            throws Exception {
        try (CacheRig rig = rig(store, transfer, 2)) {
            rig.hold();
            assertThrows(TimeoutException.class, () -> rig.cache.acquire(2, 2, TimeUnit.MILLISECONDS.toNanos(60)));
            assertEquals(0, rig.cache.openLeaseCount());
            rig.releaseHeld();
            await(() -> rig.cache.isResident(2, 2));
            assertEquals(1, rig.transfers());
            try (ExpertLease lease = rig.cache.acquire(2, 2)) {
                rig.assertLeaseBytes(lease);
            }
            assertEquals(1, rig.transfers(), "the late transfer served the next request");
            rig.cache.checkQuiescent();
        }
    }

    @Test
    void aLoaderInterruptedWhileWaitingForStagingHandsTheLoadToAWaiter() throws Exception {
        // One staging slot, held by the first transfer: the second expert's loader blocks in the store.
        try (CacheRig rig =
                new CacheRig(this.fixture, StoreKind.FILE, TransferKind.SYNTHETIC, 3, 1, System::nanoTime)) {
            rig.hold();
            Future<ExpertLease> first = acquireAsync(rig, 0, 0);
            await(() -> rig.synthetic.started() == 1);
            AtomicBoolean interruptedSeen = new AtomicBoolean();
            Thread loader = new Thread(() -> {
                try {
                    rig.cache.acquire(0, 1);
                } catch (InterruptedException interrupted) {
                    interruptedSeen.set(true);
                }
            });
            loader.start();
            await(() -> rig.cache.stats().snapshot().misses() == 2);
            Future<ExpertLease> joiner = acquireAsync(rig, 0, 1);
            await(() -> rig.cache.stats().snapshot().coalescedRequests() == 1);
            loader.interrupt();
            loader.join(10_000);
            assertTrue(interruptedSeen.get());
            assertEquals(1, rig.cache.stats().snapshot().abandonedLoads());
            rig.releaseHeld();
            // The joiner takes over the abandoned load and completes it.
            try (ExpertLease lease = joiner.get(10, TimeUnit.SECONDS);
                    ExpertLease other = first.get(10, TimeUnit.SECONDS)) {
                rig.assertLeaseBytes(lease);
                rig.assertLeaseBytes(other);
            }
            rig.cache.checkQuiescent();
        }
    }

    // -------------------------------------------------------------- indexing

    @ParameterizedTest
    @MethodSource("combos")
    void keysNeverAliasAcrossBanksAndBoundaries(StoreKind store, TransferKind transfer) throws Exception {
        int total = this.fixture.totalExperts();
        try (CacheRig rig = rig(store, transfer, total)) {
            List<ExpertLease> leases = new ArrayList<>();
            Set<Long> addresses = new HashSet<>();
            for (int bank = 0; bank < this.fixture.banks.length; bank++) {
                for (int expert = 0; expert < this.fixture.expertCount(bank); expert++) {
                    ExpertLease lease = rig.cache.acquire(bank, expert);
                    leases.add(lease);
                    assertTrue(addresses.add(lease.deviceAddress()), "two experts share a slot");
                }
            }
            assertEquals(total, rig.cache.stats().snapshot().misses());
            for (ExpertLease lease : leases) rig.assertLeaseBytes(lease);
            // Last expert of a bank, first of the next: distinct slots, distinct bytes.
            for (int bank = 0; bank + 1 < this.fixture.banks.length; bank++) {
                ExpertLease last = rig.cache.acquire(bank, this.fixture.expertCount(bank) - 1);
                ExpertLease first = rig.cache.acquire(bank + 1, 0);
                assertNotEquals(last.deviceAddress(), first.deviceAddress());
                assertFalse(java.util.Arrays.equals(
                        rig.gpu.readDevice(last.deviceAddress(), 5000),
                        rig.gpu.readDevice(first.deviceAddress(), 5000)));
                last.close();
                first.close();
            }
            assertEquals(total, rig.cache.stats().snapshot().misses(), "these were all hits");
            for (ExpertLease lease : leases) lease.close();

            // The same expert id in every bank is a different expert.
            for (int bank = 0; bank < this.fixture.banks.length; bank++)
                if (this.fixture.expertCount(bank) > 1) {
                    try (ExpertLease a = rig.cache.acquire(bank, 1)) {
                        assertEquals(bank, a.bank());
                        rig.assertLeaseBytes(a);
                    }
                }
        }
    }

    @ParameterizedTest
    @MethodSource("combos")
    void outOfRangeRequestsThrowAndChangeNothing(StoreKind store, TransferKind transfer) throws Exception {
        try (CacheRig rig = rig(store, transfer, 2)) {
            ExpertCache cache = rig.cache;
            assertThrows(IndexOutOfBoundsException.class, () -> cache.acquire(0, 8)); // one past bank 0
            assertThrows(IndexOutOfBoundsException.class, () -> cache.acquire(1, 3)); // one past bank 1
            assertThrows(IndexOutOfBoundsException.class, () -> cache.acquire(0, -1));
            assertThrows(IndexOutOfBoundsException.class, () -> cache.acquire(-1, 0));
            assertThrows(IndexOutOfBoundsException.class, () -> cache.acquire(5, 0));
            assertThrows(IndexOutOfBoundsException.class, () -> cache.acquire(0, Integer.MAX_VALUE));
            assertThrows(IndexOutOfBoundsException.class, () -> cache.tryAcquire(1, 40));
            assertThrows(IndexOutOfBoundsException.class, () -> cache.isResident(0, 8));
            assertEquals(0, cache.stats().snapshot().requests());
            assertEquals(0, rig.transfers());
            cache.checkQuiescent();
        }
    }

    @Test
    void constructionValidatesSlotGeometry() throws Exception {
        HostBackedGpu gpu = new HostBackedGpu();
        try (FileExpertStore store = new FileExpertStore(gpu, this.fixture.file, this.fixture.banks, 1);
                SyntheticExpertTransfer transfer = new SyntheticExpertTransfer()) {
            long needed = this.fixture.slotBytes();
            assertEquals(0, needed % ExpertCache.SLOT_ALIGNMENT);
            assertTrue(needed >= 9001);
            assertThrows(
                    IllegalArgumentException.class,
                    () -> new ExpertCache(store, transfer, gpu, 2, needed - ExpertCache.SLOT_ALIGNMENT));
            assertThrows(IllegalArgumentException.class, () -> new ExpertCache(store, transfer, gpu, 2, needed + 1));
            assertThrows(IllegalArgumentException.class, () -> new ExpertCache(store, transfer, gpu, 0, needed));
            assertThrows(
                    ArithmeticException.class,
                    () -> new ExpertCache(store, transfer, gpu, Integer.MAX_VALUE, Long.MAX_VALUE / 2 / 256 * 256));
            assertEquals(0, gpu.deviceAllocationSizes().size(), "a rejected cache allocates nothing");
        }
    }

    // -------------------------------------------------------------- memory bound

    @ParameterizedTest
    @MethodSource("combos")
    void deviceMemoryIsOneSlabOfExactlySlotCountTimesSlotBytes(StoreKind store, TransferKind transfer)
            throws Exception {
        try (CacheRig rig = rig(store, transfer, 5)) {
            assertEquals(List.of(5 * this.fixture.slotBytes()), rig.gpu.deviceAllocationSizes());
            assertEquals(5 * this.fixture.slotBytes(), rig.cache.capacityBytes());
            java.util.Random random = new java.util.Random(77);
            Set<Long> slotAddresses = new HashSet<>();
            for (int i = 0; i < 2000; i++) {
                int bank = random.nextInt(this.fixture.banks.length);
                int expert = random.nextInt(this.fixture.expertCount(bank));
                try (ExpertLease lease = rig.cache.acquire(bank, expert)) {
                    assertTrue(lease.deviceAddress() >= rig.cache.slabAddress());
                    assertTrue(lease.deviceAddress() + this.fixture.slotBytes()
                            <= rig.cache.slabAddress() + rig.cache.capacityBytes());
                    assertEquals(0, (lease.deviceAddress() - rig.cache.slabAddress()) % this.fixture.slotBytes());
                    slotAddresses.add(lease.deviceAddress());
                }
            }
            assertEquals(5, slotAddresses.size(), "exactly the slab's five slots are ever used");
            assertEquals(List.of(5 * this.fixture.slotBytes()), rig.gpu.deviceAllocationSizes(), "no growth");
            assertEquals(1, rig.gpu.liveDeviceAllocations());
            assertEquals(5 * this.fixture.slotBytes(), rig.gpu.liveDeviceBytes());
        }
    }

    // -------------------------------------------------------------- fences

    private static final class RecordingFence implements DeviceFence {
        final AtomicInteger awaited = new AtomicInteger();
        final AtomicInteger released = new AtomicInteger();

        @Override
        public void awaitOn(io.euhedral_execution.inference.core.gpu.GpuStream copyStream) {
            this.awaited.incrementAndGet();
        }

        @Override
        public void release() {
            this.released.incrementAndGet();
        }
    }

    @ParameterizedTest
    @MethodSource("gpuCombos")
    void aFenceIsAwaitedByTheNextRefillOfItsSlotAndThenReleased(StoreKind store, TransferKind transfer)
            throws Exception {
        try (CacheRig rig = rig(store, transfer, 1)) {
            RecordingFence fence = new RecordingFence();
            ExpertLease lease = rig.cache.acquire(0, 0);
            lease.close(fence);
            assertEquals(0, fence.awaited.get(), "nothing refilled the slot yet");
            assertEquals(0, fence.released.get());
            try (ExpertLease other = rig.cache.acquire(0, 1)) {
                rig.assertLeaseBytes(other);
            }
            assertEquals(1, fence.awaited.get());
            await(() -> fence.released.get() == 1);
            // A later refill has nothing left to wait for.
            rig.cache.acquire(0, 2).close();
            assertEquals(1, fence.awaited.get());
            assertEquals(1, fence.released.get());
        }
    }

    static Stream<Arguments> gpuCombos() {
        return combos().filter(arguments -> arguments.get()[1] != TransferKind.SYNTHETIC);
    }

    @ParameterizedTest
    @MethodSource("gpuCombos")
    void leasesClosedWithSeveralFencesRefillBehindAllOfThem(StoreKind store, TransferKind transfer) throws Exception {
        try (CacheRig rig = rig(store, transfer, 1)) {
            ExpertLease a = rig.cache.acquire(1, 0);
            ExpertLease b = rig.cache.acquire(1, 0);
            ExpertLease c = rig.cache.acquire(1, 0);
            RecordingFence fenceA = new RecordingFence();
            RecordingFence fenceB = new RecordingFence();
            a.close(fenceA);
            b.close(fenceB);
            c.close(); // a lease without a fence adds none
            rig.cache.acquire(1, 1).close();
            assertEquals(1, fenceA.awaited.get());
            assertEquals(1, fenceB.awaited.get());
            await(() -> fenceA.released.get() == 1 && fenceB.released.get() == 1);
        }
    }

    @ParameterizedTest
    @MethodSource("gpuCombos")
    void aFailedRefillKeepsTheFenceForTheNextAttempt(StoreKind store, TransferKind transfer) throws Exception {
        try (CacheRig rig = rig(store, transfer, 1)) {
            RecordingFence fence = new RecordingFence();
            rig.cache.acquire(0, 0).close(fence);
            rig.failNextTransfers(1);
            assertThrows(ExpertTransferException.class, () -> rig.cache.acquire(0, 1));
            assertEquals(0, fence.released.get(), "the failed refill did not consume the fence");
            try (ExpertLease lease = rig.cache.acquire(0, 1)) {
                rig.assertLeaseBytes(lease);
            }
            await(() -> fence.released.get() == 1);
            assertTrue(fence.awaited.get() >= 1);
            assertEquals(1, fence.released.get(), "released exactly once overall");
        }
    }

    @Test
    void aRefillDoesNotOvertakeTheKernelsAFenceStandsFor() throws Exception {
        try (CacheRig rig = rig(StoreKind.ARENA, TransferKind.GPU_ASYNC, 1)) {
            FakeStream compute = new FakeStream();
            try {
                ExpertLease lease = rig.cache.acquire(0, 0);
                byte[] before = rig.gpu.readDevice(lease.deviceAddress(), lease.byteSize());
                long address = lease.deviceAddress();
                CountDownLatch kernels = new CountDownLatch(1);
                compute.stall(kernels);
                long marker = compute.openMarker();
                compute.mark(marker); // recorded once the stalled "kernels" finish
                lease.close(StreamFence.owning(compute, marker));

                Future<ExpertLease> refill = acquireAsync(rig, 0, 1);
                assertThrows(TimeoutException.class, () -> refill.get(200, TimeUnit.MILLISECONDS));
                assertArrayEquals(
                        before,
                        rig.gpu.readDevice(address, before.length),
                        "the slot is untouched while the kernels that read it have not finished");
                assertFalse(compute.markerClosed(marker));
                kernels.countDown();
                try (ExpertLease next = refill.get(10, TimeUnit.SECONDS)) {
                    assertEquals(address, next.deviceAddress());
                    rig.assertLeaseBytes(next);
                }
                await(() -> compute.markerClosed(marker));
            } finally {
                compute.close();
            }
        }
    }

    @Test
    void closeWithFencesReleasesPendingOnes() throws Exception {
        RecordingFence fence = new RecordingFence();
        try (CacheRig rig = rig(StoreKind.ARENA, TransferKind.SYNTHETIC, 2)) {
            rig.cache.acquire(0, 0).close(fence);
        }
        assertEquals(1, fence.released.get());
        assertEquals(0, fence.awaited.get());
        // A fence given to an already-closed lease is released at once and never used.
        RecordingFence late = new RecordingFence();
        try (CacheRig rig = rig(StoreKind.ARENA, TransferKind.SYNTHETIC, 2)) {
            ExpertLease lease = rig.cache.acquire(0, 0);
            lease.close();
            lease.close(late);
            assertEquals(1, late.released.get());
        }
    }

    // -------------------------------------------------------------- statistics

    @Test
    void statisticsCountEveryKindOfRequestAndTime() throws Exception {
        AtomicLong clock = new AtomicLong();
        try (CacheRig rig = new CacheRig(this.fixture, StoreKind.ARENA, TransferKind.SYNTHETIC, 2, 2, clock::get)) {
            rig.hold();
            Future<ExpertLease> first = acquireAsync(rig, 0, 0);
            Future<ExpertLease> joiner = acquireAsync(rig, 0, 0);
            await(() -> rig.cache.stats().snapshot().coalescedRequests() == 1);
            await(() -> rig.synthetic.started() == 1);
            Thread.sleep(50);
            clock.addAndGet(5_000_000);
            rig.releaseHeld();
            ExpertLease a = first.get(10, TimeUnit.SECONDS);
            ExpertLease b = joiner.get(10, TimeUnit.SECONDS);
            ExpertCacheStats.Snapshot loaded = rig.cache.stats().snapshot();
            assertEquals(5_000_000, loaded.transferNanos());
            assertEquals(this.fixture.banks[0].recordBytes(0), loaded.transferBytes());
            assertTrue(
                    loaded.loadWaitNanos() >= 5_000_000 && loaded.loadWaitNanos() <= 10_000_000,
                    "acquirers waited for the transfer: " + loaded.loadWaitNanos());
            assertEquals(2, loaded.slotCount());
            assertEquals(this.fixture.slotBytes(), loaded.slotBytes());
            assertEquals(2 * this.fixture.slotBytes(), loaded.capacityBytes());
            a.close();
            b.close();
            rig.cache.acquire(0, 0).close(); // hit
            rig.cache.acquire(0, 1).close();
            rig.cache.acquire(0, 2).close(); // evicts
            ExpertCacheStats.Snapshot stats = rig.cache.stats().snapshot();
            assertEquals(1, stats.hits());
            assertEquals(3, stats.misses());
            assertEquals(1, stats.coalescedRequests());
            assertEquals(1, stats.evictions());
            assertEquals(5, stats.requests());
            assertEquals(0, stats.failedTransfers());
            assertEquals(1, stats.peakSlotsInUse());
            assertTrue(stats.transferBytesPerSecond() > 0);
        }
    }

    // -------------------------------------------------------------- closing

    @ParameterizedTest
    @MethodSource("combos")
    void closeReleasesTheSlabTheStoreAndTheTransferExactlyOnce(StoreKind store, TransferKind transfer)
            throws Exception {
        CacheRig rig = rig(store, transfer, 3);
        rig.cache.acquire(0, 0).close();
        rig.cache.acquire(1, 1).close();
        rig.cache.close();
        rig.cache.close();
        assertTrue(rig.cache.isClosed());
        rig.gpu.assertAllReleased();
        assertEquals(1, rig.gpu.deviceFrees());
        assertEquals(1, rig.gpu.hostFrees());
        if (rig.synthetic != null) assertEquals(1, rig.synthetic.closes());
        else assertEquals(rig.gpu.streamsOpened(), rig.gpu.streamsClosed());
        assertThrows(IllegalStateException.class, () -> rig.cache.acquire(0, 0));
        assertThrows(IllegalStateException.class, () -> rig.cache.tryAcquire(0, 0));
        assertThrows(IllegalStateException.class, () -> rig.store.open(0, 0));
    }

    @ParameterizedTest
    @MethodSource("combos")
    void closeInvalidatesLeasesThatAreStillOpen(StoreKind store, TransferKind transfer) throws Exception {
        CacheRig rig = rig(store, transfer, 3);
        ExpertLease open = rig.cache.acquire(0, 0);
        ExpertLease alsoOpen = rig.cache.acquire(0, 1);
        alsoOpen.close();
        assertTrue(open.isValid());
        rig.cache.close();
        assertFalse(open.isValid());
        assertEquals(1, rig.cache.stats().snapshot().forcedLeases());
        open.close(); // a no-op: the slab is gone
        open.close(new RecordingFence());
        rig.gpu.assertAllReleased();
    }

    @ParameterizedTest
    @MethodSource("combos")
    void closeAfterFailuresStillReleasesEverything(StoreKind store, TransferKind transfer) throws Exception {
        CacheRig rig = rig(store, transfer, 2);
        rig.failNextTransfers(2);
        for (int i = 0; i < 2; i++) {
            int expert = i;
            assertThrows(ExpertTransferException.class, () -> rig.cache.acquire(0, expert));
        }
        rig.cache.acquire(0, 2).close();
        rig.cache.close();
        rig.gpu.assertAllReleased();
    }

    @ParameterizedTest
    @MethodSource("holdableCombos")
    void closeWaitsForTransfersInFlightThenReleasesEverything(StoreKind store, TransferKind transfer) throws Exception {
        CacheRig rig = rig(store, transfer, 2);
        rig.hold();
        Future<ExpertLease> loading = acquireAsync(rig, 0, 0);
        await(() -> rig.cache.stats().snapshot().misses() == 1);
        Future<?> closing = this.pool.submit(rig.cache::close);
        assertThrows(TimeoutException.class, () -> closing.get(200, TimeUnit.MILLISECONDS));
        assertEquals(1, rig.gpu.liveDeviceAllocations(), "the slab stays while a transfer may still write it");
        rig.releaseHeld();
        closing.get(10, TimeUnit.SECONDS);
        // The acquirer that was waiting is told the cache closed.
        ExecutionException failure = assertThrows(ExecutionException.class, () -> loading.get(10, TimeUnit.SECONDS));
        assertInstanceOf(IllegalStateException.class, failure.getCause());
        rig.gpu.assertAllReleased();
    }

    @Test
    void aCloseTimeoutKeepsTheResourcesAndALaterCloseFinishes() throws Exception {
        HostBackedGpu gpu = new HostBackedGpu();
        HostExpertStore store = new ArenaExpertStore(gpu, this.fixture.file, this.fixture.banks, 2);
        SyntheticExpertTransfer transfer = new SyntheticExpertTransfer();
        java.util.concurrent.Semaphore gate = new java.util.concurrent.Semaphore(0);
        transfer.gate(gate);
        ExpertCache cache = new ExpertCache(
                store,
                transfer,
                gpu,
                2,
                this.fixture.slotBytes(),
                System::nanoTime,
                TimeUnit.MILLISECONDS.toNanos(100));
        Future<ExpertLease> loading = this.pool.submit(() -> cache.acquire(0, 0));
        await(() -> transfer.started() == 1);
        IllegalStateException failure = assertThrows(IllegalStateException.class, cache::close);
        assertTrue(failure.getMessage().contains("in flight"), failure.getMessage());
        assertEquals(1, gpu.liveDeviceAllocations());
        assertEquals(1, gpu.liveHostAllocations());
        assertEquals(0, transfer.closes());
        gate.release(10);
        ExecutionException loaded = assertThrows(ExecutionException.class, () -> loading.get(10, TimeUnit.SECONDS));
        assertInstanceOf(IllegalStateException.class, loaded.getCause());
        cache.close();
        cache.close();
        gpu.assertAllReleased();
        assertEquals(1, transfer.closes());
    }

    @Test
    void aFailingTransferCloseKeepsTheSlabAndTheStoreForARetry() throws Exception {
        HostBackedGpu gpu = new HostBackedGpu();
        HostExpertStore store = new ArenaExpertStore(gpu, this.fixture.file, this.fixture.banks, 2);
        SyntheticExpertTransfer inner = new SyntheticExpertTransfer();
        AtomicInteger attempts = new AtomicInteger();
        ExpertTransfer flaky = new ExpertTransfer() {
            @Override
            public void start(HostRecord record, long deviceAddress, DeviceFence waitFor, Completion done)
                    throws InterruptedException {
                inner.start(record, deviceAddress, waitFor, done);
            }

            @Override
            public void close() {
                if (attempts.incrementAndGet() == 1) throw new IllegalStateException("copy stream busy");
                inner.close();
            }
        };
        ExpertCache cache = new ExpertCache(store, flaky, gpu, 2, this.fixture.slotBytes());
        cache.acquire(0, 0).close();
        assertThrows(IllegalStateException.class, cache::close);
        assertEquals(1, gpu.liveDeviceAllocations(), "in-flight copies are unproven: the slab is kept");
        assertEquals(1, gpu.liveHostAllocations());
        cache.close();
        cache.close();
        gpu.assertAllReleased();
        assertEquals(2, attempts.get());
    }

    @ParameterizedTest
    @MethodSource("combos")
    void closeAbortsAcquirersWaitingForASlot(StoreKind store, TransferKind transfer) throws Exception {
        CacheRig rig = rig(store, transfer, 1);
        ExpertLease held = rig.cache.acquire(0, 0);
        Future<ExpertLease> waiting = acquireAsync(rig, 0, 1);
        Thread.sleep(100);
        assertFalse(waiting.isDone());
        rig.cache.close();
        ExecutionException failure = assertThrows(ExecutionException.class, () -> waiting.get(10, TimeUnit.SECONDS));
        assertInstanceOf(IllegalStateException.class, failure.getCause());
        assertFalse(held.isValid());
        rig.gpu.assertAllReleased();
    }
}
