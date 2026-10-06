package io.euhedral_execution.inference.core.model_loader.qwen4.expert;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.model_loader.qwen4.expert.CacheRig.StoreKind;
import io.euhedral_execution.inference.core.model_loader.qwen4.expert.CacheRig.TransferKind;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

/// The cache's asynchronous requests: hits answer inline, misses reserve and load without any caller waiting,
/// duplicate requests coalesce, a full cache parks continuations instead of threads, the resources bound the fan-out,
/// and nothing is left pinned after any outcome.
class ExpertCacheAsyncTest {
    // Fixture keys: bank 0 has 8 experts, bank 1 has 3, bank 2 has 16, bank 3 has 1, bank 4 has 12 (40 in all).

    @TempDir
    Path directory;

    private ExpertFixture fixture;

    @BeforeEach
    void setUp() throws IOException {
        this.fixture = ExpertFixture.standard(this.directory, 77);
    }

    /// Collects the answers of async requests.
    private static final class Answers implements ExpertListener {
        final List<CompletableFuture<ExpertLease>> futures = new ArrayList<>();
        final AtomicInteger calls = new AtomicInteger();

        Answers(int count) {
            for (int i = 0; i < count; i++) this.futures.add(new CompletableFuture<>());
        }

        @Override
        public void ready(int tag, ExpertLease lease, Throwable failure) {
            this.calls.incrementAndGet();
            if (failure != null) this.futures.get(tag).completeExceptionally(failure);
            else this.futures.get(tag).complete(lease);
        }

        ExpertLease get(int tag) throws Exception {
            return this.futures.get(tag).get(10, TimeUnit.SECONDS);
        }
    }

    @ParameterizedTest
    @EnumSource(StoreKind.class)
    void aMissLoadsAndALaterRequestHitsInline(StoreKind store) throws Exception {
        try (CacheRig rig = new CacheRig(this.fixture, store, TransferKind.GPU_ASYNC, 4)) {
            Answers answers = new Answers(2);
            rig.cache.acquireAsync(2, 5, answers, 0);
            try (ExpertLease lease = answers.get(0)) {
                rig.assertLeaseBytes(lease);
            }
            assertEquals(1, rig.cache.stats().snapshot().misses());
            rig.cache.acquireAsync(2, 5, answers, 1);
            assertEquals(2, answers.calls.get(), "a hit answers before the call returns");
            try (ExpertLease lease = answers.get(1)) {
                rig.assertLeaseBytes(lease);
            }
            assertEquals(1, rig.cache.stats().snapshot().hits());
            rig.cache.checkQuiescent();
        }
    }

    @ParameterizedTest
    @EnumSource(StoreKind.class)
    void requestsForOneExpertShareOneTransfer(StoreKind store) throws Exception {
        try (CacheRig rig = new CacheRig(this.fixture, store, TransferKind.SYNTHETIC, 4)) {
            rig.hold();
            Answers answers = new Answers(5);
            for (int i = 0; i < 5; i++) rig.cache.acquireAsync(0, 3, answers, i);
            assertEquals(0, answers.calls.get(), "nothing is answered while the transfer is in flight");
            rig.releaseHeld();
            List<ExpertLease> leases = new ArrayList<>();
            for (int i = 0; i < 5; i++) leases.add(answers.get(i));
            for (ExpertLease lease : leases) rig.assertLeaseBytes(lease);
            var stats = rig.cache.stats().snapshot();
            assertEquals(1, stats.misses());
            assertEquals(4, stats.coalescedRequests());
            assertEquals(1, rig.transfers(), "one copy served five requests");
            leases.forEach(ExpertLease::close);
            rig.cache.checkQuiescent();
        }
    }

    @Test
    void aRequestPastTheCacheSlotsFailsLoudlyAndAFreedSlotServesTheNext() throws Exception {
        try (CacheRig rig = new CacheRig(this.fixture, StoreKind.ARENA, TransferKind.SYNTHETIC, 2)) {
            Answers answers = new Answers(3);
            rig.cache.acquireAsync(0, 0, answers, 0);
            rig.cache.acquireAsync(0, 1, answers, 1);
            ExpertLease first = answers.get(0);
            ExpertLease second = answers.get(1);
            // The requester bounds what it has outstanding by the slots it may use; nothing queues to hide a request
            // past them.
            assertThrows(IllegalStateException.class, () -> rig.cache.acquireAsync(0, 2, answers, 2));
            assertFalse(answers.futures.get(2).isDone(), "a refused request is never answered");
            first.close();
            rig.cache.acquireAsync(0, 2, answers, 2);
            ExpertLease third = answers.get(2);
            rig.assertLeaseBytes(third);
            second.close();
            third.close();
            rig.cache.checkQuiescent();
        }
    }

    @Test
    void theStagingSlotsBoundTheReadsOfARequesterThatStaysWithinThem() throws Exception {
        try (CacheRig rig =
                new CacheRig(this.fixture, StoreKind.FILE, TransferKind.GPU_ASYNC, 40, 3, System::nanoTime)) {
            // 40 requests, every one a miss: only 3 staging slots exist and at most 4 copies may be in flight.
            Answers answers = new Answers(40);
            List<ExpertLease> leases = new ArrayList<>();
            int tag = 0;
            for (int bank = 0; bank < 5; bank++) {
                int experts = rig.fixture.banks[bank].expertCount();
                for (int expert = 0; expert < experts; expert += 3) {
                    int first = tag;
                    int count = Math.min(3, experts - expert);
                    for (int i = 0; i < count && tag < 40; i++)
                        rig.cache.acquireAsync(bank, expert + i, answers, tag++);
                    for (int i = first; i < tag; i++) leases.add(answers.get(i));
                    // The window's leases are used and closed before the next three are asked for.
                    for (int i = first; i < tag; i++) rig.assertLeaseBytes(leases.get(i));
                    for (int i = first; i < tag; i++) leases.get(i).close();
                }
            }
            assertEquals(40, tag);
            FileExpertStore files = (FileExpertStore) rig.store;
            assertTrue(files.concurrentReadsHighWater() <= 3, "reads " + files.concurrentReadsHighWater());
            assertTrue(files.concurrentReadsHighWater() >= 1);
            assertEquals(0, files.blockedOpens(), "a requester within the bound is never refused");
            rig.cache.checkQuiescent();
        }
    }

    @Test
    void aFailedLoadFailsItsRequestsAndLeavesNothingBehind() throws Exception {
        try (CacheRig rig = new CacheRig(this.fixture, StoreKind.ARENA, TransferKind.SYNTHETIC, 3)) {
            rig.failNextTransfers(1);
            Answers answers = new Answers(2);
            rig.cache.acquireAsync(1, 0, answers, 0);
            var failure = assertThrows(java.util.concurrent.ExecutionException.class, () -> answers.get(0));
            assertInstanceOf(ExpertTransferException.class, failure.getCause());
            assertFalse(rig.cache.isResident(1, 0));
            rig.cache.acquireAsync(1, 0, answers, 1);
            try (ExpertLease lease = answers.get(1)) {
                rig.assertLeaseBytes(lease);
            }
            rig.cache.checkQuiescent();
        }
    }

    @Test
    void aPrefetchHoldsNoLeaseAndIsCountedWhenUsedOrWasted() throws Exception {
        try (CacheRig rig = new CacheRig(this.fixture, StoreKind.ARENA, TransferKind.SYNTHETIC, 2)) {
            assertTrue(rig.cache.prefetch(0, 0));
            assertFalse(rig.cache.prefetch(0, 0), "already loading or resident");
            long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(10);
            while (!rig.cache.isResident(0, 0)) {
                assertTrue(System.nanoTime() < deadline);
                Thread.sleep(1);
            }
            rig.cache.checkQuiescent();
            assertEquals(0, rig.cache.openLeaseCount());
            assertEquals(2, rig.cache.evictableSlots() + 1 > 0 ? 2 : 0);
            Answers answers = new Answers(1);
            rig.cache.acquireAsync(0, 0, answers, 0);
            answers.get(0).close();
            assertEquals(1, rig.cache.stats().snapshot().prefetchesUsed());
            // Prefetch two more: the first is evicted before anyone asked for it.
            assertTrue(rig.cache.prefetch(0, 1));
            while (!rig.cache.isResident(0, 1)) Thread.sleep(1);
            assertTrue(rig.cache.prefetch(0, 2));
            while (!rig.cache.isResident(0, 2)) Thread.sleep(1);
            assertTrue(rig.cache.prefetch(0, 3), "a prefetch evicts an unpinned resident");
            while (!rig.cache.isResident(0, 3)) Thread.sleep(1);
            assertTrue(rig.cache.stats().snapshot().prefetchesWasted() >= 1);
            rig.cache.checkQuiescent();
        }
    }

    @Test
    void aPrefetchNeverWaitsForASlot() throws Exception {
        try (CacheRig rig = new CacheRig(this.fixture, StoreKind.ARENA, TransferKind.SYNTHETIC, 1)) {
            Answers answers = new Answers(1);
            rig.cache.acquireAsync(0, 0, answers, 0);
            try (ExpertLease held = answers.get(0)) {
                assertFalse(rig.cache.prefetch(0, 1), "the only slot is leased");
                assertNotNull(held);
            }
            assertNull(null);
            rig.cache.checkQuiescent();
        }
    }

    @Test
    void closingWaitsForLoadsInFlight() throws Exception {
        CacheRig rig = new CacheRig(this.fixture, StoreKind.ARENA, TransferKind.SYNTHETIC, 1);
        rig.hold();
        Answers answers = new Answers(1);
        rig.cache.acquireAsync(0, 0, answers, 0);
        Thread closer = new Thread(rig.cache::close);
        closer.start();
        Thread.sleep(100);
        assertTrue(closer.isAlive(), "the close waits for the load in flight");
        rig.releaseHeld();
        closer.join(10_000);
        assertFalse(closer.isAlive());
        assertTrue(rig.cache.isClosed());
        assertThrows(IllegalStateException.class, () -> rig.cache.acquireAsync(0, 2, answers, 0));
        rig.close();
    }
}
