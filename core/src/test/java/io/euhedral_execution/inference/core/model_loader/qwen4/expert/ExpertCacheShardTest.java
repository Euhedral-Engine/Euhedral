package io.euhedral_execution.inference.core.model_loader.qwen4.expert;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.gpu.GpuStream;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/// The shard of the cache is single-owner: these tests are its owner, so they call it with no lock
/// and nothing else running. A miss is a load whose lease carries the marker of its copy, so it
/// exists before the bytes arrive, and its slot stays unevictable until the copy retired.
/// Everything else is replaced least recently used first.
class ExpertCacheShardTest {

    @TempDir
    Path directory;

    private final HostBackedGpu gpu = new HostBackedGpu();
    private final List<FakeStream> streams = new ArrayList<>();
    private ExpertFixture fixture;
    private ExpertCache cache;

    @BeforeEach
    void setUp() throws IOException {
        this.fixture = ExpertFixture.standard(this.directory, 91);
        this.gpu.streamFactory(() -> {
            FakeStream stream = new FakeStream();
            synchronized (this.streams) {
                this.streams.add(stream);
            }
            return stream;
        });
    }

    @AfterEach
    void tearDown() {
        if (this.cache != null) this.cache.close();
        this.gpu.assertAllReleased();
    }

    private ExpertCache cache(int slots, int shards) throws IOException {
        return cache(slots, shards, ReplacementPolicy.GLOBAL_LRU);
    }

    private ExpertCache cache(int slots, int shards, ReplacementPolicy policy) throws IOException {
        var store = new FileExpertStore(this.gpu, this.fixture.file, this.fixture.banks, 4);
        var transfer = new GpuExpertTransfer(this.gpu, 4);
        this.cache = new ExpertCache(store, transfer, this.gpu, slots, this.fixture.slotBytes(), shards, policy);
        return this.cache;
    }

    /// Copy lane `lane`'s stream: the transfer's marker stream is the first one opened.
    private FakeStream lane(int lane) {
        synchronized (this.streams) {
            return this.streams.get(1 + lane);
        }
    }

    private static final class Retired implements GpuStream.RetirementListener {
        final CompletableFuture<Long> ticket = new CompletableFuture<>();

        @Override
        public void retired(long ticket, boolean driverThread) {
            this.ticket.complete(ticket);
        }
    }

    /// A claim that must be a miss, with room.
    private ExpertCacheShard.Load miss(ExpertCacheShard shard, int bank, int expert) {
        var ticket = new ExpertCacheShard.Ticket();
        shard.claim(bank, expert, true, ticket);
        assertNull(ticket.lease(), "a cold expert is a miss");
        assertNotNull(ticket.load(), "and there was room");
        return ticket.load();
    }

    private ExpertCacheShard shardOf(int bank, int expert) {
        return this.cache.shard(this.cache.shardOf(bank, expert));
    }

    @Test
    void aMissIsLeasedBeforeItsBytesArriveAndTheLeaseCarriesTheCopysMarker() throws Exception {
        cache(2, 1);
        CountDownLatch release = new CountDownLatch(1);
        lane(0).stall(release);
        ExpertCacheShard shard = this.cache.shard(0);
        ExpertCacheShard.Load load = miss(shard, 2, 5);
        Retired retired = new Retired();
        HostRecord record = this.cache.store().open(2, 5, 0);
        this.cache.transfer().stream(0, record, load.deviceAddress(), load.fence(), load.readyMarker(), retired);
        ExpertLease lease = load.submitted();
        assertNotEquals(0, lease.readyMarker(), "the lease stands for bytes that are still on their way");
        assertFalse(retired.ticket.isDone(), "the copy has not retired: the lease did not wait for it");
        release.countDown();
        load.retired(this.cache.transfer().confirm(0, retired.ticket.get(10, TimeUnit.SECONDS)));
        assertArrayEquals(this.fixture.record(2, 5), this.gpu.readDevice(lease.deviceAddress(), lease.byteSize()));
        lease.close();
        shard.checkQuiescent();
        assertEquals(1, shard.stats().snapshot().misses());
    }

    @Test
    void aSlotWhoseLeaseWasClosedStaysUnevictableUntilItsCopyRetired() throws Exception {
        cache(2, 1);
        CountDownLatch release = new CountDownLatch(1);
        lane(0).stall(release);
        ExpertCacheShard shard = this.cache.shard(0);
        ExpertCacheShard.Load load = miss(shard, 0, 1);
        Retired retired = new Retired();
        HostRecord record = this.cache.store().open(0, 1, 0);
        this.cache.transfer().stream(0, record, load.deviceAddress(), load.fence(), load.readyMarker(), retired);
        // The kernels that read the slot were submitted behind the marker, so the lease closes at once.
        load.submitted().close();
        shard.release(0, 1, null);
        assertEquals(0, shard.evictableSlots(), "the copy is still writing the slot");
        ExpertCacheShard.Load other = miss(shard, 0, 2);
        var ticket = new ExpertCacheShard.Ticket();
        shard.claim(0, 3, true, ticket);
        assertTrue(ticket.waiting(), "every slot is pinned or loading");
        release.countDown();
        load.retired(this.cache.transfer().confirm(0, retired.ticket.get(10, TimeUnit.SECONDS)));
        assertEquals(1, shard.evictableSlots(), "the retired copy's slot may be evicted now");
        other.failed(new IllegalStateException("not loaded in this test"));
    }

    @Test
    void aRequestThatFindsTheCopySubmittedIsLeasedWithTheSameMarker() throws Exception {
        cache(2, 1);
        CountDownLatch release = new CountDownLatch(1);
        lane(0).stall(release);
        ExpertCacheShard shard = this.cache.shard(0);
        ExpertCacheShard.Load load = miss(shard, 1, 1);
        Retired retired = new Retired();
        HostRecord record = this.cache.store().open(1, 1, 0);
        this.cache.transfer().stream(0, record, load.deviceAddress(), load.fence(), load.readyMarker(), retired);
        ExpertLease first = load.submitted();
        var again = new ExpertCacheShard.Ticket();
        shard.claim(1, 1, true, again);
        assertNull(again.load());
        assertEquals(first.readyMarker(), again.lease().readyMarker());
        assertEquals(1, shard.stats().snapshot().coalescedRequests());
        release.countDown();
        load.retired(this.cache.transfer().confirm(0, retired.ticket.get(10, TimeUnit.SECONDS)));
        first.close();
        again.lease().close();
        shard.checkQuiescent();
    }

    @Test
    void aResidentExpertIsAHitWithNothingToWaitFor() throws Exception {
        cache(2, 1);
        ExpertTestSupport.acquire(this.cache, 3, 0).close();
        var hit = new ExpertCacheShard.Ticket();
        this.cache.shard(0).claim(3, 0, false, hit);
        assertNull(hit.load());
        assertEquals(0, hit.lease().readyMarker(), "its bytes are in the slot already");
        assertEquals(1, this.cache.shard(0).stats().snapshot().hits());
        hit.lease().close();
    }

    @Test
    void aMissThatMayNotLoadOrFindsNoRoomWaitsAndChangesNothing() throws Exception {
        cache(1, 1);
        ExpertCacheShard shard = this.cache.shard(0);
        var ticket = new ExpertCacheShard.Ticket();
        shard.claim(0, 0, false, ticket);
        assertTrue(ticket.waiting());
        assertEquals(0, shard.stats().snapshot().misses());
        ExpertLease held = ExpertTestSupport.acquire(this.cache, 0, 1);
        shard.claim(0, 2, true, ticket);
        assertTrue(ticket.waiting(), "the only slot is pinned");
        assertFalse(shard.hasRoom());
        held.close();
        assertTrue(shard.hasRoom());
        shard.claim(0, 2, true, ticket);
        assertNotNull(ticket.load());
        ticket.load().failed(new IllegalStateException("not loaded in this test"));
    }

    @Test
    void aLoadThatNeverSubmittedReturnsItsSlot() throws Exception {
        cache(2, 1);
        ExpertCacheShard shard = this.cache.shard(0);
        miss(shard, 2, 2).failed(new IllegalStateException("the record could not be read"));
        shard.checkQuiescent();
        assertEquals(1, shard.stats().snapshot().failedTransfers());
        miss(shard, 2, 2).failed(new IllegalStateException("not loaded in this test"));
    }

    @Test
    void aDeviceFailureAtRetirementUnmapsTheExpertButKeepsItsLeaseValid() throws Exception {
        cache(2, 1);
        lane(0).failNextRetirements(1);
        ExpertCacheShard shard = this.cache.shard(0);
        ExpertCacheShard.Load load = miss(shard, 0, 6);
        Retired retired = new Retired();
        HostRecord record = this.cache.store().open(0, 6, 0);
        this.cache.transfer().stream(0, record, load.deviceAddress(), load.fence(), load.readyMarker(), retired);
        ExpertLease lease = load.submitted();
        load.retired(this.cache.transfer().confirm(0, retired.ticket.get(10, TimeUnit.SECONDS)));
        assertTrue(lease.isValid(), "the holder closes its lease as usual");
        assertFalse(this.cache.isResident(0, 6), "a failed copy is not served to later requests");
        lease.close();
        shard.release(0, 1, null);
        assertEquals(1, shard.stats().snapshot().failedTransfers());
        miss(shard, 0, 6).failed(new IllegalStateException("not loaded in this test"));
    }

    @Test
    void theLeastRecentlyUsedResidentIsEvictedFirst() throws Exception {
        cache(3, 1);
        for (int expert = 0; expert < 3; expert++)
            ExpertTestSupport.acquire(this.cache, 2, expert).close();
        // 0 becomes the most recently used; 1 is now the least.
        ExpertTestSupport.acquire(this.cache, 2, 0).close();
        ExpertTestSupport.acquire(this.cache, 2, 3).close();
        assertFalse(this.cache.isResident(2, 1), "the least recently used was evicted");
        assertTrue(this.cache.isResident(2, 0));
        assertTrue(this.cache.isResident(2, 2));
        assertTrue(this.cache.isResident(2, 3));
        assertEquals(1, this.cache.shard(0).stats().snapshot().evictions());
        this.cache.shard(0).checkQuiescent();
    }

    @Test
    void aLeasedExpertIsNeverEvicted() throws Exception {
        cache(2, 1);
        ExpertLease pinned = ExpertTestSupport.acquire(this.cache, 2, 0);
        ExpertTestSupport.acquire(this.cache, 2, 1).close();
        ExpertTestSupport.acquire(this.cache, 2, 2).close();
        assertTrue(this.cache.isResident(2, 0), "the leased slot was left alone");
        assertFalse(this.cache.isResident(2, 1));
        assertArrayEquals(this.fixture.record(2, 0), this.gpu.readDevice(pinned.deviceAddress(), pinned.byteSize()));
        pinned.close();
        this.cache.shard(0).checkQuiescent();
    }

    /// A fence that records its release.
    private static final class RecordingFence implements DeviceFence {
        final AtomicInteger released = new AtomicInteger();
        final AtomicInteger awaited = new AtomicInteger();

        @Override
        public void awaitOn(GpuStream copyStream) {
            this.awaited.incrementAndGet();
        }

        @Override
        public void release() {
            this.released.incrementAndGet();
        }
    }

    @Test
    void theRefillOfASlotIsOrderedBehindTheFenceItsLeaseClosedWith() throws Exception {
        cache(1, 1);
        ExpertCacheShard shard = this.cache.shard(0);
        ExpertLease lease = ExpertTestSupport.acquire(this.cache, 2, 0);
        RecordingFence fence = new RecordingFence();
        lease.close(fence);
        shard.release(0, 1, null);
        ExpertCacheShard.Load next = miss(shard, 2, 1);
        assertSame(fence, next.fence(), "the copy waits for the kernels that read the slot");
        assertEquals(0, fence.released.get());
        Retired retired = new Retired();
        HostRecord record = this.cache.store().open(2, 1, 0);
        this.cache.transfer().stream(0, record, next.deviceAddress(), next.fence(), next.readyMarker(), retired);
        assertEquals(1, fence.awaited.get(), "the lane's stream was ordered behind it on the device");
        ExpertLease second = next.submitted();
        next.retired(this.cache.transfer().confirm(0, retired.ticket.get(10, TimeUnit.SECONDS)));
        assertEquals(1, fence.released.get(), "released once the refill ended");
        second.close();
    }

    @Test
    void shardsPartitionTheSlotsAndTheExpertsAndWorkIndependently() throws Exception {
        cache(10, 3);
        assertEquals(3, this.cache.shardCount());
        int slots = 0;
        for (int shard = 0; shard < 3; shard++) slots += this.cache.shard(shard).slotCount();
        assertEquals(10, slots);
        Set<Integer> used = new HashSet<>();
        List<ExpertLease> leases = new ArrayList<>();
        for (int bank = 0; bank < 3; bank++) {
            for (int expert = 0; expert < this.fixture.expertCount(bank); expert++) {
                int shard = this.cache.shardOf(bank, expert);
                assertEquals(shard, this.cache.shardOf(bank, expert), "an expert always maps to one shard");
                used.add(shard);
                if (leases.size() < 6) leases.add(ExpertTestSupport.acquire(this.cache, bank, expert));
            }
        }
        assertEquals(Set.of(0, 1, 2), used, "the experts spread over every shard");
        for (ExpertLease lease : leases)
            assertArrayEquals(
                    this.fixture.record(lease.bank(), lease.expert()),
                    this.gpu.readDevice(lease.deviceAddress(), lease.byteSize()));
        for (ExpertLease lease : leases) lease.close();
        this.cache.checkQuiescent();
        assertEquals(6, this.cache.stats().snapshot().misses(), "the shards' counters read as one");
    }

    @Test
    void closingInvalidatesTheLeasesStillOpen() throws Exception {
        cache(2, 1);
        ExpertLease lease = ExpertTestSupport.acquire(this.cache, 0, 0);
        this.cache.close();
        assertFalse(lease.isValid());
        lease.close();
        this.cache = null;
    }

    /// The model visits its layers in a cycle and each layer reuses its own few experts. Ten slots cannot
    /// hold the 13 hot experts of the five banks: a global recency order evicts each one just before it is
    /// needed again, while a quota per bank keeps the experts of the banks that fit.
    @Test
    void aLayerAwarePolicyKeepsWhatACyclicPassWouldEvictFromAGlobalRecencyOrder() throws Exception {
        long[] hits = new long[2];
        for (ReplacementPolicy policy : ReplacementPolicy.values()) {
            if (this.cache != null) this.cache.close();
            cache(10, 1, policy);
            int[] hotPerBank = {3, 3, 3, 1, 3};
            for (int pass = 0; pass < 50; pass++)
                for (int bank = 0; bank < 5; bank++)
                    for (int hot = 0; hot < hotPerBank[bank]; hot++) {
                        try (ExpertLease lease = ExpertTestSupport.acquire(this.cache, bank, hot)) {
                            assertArrayEquals(
                                    this.fixture.record(bank, hot),
                                    this.gpu.readDevice(lease.deviceAddress(), lease.byteSize()));
                        }
                    }
            this.cache.checkQuiescent();
            hits[policy.ordinal()] = this.cache.stats().snapshot().hits();
        }
        long requests = 50 * 13;
        System.out.printf(
                "cyclic trace, 10 slots: partitioned %.3f, global %.3f%n",
                (double) hits[0] / requests, (double) hits[1] / requests);
        assertTrue(hits[1] < requests / 20, "a global order thrashes on the cycle: " + hits[1]);
        assertTrue(hits[0] > requests / 3, "the partitioned policy keeps a share of every pass: " + hits[0]);
    }

    @Test
    void aFailedLoadReturnsItsSlotToItsLayerUnderThePartitionedPolicy() throws Exception {
        cache(4, 1, ReplacementPolicy.BANK_PARTITIONED);
        ExpertCacheShard shard = this.cache.shard(0);
        ExpertCacheShard.Load load = miss(shard, 2, 0);
        load.failed(new IllegalStateException("not loaded in this test"));
        shard.checkQuiescent();
        for (int expert = 0; expert < 6; expert++)
            ExpertTestSupport.acquire(this.cache, 2, expert).close();
        for (int expert = 0; expert < 4; expert++)
            ExpertTestSupport.acquire(this.cache, 4, expert).close();
        shard.checkQuiescent();
        assertEquals(4, shard.evictableSlots());
    }
}
