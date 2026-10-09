package io.euhedral_execution.inference.core.model.qwen4.expert;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.core.frames.AbstractFrame;
import io.euhedral_execution.inference.core.model.qwen4.ExpertCacheOwner;
import io.euhedral_execution.inference.core.runtime.graph.AsyncReads;
import io.euhedral_execution.inference.core.runtime.graph.FrameLake;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BooleanSupplier;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/// The owner of the cache's bookkeeping, a source of the lattice: a miss becomes frames whose dependencies carry the
/// load from the read to the device copy, and everything that changes the bookkeeping (the fetch, the retirement, a
/// failure, a lease's release) is a record the source applies when a worker polls it; the reads and the copy's
/// submission are frames. The tests are the lattice: the lake collects the frames that are published and they poll
/// the source and run the frames, one at a time, as a worker would.
class ExpertCacheOwnerTest {

    @TempDir
    Path directory;

    private final HostBackedGpu gpu = new HostBackedGpu();
    private final Lake lake = new Lake();
    private ExpertFixture fixture;
    private ExpertCache cache;
    private ExpertCacheOwner owner;
    /// Asynchronous reads, when a test uses them: their completions are polled into the lake.
    private AsyncReads reads;

    @BeforeEach
    void setUp() throws IOException {
        this.fixture = ExpertFixture.standard(this.directory, 33);
        this.gpu.streamFactory(FakeStream::new);
    }

    @AfterEach
    void tearDown() {
        if (this.cache != null) this.cache.close();
        if (this.reads != null) this.reads.close();
        this.gpu.assertAllReleased();
    }

    /// A cache of `slots` slots whose store pins `buffers` staging buffers up front.
    private void build(int slots, int buffers) throws IOException {
        build(slots, new FileExpertStore(this.gpu, this.fixture.file, this.fixture.banks, buffers));
    }

    private void build(int slots, FileExpertStore store) {
        var transfer = new GpuExpertTransfer(this.gpu, 1);
        this.cache = new ExpertCache(store, transfer, this.gpu, slots, this.fixture.slotBytes(), 1);
        this.owner = new ExpertCacheOwner(this.cache, this.lake);
    }

    /// The pool of ready work: it collects what is published, from any thread.
    private static final class Lake implements FrameLake {
        final ConcurrentLinkedQueue<AbstractFrame> ready = new ConcurrentLinkedQueue<>();
        final AtomicInteger admitted = new AtomicInteger();
        final AtomicInteger terminated = new AtomicInteger();
        final AtomicInteger fromCallbacks = new AtomicInteger();

        @Override
        public void publish(AbstractFrame frame) {
            this.ready.add(frame);
        }

        @Override
        public void publishFromCallback(AbstractFrame frame) {
            this.fromCallbacks.incrementAndGet();
            this.ready.add(frame);
        }

        @Override
        public void admit() {
            this.admitted.incrementAndGet();
        }

        @Override
        public void admitDuringDrain() {
            this.admitted.incrementAndGet();
        }

        @Override
        public void terminated() {
            this.terminated.incrementAndGet();
        }

        long readyOf(String kind) {
            return this.ready.stream()
                    .filter(frame -> frame.getClass().getSimpleName().equals(kind))
                    .count();
        }
    }

    /// What a fetch records of what it is told.
    private static final class Fetch implements ExpertCacheOwner.Fetch {
        final List<ExpertLease> leases = Collections.synchronizedList(new ArrayList<>());
        final List<Throwable> failures = Collections.synchronizedList(new ArrayList<>());

        volatile boolean stopped;
        final AtomicInteger abandoned = new AtomicInteger();

        @Override
        public boolean stopped() {
            return this.stopped;
        }

        @Override
        public void abandoned() {
            this.abandoned.incrementAndGet();
        }

        @Override
        public void arrived(ExpertLease lease) {
            this.leases.add(lease);
        }

        @Override
        public void failed(Throwable failure) {
            this.failures.add(failure);
        }

        boolean ended() {
            return this.leases.size() + this.failures.size() + this.abandoned.get() == 1;
        }

        ExpertLease lease() {
            return this.leases.get(0);
        }
    }

    /// Polls the source and runs what is published, one frame at a time as a worker would, until `condition` holds;
    /// returns the frames run.
    private int drive(BooleanSupplier condition) throws InterruptedException {
        int frames = 0;
        long deadline = System.nanoTime() + 20_000_000_000L;
        while (!condition.getAsBoolean()) {
            if (System.nanoTime() > deadline) throw new AssertionError("the load did not get there");
            this.owner.poll();
            if (this.reads != null) this.reads.getDelegate().pull(this.lake::publish, ready -> false, Long.MAX_VALUE);
            AbstractFrame frame = this.lake.ready.poll();
            if (frame == null) {
                Thread.sleep(0, 200_000);
                continue;
            }
            frame.execute();
            frame.doFinally();
            frames++;
        }
        return frames;
    }

    /// Every load ended and the lake was told so.
    private void drained() throws InterruptedException {
        drive(() -> this.owner.loadsInFlight() == 0 && this.lake.ready.isEmpty() && !this.owner.hasRecords());
        assertEquals(this.lake.admitted.get(), this.lake.terminated.get());
    }

    /// Requests every expert of `bank` in `experts` as fetch stages do: a request that finds the cache full waits in
    /// the source while the frames run, until each expert is held.
    private List<Fetch> fetchAll(int bank, int... experts) throws InterruptedException {
        List<Fetch> targets = new ArrayList<>();
        for (int expert : experts) {
            Fetch target = new Fetch();
            targets.add(target);
            this.owner.request(target, bank, expert);
        }
        drive(() -> targets.stream().allMatch(Fetch::ended));
        return targets;
    }

    /// Fetches as a fetch stage does, on the test thread (which plays the owner's frames).
    private Fetch fetch(int bank, int expert, ExpertCacheOwner.Outcome expected) {
        Fetch target = new Fetch();
        assertEquals(expected, this.owner.fetch(target, bank, expert));
        return target;
    }

    /// The lookahead loads what a layer is predicted to ask for into device slots, holds them for the layer, and never
    /// more than a third of the cache.
    @Test
    void aLookaheadLoadsTheWantedRecordsAndHoldsAThirdOfTheCacheUntilTheLayerIsDone() throws Exception {
        build(9, 4);
        this.owner.publishAhead(3, 2, new int[] {0, 1, 2, 3, 4});
        drive(() -> this.owner.aheadLoads() == 3 && this.owner.loadsInFlight() == 0);
        drive(() -> this.lake.ready.isEmpty() && !this.owner.hasRecords());
        for (int expert = 0; expert < 3; expert++) assertTrue(this.cache.isResident(2, expert), "expert " + expert);
        assertFalse(this.cache.isResident(2, 3), "the hold limit: a third of nine slots");
        assertEquals(3, this.cache.openLeaseCount(), "each loaded record is held");
        // The layer's own fetches find what was loaded, and the layer's end releases the holds, which lets the rest in.
        Fetch own = fetch(2, 1, ExpertCacheOwner.Outcome.LEASED);
        own.lease().close();
        this.owner.aheadDone(3, false);
        drive(() -> this.owner.aheadLoads() == 3 && this.cache.openLeaseCount() == 0 && !this.owner.hasRecords());
        drained();
        assertEquals(3, this.owner.aheadLoads(), "the layer is done: what is left of its wants is dropped");
        this.cache.checkQuiescent();
    }

    @Test
    void aLookaheadForALayerThatAsksForItsExpertsItselfLoadsNothing() throws Exception {
        build(9, 4);
        this.owner.aheadPassed(5);
        this.owner.publishAhead(5, 2, new int[] {0, 1});
        this.owner.poll();
        drained();
        assertEquals(0, this.owner.aheadLoads());
        assertTrue(this.lake.ready.isEmpty());
        this.owner.aheadDone(5, true);
        this.owner.publishAhead(4, 2, new int[] {7});
        drive(() -> this.owner.aheadLoads() == 1 && this.owner.loadsInFlight() == 0);
        this.owner.aheadDone(4, true);
        drained();
        this.cache.checkQuiescent();
    }

    @Test
    void aResidentExpertIsALeaseAtOnceWithNoFrame() throws Exception {
        build(4, 2);
        ExpertTestSupport.acquire(this.cache, 0, 1).close();
        drive(this.lake.ready::isEmpty);
        Fetch hit = fetch(0, 1, ExpertCacheOwner.Outcome.LEASED);
        assertEquals(0, hit.lease().readyMarker(), "the bytes are in the slot already");
        assertTrue(this.lake.ready.isEmpty(), "a hit publishes nothing");
        assertArrayEquals(
                this.fixture.record(0, 1),
                this.gpu.readDevice(hit.lease().deviceAddress(), hit.lease().byteSize()));
        hit.lease().close();
        drained();
        this.cache.checkQuiescent();
    }

    @Test
    void aMissIsAReadThenItsSubmissionWhichHandsOverTheLeaseThenTheOwnersRetire() throws Exception {
        build(4, 2);
        Fetch miss = fetch(2, 3, ExpertCacheOwner.Outcome.LOADING);
        assertEquals(1, this.lake.readyOf("Part"), "the load's first frame: its read, in one part");
        assertFalse(this.owner.hasRecords());
        assertEquals(2, drive(miss::ended), "the read, then the submission that hands the lease over");
        assertNotEquals(0, miss.lease().readyMarker(), "the lease carries the marker of its copy");
        drained();
        assertTrue(
                this.owner.loadsInFlight() == 0, "the copy's retirement, posted by the driver callback, was applied");
        assertArrayEquals(
                this.fixture.record(2, 3),
                this.gpu.readDevice(miss.lease().deviceAddress(), miss.lease().byteSize()));
        miss.lease().close();
        assertEquals(1, this.cache.openLeaseCount(), "a closed lease changes nothing until the owner runs its release");
        assertTrue(this.owner.hasRecords(), "the release is posted, and the owner applies it when it is polled");
        drained();
        assertEquals(0, this.cache.openLeaseCount());
        this.cache.checkQuiescent();
    }

    @Test
    void aReadInPartsFansOutAndJoinsIntoTheSubmission() throws Exception {
        build(
                3,
                new FileExpertStore(
                        this.gpu,
                        new FileRecordSource(this.fixture.file, this.fixture.banks),
                        null,
                        this.fixture.banks,
                        2,
                        4));
        Fetch miss = fetch(2, 4, ExpertCacheOwner.Outcome.LOADING);
        assertEquals(4, this.lake.readyOf("Part"), "the parts are published side by side");
        for (int part = 0; part < 3; part++) this.lake.ready.poll().execute();
        assertEquals(0, this.lake.readyOf("Submit"), "the join waits for every part");
        this.lake.ready.poll().execute();
        assertEquals(1, this.lake.readyOf("Submit"), "the last part's arrival publishes the submission");
        assertFalse(this.owner.hasRecords(), "submitting the copy posts nothing to the owner");
        assertEquals(2, this.cache.store().stagingBuffers(), "the parts share one buffer: nothing more was pinned");
        drive(miss::ended);
        drained();
        assertArrayEquals(
                this.fixture.record(2, 4),
                this.gpu.readDevice(miss.lease().deviceAddress(), miss.lease().byteSize()));
        miss.lease().close();
        drained();
    }

    /// A prefetch reads a record into the pinned tier with no device load; a fetch that finds it still being read
    /// publishes itself again, and once it is in, the fetch copies it from the tier slot.
    @Test
    void aPrefetchReadsIntoTheTierAndAFetchWaitsForItThenCopiesFromIt() throws Exception {
        org.junit.jupiter.api.Assumptions.assumeTrue(AsyncReads.available(), "no io_uring on this machine");
        this.reads = AsyncReads.open();
        this.reads.getDelegate().addDownstream(new io.euhedral_execution.core.generics.LatticeReceiver() {
            @Override
            public void addUpstream(io.euhedral_execution.core.generics.LatticeSource upstream) {}

            @Override
            public void push(AbstractFrame frame) {
                ExpertCacheOwnerTest.this.lake.publish(frame);
            }

            @Override
            public void onComplete() {}

            @Override
            public void onError(Throwable error) {
                throw new AssertionError(error);
            }
        });
        var tier = new RamTier(this.fixture.banks, 10, 1, ReplacementPolicy.FREQUENCY, this.gpu, true);
        build(
                6,
                new FileExpertStore(
                        this.gpu,
                        new FileRecordSource(this.fixture.file, this.fixture.banks, this.reads),
                        tier,
                        this.fixture.banks,
                        2,
                        1));
        this.owner.prefetch(0, new int[] {5, 6, 7}, 2);
        assertTrue(tier.shard(0).isFilling(0, 5) && tier.shard(0).isFilling(0, 6), "two reads, the budget");
        assertFalse(tier.shard(0).isFilling(0, 7));
        assertFalse(this.cache.shard(0).holds(0, 5), "a prefetch takes no device slot");
        fetch(0, 5, ExpertCacheOwner.Outcome.FULL);
        drive(() -> tier.shard(0).isResident(0, 5) && tier.shard(0).isResident(0, 6));
        Fetch hit = fetch(0, 5, ExpertCacheOwner.Outcome.LOADING);
        drive(hit::ended);
        drained();
        assertArrayEquals(
                this.fixture.record(0, 5),
                this.gpu.readDevice(hit.lease().deviceAddress(), hit.lease().byteSize()));
        hit.lease().close();
        drained();
        assertEquals(2, this.owner.prefetchCounts()[0]);
        assertEquals(1, tier.prefetchesUsed());
        assertEquals(0, this.owner.readsInFlight());
        tier.checkInvariants();
        this.cache.checkQuiescent();
    }

    @Test
    void aReadInPartsSubmitsItsReadsAndTheirCompletionsAreFrames() throws Exception {
        org.junit.jupiter.api.Assumptions.assumeTrue(AsyncReads.available(), "no io_uring on this machine");
        this.reads = AsyncReads.open();
        this.reads.getDelegate().addDownstream(new io.euhedral_execution.core.generics.LatticeReceiver() {
            @Override
            public void addUpstream(io.euhedral_execution.core.generics.LatticeSource upstream) {}

            @Override
            public void push(AbstractFrame frame) {
                ExpertCacheOwnerTest.this.lake.publish(frame);
            }

            @Override
            public void onComplete() {}

            @Override
            public void onError(Throwable error) {
                throw new AssertionError(error);
            }
        });
        var tier = new RamTier(this.fixture.banks, 10, 1, ReplacementPolicy.BANK_PARTITIONED);
        build(
                6,
                new FileExpertStore(
                        this.gpu,
                        new FileRecordSource(this.fixture.file, this.fixture.banks, this.reads),
                        tier,
                        this.fixture.banks,
                        2,
                        4));
        Fetch miss = fetch(0, 5, ExpertCacheOwner.Outcome.LOADING);
        assertEquals(4, this.lake.readyOf("Part"));
        for (int part = 0; part < 4; part++) this.lake.ready.poll().execute();
        assertEquals(0, this.lake.readyOf("Submit"), "the parts submitted their reads and ended");
        drive(miss::ended);
        drained();
        assertEquals(0, this.reads.inFlight());
        assertArrayEquals(
                this.fixture.record(0, 5),
                this.gpu.readDevice(miss.lease().deviceAddress(), miss.lease().byteSize()));
        miss.lease().close();
        drained();
        assertTrue(tier.shard(0).isResident(0, 5), "the fill kept the record");
        tier.checkInvariants();
        this.cache.checkQuiescent();
    }

    /// A load holds one of the disk's reads only while it reads: the read is back once the record is in, before the
    /// device copy retires.
    @Test
    void aLoadGivesItsReadBackWhenTheRecordIsInNotWhenItsCopyRetires() throws Exception {
        build(4, 2);
        Fetch miss = fetch(2, 0, ExpertCacheOwner.Outcome.LOADING);
        assertEquals(1, this.owner.readsInFlight(), "the fetch took a read");
        drive(miss::ended);
        assertEquals(0, this.owner.readsInFlight(), "the record is in: the read is back");
        drained();
        assertEquals(0, this.owner.readsInFlight());
        miss.lease().close();
        drained();
        this.cache.checkQuiescent();
    }

    @Test
    void aMissNeedsAStagingBufferAsWellAsASlotAndFindsTheCacheFullWithoutOne() throws Exception {
        build(4, 1);
        Fetch first = fetch(2, 0, ExpertCacheOwner.Outcome.LOADING);
        Fetch second = fetch(2, 1, ExpertCacheOwner.Outcome.FULL);
        assertTrue(this.cache.isResident(2, 0) || this.cache.shard(0).loadingCount() == 1);
        assertEquals(1, this.cache.shard(0).loadingCount(), "the full fetch reserved no slot");
        // A hit needs no buffer.
        drive(first::ended);
        drained();
        Fetch hit = fetch(2, 0, ExpertCacheOwner.Outcome.LEASED);
        hit.lease().close();
        // The buffer came back with the copy's retirement: the second miss loads.
        second = fetch(2, 1, ExpertCacheOwner.Outcome.LOADING);
        Fetch loaded = second;
        drive(loaded::ended);
        drained();
        assertArrayEquals(
                this.fixture.record(2, 1),
                this.gpu.readDevice(
                        loaded.lease().deviceAddress(), loaded.lease().byteSize()));
        first.lease().close();
        loaded.lease().close();
        drained();
        assertEquals(1, this.cache.store().stagingBuffers(), "the staging buffers are a fixed pool");
        this.cache.checkQuiescent();
    }

    @Test
    void aFetchFindsEverySlotPinnedAndTriesAgainOnceALeaseWasReleased() throws Exception {
        build(2, 2);
        Fetch a = fetch(2, 0, ExpertCacheOwner.Outcome.LOADING);
        Fetch b = fetch(2, 1, ExpertCacheOwner.Outcome.LOADING);
        drive(() -> a.ended() && b.ended());
        drained();
        Fetch c = fetch(2, 2, ExpertCacheOwner.Outcome.FULL);
        assertTrue(this.lake.ready.isEmpty(), "nothing changed and nothing waits");
        assertEquals(1, this.owner.fullFetches());
        a.lease().close();
        drained();
        c = fetch(2, 2, ExpertCacheOwner.Outcome.LOADING);
        Fetch done = c;
        drive(done::ended);
        b.lease().close();
        done.lease().close();
        drained();
        this.cache.checkQuiescent();
    }

    /// A request that finds the cache full waits in the source: polling it again and again asks the cache nothing
    /// more, and it is served once a lease is released.
    @Test
    @org.junit.jupiter.api.Timeout(30)
    void aRequestForAFullCacheWaitsInTheSourceUntilALeaseIsReleased() throws Exception {
        build(2, 2);
        Fetch a = fetch(2, 0, ExpertCacheOwner.Outcome.LOADING);
        Fetch b = fetch(2, 1, ExpertCacheOwner.Outcome.LOADING);
        drive(() -> a.ended() && b.ended());
        drained();
        Fetch c = new Fetch();
        this.owner.request(c, 2, 2);
        this.owner.poll();
        assertEquals(1, this.owner.blockedFetches(), "no slot is free or evictable: the request waits");
        assertEquals(1, this.owner.fullFetches());
        for (int poll = 0; poll < 1_000; poll++) this.owner.poll();
        assertEquals(1, this.owner.fullFetches(), "polling a source whose cache did not change asks nothing");
        assertFalse(c.ended());
        a.lease().close();
        drive(c::ended);
        assertEquals(0, this.owner.blockedFetches());
        drained();
        b.lease().close();
        c.lease().close();
        drained();
        this.cache.checkQuiescent();
    }

    /// Fetches that wait for the same thing are asked once per change, not once each: the first that finds the staging
    /// buffers all in use is the answer for the ones behind it.
    @Test
    void fetchesWaitingForTheSameStagingBufferAreAskedOncePerChange() throws Exception {
        build(8, 1);
        Fetch first = fetch(2, 0, ExpertCacheOwner.Outcome.LOADING);
        List<Fetch> waiting = new ArrayList<>();
        for (int expert = 1; expert < 5; expert++) {
            Fetch target = new Fetch();
            waiting.add(target);
            this.owner.request(target, 2, expert);
        }
        this.owner.poll();
        assertEquals(4, this.owner.blockedFetches());
        long before = this.owner.fullFetches();
        drive(first::ended);
        drained();
        first.lease().close();
        drive(() -> waiting.stream().allMatch(Fetch::ended));
        drained();
        for (Fetch target : waiting) target.lease().close();
        drained();
        assertEquals(0, this.owner.blockedFetches());
        assertTrue(this.owner.fullFetches() - before < 4 * 8, "waiting fetches were not asked again on every change");
        this.cache.checkQuiescent();
    }

    /// A request whose quantum stopped while it waited is abandoned without anything having to free a slot.
    @Test
    @org.junit.jupiter.api.Timeout(30)
    void aWaitingRequestOfAStoppedQuantumIsAbandoned() throws Exception {
        build(1, 2);
        Fetch held = fetch(2, 0, ExpertCacheOwner.Outcome.LOADING);
        drive(held::ended);
        drained();
        Fetch waiting = new Fetch();
        this.owner.request(waiting, 2, 1);
        this.owner.poll();
        assertEquals(1, this.owner.blockedFetches());
        waiting.stopped = true;
        drive(waiting::ended);
        assertEquals(1, waiting.abandoned.get());
        assertEquals(0, this.owner.blockedFetches());
        held.lease().close();
        drained();
        this.cache.checkQuiescent();
    }

    @Test
    void twoBlocksAskingForTheSameExpertsAtOnceBothFinishWithOneLoadEach() throws Exception {
        build(4, 2);
        // Two blocks' fetches, interleaved: experts 2 and 3 are asked by both, the second time while their loads
        // may not have submitted their copies yet.
        List<Fetch> fetches = fetchAll(2, 1, 2, 2, 3, 3, 4);
        drained();
        for (Fetch fetch : fetches) assertTrue(fetch.ended() && fetch.lease() != null);
        assertEquals(
                4,
                this.cache.shard(this.cache.shardOf(2, 1)).stats().snapshot().misses() + otherShardMisses(2, 1),
                "one load per expert");
        for (Fetch fetch : fetches) fetch.lease().close();
        drained();
        this.cache.checkQuiescent();
    }

    /// Misses of every shard but the one holding (`bank`, `expert`).
    private long otherShardMisses(int bank, int expert) {
        long misses = 0;
        int own = this.cache.shardOf(bank, expert);
        for (int shard = 0; shard < this.cache.shardCount(); shard++)
            if (shard != own)
                misses += this.cache.shard(shard).stats().snapshot().misses();
        return misses;
    }

    @Test
    void aCopyThatCannotBeSubmittedFailsTheFetchAndReturnsItsSlotAndBuffer() throws Exception {
        build(2, 2);
        this.gpu.failNextHostCopies(1);
        Fetch failed = fetch(2, 4, ExpertCacheOwner.Outcome.LOADING);
        drive(failed::ended);
        assertEquals(1, failed.failures.size());
        assertTrue(failed.leases.isEmpty());
        drained();
        this.cache.checkQuiescent();
        assertEquals(1, this.cache.shard(0).stats().snapshot().failedTransfers());
        Fetch again = fetch(2, 4, ExpertCacheOwner.Outcome.LOADING);
        drive(again::ended);
        again.lease().close();
        drained();
        assertEquals(2, this.cache.store().stagingBuffers(), "the failed load's buffer came back");
    }

    @Test
    void aDeviceMissAfterItsEvictionIsAnsweredFromTheRamTierWithoutReadingTheArtifact() throws Exception {
        build(2, ExpertTestSupport.ramStore(this.gpu, this.fixture.file, this.fixture.banks, 2, 10, 1));
        var tier = ((FileExpertStore) this.cache.store()).ramTier();
        for (int expert = 0; expert < 4; expert++) {
            Fetch miss = fetch(2, expert, ExpertCacheOwner.Outcome.LOADING);
            drive(miss::ended);
            miss.lease().close();
            drained();
        }
        assertEquals(4, tier.stats().totalMisses(), "every first touch fills a slot");
        assertEquals(0, tier.stats().totalHits());
        long artifactReads = this.cache.store().bytesRead();
        Fetch again = fetch(2, 0, ExpertCacheOwner.Outcome.LOADING);
        drive(again::ended);
        drained();
        assertNotEquals(0, again.lease().readyMarker(), "the device missed: a copy was made");
        assertArrayEquals(
                this.fixture.record(2, 0),
                this.gpu.readDevice(again.lease().deviceAddress(), again.lease().byteSize()));
        again.lease().close();
        drained();
        assertEquals(1, tier.stats().totalHits(), "the device miss was a RAM hit");
        assertEquals(artifactReads, this.cache.store().bytesRead(), "and read nothing from the artifact");
        this.cache.checkQuiescent();
        tier.checkInvariants();
        assertEquals(0, this.cache.store().tier(0).pinnedSlots(), "every tier slot was released");
    }

    @Test
    void aPinnedTierIsCopiedFromInPlaceWithNoStagingBuffer() throws Exception {
        var tier = new RamTier(this.fixture.banks, 10, 1, ReplacementPolicy.BANK_PARTITIONED, this.gpu);
        assertTrue(tier.pinned());
        var store = new FileExpertStore(
                this.gpu, new FileRecordSource(this.fixture.file, this.fixture.banks), tier, this.fixture.banks, 1, 2);
        build(1, store);
        // A record read from the artifact reserves a staging buffer with its slot, and a fill of a pinned tier gives
        // it back at once: the load reads into the tier slot.
        Fetch fill = fetch(2, 0, ExpertCacheOwner.Outcome.LOADING);
        int free = store.acquireStaging();
        assertTrue(free >= 0, "the fill gave its staging buffer back");
        store.releaseStaging(free);
        drive(fill::ended);
        assertTrue(fill.failures.isEmpty(), "a fill reads into its tier slot and is copied from there");
        drained();
        fill.lease().close();
        drained();
        // Another expert takes the device's only slot; the first is then a hit in the tier.
        Fetch other = fetch(2, 1, ExpertCacheOwner.Outcome.LOADING);
        drive(other::ended);
        drained();
        other.lease().close();
        drained();
        // The one staging buffer is held: a hit in a pinned tier needs none.
        int held = store.acquireStaging();
        assertEquals(-1, store.acquireStaging());
        Fetch hit = fetch(2, 0, ExpertCacheOwner.Outcome.LOADING);
        assertTrue(hit.ended(), "the fetch submitted the copy from the tier slot itself");
        assertTrue(this.lake.ready.isEmpty() || this.lake.readyOf("Copy") == 0, "with no frame of its own");
        drained();
        assertArrayEquals(
                this.fixture.record(2, 0),
                this.gpu.readDevice(hit.lease().deviceAddress(), hit.lease().byteSize()));
        hit.lease().close();
        drained();
        assertEquals(1, tier.stats().totalHits());
        store.releaseStaging(held);
        tier.checkInvariants();
        this.cache.checkQuiescent();
    }

    /// A fetch that has to wait asks the tier nothing: planning a fill would take a slot, and evict its record, for
    /// a load that does not start, again on every try.
    @Test
    void aFetchThatWaitsLeavesTheTierAsItWas() throws Exception {
        var tier = new RamTier(this.fixture.banks, 1, 1, ReplacementPolicy.BANK_PARTITIONED, this.gpu);
        var store = new FileExpertStore(
                this.gpu, new FileRecordSource(this.fixture.file, this.fixture.banks), tier, this.fixture.banks, 1, 1);
        build(1, store);
        Fetch fill = fetch(2, 0, ExpertCacheOwner.Outcome.LOADING);
        drive(fill::ended);
        drained();
        fill.lease().close();
        drained();
        assertTrue(tier.shard(0).isResident(2, 0));
        int held = store.acquireStaging();
        for (int attempt = 0; attempt < 3; attempt++) fetch(2, 1, ExpertCacheOwner.Outcome.FULL);
        assertTrue(tier.shard(0).isResident(2, 0), "the tier kept its record");
        assertEquals(0, tier.stats().totalEvictions());
        store.releaseStaging(held);
        Fetch other = fetch(2, 1, ExpertCacheOwner.Outcome.LOADING);
        drive(other::ended);
        drained();
        other.lease().close();
        drained();
        tier.checkInvariants();
        this.cache.checkQuiescent();
    }

    @Test
    void aReadThatFailsGivesItsTierSlotBackAndAFailedCopyKeepsTheRecordItRead() throws Exception {
        var failingReads = new RecordSource() {
            private final FileRecordSource real = new FileRecordSource(fixture.file, fixture.banks);
            private int calls;

            @Override
            public void read(ExpertBank bank, int expert, java.lang.foreign.MemorySegment destination)
                    throws IOException, InterruptedException {
                if (this.calls++ == 0) throw new IOException("injected");
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
        var tier = new RamTier(this.fixture.banks, 10, 1, ReplacementPolicy.BANK_PARTITIONED);
        build(4, new FileExpertStore(this.gpu, failingReads, tier, this.fixture.banks, 2));
        Fetch failed = fetch(2, 0, ExpertCacheOwner.Outcome.LOADING);
        drive(failed::ended);
        drained();
        assertEquals(1, failed.failures.size());
        assertEquals(0, tier.stats().residentExperts(), "the failed read left nothing in its slot");
        assertEquals(0, tier.shard(0).pinnedSlots());
        tier.checkInvariants();

        this.gpu.failNextHostCopies(1);
        Fetch copy = fetch(2, 1, ExpertCacheOwner.Outcome.LOADING);
        drive(copy::ended);
        drained();
        assertEquals(1, copy.failures.size());
        assertTrue(tier.shard(0).isResident(2, 1), "the record was read: the tier keeps it");
        tier.checkInvariants();
        assertEquals(0, tier.shard(0).pinnedSlots());
    }

    @Test
    void anArtifactReadInPartsLoadsTheSameBytesWithAndWithoutATierSlot() throws Exception {
        var tier = new RamTier(this.fixture.banks, 10, 1, ReplacementPolicy.BANK_PARTITIONED);
        build(
                6,
                new FileExpertStore(
                        this.gpu,
                        new FileRecordSource(this.fixture.file, this.fixture.banks),
                        tier,
                        this.fixture.banks,
                        4,
                        4));
        assertEquals(4, this.cache.store().readParts());
        // Banks 0 (9000-byte records) and 2 (8192 bytes at page-aligned offsets): the parts end at page
        // boundaries, so the last parts are short or empty. Eighteen experts through a 10-slot tier fill it,
        // and the rest bypass or evict.
        for (int bank : new int[] {0, 2, 4}) {
            List<Fetch> misses = fetchAll(bank, 0, 1, 2, 3, 4, 5);
            drained();
            for (int expert = 0; expert < 6; expert++) {
                ExpertLease lease = misses.get(expert).lease();
                assertArrayEquals(
                        this.fixture.record(bank, expert),
                        this.gpu.readDevice(lease.deviceAddress(), lease.byteSize()));
                lease.close();
            }
            drained();
        }
        tier.checkInvariants();
        assertEquals(0, tier.shard(0).pinnedSlots());
        this.cache.checkQuiescent();
    }

    @Test
    void aFailedPartFailsTheLoadOnceAndGivesItsTierSlotBack() throws Exception {
        var failingPart = new RecordSource() {
            private final FileRecordSource real = new FileRecordSource(fixture.file, fixture.banks);

            @Override
            public boolean ranged() {
                return true;
            }

            @Override
            public void read(ExpertBank bank, int expert, java.lang.foreign.MemorySegment destination)
                    throws IOException, InterruptedException {
                this.real.read(bank, expert, destination);
            }

            @Override
            public void readRange(ExpertBank bank, int expert, long from, java.lang.foreign.MemorySegment destination)
                    throws IOException, InterruptedException {
                if (from > 0 && expert == 2) throw new IOException("injected");
                this.real.readRange(bank, expert, from, destination);
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
        var tier = new RamTier(this.fixture.banks, 10, 1, ReplacementPolicy.BANK_PARTITIONED);
        build(4, new FileExpertStore(this.gpu, failingPart, tier, this.fixture.banks, 4, 4));
        Fetch[] targets = {new Fetch(), new Fetch(), new Fetch()};
        for (int i = 0; i < 3; i++) this.owner.fetch(targets[i], 2, 1 + i);
        drive(() -> targets[0].ended() && targets[1].ended() && targets[2].ended());
        assertEquals(1, targets[1].failures.size(), "the failed load fails its fetch once");
        assertNull(targets[1].leases.isEmpty() ? null : targets[1].lease());
        targets[0].lease().close();
        targets[2].lease().close();
        drained();
        assertFalse(tier.shard(0).isResident(2, 2));
        assertTrue(tier.shard(0).isResident(2, 1));
        assertEquals(0, tier.shard(0).pinnedSlots());
        tier.checkInvariants();
        this.cache.checkQuiescent();
    }
}
