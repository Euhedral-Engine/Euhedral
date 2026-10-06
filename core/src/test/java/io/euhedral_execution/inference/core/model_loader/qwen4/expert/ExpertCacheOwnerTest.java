package io.euhedral_execution.inference.core.model_loader.qwen4.expert;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.core.frames.AbstractFrame;
import io.euhedral_execution.inference.core.qwen4.ExpertCacheOwner;
import io.euhedral_execution.inference.core.scheduling.graph.AsyncReads;
import io.euhedral_execution.inference.core.scheduling.graph.FrameLake;
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

/// The owner of the cache's bookkeeping: a miss becomes frames whose dependencies carry the load from the read to
/// the device copy, and everything that changes the bookkeeping (the fetch, the retirement, a failure, a lease's
/// release) is a frame routed with the owner's hash; the reads and the copy's submission are not. The tests are the
/// lattice: the lake collects what is published
/// and they run it, one frame at a time, as a worker would.
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

        long readyForTheOwner() {
            return this.ready.stream()
                    .filter(frame -> frame.getRoutingHash() == ExpertCacheOwner.HASH)
                    .count();
        }
    }

    /// What a fetch records of what it is told.
    private static final class Fetch implements ExpertCacheOwner.Fetch {
        final List<ExpertLease> leases = Collections.synchronizedList(new ArrayList<>());
        final List<Throwable> failures = Collections.synchronizedList(new ArrayList<>());

        @Override
        public boolean stopped() {
            return false;
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
            return this.leases.size() + this.failures.size() == 1;
        }

        ExpertLease lease() {
            return this.leases.get(0);
        }
    }

    /// Runs what is published, one frame at a time as a worker would, until `condition` holds; returns the frames
    /// run.
    private int drive(BooleanSupplier condition) throws InterruptedException {
        int frames = 0;
        long deadline = System.nanoTime() + 20_000_000_000L;
        while (!condition.getAsBoolean()) {
            if (System.nanoTime() > deadline) throw new AssertionError("the load did not get there");
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
        drive(() -> this.owner.loadsInFlight() == 0 && this.lake.ready.isEmpty());
        assertEquals(this.lake.admitted.get(), this.lake.terminated.get());
    }

    /// Fetches every expert of `bank` in `experts` as fetch stages do: a fetch that finds the cache full tries
    /// again while the frames run, until each expert is held.
    private List<Fetch> fetchAll(int bank, int... experts) throws InterruptedException {
        List<Fetch> targets = new ArrayList<>();
        boolean[] started = new boolean[experts.length];
        for (int i = 0; i < experts.length; i++) targets.add(new Fetch());
        drive(() -> {
            for (int i = 0; i < experts.length; i++)
                if (!started[i])
                    started[i] = this.owner.fetch(targets.get(i), bank, experts[i]) != ExpertCacheOwner.Outcome.FULL;
            return targets.stream().allMatch(Fetch::ended);
        });
        return targets;
    }

    /// Fetches as a fetch stage does, on the test thread (which plays the owner's frames).
    private Fetch fetch(int bank, int expert, ExpertCacheOwner.Outcome expected) {
        Fetch target = new Fetch();
        assertEquals(expected, this.owner.fetch(target, bank, expert));
        return target;
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
        assertEquals(0, this.lake.readyForTheOwner());
        assertEquals(2, drive(miss::ended), "the read, then the submission that hands the lease over");
        assertNotEquals(0, miss.lease().readyMarker(), "the lease carries the marker of its copy");
        drained();
        assertEquals(1, this.lake.fromCallbacks.get(), "the copy's retirement is a frame a driver callback published");
        assertArrayEquals(
                this.fixture.record(2, 3),
                this.gpu.readDevice(miss.lease().deviceAddress(), miss.lease().byteSize()));
        miss.lease().close();
        assertEquals(1, this.cache.openLeaseCount(), "a closed lease changes nothing until the owner runs its release");
        assertEquals(1, this.lake.readyForTheOwner());
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
        assertEquals(0, this.lake.readyForTheOwner(), "submitting the copy touches none of the owner's state");
        assertEquals(2, this.cache.store().stagingBuffers(), "the parts share one buffer: nothing more was pinned");
        drive(miss::ended);
        drained();
        assertArrayEquals(
                this.fixture.record(2, 4),
                this.gpu.readDevice(miss.lease().deviceAddress(), miss.lease().byteSize()));
        miss.lease().close();
        drained();
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
