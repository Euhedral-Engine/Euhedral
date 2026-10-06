package io.euhedral_execution.inference.core.model_loader.qwen4.expert;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.core.frames.AbstractFrame;
import io.euhedral_execution.core.generics.LatticeReceiver;
import io.euhedral_execution.core.generics.LatticeSource;
import io.euhedral_execution.inference.core.qwen4.ExpertBlock;
import io.euhedral_execution.inference.core.qwen4.ExpertSource;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BooleanSupplier;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/// The serial source that owns a shard of the cache: it generates its frames from `request` and
/// `pull`, walks a block's experts in order, and answers the block with leases as the loads
/// progress. The tests are the lattice: they pull, run the frames the source hands out, and read
/// what the block was told.
class ExpertSourceTest {

    @TempDir
    Path directory;

    private final HostBackedGpu gpu = new HostBackedGpu();
    private ExpertFixture fixture;
    private ExpertCache cache;
    private ExpertSource source;

    @BeforeEach
    void setUp() throws IOException {
        this.fixture = ExpertFixture.standard(this.directory, 33);
        this.gpu.streamFactory(FakeStream::new);
    }

    @AfterEach
    void tearDown() {
        if (this.cache != null) this.cache.close();
        this.gpu.assertAllReleased();
    }

    private void build(int slots, int lanes) throws IOException {
        build(slots, lanes, new FileExpertStore(this.gpu, this.fixture.file, this.fixture.banks, lanes));
    }

    private void build(int slots, int lanes, FileExpertStore store) {
        var transfer = new GpuExpertTransfer(this.gpu, lanes);
        this.cache = new ExpertCache(store, transfer, this.gpu, slots, this.fixture.slotBytes(), 1);
        this.source = new ExpertSource(this.cache, 0);
        this.source.addDownstream(new LatticeReceiver() {
            @Override
            public void addUpstream(LatticeSource upstream) {}

            @Override
            public void push(AbstractFrame frame) {}

            @Override
            public void onComplete() {}

            @Override
            public void onError(Throwable error) {
                throw new AssertionError(error);
            }
        });
    }

    /// What a block records of what the source tells it.
    private final class Block implements ExpertBlock {
        final int bank;
        final int[] experts;
        final List<Object[]> arrivals = Collections.synchronizedList(new ArrayList<>());
        final List<Throwable> failures = Collections.synchronizedList(new ArrayList<>());
        final AtomicBoolean stopped = new AtomicBoolean();
        final AtomicInteger done = new AtomicInteger();

        Block(int bank, int... experts) {
            this.bank = bank;
            this.experts = experts;
        }

        @Override
        public int bank() {
            return this.bank;
        }

        @Override
        public int expertAt(int position) {
            return this.experts[position];
        }

        @Override
        public boolean stopped() {
            return this.stopped.get();
        }

        @Override
        public void arrive(int position, ExpertLease lease) {
            this.arrivals.add(new Object[] {position, lease});
        }

        @Override
        public void failed(Throwable failure) {
            this.failures.add(failure);
        }

        @Override
        public void shardDone() {
            this.done.incrementAndGet();
        }

        ExpertLease lease(int position) {
            for (Object[] arrival : this.arrivals) if ((int) arrival[0] == position) return (ExpertLease) arrival[1];
            return null;
        }

        ExpertSource.Work work() {
            int[] positions = new int[this.experts.length];
            for (int i = 0; i < positions.length; i++) positions[i] = i;
            return new ExpertSource.Work().set(this, positions, positions.length);
        }
    }

    /// Pulls, running the frames the source hands out as the lattice's workers would, until
    /// `condition` holds.
    private int drive(BooleanSupplier condition) throws InterruptedException {
        int frames = 0;
        long deadline = System.nanoTime() + 20_000_000_000L;
        while (!condition.getAsBoolean()) {
            if (System.nanoTime() > deadline) throw new AssertionError("the source did not get there");
            frames += (int) this.source.pull(
                    frame -> {
                        frame.execute();
                        frame.doFinally();
                    },
                    frame -> false,
                    Long.MAX_VALUE);
            Thread.sleep(0, 200_000);
        }
        return frames;
    }

    @Test
    void residentExpertsAreLeasedWithoutAnyFrame() throws Exception {
        build(4, 2);
        for (int expert = 0; expert < 3; expert++)
            ExpertTestSupport.acquire(this.cache, 0, expert).close();
        Block block = new Block(0, 0, 1, 2);
        this.source.submit(block.work());
        int frames = drive(() -> block.done.get() == 1);
        assertEquals(0, frames, "a hit needs no load frame");
        assertEquals(3, block.arrivals.size());
        for (int position = 0; position < 3; position++) {
            ExpertLease lease = block.lease(position);
            assertEquals(0, lease.readyMarker(), "the bytes are in the slot already");
            assertArrayEquals(
                    this.fixture.record(0, position), this.gpu.readDevice(lease.deviceAddress(), lease.byteSize()));
            lease.close();
        }
        drive(() -> this.cache.openLeaseCount() == 0);
    }

    @Test
    void aMissBecomesALoadFrameAndItsLeaseArrivesCarryingTheCopysMarker() throws Exception {
        build(4, 2);
        Block block = new Block(2, 3, 5);
        this.source.submit(block.work());
        int frames = drive(() -> block.done.get() == 1);
        assertEquals(2, frames, "each miss is one load frame, generated by the source");
        assertEquals(2, block.arrivals.size());
        for (int position = 0; position < 2; position++) {
            ExpertLease lease = block.lease(position);
            assertNotEquals(0, lease.readyMarker(), "the lease carries the marker of its copy");
            assertArrayEquals(
                    this.fixture.record(2, block.experts[position]),
                    this.gpu.readDevice(lease.deviceAddress(), lease.byteSize()));
            lease.close();
        }
        assertTrue(block.failures.isEmpty());
        drive(() -> this.cache.openLeaseCount() == 0);
        this.cache.checkQuiescent();
    }

    @Test
    void theWalkWaitsForASlotAndResumesWhenALeaseCloses() throws Exception {
        build(2, 2);
        Block block = new Block(2, 0, 1, 2, 3);
        this.source.submit(block.work());
        drive(() -> block.arrivals.size() == 2);
        // Both slots hold leases of this block: the third expert has no room, and nothing is claimed for it.
        for (int i = 0; i < 20; i++) this.source.pull(frame -> {}, frame -> false, 0);
        drive(() -> true);
        assertEquals(2, block.arrivals.size(), "the walk waits for room instead of failing or queueing a thread");
        assertEquals(0, block.done.get());
        block.lease(0).close();
        drive(() -> block.arrivals.size() == 3);
        block.lease(1).close();
        drive(() -> block.arrivals.size() == 4 && block.done.get() == 1);
        for (int position = 0; position < 4; position++) {
            ExpertLease lease = block.lease(position);
            if (lease != null && lease.isValid()) lease.close();
        }
        drive(() -> this.cache.openLeaseCount() == 0);
        this.cache.checkQuiescent();
    }

    @Test
    void aStoppedBlockLoadsNothingYetEveryExpertArrives() throws Exception {
        build(4, 2);
        Block block = new Block(2, 0, 1, 2);
        block.stopped.set(true);
        this.source.submit(block.work());
        int frames = drive(() -> block.done.get() == 1);
        assertEquals(0, frames);
        assertEquals(3, block.arrivals.size());
        for (Object[] arrival : block.arrivals) assertNull(arrival[1], "no lease for a quantum that stopped");
        assertEquals(0, this.cache.shard(0).stats().snapshot().misses());
    }

    @Test
    void aCopyThatCannotBeSubmittedFailsTheBlockAndReturnsItsSlot() throws Exception {
        build(2, 2);
        this.gpu.failNextHostCopies(1);
        Block block = new Block(2, 4, 5);
        this.source.submit(block.work());
        drive(() -> block.done.get() == 1);
        assertEquals(1, block.failures.size(), "the failed load fails the block");
        assertEquals(2, block.arrivals.size(), "and still arrives, so the quantum's stages end");
        int leases = 0;
        for (Object[] arrival : block.arrivals)
            if (arrival[1] != null) {
                leases++;
                ((ExpertLease) arrival[1]).close();
            }
        assertEquals(1, leases, "the other expert loaded");
        drive(() -> this.cache.openLeaseCount() == 0);
        this.cache.checkQuiescent();
        assertEquals(1, this.cache.shard(0).stats().snapshot().failedTransfers());
    }

    @Test
    void aLeaseClosedFromAnotherThreadReachesTheOwnerOnlyThroughItsMailbox() throws Exception {
        build(2, 2);
        Block block = new Block(2, 6);
        this.source.submit(block.work());
        drive(() -> block.done.get() == 1);
        ExpertLease lease = block.lease(0);
        Thread closer = new Thread(lease::close);
        closer.start();
        closer.join();
        assertEquals(1, this.cache.openLeaseCount(), "the cache is untouched until the owner handles the release");
        drive(() -> this.cache.openLeaseCount() == 0);
        assertFalse(lease.isValid());
        assertNotNull(this.cache.shard(0));
    }

    @Test
    void aDeviceMissAfterItsEvictionIsAnsweredFromTheRamTierWithoutReadingTheArtifact() throws Exception {
        build(2, 2, ExpertTestSupport.ramStore(this.gpu, this.fixture.file, this.fixture.banks, 2, 10, 1));
        var tier = ((FileExpertStore) this.cache.store()).ramTier();
        Block first = new Block(2, 0, 1, 2, 3);
        this.source.submit(first.work());
        // Four experts through two device slots: close each lease as it arrives so the walk goes on.
        boolean[] closed = new boolean[4];
        drive(() -> {
            for (int position = 0; position < 4; position++) {
                ExpertLease lease = first.lease(position);
                if (lease != null && !closed[position]) {
                    closed[position] = true;
                    lease.close();
                }
            }
            return first.done.get() == 1 && this.cache.openLeaseCount() == 0;
        });
        assertEquals(4, tier.stats().totalMisses(), "every first touch fills a slot");
        assertEquals(0, tier.stats().totalHits());
        long artifactReads = this.cache.store().bytesRead();

        Block again = new Block(2, 0);
        this.source.submit(again.work());
        drive(() -> again.done.get() == 1);
        ExpertLease lease = again.lease(0);
        assertNotEquals(0, lease.readyMarker(), "the device missed: a copy was made");
        assertArrayEquals(this.fixture.record(2, 0), this.gpu.readDevice(lease.deviceAddress(), lease.byteSize()));
        lease.close();
        drive(() -> this.cache.openLeaseCount() == 0);
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
        var tier = new RamTier(this.fixture.banks, 10, 1, RamTier.Policy.BANK_PARTITIONED);
        build(4, 2, new FileExpertStore(this.gpu, failingReads, tier, this.fixture.banks, 2));

        Block failed = new Block(2, 0);
        this.source.submit(failed.work());
        drive(() -> failed.done.get() == 1);
        assertEquals(1, failed.failures.size());
        assertEquals(0, tier.stats().residentExperts(), "the failed read left nothing in its slot");
        assertEquals(0, tier.shard(0).pinnedSlots());
        tier.checkInvariants();

        this.gpu.failNextHostCopies(1);
        Block copy = new Block(2, 1);
        this.source.submit(copy.work());
        drive(() -> copy.done.get() == 1);
        assertEquals(1, copy.failures.size());
        assertTrue(tier.shard(0).isResident(2, 1), "the record was read: the tier keeps it");
        tier.checkInvariants();
        assertEquals(0, tier.shard(0).pinnedSlots());
    }
}
