package io.euhedral_execution.inference.core.model_loader.qwen4.expert;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/// The host tier: a record is read from the artifact once and answered from memory after, a resident tier
/// preloads everything and reads nothing afterwards, a bounded tier stays within its slots and returns the
/// artifact's bytes, a shard's pinned slot is never taken, shards are independent, and a layer-aware policy
/// keeps what a cyclic pass would evict from a global LRU. Each shard is driven by one thread at a time, as
/// its owner drives it.
class RamTierTest {
    // Fixture keys: bank 0 has 8 experts, bank 1 has 3, bank 2 has 16, bank 3 has 1, bank 4 has 12 (40 in all).

    @TempDir
    Path directory;

    private final HostBackedGpu gpu = new HostBackedGpu();
    private ExpertFixture fixture;
    private FileExpertStore store;

    @BeforeEach
    void setUp() throws IOException {
        this.fixture = ExpertFixture.standard(this.directory, 101);
    }

    @AfterEach
    void tearDown() {
        if (this.store != null) this.store.close();
        this.gpu.assertAllReleased();
    }

    private FileExpertStore store(int slots, int shards, ReplacementPolicy policy, RecordSource source)
            throws IOException {
        RamTier tier = new RamTier(this.fixture.banks, slots, shards, policy);
        this.store = new FileExpertStore(this.gpu, source, tier, this.fixture.banks, 4);
        return this.store;
    }

    private FileExpertStore store(int slots, int shards, ReplacementPolicy policy) throws IOException {
        return store(slots, shards, policy, new FileRecordSource(this.fixture.file, this.fixture.banks));
    }

    private int shardOf(int bank, int expert) {
        return ExpertKeys.shardOf(
                new ExpertKeys(this.fixture.banks).key(bank, expert),
                this.store.ramTier().shards());
    }

    /// One load as a serial source drives it: plan, open, settle. Returns the staged bytes.
    private byte[] load(int bank, int expert, int lane) throws Exception {
        RamTierShard shard = this.store.tier(shardOf(bank, expert));
        TierDirective directive = new TierDirective();
        shard.plan(bank, expert, directive);
        HostRecord record;
        try {
            record = this.store.open(bank, expert, lane, directive);
        } catch (Exception | Error failure) {
            if (directive.mode() == TierDirective.Mode.FILL) shard.abandoned(directive);
            else if (directive.mode() == TierDirective.Mode.HIT) shard.used(directive);
            throw failure;
        }
        byte[] bytes = MemorySegment.ofAddress(record.hostAddress())
                .reinterpret(record.byteSize())
                .toArray(ValueLayout.JAVA_BYTE);
        if (directive.mode() == TierDirective.Mode.FILL) shard.filled(directive);
        else if (directive.mode() == TierDirective.Mode.HIT) shard.used(directive);
        return bytes;
    }

    @Test
    void aRecordIsReadFromTheArtifactOnceAndThenFromMemory() throws Exception {
        store(10, 1, ReplacementPolicy.BANK_PARTITIONED);
        assertArrayEquals(this.fixture.record(2, 5), load(2, 5, 0));
        long afterFirst = this.store.bytesRead();
        assertEquals(this.fixture.banks[2].recordBytes(5), afterFirst);
        for (int i = 0; i < 4; i++) assertArrayEquals(this.fixture.record(2, 5), load(2, 5, 0));
        assertEquals(afterFirst, this.store.bytesRead(), "no further artifact reads");
        var stats = this.store.ramTier().stats();
        assertEquals(4, stats.hits()[2]);
        assertEquals(1, stats.misses()[2]);
        assertEquals(1, stats.residentExperts());
        assertEquals(
                4 * this.fixture.banks[2].recordBytes(5),
                this.store.ramCopyBytes() - this.fixture.banks[2].recordBytes(5));
        this.store.ramTier().checkInvariants();
    }

    @Test
    void aResidentTierPreloadsEverythingAndReadsNothingAfterwards() throws Exception {
        store(40, 3, ReplacementPolicy.BANK_PARTITIONED);
        RamTier tier = this.store.ramTier();
        assertTrue(tier.isResident());
        tier.preload(new FileRecordSource(this.fixture.file, this.fixture.banks), 4);
        assertEquals(40, tier.slotCount());
        assertEquals(40, tier.stats().residentExperts());
        assertTrue(tier.stats().preloadBytesPerSecond() > 0);
        long before = this.store.bytesRead();
        for (int bank = 0; bank < 5; bank++)
            for (int expert = 0; expert < this.fixture.banks[bank].expertCount(); expert++)
                assertArrayEquals(this.fixture.record(bank, expert), load(bank, expert, 0));
        assertEquals(before, this.store.bytesRead(), "inference reads no expert from the artifact");
        assertEquals(40, tier.stats().totalHits());
        assertEquals(0, tier.stats().totalMisses(), "the preload is not a miss");
        assertEquals(0, tier.stats().totalEvictions());
        tier.checkInvariants();
    }

    @Test
    void preloadRefusesATierThatCannotHoldEverything() throws Exception {
        store(20, 2, ReplacementPolicy.GLOBAL_LRU);
        assertFalse(this.store.ramTier().isResident());
        assertThrows(
                IllegalStateException.class,
                () -> this.store.ramTier().preload(new FileRecordSource(this.fixture.file, this.fixture.banks), 2));
    }

    @Test
    void aBoundedTierStaysWithinItsSlotsAndReturnsTheArtifactsBytes() throws Exception {
        for (ReplacementPolicy policy : ReplacementPolicy.values()) {
            if (this.store != null) this.store.close();
            store(12, 2, policy);
            SplittableRandom random = new SplittableRandom(7);
            for (int i = 0; i < 2000; i++) {
                int bank = random.nextInt(5);
                int expert = random.nextInt(this.fixture.banks[bank].expertCount());
                assertArrayEquals(this.fixture.record(bank, expert), load(bank, expert, 0));
                assertTrue(this.store.ramTier().stats().residentExperts() <= 12);
            }
            this.store.ramTier().checkInvariants();
            var stats = this.store.ramTier().stats();
            assertTrue(stats.totalEvictions() > 0, policy + " must evict 40 experts through 12 slots");
            assertEquals(2000, stats.totalHits() + stats.totalMisses() + stats.totalBypasses());
            assertEquals(0, stats.totalBypasses(), "with nothing pinned a slot is always found");
        }
    }

    /// The model visits its layers in a cycle and each layer reuses its own few experts. Ten slots cannot
    /// hold the 15 hot experts of the five banks: a global LRU evicts each one just before it is needed
    /// again, while a quota per bank keeps the experts of the banks that fit.
    @Test
    void aLayerAwarePolicyKeepsWhatACyclicPassWouldEvictFromAGlobalLru() throws Exception {
        double global = cyclicHitRate(ReplacementPolicy.GLOBAL_LRU);
        this.store.close();
        double partitioned = cyclicHitRate(ReplacementPolicy.BANK_PARTITIONED);
        System.out.printf("cyclic trace, 10 slots: global LRU %.3f, bank partitioned %.3f%n", global, partitioned);
        assertTrue(global < 0.05, "a global LRU thrashes on the cycle: " + global);
        assertTrue(partitioned > 0.3, "the partitioned policy keeps a share of every pass: " + partitioned);
    }

    private double cyclicHitRate(ReplacementPolicy policy) throws Exception {
        store(10, 1, policy);
        int[] hotPerBank = {3, 3, 3, 1, 3};
        for (int pass = 0; pass < 50; pass++)
            for (int bank = 0; bank < 5; bank++) for (int hot = 0; hot < hotPerBank[bank]; hot++) load(bank, hot, 0);
        this.store.ramTier().checkInvariants();
        return this.store.ramTier().stats().hitRate();
    }

    @Test
    void aPinnedSlotIsNeverTakenAndALoadWithoutASlotBypasses() throws Exception {
        store(1, 1, ReplacementPolicy.GLOBAL_LRU);
        RamTierShard shard = this.store.tier(0);
        assertArrayEquals(this.fixture.record(0, 0), load(0, 0, 0));

        // The record's slot is pinned by a hit in flight: a record that needs a slot finds none.
        TierDirective hit = new TierDirective();
        shard.plan(0, 0, hit);
        assertEquals(TierDirective.Mode.HIT, hit.mode());
        TierDirective other = new TierDirective();
        shard.plan(0, 1, other);
        assertEquals(TierDirective.Mode.BYPASS, other.mode());
        assertArrayEquals(this.fixture.record(0, 1), bytes(this.store.open(0, 1, 1, other)));
        assertTrue(shard.isResident(0, 0), "the pinned record stayed");
        assertEquals(1, shard.pinnedSlots());

        shard.used(hit);
        assertEquals(0, shard.pinnedSlots());
        assertArrayEquals(this.fixture.record(0, 1), load(0, 1, 0));
        assertFalse(shard.isResident(0, 0), "once unpinned it is the victim");
        assertTrue(shard.isResident(0, 1));
        shard.checkInvariants();
        assertEquals(1, this.store.ramTier().stats().totalEvictions());
    }

    @Test
    void aFillingSlotIsUnavailableUntilItsFillIsSettled() throws Exception {
        store(1, 1, ReplacementPolicy.GLOBAL_LRU);
        RamTierShard shard = this.store.tier(0);
        TierDirective fill = new TierDirective();
        shard.plan(0, 0, fill);
        assertEquals(TierDirective.Mode.FILL, fill.mode());
        TierDirective same = new TierDirective();
        shard.plan(0, 0, same);
        assertEquals(TierDirective.Mode.BYPASS, same.mode(), "the record is being filled: read around it");
        TierDirective other = new TierDirective();
        shard.plan(0, 1, other);
        assertEquals(TierDirective.Mode.BYPASS, other.mode());
        shard.filled(fill);
        shard.checkInvariants();
        assertTrue(shard.isResident(0, 0));
    }

    @Test
    void aFailedFillReturnsItsSlot() throws Exception {
        RecordSource failing = new RecordSource() {
            private final FileRecordSource real =
                    new FileRecordSource(RamTierTest.this.fixture.file, RamTierTest.this.fixture.banks);
            private int calls;

            @Override
            public void read(ExpertBank bank, int expert, MemorySegment destination)
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
        store(2, 1, ReplacementPolicy.GLOBAL_LRU, failing);
        assertThrows(IOException.class, () -> load(0, 0, 0));
        RamTierShard shard = this.store.tier(0);
        shard.checkInvariants();
        assertEquals(0, this.store.ramTier().stats().residentExperts());
        assertEquals(0, shard.pinnedSlots());
        assertArrayEquals(this.fixture.record(0, 0), load(0, 0, 0));
        assertEquals(1, this.store.ramTier().stats().residentExperts());
        shard.checkInvariants();
    }

    @Test
    void shardsAreIndependentAndEachRunsOnItsOwnThread() throws Exception {
        int shards = 3;
        store(24, shards, ReplacementPolicy.BANK_PARTITIONED);
        for (int shard = 0; shard < shards; shard++) {
            int mine = shard;
            assertTrue(
                    java.util.stream.IntStream.range(0, 40).anyMatch(key -> ExpertKeys.shardOf(key, shards) == mine),
                    "every shard owns a record of the fixture");
        }
        ExecutorService pool = Executors.newFixedThreadPool(shards);
        try {
            List<Future<?>> done = new ArrayList<>();
            for (int shard = 0; shard < shards; shard++) {
                int mine = shard;
                done.add(pool.submit(() -> {
                    SplittableRandom random = new SplittableRandom(mine);
                    for (int i = 0; i < 600; i++) {
                        int bank;
                        int expert;
                        do {
                            bank = random.nextInt(5);
                            expert = random.nextInt(this.fixture.banks[bank].expertCount());
                        } while (shardOf(bank, expert) != mine);
                        assertArrayEquals(this.fixture.record(bank, expert), load(bank, expert, mine));
                    }
                    return null;
                }));
            }
            for (Future<?> f : done) f.get(60, TimeUnit.SECONDS);
        } finally {
            pool.shutdownNow();
        }
        this.store.ramTier().checkInvariants();
        var stats = this.store.ramTier().stats();
        assertEquals(shards * 600, stats.totalHits() + stats.totalMisses() + stats.totalBypasses());
        assertTrue(stats.residentExperts() <= 24);
        assertEquals(24 * this.store.ramTier().slotBytes(), this.store.ramTier().capacityBytes());
    }

    private static byte[] bytes(HostRecord record) {
        return MemorySegment.ofAddress(record.hostAddress())
                .reinterpret(record.byteSize())
                .toArray(ValueLayout.JAVA_BYTE);
    }
}
