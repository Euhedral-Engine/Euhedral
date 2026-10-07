package io.euhedral_execution.inference.core.model.qwen4.expert;

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

    private FileExpertStore admitting(int slots, ReplacementPolicy policy) throws IOException {
        RamTier tier = new RamTier(this.fixture.banks, slots, 1, policy, null, true);
        this.store = new FileExpertStore(
                this.gpu, new FileRecordSource(this.fixture.file, this.fixture.banks), tier, this.fixture.banks, 4);
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

    /// A bounded tier starts with each layer's share of its slots filled by the layer's lowest experts, so the
    /// first pass finds them in memory.
    @Test
    void aBoundedTierPreloadsEachLayersShare() throws Exception {
        for (int shards : new int[] {1, 2}) {
            if (this.store != null) this.store.close();
            store(20, shards, ReplacementPolicy.BANK_PARTITIONED);
            RamTier tier = this.store.ramTier();
            assertFalse(tier.isResident());
            tier.preload(new FileRecordSource(this.fixture.file, this.fixture.banks), 3);
            tier.checkInvariants();
            var stats = tier.stats();
            int[] experts = {8, 3, 16, 1, 12};
            int preloaded = stats.residentExperts();
            assertTrue(preloaded <= 20 && preloaded >= 20 - 5 * shards, "the shares fill the tier: " + preloaded);
            for (int bank = 0; bank < 5; bank++)
                assertTrue(
                        Math.abs(stats.residentPerBank()[bank] - 20.0 * experts[bank] / 40) <= shards,
                        "bank " + bank + " has its share: " + stats.residentPerBank()[bank]);
            long before = this.store.bytesRead();
            for (int bank = 0; bank < 5; bank++)
                for (int expert = 0; expert < experts[bank]; expert++)
                    if (this.store.tier(shardOf(bank, expert)).isResident(bank, expert))
                        assertArrayEquals(this.fixture.record(bank, expert), load(bank, expert, 0));
            assertEquals(before, this.store.bytesRead(), "a preloaded record is not read again");
            assertEquals(preloaded, tier.stats().totalHits());
            tier.checkInvariants();
        }
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

    /// A prefill visits every expert of a layer once per chunk. Eight slots cannot hold the 16 experts of bank 2:
    /// recency replaces each record just before the next sweep needs it, while admission keeps the records the
    /// tier has and serves half of every sweep.
    @Test
    void admissionKeepsAShareOfARepeatedSweepThatRecencyThrashes() throws Exception {
        double recency = sweepHitRate(false);
        this.store.close();
        double admitted = sweepHitRate(true);
        System.out.printf("sweeps of 16 records through 8 slots: recency %.3f, admission %.3f%n", recency, admitted);
        assertTrue(recency < 0.05, "recency thrashes on the sweep: " + recency);
        assertTrue(admitted > 0.4, "admission keeps half of every sweep: " + admitted);
    }

    private double sweepHitRate(boolean admission) throws Exception {
        if (admission) admitting(8, ReplacementPolicy.GLOBAL_LRU);
        else store(8, 1, ReplacementPolicy.GLOBAL_LRU);
        for (int sweep = 0; sweep < 20; sweep++)
            for (int expert = 0; expert < 16; expert++)
                assertArrayEquals(this.fixture.record(2, expert), load(2, expert, 0));
        this.store.ramTier().checkInvariants();
        return this.store.ramTier().stats().hitRate();
    }

    /// A preloaded record holds its slot only until a request needs it: with admission, the first request of a
    /// record the tier lacks replaces one, as it would take a free slot; a preloaded record that was asked for is
    /// an ordinary record from then on.
    @Test
    void aPreloadedRecordGivesWayToTheFirstRequestOfAnother() throws Exception {
        admitting(8, ReplacementPolicy.GLOBAL_LRU);
        RamTier tier = this.store.ramTier();
        tier.preload(new FileRecordSource(this.fixture.file, this.fixture.banks), 2);
        RamTierShard shard = this.store.tier(0);
        assertTrue(shard.isResident(2, 0));
        // The shares round down: two slots are free and take the first two records.
        assertEquals(6, tier.stats().residentExperts());
        load(2, 14, 0);
        load(2, 13, 0);
        assertEquals(0, tier.stats().totalEvictions());
        assertFalse(shard.isResident(2, 15));
        assertArrayEquals(this.fixture.record(2, 15), load(2, 15, 0));
        assertTrue(shard.isResident(2, 15), "the first request replaced a preloaded record");
        assertEquals(1, tier.stats().totalEvictions());
        load(2, 0, 0);
        assertArrayEquals(this.fixture.record(4, 11), load(4, 11, 0));
        assertTrue(shard.isResident(2, 0), "a preloaded record that was asked for is not a placeholder any more");
        tier.checkInvariants();
    }

    private int planned(RamTierShard shard, int bank, int expert, boolean scan) {
        TierDirective directive = new TierDirective();
        shard.plan(bank, expert, scan, directive);
        TierDirective.Mode mode = directive.mode();
        if (mode == TierDirective.Mode.FILL) shard.filled(directive);
        else if (mode == TierDirective.Mode.HIT) shard.used(directive);
        return mode.ordinal();
    }

    /// Under the frequency policy the victim is the least requested record; a decode step's record always enters,
    /// and a prefill's only in place of a record requested less often.
    @Test
    void theFrequencyPolicyEvictsTheLeastRequestedAndAdmitsOnlyAPrefillsRecords() throws Exception {
        RamTier tier = new RamTier(this.fixture.banks, 2, 1, ReplacementPolicy.FREQUENCY, null, true);
        this.store = new FileExpertStore(
                this.gpu, new FileRecordSource(this.fixture.file, this.fixture.banks), tier, this.fixture.banks, 4);
        RamTierShard shard = this.store.tier(0);
        for (int i = 0; i < 3; i++) planned(shard, 2, 0, false);
        planned(shard, 2, 1, false);
        assertTrue(shard.isResident(2, 0) && shard.isResident(2, 1));
        // A prefill's first request of a record: the victim (2, 1) was requested once, more than never.
        planned(shard, 2, 2, true);
        assertFalse(shard.isResident(2, 2), "a prefill's record does not replace one requested more often");
        // A decode step's record enters, in place of the least requested record.
        planned(shard, 2, 3, false);
        assertTrue(shard.isResident(2, 3));
        assertTrue(shard.isResident(2, 0), "the most requested record stays");
        assertFalse(shard.isResident(2, 1));
        tier.checkInvariants();
    }

    /// A record asked for more often than the victim replaces it; until then it is read around the tier.
    @Test
    void aRecordAskedForMoreOftenThanTheVictimReplacesIt() throws Exception {
        admitting(1, ReplacementPolicy.GLOBAL_LRU);
        RamTierShard shard = this.store.tier(0);
        load(0, 0, 0);
        assertTrue(shard.isResident(0, 0), "a free slot takes the first record");
        load(0, 1, 0);
        load(0, 1, 0);
        assertTrue(shard.isResident(0, 0), "asked for once before, as often as the victim: read around the tier");
        assertFalse(shard.isResident(0, 1));
        assertArrayEquals(this.fixture.record(0, 1), load(0, 1, 0));
        assertTrue(shard.isResident(0, 1), "asked for twice before, more often than the victim: it replaces it");
        var stats = this.store.ramTier().stats();
        assertEquals(2, stats.totalBypasses());
        assertEquals(1, stats.totalEvictions());
        shard.checkInvariants();
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
