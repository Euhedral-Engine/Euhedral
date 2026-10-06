package io.euhedral_execution.inference.core.model_loader.qwen4.expert;

import java.util.Arrays;
import java.util.concurrent.atomic.AtomicIntegerArray;
import java.util.concurrent.atomic.AtomicLongArray;

/// One shard of the host tier: the bookkeeping for the slots of ordinary memory that hold the records of
/// the experts that hash to it, and nothing else. It has one owner, the owner of the device cache shard
/// with the same index, which confines it, and every method but the readers of the counters runs there: the directory,
/// the slot states, the recency lists and the pins are plain arrays.
///
/// A slot is free, filling or ready. A load asks [#plan] what to do: copy a ready record out of its slot
/// (a hit, which pins the slot until [#used]), read the artifact into a slot reserved for it (a fill,
/// which pins the slot until [#filled] or [#abandoned]), or read the artifact into the staging slot alone
/// when no slot may be taken (a bypass). A slot that is filling or pinned is on no recency list, or is
/// skipped by the victim search, so it is never taken while a load uses it. The load itself, the copy or
/// the read, runs elsewhere and touches only the slot's memory; the shard's state changes only here.
///
/// ## Replacement
///
/// The model visits layer 0 to the last and starts again, so the reuse distance of an expert is a whole
/// pass and a global LRU evicts exactly what the next pass needs. [ReplacementPolicy#BANK_PARTITIONED]
/// gives each layer (bank) a quota of the shard's slots in proportion to its experts. A layer under its
/// quota that needs a slot takes it from the layer furthest over its own quota, and a layer at its quota
/// replaces its own least recently used record, so a layer cannot flush the others. Slots no layer is
/// using are borrowed freely. [ReplacementPolicy#GLOBAL_LRU] is the baseline.
///
/// ## Admission
///
/// With admission, a record that finds no free slot replaces the victim only if it was asked for more often
/// than the victim was. A prefill visits nearly every expert of a layer once per chunk, so recency alone
/// replaces each record just before the next chunk needs it, and the tier serves almost nothing; with
/// admission the records already in the tier stay, and a share of every chunk is served from memory. A record
/// that loses reads the artifact into staging (a bypass). The counts are halved every [#AGE_PER_SLOT]
/// requests per slot, so a record that was hot long ago does not hold its slot forever.
public final class RamTierShard {
    private static final byte FREE = 0;
    private static final byte FILLING = 1;
    private static final byte READY = 2;
    private static final int NONE = -1;
    /// Requests per slot between halvings of the request counts.
    static final int AGE_PER_SLOT = 10;

    private final RamTier tier;
    private final ExpertKeys keys;
    private final int firstSlot;
    private final int slots;
    private final boolean partitioned;
    private final boolean resident;
    private final boolean admission;
    /// [ReplacementPolicy#FREQUENCY]: the victim is the least requested of a sample of ready slots.
    private final boolean frequency;
    /// Slots sampled for a victim under [ReplacementPolicy#FREQUENCY].
    static final int SAMPLE = 32;
    /// Each slot's latest plan, by [#clock], under [ReplacementPolicy#FREQUENCY].
    private final long[] lastUse;
    private long clock;
    private final java.util.SplittableRandom random = new java.util.SplittableRandom(0x5eed);
    /// Requests of each record since the counts were last halved, when admission is on.
    private final int[] requests;
    private int sinceAging;

    private final int[] directory;
    private final int[] slotKey;
    private final byte[] state;
    private final int[] pins;
    private final int[] prev;
    private final int[] next;
    private final int[] head;
    private final int[] tail;
    private final int[] used;
    private final int[] quota;
    private final int[] free;
    private int freeTop;

    private final AtomicLongArray hits;
    private final AtomicLongArray misses;
    private final AtomicLongArray evictions;
    private final AtomicLongArray bypasses;
    private final AtomicIntegerArray readyPerBank;

    RamTierShard(
            RamTier tier,
            ExpertKeys keys,
            int index,
            int firstSlot,
            int slots,
            boolean partitioned,
            boolean frequency,
            boolean admission) {
        this.tier = tier;
        this.keys = keys;
        this.firstSlot = firstSlot;
        this.slots = slots;
        this.partitioned = partitioned;
        int banks = keys.bankCount();
        int lists = partitioned ? banks : 1;
        this.directory = new int[keys.keyCount()];
        Arrays.fill(this.directory, NONE);
        this.slotKey = new int[slots];
        this.state = new byte[slots];
        this.pins = new int[slots];
        this.prev = new int[slots];
        this.next = new int[slots];
        this.head = new int[lists];
        this.tail = new int[lists];
        Arrays.fill(this.head, NONE);
        Arrays.fill(this.tail, NONE);
        this.used = new int[lists];
        this.quota = new int[lists];
        this.free = new int[slots];
        for (int slot = slots - 1; slot >= 0; slot--) this.free[this.freeTop++] = slot;
        this.hits = new AtomicLongArray(banks);
        this.misses = new AtomicLongArray(banks);
        this.evictions = new AtomicLongArray(banks);
        this.bypasses = new AtomicLongArray(banks);
        this.readyPerBank = new AtomicIntegerArray(banks);

        int[] owned = new int[banks];
        int ownedTotal = 0;
        for (int key = 0; key < keys.keyCount(); key++) {
            if (ExpertKeys.shardOf(key, tier.shards()) != index) continue;
            owned[keys.bankOf(key)]++;
            ownedTotal++;
        }
        this.resident = slots >= ownedTotal;
        this.admission = admission && !this.resident;
        this.frequency = frequency && !this.resident;
        this.requests = this.admission || this.frequency ? new int[keys.keyCount()] : null;
        this.lastUse = this.frequency ? new long[slots] : null;
        if (partitioned && ownedTotal > 0)
            for (int bank = 0; bank < banks; bank++) this.quota[bank] = (int) ((long) slots * owned[bank] / ownedTotal);
    }

    /// Whether the tier is pinned host memory, which the device's copies read in place.
    public boolean pinned() {
        return this.tier.pinned();
    }

    /// Whether every record of this shard has a slot: nothing is ever evicted or filled.
    public boolean isResident() {
        return this.resident;
    }

    public int slotCount() {
        return this.slots;
    }

    private int list(int bank) {
        return this.partitioned ? bank : 0;
    }

    // ---------------------------------------------------------------- planning a load

    /// Decides what the load of `(bank, expert)` does with the tier, and writes it to `out`, for a load a prefill
    /// chunk asked for. A hit or a fill pins the slot until [#used], [#filled] or [#abandoned].
    public void plan(int bank, int expert, TierDirective out) {
        plan(bank, expert, true, out);
    }

    /// As above, for a load a prefill chunk (`scan`) or a decode step asked for: under
    /// [ReplacementPolicy#FREQUENCY] only a scan's records are subject to admission.
    public void plan(int bank, int expert, boolean scan, TierDirective out) {
        int key = this.keys.key(bank, expert);
        int asked = this.requests != null ? request(key) : 0;
        boolean admit = this.admission && (scan || !this.frequency);
        this.clock++;
        int slot = this.directory[key];
        if (slot != NONE) {
            if (this.state[slot] == READY) {
                this.pins[slot]++;
                if (!this.resident) touch(slot, bank);
                if (this.frequency) this.lastUse[slot] = this.clock;
                count(this.hits, bank);
                out.set(TierDirective.Mode.HIT, this.firstSlot + slot, this.tier.address(this.firstSlot + slot));
            } else {
                count(this.bypasses, bank);
                out.set(TierDirective.Mode.BYPASS, -1, 0);
            }
            return;
        }
        slot = this.resident ? NONE : take(bank, asked, admit);
        if (slot == NONE) {
            count(this.bypasses, bank);
            out.set(TierDirective.Mode.BYPASS, -1, 0);
            return;
        }
        count(this.misses, bank);
        this.state[slot] = FILLING;
        this.pins[slot] = 1;
        this.slotKey[slot] = key;
        this.directory[key] = slot;
        this.used[list(bank)]++;
        if (this.frequency) this.lastUse[slot] = this.clock;
        out.set(TierDirective.Mode.FILL, this.firstSlot + slot, this.tier.address(this.firstSlot + slot));
    }

    /// The load that planned a hit is done with the slot.
    public void used(TierDirective directive) {
        int slot = directive.slot() - this.firstSlot;
        this.pins[slot]--;
    }

    /// The fill that planned `directive` read the record into its slot: the record is there for the next
    /// load.
    public void filled(TierDirective directive) {
        int slot = directive.slot() - this.firstSlot;
        int bank = this.keys.bankOf(this.slotKey[slot]);
        this.state[slot] = READY;
        this.pins[slot] = 0;
        link(slot, bank);
        this.readyPerBank.lazySet(bank, this.readyPerBank.get(bank) + 1);
    }

    /// The fill failed: the slot holds nothing usable and goes back to the free slots.
    public void abandoned(TierDirective directive) {
        int slot = directive.slot() - this.firstSlot;
        int key = this.slotKey[slot];
        this.directory[key] = NONE;
        this.used[list(this.keys.bankOf(key))]--;
        this.state[slot] = FREE;
        this.pins[slot] = 0;
        this.free[this.freeTop++] = slot;
    }

    // ---------------------------------------------------------------- slots

    /// Counts a request of `key` and returns how often it was asked for before it.
    private int request(int key) {
        int before = this.requests[key];
        if (before < Integer.MAX_VALUE) this.requests[key] = before + 1;
        if (++this.sinceAging >= AGE_PER_SLOT * this.slots) {
            this.sinceAging = 0;
            for (int k = 0; k < this.requests.length; k++) this.requests[k] >>= 1;
        }
        return before;
    }

    /// A free slot, or the slot of a victim chosen by the policy, or [#NONE] when every slot is filling or
    /// pinned, or (with admission) when the victim was asked for at least as often as the record was before
    /// this request (`asked`).
    private int take(int bank, int asked, boolean admit) {
        if (this.freeTop > 0) return this.free[--this.freeTop];
        int victim = this.frequency ? leastRequested() : leastRecent(bank);
        if (victim == NONE) return NONE;
        if (admit && this.requests[this.slotKey[victim]] >= asked) return NONE;
        evict(victim);
        return victim;
    }

    /// The least requested of [#SAMPLE] ready, unpinned slots drawn at random, the least recently planned among
    /// equals; when the sample finds none, the first such slot; [#NONE] when every slot is filling or pinned.
    private int leastRequested() {
        int victim = NONE;
        for (int i = 0; i < SAMPLE; i++) {
            int slot = this.random.nextInt(this.slots);
            if (this.state[slot] != READY || this.pins[slot] != 0) continue;
            if (victim == NONE || before(slot, victim)) victim = slot;
        }
        if (victim != NONE) return victim;
        for (int slot = 0; slot < this.slots; slot++)
            if (this.state[slot] == READY && this.pins[slot] == 0) return slot;
        return NONE;
    }

    private boolean before(int slot, int other) {
        int a = this.requests[this.slotKey[slot]];
        int b = this.requests[this.slotKey[other]];
        return a < b || (a == b && this.lastUse[slot] < this.lastUse[other]);
    }

    /// The victim the recency policies choose, or [#NONE].
    private int leastRecent(int bank) {
        int own = list(bank);
        int victim = NONE;
        if (this.used[own] > 0 && this.used[own] >= this.quota[own]) victim = unpinned(own);
        if (victim == NONE) {
            int bestOver = Integer.MIN_VALUE;
            for (int list = 0; list < this.head.length; list++) {
                int over = this.used[list] - this.quota[list];
                if (over <= bestOver) continue;
                int candidate = unpinned(list);
                if (candidate == NONE) continue;
                bestOver = over;
                victim = candidate;
            }
        }
        return victim;
    }

    private int unpinned(int list) {
        for (int slot = this.head[list]; slot != NONE; slot = this.next[slot]) if (this.pins[slot] == 0) return slot;
        return NONE;
    }

    private void evict(int slot) {
        int key = this.slotKey[slot];
        int bank = this.keys.bankOf(key);
        unlink(slot, bank);
        this.directory[key] = NONE;
        this.used[list(bank)]--;
        this.state[slot] = FREE;
        count(this.evictions, bank);
        this.readyPerBank.lazySet(bank, this.readyPerBank.get(bank) - 1);
    }

    private void touch(int slot, int bank) {
        unlink(slot, bank);
        link(slot, bank);
    }

    private void link(int slot, int bank) {
        int list = list(bank);
        this.prev[slot] = this.tail[list];
        this.next[slot] = NONE;
        if (this.tail[list] != NONE) this.next[this.tail[list]] = slot;
        else this.head[list] = slot;
        this.tail[list] = slot;
    }

    private void unlink(int slot, int bank) {
        int list = list(bank);
        int before = this.prev[slot];
        int after = this.next[slot];
        if (before != NONE) this.next[before] = after;
        else this.head[list] = after;
        if (after != NONE) this.prev[after] = before;
        else this.tail[list] = before;
    }

    private static void count(AtomicLongArray counters, int bank) {
        counters.lazySet(bank, counters.get(bank) + 1);
    }

    // ---------------------------------------------------------------- resident tier

    /// Gives every record of this shard its slot, ready, in key order. Before any load; the records are
    /// read into the slots by the preload.
    void assignResident(int index) {
        if (!this.resident) throw new IllegalStateException("the shard has fewer slots than records");
        int slot = 0;
        for (int key = 0; key < this.keys.keyCount(); key++) {
            if (ExpertKeys.shardOf(key, this.tier.shards()) != index) continue;
            int bank = this.keys.bankOf(key);
            this.directory[key] = slot;
            this.slotKey[slot] = key;
            this.state[slot] = READY;
            this.used[list(bank)]++;
            this.readyPerBank.lazySet(bank, this.readyPerBank.get(bank) + 1);
            slot++;
        }
        this.freeTop = 0;
        for (int free = slot; free < this.slots; free++) this.free[this.freeTop++] = free;
    }

    /// Gives each bank its share of this shard's slots (its quota, in proportion to its experts), filled by its
    /// lowest experts in consecutive slots, ready, at the cold end of the recency order. Before any load, for a
    /// bounded tier; the startup fill reads the records into the slots. They hold their slots only until a request
    /// needs one: with admission each counts as asked for less than never (-1), so the first request of any record
    /// replaces it as it would take a free slot, and the tier adapts as an empty one would.
    void assignShare(int index) {
        if (this.resident) return;
        int banks = this.keys.bankCount();
        int[] owned = new int[banks];
        int ownedTotal = 0;
        for (int key = 0; key < this.keys.keyCount(); key++) {
            if (ExpertKeys.shardOf(key, this.tier.shards()) != index) continue;
            owned[this.keys.bankOf(key)]++;
            ownedTotal++;
        }
        int[] taken = new int[banks];
        int slot = 0;
        for (int key = 0; key < this.keys.keyCount() && ownedTotal > 0; key++) {
            if (ExpertKeys.shardOf(key, this.tier.shards()) != index) continue;
            int bank = this.keys.bankOf(key);
            if (taken[bank] >= (int) ((long) this.slots * owned[bank] / ownedTotal)) continue;
            taken[bank]++;
            this.directory[key] = slot;
            this.slotKey[slot] = key;
            this.state[slot] = READY;
            this.pins[slot] = 0;
            this.used[list(bank)]++;
            if (this.requests != null) this.requests[key] = -1;
            link(slot, bank);
            this.readyPerBank.lazySet(bank, this.readyPerBank.get(bank) + 1);
            slot++;
        }
        this.freeTop = 0;
        for (int free = this.slots - 1; free >= slot; free--) this.free[this.freeTop++] = free;
    }

    /// The global slot of `key`'s record, or -1 when it has none. Read it on the owner, or quiescent.
    int slotOf(int key) {
        int slot = this.directory[key];
        return slot == NONE ? -1 : this.firstSlot + slot;
    }

    // ---------------------------------------------------------------- inspection (quiescent)

    /// Checks the bookkeeping against itself: every directory entry names a slot that names it back, the
    /// recency lists hold exactly the ready slots, and the counts agree. Read it with the shard quiescent.
    public void checkInvariants() {
        int mapped = 0;
        for (int key = 0; key < this.directory.length; key++) {
            int slot = this.directory[key];
            if (slot == NONE) continue;
            mapped++;
            if (this.slotKey[slot] != key) throw new IllegalStateException("slot " + slot + " does not hold " + key);
            if (this.state[slot] == FREE) throw new IllegalStateException("free slot " + slot + " is mapped");
        }
        int[] listed = new int[this.head.length];
        int linked = 0;
        for (int list = 0; list < this.head.length; list++) {
            int before = NONE;
            for (int slot = this.head[list]; slot != NONE; slot = this.next[slot]) {
                if (this.prev[slot] != before) throw new IllegalStateException("slot " + slot + " has a broken link");
                if (this.state[slot] != READY) throw new IllegalStateException("slot " + slot + " is listed unready");
                before = slot;
                listed[list]++;
                linked++;
            }
            if (before != this.tail[list]) throw new IllegalStateException("the tail of list " + list);
        }
        int ready = 0;
        int filling = 0;
        for (int slot = 0; slot < this.slots; slot++) {
            if (this.state[slot] == READY) ready++;
            if (this.state[slot] == FILLING) filling++;
            if (this.pins[slot] < 0) throw new IllegalStateException("slot " + slot + " is unpinned too often");
        }
        if (!this.resident && linked != ready)
            throw new IllegalStateException("listed " + linked + " of " + ready + " ready");
        if (mapped != ready + filling) throw new IllegalStateException("the directory holds " + mapped);
        if (this.freeTop != this.slots - ready - filling) throw new IllegalStateException("free slots " + this.freeTop);
        int total = 0;
        for (int list = 0; list < this.used.length; list++) total += this.used[list];
        if (total != ready + filling) throw new IllegalStateException("used " + total);
    }

    /// Slots filling or pinned. Zero when no load is in flight. Read it with the shard quiescent.
    public int pinnedSlots() {
        int count = 0;
        for (int slot = 0; slot < this.slots; slot++) if (this.state[slot] == FILLING || this.pins[slot] > 0) count++;
        return count;
    }

    /// Whether the record has a ready slot. Read it on the owner, or with the shard quiescent.
    public boolean isResident(int bank, int expert) {
        int slot = this.directory[this.keys.key(bank, expert)];
        return slot != NONE && this.state[slot] == READY;
    }

    void addTo(long[] h, long[] m, long[] e, long[] b, int[] ready) {
        for (int bank = 0; bank < h.length; bank++) {
            h[bank] += this.hits.get(bank);
            m[bank] += this.misses.get(bank);
            e[bank] += this.evictions.get(bank);
            b[bank] += this.bypasses.get(bank);
            ready[bank] += this.readyPerBank.get(bank);
        }
    }
}
