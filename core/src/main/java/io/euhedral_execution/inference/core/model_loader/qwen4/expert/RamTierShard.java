package io.euhedral_execution.inference.core.model_loader.qwen4.expert;

import java.util.Arrays;
import java.util.concurrent.atomic.AtomicIntegerArray;
import java.util.concurrent.atomic.AtomicLongArray;

/// One shard of the host tier: the bookkeeping for the slots of ordinary memory that hold the records of
/// the experts that hash to it, and nothing else. It has one owner, the serial source of the device cache
/// shard with the same index, and every method but the readers of the counters runs there: the directory,
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
public final class RamTierShard {
    private static final byte FREE = 0;
    private static final byte FILLING = 1;
    private static final byte READY = 2;
    private static final int NONE = -1;

    private final RamTier tier;
    private final ExpertKeys keys;
    private final int firstSlot;
    private final int slots;
    private final boolean partitioned;
    private final boolean resident;

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

    RamTierShard(RamTier tier, ExpertKeys keys, int index, int firstSlot, int slots, boolean partitioned) {
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
        if (partitioned && ownedTotal > 0)
            for (int bank = 0; bank < banks; bank++) this.quota[bank] = (int) ((long) slots * owned[bank] / ownedTotal);
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

    /// Decides what the load of `(bank, expert)` does with the tier, and writes it to `out`. A hit or a
    /// fill pins the slot until [#used], [#filled] or [#abandoned].
    public void plan(int bank, int expert, TierDirective out) {
        int key = this.keys.key(bank, expert);
        int slot = this.directory[key];
        if (slot != NONE) {
            if (this.state[slot] == READY) {
                this.pins[slot]++;
                if (!this.resident) touch(slot, bank);
                count(this.hits, bank);
                out.set(TierDirective.Mode.HIT, this.firstSlot + slot, this.tier.address(this.firstSlot + slot));
            } else {
                count(this.bypasses, bank);
                out.set(TierDirective.Mode.BYPASS, -1, 0);
            }
            return;
        }
        slot = this.resident ? NONE : take(bank);
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

    /// A free slot, or the slot of a victim chosen by the policy, or [#NONE] when every slot is filling or
    /// pinned.
    private int take(int bank) {
        if (this.freeTop > 0) return this.free[--this.freeTop];
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
        if (victim == NONE) return NONE;
        evict(victim);
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

    /// The global slot of a resident record.
    int residentSlot(int key) {
        return this.firstSlot + this.directory[key];
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

    /// Whether the record has a ready slot. Read it with the shard quiescent.
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
