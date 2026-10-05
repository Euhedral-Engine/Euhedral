package io.euhedral_execution.inference.core.qwen4;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

/// Host-side planning of the routed experts of one chunk of rows: which experts run in which wave, and the work
/// description the expert kernels of a wave read (docs/FLASH_NEXT_EXPERTS.md).
///
/// A chunk's router output is `topKIds`/`topKWeights`, `[rows][topK]` row-major, ids in `[0, experts]` (the id
/// `experts` is padding and is skipped, as upstream skips it) and BF16 weight bits. [#plan] groups the (row,
/// weight) pairs by expert; the experts that received pairs, in ascending id order, are cut into waves of at most
/// `maxWaveExperts` experts and `maxWavePairs` pairs. The caller acquires the experts of a wave from the cache,
/// [#fill]s a descriptor, writes the slot addresses ([#setSlot]), copies the descriptor to the device and runs
/// [Qwen4ExpertOps#runWave]; waves are run in order, which is what makes each token's additions ascend in expert
/// id across waves.
///
/// A pair's position inside an expert is ascending row, then ascending top-k position. Everything is preallocated
/// by the constructor; `plan` and `fill` allocate nothing.
///
/// The descriptor is one block of host memory (little-endian, 16-byte aligned sections) that the device reads in
/// place after one copy:
///
/// | section | content | bytes |
/// |---|---|---|
/// | slots | device address of each expert record of the wave, written by the caller | 8 E |
/// | items | per work item: slot index, first pair, pair count (1..8), 0 | 16 I |
/// | pairs | per pair: row (int), BF16 routing weight (int, low 16 bits) | 8 P |
/// | row offsets | `rows + 1` offsets into the row lists | 4 (T + 1) |
/// | row pairs | for each row the wave's pair indices, ascending expert order | 4 P |
///
/// E, I, P and T are the wave capacities: `maxWaveExperts`, `maxWavePairs / 8 + maxWaveExperts`, `maxWavePairs` and
/// `maxRows`.
public final class Qwen4ExpertWave {

    /// Pairs per work item: the token columns of one MMA.
    public static final int ITEM_PAIRS = 8;

    private static final ValueLayout.OfInt INT = ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfLong LONG = ValueLayout.JAVA_LONG_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

    private final int experts;
    private final int topK;
    private final int maxRows;
    private final int maxWaveExperts;
    private final int maxWavePairs;

    private final int[] expertStart;
    private final int[] expertFill;
    private final int[] sortedRow;
    private final int[] sortedWeight;
    private final int[] active;
    private final int[] waveFirst;
    private final int[] rowCursor;

    private final int slotsOffset;
    private final int itemsOffset;
    private final int pairsOffset;
    private final int rowOffsetsOffset;
    private final int rowPairsOffset;
    private final int descriptorBytes;
    private final int itemCapacity;

    private int rows;
    private int activeCount;
    private int waveCount;

    /// @param experts routed experts per layer (512)
    /// @param topK experts per row (10)
    /// @param maxRows most rows of a chunk
    /// @param maxWaveExperts most experts of a wave (the expert cache's slots the wave may hold at once)
    /// @param maxWavePairs most pairs of a wave; at least `maxRows`, so that any expert fits a wave
    public Qwen4ExpertWave(int experts, int topK, int maxRows, int maxWaveExperts, int maxWavePairs) {
        if (experts <= 0 || topK <= 0 || maxRows <= 0 || maxWaveExperts <= 0)
            throw new IllegalArgumentException("extents must be positive");
        if (maxWavePairs < maxRows)
            throw new IllegalArgumentException("a wave must hold at least one expert's pairs: " + maxWavePairs);
        this.experts = experts;
        this.topK = topK;
        this.maxRows = maxRows;
        this.maxWaveExperts = maxWaveExperts;
        this.maxWavePairs = maxWavePairs;
        int totalPairs = Math.multiplyExact(maxRows, topK);
        this.expertStart = new int[experts + 2];
        this.expertFill = new int[experts + 1];
        this.sortedRow = new int[totalPairs];
        this.sortedWeight = new int[totalPairs];
        this.active = new int[Math.min(experts, totalPairs)];
        this.waveFirst = new int[this.active.length + 1];
        this.rowCursor = new int[maxRows];
        this.itemCapacity = maxWavePairs / ITEM_PAIRS + maxWaveExperts;
        int offset = 0;
        this.slotsOffset = offset;
        offset = align(offset + 8 * maxWaveExperts);
        this.itemsOffset = offset;
        offset = align(offset + 16 * this.itemCapacity);
        this.pairsOffset = offset;
        offset = align(offset + 8 * maxWavePairs);
        this.rowOffsetsOffset = offset;
        offset = align(offset + 4 * (maxRows + 1));
        this.rowPairsOffset = offset;
        this.descriptorBytes = align(offset + 4 * maxWavePairs);
    }

    private static int align(int bytes) {
        return (bytes + 15) & ~15;
    }

    /// Device scratch of a wave of at most `maxPairs` pairs: `act` `[pairs][inter]` and `weighted` `[pairs][hidden]`,
    /// both BF16, each 16-byte aligned (`act` first; see [Qwen4ExpertOps.Scratch]).
    public static long scratchBytes(int maxPairs, int inter, int hidden) {
        return Qwen4ExpertOps.Scratch.bytes(maxPairs, inter, hidden);
    }

    /// [#scratchBytes(int, int, int)] for Flash-Next's expert geometry (640 intermediate, 2560 hidden).
    public static long scratchBytes(int maxPairs) {
        return scratchBytes(maxPairs, 640, 2560);
    }

    /// Bytes of the descriptor [#fill] writes (host and device buffers of this size).
    public int descriptorBytes() {
        return this.descriptorBytes;
    }

    public int slotsOffset() {
        return this.slotsOffset;
    }

    public int itemsOffset() {
        return this.itemsOffset;
    }

    public int pairsOffset() {
        return this.pairsOffset;
    }

    public int rowOffsetsOffset() {
        return this.rowOffsetsOffset;
    }

    public int rowPairsOffset() {
        return this.rowPairsOffset;
    }

    public int maxWaveExperts() {
        return this.maxWaveExperts;
    }

    public int maxWavePairs() {
        return this.maxWavePairs;
    }

    public int maxRows() {
        return this.maxRows;
    }

    /// Groups the chunk's pairs by expert and cuts the experts that have pairs into waves.
    public void plan(int rows, int[] topKIds, short[] topKWeights) {
        if (rows < 0 || rows > this.maxRows)
            throw new IllegalArgumentException("rows " + rows + " exceeds " + this.maxRows);
        int pairs = rows * this.topK;
        if (topKIds.length < pairs || topKWeights.length < pairs)
            throw new IllegalArgumentException("router output shorter than rows * topK");
        this.rows = rows;
        java.util.Arrays.fill(this.expertStart, 0);
        for (int i = 0; i < pairs; i++) {
            int id = topKIds[i];
            if (id < 0 || id > this.experts) throw new IllegalArgumentException("expert id " + id + " out of range");
            if (id < this.experts) this.expertStart[id + 2]++;
        }
        // expertStart[e + 1] is the first pair of expert e after the prefix sum; expertFill walks it.
        for (int e = 0; e < this.experts; e++) this.expertStart[e + 2] += this.expertStart[e + 1];
        for (int e = 0; e < this.experts; e++) this.expertFill[e] = this.expertStart[e + 1];
        for (int i = 0; i < pairs; i++) {
            int id = topKIds[i];
            if (id == this.experts) continue;
            int at = this.expertFill[id]++;
            this.sortedRow[at] = i / this.topK;
            this.sortedWeight[at] = topKWeights[i] & 0xffff;
        }
        // Waves: consecutive active experts until the expert or the pair limit is reached.
        this.activeCount = 0;
        this.waveCount = 0;
        int wavePairs = 0;
        int waveExperts = 0;
        for (int e = 0; e < this.experts; e++) {
            int count = this.expertStart[e + 2] - this.expertStart[e + 1];
            if (count == 0) continue;
            if (count > this.maxWavePairs)
                throw new IllegalArgumentException("expert " + e + " has " + count + " pairs, more than a wave holds");
            if (waveExperts == this.maxWaveExperts || wavePairs + count > this.maxWavePairs) {
                waveExperts = 0;
                wavePairs = 0;
            }
            if (waveExperts == 0) this.waveFirst[this.waveCount++] = this.activeCount;
            this.active[this.activeCount++] = e;
            waveExperts++;
            wavePairs += count;
        }
        this.waveFirst[this.waveCount] = this.activeCount;
    }

    public int rows() {
        return this.rows;
    }

    /// Experts that received at least one pair.
    public int activeExperts() {
        return this.activeCount;
    }

    public int waveCount() {
        return this.waveCount;
    }

    /// Experts of wave `wave`.
    public int waveExpertCount(int wave) {
        return this.waveFirst[checkWave(wave) + 1] - this.waveFirst[wave];
    }

    /// The id of the `index`-th expert (ascending) of wave `wave`: slot `index` of the descriptor.
    public int waveExpert(int wave, int index) {
        if (index < 0 || index >= waveExpertCount(wave)) throw new IndexOutOfBoundsException(index);
        return this.active[this.waveFirst[wave] + index];
    }

    /// Pairs of wave `wave`.
    public int wavePairCount(int wave) {
        int first = this.waveFirst[checkWave(wave)];
        int last = this.waveFirst[wave + 1];
        return this.expertStart[this.active[last - 1] + 2] - this.expertStart[this.active[first] + 1];
    }

    /// Work items of wave `wave`: the grid height of the expert kernels.
    public int waveItemCount(int wave) {
        int items = 0;
        for (int i = this.waveFirst[checkWave(wave)]; i < this.waveFirst[wave + 1]; i++) {
            int e = this.active[i];
            items += (this.expertStart[e + 2] - this.expertStart[e + 1] + ITEM_PAIRS - 1) / ITEM_PAIRS;
        }
        return items;
    }

    /// Pairs of expert `expert` in this plan (0 for an expert that received none).
    public int expertPairCount(int expert) {
        return this.expertStart[expert + 2] - this.expertStart[expert + 1];
    }

    private int checkWave(int wave) {
        if (wave < 0 || wave >= this.waveCount) throw new IndexOutOfBoundsException("wave " + wave);
        return wave;
    }

    /// Writes the descriptor of wave `wave` into the first [#descriptorBytes] bytes of `descriptor` (host memory,
    /// 16-byte aligned if it will be copied by DMA). Everything but the slot addresses; see [#setSlot].
    public void fill(int wave, MemorySegment descriptor) {
        checkWave(wave);
        if (descriptor.byteSize() < this.descriptorBytes)
            throw new IllegalArgumentException("descriptor segment smaller than " + this.descriptorBytes);
        int first = this.waveFirst[wave];
        int last = this.waveFirst[wave + 1];
        int base = this.expertStart[this.active[first] + 1];
        int itemIndex = 0;
        for (int i = first; i < last; i++) {
            int e = this.active[i];
            int begin = this.expertStart[e + 1] - base;
            int count = this.expertStart[e + 2] - this.expertStart[e + 1];
            for (int g = 0; g < count; g += ITEM_PAIRS) {
                long at = this.itemsOffset + 16L * itemIndex++;
                descriptor.set(INT, at, i - first);
                descriptor.set(INT, at + 4, begin + g);
                descriptor.set(INT, at + 8, Math.min(ITEM_PAIRS, count - g));
                descriptor.set(INT, at + 12, 0);
            }
        }
        int pairs = this.expertStart[this.active[last - 1] + 2] - base;
        for (int p = 0; p < pairs; p++) {
            long at = this.pairsOffset + 8L * p;
            descriptor.set(INT, at, this.sortedRow[base + p]);
            descriptor.set(INT, at + 4, this.sortedWeight[base + p]);
        }
        // Row lists: a counting sort of the wave's pairs by row, ascending pair index (ascending expert) inside a row.
        java.util.Arrays.fill(this.rowCursor, 0, this.rows, 0);
        for (int p = 0; p < pairs; p++) this.rowCursor[this.sortedRow[base + p]]++;
        int running = 0;
        for (int r = 0; r < this.rows; r++) {
            descriptor.set(INT, this.rowOffsetsOffset + 4L * r, running);
            int count = this.rowCursor[r];
            this.rowCursor[r] = running;
            running += count;
        }
        descriptor.set(INT, this.rowOffsetsOffset + 4L * this.rows, running);
        for (int r = this.rows + 1; r <= this.maxRows; r++)
            descriptor.set(INT, this.rowOffsetsOffset + 4L * r, running);
        for (int p = 0; p < pairs; p++) {
            int row = this.sortedRow[base + p];
            descriptor.set(INT, this.rowPairsOffset + 4L * this.rowCursor[row]++, p);
        }
    }

    /// Writes the device address of slot `index`'s expert record into a descriptor [#fill] wrote.
    public void setSlot(MemorySegment descriptor, int index, long address) {
        if (index < 0 || index >= this.maxWaveExperts) throw new IndexOutOfBoundsException(index);
        descriptor.set(LONG, this.slotsOffset + 8L * index, address);
    }
}
