package io.euhedral_execution.inference.core.model.qwen4;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.Arrays;

/// Host-side grouping of a chunk's routed (row, expert) pairs by expert, and the description the expert
/// kernels and the combine read (docs/FLASH_NEXT_EXPERTS.md).
///
/// A chunk's router output is `topKIds`/`topKWeights`, `[rows][topK]` row-major, ids in `[0, experts]` (the id
/// `experts` is padding and is skipped, as upstream skips it) and BF16 weight bits. [#plan] groups the pairs by
/// expert; the experts that received pairs are the block's *active* experts, in ascending id order, and the
/// `i`-th of them is computed on its own: its work items are contiguous ([#itemStart], [#itemCount]) and name
/// slot `i` of the descriptor's slot table, which holds that expert's record address. A pair's position inside
/// an expert is ascending row, then ascending top-k position. The combine adds every row's pairs in ascending
/// expert order, so the sum is the same whichever experts finish first.
///
/// The descriptor is one block of host memory (little-endian, 16-byte aligned sections) that the device reads in
/// place after one copy:
///
/// | section | content | bytes |
/// |---|---|---|
/// | slots | device address of each active expert's record | 8 E |
/// | items | per work item: slot index, first pair, pair count (1..16), 0 | 16 I |
/// | pairs | per pair: row (int), BF16 routing weight (int, low 16 bits) | 8 P |
/// | row offsets | `rows + 1` offsets into the row lists | 4 (T + 1) |
/// | row pairs | for each row its pair indices, ascending expert order | 4 P |
///
/// E, I, P and T are the capacities: `min(experts, maxRows * topK)`, `P / 16 + E`, `maxRows * topK` and
/// `maxRows`. Everything is preallocated by the constructor; `plan` and `fill` allocate nothing.
public final class ExpertRouting {

    /// Pairs per work item: the token columns of two MMA tiles.
    public static final int ITEM_PAIRS = 16;

    private static final ValueLayout.OfInt INT = ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfLong LONG = ValueLayout.JAVA_LONG_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

    private final int experts;
    private final int topK;
    private final int maxRows;
    private final int maxPairs;
    private final int maxActive;

    private final int[] expertStart;
    private final int[] expertFill;
    private final int[] sortedRow;
    private final int[] sortedWeight;
    private final int[] active;
    private final int[] itemStart;
    private final int[] rowCursor;

    private final int slotsOffset;
    private final int itemsOffset;
    private final int pairsOffset;
    private final int rowOffsetsOffset;
    private final int rowPairsOffset;
    private final int descriptorBytes;

    private int rows;
    private int activeCount;

    /// @param experts routed experts per layer (512)
    /// @param topK experts per row (10)
    /// @param maxRows most rows of a chunk
    public ExpertRouting(int experts, int topK, int maxRows) {
        if (experts <= 0 || topK <= 0 || maxRows <= 0) throw new IllegalArgumentException("extents must be positive");
        this.experts = experts;
        this.topK = topK;
        this.maxRows = maxRows;
        this.maxPairs = Math.multiplyExact(maxRows, topK);
        this.maxActive = Math.min(experts, this.maxPairs);
        this.expertStart = new int[experts + 2];
        this.expertFill = new int[experts + 1];
        this.sortedRow = new int[this.maxPairs];
        this.sortedWeight = new int[this.maxPairs];
        this.active = new int[this.maxActive];
        this.itemStart = new int[this.maxActive + 1];
        this.rowCursor = new int[maxRows];
        int itemCapacity = this.maxPairs / ITEM_PAIRS + this.maxActive;
        int offset = 0;
        this.slotsOffset = offset;
        offset = align(offset + 8 * this.maxActive);
        this.itemsOffset = offset;
        offset = align(offset + 16 * itemCapacity);
        this.pairsOffset = offset;
        offset = align(offset + 8 * this.maxPairs);
        this.rowOffsetsOffset = offset;
        offset = align(offset + 4 * (maxRows + 1));
        this.rowPairsOffset = offset;
        this.descriptorBytes = align(offset + 4 * this.maxPairs);
    }

    private static int align(int bytes) {
        return (bytes + 15) & ~15;
    }

    /// Device scratch of a chunk: `act` `[pairs][inter]` and `weighted` `[pairs][hidden]`, both BF16, each
    /// 16-byte aligned (`act` first; see [ExpertOps.Scratch]).
    public static long scratchBytes(int maxPairs, int inter, int hidden) {
        return ExpertOps.Scratch.bytes(maxPairs, inter, hidden);
    }

    /// [#scratchBytes(int, int, int)] for Flash-Next's expert geometry (640 intermediate, 2560 hidden).
    public static long scratchBytes(int maxPairs) {
        return scratchBytes(maxPairs, 640, 2560);
    }

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

    public int maxRows() {
        return this.maxRows;
    }

    public int maxPairs() {
        return this.maxPairs;
    }

    /// The most experts a chunk can name: one per pair, at most every expert.
    public int maxActive() {
        return this.maxActive;
    }

    /// Groups the chunk's pairs by expert.
    public void plan(int rows, int[] topKIds, short[] topKWeights) {
        if (rows < 0 || rows > this.maxRows)
            throw new IllegalArgumentException("rows " + rows + " exceeds " + this.maxRows);
        int pairs = rows * this.topK;
        if (topKIds.length < pairs || topKWeights.length < pairs)
            throw new IllegalArgumentException("router output shorter than rows * topK");
        this.rows = rows;
        Arrays.fill(this.expertStart, 0);
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
        this.activeCount = 0;
        int items = 0;
        for (int e = 0; e < this.experts; e++) {
            int count = this.expertStart[e + 2] - this.expertStart[e + 1];
            if (count == 0) continue;
            this.itemStart[this.activeCount] = items;
            this.active[this.activeCount++] = e;
            items += (count + ITEM_PAIRS - 1) / ITEM_PAIRS;
        }
        this.itemStart[this.activeCount] = items;
    }

    public int rows() {
        return this.rows;
    }

    /// Experts that received at least one pair.
    public int activeExperts() {
        return this.activeCount;
    }

    /// The id of the `index`-th active expert (ascending).
    public int activeExpert(int index) {
        return this.active[checkActive(index)];
    }

    /// The first work item of the `index`-th active expert, and how many it has.
    public int itemStart(int index) {
        return this.itemStart[checkActive(index)];
    }

    public int itemCount(int index) {
        return this.itemStart[checkActive(index) + 1] - this.itemStart[index];
    }

    /// Pairs of the chunk.
    public int pairCount() {
        return this.expertStart[this.experts + 1];
    }

    /// Pairs of expert `expert` in this plan (0 for an expert that received none).
    public int expertPairCount(int expert) {
        return this.expertStart[expert + 2] - this.expertStart[expert + 1];
    }

    private int checkActive(int index) {
        if (index < 0 || index >= this.activeCount) throw new IndexOutOfBoundsException("active expert " + index);
        return index;
    }

    /// Writes the descriptor into the first [#descriptorBytes] bytes of `descriptor` (host memory, 16-byte
    /// aligned if it will be copied by DMA): everything but the slot addresses ([#setSlot]).
    public void fill(MemorySegment descriptor) {
        if (descriptor.byteSize() < this.descriptorBytes)
            throw new IllegalArgumentException("descriptor segment smaller than " + this.descriptorBytes);
        int itemIndex = 0;
        for (int i = 0; i < this.activeCount; i++) {
            int e = this.active[i];
            int begin = this.expertStart[e + 1];
            int count = this.expertStart[e + 2] - begin;
            for (int g = 0; g < count; g += ITEM_PAIRS) {
                long at = this.itemsOffset + 16L * itemIndex++;
                descriptor.set(INT, at, i);
                descriptor.set(INT, at + 4, begin + g);
                descriptor.set(INT, at + 8, Math.min(ITEM_PAIRS, count - g));
                descriptor.set(INT, at + 12, 0);
            }
        }
        int pairs = pairCount();
        for (int p = 0; p < pairs; p++) {
            long at = this.pairsOffset + 8L * p;
            descriptor.set(INT, at, this.sortedRow[p]);
            descriptor.set(INT, at + 4, this.sortedWeight[p]);
        }
        // Row lists: a counting sort of the pairs by row, ascending pair index (ascending expert) inside a row.
        Arrays.fill(this.rowCursor, 0, this.rows, 0);
        for (int p = 0; p < pairs; p++) this.rowCursor[this.sortedRow[p]]++;
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
            int row = this.sortedRow[p];
            descriptor.set(INT, this.rowPairsOffset + 4L * this.rowCursor[row]++, p);
        }
    }

    /// Writes the record address of the `index`-th active expert into a descriptor [#fill] wrote.
    public void setSlot(MemorySegment descriptor, int index, long address) {
        descriptor.set(LONG, this.slotsOffset + 8L * checkActive(index), address);
    }
}
