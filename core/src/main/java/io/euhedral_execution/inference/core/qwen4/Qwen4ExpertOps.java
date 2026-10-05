package io.euhedral_execution.inference.core.qwen4;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.gpu.Qwen4Kernel;
import io.euhedral_execution.inference.core.gpu.Qwen4KernelArguments;
import io.euhedral_execution.inference.core.model_loader.qwen4.expert.ExpertBank;

/// Typed launches of the routed-expert kernels (native/src/qwen4/experts.cuh, docs/FLASH_NEXT_EXPERTS.md): the
/// grouped NVFP4 math of a wave of resident experts. Every method queues on the GPU's selected stream.
///
/// The experts stay NVFP4 in their cache slots. Activations are BF16 for any number of rows, accumulation is FP32
/// on BF16 tensor cores, so the result for a (row, expert) pair is the same bits whatever else is in the wave.
public final class Qwen4ExpertOps {

    /// Row alignment the kernels need of every activation, scratch and output address.
    public static final int ALIGNMENT = 16;

    /// Output columns per CTA of the gate_up kernel and rows per CTA of the down kernel.
    private static final int GATE_UP_COLUMNS = 32;

    private static final int DOWN_ROWS = 128;
    private static final int COMBINE_THREADS = 256;

    private static final ThreadLocal<Qwen4KernelArguments> ARGUMENTS =
            ThreadLocal.withInitial(Qwen4KernelArguments::new);

    private Qwen4ExpertOps() {}

    private static Qwen4KernelArguments arguments() {
        return ARGUMENTS.get().clear();
    }

    /// Shapes and record offsets of an expert record: `gate_up` `[2 * inter][hidden]` at `gateUpOffset`, `down`
    /// `[hidden][inter]` at `downOffset`, both complete NVFP4 tensors (row-split-k128-v1).
    public record Geometry(int hidden, int inter, long gateUpOffset, long downOffset) {
        public Geometry {
            if (hidden <= 0 || inter <= 0) throw new IllegalArgumentException("extents must be positive");
            if (hidden % 128 != 0 || inter % 128 != 0)
                throw new IllegalArgumentException(
                        "hidden and inter must be multiples of 128, got " + hidden + ", " + inter);
            if (inter % GATE_UP_COLUMNS != 0 || hidden % DOWN_ROWS != 0)
                throw new IllegalArgumentException(
                        "inter must be a multiple of " + GATE_UP_COLUMNS + " and hidden of " + DOWN_ROWS);
            if (gateUpOffset < 0 || downOffset < 0 || gateUpOffset % 256 != 0 || downOffset % 256 != 0)
                throw new IllegalArgumentException("tensor offsets must be 256-byte aligned");
        }

        /// Flash-Next's experts: 2560 hidden, 640 intermediate, `gate_up` at 0 and `down` at 1,843,456.
        public static Geometry flashNext() {
            return new Geometry(2560, 640, 0, 1_843_456);
        }

        /// The geometry of a bank's records, checked against its projections.
        public static Geometry of(ExpertBank bank) {
            var gateUp = bank.projection("gate_up");
            var down = bank.projection("down");
            if (gateUp.shape().length != 2 || down.shape().length != 2)
                throw new IllegalArgumentException(bank.name() + ": expert projections must be matrices");
            int inter = Math.toIntExact(down.shape()[1]);
            int hidden = Math.toIntExact(down.shape()[0]);
            if (gateUp.shape()[0] != 2L * inter || gateUp.shape()[1] != hidden)
                throw new IllegalArgumentException(bank.name() + ": gate_up is not [2 * inter][hidden]");
            return new Geometry(hidden, inter, gateUp.recordOffset(), down.recordOffset());
        }
    }

    /// The device scratch of a wave, one block: `act` `[pairs][inter]` at offset 0, then `weighted` `[pairs][hidden]`
    /// at 16-byte alignment, both BF16.
    public static final class Scratch {
        private Scratch() {}

        public static long actBytes(int pairs, int inter) {
            return align(2L * pairs * inter);
        }

        public static long weightedOffset(int pairs, int inter) {
            return actBytes(pairs, inter);
        }

        public static long bytes(int maxPairs, int inter, int hidden) {
            return actBytes(maxPairs, inter) + align(2L * maxPairs * hidden);
        }

        private static long align(long bytes) {
            return (bytes + ALIGNMENT - 1) & -(long) ALIGNMENT;
        }
    }

    /// act of every pair of the wave: gate and up of the pair's row against the item's expert, SwiGLU.
    ///
    /// @param slots device address of the wave's slot table (8 bytes per expert record address)
    /// @param items device address of the work items (16 bytes each), `itemCount` of them
    /// @param pairs device address of the pairs (row, weight)
    /// @param x activations `[rows][hidden]` BF16
    /// @param act output `[pairs][inter]` BF16
    public static void gateUpSwiGlu(
            ExecutionGpu gpu, Geometry geometry, long slots, long items, int itemCount, long pairs, long x, long act) {
        if (itemCount <= 0) throw new IllegalArgumentException("a wave has at least one work item");
        requireAligned(slots, items, pairs, x, act);
        gpu.launchQwen4(
                Qwen4Kernel.EXPERT_GATE_UP_SWIGLU_BF16,
                geometry.inter() / GATE_UP_COLUMNS,
                itemCount,
                1,
                128,
                1,
                1,
                0,
                arguments()
                        .pointer(slots)
                        .pointer(items)
                        .pointer(pairs)
                        .pointer(x)
                        .pointer(act)
                        .int32(Math.toIntExact(geometry.gateUpOffset()))
                        .int32(geometry.hidden())
                        .int32(geometry.inter()));
    }

    /// weighted of every pair of the wave: the down projection of the pair's act row times its routing weight.
    public static void downWeighted(
            ExecutionGpu gpu,
            Geometry geometry,
            long slots,
            long items,
            int itemCount,
            long pairs,
            long act,
            long weighted) {
        if (itemCount <= 0) throw new IllegalArgumentException("a wave has at least one work item");
        requireAligned(slots, items, pairs, act, weighted);
        gpu.launchQwen4(
                Qwen4Kernel.EXPERT_DOWN_BF16,
                geometry.hidden() / DOWN_ROWS,
                itemCount,
                1,
                128,
                1,
                1,
                0,
                arguments()
                        .pointer(slots)
                        .pointer(items)
                        .pointer(pairs)
                        .pointer(act)
                        .pointer(weighted)
                        .int32(Math.toIntExact(geometry.downOffset()))
                        .int32(geometry.inter())
                        .int32(geometry.hidden()));
    }

    /// Adds every row's weighted outputs of the wave to its row of `out`, one BF16 addition at a time, in the
    /// order of the row's pair list (the entries of `rowPairs` from `rowOffsets[row]` up to `rowOffsets[row + 1]`).
    /// With `zeroFirst` the sums start at zero and rows without pairs are zeroed: the first wave of a layer.
    /// Without it a row without pairs is not touched.
    public static void combine(
            ExecutionGpu gpu,
            int hidden,
            long weighted,
            long rowOffsets,
            long rowPairs,
            long out,
            int rows,
            boolean zeroFirst) {
        if (rows <= 0 || hidden <= 0 || hidden % 8 != 0)
            throw new IllegalArgumentException("combine needs positive rows and hidden divisible by 8");
        requireAligned(weighted, rowOffsets, rowPairs, out);
        int vectors = hidden / 8;
        gpu.launchQwen4(
                Qwen4Kernel.EXPERT_COMBINE_BF16,
                (vectors + COMBINE_THREADS - 1) / COMBINE_THREADS,
                rows,
                1,
                COMBINE_THREADS,
                1,
                1,
                0,
                arguments()
                        .pointer(weighted)
                        .pointer(rowOffsets)
                        .pointer(rowPairs)
                        .pointer(out)
                        .int32(hidden)
                        .int32(zeroFirst ? 1 : 0));
    }

    /// Runs wave `wave` of `plan`: gate_up and SwiGLU, down and routing weights, then the accumulation into `out`.
    /// `descriptor` is the device copy of the descriptor [Qwen4ExpertWave#fill] wrote, with the slot addresses set;
    /// `scratch` is [Qwen4ExpertWave#scratchBytes] bytes (see [Scratch]); `x` and `out` are `[rows][hidden]` BF16
    /// of the chunk. Waves of a chunk must be run in order, the first with `firstWave`, on one stream; the experts'
    /// leases may be released (with a fence recorded after this call) once it returns.
    public static void runWave(
            ExecutionGpu gpu,
            Geometry geometry,
            Qwen4ExpertWave plan,
            int wave,
            long descriptor,
            long x,
            long scratch,
            long out,
            boolean firstWave) {
        int items = plan.waveItemCount(wave);
        int pairs = plan.wavePairCount(wave);
        long act = scratch;
        long weighted = scratch + Scratch.weightedOffset(pairs, geometry.inter());
        long slots = descriptor + plan.slotsOffset();
        long itemTable = descriptor + plan.itemsOffset();
        long pairTable = descriptor + plan.pairsOffset();
        gateUpSwiGlu(gpu, geometry, slots, itemTable, items, pairTable, x, act);
        downWeighted(gpu, geometry, slots, itemTable, items, pairTable, act, weighted);
        combine(
                gpu,
                geometry.hidden(),
                weighted,
                descriptor + plan.rowOffsetsOffset(),
                descriptor + plan.rowPairsOffset(),
                out,
                plan.rows(),
                firstWave);
    }

    private static void requireAligned(long... addresses) {
        for (long address : addresses) {
            if (address == 0 || address % ALIGNMENT != 0)
                throw new IllegalArgumentException("address " + address + " is not " + ALIGNMENT + "-byte aligned");
        }
    }
}
