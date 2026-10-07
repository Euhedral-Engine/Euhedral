package io.euhedral_execution.inference.core.qwen4;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.gpu.KernelArguments;
import io.euhedral_execution.inference.core.gpu.Qwen4Kernel;
import io.euhedral_execution.inference.core.model_loader.qwen4.expert.ExpertBank;

/// Typed launches of the routed-expert kernels (native/src/qwen4/experts.cuh, docs/FLASH_NEXT_EXPERTS.md): the
/// NVFP4 math of resident experts, one expert or several per launch. Every method queues on the GPU's selected stream.
///
/// The experts stay NVFP4 in their cache slots. Activations are BF16 for any number of rows, accumulation is FP32
/// on BF16 tensor cores, so the result for a (row, expert) pair is the same bits whatever else is in the launch.
public final class Qwen4ExpertOps {

    /// Row alignment the kernels need of every activation, scratch and output address.
    public static final int ALIGNMENT = 16;

    /// Output columns per CTA of the gate_up kernel and rows per CTA of the down kernel.
    private static final int GATE_UP_COLUMNS = 32;

    private static final int DOWN_ROWS = 128;
    private static final int COMBINE_THREADS = 256;

    private static final ThreadLocal<KernelArguments> ARGUMENTS = ThreadLocal.withInitial(KernelArguments::new);

    private Qwen4ExpertOps() {}

    private static KernelArguments arguments() {
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

    /// The device scratch of a chunk, one block: `act` `[pairs][inter]` at offset 0, then `weighted` `[pairs][hidden]`
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

    /// act of every pair of the given work items: gate and up of the pair's row against the item's expert, SwiGLU.
    ///
    /// @param slots device address of the slot table (8 bytes per expert record address)
    /// @param items device address of the work items (16 bytes each), `itemCount` of them
    /// @param pairs device address of the pairs (row, weight)
    /// @param x activations `[rows][hidden]` BF16
    /// @param act output `[pairs][inter]` BF16
    public static void gateUpSwiGlu(
            ExecutionGpu gpu, Geometry geometry, long slots, long items, int itemCount, long pairs, long x, long act) {
        if (itemCount <= 0) throw new IllegalArgumentException("an expert has at least one work item");
        requireAligned(slots, items, pairs, x, act);
        gpu.launchTableKernel(
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

    /// weighted of every pair of the given work items: the down projection of the pair's act row times its routing
    /// weight.
    public static void downWeighted(
            ExecutionGpu gpu,
            Geometry geometry,
            long slots,
            long items,
            int itemCount,
            long pairs,
            long act,
            long weighted) {
        if (itemCount <= 0) throw new IllegalArgumentException("an expert has at least one work item");
        requireAligned(slots, items, pairs, act, weighted);
        gpu.launchTableKernel(
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

    /// Adds every row's weighted outputs to its row of `out`, one BF16 addition at a time, in the
    /// order of the row's pair list (the entries of `rowPairs` from `rowOffsets[row]` up to `rowOffsets[row + 1]`).
    /// With `zeroFirst` the sums start at zero and rows without pairs are zeroed; without it they add to `out` and
    /// a row without pairs is not touched.
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
        gpu.launchTableKernel(
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

    /// The `index`-th active expert of `routing` on its own: gate_up and SwiGLU, then down and the routing
    /// weights, over its work items only, into its pairs' rows of `scratch` (see [Scratch], laid out for the
    /// chunk's pairs). `descriptor` is the device copy of the descriptor [Qwen4ExpertRouting#fill] wrote,
    /// with this expert's slot address set; `x` is `[rows][hidden]` BF16 of the chunk. Experts are independent:
    /// they may run in any order, on any streams. The expert's lease may be released (with a fence recorded after
    /// this call) once it returns.
    public static void runExpert(
            ExecutionGpu gpu,
            Geometry geometry,
            Qwen4ExpertRouting routing,
            int index,
            long descriptor,
            long x,
            long scratch) {
        long act = scratch;
        long weighted = scratch + Scratch.weightedOffset(routing.pairCount(), geometry.inter());
        long slots = descriptor + routing.slotsOffset();
        long items = descriptor + routing.itemsOffset() + 16L * routing.itemStart(index);
        long pairs = descriptor + routing.pairsOffset();
        int count = routing.itemCount(index);
        gateUpSwiGlu(gpu, geometry, slots, items, count, pairs, x, act);
        downWeighted(gpu, geometry, slots, items, count, pairs, act, weighted);
    }

    /// The ordered accumulation of the chunk: every row's weighted outputs added to its row of `out` in ascending
    /// expert order, after every expert ran ([#runExpert]); rows without pairs are zeroed.
    public static void combineExperts(
            ExecutionGpu gpu, Geometry geometry, Qwen4ExpertRouting routing, long descriptor, long scratch, long out) {
        combine(
                gpu,
                geometry.hidden(),
                scratch + Scratch.weightedOffset(routing.pairCount(), geometry.inter()),
                descriptor + routing.rowOffsetsOffset(),
                descriptor + routing.rowPairsOffset(),
                out,
                routing.rows(),
                true);
    }

    private static void requireAligned(long... addresses) {
        for (long address : addresses) {
            if (address == 0 || address % ALIGNMENT != 0)
                throw new IllegalArgumentException("address " + address + " is not " + ALIGNMENT + "-byte aligned");
        }
    }
}
