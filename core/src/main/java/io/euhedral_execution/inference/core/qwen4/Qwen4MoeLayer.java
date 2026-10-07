package io.euhedral_execution.inference.core.qwen4;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.gpu.GpuStream;
import io.euhedral_execution.inference.core.model_loader.qwen4.expert.ExpertLease;
import io.euhedral_execution.inference.core.model_loader.qwen4.expert.StreamFence;
import io.euhedral_execution.inference.core.scheduling.graph.LanePool;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.concurrent.atomic.AtomicLongArray;
import java.util.concurrent.atomic.LongAdder;

/// The sparse MoE block of a Flash-Next layer (Qwen4ExpTextSparseMoeBlock): a router chooses ten of
/// 512 experts per token, a shared expert runs for every token, and
///
/// ```
/// out = routed experts (weighted, summed in ascending expert order) + sigmoid(shared gate) * shared expert
/// ```
///
/// The routed experts are not on the device: they pass through the [ExpertCache]. One instance is
/// the block resources of one stage graph (its route readback, its descriptor, the leases a block
/// holds); the stages of the graph use it:
///
/// 1. [#submitRouting] queues the router and the copy of its choice to the host; the plan stage follows
///    it across a device-completion edge, and the shared expert ([#submitShared]) is a side branch;
/// 2. [#plan] (on the host, once the routing is readable) groups the (token, expert) pairs by expert
///    ([Qwen4ExpertRouting]) and [#submitPlan] copies the block's description to the device;
/// 3. every active expert is computed on its own once it is held ([#hold], [#submitExpert]): its kernels
///    wait on the device for its copy's marker, and its lease closes behind a marker recorded after them,
///    so a lease is held only while its kernels are being submitted. Experts are independent of one
///    another and may run in any order, on any lanes;
/// 4. [#submitFinish] adds every row's expert outputs in ascending expert order (the order the sum needs,
///    whatever order the experts ran in) and combines them with the shared expert's gated output.
///
/// No device pointer into a slot outlives its lease: the kernels read the slot address from the
/// descriptor while the lease is held, and the fence orders their completion before the slot's next use.
public final class Qwen4MoeLayer implements AutoCloseable {

    /// The block's weights: the BF16 router and shared-expert gate, the NVFP4 shared expert, and (null when the
    /// shared expert's width suits the native prefill kernels as it is) its down projection with its input width
    /// padded to [#paddedWidth], zero columns, for prefill.
    public record Weights(
            Qwen4Weight router,
            Qwen4Weight sharedGateProj,
            Qwen4Weight sharedUpProj,
            Qwen4Weight sharedDownProj,
            Qwen4Weight sharedExpertGate,
            Qwen4Weight sharedDownPadded) {}

    /// The shared expert's width as a prefill's down projection reads it: a multiple of 256, as the native NVFP4
    /// kernels need for more than 64 rows (their weight scales are fetched 256 values at a time). The added columns
    /// are zeros in the activation and in the padded weights, so every product they add is 0.
    public static int paddedWidth(int width) {
        return io.euhedral_execution.inference.core.model_loader.qwen4.Qwen4ResidencyPlanner.paddedSharedWidth(width);
    }

    /// The device buffers of one block ([#scratchBytes]).
    public record Scratch(
            long logits,
            long ids,
            long routeWeights,
            long sharedGate,
            long sharedUp,
            long sharedAct,
            long shared,
            long gateRaw,
            long routed,
            long experts) {}

    private static final ValueLayout.OfInt INT = ValueLayout.JAVA_INT_UNALIGNED;
    private static final ValueLayout.OfShort SHORT = ValueLayout.JAVA_SHORT_UNALIGNED;

    private final ExecutionGpu gpu;
    private final Qwen4ExpertOps.Geometry geometry;
    private final int hidden;
    private final int experts;
    private final int topK;
    private final int sharedInter;
    private final int maxRows;
    private final Qwen4ExpertRouting routing;
    private final ExecutionGpu.ReadbackBuffer routeIds;
    // Per layer, a later layer's router applied to the layer's input, for one row (allocated on first use): the
    // prediction a prefetch reads ahead by, and that a demand recording reports. One buffer per layer, as the
    // predictions of several layers may be in flight at once.
    private final java.util.Map<Integer, ExecutionGpu.ReadbackBuffer> predictedLogits =
            new java.util.concurrent.ConcurrentHashMap<>();
    private final java.util.Map<Integer, Long> predictedDevice = new java.util.concurrent.ConcurrentHashMap<>();
    private final ExecutionGpu.ReadbackBuffer routeWeights;
    private final ExecutionGpu.UploadBuffer hostDescriptor;
    private final long deviceDescriptor;
    private final int[] ids;
    private final short[] weights;
    /// One fence marker per lane, re-recorded after every expert's kernels on that lane: waiting for it waits for
    /// its latest recording, so a slot's fence never holds more than one marker per lane.
    private final AtomicLongArray laneFences = new AtomicLongArray(LanePool.MAX_LANES);
    private final GpuStream[] fenceStreams = new GpuStream[LanePool.MAX_LANES];

    // The leases the block holds, by active expert: stored by the expert's fetch, closed by its expert stage, or
    // by an abandoned block.
    private final ExpertLease[] held;
    private int bank;
    private int lastUnique;
    private final Metrics metrics;
    private boolean closed;

    /// Counters that every graph of a plan adds to.
    public static final class Metrics {
        final LongAdder routeWaitNanos = new LongAdder();
        final LongAdder expertWaitNanos = new LongAdder();
        final LongAdder blocks = new LongAdder();
        final LongAdder leased = new LongAdder();

        public Counters counters() {
            return new Counters(
                    this.routeWaitNanos.sum(), this.expertWaitNanos.sum(), this.blocks.sum(), this.leased.sum());
        }
    }

    /// Host time between the router's copy being queued and the plan stage reading it, host time from the
    /// plan to the combine (the block's experts being fetched and computed), and the blocks run and
    /// experts leased since the counters were built.
    public record Counters(long routeWaitNanos, long expertWaitNanos, long blocks, long leased) {}

    /// @param maxRows the most rows of a chunk
    public Qwen4MoeLayer(
            ExecutionGpu gpu,
            Qwen4ExpertOps.Geometry geometry,
            int experts,
            int topK,
            int sharedInter,
            int maxRows,
            Metrics metrics) {
        this.gpu = gpu;
        this.geometry = geometry;
        this.hidden = geometry.hidden();
        this.experts = experts;
        this.topK = topK;
        this.sharedInter = sharedInter;
        this.maxRows = maxRows;
        this.metrics = metrics;
        this.routing = new Qwen4ExpertRouting(experts, topK, maxRows);
        int maxPairs = this.routing.maxPairs();
        this.ids = new int[maxPairs];
        this.weights = new short[maxPairs];
        this.held = new ExpertLease[this.routing.maxActive()];
        this.routeIds = gpu.allocateReadbackBuffer(4L * maxPairs);
        this.routeWeights = gpu.allocateReadbackBuffer(2L * maxPairs);
        ExecutionGpu.UploadBuffer host = null;
        long device = 0;
        try {
            host = gpu.allocateUploadBuffer(this.routing.descriptorBytes());
            device = gpu.allocate(this.routing.descriptorBytes());
        } catch (Throwable failure) {
            if (host != null) host.close();
            this.routeIds.close();
            this.routeWeights.close();
            throw failure;
        }
        this.hostDescriptor = host;
        this.deviceDescriptor = device;
    }

    /// The most experts a block can name: the expert stages of a layer.
    public static int maxExperts(int experts, int maxRows, int topK) {
        return Math.min(experts, Math.multiplyExact(maxRows, topK));
    }

    /// Device bytes of the scratch for `rows` rows.
    public long scratchBytes(int rows) {
        long bf16 = Short.BYTES;
        long maxPairs = (long) rows * this.topK;
        return align(rows * (long) this.experts * bf16) // logits
                + align(maxPairs * 4) // ids
                + align(maxPairs * bf16) // route weights
                + 2 * align(rows * (long) this.sharedInter * bf16) // shared gate, up
                + align(rows * (long) paddedWidth(this.sharedInter) * bf16) // shared act, padded for prefill
                + 2 * align(rows * (long) this.hidden * bf16) // shared, routed
                + align(rows * bf16) // shared expert gate
                + align(Qwen4ExpertRouting.scratchBytes((int) maxPairs, this.geometry.inter(), this.hidden));
    }

    public Scratch scratch(long base, int rows) {
        long bf16 = Short.BYTES;
        long maxPairs = (long) rows * this.topK;
        long logits = base;
        long idsAddress = logits + align(rows * (long) this.experts * bf16);
        long routeWeightsAddress = idsAddress + align(maxPairs * 4);
        long sharedGate = routeWeightsAddress + align(maxPairs * bf16);
        long sharedUp = sharedGate + align(rows * (long) this.sharedInter * bf16);
        long sharedAct = sharedUp + align(rows * (long) this.sharedInter * bf16);
        long shared = sharedAct + align(rows * (long) paddedWidth(this.sharedInter) * bf16);
        long routed = shared + align(rows * (long) this.hidden * bf16);
        long gateRaw = routed + align(rows * (long) this.hidden * bf16);
        long expertScratch = gateRaw + align(rows * bf16);
        return new Scratch(
                logits,
                idsAddress,
                routeWeightsAddress,
                sharedGate,
                sharedUp,
                sharedAct,
                shared,
                gateRaw,
                routed,
                expertScratch);
    }

    private static long align(long bytes) {
        return (bytes + 255) & -256L;
    }

    // ---------------------------------------------------------------- the pieces of a block

    /// Queues the router (logits, softmax, top ten, renormalized weights) and the copy of its
    /// choice to the host, on the layer's stream. The caller arms a retirement boundary after this,
    /// then calls [#submitShared].
    public void submitRouting(Weights weights, long input, int rows, Scratch scratch) {
        if (rows <= 0 || rows > this.maxRows) throw new IllegalArgumentException("rows " + rows);
        Qwen4Ops.linearBf16(
                this.gpu, input, weights.router().address(), scratch.logits(), rows, this.hidden, this.experts);
        Qwen4MoeOps.router(
                this.gpu, scratch.logits(), scratch.ids(), scratch.routeWeights(), rows, this.experts, this.topK);
        this.gpu.copyDeviceToReadback(this.routeIds, scratch.ids(), 4L * rows * this.topK);
        this.gpu.copyDeviceToReadback(this.routeWeights, scratch.routeWeights(), 2L * rows * this.topK);
    }

    /// Queues the shared expert and its gate behind the routing, on the layer's stream.
    public void submitShared(Weights weights, long input, int rows, Scratch scratch) {
        sharedExpert(weights, input, rows, scratch);
    }

    /// Rows above which a block is a prefill: the decode kernels take 1 to 8 rows.
    private static final int PREFILL_ROWS = 8;

    /// The shared expert's down projection at its own width.
    private void sharedDown(Weights weights, int rows, Scratch scratch) {
        Qwen4Weight down = weights.sharedDownProj();
        this.gpu.linearNvfp4Bf16(
                scratch.sharedAct(),
                down.address(),
                scratch.shared(),
                rows,
                this.sharedInter,
                this.hidden,
                down.bytes());
    }

    /// The shared expert and its gate: the SwiGLU MLP of every row, and the gate's raw projection.
    private void sharedExpert(Weights weights, long input, int rows, Scratch scratch) {
        Qwen4Weight gate = weights.sharedGateProj();
        this.gpu.linearNvfp4Bf16(
                input, gate.address(), scratch.sharedGate(), rows, this.hidden, this.sharedInter, gate.bytes());
        Qwen4Weight up = weights.sharedUpProj();
        this.gpu.linearNvfp4Bf16(
                input, up.address(), scratch.sharedUp(), rows, this.hidden, this.sharedInter, up.bytes());
        Qwen4Weight padded = weights.sharedDownPadded();
        if (rows > PREFILL_ROWS && padded != null) {
            // A prefill: the native kernels take the padded width, where the unpadded one would run the scalar
            // reference (more than 64 rows need input widths in multiples of 256).
            int width = paddedWidth(this.sharedInter);
            Qwen4MoeOps.swiGluPadded(
                    this.gpu,
                    scratch.sharedGate(),
                    scratch.sharedUp(),
                    scratch.sharedAct(),
                    rows,
                    this.sharedInter,
                    width);
            this.gpu.linearNvfp4Bf16(
                    scratch.sharedAct(), padded.address(), scratch.shared(), rows, width, this.hidden, padded.bytes());
        } else {
            Qwen4MoeOps.swiGlu(
                    this.gpu, scratch.sharedGate(), scratch.sharedUp(), scratch.sharedAct(), rows * this.sharedInter);
            sharedDown(weights, rows, scratch);
        }
        Qwen4Ops.linearBf16(
                this.gpu, input, weights.sharedExpertGate().address(), scratch.gateRaw(), rows, this.hidden, 1);
    }

    /// Reads the routing the device copied (readable once the boundary armed after [#submitRouting]
    /// retired), groups the block's pairs by expert, for the layer whose experts are bank `bank`, and writes
    /// the block's description. Returns the active experts.
    public int plan(int bank, int rows) {
        int pairs = rows * this.topK;
        MemorySegment.copy(this.routeIds.segment(), INT, 0, this.ids, 0, pairs);
        MemorySegment.copy(this.routeWeights.segment(), SHORT, 0, this.weights, 0, pairs);
        this.routing.plan(rows, this.ids, this.weights);
        this.routing.fill(this.hostDescriptor.segment());
        this.lastUnique = this.routing.activeExperts();
        this.bank = bank;
        this.metrics.blocks.increment();
        return this.lastUnique;
    }

    /// Queues the copy of the block's description to the device, on the lane the plan stage chose.
    public void submitPlan() {
        this.gpu.copyUploadToDevice(this.deviceDescriptor, this.hostDescriptor);
    }

    /// Experts of the planned block.
    public int activeExperts() {
        return this.routing.activeExperts();
    }

    /// The id of the `index`-th active expert of the planned block.
    public int activeExpert(int index) {
        return this.routing.activeExpert(index);
    }

    public int bank() {
        return this.bank;
    }

    /// The most experts a block can name.
    public int maxExperts() {
        return this.held.length;
    }

    /// Stores the lease of the `index`-th active expert, for its expert stage.
    public void hold(int index, ExpertLease lease) {
        this.held[index] = lease;
    }

    /// Adds host time spent waiting: for the routing, and from the plan to the combine.
    public void chargeRouteWait(long nanos) {
        this.metrics.routeWaitNanos.add(nanos);
    }

    public void chargeExpertWait(long nanos) {
        this.metrics.expertWaitNanos.add(nanos);
    }

    /// Queues the kernels of the `index`-th active expert, whose lease is held, on `stream` (lane `lane` of
    /// the graph's pool), and closes its lease behind a fence on that lane. The lease is closed even when this
    /// throws.
    public void submitExpert(int index, GpuStream stream, int lane, long input, Scratch scratch) {
        ExpertLease lease = this.held[index];
        this.held[index] = null;
        if (lease == null) throw new IllegalStateException("expert " + index + " of the block is not held");
        StreamFence fence = null;
        try {
            // The expert's record address, into its entry of the description the device reads.
            MemorySegment host = this.hostDescriptor.segment();
            this.routing.setSlot(host, index, lease.deviceAddress());
            long entry = this.routing.slotsOffset() + 8L * index;
            this.gpu.copyHostWeightsToDevice(this.deviceDescriptor + entry, host.address() + entry, 8);
            // An expert whose copy was only submitted is waited for here, on the device.
            if (lease.readyMarker() != 0) stream.await(lease.readyMarker());
            Qwen4ExpertOps.runExpert(
                    this.gpu, this.geometry, this.routing, index, this.deviceDescriptor, input, scratch.experts());
            long marker = laneFence(stream, lane);
            stream.mark(marker);
            fence = new StreamFence(stream, marker);
            this.metrics.leased.increment();
        } finally {
            if (fence != null) lease.close(fence);
            else lease.close();
        }
    }

    /// The fence marker of `lane`, opened on first use.
    private long laneFence(GpuStream stream, int lane) {
        long marker = this.laneFences.get(lane);
        if (marker != 0) return marker;
        long opened = stream.openMarker();
        if (this.laneFences.compareAndSet(lane, 0, opened)) {
            this.fenceStreams[lane] = stream;
            return opened;
        }
        stream.closeMarker(opened);
        return this.laneFences.get(lane);
    }

    /// Closes every lease the block still holds, without a fence: no device work reads a lease that
    /// was never submitted. For a block that ended early; no fetch may be outstanding.
    public void abandon() {
        for (int i = 0; i < this.held.length; i++) {
            ExpertLease lease = this.held[i];
            if (lease == null) continue;
            this.held[i] = null;
            lease.close();
        }
    }

    /// Queues the ordered accumulation of the experts' outputs, after every expert ran, then its combination
    /// with the shared expert's gated output.
    public void submitFinish(long output, int rows, Scratch scratch) {
        if (this.routing.activeExperts() > 0)
            Qwen4ExpertOps.combineExperts(
                    this.gpu, this.geometry, this.routing, this.deviceDescriptor, scratch.experts(), scratch.routed());
        else this.gpu.zeroDeviceMemory(scratch.routed(), 2L * rows * this.hidden);
        Qwen4MoeOps.finish(this.gpu, scratch.routed(), scratch.shared(), scratch.gateRaw(), output, rows, this.hidden);
    }

    private void release() {
        for (ExecutionGpu.ReadbackBuffer buffer : this.predictedLogits.values()) buffer.close();
        for (long device : this.predictedDevice.values()) this.gpu.free(device);
        this.gpu.free(this.deviceDescriptor);
        this.hostDescriptor.close();
        this.routeIds.close();
        this.routeWeights.close();
    }

    private static float bf16(short bits) {
        return Float.intBitsToFloat((bits & 0xFFFF) << 16);
    }

    /// Queues a later layer's router (`ahead`) applied to `layer`'s single row: a prediction of the experts that
    /// layer will ask for, read back for [#takePrediction].
    void submitPrediction(int layer, Weights ahead, long input) {
        long device = this.predictedDevice.computeIfAbsent(layer, l -> this.gpu.allocate(2L * this.experts));
        ExecutionGpu.ReadbackBuffer logits =
                this.predictedLogits.computeIfAbsent(layer, l -> this.gpu.allocateReadbackBuffer(2L * this.experts));
        Qwen4Ops.linearBf16(this.gpu, input, ahead.router().address(), device, 1, this.hidden, this.experts);
        this.gpu.copyDeviceToReadback(logits, device, 2L * this.experts);
    }

    /// The `k` experts `layer`'s prediction ([#submitPrediction], retired) ranks best, best first. A selection over
    /// the read-back logits: no sort, no boxing.
    int[] takePrediction(int layer, int k) {
        k = Math.min(k, this.experts);
        int[] best = new int[k];
        float[] value = new float[k];
        int held = 0;
        MemorySegment logits = this.predictedLogits.get(layer).segment();
        for (int e = 0; e < this.experts; e++) {
            float v = bf16(logits.getAtIndex(SHORT, e));
            if (held == k && v <= value[k - 1]) continue;
            int at = held < k ? held++ : k - 1;
            while (at > 0 && value[at - 1] < v) {
                value[at] = value[at - 1];
                best[at] = best[at - 1];
                at--;
            }
            value[at] = v;
            best[at] = e;
        }
        return best;
    }

    /// Reports the latest block's distinct experts and their routed rows to `demand`, and `prediction` (the next
    /// layer's experts ranked by its router) when there is one.
    void reportDemand(
            Qwen4ExecutionPlan.ExpertDemand demand, int layer, int ahead, int nextBank, int rows, int[] prediction) {
        if (prediction != null) demand.prediction(ahead, nextBank, prediction);
        int bank = this.bank;
        int count = this.routing.activeExperts();
        int[] experts = new int[count];
        int[] pairs = new int[count];
        for (int i = 0; i < count; i++) {
            experts[i] = this.routing.activeExpert(i);
            pairs[i] = this.routing.expertPairCount(experts[i]);
        }
        demand.block(layer, bank, rows, experts, pairs, count);
    }

    /// Distinct experts the latest block routed to, and the waves it ran in.
    public int lastUniqueExperts() {
        return this.lastUnique;
    }

    /// Releases the buffers. Every lane must have retired the blocks' work.
    @Override
    public void close() {
        if (this.closed) return;
        this.closed = true;
        for (int lane = 0; lane < this.fenceStreams.length; lane++) {
            long marker = this.laneFences.get(lane);
            if (marker != 0) this.fenceStreams[lane].closeMarker(marker);
        }
        release();
    }
}
