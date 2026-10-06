package io.euhedral_execution.inference.core.qwen4;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.gpu.GpuStream;
import io.euhedral_execution.inference.core.model_loader.qwen4.expert.ExpertCache;
import io.euhedral_execution.inference.core.model_loader.qwen4.expert.ExpertLease;
import io.euhedral_execution.inference.core.model_loader.qwen4.expert.StreamFence;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.concurrent.atomic.LongAdder;

/// The sparse MoE block of a Flash-Next layer (Qwen4ExpTextSparseMoeBlock): a router chooses ten of
/// 512 experts per token, a shared expert runs for every token, and
///
/// ```
/// out = routed experts (weighted, summed in ascending expert order) + sigmoid(shared gate) * shared expert
/// ```
///
/// The routed experts are not on the device: they pass through the [ExpertCache]. One instance is
/// the block resources of one stage graph (its route readback, its wave descriptors, the leases a
/// block holds); the stages of the graph use it in turn, never concurrently:
///
/// 1. [#submitRouting] queues the router and the copy of its choice to the host. The plan stage follows it
///    across a device-completion edge, so the host learns the routing without waiting for work that does not
///    feed its decision, and the shared expert ([#submitShared]) is a side branch that overlaps the host's
///    planning and the experts' movement;
/// 2. [#plan] (on the host, once the routing is readable) groups the (token, expert) pairs by expert and cuts
///    the experts into waves that fit the cache ([Qwen4ExpertWave]): a chunk of 512 tokens can name more
///    experts than a minimal cache holds, so no wave may need more than the slots it can lease;
/// 3. each wave's experts are loaded by a load stage and held ([#hold]); [#submitWave] copies the wave's
///    descriptor (slot addresses, work items, pairs) to the device, runs the grouped expert kernels, records
///    a stream marker behind them and closes every lease with that marker as its fence. Nothing waits for the
///    kernels: the cache's copy stream orders any later refill of a slot behind the marker on the device, so a
///    lease is held only while the kernels are being submitted;
/// 4. [#submitFinish] combines the routed sum with the shared expert's gated output.
///
/// No device pointer into a slot outlives its lease: the kernels read the slot addresses from the
/// descriptor the wave built while it held the leases, and the fence orders their completion before
/// the slot's next use.
///
/// One block is in flight at a time (a graph runs one quantum, and its layers in turn), so the
/// per-block state is plain fields.
public final class Qwen4MoeLayer implements AutoCloseable {

    /// The block's weights: the BF16 router and shared-expert gate, the NVFP4 shared expert.
    public record Weights(
            Qwen4Weight router,
            Qwen4Weight sharedGateProj,
            Qwen4Weight sharedUpProj,
            Qwen4Weight sharedDownProj,
            Qwen4Weight sharedExpertGate) {}

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
    private final ExpertCache cache;
    private final Qwen4ExpertOps.Geometry geometry;
    private final int hidden;
    private final int experts;
    private final int topK;
    private final int sharedInter;
    private final int maxRows;
    private final Qwen4ExpertWave wave;
    private final ExecutionGpu.ReadbackBuffer routeIds;
    private final ExecutionGpu.ReadbackBuffer routeWeights;
    private final ExecutionGpu.UploadBuffer[] hostDescriptors;
    private final long[] deviceDescriptors;
    private final int[] ids;
    private final short[] weights;
    private long fenceId;
    private GpuStream fenceOwner;

    private final int maxWaveExperts;

    // The leases the block holds, by position in the flattened list of its waves' experts (wave 0's first). A lease is
    // stored by the load's completion and closed by the wave that submits it, or by an abandoned block.
    private final ExpertLease[] held;
    private int[] waveStart;
    private int bank;

    private int lastUnique;
    private int lastWaves;
    private final Metrics metrics;
    private boolean closed;

    /// Counters that every graph of a plan adds to.
    public static final class Metrics {
        final LongAdder routeWaitNanos = new LongAdder();
        final LongAdder expertWaitNanos = new LongAdder();
        final LongAdder waves = new LongAdder();
        final LongAdder leased = new LongAdder();

        public Counters counters() {
            return new Counters(
                    this.routeWaitNanos.sum(), this.expertWaitNanos.sum(), this.waves.sum(), this.leased.sum());
        }
    }

    /// Host time between the router's copy being queued and the plan stage reading it, the host
    /// time the chain had nothing to do but wait for a wave's experts, and the waves run and
    /// experts leased since the counters were built.
    public record Counters(long routeWaitNanos, long expertWaitNanos, long waves, long leased) {}

    /// @param maxRows the most rows of a chunk
    /// @param maxWaveExperts the most experts one wave leases (at most the cache's slots)
    public Qwen4MoeLayer(
            ExecutionGpu gpu,
            ExpertCache cache,
            Qwen4ExpertOps.Geometry geometry,
            int experts,
            int topK,
            int sharedInter,
            int maxRows,
            int maxWaveExperts,
            Metrics metrics) {
        if (maxWaveExperts <= 0 || maxWaveExperts > cache.slotCount())
            throw new IllegalArgumentException("a wave of " + maxWaveExperts + " experts needs that many cache slots");
        this.gpu = gpu;
        this.cache = cache;
        this.geometry = geometry;
        this.hidden = geometry.hidden();
        this.experts = experts;
        this.topK = topK;
        this.sharedInter = sharedInter;
        this.maxRows = maxRows;
        this.maxWaveExperts = maxWaveExperts;
        this.metrics = metrics;
        int maxPairs = Math.multiplyExact(maxRows, topK);
        this.wave = new Qwen4ExpertWave(experts, topK, maxRows, maxWaveExperts, maxPairs);
        this.ids = new int[maxPairs];
        this.weights = new short[maxPairs];
        this.held = new ExpertLease[Math.min(experts, maxPairs)];
        this.waveStart = new int[this.held.length + 2];
        this.routeIds = gpu.allocateReadbackBuffer(4L * maxPairs);
        this.routeWeights = gpu.allocateReadbackBuffer(2L * maxPairs);
        int descriptors = maxWaves(experts, maxPairs, maxWaveExperts);
        this.hostDescriptors = new ExecutionGpu.UploadBuffer[descriptors];
        this.deviceDescriptors = new long[descriptors];
        try {
            for (int i = 0; i < descriptors; i++) {
                this.hostDescriptors[i] = gpu.allocateUploadBuffer(this.wave.descriptorBytes());
                this.deviceDescriptors[i] = gpu.allocate(this.wave.descriptorBytes());
            }
        } catch (Throwable failure) {
            release();
            throw failure;
        }
    }

    /// Most waves a block of `maxPairs` pairs can have when waves hold `maxWaveExperts` experts.
    public static int maxWaves(int experts, int maxPairs, int maxWaveExperts) {
        return (Math.min(experts, maxPairs) + maxWaveExperts - 1) / maxWaveExperts;
    }

    /// Device bytes of the scratch for `rows` rows.
    public long scratchBytes(int rows) {
        long bf16 = Short.BYTES;
        long maxPairs = (long) rows * this.topK;
        return align(rows * (long) this.experts * bf16) // logits
                + align(maxPairs * 4) // ids
                + align(maxPairs * bf16) // route weights
                + 3 * align(rows * (long) this.sharedInter * bf16) // shared gate, up, act
                + 2 * align(rows * (long) this.hidden * bf16) // shared, routed
                + align(rows * bf16) // shared expert gate
                + align(Qwen4ExpertWave.scratchBytes((int) maxPairs, this.geometry.inter(), this.hidden));
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
        long shared = sharedAct + align(rows * (long) this.sharedInter * bf16);
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

    /// The shared expert and its gate: the SwiGLU MLP of every row, and the gate's raw projection.
    private void sharedExpert(Weights weights, long input, int rows, Scratch scratch) {
        Qwen4Weight gate = weights.sharedGateProj();
        this.gpu.linearNvfp4Bf16(
                input, gate.address(), scratch.sharedGate(), rows, this.hidden, this.sharedInter, gate.bytes());
        Qwen4Weight up = weights.sharedUpProj();
        this.gpu.linearNvfp4Bf16(
                input, up.address(), scratch.sharedUp(), rows, this.hidden, this.sharedInter, up.bytes());
        Qwen4MoeOps.swiGlu(
                this.gpu, scratch.sharedGate(), scratch.sharedUp(), scratch.sharedAct(), rows * this.sharedInter);
        Qwen4Weight down = weights.sharedDownProj();
        this.gpu.linearNvfp4Bf16(
                scratch.sharedAct(),
                down.address(),
                scratch.shared(),
                rows,
                this.sharedInter,
                this.hidden,
                down.bytes());
        Qwen4Ops.linearBf16(
                this.gpu, input, weights.sharedExpertGate().address(), scratch.gateRaw(), rows, this.hidden, 1);
    }

    /// Reads the routing the device copied (readable once the boundary armed after [#submitRouting]
    /// retired) and plans the waves of the block over `rows` rows, for the layer whose experts are
    /// bank `bank`. Returns the waves.
    public int plan(int bank, int rows) {
        int pairs = rows * this.topK;
        MemorySegment.copy(this.routeIds.segment(), INT, 0, this.ids, 0, pairs);
        MemorySegment.copy(this.routeWeights.segment(), SHORT, 0, this.weights, 0, pairs);
        this.wave.plan(rows, this.ids, this.weights);
        this.lastUnique = this.wave.activeExperts();
        this.lastWaves = this.wave.waveCount();
        this.bank = bank;
        int total = 0;
        for (int w = 0; w < this.lastWaves; w++) {
            this.waveStart[w] = total;
            total += this.wave.waveExpertCount(w);
        }
        this.waveStart[this.lastWaves] = total;
        // A sentinel past the end keeps waveOf's scan bounded.
        this.waveStart[this.lastWaves + 1] = Integer.MAX_VALUE;
        return this.lastWaves;
    }

    /// Waves of the planned block.
    public int waveCount() {
        return this.wave.waveCount();
    }

    /// Experts of the planned block, in wave order: the positions the loads and the held leases are
    /// indexed by.
    public int activeExperts() {
        return this.wave.activeExperts();
    }

    /// Position of wave `w`'s first expert in the block's flattened list (`waveStart(waveCount())`
    /// is the total).
    public int waveStart(int w) {
        return this.waveStart[w];
    }

    /// The wave that the expert at flattened position `position` belongs to.
    public int waveOf(int position) {
        int w = 0;
        while (this.waveStart[w + 1] <= position) w++;
        return w;
    }

    /// The expert id of the `index`th expert of wave `w`.
    public int expertOf(int w, int index) {
        return this.wave.waveExpert(w, index);
    }

    /// The expert id at flattened position `position`, and the bank it is asked from.
    public int expertAt(int position) {
        int w = waveOf(position);
        return this.wave.waveExpert(w, position - this.waveStart[w]);
    }

    public int bank() {
        return this.bank;
    }

    /// Most experts a block can name: the length of its flattened list of waves.
    public int maxExperts() {
        return this.held.length;
    }

    /// Most waves a block can have: the descriptors one block uses.
    public int maxWaves() {
        return this.hostDescriptors.length;
    }

    /// The largest number of experts one wave holds.
    public int maxWaveExperts() {
        return this.maxWaveExperts;
    }

    /// Stores the lease that the load of position `position` produced, for the wave that will
    /// submit it.
    public void hold(int position, ExpertLease lease) {
        this.held[position] = lease;
    }

    /// Adds host time spent with nothing to do but wait: for the routing, and for a wave's experts.
    public void chargeRouteWait(long nanos) {
        this.metrics.routeWaitNanos.add(nanos);
    }

    public void chargeExpertWait(long nanos) {
        this.metrics.expertWaitNanos.add(nanos);
    }

    /// Queues the kernels of wave `w`, whose experts are all held, on the lane `stream` that the
    /// wave's stage chose, and closes its leases behind a fence on that lane. The wave's leases are
    /// closed even when this throws.
    public void submitWave(int w, GpuStream stream, long input, Scratch scratch) {
        int first = this.waveStart[w];
        int count = this.wave.waveExpertCount(w);
        StreamFence fence = null;
        try {
            MemorySegment descriptor = this.hostDescriptors[w].segment();
            this.wave.fill(w, descriptor);
            for (int i = 0; i < count; i++) {
                ExpertLease lease = this.held[first + i];
                if (lease == null)
                    throw new IllegalStateException("wave " + w + " is missing the lease of expert " + i);
                this.wave.setSlot(descriptor, i, lease.deviceAddress());
            }
            // An expert whose copy was only submitted is waited for here, on the device: the lane that runs the kernels
            // waits for the marker the copy recorded, and no host thread waits for the bytes.
            for (int i = 0; i < count; i++) {
                long ready = this.held[first + i].readyMarker();
                if (ready != 0) stream.await(ready);
            }
            long deviceDescriptor = this.deviceDescriptors[w];
            this.gpu.copyUploadToDevice(deviceDescriptor, this.hostDescriptors[w]);
            Qwen4ExpertOps.runWave(
                    this.gpu,
                    this.geometry,
                    this.wave,
                    w,
                    deviceDescriptor,
                    input,
                    scratch.experts(),
                    scratch.routed(),
                    w == 0);
            // The kernels that read the slots are queued; the marker recorded behind them is every lease's fence.
            long marker = fenceMarker(stream);
            stream.mark(marker);
            fence = new StreamFence(stream, marker);
            this.metrics.leased.add(count);
            this.metrics.waves.increment();
        } finally {
            for (int i = 0; i < count; i++) {
                ExpertLease lease = this.held[first + i];
                if (lease == null) continue;
                this.held[first + i] = null;
                if (fence != null) lease.close(fence);
                else lease.close();
            }
        }
    }

    /// The marker the waves' leases are fenced with. Every wave of a block follows the one before
    /// it across the device (a stage waits for its predecessor's marker), so a recording on a later
    /// wave's lane stands after every earlier wave's kernels.
    private long fenceMarker(GpuStream stream) {
        if (this.fenceId == 0) {
            this.fenceId = stream.openMarker();
            this.fenceOwner = stream;
        }
        return this.fenceId;
    }

    /// Closes every lease the block still holds, without a fence: no device work reads a lease that
    /// was never submitted. For a block that ended early; no load may be outstanding.
    public void abandon() {
        for (int i = 0; i < this.held.length; i++) {
            ExpertLease lease = this.held[i];
            if (lease == null) continue;
            this.held[i] = null;
            lease.close();
        }
    }

    /// Queues the combination of the routed sum and the shared expert's gated output.
    public void submitFinish(long output, int rows, Scratch scratch) {
        Qwen4MoeOps.finish(this.gpu, scratch.routed(), scratch.shared(), scratch.gateRaw(), output, rows, this.hidden);
    }

    private void release() {
        for (int i = 0; i < this.hostDescriptors.length; i++) {
            if (this.deviceDescriptors[i] != 0) this.gpu.free(this.deviceDescriptors[i]);
            if (this.hostDescriptors[i] != null) this.hostDescriptors[i].close();
        }
        this.routeIds.close();
        this.routeWeights.close();
    }

    /// Distinct experts the latest block routed to, and the waves it ran in.
    public int lastUniqueExperts() {
        return this.lastUnique;
    }

    public int lastWaves() {
        return this.lastWaves;
    }

    /// Releases the buffers. Every lane must have retired the blocks' work.
    @Override
    public void close() {
        if (this.closed) return;
        this.closed = true;
        if (this.fenceId != 0) this.fenceOwner.closeMarker(this.fenceId);
        release();
    }
}
