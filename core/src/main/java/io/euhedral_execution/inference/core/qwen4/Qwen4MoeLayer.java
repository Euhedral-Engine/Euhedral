package io.euhedral_execution.inference.core.qwen4;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.gpu.GpuStream;
import io.euhedral_execution.inference.core.model_loader.qwen4.expert.ExpertCache;
import io.euhedral_execution.inference.core.model_loader.qwen4.expert.ExpertLease;
import io.euhedral_execution.inference.core.model_loader.qwen4.expert.StreamFence;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/// The sparse MoE block of a Flash-Next layer (Qwen4ExpTextSparseMoeBlock): a router chooses ten of 512 experts per
/// token, a shared expert runs for every token, and
///
/// ```
/// out = routed experts (weighted, summed in ascending expert order) + sigmoid(shared gate) * shared expert
/// ```
///
/// The routed experts are not on the device: they pass through the [ExpertCache]. A layer is
///
/// 1. on the device: the router (logits, softmax, top ten, renormalized weights) and the shared expert, then a copy of
///    the routing to the host. This is the one host wait of the block: the host needs the choice to know which experts
///    to bring in;
/// 2. on the host: group the (token, expert) pairs by expert and cut the experts into waves that fit the cache
///    ([Qwen4ExpertWave]); a chunk of 512 tokens can name more experts than a minimal cache holds, so no wave may
///    need more than the slots it can lease;
/// 3. for each wave: lease its experts from the cache, copy the wave's descriptor (slot addresses, work items, pairs)
///    to the device, run the grouped expert kernels, record a stream marker behind them and close every lease with
///    that marker as its fence. Nothing waits for the kernels: the cache's copy stream orders any later refill of a
///    slot behind the marker on the device, so a lease is held only while the kernels are being submitted;
/// 4. combine the routed sum with the shared expert's gated output.
///
/// No device pointer into a slot outlives its lease: the kernels read the slot addresses from the descriptor the wave
/// built while it held the leases, and the fence orders their completion before the slot's next use.
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
    private final GpuStream stream;
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
    private final ExpertLease[] leases;
    private final long marker;
    private boolean closed;

    /// @param maxRows the most rows of a chunk
    /// @param maxWaveExperts the most experts one wave leases (at most the cache's slots)
    public Qwen4MoeLayer(
            ExecutionGpu gpu,
            GpuStream stream,
            ExpertCache cache,
            Qwen4ExpertOps.Geometry geometry,
            int experts,
            int topK,
            int sharedInter,
            int maxRows,
            int maxWaveExperts) {
        if (maxWaveExperts <= 0 || maxWaveExperts > cache.slotCount())
            throw new IllegalArgumentException("a wave of " + maxWaveExperts + " experts needs that many cache slots");
        this.gpu = gpu;
        this.stream = stream;
        this.cache = cache;
        this.geometry = geometry;
        this.hidden = geometry.hidden();
        this.experts = experts;
        this.topK = topK;
        this.sharedInter = sharedInter;
        this.maxRows = maxRows;
        int maxPairs = Math.multiplyExact(maxRows, topK);
        this.wave = new Qwen4ExpertWave(experts, topK, maxRows, maxWaveExperts, maxPairs);
        this.ids = new int[maxPairs];
        this.weights = new short[maxPairs];
        this.leases = new ExpertLease[maxWaveExperts];
        this.routeIds = gpu.allocateReadbackBuffer(4L * maxPairs);
        this.routeWeights = gpu.allocateReadbackBuffer(2L * maxPairs);
        int descriptors = (Math.min(experts, maxPairs) + maxWaveExperts - 1) / maxWaveExperts + 1;
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
        this.marker = stream.openMarker();
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

    /// Runs the block over `rows` rows at `input` (rows x hidden) into `output`, for the layer whose experts are bank
    /// `bank` of the cache. Everything is queued on the layer's stream; the host waits once, for the routing.
    public void run(Weights weights, int bank, long input, int rows, Scratch scratch, long output)
            throws InterruptedException {
        if (rows <= 0 || rows > this.maxRows) throw new IllegalArgumentException("rows " + rows);
        this.stream.submit(
                () -> {
                    Qwen4Ops.linearBf16(
                            this.gpu,
                            input,
                            weights.router().address(),
                            scratch.logits(),
                            rows,
                            this.hidden,
                            this.experts);
                    Qwen4MoeOps.router(
                            this.gpu,
                            scratch.logits(),
                            scratch.ids(),
                            scratch.routeWeights(),
                            rows,
                            this.experts,
                            this.topK);
                    this.gpu.copyDeviceToReadback(this.routeIds, scratch.ids(), 4L * rows * this.topK);
                    this.gpu.copyDeviceToReadback(this.routeWeights, scratch.routeWeights(), 2L * rows * this.topK);
                    sharedExpert(weights, input, rows, scratch);
                },
                false);
        Qwen4Streams.awaitCompletion(this.stream);

        int pairs = rows * this.topK;
        MemorySegment.copy(this.routeIds.segment(), INT, 0, this.ids, 0, pairs);
        MemorySegment.copy(this.routeWeights.segment(), SHORT, 0, this.weights, 0, pairs);
        this.wave.plan(rows, this.ids, this.weights);

        for (int w = 0; w < this.wave.waveCount(); w++) runWave(bank, w, input, rows, scratch);

        this.stream.submit(
                () -> Qwen4MoeOps.finish(
                        this.gpu, scratch.routed(), scratch.shared(), scratch.gateRaw(), output, rows, this.hidden),
                false);
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

    private void runWave(int bank, int w, long input, int rows, Scratch scratch) throws InterruptedException {
        int count = this.wave.waveExpertCount(w);
        int acquired = 0;
        try {
            MemorySegment descriptor = this.hostDescriptors[w].segment();
            this.wave.fill(w, descriptor);
            for (int i = 0; i < count; i++) {
                ExpertLease lease = this.cache.acquire(bank, this.wave.waveExpert(w, i));
                this.leases[acquired++] = lease;
                this.wave.setSlot(descriptor, i, lease.deviceAddress());
            }
            long deviceDescriptor = this.deviceDescriptors[w];
            boolean first = w == 0;
            this.stream.submit(
                    () -> {
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
                                first);
                    },
                    false);
            // The kernels that read the slots are queued; the marker recorded behind them is every lease's fence.
            this.stream.mark(this.marker);
        } finally {
            StreamFence fence = new StreamFence(this.stream, this.marker);
            for (int i = 0; i < acquired; i++) {
                this.leases[i].close(fence);
                this.leases[i] = null;
            }
        }
    }

    private void release() {
        for (int i = 0; i < this.hostDescriptors.length; i++) {
            if (this.deviceDescriptors[i] != 0) this.gpu.free(this.deviceDescriptors[i]);
            if (this.hostDescriptors[i] != null) this.hostDescriptors[i].close();
        }
        this.routeIds.close();
        this.routeWeights.close();
    }

    /// Releases the buffers. The stream must have retired every block's work.
    @Override
    public void close() {
        if (this.closed) return;
        this.closed = true;
        this.stream.closeMarker(this.marker);
        release();
    }
}
