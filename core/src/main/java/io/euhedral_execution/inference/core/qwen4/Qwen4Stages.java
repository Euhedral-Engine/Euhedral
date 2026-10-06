package io.euhedral_execution.inference.core.qwen4;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.scheduling.graph.StageFrame;
import io.euhedral_execution.inference.core.scheduling.graph.StageGraph;
import java.util.concurrent.atomic.AtomicInteger;

/// The stage frames of a [Qwen4Shape]. Each is a [StageFrame]: it submits its device work to the
/// lane it was given (or does its host work) and its completion makes its successors available. A
/// stage reads what it needs from the quantum bound to its graph and the graph's storage, and from
/// nothing that another stage is still writing: every such dependency is an edge of the shape.
final class Qwen4Stages {

    private Qwen4Stages() {}

    /// Whether the critical chain's stages keep one routing hash (the lane of the step); experiment
    /// switch.
    static final boolean ORDERED = System.getenv("EUHEDRAL_QWEN4_UNORDERED") == null;

    static StageFrame create(StageGraph graph, int stage, Qwen4Shape shape, Qwen4Shape.Spec spec) {
        int layer = spec.layer();
        int wave = spec.wave();
        return switch (spec.kind()) {
            case EMBED -> new Embed(graph, stage, shape);
            case PLEIDS -> new PleIds(graph, stage, shape, layer);
            case PLEGATHER -> new PleGather(graph, stage, shape, layer, wave);
            case PLE -> new Ple(graph, stage, shape, layer);
            case MIX -> new Mix(graph, stage, shape, layer);
            case ATTENTION -> new Attention(graph, stage, shape, layer, false);
            case INJECT -> new Inject(graph, stage, shape, layer);
            case BLOCK -> new Attention(graph, stage, shape, layer, true);
            case MID -> new Observe(graph, stage, shape, layer, true);
            case ROUTE -> new Route(graph, stage, shape, layer);
            case SHARED -> new Shared(graph, stage, shape, layer);
            case PLAN -> new Plan(graph, stage, shape, layer);
            case LOAD -> new Load(graph, stage, shape, layer, wave);
            case WAVE -> new Wave(graph, stage, shape, layer, wave);
            case DRAIN -> new Drain(graph, stage, shape, layer);
            case FINISH -> new Finish(graph, stage, shape, layer);
            case ENDINJECT -> new EndInject(graph, stage, shape, layer);
            case OBSERVE -> new Observe(graph, stage, shape, layer, false);
            case HEAD -> new Head(graph, stage, shape);
        };
    }

    /// What every stage reads.
    private abstract static class Base extends StageFrame {
        final Qwen4Shape shape;
        final int layer;

        Base(StageGraph graph, int stage, Qwen4Shape shape, int layer) {
            this(graph, stage, shape, layer, ORDERED);
        }

        Base(StageGraph graph, int stage, Qwen4Shape shape, int layer, boolean ordered) {
            super(graph, stage, ordered);
            this.shape = shape;
            this.layer = layer;
        }

        final Qwen4Quantum quantum() {
            return (Qwen4Quantum) graph().quantum();
        }

        final Qwen4ExecutionPlan plan() {
            return this.shape.plan();
        }

        final Qwen4GraphStorage storage() {
            return quantum().storage();
        }

        final ExecutionGpu gpu() {
            return plan().gpu();
        }

        final int rows() {
            return quantum().rows();
        }

        final Qwen4HyperConnection.Scratch hyperScratch() {
            return plan().hyperConnection().scratch(storage().hcScratch(), rows());
        }

        /// Starts timing `component` in a diagnostic shape.
        final void tick(int component) {
            if (this.shape.diagnostic()) quantum().tick(component);
        }

        final void attentionMix() {
            Qwen4ExecutionPlan plan = plan();
            Qwen4GraphStorage storage = storage();
            plan.hyperConnection()
                    .mix(
                            gpu(),
                            plan.weights().attentionResidual(this.layer),
                            storage.state(),
                            hyperScratch(),
                            storage.mixed(),
                            rows());
        }

        /// Runs the layer's attention (sparse or recurrent) over the mixed input into the block
        /// output.
        final void attention(Qwen4Sequence sequence) {
            Qwen4ExecutionPlan plan = plan();
            Qwen4GraphStorage storage = storage();
            if (plan.isSparse(this.layer))
                plan.qsa()
                        .run(
                                gpu(),
                                plan.weights().qsa(this.layer),
                                sequence.qsa(this.layer),
                                storage.mixed(),
                                rows(),
                                storage.blockOutput(),
                                plan.qsa().scratch(storage.layerScratch(), rows()),
                                0);
            else
                plan.gdn()
                        .run(
                                gpu(),
                                plan.weights().gdn(this.layer),
                                sequence.gdn(this.layer),
                                storage.mixed(),
                                rows(),
                                plan.gdn().scratch(storage.layerScratch(), rows()),
                                storage.blockOutput());
        }

        /// The attention block's injection into the residual streams, then the mix that feeds the
        /// MoE block.
        final void attentionInject() {
            Qwen4ExecutionPlan plan = plan();
            Qwen4GraphStorage storage = storage();
            Qwen4HyperConnection hyper = plan.hyperConnection();
            hyper.inject(gpu(), storage.state(), storage.blockOutput(), hyperScratch(), storage.state(), rows());
            hyper.mix(
                    gpu(),
                    plan.weights().moeResidual(this.layer),
                    storage.state(),
                    hyperScratch(),
                    storage.mixed(),
                    rows());
        }

        final void blockInject() {
            Qwen4GraphStorage storage = storage();
            plan().hyperConnection()
                    .inject(gpu(), storage.state(), storage.blockOutput(), hyperScratch(), storage.state(), rows());
        }
    }

    /// The token ids to the device, their embeddings, and the embedding repeated over the residual
    /// streams.
    private static final class Embed extends Base {
        Embed(StageGraph graph, int stage, Qwen4Shape shape) {
            super(graph, stage, shape, -1);
        }

        @Override
        protected void submit() {
            tick(Qwen4ExecutionPlan.Timings.EMBEDDING);
            Qwen4ExecutionPlan plan = plan();
            Qwen4GraphStorage storage = storage();
            gpu().copyUploadToDevice(storage.tokensDevice(), storage.tokenUpload());
            Qwen4Ops.embedding(
                    gpu(),
                    plan.embedding().address(),
                    storage.tokensDevice(),
                    storage.embedded(),
                    rows(),
                    plan.hidden(),
                    plan.vocabulary());
            Qwen4Ops.repeatStreams(gpu(), storage.embedded(), storage.state(), rows(), plan.streams(), plan.hidden());
        }
    }

    /// The n-gram row ids of the chunk's tokens and the staging buffer their records are gathered into.
    /// Host work that needs nothing from the device: a root of the graph. The stage owns the buffer for the
    /// quantum, whichever stages ran.
    private static final class PleIds extends Base {
        PleIds(StageGraph graph, int stage, Qwen4Shape shape, int layer) {
            super(graph, stage, shape, layer);
        }

        @Override
        protected boolean host() {
            return true;
        }

        @Override
        protected void submit() {
            Qwen4ExecutionPlan plan = plan();
            Qwen4GraphStorage storage = storage();
            Qwen4Quantum quantum = quantum();
            storage.pleUpload = null;
            storage.pleCount = plan.ple()
                    .prepare(quantum.sequence().ple(), quantum.tokens(), quantum.offset(), rows(), storage.pleRowIds);
            storage.pleUpload = gpu().allocateUploadBuffer(plan.ple().recordsBytes(rows()));
        }

        /// The device stopped reading the staging buffer, or the quantum ended without reading it.
        @Override
        protected void retired(boolean committed) {
            Qwen4GraphStorage storage = storage();
            ExecutionGpu.UploadBuffer finished = storage.pleUpload;
            storage.pleUpload = null;
            if (finished != null) finished.close();
        }
    }

    /// One part of the gather of the chunk's n-gram records: a range of the rows, read from the table
    /// (a mapped file whose pages may have to be faulted in) into the staging buffer. The parts are
    /// stages of their own, so their reads overlap on different workers.
    private static final class PleGather extends Base {
        private final int part;

        PleGather(StageGraph graph, int stage, Qwen4Shape shape, int layer, int part) {
            super(graph, stage, shape, layer, false);
            this.part = part;
        }

        @Override
        protected boolean host() {
            return true;
        }

        /// Rows below which a gather is not worth splitting: a part is a frame, and its hop costs more than
        /// a few dozen rows of copying.
        private static final int MIN_PART_ROWS = 64;

        private int parts() {
            return (int) Math.max(1, Math.min(Qwen4Shape.PLE_PARTS, storage().pleCount / MIN_PART_ROWS));
        }

        private int from() {
            return this.part >= parts() ? 0 : (int) ((long) storage().pleCount * this.part / parts());
        }

        private int to() {
            return this.part >= parts() ? 0 : (int) ((long) storage().pleCount * (this.part + 1) / parts());
        }

        @Override
        protected boolean skips() {
            return from() >= to();
        }

        @Override
        protected void submit() {
            Qwen4GraphStorage storage = storage();
            plan().ple().gather(storage.pleRowIds, from(), to(), storage.pleUpload.segment());
        }
    }

    /// The per-layer embedding: the gathered n-gram records are copied to the device, expanded, and added
    /// to the residual state.
    private static final class Ple extends Base {
        Ple(StageGraph graph, int stage, Qwen4Shape shape, int layer) {
            super(graph, stage, shape, layer);
        }

        @Override
        protected void submit() {
            tick(Qwen4ExecutionPlan.Timings.PLE);
            Qwen4ExecutionPlan plan = plan();
            Qwen4GraphStorage storage = storage();
            Qwen4Quantum quantum = quantum();
            var scratch = plan.ple().scratch(storage.layerScratch(), rows());
            plan.ple()
                    .apply(
                            gpu(),
                            plan.weights().ple(this.layer),
                            quantum.sequence().ple(),
                            rows(),
                            storage.state(),
                            scratch,
                            storage.pleOutput(),
                            storage.pleUpload);
            gpu().residualAddBf16(
                            storage.state(),
                            storage.pleOutput(),
                            storage.state(),
                            rows(),
                            plan.hyperConnection().stateWidth());
        }
    }

    /// The mix that feeds the attention block (diagnostic shape).
    private static final class Mix extends Base {
        Mix(StageGraph graph, int stage, Qwen4Shape shape, int layer) {
            super(graph, stage, shape, layer);
        }

        @Override
        protected void submit() {
            tick(Qwen4ExecutionPlan.Timings.RESIDUAL);
            attentionMix();
        }
    }

    /// The attention block: sparse (QSA) or recurrent (GDN). With `whole` it is the layer's entire
    /// attention half: the mix, the block, and the injection with the mix that feeds the MoE block;
    /// otherwise (diagnostic shape) just the block. A sparse layer's key/value state is committed
    /// when the quantum commits and discarded otherwise.
    private static final class Attention extends Base {
        private final boolean whole;
        private Qwen4QsaState opened;

        Attention(StageGraph graph, int stage, Qwen4Shape shape, int layer, boolean whole) {
            super(graph, stage, shape, layer);
            this.whole = whole;
        }

        @Override
        protected void submit() {
            tick(plan().isSparse(this.layer) ? Qwen4ExecutionPlan.Timings.QSA : Qwen4ExecutionPlan.Timings.GDN);
            this.opened = null;
            Qwen4Sequence sequence = quantum().sequence();
            if (plan().isSparse(this.layer)) this.opened = sequence.qsa(this.layer);
            if (this.whole) attentionMix();
            attention(sequence);
            if (this.whole) attentionInject();
        }

        @Override
        protected void retired(boolean committed) {
            Qwen4QsaState state = this.opened;
            this.opened = null;
            if (state == null) return;
            if (committed) state.commit();
            else state.discard();
        }
    }

    /// The attention block's injection and the mix that feeds the MoE block (diagnostic shape).
    private static final class Inject extends Base {
        Inject(StageGraph graph, int stage, Qwen4Shape shape, int layer) {
            super(graph, stage, shape, layer);
        }

        @Override
        protected void submit() {
            tick(Qwen4ExecutionPlan.Timings.RESIDUAL);
            attentionInject();
        }
    }

    /// A test hook that reads the state the device just finished (diagnostic shape): after the
    /// attention half (`middle`) or after the layer. It does nothing when no hook is set.
    private static final class Observe extends Base {
        private final boolean middle;

        Observe(StageGraph graph, int stage, Qwen4Shape shape, int layer, boolean middle) {
            super(graph, stage, shape, layer);
            this.middle = middle;
        }

        private Qwen4ExecutionPlan.Observer observer() {
            return this.middle ? plan().midObserver() : plan().observer();
        }

        @Override
        protected boolean host() {
            return true;
        }

        @Override
        protected boolean skips() {
            return observer() == null;
        }

        @Override
        protected void submit() {
            try {
                observer().layerFinished(this.layer, storage().state(), rows());
            } catch (InterruptedException interrupted) {
                Thread.currentThread().interrupt();
                throw new IllegalStateException("an observer was interrupted", interrupted);
            }
        }
    }

    /// The router and the copy of its choice to the host.
    private static final class Route extends Base {
        Route(StageGraph graph, int stage, Qwen4Shape shape, int layer) {
            super(graph, stage, shape, layer);
        }

        @Override
        protected void submit() {
            tick(Qwen4ExecutionPlan.Timings.MOE);
            Qwen4ExecutionPlan plan = plan();
            Qwen4GraphStorage storage = storage();
            Qwen4MoeLayer moe = storage.moe();
            storage.bank = plan.bankOrdinal(this.layer);
            storage.moeBlock = moe.scratch(storage.moeScratch(), rows());
            if (plan.traceOn()) storage.traceBefore = plan.expertStats().snapshot();
            moe.submitRouting(plan.weights().moe(this.layer), storage.mixed(), rows(), storage.moeBlock);
            storage.routeArmedNanos = System.nanoTime();
        }
    }

    /// The shared expert: a side branch that overlaps the host's planning and the experts'
    /// movement.
    private static final class Shared extends Base {
        Shared(StageGraph graph, int stage, Qwen4Shape shape, int layer) {
            super(graph, stage, shape, layer, false);
        }

        @Override
        protected void submit() {
            Qwen4GraphStorage storage = storage();
            storage.moe().submitShared(plan().weights().moe(this.layer), storage.mixed(), rows(), storage.moeBlock);
        }
    }

    /// Reads the router's choice (it reaches this stage across a device-completion edge) and plans
    /// the block's waves.
    private static final class Plan extends Base {
        Plan(StageGraph graph, int stage, Qwen4Shape shape, int layer) {
            super(graph, stage, shape, layer);
        }

        @Override
        protected boolean host() {
            return true;
        }

        @Override
        protected void submit() {
            Qwen4GraphStorage storage = storage();
            Qwen4MoeLayer moe = storage.moe();
            long now = System.nanoTime();
            moe.chargeRouteWait(now - storage.routeArmedNanos);
            storage.waves = moe.plan(storage.bank, rows());
            postShares();
        }

        /// Hands the block to the sources that own the cache's shards: each gets the positions of
        /// the experts that hash to it, in order, and generates the loads it needs. The wave
        /// stages' arrival points are readied first.
        private void postShares() {
            Qwen4ExecutionPlan plan = plan();
            Qwen4GraphStorage storage = storage();
            Qwen4MoeLayer moe = storage.moe();
            int total = moe.waveStart(storage.waves);
            long now = System.nanoTime();
            for (int w = 0; w < storage.waves; w++) {
                Qwen4GraphStorage.Arrivals arrivals =
                        (Qwen4GraphStorage.Arrivals) graph().stage(this.shape.loadStage(this.layer, w));
                arrivals.expect(moe.waveStart(w + 1) - moe.waveStart(w));
                storage.arrivals[w] = arrivals;
                storage.loadBegin[w] = now;
            }
            ExpertSource[] sources = plan.expertSources();
            java.util.Arrays.fill(storage.shareCounts, 0);
            var cache = plan.expertCache();
            for (int p = 0; p < total; p++) {
                int shard = cache.shardOf(storage.bank, moe.expertAt(p));
                storage.sharePositions[shard][storage.shareCounts[shard]++] = p;
            }
            storage.block.bind(quantum(), storage.bank);
            int shares = 0;
            for (int shard = 0; shard < sources.length; shard++) if (storage.shareCounts[shard] != 0) shares++;
            Drain drain = (Drain) graph().stage(this.shape.drainStage(this.layer));
            drain.expect(shares);
            storage.drain = drain;
            for (int shard = 0; shard < sources.length; shard++) {
                if (storage.shareCounts[shard] == 0) continue;
                sources[shard].submit(storage.shares[shard].set(
                        storage.block, storage.sharePositions[shard], storage.shareCounts[shard]));
            }
        }
    }

    /// The arrival point of one wave's experts: the stage ends when every expert of the wave has
    /// been taken from the cache or its copy submitted (each item reports its arrival), from
    /// whichever frame reports the last. It does no work itself; a wave the block does not have
    /// does nothing.
    private static final class Load extends Base implements Qwen4GraphStorage.Arrivals {
        private final int wave;
        private final AtomicInteger pending = new AtomicInteger();

        Load(StageGraph graph, int stage, Qwen4Shape shape, int layer, int wave) {
            super(graph, stage, shape, layer);
            this.wave = wave;
        }

        @Override
        protected boolean host() {
            return true;
        }

        @Override
        protected boolean skips() {
            return this.wave >= storage().waves;
        }

        /// The plan stage sets the count before it spawns an item, and this stage's own part (the
        /// extra one) arrives when it runs.
        @Override
        public void expect(int count) {
            this.pending.set(count + 1);
        }

        @Override
        public void arrived() {
            settle();
        }

        @Override
        protected void submit() {
            deferCompletion();
            settle();
        }

        private void settle() {
            if (this.pending.decrementAndGet() == 0) completeDeferred();
        }
    }

    /// The end of a block's claims: this stage ends when every source finished its share (each
    /// expert taken, each copy retired), so the next layer's plan only posts to sources that are
    /// free. It does no work itself.
    private static final class Drain extends Base implements Qwen4GraphStorage.Drain {
        private final AtomicInteger pending = new AtomicInteger();

        Drain(StageGraph graph, int stage, Qwen4Shape shape, int layer) {
            super(graph, stage, shape, layer);
        }

        @Override
        protected boolean host() {
            return true;
        }

        /// The plan stage sets the count before it posts the first share; this stage's own part is
        /// the extra one.
        @Override
        public void expect(int shares) {
            this.pending.set(shares + 1);
        }

        @Override
        public void done() {
            settle();
        }

        @Override
        protected void submit() {
            deferCompletion();
            settle();
        }

        private void settle() {
            if (this.pending.decrementAndGet() == 0) completeDeferred();
        }
    }

    /// One wave's kernels, over the experts the wave's load holds; the leases close behind a marker
    /// of the lane.
    private static final class Wave extends Base {
        private final int wave;

        Wave(StageGraph graph, int stage, Qwen4Shape shape, int layer, int wave) {
            super(graph, stage, shape, layer);
            this.wave = wave;
        }

        @Override
        protected boolean skips() {
            return this.wave >= storage().waves;
        }

        @Override
        protected void submit() {
            Qwen4GraphStorage storage = storage();
            Qwen4MoeLayer moe = storage.moe();
            long now = System.nanoTime();
            moe.chargeExpertWait(now - storage.loadBegin[this.wave]);
            moe.submitWave(this.wave, laneStream(), storage.mixed(), storage.moeBlock);
        }
    }

    /// Combines the routed sum with the shared expert's gated output and, outside the diagnostic
    /// shape, injects the block into the residual streams.
    private static final class Finish extends Base {
        Finish(StageGraph graph, int stage, Qwen4Shape shape, int layer) {
            super(graph, stage, shape, layer);
        }

        @Override
        protected void submit() {
            Qwen4ExecutionPlan plan = plan();
            Qwen4GraphStorage storage = storage();
            Qwen4MoeLayer moe = storage.moe();
            moe.submitFinish(storage.blockOutput(), rows(), storage.moeBlock);
            if (plan.traceOn())
                plan.reportTrace(
                        this.layer,
                        rows(),
                        moe.lastUniqueExperts(),
                        moe.lastWaves(),
                        storage.traceBefore,
                        plan.expertStats().snapshot());
            if (!this.shape.diagnostic()) blockInject();
        }
    }

    /// The MoE block's injection into the residual streams (diagnostic shape).
    private static final class EndInject extends Base {
        EndInject(StageGraph graph, int stage, Qwen4Shape shape, int layer) {
            super(graph, stage, shape, layer);
        }

        @Override
        protected void submit() {
            tick(Qwen4ExecutionPlan.Timings.RESIDUAL);
            blockInject();
        }
    }

    /// The output head over the last row, when the caller wants logits.
    private static final class Head extends Base {
        Head(StageGraph graph, int stage, Qwen4Shape shape) {
            super(graph, stage, shape, -1);
        }

        @Override
        protected boolean skips() {
            return quantum().sink() == null;
        }

        @Override
        protected void submit() {
            tick(Qwen4ExecutionPlan.Timings.HEAD);
            Qwen4ExecutionPlan plan = plan();
            Qwen4GraphStorage storage = storage();
            Qwen4HyperConnection hyper = plan.hyperConnection();
            long last = storage.state() + (long) (rows() - 1) * hyper.stateWidth() * Short.BYTES;
            hyper.mix(
                    gpu(),
                    plan.weights().finalMixer(),
                    last,
                    hyper.scratch(storage.hcScratch(), 1),
                    storage.finalMixed(),
                    1);
            Qwen4Ops.linearBf16(
                    gpu(),
                    storage.finalMixed(),
                    plan.head().address(),
                    storage.logits(),
                    1,
                    plan.hidden(),
                    plan.vocabulary());
            quantum().sink().queue(storage.logits());
        }
    }
}
