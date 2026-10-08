package io.euhedral_execution.inference.core.model.qwen4;

import io.euhedral_execution.core.frames.AbstractFrame;
import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.model.qwen4.expert.ExpertLease;
import io.euhedral_execution.inference.core.runtime.graph.StageFrame;
import io.euhedral_execution.inference.core.runtime.graph.StageGraph;

/// The stage frames of a [Shape]. Each is a [StageFrame]: it submits its device work to the
/// lane it was given (or does its host work) and its completion makes its successors available. A
/// stage reads what it needs from the quantum bound to its graph and the graph's storage, and from
/// nothing that another stage is still writing: every such dependency is an edge of the shape.
final class Stages {

    private Stages() {}

    static StageFrame create(StageGraph graph, int stage, Shape shape, Shape.Spec spec) {
        return create(graph, stage, shape, spec, 0);
    }

    /// The frame of `spec` in chunk `chunk` of a prompt graph, at stage `stage`.
    static StageFrame create(StageGraph graph, int stage, Shape shape, Shape.Spec spec, int chunk) {
        StageFrame frame = of(graph, stage, shape, spec);
        ((Base) frame).chunk = chunk;
        return frame;
    }

    private static StageFrame of(StageGraph graph, int stage, Shape shape, Shape.Spec spec) {
        int layer = spec.layer();
        int index = spec.index();
        return switch (spec.kind()) {
            case EMBED -> new Embed(graph, stage, shape);
            case PLEIDS -> new PleIds(graph, stage, shape, layer);
            case PLEGATHER -> new PleGather(graph, stage, shape, layer, index);
            case PLE -> new Ple(graph, stage, shape, layer);
            case MIX -> new Mix(graph, stage, shape, layer);
            case ATTENTION -> new Attention(graph, stage, shape, layer, false);
            case INJECT -> new Inject(graph, stage, shape, layer);
            case BLOCK -> new Attention(graph, stage, shape, layer, true);
            case MID -> new Observe(graph, stage, shape, layer, true);
            case ROUTE -> new Route(graph, stage, shape, layer);
            case SHARED -> new Shared(graph, stage, shape, layer);
            case PLAN -> new Plan(graph, stage, shape, layer);
            case PREDICT -> new Predict(graph, stage, shape, layer);
            case PREFETCH -> new Prefetch(graph, stage, shape, layer);
            case FETCH -> new Fetch(graph, stage, shape, layer, index);
            case EXPERT -> new Expert(graph, stage, shape, layer, index);
            case FINISH -> new Finish(graph, stage, shape, layer);
            case ENDINJECT -> new EndInject(graph, stage, shape, layer);
            case OBSERVE -> new Observe(graph, stage, shape, layer, false);
            case HEAD -> new Head(graph, stage, shape);
        };
    }

    /// What every stage reads.
    private abstract static class Base extends StageFrame {
        final Shape shape;
        final int layer;
        /// The chunk of a prompt graph this stage belongs to; 0 in any other graph.
        int chunk;

        Base(StageGraph graph, int stage, Shape shape, int layer) {
            this(graph, stage, shape, layer, false);
        }

        /// A stage routed to the owner of some state, by its hash.
        Base(StageGraph graph, int stage, Shape shape, int layer, long ownerHash) {
            super(graph, stage, ownerHash);
            this.shape = shape;
            this.layer = layer;
        }

        Base(StageGraph graph, int stage, Shape shape, int layer, boolean ordered) {
            super(graph, stage, ordered);
            this.shape = shape;
            this.layer = layer;
        }

        /// The chunk of the quantum this stage runs.
        final Quantum.Chunk chunk() {
            return quantum().chunk(this.chunk);
        }

        /// The n-gram rows of this stage's chunk.
        final Quantum.PleRows pleRows() {
            return quantum().pleRows(this.chunk);
        }

        final Quantum quantum() {
            return (Quantum) graph().quantum();
        }

        final ExecutionPlan plan() {
            return this.shape.plan();
        }

        final Workspace storage() {
            return quantum().storage();
        }

        final MoeLayer moe() {
            return quantum().moe();
        }

        /// Runs `work` with the workspace's expansion scratch bound when the shape declared this stage a
        /// scratch user ([Shape#takesScratch]): its edges, and the workspace's across graphs, give it the
        /// region alone.
        final void withScratch(Runnable work) {
            Workspace storage = storage();
            if (this.shape.takesScratch(stage()) && storage.scratchAddress() != 0)
                gpu().withScratch(storage.scratchAddress(), storage.scratchBytes(), work);
            else work.run();
        }

        final ExecutionGpu gpu() {
            return plan().gpu();
        }

        final int rows() {
            return chunk().rows();
        }

        final HyperConnection.Scratch hyperScratch() {
            return plan().hyperConnection().scratch(storage().hcScratch(), rows());
        }

        /// Starts timing `component` in a diagnostic shape.
        final void tick(int component) {
            if (this.shape.diagnostic()) quantum().tick(component);
        }

        final void attentionMix() {
            ExecutionPlan plan = plan();
            Workspace storage = storage();
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
        final void attention(Sequence sequence) {
            ExecutionPlan plan = plan();
            Workspace storage = storage();
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
            ExecutionPlan plan = plan();
            Workspace storage = storage();
            HyperConnection hyper = plan.hyperConnection();
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
            Workspace storage = storage();
            plan().hyperConnection()
                    .inject(gpu(), storage.state(), storage.blockOutput(), hyperScratch(), storage.state(), rows());
        }
    }

    /// The token ids to the device, their embeddings, and the embedding repeated over the residual
    /// streams.
    private static final class Embed extends Base {
        Embed(StageGraph graph, int stage, Shape shape) {
            super(graph, stage, shape, -1);
        }

        @Override
        protected void submit() {
            tick(ExecutionPlan.Timings.EMBEDDING);
            ExecutionPlan plan = plan();
            Workspace storage = storage();
            Quantum quantum = quantum();
            if (quantum.chunkCount() == 1) gpu().copyUploadToDevice(storage.tokensDevice(), storage.tokenUpload());
            else gpu().copyHostToDeviceAsync(storage.tokensDevice(), quantum.promptTokens(this.chunk), 4L * rows());
            Ops.embedding(
                    gpu(),
                    plan.embedding().address(),
                    storage.tokensDevice(),
                    storage.embedded(),
                    rows(),
                    plan.hidden(),
                    plan.vocabulary());
            Ops.repeatStreams(gpu(), storage.embedded(), storage.state(), rows(), plan.streams(), plan.hidden());
        }
    }

    /// The n-gram row ids of the chunk's tokens and the staging buffer their records are gathered into.
    /// Host work that needs nothing from the device: a root of the graph. The stage owns the buffer for the
    /// quantum, whichever stages ran.
    private static final class PleIds extends Base {
        PleIds(StageGraph graph, int stage, Shape shape, int layer) {
            super(graph, stage, shape, layer);
        }

        @Override
        protected boolean host() {
            return true;
        }

        @Override
        protected void submit() {
            ExecutionPlan plan = plan();
            Workspace storage = storage();
            Quantum quantum = quantum();
            Quantum.PleRows ple = pleRows();
            ple.upload = null;
            ple.count = plan.ple()
                    .prepare(quantum.sequence().ple(), quantum.tokens(), chunk().offset(), rows(), ple.rowIds);
            ple.upload = gpu().allocateUploadBuffer(plan.ple().recordsBytes(rows()));
        }

        /// The device stopped reading the staging buffer, or the quantum ended without reading it.
        @Override
        protected void retired(boolean committed) {
            Quantum.PleRows ple = pleRows();
            ExecutionGpu.UploadBuffer finished = ple.upload;
            ple.upload = null;
            if (finished != null) finished.close();
        }
    }

    /// One part of the gather of the chunk's n-gram records: a range of the rows, read from the table
    /// (a mapped file whose pages may have to be faulted in) into the staging buffer. The parts are
    /// stages of their own, so their reads overlap on different workers.
    private static final class PleGather extends Base {
        private final int part;

        PleGather(StageGraph graph, int stage, Shape shape, int layer, int part) {
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
            return (int) Math.max(1, Math.min(Shape.PLE_PARTS, pleRows().count / MIN_PART_ROWS));
        }

        private int from() {
            return this.part >= parts() ? 0 : (int) ((long) pleRows().count * this.part / parts());
        }

        private int to() {
            return this.part >= parts() ? 0 : (int) ((long) pleRows().count * (this.part + 1) / parts());
        }

        @Override
        protected boolean skips() {
            return from() >= to();
        }

        @Override
        protected void submit() {
            Workspace storage = storage();
            Quantum.PleRows ple = pleRows();
            plan().ple().gather(ple.rowIds, from(), to(), ple.upload.segment());
        }
    }

    /// The per-layer embedding: the gathered n-gram records are copied to the device, expanded, and added
    /// to the residual state.
    private static final class Ple extends Base {
        Ple(StageGraph graph, int stage, Shape shape, int layer) {
            super(graph, stage, shape, layer);
        }

        @Override
        protected void submit() {
            withScratch(this::apply);
        }

        private void apply() {
            tick(ExecutionPlan.Timings.PLE);
            ExecutionPlan plan = plan();
            Workspace storage = storage();
            Quantum quantum = quantum();
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
                            pleRows().upload);
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
        Mix(StageGraph graph, int stage, Shape shape, int layer) {
            super(graph, stage, shape, layer);
        }

        @Override
        protected void submit() {
            tick(ExecutionPlan.Timings.RESIDUAL);
            attentionMix();
        }
    }

    /// The attention block: sparse (QSA) or recurrent (GDN). With `whole` it is the layer's entire
    /// attention half: the mix, the block, and the injection with the mix that feeds the MoE block;
    /// otherwise (diagnostic shape) just the block. A sparse layer's key/value state is committed
    /// when the quantum commits and discarded otherwise.
    private static final class Attention extends Base {
        private final boolean whole;
        private QsaState opened;

        Attention(StageGraph graph, int stage, Shape shape, int layer, boolean whole) {
            super(graph, stage, shape, layer);
            this.whole = whole;
        }

        @Override
        protected void submit() {
            withScratch(this::block);
        }

        private void block() {
            tick(plan().isSparse(this.layer) ? ExecutionPlan.Timings.QSA : ExecutionPlan.Timings.GDN);
            this.opened = null;
            Sequence sequence = quantum().sequence();
            if (plan().isSparse(this.layer)) this.opened = sequence.qsa(this.layer);
            if (this.whole) attentionMix();
            attention(sequence);
            if (this.whole) attentionInject();
        }

        @Override
        protected void retired(boolean committed) {
            QsaState state = this.opened;
            this.opened = null;
            if (state == null) return;
            if (committed) state.commit();
            else state.discard();
        }
    }

    /// The attention block's injection and the mix that feeds the MoE block (diagnostic shape).
    private static final class Inject extends Base {
        Inject(StageGraph graph, int stage, Shape shape, int layer) {
            super(graph, stage, shape, layer);
        }

        @Override
        protected void submit() {
            tick(ExecutionPlan.Timings.RESIDUAL);
            attentionInject();
        }
    }

    /// A test hook that reads the state the device just finished (diagnostic shape): after the
    /// attention half (`middle`) or after the layer. It does nothing when no hook is set.
    private static final class Observe extends Base {
        private final boolean middle;

        Observe(StageGraph graph, int stage, Shape shape, int layer, boolean middle) {
            super(graph, stage, shape, layer);
            this.middle = middle;
        }

        private ExecutionPlan.Observer observer() {
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
        Route(StageGraph graph, int stage, Shape shape, int layer) {
            super(graph, stage, shape, layer);
        }

        @Override
        protected void submit() {
            tick(ExecutionPlan.Timings.MOE);
            ExecutionPlan plan = plan();
            Workspace storage = storage();
            MoeLayer moe = moe();
            storage.bank = plan.bankOrdinal(this.layer);
            storage.moeBlock = moe.scratch(storage.moeScratch(), rows());
            if (plan.traceOn()) storage.traceBefore = plan.expertStats().snapshot();
            moe.submitRouting(plan.weights().moe(this.layer), storage.mixed(), rows(), storage.moeBlock);
            storage.routeArmedNanos = System.nanoTime();
        }
    }

    /// A later layer's router applied to this decode step's input: the prediction a prefetch reads ahead by.
    private static final class Predict extends Base {
        Predict(StageGraph graph, int stage, Shape shape, int layer) {
            super(graph, stage, shape, layer);
        }

        @Override
        protected void submit() {
            int ahead = this.layer + ExpertCacheOwner.prefetchDistance();
            moe().submitPrediction(
                            this.layer, plan().weights().moe(ahead), storage().mixed());
        }
    }

    /// Reads the prediction back on the host and hands it to the cache's owner, which reads what the device and the
    /// host tier lack into the tier; a demand recording also gets it.
    private static final class Prefetch extends Base {
        Prefetch(StageGraph graph, int stage, Shape shape, int layer) {
            super(graph, stage, shape, layer);
        }

        @Override
        protected boolean host() {
            return true;
        }

        @Override
        protected void submit() {
            ExecutionPlan plan = plan();
            int ahead = this.layer + ExpertCacheOwner.prefetchDistance();
            int bank = plan.bankOrdinal(ahead);
            ExecutionPlan.ExpertDemand demand = plan.demandListener();
            int candidates = ExpertCacheOwner.prefetchCandidates();
            int[] ranked = moe().takePrediction(this.layer, Math.max(candidates, demand != null ? 64 : 0));
            plan.expertOwner()
                    .publishPrefetch(bank, java.util.Arrays.copyOf(ranked, Math.min(ranked.length, candidates)));
            if (demand != null) demand.prediction(ahead, bank, ranked);
        }
    }

    /// The shared expert: a side branch that overlaps the host's planning and the experts'
    /// movement.
    private static final class Shared extends Base {
        Shared(StageGraph graph, int stage, Shape shape, int layer) {
            super(graph, stage, shape, layer, false);
        }

        @Override
        protected void submit() {
            withScratch(() -> {
                Workspace storage = storage();
                moe().submitShared(plan().weights().moe(this.layer), storage.mixed(), rows(), storage.moeBlock);
            });
        }
    }

    /// Reads the router's choice (it reaches this stage across a device-completion edge), groups the block's
    /// pairs by expert, and copies the block's description to the device for the expert stages.
    private static final class Plan extends Base {
        Plan(StageGraph graph, int stage, Shape shape, int layer) {
            super(graph, stage, shape, layer);
        }

        @Override
        protected void submit() {
            Workspace storage = storage();
            MoeLayer moe = moe();
            long now = System.nanoTime();
            moe.chargeRouteWait(now - storage.routeArmedNanos);
            storage.plannedNanos = now;
            storage.experts = moe.plan(storage.bank, rows());
            ExecutionPlan.ExpertDemand demand = plan().demandListener();
            if (demand != null) moe.reportDemand(demand, this.layer, -1, -1, rows(), null);
            moe.submitPlan();
        }
    }

    /// Takes the `index`-th active expert of the block from the cache. It runs where the cache's bookkeeping
    /// lives (its routing hash is the cache owner's, so it runs in order with every other frame that touches
    /// that state): a resident expert is a lease at once and the stage ends; a missing one reserves a slot and
    /// starts its load, and the stage ends when the load hands it the lease. When every slot that could hold
    /// the expert is pinned, the stage publishes itself again and tries once more when the lattice runs it, as
    /// any frame that finds its resource full does; no other expert waits for it.
    private static final class Fetch extends Base implements ExpertCacheOwner.Fetch {
        private final int index;
        private final Retry retry = new Retry();

        Fetch(StageGraph graph, int stage, Shape shape, int layer, int index) {
            super(graph, stage, shape, layer, ExpertCacheOwner.HASH);
            this.index = index;
        }

        @Override
        protected boolean host() {
            return true;
        }

        @Override
        protected boolean skips() {
            return this.index >= storage().experts;
        }

        @Override
        protected void submit() {
            deferCompletion();
            fetch();
        }

        /// Asks the cache for the expert. Runs on the owner's frames only (this stage, or its retry).
        private void fetch() {
            if (stopped()) {
                completeDeferred();
                return;
            }
            Workspace storage = storage();
            int expert = moe().activeExpert(this.index);
            if (plan().expertOwner().fetch(this, storage.bank, expert) == ExpertCacheOwner.Outcome.FULL)
                graph().lake().publish(this.retry);
        }

        @Override
        public boolean stopped() {
            return quantum().stopRequested();
        }

        @Override
        public boolean scan() {
            return rows() > 1;
        }

        @Override
        public void arrived(ExpertLease lease) {
            if (stopped()) lease.close();
            else moe().hold(this.index, lease);
            completeDeferred();
        }

        @Override
        public void failed(Throwable failure) {
            quantum().fail(failure);
            completeDeferred();
        }

        /// The fetch once more, routed to the owner.
        private final class Retry extends AbstractFrame {
            Retry() {
                super(ExpertCacheOwner.HASH);
            }

            @Override
            public void execute() {
                fetch();
            }

            @Override
            public void doFinally() {}

            @Override
            public void doFinallyWithError(Throwable rejection) {
                failed(new IllegalStateException("the lattice rejected an expert fetch", rejection));
            }
        }
    }

    /// The kernels of the `index`-th active expert, once it is held; its lease closes behind a marker of the
    /// lane. Independent of every other expert.
    private static final class Expert extends Base {
        private final int index;

        Expert(StageGraph graph, int stage, Shape shape, int layer, int index) {
            super(graph, stage, shape, layer, false);
            this.index = index;
        }

        @Override
        protected boolean skips() {
            return this.index >= storage().experts;
        }

        @Override
        protected void submit() {
            Workspace storage = storage();
            moe().submitExpert(this.index, laneStream(), laneIndex(), storage.mixed(), storage.moeBlock);
        }
    }

    /// Combines the routed sum with the shared expert's gated output and, outside the diagnostic
    /// shape, injects the block into the residual streams.
    private static final class Finish extends Base {
        Finish(StageGraph graph, int stage, Shape shape, int layer) {
            super(graph, stage, shape, layer);
        }

        @Override
        protected void submit() {
            ExecutionPlan plan = plan();
            Workspace storage = storage();
            MoeLayer moe = moe();
            moe.chargeExpertWait(System.nanoTime() - storage.plannedNanos);
            moe.submitFinish(storage.blockOutput(), rows(), storage.moeBlock);
            if (plan.traceOn())
                plan.reportTrace(
                        this.layer,
                        rows(),
                        moe.lastUniqueExperts(),
                        storage.traceBefore,
                        plan.expertStats().snapshot());
            if (!this.shape.diagnostic()) blockInject();
        }
    }

    /// The MoE block's injection into the residual streams (diagnostic shape).
    private static final class EndInject extends Base {
        EndInject(StageGraph graph, int stage, Shape shape, int layer) {
            super(graph, stage, shape, layer);
        }

        @Override
        protected void submit() {
            tick(ExecutionPlan.Timings.RESIDUAL);
            blockInject();
        }
    }

    /// The output head over the last row, when the caller wants logits.
    private static final class Head extends Base {
        Head(StageGraph graph, int stage, Shape shape) {
            super(graph, stage, shape, -1);
        }

        @Override
        protected boolean skips() {
            return quantum().sink() == null || !chunk().last();
        }

        @Override
        protected void submit() {
            tick(ExecutionPlan.Timings.HEAD);
            ExecutionPlan plan = plan();
            Workspace storage = storage();
            HyperConnection hyper = plan.hyperConnection();
            long last = storage.state() + (long) (rows() - 1) * hyper.stateWidth() * Short.BYTES;
            hyper.mix(
                    gpu(),
                    plan.weights().finalMixer(),
                    last,
                    hyper.scratch(storage.hcScratch(), 1),
                    storage.finalMixed(),
                    1);
            Ops.linearBf16(
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
