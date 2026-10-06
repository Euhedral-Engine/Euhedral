package io.euhedral_execution.inference.core.qwen4;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.model_loader.qwen4.NgramStore;
import io.euhedral_execution.inference.core.model_loader.qwen4.Qwen4Config;
import io.euhedral_execution.inference.core.model_loader.qwen4.Qwen4LayerType;
import io.euhedral_execution.inference.core.model_loader.qwen4.Qwen4Model;
import io.euhedral_execution.inference.core.model_loader.qwen4.expert.ExpertCache;
import io.euhedral_execution.inference.core.model_loader.qwen4.expert.ExpertCacheStats;
import io.euhedral_execution.inference.core.scheduling.EuhedralInferenceRuntime;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.atomic.LongAdder;

/// What a loaded Flash-Next model runs, as the shape of the work rather than a runner of it:
///
/// ```
/// tokens -> embedding rows -> repeated over 4 streams
///   -> layers 0..47: [PLE] mix -> GDN | QSA -> inject -> mix -> MoE -> inject
///   -> final mix of the last row -> output head -> logits
/// ```
///
/// A chunk of tokens (a prefill chunk or a decode token) is a quantum of a [Qwen4Shape]: a static
/// DAG of stage frames that the lattice's runtime instantiates, runs and recycles as it does for
/// the dense model. Each layer's MoE block is part of that DAG: the router and its choice, the
/// shared expert as a side branch, the plan that reads the choice, and per wave of experts a load
/// stage (which completes when the wave's experts are resident) and a wave stage (which runs their
/// kernels). A stage's completion is what makes its successors available; nothing here decides who
/// runs, waits, or in what order beyond the edges of the shape. The plan owns the model-level
/// resources that stages read (weights, the layers' operators, the expert cache); each graph owns
/// its workspace ([Qwen4GraphStorage]); and the plan serves one quantum at a time through a
/// completion chain, because a block's experts are sized to the cache's slots.
public final class Qwen4ExecutionPlan implements AutoCloseable {

    /// Most tokens of one chunk when the residency plan does not say otherwise.
    public static final int MAX_ROWS = 512;

    /// Receives the logits row of a step: called while the head stage's lane is selected, after the
    /// output head's launch, so a copy it queues is ordered behind the head.
    @FunctionalInterface
    public interface LogitsSink {
        void queue(long logitsAddress);
    }

    /// Wall time per component of the steps run while [#timings] is set: each component is followed
    /// by a wait for the device, so the parts add up to a slower step than an untimed one
    /// (diagnostics, never production).
    public static final class Timings {
        public static final int EMBEDDING = 0, PLE = 1, RESIDUAL = 2, GDN = 3, QSA = 4, MOE = 5, HEAD = 6;
        public static final String[] NAMES = {
            "embedding", "per-layer embedding", "gated residual", "GDN", "QSA", "MoE", "output head"
        };
        public final long[] nanos = new long[NAMES.length];
        public long steps;
        public long rows;
    }

    /// What a layer's MoE block asked of the expert cache, for the real routing of real inference:
    /// the distinct experts its tokens named and the cache's counters before and after it.
    @FunctionalInterface
    public interface ExpertTrace {
        void layer(
                int layer,
                int rows,
                int uniqueExperts,
                ExpertCacheStats.Snapshot before,
                ExpertCacheStats.Snapshot after);
    }

    /// Test hook: the residual state after each layer, read after the device finished it.
    public interface Observer {
        void layerFinished(int layer, long stateAddress, int rows) throws InterruptedException;
    }

    /// Receives the end of a step, on the worker that retired it (or that found it failed):
    /// `failure` is null when the step committed. It must not block.
    @FunctionalInterface
    public interface Listener {
        void finished(Throwable failure);
    }

    /// Stops a step at its next stage.
    public interface Handle {
        void cancel();
    }

    /// A test's way into a single layer's quantum: `provide` queues the residual state the layer
    /// starts from (on the quantum's home lane, before its stages) and `collect` reads the state
    /// after the device finished.
    interface StateExchange {
        void provide(ExecutionGpu gpu, long state);

        void collect(ExecutionGpu gpu, long state);
    }

    /// The layers a quantum runs and what surrounds them.
    record Range(int firstLayer, int endLayer, boolean embeds, boolean head, boolean advances) {}

    /// Everything that distinguishes one shape of the plan from another.
    record ShapeKey(Range range, int rows, boolean diagnostic) {}

    private final ExecutionGpu gpu;
    private final EuhedralInferenceRuntime runtime;
    private final Qwen4Model model;
    private final Qwen4Config config;
    private final Qwen4Weights weights;
    private final Qwen4HyperConnection hyperConnection;
    private final Qwen4GdnLayer gdn;
    private final Qwen4Ple ple;
    private final Qwen4QsaLayer qsa;
    private final Qwen4ExpertOps.Geometry geometry;
    private final Qwen4MoeLayer.Metrics metrics = new Qwen4MoeLayer.Metrics();
    private final int pleLayer;
    private final int hidden;
    private final int streams;
    private final int vocabulary;
    private final int maxTokens;
    /// The owner of the expert cache's bookkeeping, which the frames routed to it change.
    private final ExpertCacheOwner expertOwner;
    private final boolean[] sparse;
    private final int[] bankOrdinals;
    private final Qwen4Weight embedding;
    private final Qwen4Weight head;
    private final ConcurrentHashMap<ShapeKey, Qwen4Shape> shapes = new ConcurrentHashMap<>();
    /// The workspaces, one per row capacity of [#rowBuckets]: allocated with the plan, shared by
    /// the graphs of a capacity, freed by [#close].
    private final Qwen4GraphStorage[] workspaces;

    /// The row capacities of the plan's shapes: a decode token, a short chunk, a full chunk.
    private final int[] rowBuckets;
    private final int maxRows;
    /// The last quantum admitted: a new quantum registers as its successor, and its conclusion
    /// starts the new one.
    private final AtomicReference<Qwen4Quantum> chainTail = new AtomicReference<>();
    private final AtomicInteger active = new AtomicInteger();
    private final LongAdder stepsRun = new LongAdder();
    private volatile Timings timings;
    private volatile ExpertTrace trace;
    private volatile ExpertDemand demand;
    private volatile Observer observer;
    private volatile Observer midObserver;
    private volatile boolean closed;

    /// @param maxContextTokens the longest sequence this plan's sequences hold
    /// @param runtime the lattice runtime that instantiates and runs the plan's graphs
    public Qwen4ExecutionPlan(
            ExecutionGpu gpu, Qwen4Model model, int maxContextTokens, EuhedralInferenceRuntime runtime) {
        this.gpu = Objects.requireNonNull(gpu, "gpu");
        this.runtime = Objects.requireNonNull(runtime, "runtime");
        this.model = model;
        this.config = model.artifact().config();
        this.maxTokens = maxContextTokens;
        this.maxRows = model.plan().prefillChunkTokens();
        this.rowBuckets = new int[] {1, 16, this.maxRows};
        this.workspaces = new Qwen4GraphStorage[this.rowBuckets.length];
        this.hidden = this.config.text().hiddenSize();
        this.streams = this.config.hyperConnection().count();
        this.vocabulary = this.config.text().vocabSize();
        this.weights = new Qwen4Weights(model, gpu);
        float epsilon = (float) this.config.text().rmsNormEpsilon();
        this.hyperConnection = new Qwen4HyperConnection(
                this.streams, this.hidden, this.config.hyperConnection().lowrank(), epsilon);
        Qwen4Config.Gdn g = this.config.gdn();
        this.gdn = new Qwen4GdnLayer(
                this.hidden,
                g.numKeyHeads(),
                g.numValueHeads(),
                g.keyHeadDim(),
                g.valueHeadDim(),
                g.convKernelDim(),
                epsilon,
                this.config.attention().outputGate());
        Qwen4Config.Ngram n = this.config.ngram();
        NgramStore store = model.ngram();
        this.ple = new Qwen4Ple(
                store,
                new Qwen4NgramIds(
                        n.size(),
                        n.headsPerNgram(),
                        n.layerMultipliers(),
                        n.headsVocabSizes(),
                        n.headsOffsets(),
                        this.config.text().eosTokenId()),
                this.hidden,
                this.streams,
                this.config.ple().embedDim(),
                this.config.ple().convKernelSize(),
                n.size(),
                epsilon);
        this.pleLayer = this.config.ple().layers()[0];
        this.qsa = new Qwen4QsaLayer(Qwen4QsaLayer.Config.of(this.config, maxContextTokens));
        int layers = this.config.text().numLayers();
        this.sparse = new boolean[layers];
        for (int l = 0; l < layers; l++) this.sparse[l] = this.config.layerType(l) == Qwen4LayerType.SPARSE_ATTENTION;
        this.bankOrdinals = new int[layers];
        for (int l = 0; l < layers; l++) this.bankOrdinals[l] = model.bankOrdinal("text/layers/" + l + "/moe/experts");
        this.geometry = Qwen4ExpertOps.Geometry.of(model.expertBanks()[0]);
        this.expertOwner = new ExpertCacheOwner(model.expertCache(), runtime.lake());
        // The artifact's reads complete as frames: the workers poll their sink like every other.
        if (model.asyncReads() != null) runtime.lake().attach(model.asyncReads().getDelegate());
        this.embedding = this.weights.embedding();
        this.head = this.weights.head();
        // The workspaces are allocated now, not by the first step: a model that cannot hold them fails to load, and the
        // residency plan's reserve is spent at a known time.
        try {
            for (int i = 0; i < this.rowBuckets.length; i++)
                this.workspaces[i] = new Qwen4GraphStorage(this, gpu, this.rowBuckets[i]);
        } catch (Throwable failure) {
            closeWorkspaces();
            throw failure;
        }
    }

    private void closeWorkspaces() {
        for (int i = 0; i < this.workspaces.length; i++) {
            Qwen4GraphStorage storage = this.workspaces[i];
            this.workspaces[i] = null;
            if (storage != null) storage.close();
        }
    }

    // ---------------------------------------------------------------- starting quanta

    /// A new sequence of up to `maxContextTokens` positions (the plan's), at position 0.
    public Qwen4Sequence newSequence() {
        return new Qwen4Sequence(this.gpu, this.gdn, this.ple, this.sparse, this.qsaKeyValueWidth(), this.maxTokens);
    }

    private int qsaKeyValueWidth() {
        return this.config.attention().numKvHeads() * this.config.attention().headDim();
    }

    /// Starts `rows` tokens (`tokens[offset ..]`) at the sequence's position as one quantum and
    /// returns a handle that can cancel it. With a `sink` the last row's logits are computed and
    /// offered to it; the quantum commits (the sequence advances past the rows) when its device
    /// work retired, and then `listener` runs, on the worker that retired it. Nothing blocks the
    /// caller: when another quantum is running this one starts when that one concludes.
    ///
    /// `tokens` must stay unchanged until the listener ran.
    public Handle start(
            Qwen4Sequence sequence, int[] tokens, int offset, int rows, LogitsSink sink, Listener listener) {
        return begin(
                sequence,
                tokens,
                offset,
                rows,
                sink,
                new Range(0, this.sparse.length, true, true, true),
                null,
                listener);
    }

    /// Starts one layer of a chunk on the residual state `exchange` provides (tests: each layer fed
    /// the reference's own input). The chunk's other layers run through their own calls at the same
    /// position, and [#finishChunk] advances the sequence once all of them did.
    void startLayer(
            Qwen4Sequence sequence,
            int layer,
            int[] tokens,
            int offset,
            int rows,
            StateExchange exchange,
            Listener listener) {
        begin(
                sequence,
                tokens,
                offset,
                rows,
                null,
                new Range(layer, layer + 1, false, false, false),
                Objects.requireNonNull(exchange, "exchange"),
                listener);
    }

    /// Advances `sequence` past a chunk whose layers ran through [#startLayer].
    void finishChunk(Qwen4Sequence sequence, int rows) {
        sequence.advance(rows);
    }

    private Handle begin(
            Qwen4Sequence sequence,
            int[] tokens,
            int offset,
            int rows,
            LogitsSink sink,
            Range range,
            StateExchange exchange,
            Listener listener) {
        if (this.closed) throw new IllegalStateException("the plan is closed");
        if (rows <= 0 || rows > this.maxRows) throw new IllegalArgumentException("rows " + rows);
        if (range.advances() && sequence.position() + rows > sequence.maxTokens())
            throw new IllegalStateException("the sequence would exceed its " + sequence.maxTokens() + " positions");
        boolean diagnostic = timingsOn() || hasObserver() || hasMidObserver();
        Qwen4Shape shape = this.shapes.computeIfAbsent(
                new ShapeKey(range, rowBucket(rows), diagnostic), key -> new Qwen4Shape(this, key));
        Qwen4Quantum quantum = new Qwen4Quantum(this, shape, sequence, tokens, offset, rows, sink, exchange, listener);
        quantum.enter();
        return quantum;
    }

    /// A graph's hold on the workspace of capacity `rows`: the graphs of a capacity take turns on
    /// it.
    Qwen4GraphStorage.Lease leaseStorage(int rows) {
        for (int i = 0; i < this.rowBuckets.length; i++)
            if (this.rowBuckets[i] == rows) return new Qwen4GraphStorage.Lease(this.workspaces[i]);
        throw new IllegalArgumentException("no workspace of " + rows + " rows");
    }

    /// Most tokens of one chunk: the residency plan's prefill chunk.
    public int maxRows() {
        return this.maxRows;
    }

    /// The row capacity of the shape that serves `rows` rows: a decode token, a short chunk, a full
    /// chunk. The capacity sizes the shape's workspace and the most experts its MoE blocks can name,
    /// so a decode graph is small and has no expert stage that a token could not use.
    int rowBucket(int rows) {
        if (rows <= 1) return 1;
        if (rows <= 16) return 16;
        return this.maxRows;
    }

    // ---------------------------------------------------------------- what the shapes and their stages read

    EuhedralInferenceRuntime runtime() {
        return this.runtime;
    }

    AtomicReference<Qwen4Quantum> chainTail() {
        return this.chainTail;
    }

    Qwen4MoeLayer newMoeLayer(int rows) {
        return new Qwen4MoeLayer(
                this.gpu,
                this.geometry,
                this.config.moe().numExperts(),
                this.config.moe().expertsPerToken(),
                this.config.moe().sharedExpertIntermediateSize(),
                rows,
                this.metrics);
    }

    /// The most experts a block of a shape of `rows` rows can name: the fetch and expert stages of a layer.
    int maxExperts(int rows) {
        return Qwen4MoeLayer.maxExperts(
                this.config.moe().numExperts(), rows, this.config.moe().expertsPerToken());
    }

    /// The loads' timings.
    public ExpertCacheOwner.Timings expertTimings() {
        return this.expertOwner.timings();
    }

    /// Fetches that found every slot pinned and tried again.
    public long expertFullFetches() {
        return this.expertOwner.fullFetches();
    }

    /// [ExpertCacheOwner#prefetchCounts()]. Any thread.
    public long[] expertPrefetches() {
        return this.expertOwner.prefetchCounts();
    }

    /// [ExpertCacheOwner#fullCauses()]. Any thread.
    public long[] expertFullCauses() {
        return this.expertOwner.fullCauses();
    }

    /// Expert loads fetched whose copies have not retired yet. Any thread.
    public int expertLoadsInFlight() {
        return this.expertOwner.loadsInFlight();
    }

    /// The owner of the expert cache's bookkeeping.
    ExpertCacheOwner expertOwner() {
        return this.expertOwner;
    }

    int layers() {
        return this.sparse.length;
    }

    ExecutionGpu gpu() {
        return this.gpu;
    }

    ExpertCache expertCache() {
        return this.model.expertCache();
    }

    ExpertCacheStats expertStats() {
        return this.model.expertCache().stats();
    }

    Qwen4Weights weights() {
        return this.weights;
    }

    Qwen4HyperConnection hyperConnection() {
        return this.hyperConnection;
    }

    Qwen4GdnLayer gdn() {
        return this.gdn;
    }

    Qwen4Ple ple() {
        return this.ple;
    }

    Qwen4QsaLayer qsa() {
        return this.qsa;
    }

    int pleLayer() {
        return this.pleLayer;
    }

    boolean isSparse(int layer) {
        return this.sparse[layer];
    }

    int bankOrdinal(int layer) {
        return this.bankOrdinals[layer];
    }

    int hidden() {
        return this.hidden;
    }

    int streams() {
        return this.streams;
    }

    int vocabulary() {
        return this.vocabulary;
    }

    Qwen4Weight embedding() {
        return this.embedding;
    }

    Qwen4Weight head() {
        return this.head;
    }

    // ---------------------------------------------------------------- diagnostics

    public Qwen4MoeLayer.Counters moeCounters() {
        return this.metrics.counters();
    }

    /// The experts each MoE block asks for, in the order the blocks are planned: a recording of the cache's demand
    /// that a simulator can replay. `experts[0..count)` are the block's distinct experts (ascending) in bank `bank`
    /// (the cache's ordinal of the layer's experts), and `pairs` the routed rows of each; the listener may keep the
    /// arrays.
    @FunctionalInterface
    public interface ExpertDemand {
        void block(int layer, int bank, int rows, int[] experts, int[] pairs, int count);

        /// The next layer's router applied to a single-row block's input: `layer`'s experts in bank `bank`, ranked
        /// by that router's logits, best first. Reported only when [#predicts()].
        default void prediction(int layer, int bank, int[] ranked) {}

        /// Whether to run and report the next layer's router on every single-row block.
        default boolean predicts() {
            return false;
        }
    }

    /// Reports every MoE block's demand to `demand` (null stops it). Diagnostics, never production.
    public void demand(ExpertDemand demand) {
        this.demand = demand;
    }

    ExpertDemand demandListener() {
        return this.demand;
    }

    /// Reports every MoE block's use of the expert cache to `trace` (null stops it).
    public void trace(ExpertTrace trace) {
        this.trace = trace;
    }

    /// Starts (or with null stops) timing the components of every step.
    public void timings(Timings timings) {
        this.timings = timings;
    }

    /// Takes the residual state of each layer's end for the observer (tests); null stops it.
    void observe(Observer observer) {
        this.observer = observer;
    }

    /// Takes the state after each layer's attention injection, before its MoE block (tests); null
    /// stops it.
    void observeMid(Observer observer) {
        this.midObserver = observer;
    }

    public int vocabularySize() {
        return this.vocabulary;
    }

    boolean timingsOn() {
        return this.timings != null;
    }

    void charge(int component, long nanos) {
        Timings current = this.timings;
        if (current != null) current.nanos[component] += nanos;
    }

    void stepCompleted(int rows) {
        Timings current = this.timings;
        if (current != null) {
            current.steps++;
            current.rows += rows;
        }
        this.stepsRun.increment();
    }

    boolean traceOn() {
        return this.trace != null;
    }

    void reportTrace(
            int layer, int rows, int uniqueExperts, ExpertCacheStats.Snapshot before, ExpertCacheStats.Snapshot after) {
        ExpertTrace current = this.trace;
        if (current != null) current.layer(layer, rows, uniqueExperts, before, after);
    }

    boolean hasObserver() {
        return this.observer != null;
    }

    Observer observer() {
        return this.observer;
    }

    boolean hasMidObserver() {
        return this.midObserver != null;
    }

    Observer midObserver() {
        return this.midObserver;
    }

    void quantumStarted() {
        this.active.incrementAndGet();
    }

    void quantumEnded() {
        this.active.decrementAndGet();
    }

    /// Steps that retired.
    public long stepsRun() {
        return this.stepsRun.sum();
    }

    /// Steps running now (the others of the chain have not started).
    public int stepsInFlight() {
        return this.active.get();
    }

    /// Ends the plan. The runtime that runs its graphs closes them (and their workspaces) when it
    /// closes, after every quantum retired.
    @Override
    public void close() {
        this.closed = true;
        closeWorkspaces();
    }
}
