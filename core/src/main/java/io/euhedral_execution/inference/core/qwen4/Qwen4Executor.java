package io.euhedral_execution.inference.core.qwen4;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.gpu.GpuStream;
import io.euhedral_execution.inference.core.model_loader.qwen4.NgramStore;
import io.euhedral_execution.inference.core.model_loader.qwen4.Qwen4Config;
import io.euhedral_execution.inference.core.model_loader.qwen4.Qwen4LayerType;
import io.euhedral_execution.inference.core.model_loader.qwen4.Qwen4Model;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.List;

/// Runs a loaded Flash-Next model one chunk of tokens at a time:
///
/// ```
/// tokens -> embedding rows -> repeated over 4 streams
///   -> layers 0..47: [PLE] mix -> GDN | QSA -> inject -> mix -> MoE -> inject
///   -> final mix of the last row -> output head -> logits
/// ```
///
/// One executor owns one stream and the workspace of a chunk of up to [#MAX_ROWS] tokens; a [Qwen4Sequence] carries
/// each sequence's state. Prefill chunks and decode steps are the same code over different row counts. A step queues
/// everything on the stream and waits for the device once per MoE block (the routing decides which experts to bring
/// in) and once at its end. The step runs uncaptured: its dynamic parts are the expert choice of every layer
/// (host-driven waves over the cache) and the position, see docs/FLASH_NEXT_EXECUTION.md.
public final class Qwen4Executor implements AutoCloseable {

    /// Most tokens of one chunk.
    public static final int MAX_ROWS = 512;

    /// Receives the logits row of a step: called while the step's stream is selected, after the output head's
    /// launch, so a copy it queues is ordered behind the head.
    @FunctionalInterface
    public interface LogitsSink {
        void queue(long logitsAddress);
    }

    /// Test hook: the residual state after each layer, read after the device finished it.
    public interface Observer {
        void layerFinished(int layer, long stateAddress, int rows) throws InterruptedException;
    }

    private final ExecutionGpu gpu;
    private final Qwen4Model model;
    private final Qwen4Config config;
    private final GpuStream stream;
    private final Qwen4Weights weights;
    private final Qwen4HyperConnection hyperConnection;
    private final Qwen4GdnLayer gdn;
    private final Qwen4Ple ple;
    private final Qwen4QsaLayer qsa;
    private final Qwen4MoeLayer moe;
    private final int pleLayer;
    private final int hidden;
    private final int streams;
    private final int vocabulary;
    private final int maxTokens;
    private final boolean[] sparse;
    private final Qwen4Weight embedding;
    private final Qwen4Weight head;

    private final long tokensDevice;
    private final ExecutionGpu.UploadBuffer tokenUpload;
    private final long embedded;
    private final long state;
    private final long pleOutput;
    private final long mixed;
    private final long blockOutput;
    private final long finalMixed;
    private final long logits;
    private final long hcScratchAddress;
    private final long layerScratchAddress;
    private final long moeScratchAddress;
    private final List<Long> allocations = new ArrayList<>();
    private Observer observer;
    private Observer midObserver;
    private boolean closed;

    /// @param maxContextTokens the longest sequence this executor's sequences hold
    public Qwen4Executor(ExecutionGpu gpu, Qwen4Model model, int maxContextTokens) {
        this.gpu = gpu;
        this.model = model;
        this.config = model.artifact().config();
        this.maxTokens = maxContextTokens;
        this.hidden = this.config.text().hiddenSize();
        this.streams = this.config.hyperConnection().count();
        this.vocabulary = this.config.text().vocabSize();
        this.stream = gpu.openStream();
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
        var banks = model.expertBanks();
        int slots = model.expertCache().slotCount();
        this.moe = new Qwen4MoeLayer(
                gpu,
                this.stream,
                model.expertCache(),
                Qwen4ExpertOps.Geometry.of(banks[0]),
                this.config.moe().numExperts(),
                this.config.moe().expertsPerToken(),
                this.config.moe().sharedExpertIntermediateSize(),
                MAX_ROWS,
                Math.min(slots, 32));
        this.embedding = this.weights.embedding();
        this.head = this.weights.head();
        this.tokenUpload = gpu.allocateUploadBuffer(4L * MAX_ROWS);
        try {
            long bf16 = Short.BYTES;
            long width = (long) this.streams * this.hidden;
            this.tokensDevice = allocate(4L * MAX_ROWS);
            this.embedded = allocate(MAX_ROWS * (long) this.hidden * bf16);
            this.state = allocate(MAX_ROWS * width * bf16);
            this.pleOutput = allocate(MAX_ROWS * width * bf16);
            this.mixed = allocate(MAX_ROWS * (long) this.hidden * bf16);
            this.blockOutput = allocate(MAX_ROWS * (long) this.hidden * bf16);
            this.finalMixed = allocate((long) this.hidden * bf16);
            this.logits = allocate((long) this.vocabulary * bf16);
            this.hcScratchAddress = allocate(this.hyperConnection.scratchBytes(MAX_ROWS));
            long layerBytes = Math.max(
                    Math.max(this.gdn.scratchBytes(MAX_ROWS), this.qsa.scratchBytes(MAX_ROWS)),
                    this.ple.scratchBytes(MAX_ROWS));
            this.layerScratchAddress = allocate(layerBytes);
            this.moeScratchAddress = allocate(this.moe.scratchBytes(MAX_ROWS));
        } catch (Throwable failure) {
            releaseBuffers();
            this.tokenUpload.close();
            this.moe.close();
            throw failure;
        }
    }

    private long allocate(long bytes) {
        long address = this.gpu.allocate(bytes);
        this.allocations.add(address);
        return address;
    }

    private void releaseBuffers() {
        for (long address : this.allocations) this.gpu.free(address);
        this.allocations.clear();
    }

    /// A new sequence of up to `maxContextTokens` positions (the executor's), at position 0.
    public Qwen4Sequence newSequence() {
        return new Qwen4Sequence(this.gpu, this.gdn, this.ple, this.sparse, this.qsaKeyValueWidth(), this.maxTokens);
    }

    private int qsaKeyValueWidth() {
        return this.config.attention().numKvHeads() * this.config.attention().headDim();
    }

    /// Takes the residual state of each layer's end for the observer (tests); null stops it.
    void observe(Observer observer) {
        this.observer = observer;
    }

    /// Takes the state after each layer's attention injection, before its MoE block (tests); null stops it.
    void observeMid(Observer observer) {
        this.midObserver = observer;
    }

    /// The device address of the residual state, `[rows][streams * hidden]` BF16 (tests).
    long stateAddress() {
        return this.state;
    }

    public int vocabularySize() {
        return this.vocabulary;
    }

    /// Runs one layer of a chunk on a residual state the caller supplies and returns the state after it (tests: each
    /// layer fed the reference's own input). The chunk's other layers run through their own calls at the same
    /// position; [#finishChunk] advances the sequence once all of them did.
    short[] runSingleLayer(Qwen4Sequence sequence, int layer, int[] tokens, int offset, int rows, short[] stateIn)
            throws InterruptedException {
        int width = this.hyperConnection.stateWidth();
        if (stateIn.length != rows * width) throw new IllegalArgumentException("state length");
        try (java.lang.foreign.Arena arena = java.lang.foreign.Arena.ofConfined()) {
            var host = arena.allocate((long) stateIn.length * Short.BYTES, 16);
            java.lang.foreign.MemorySegment.copy(stateIn, 0, host, ValueLayout.JAVA_SHORT, 0, stateIn.length);
            this.gpu.copyHostToDevice(this.state, host, host.byteSize());
            // A copy from pageable memory may return before its DMA ended, and the step's stream does not wait for it.
            this.gpu.synchronize();
            for (int i = 0; i < rows; i++)
                this.tokenUpload.segment().set(ValueLayout.JAVA_INT_UNALIGNED, 4L * i, tokens[offset + i]);
            List<ExecutionGpu.UploadBuffer> staged = new ArrayList<>();
            List<Qwen4QsaState> opened = new ArrayList<>();
            boolean finished = false;
            try {
                runLayer(sequence, layer, tokens, offset, rows, staged, opened);
                Qwen4Streams.awaitCompletion(this.stream);
                for (Qwen4QsaState attention : opened) attention.commit();
                opened.clear();
                finished = true;
            } finally {
                if (!finished) for (Qwen4QsaState attention : opened) attention.discard();
                for (ExecutionGpu.UploadBuffer upload : staged) upload.close();
            }
            var back = arena.allocate((long) stateIn.length * Short.BYTES, 16);
            this.gpu.copyDeviceToHost(back, this.state, back.byteSize());
            short[] out = new short[stateIn.length];
            java.lang.foreign.MemorySegment.copy(back, ValueLayout.JAVA_SHORT, 0, out, 0, out.length);
            return out;
        }
    }

    /// Advances `sequence` past a chunk whose layers ran through [#runSingleLayer].
    void finishChunk(Qwen4Sequence sequence, int rows) {
        sequence.advance(rows);
    }

    /// Runs `rows` tokens (`tokens[offset ..]`) at the sequence's position and advances it. With a `sink` the
    /// last row's logits are computed and offered to it. Returns after the device finished the step.
    public void step(Qwen4Sequence sequence, int[] tokens, int offset, int rows, LogitsSink sink)
            throws InterruptedException {
        if (this.closed) throw new IllegalStateException("the executor is closed");
        if (rows <= 0 || rows > MAX_ROWS) throw new IllegalArgumentException("rows " + rows);
        if (sequence.position() + rows > sequence.maxTokens())
            throw new IllegalStateException("the sequence would exceed its " + sequence.maxTokens() + " positions");
        long start = sequence.position();
        List<ExecutionGpu.UploadBuffer> staged = new ArrayList<>();
        List<Qwen4QsaState> opened = new ArrayList<>();
        boolean finished = false;
        try {
            for (int i = 0; i < rows; i++)
                this.tokenUpload.segment().set(ValueLayout.JAVA_INT_UNALIGNED, 4L * i, tokens[offset + i]);
            this.stream.submit(
                    () -> {
                        this.gpu.copyUploadToDevice(this.tokensDevice, this.tokenUpload);
                        Qwen4Ops.embedding(
                                this.gpu,
                                this.embedding.address(),
                                this.tokensDevice,
                                this.embedded,
                                rows,
                                this.hidden,
                                this.vocabulary);
                        Qwen4Ops.repeatStreams(this.gpu, this.embedded, this.state, rows, this.streams, this.hidden);
                    },
                    false);
            for (int layer = 0; layer < this.sparse.length; layer++) {
                runLayer(sequence, layer, tokens, offset, rows, staged, opened);
                if (this.observer != null) {
                    Qwen4Streams.awaitCompletion(this.stream);
                    this.observer.layerFinished(layer, this.state, rows);
                }
            }
            if (sink != null) {
                this.stream.submit(
                        () -> {
                            long last =
                                    this.state + (long) (rows - 1) * this.hyperConnection.stateWidth() * Short.BYTES;
                            this.hyperConnection.mix(
                                    this.gpu,
                                    this.weights.finalMixer(),
                                    last,
                                    this.hyperConnection.scratch(this.hcScratchAddress, 1),
                                    this.finalMixed,
                                    1);
                            Qwen4Ops.linearBf16(
                                    this.gpu,
                                    this.finalMixed,
                                    this.head.address(),
                                    this.logits,
                                    1,
                                    this.hidden,
                                    this.vocabulary);
                            sink.queue(this.logits);
                        },
                        false);
            }
            Qwen4Streams.awaitCompletion(this.stream);
            for (Qwen4QsaState attention : opened) attention.commit();
            opened.clear();
            sequence.advance(rows);
            finished = true;
        } finally {
            if (!finished) {
                for (Qwen4QsaState attention : opened) attention.discard();
                try {
                    Qwen4Streams.awaitCompletion(this.stream);
                } catch (RuntimeException failure) {
                    // The device failure surfaces from the exception already propagating.
                }
            }
            for (ExecutionGpu.UploadBuffer upload : staged) upload.close();
        }
        if (start < 0) throw new IllegalStateException();
    }

    /// One decoder layer over the state buffer.
    private void runLayer(
            Qwen4Sequence sequence,
            int layer,
            int[] tokens,
            int offset,
            int rows,
            List<ExecutionGpu.UploadBuffer> staged,
            List<Qwen4QsaState> opened)
            throws InterruptedException {
        long hcAddress = this.hcScratchAddress;
        var hc = this.hyperConnection.scratch(hcAddress, rows);
        long start = sequence.position();
        this.stream.submit(
                () -> {
                    if (layer == this.pleLayer) {
                        var scratch = this.ple.scratch(this.layerScratchAddress, rows);
                        staged.add(this.ple.apply(
                                this.gpu,
                                this.weights.ple(layer),
                                sequence.ple(),
                                tokens,
                                offset,
                                rows,
                                this.state,
                                scratch,
                                this.pleOutput));
                        this.gpu.residualAddBf16(
                                this.state, this.pleOutput, this.state, rows, this.hyperConnection.stateWidth());
                    }
                    this.hyperConnection.mix(
                            this.gpu, this.weights.attentionResidual(layer), this.state, hc, this.mixed, rows);
                    if (this.sparse[layer]) {
                        Qwen4QsaState attention = sequence.qsa(layer);
                        opened.add(attention);
                        this.qsa.run(
                                this.gpu,
                                this.weights.qsa(layer),
                                attention,
                                this.mixed,
                                rows,
                                this.blockOutput,
                                this.qsa.scratch(this.layerScratchAddress, rows),
                                0);
                    } else {
                        this.gdn.run(
                                this.gpu,
                                this.weights.gdn(layer),
                                sequence.gdn(layer),
                                this.mixed,
                                rows,
                                this.gdn.scratch(this.layerScratchAddress, rows),
                                this.blockOutput);
                    }
                    this.hyperConnection.inject(this.gpu, this.state, this.blockOutput, hc, this.state, rows);
                    this.hyperConnection.mix(
                            this.gpu, this.weights.moeResidual(layer), this.state, hc, this.mixed, rows);
                },
                false);
        if (this.midObserver != null) {
            Qwen4Streams.awaitCompletion(this.stream);
            this.midObserver.layerFinished(layer, this.state, rows);
        }
        this.moe.run(
                this.weights.moe(layer),
                this.model.bankOrdinal("text/layers/" + layer + "/moe/experts"),
                this.mixed,
                rows,
                this.moe.scratch(this.moeScratchAddress, rows),
                this.blockOutput);
        this.stream.submit(
                () -> this.hyperConnection.inject(this.gpu, this.state, this.blockOutput, hc, this.state, rows), false);
    }

    @Override
    public void close() {
        if (this.closed) return;
        this.closed = true;
        try {
            Qwen4Streams.awaitCompletion(this.stream);
        } catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
        }
        this.moe.close();
        this.tokenUpload.close();
        releaseBuffers();
        this.stream.close();
    }
}
