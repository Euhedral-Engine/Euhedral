package io.euhedral_execution.inference.core.model.qwen38;

import io.euhedral_execution.inference.core.generation.DeviceLogits;
import io.euhedral_execution.inference.core.generation.HostLogits;
import io.euhedral_execution.inference.core.generation.LogitsRequirement;
import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.gpu.GpuStream;
import io.euhedral_execution.inference.core.model.qwen38.prefix.PromptCheckpoints;
import io.euhedral_execution.inference.core.model.qwen38.speculative.DFlash2Proposal;
import io.euhedral_execution.inference.core.model.qwen38.speculative.SpeculativeAcceptance;
import io.euhedral_execution.inference.core.runtime.graph.AbstractQuantum;
import io.euhedral_execution.inference.core.runtime.graph.CaptureFingerprint;
import io.euhedral_execution.inference.core.runtime.graph.SequenceOwner;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.CancellationException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Consumer;

/// Mutable state for one inference quantum: its token range, its admission to the sequence, workspace, and outcome.
///
/// While it runs, the quantum is bound to a reusable stage graph whose frames perform the operations.
/// Its sequence reference retains sequence-lifetime state, which is published only at retirement.
public final class Quantum extends AbstractQuantum implements Sequence.Work {

    /// Decode quanta that start below this position overlap registered kernels with their predecessor
    /// (programmatic dependent launch). It won 12 of 12 paired forks, about +1% decode, at a 64-token
    /// context and only 10 of 12 at 1024, so longer contexts keep ordinary launches.
    static final long OVERLAP_MAX_START_POSITION = 1024;

    private static final Runnable NO_OP = () -> {};

    public enum ExecutionKind {
        PREFILL,
        DECODE,
        /// Speculative verification: decode's topology over several rows, each computed bit for bit as
        /// one-row decode at its position (row-exact execution), with logits for every row.
        VERIFY,
        /// Drafting, which leaves the base sequence position where it is: rows of the MTP layer at MTP positions,
        /// writing only the MTP layer's cache (docs/MTP_CONTRACT.md §2), or a DFlash2 block, which proposes the
        /// tokens after its anchor and writes no state (docs/DFLASH2.md).
        DRAFT,
        /// DFlash2 context rows: the drafter's keys and values of committed target rows at their positions, from the
        /// target's tapped hidden rows, into the drafter's own cache. The base sequence position does not move.
        DRAFT_CONTEXT
    }

    public enum Status {
        SUCCESS,
        CANCELLED,
        FAILED
    }

    public record Outcome(Status status, Throwable failure) {}

    static final class DuplicateAdmissionException extends IllegalStateException {
        DuplicateAdmissionException() {
            super("quantum was already submitted");
        }
    }

    private final ExecutionPlan plan;
    /// The view of `plan` this quantum runs.
    private final Shape shape;
    private final Sequence sequence;
    private final ExecutionKind kind;
    private final LogitsRequirement logitsRequirement;
    private final HostLogits hostLogits;
    private final long startPosition;
    private final int[] tokenIds;
    private final AtomicBoolean submitted = new AtomicBoolean();
    /// Whether the sequence admitted this quantum; only an admitted quantum retires the sequence's work.
    private boolean admitted;
    /// The position the sequence admitted this quantum at.
    private long admittedAt;
    /// Whether its device work retired, so that it may conclude once the quanta admitted before it concluded.
    private volatile boolean ready;
    /// The conclusion it runs in the sequence's admission order.
    private Runnable pendingConclusion;
    private Workspace workspace;
    /// The pinned staging of the quantum's input record, held until its device work retired.
    private ExecutionGpu.UploadBuffer inputUpload;
    private DeviceLogits logitsOutput;
    private ExecutionGpu gpu;
    private Consumer<? super Quantum> terminalConsumer;
    private volatile Outcome conclusion;
    /// DRAFT: the device rows of base (or MTP) hidden state that seed this quantum's MTP rows, and how
    /// many of its rows the MTP cache commits (catch-up rows commit, recursive draft rows do not).
    private long draftSeedAddress;
    /// DRAFT seeded from the sequence's draft seed rows: the first of them, read when its stage runs; else -1.
    private int draftSeedRow = -1;
    private int draftCommittedRows;
    /// Base quanta of a speculative session also keep every row's post-final-norm hidden for drafting.
    private boolean seedsDraft;
    /// A DFlash2 block's host copy of its proposal, queued by its selector stage.
    private DFlash2Proposal proposal;
    /// The quantum's rows in chunks: a prompt's chunks, or one chunk of all its rows.
    private final Chunk[] chunks;
    /// A prompt's prefix checkpoints, and for each chunk the checkpoint copied after it (-1: none).
    private PromptCheckpoints checkpoints;
    private int[] checkpointOf;

    /// One chunk of a quantum's rows: its place in the quantum, its first position, its rows, and the offset of its
    /// rows in the quantum's input record. Only the last chunk produces logits.
    public record Chunk(int index, long startPosition, int rows, int inputOffset, boolean last) {
        /// The logits rows this chunk produces under `requirement`.
        public int logitsRows(LogitsRequirement requirement) {
            return this.last ? requirement.outputRows(this.rows) : 0;
        }
    }

    public Quantum(ExecutionPlan plan, Sequence sequence, ExecutionKind kind, long startPosition, int[] tokenIds) {
        this(plan, sequence, kind, startPosition, tokenIds, LogitsRequirement.ALL_TOKENS);
    }

    public Quantum(
            ExecutionPlan plan,
            Sequence sequence,
            ExecutionKind kind,
            long startPosition,
            int[] tokenIds,
            LogitsRequirement logitsRequirement) {
        this(plan, sequence, kind, startPosition, tokenIds, logitsRequirement, null);
    }

    /// A quantum whose caller samples its final logits row on the host. The row is copied into
    /// `hostLogits` on the quantum's stream before retirement, and the device logits stay in the
    /// executing graph's storage; [#logitsOutput] is then empty.
    public Quantum(
            ExecutionPlan plan,
            Sequence sequence,
            ExecutionKind kind,
            long startPosition,
            int[] tokenIds,
            LogitsRequirement logitsRequirement,
            HostLogits hostLogits) {
        this.plan = Objects.requireNonNull(plan, "plan");
        this.shape = plan.forExecution(kind, Objects.requireNonNull(tokenIds, "tokenIds").length);
        this.logitsRequirement = Objects.requireNonNull(logitsRequirement, "logitsRequirement");
        if (hostLogits != null) {
            if (logitsRequirement == LogitsRequirement.NONE)
                throw new IllegalArgumentException("host logits require a logits row");
            int vocabulary = kind == ExecutionKind.DRAFT
                    ? this.plan.draftVocabularySize()
                    : this.plan.weights().config().vocabSize();
            if (hostLogits.vocabularySize() != vocabulary)
                throw new IllegalArgumentException("host logits do not match the model vocabulary");
        }
        this.hostLogits = hostLogits;
        this.sequence = Objects.requireNonNull(sequence, "sequence");
        this.kind = Objects.requireNonNull(kind, "kind");
        Objects.requireNonNull(tokenIds, "tokenIds");
        if (startPosition < 0 || tokenIds.length == 0) {
            throw new IllegalArgumentException("invalid token range");
        }
        this.startPosition = startPosition;
        this.tokenIds = tokenIds.clone();
        this.chunks = new Chunk[] {new Chunk(0, startPosition, tokenIds.length, 0, true)};
    }

    /// A prompt quantum: `tokens` from `startPosition` in chunks of `chunkRows` rows (the remainder last), run as one
    /// graph whose chunks are copies of the view a chunk selects. Only the last chunk produces logits. The graph
    /// binds one view's workspace, so its last chunk must select the same view as the others: a shorter remainder
    /// that selects another (the small prefill view) runs as a quantum of its own.
    static Quantum prompt(
            ExecutionPlan plan,
            Sequence sequence,
            long startPosition,
            int[] tokens,
            int chunkRows,
            LogitsRequirement last,
            HostLogits hostLogits) {
        if (chunkRows <= 0) throw new IllegalArgumentException("chunkRows must be positive");
        int rest = tokens.length % chunkRows;
        if (tokens.length > chunkRows
                && rest != 0
                && plan.forExecution(ExecutionKind.PREFILL, rest)
                        != plan.forExecution(ExecutionKind.PREFILL, chunkRows))
            throw new IllegalArgumentException(
                    "a prompt's last chunk of " + rest + " rows selects another view than its chunks of " + chunkRows);
        return new Quantum(plan, sequence, startPosition, tokens, chunkRows, last, hostLogits);
    }

    private Quantum(
            ExecutionPlan plan,
            Sequence sequence,
            long startPosition,
            int[] tokens,
            int chunkRows,
            LogitsRequirement last,
            HostLogits hostLogits) {
        this.plan = Objects.requireNonNull(plan, "plan");
        Objects.requireNonNull(tokens, "tokens");
        if (startPosition < 0 || tokens.length == 0) throw new IllegalArgumentException("invalid token range");
        this.kind = ExecutionKind.PREFILL;
        this.shape = plan.forExecution(ExecutionKind.PREFILL, Math.min(chunkRows, tokens.length));
        this.logitsRequirement = Objects.requireNonNull(last, "last");
        if (last == LogitsRequirement.ALL_TOKENS)
            throw new IllegalArgumentException("a prompt quantum produces at most its last chunk's logits");
        if (hostLogits != null && last == LogitsRequirement.NONE)
            throw new IllegalArgumentException("host logits require a logits row");
        this.hostLogits = hostLogits;
        this.sequence = Objects.requireNonNull(sequence, "sequence");
        this.startPosition = startPosition;
        this.tokenIds = tokens.clone();
        int count = (tokens.length + chunkRows - 1) / chunkRows;
        this.chunks = new Chunk[count];
        for (int index = 0; index < count; index++) {
            int offset = index * chunkRows;
            int rows = Math.min(chunkRows, tokens.length - offset);
            this.chunks[index] = new Chunk(index, startPosition + offset, rows, offset, index == count - 1);
        }
    }

    /// Takes `checkpoints` inside the prompt graph: after each chunk that ends at a checkpoint the cache reserved,
    /// a stage copies the sequence's state into it. Before submission only.
    Quantum withCheckpoints(PromptCheckpoints checkpoints) {
        if (this.submitted.get()) throw new IllegalStateException("the quantum was already submitted");
        this.checkpoints = Objects.requireNonNull(checkpoints, "checkpoints");
        this.checkpointOf = new int[this.chunks.length];
        java.util.Arrays.fill(this.checkpointOf, -1);
        for (int index = 0; index < checkpoints.count(); index++) {
            if (!checkpoints.copies(index)) continue;
            for (Chunk chunk : this.chunks)
                if (chunk.startPosition() + chunk.rows() == checkpoints.position(index))
                    this.checkpointOf[chunk.index()] = index;
        }
        return this;
    }

    /// The chunks after which the prompt graph copies a checkpoint, ascending; empty without checkpoints.
    java.util.List<Integer> checkpointChunks() {
        java.util.List<Integer> chunks = new java.util.ArrayList<>();
        if (this.checkpointOf != null)
            for (int index = 0; index < this.checkpointOf.length; index++)
                if (this.checkpointOf[index] >= 0) chunks.add(index);
        return chunks;
    }

    /// Queues the copies of the checkpoint taken after chunk `chunk` ([Stages.Checkpoint]).
    void checkpointAfter(int chunk, ExecutionGpu gpu) {
        int index = this.checkpointOf == null ? -1 : this.checkpointOf[chunk];
        if (index >= 0) this.checkpoints.copy(index, this.sequence, gpu);
    }

    /// Chunk `index` of the quantum's rows.
    public Chunk chunk(int index) {
        return this.chunks[index];
    }

    public int chunkCount() {
        return this.chunks.length;
    }

    /// The most rows one chunk has: what the workspace holds at a time.
    public int chunkRows() {
        return this.chunks[0].rows();
    }

    /// The plan whose weights this quantum runs.
    public ExecutionPlan plan() {
        return this.plan;
    }

    /// The view of [#plan] this quantum runs.
    public Shape shape() {
        return this.shape;
    }

    public Sequence sequenceState() {
        return this.sequence;
    }

    public int inputTokenCount() {
        return this.tokenIds.length;
    }

    public LogitsRequirement logitsRequirement() {
        return this.logitsRequirement;
    }

    public int logitsRowCount() {
        return this.logitsRequirement.outputRows(this.tokenIds.length);
    }

    public long startPosition() {
        return this.startPosition;
    }

    public int[] inputTokenIds() {
        return this.tokenIds.clone();
    }

    public Workspace workspace() {
        Workspace current = this.workspace;
        if (current == null) {
            throw new IllegalStateException("workspace has not been allocated");
        }
        return current;
    }

    /// Returns GPU-resident logits after successful completion; the caller owns and must close them.
    /// Empty for a quantum that samples on the host.
    public Optional<DeviceLogits> logitsOutput() {
        return Optional.ofNullable(this.logitsOutput);
    }

    /// Called by the stage that produced this quantum's logits rows at `address`, with the quantum's
    /// stream selected. A host-sampling quantum queues the copy of its final row here, ahead of its
    /// retirement boundary.
    /// A DRAFT quantum whose MTP rows are seeded by the BF16 hidden rows at `seedAddress`, committing its
    /// first `committedRows` MTP cache rows.
    public Quantum withDraftSeed(long seedAddress, int committedRows) {
        if (this.kind != ExecutionKind.DRAFT) throw new IllegalStateException("only drafting takes a seed");
        if (seedAddress == 0 || committedRows < 0 || committedRows > this.tokenIds.length)
            throw new IllegalArgumentException("invalid draft seed");
        this.draftSeedAddress = seedAddress;
        this.draftCommittedRows = committedRows;
        return this;
    }

    /// A DRAFT quantum whose MTP rows are seeded by the sequence's draft seed rows from `firstRow`, committing its
    /// first `committedRows` MTP cache rows. The rows are found when its stage runs, after the seeding quantum's
    /// writer (the carried seed-row edge): so a catch-up can be admitted before that quantum ran.
    public Quantum withSeedRows(int firstRow, int committedRows) {
        if (this.kind != ExecutionKind.DRAFT) throw new IllegalStateException("only drafting takes a seed");
        if (firstRow < 0 || committedRows < 0 || committedRows > this.tokenIds.length)
            throw new IllegalArgumentException("invalid draft seed");
        this.draftSeedRow = firstRow;
        this.draftCommittedRows = committedRows;
        return this;
    }

    /// The seed rows' device address; for [#withSeedRows], as the sequence's seed rows are now.
    public long draftSeedAddress() {
        if (this.draftSeedRow < 0) return this.draftSeedAddress;
        var states = (AttentionStates) this.sequence.kvCacheState();
        int hidden = this.plan.weights().config().hiddenSize();
        return states.seedRows() + (long) this.draftSeedRow * hidden * Short.BYTES;
    }

    /// Keeps every row's post-final-norm hidden in the sequence's draft seed rows.
    public Quantum seedingDraft() {
        if (drafting()) throw new IllegalStateException("drafting does not seed itself");
        this.seedsDraft = true;
        return this;
    }

    public boolean seedsDraft() {
        return this.seedsDraft;
    }

    /// Binds the host copy that a DFlash2 block quantum's proposal is queued into before it retires.
    public Quantum withProposal(DFlash2Proposal proposal) {
        if (this.kind != ExecutionKind.DRAFT) throw new IllegalStateException("only a draft block proposes tokens");
        this.proposal = Objects.requireNonNull(proposal, "proposal");
        return this;
    }

    public DFlash2Proposal proposal() {
        return this.proposal;
    }

    /// Whether this quantum leaves the base sequence position where it is (MTP rows, DFlash2 blocks and context).
    boolean drafting() {
        return this.kind == ExecutionKind.DRAFT || this.kind == ExecutionKind.DRAFT_CONTEXT;
    }

    /// The acceptance rule of a VERIFY quantum; null commits every row (tests may force a count).
    private SpeculativeAcceptance acceptance;
    private int forcedCommittedRows;

    /// Binds the acceptance of a VERIFY quantum before submission.
    public Quantum withAcceptance(SpeculativeAcceptance acceptance) {
        if (this.kind != ExecutionKind.VERIFY) throw new IllegalStateException("only verification resolves acceptance");
        this.acceptance = Objects.requireNonNull(acceptance, "acceptance");
        return this;
    }

    /// Commits only the first `rows` verified rows regardless of the selections (state tests).
    Quantum withCommittedRows(int rows) {
        if (this.kind != ExecutionKind.VERIFY || rows <= 0 || rows > this.tokenIds.length)
            throw new IllegalArgumentException("invalid committed row count");
        this.forcedCommittedRows = rows;
        return this;
    }

    public SpeculativeAcceptance acceptance() {
        return this.acceptance;
    }

    /// Rows whose state this quantum commits: all rows, or a verification's accepted prefix. Resolved on
    /// first use after the device work retired, from the verified rows' greedy selections.
    public int committedRowCount() {
        if (drafting()) return this.draftCommittedRows;
        if (this.forcedCommittedRows > 0) return this.forcedCommittedRows;
        if (this.acceptance == null) return this.tokenIds.length;
        if (!this.acceptance.resolved()) this.acceptance.resolve(this.hostLogits.selectedTokens());
        return this.acceptance.committedRows();
    }

    public void logitsProduced(long address) {
        if (this.hostLogits == null) return;
        if (this.kind == ExecutionKind.VERIFY) this.hostLogits.queueRowSelections(address, logitsRowCount());
        else this.hostLogits.queueFinalRow(address, logitsRowCount());
    }

    @Override
    public void cancel() {
        this.sequence.cancel();
    }

    /// Reports whether this quantum must stop admitting dependent instructions.
    public boolean hasFailureOrCancellation() {
        return failure() != null || this.sequence.cancellationRequested();
    }

    @Override
    public boolean stopRequested() {
        return hasFailureOrCancellation();
    }

    @Override
    public boolean overlapLaunches() {
        return (this.kind == ExecutionKind.DECODE || this.kind == ExecutionKind.VERIFY)
                && this.startPosition < OVERLAP_MAX_START_POSITION;
    }

    /// Quanta of at most this many rows are captured: decode, verification and draft quanta.
    static final int MAX_CAPTURED_ROWS = 8;

    /// Decode, verification and draft quanta are captured (docs/CUDA_GRAPHS.md). The key holds what their
    /// stages' submissions depend on apart from the input record: the kind, rows and outputs, the launch
    /// geometry of decode attention, and a fingerprint of the workspace and
    /// sequence-owned device addresses. A quantum whose reservation would allocate KV pages or upload a page
    /// table is not captured; neither are quanta under exact numerics or with host-staged weights.
    @Override
    public Object captureKey() {
        if (this.kind == ExecutionKind.PREFILL || this.tokenIds.length > MAX_CAPTURED_ROWS) return null;
        if (this.workspace == null || this.gpu == null || this.gpu.exactNumerics()) return null;
        // Its seed rows are found only when its stage runs.
        if (this.draftSeedRow >= 0) return null;
        if (!this.shape.hasFirstLayer()) return null;
        if (!(this.sequence.kvCacheState() instanceof AttentionStates attention)
                || !(this.sequence.recurrentState() instanceof GdnStates recurrent)) return null;
        // A DFlash2 quantum appends to no paged cache; its ring is fixed.
        boolean dflash = drafting() && this.plan.draftsWithDFlash2();
        if (!dflash && !attention.reserves(this.startPosition, this.tokenIds.length, this.kind == ExecutionKind.DRAFT))
            return null;
        CaptureFingerprint fingerprint = new CaptureFingerprint();
        this.workspace.fingerprint(fingerprint);
        attention.fingerprint(fingerprint);
        recurrent.fingerprint(fingerprint);
        if (this.hostLogits != null) this.hostLogits.fingerprint(fingerprint);
        if (this.proposal != null) this.proposal.fingerprint(fingerprint);
        Qwen38Config config = this.plan.weights().config();
        return new CaptureKey(new long[] {
            this.kind.ordinal(),
            this.tokenIds.length,
            this.logitsRequirement.ordinal(),
            this.hostLogits == null ? 0 : 1,
            this.seedsDraft ? 1 : 0,
            this.draftSeedAddress,
            this.draftCommittedRows,
            attentionGeometry(config.numAttentionHeads() / config.numKeyValueHeads()),
            fingerprint.value()
        });
    }

    /// The split counts that size decode attention's launches (host dispatch in qwen_layer_ops.c): rows
    /// below 2048 keys split into 48-key spans, rows from 2048 keys (query-head groups up to 8) into 32-key
    /// spans, at most 64 each. Every other position-dependent value is read from the input record.
    private long attentionGeometry(int group) {
        long first = this.startPosition + 1, last = this.startPosition + this.tokenIds.length;
        long from = group <= 8 ? 2048 : Long.MAX_VALUE;
        long below = first < from ? Math.min(64, (Math.min(last, from - 1) + 47) / 48) : 0;
        long above = last >= from ? Math.min(64, (last + 31) / 32) : 0;
        return below << 8 | above;
    }

    private static final class CaptureKey {
        private final long[] words;
        private final int hash;

        CaptureKey(long[] words) {
            this.words = words;
            this.hash = java.util.Arrays.hashCode(words);
        }

        @Override
        public boolean equals(Object other) {
            return other instanceof CaptureKey key && java.util.Arrays.equals(this.words, key.words);
        }

        @Override
        public int hashCode() {
            return this.hash;
        }
    }

    public ExecutionKind kind() {
        return this.kind;
    }

    /// Claims this quantum's single admission. A claimed quantum always reaches a terminal outcome, so
    /// a failed admission can never be retried into a second sequence admission or workspace.
    public void claim() {
        if (!this.submitted.compareAndSet(false, true)) {
            throw new DuplicateAdmissionException();
        }
    }

    /// Package-private hook to deterministically exercise cancellation at the sequence-admission boundary.
    void begin(ExecutionGpu gpu, Runnable beforeClaim) {
        claim();
        if (!begin(gpu, null, null, new WorkspaceStorage(gpu), null, beforeClaim)) concludeUnstarted();
    }

    /// Joins the sequence's admission order and binds the executing graph's `storage` with `stream` selected, so
    /// any initialization it queues precedes every stage of the quantum. Returns whether the quantum proceeds to
    /// its stages. Otherwise the caller concludes it ([#concludeUnstarted()]) once the stream is no longer
    /// selected.
    public boolean begin(
            ExecutionGpu gpu, GpuStream stream, Consumer<? super Quantum> terminalConsumer, WorkspaceStorage storage) {
        return begin(gpu, stream, terminalConsumer, storage, null, NO_OP);
    }

    /// As [#begin(ExecutionGpu, GpuStream, Consumer, WorkspaceStorage)], binding the runtime's `shared` workspace
    /// (the graph's `storage` keeps only the input record and the logits).
    public boolean begin(
            ExecutionGpu gpu,
            GpuStream stream,
            Consumer<? super Quantum> terminalConsumer,
            WorkspaceStorage storage,
            SharedWorkspace shared) {
        return begin(gpu, stream, terminalConsumer, storage, shared, NO_OP);
    }

    private boolean begin(
            ExecutionGpu gpu,
            GpuStream stream,
            Consumer<? super Quantum> terminalConsumer,
            WorkspaceStorage storage,
            SharedWorkspace shared,
            Runnable beforeClaim) {
        this.gpu = Objects.requireNonNull(gpu, "gpu");
        Objects.requireNonNull(storage, "storage");
        this.terminalConsumer = terminalConsumer;
        if (this.sequence.cancellationRequested()) {
            // Cancelled before it began: it concludes without a stage.
            return false;
        }
        try {
            long end = Math.addExact(this.startPosition, this.tokenIds.length);
            if (end < 0) {
                throw new IllegalArgumentException("token range overflows");
            }
            if (shared != null && chunkRows() > shared.maxRows())
                throw new IllegalArgumentException("a quantum of " + chunkRows()
                        + " rows exceeds the workspace sized at load for " + shared.maxRows());
            int vocabulary = this.plan.weights().config().vocabSize();
            for (int index = 0; index < this.tokenIds.length; index++) {
                if (this.tokenIds[index] < 0 || this.tokenIds[index] >= vocabulary) {
                    throw new IllegalArgumentException("token ID outside vocabulary at index " + index);
                }
            }
            beforeClaim.run();
            // The sequence takes several quanta in flight, but not the state no edge orders: the host row and the
            // proposal. A quantum that touches it is refused while other work on the sequence is in flight.
            if (sharesSequenceState() && this.sequence.inFlight())
                throw new IllegalStateException(
                        "another quantum of the sequence is in flight, and its state takes one quantum at a time");
            try {
                // Drafting runs at MTP positions and leaves the frontiers where they are.
                if (drafting()) {
                    this.admittedAt = this.sequence.admitAtFrontier(this);
                } else {
                    this.admittedAt = this.startPosition;
                    this.sequence.admit(this, this.startPosition, end);
                }
                this.admitted = true;
            } catch (IllegalStateException admissionFailure) {
                if (this.sequence.cancellationRequested()
                        || this.sequence.terminalState() == Sequence.TerminalState.CANCELLED) {
                    return false;
                }
                throw admissionFailure;
            }
            initializeSequenceState(gpu);
            reservePrompt();
            int hidden = this.plan.weights().config().hiddenSize();
            if (shared == null)
                this.workspace = this.shape.hasFirstLayer()
                        ? new Workspace(storage, chunkRows(), this.shape, this.logitsRequirement)
                                .withInput(this.tokenIds.length, this.chunks.length)
                        : new Workspace(storage, chunkRows(), hidden, this.shape.projectionWidths())
                                .withInput(this.tokenIds.length, this.chunks.length);
            else
                this.workspace = this.shape.hasFirstLayer()
                        ? Workspace.bound(shared, storage, chunkRows(), this.shape, this.logitsRequirement)
                                .withInput(this.tokenIds.length, this.chunks.length)
                        : Workspace.bound(shared, storage, chunkRows(), hidden, this.shape.projectionWidths())
                                .withInput(this.tokenIds.length, this.chunks.length);
            this.workspace.allocateBuffers();
            uploadInput(gpu);
            return true;
        } catch (RuntimeException | Error error) {
            // Initialization can queue zeroes before a later allocation fails. Keep every allocation
            // when the stream cannot prove that those writes stopped.
            if (stream != null) stream.recover(error);
            fail(error);
            return false;
        }
    }

    /// Queues the input record (token IDs and start position) ahead of every stage of the quantum.
    private void uploadInput(ExecutionGpu gpu) {
        ExecutionGpu.UploadBuffer upload = gpu.allocateUploadBuffer(this.workspace.inputByteSize());
        this.inputUpload = upload;
        long[] positions = new long[this.chunks.length];
        for (int index = 0; index < positions.length; index++) positions[index] = this.chunks[index].startPosition();
        this.workspace.writeInput(upload.segment(), this.tokenIds, positions);
        gpu.copyUploadToDevice(this.workspace.tokenIdsAddress(), upload);
    }

    /// A poisoned GPU cannot prove that DMA has stopped reading pinned host memory.
    private void releaseInputUpload() {
        ExecutionGpu.UploadBuffer upload = this.inputUpload;
        this.inputUpload = null;
        if (upload != null && this.gpu.completionProven()) upload.close();
    }

    /// A prompt quantum reserves every row's KV pages and table now, with its stream selected, so none of its
    /// chunks grows a table; each chunk's append then continues the previous one's.
    private void reservePrompt() {
        if (this.chunks.length == 1 || !this.shape.hasFirstLayer()) return;
        var attention = (AttentionStates) this.sequence.kvCacheState();
        LayerType[] layers = this.plan.weights().config().layerTypes();
        for (int layer = 0; layer < layers.length; layer++)
            if (layers[layer] == LayerType.FULL_ATTENTION)
                attention.forLayer(layer).prepareAppend(this.startPosition, this.tokenIds.length);
    }

    private void initializeSequenceState(ExecutionGpu gpu) {
        attachSequenceState(this.shape, this.sequence, gpu);
    }

    /// Gives `sequence` the persistent state a first quantum allocates (GDN buffers and KV pages), or checks
    /// the state it already has. Runs while the sequence has its quantum in flight.
    public static void attachSequenceState(ExecutionPlan plan, Sequence sequence, ExecutionGpu gpu) {
        attachSequenceState(plan.shape(), sequence, gpu);
    }

    /// As [#attachSequenceState(ExecutionPlan, Sequence, ExecutionGpu)] for the view `shape`: a view without a
    /// first layer leaves the sequence's state alone.
    static void attachSequenceState(Shape shape, Sequence sequence, ExecutionGpu gpu) {
        if (!shape.hasFirstLayer()) return;
        ExecutionPlan plan = shape.plan();
        Object current = sequence.recurrentState();
        Object currentKv = sequence.kvCacheState();
        Qwen38Config config = plan.weights().config();
        if (current == null) {
            AutoCloseable createdRecurrent = null;
            AutoCloseable createdKv = null;
            try {
                if (plan.weights().layers().length > 1) {
                    createdRecurrent = GdnStates.allocate(
                            gpu,
                            config.layerTypes(),
                            config.linearNumKeyHeads(),
                            config.linearNumValueHeads(),
                            config.linearKeyHeadDim(),
                            config.linearValueHeadDim(),
                            config.linearConvKernelDim());
                    createdKv = AttentionStates.allocate(
                            gpu,
                            config.layerTypes(),
                            config.numKeyValueHeads() * config.attentionHeadDim(),
                            plan.weights().mtp() != null);
                } else {
                    createdRecurrent = GdnState.allocate(
                            gpu,
                            config.linearNumKeyHeads(),
                            config.linearNumValueHeads(),
                            config.linearKeyHeadDim(),
                            config.linearValueHeadDim(),
                            config.linearConvKernelDim());
                }
                sequence.setRecurrentState(createdRecurrent);
                if (createdKv != null) sequence.setKvCacheState(createdKv);
            } catch (RuntimeException | Error attachmentFailure) {
                closeCreatedState(createdKv, attachmentFailure);
                closeCreatedState(createdRecurrent, attachmentFailure);
                throw attachmentFailure;
            }
        } else if (plan.weights().layers().length > 1) {
            if (!(current instanceof GdnStates)) {
                throw new IllegalStateException("sequence already owns incompatible full-model GDN state");
            }
            if (!(currentKv instanceof AttentionStates)) {
                throw new IllegalStateException("sequence already owns incompatible full-attention KV state");
            }
        } else if (!(current instanceof GdnState)) {
            throw new IllegalStateException("sequence already owns incompatible recurrent state");
        }
    }

    private static void closeCreatedState(AutoCloseable state, Throwable failure) {
        if (state == null) return;
        try {
            state.close();
        } catch (Exception cleanupFailure) {
            failure.addSuppressed(cleanupFailure);
        }
    }

    /// Whether the quantum touches per-sequence state no edge orders: the session's host row, or a proposal. Its
    /// layers' GDN and KV state, the MTP cache, the draft seed rows and the DFlash2 taps and ring are carried
    /// state, which the edges between the sequence's graphs order ([SequenceOwner]), so a quantum that touches
    /// only those may be admitted while another of the sequence is in flight.
    private boolean sharesSequenceState() {
        return this.hostLogits != null || this.proposal != null;
    }

    @Override
    public SequenceOwner sequenceOwner() {
        return this.sequence.owner();
    }

    /// A quantum is cancelled through its sequence.
    @Override
    protected boolean cancelRequested() {
        return this.sequence.cancellationRequested();
    }

    /// A quantum the sequence admitted concludes in the sequence's admission order: it marks itself ready and
    /// drains the sequence, which runs `conclusion` once every quantum admitted before it concluded, on whichever
    /// thread finds it ready at the head. Any other quantum concludes now.
    @Override
    public void concludeInOrder(Runnable conclusion) {
        if (!this.admitted) {
            conclusion.run();
            return;
        }
        this.pendingConclusion = conclusion;
        this.ready = true;
        this.sequence.drain();
    }

    @Override
    public boolean ready() {
        return this.ready;
    }

    @Override
    public long admittedAt() {
        return this.admittedAt;
    }

    /// Runs the conclusion in admission order. A quantum that can no longer commit (a quantum before it failed the
    /// sequence, or committed short of where it starts) fails first, so that its stages discard their work.
    @Override
    public void concluded(Throwable blocked) {
        if (blocked != null) fail(blocked);
        Runnable conclusion = this.pendingConclusion;
        this.pendingConclusion = null;
        conclusion.run();
    }

    /// Keeps the final logits (when no host row takes them) and hands the quantum to the terminal consumer while
    /// its workspace is still held.
    @Override
    protected void commit() {
        if (this.hostLogits == null && this.proposal == null) retainLogits(this.gpu);
        if (this.terminalConsumer != null) this.terminalConsumer.accept(this);
    }

    /// Releases the input upload and the workspace, whatever the outcome.
    @Override
    protected void release() {
        try {
            releaseInputUpload();
        } catch (RuntimeException | Error releaseFailure) {
            fail(releaseFailure);
        }
        if (this.workspace != null) {
            try {
                this.workspace.close();
            } catch (RuntimeException | Error cleanupFailure) {
                fail(cleanupFailure);
                try {
                    this.workspace.close();
                } catch (RuntimeException | Error retryFailure) {
                    fail(retryFailure);
                }
            }
        }
    }

    /// The sequence follows the outcome, in admission order: a failure fails it and abandons the quantum's work, a
    /// cancellation of the sequence abandons it, and a success commits it at its next position. Only a cancellation
    /// the sequence requested is CANCELLED; a failure that came with it (a release that failed after the
    /// cancellation) still fails the quantum. A quantum the sequence never admitted leaves it alone. Then the host
    /// row and the proposal become readable for a success, and the outcome is recorded for the continuation.
    @Override
    protected void settle() {
        Throwable error = failure();
        Throwable failed = error;
        boolean cancelled = false;
        if (error instanceof CancellationException cancellation && this.sequence.cancellationRequested()) {
            Throwable[] suppressed = cancellation.getSuppressed();
            if (suppressed.length == 0) {
                cancelled = true;
                failed = null;
            } else {
                failed = suppressed[0];
                for (int index = 1; index < suppressed.length; index++)
                    if (suppressed[index] != failed) failed.addSuppressed(suppressed[index]);
            }
        }
        if (error != null) failed = releaseLogits(failed);
        Outcome completed;
        // Each branch settles the quantum's admission last, so a throw means it is not settled yet.
        try {
            if (failed != null) {
                if (this.admitted) {
                    this.sequence.fail(failed);
                    this.sequence.abandon();
                }
                completed = new Outcome(Status.FAILED, failed);
            } else if (cancelled) {
                if (this.admitted) this.sequence.abandon();
                completed = new Outcome(Status.CANCELLED, null);
            } else {
                if (!this.admitted) throw new IllegalStateException("the sequence never admitted the quantum");
                this.sequence.commit(drafting() ? this.admittedAt : this.startPosition + committedRowCount());
                completed = new Outcome(Status.SUCCESS, null);
            }
        } catch (RuntimeException | Error retirementFailure) {
            // Never strand the continuation: a sequence that refuses the retirement fails the quantum, and the
            // sequence, which can no longer resume from its state.
            if (failed == null) failed = retirementFailure;
            else if (failed != retirementFailure) failed.addSuppressed(retirementFailure);
            failed = releaseLogits(failed);
            if (this.admitted) {
                this.sequence.fail(failed);
                this.sequence.abandon();
            }
            completed = new Outcome(Status.FAILED, failed);
        }
        this.admitted = false;
        // The row was copied before the retirement boundary; only a successful quantum exposes it.
        if (this.hostLogits != null) this.hostLogits.retired(completed.status() == Status.SUCCESS);
        if (this.proposal != null) this.proposal.retired(completed.status() == Status.SUCCESS);
        if (completed.failure() != null) fail(completed.failure());
        this.conclusion = completed;
    }

    /// The quantum's outcome once it retired, or null before: what a continuation reads.
    public Outcome conclusion() {
        return this.conclusion;
    }

    private void retainLogits(ExecutionGpu gpu) {
        if (this.workspace == null || !this.workspace.hasBuffer(ExecutionPlan.Buffer.LOGITS)) return;
        long address = this.workspace.detachAddress(ExecutionPlan.Buffer.LOGITS);
        try {
            this.logitsOutput = new DeviceLogits(
                    gpu, address, logitsRowCount(), this.plan.weights().config().vocabSize());
        } catch (RuntimeException | Error constructionFailure) {
            try {
                gpu.free(address);
            } catch (Throwable cleanupFailure) {
                constructionFailure.addSuppressed(cleanupFailure);
            }
            throw constructionFailure;
        }
    }

    private Throwable releaseLogits(Throwable priorFailure) {
        if (this.logitsOutput == null) return priorFailure;
        try {
            this.logitsOutput.close();
            this.logitsOutput = null;
        } catch (Throwable cleanupFailure) {
            if (priorFailure != null) priorFailure.addSuppressed(cleanupFailure);
            else priorFailure = cleanupFailure;
        }
        return priorFailure;
    }
}
