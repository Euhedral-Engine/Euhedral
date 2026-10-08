package io.euhedral_execution.inference.core.model.qwen4;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.model.qwen4.expert.ExpertCacheStats;
import io.euhedral_execution.inference.core.runtime.graph.GraphStorage;
import java.util.ArrayList;
import java.util.List;

/// The plan's one workspace: the device buffers of a chunk of up to `rows` tokens (the residual state, the
/// activations, the logits row, the scratch of each kind of layer, the expansion scratch of the NVFP4 linears),
/// sized at load for the plan's chunk, and the block resources of its MoE layers ([MoeLayer]) for each row capacity:
/// a decode token's block description stays as small as the token. The graphs of every shape
/// (every row capacity, the diagnostic variants, a test's single layers) take turns on it through a [Lease]: the
/// plan serves one quantum at a time, and the workspace's owner orders a graph's first device stages behind the
/// previous graph's last ones ([Shape#workspaceBuffers]).
final class Workspace {

    /// A graph's hold on the workspace, in the runtime's terms, with the MoE block resources of its shape's row
    /// capacity. The plan owns the workspace and frees it; closing a lease only ends the graph's use.
    static final class Lease implements GraphStorage {
        private final Workspace storage;
        private final MoeLayer moe;
        private boolean closed;

        Lease(Workspace storage, int rows) {
            this.storage = storage;
            this.moe = storage.moe(rows);
        }

        Workspace storage() {
            return this.storage;
        }

        MoeLayer moe() {
            return this.moe;
        }

        @Override
        public long retainedBytes() {
            return this.closed ? 0 : this.storage.retainedBytes();
        }

        @Override
        public boolean isClosed() {
            return this.closed;
        }

        @Override
        public void close() {
            this.closed = true;
        }
    }

    private final ExecutionGpu gpu;
    private final int rows;
    /// The MoE block resources of each row capacity of [ExecutionPlan#rowBucket], ascending; the last is `rows`'.
    private final int[] capacities;
    private final MoeLayer[] moes;
    private final ExecutionGpu.UploadBuffer tokenUpload;
    private final long tokensDevice;
    private final long embedded;
    private final long state;
    private final long pleOutput;
    private final long mixed;
    private final long blockOutput;
    private final long finalMixed;
    private final long logits;
    private final long hcScratch;
    private final long layerScratch;
    private final long moeScratch;
    /// The expansion scratch the native NVFP4 route takes, bound around the stages that declare it
    /// ([Shape#takesScratch]); 0 when no linear takes that route at `rows` rows.
    private final long expansion;
    private final long expansionBytes;
    private final List<Long> allocations = new ArrayList<>();
    private long retained;
    private boolean closed;

    // The block in flight, written by one stage and read by the stages that follow it across an edge.
    MoeLayer.Scratch moeBlock;
    int bank;
    int experts;
    long routeArmedNanos;
    long plannedNanos;
    ExpertCacheStats.Snapshot traceBefore;
    /// The n-gram rows of the chunk, their staging buffer and how many there are: written by the stage that
    /// computes the ids and gathers them, read by the stage that copies them to the device.
    final long[] pleRowIds;

    ExecutionGpu.UploadBuffer pleUpload;
    int pleCount;

    Workspace(ExecutionPlan plan, ExecutionGpu gpu, int rows) {
        this.gpu = gpu;
        this.rows = rows;
        int hidden = plan.hidden();
        this.pleRowIds = new long[rows * plan.ple().rowsPerToken()];
        this.capacities = plan.rowBuckets();
        this.moes = new MoeLayer[this.capacities.length];
        ExecutionGpu.UploadBuffer upload = null;
        try {
            for (int i = 0; i < this.capacities.length; i++) this.moes[i] = plan.newMoeLayer(this.capacities[i]);
            upload = gpu.allocateUploadBuffer(4L * rows);
        } catch (Throwable failure) {
            closeMoes();
            throw failure;
        }
        this.tokenUpload = upload;
        MoeLayer moe = this.moes[this.moes.length - 1];
        try {
            long bf16 = Short.BYTES;
            long width = (long) plan.streams() * hidden;
            this.tokensDevice = allocate(4L * rows);
            this.embedded = allocate(rows * (long) hidden * bf16);
            this.state = allocate(rows * width * bf16);
            this.pleOutput = allocate(rows * width * bf16);
            this.mixed = allocate(rows * (long) hidden * bf16);
            this.blockOutput = allocate(rows * (long) hidden * bf16);
            this.finalMixed = allocate((long) hidden * bf16);
            this.logits = allocate((long) plan.vocabulary() * bf16);
            this.hcScratch = allocate(plan.hyperConnection().scratchBytes(rows));
            long layerBytes = Math.max(
                    Math.max(plan.gdn().scratchBytes(rows), plan.qsa().scratchBytes(rows)),
                    plan.ple().scratchBytes(rows));
            this.layerScratch = allocate(layerBytes);
            this.moeScratch = allocate(moe.scratchBytes(rows));
            this.expansionBytes = Math.max(
                    Math.max(
                            plan.gdn().linearScratchBytes(gpu, rows), plan.qsa().linearScratchBytes(gpu, rows)),
                    Math.max(plan.ple().linearScratchBytes(gpu, rows), moe.linearScratchBytes(rows)));
            this.expansion = this.expansionBytes > 0 ? allocate(this.expansionBytes) : 0;
        } catch (Throwable failure) {
            releaseBuffers();
            this.tokenUpload.close();
            closeMoes();
            throw failure;
        }
    }

    private long allocate(long bytes) {
        long address = this.gpu.allocate(bytes);
        this.allocations.add(address);
        this.retained += bytes;
        return address;
    }

    private void releaseBuffers() {
        for (long address : this.allocations) this.gpu.free(address);
        this.allocations.clear();
        this.retained = 0;
    }

    /// Most rows a quantum on this storage may have.
    int rows() {
        return this.rows;
    }

    /// The MoE block resources of the capacity that serves a shape of `rows` rows.
    MoeLayer moe(int rows) {
        for (int i = 0; i < this.capacities.length; i++) if (rows <= this.capacities[i]) return this.moes[i];
        throw new IllegalArgumentException("no MoE block resources for " + rows + " rows");
    }

    private void closeMoes() {
        RuntimeException failure = null;
        for (int i = 0; i < this.moes.length; i++) {
            MoeLayer moe = this.moes[i];
            this.moes[i] = null;
            if (moe == null) continue;
            try {
                moe.close();
            } catch (RuntimeException closeFailure) {
                if (failure == null) failure = closeFailure;
                else failure.addSuppressed(closeFailure);
            }
        }
        if (failure != null) throw failure;
    }

    ExecutionGpu.UploadBuffer tokenUpload() {
        return this.tokenUpload;
    }

    long tokensDevice() {
        return this.tokensDevice;
    }

    long embedded() {
        return this.embedded;
    }

    /// The residual state, `[rows][streams * hidden]` BF16.
    long state() {
        return this.state;
    }

    long pleOutput() {
        return this.pleOutput;
    }

    long mixed() {
        return this.mixed;
    }

    long blockOutput() {
        return this.blockOutput;
    }

    long finalMixed() {
        return this.finalMixed;
    }

    long logits() {
        return this.logits;
    }

    long hcScratch() {
        return this.hcScratch;
    }

    long layerScratch() {
        return this.layerScratch;
    }

    long moeScratch() {
        return this.moeScratch;
    }

    long scratchAddress() {
        return this.expansion;
    }

    long scratchBytes() {
        return this.expansionBytes;
    }

    long retainedBytes() {
        return this.retained;
    }

    boolean isClosed() {
        return this.closed;
    }

    void close() {
        if (this.closed) return;
        this.closed = true;
        try {
            closeMoes();
        } finally {
            try {
                this.tokenUpload.close();
            } finally {
                releaseBuffers();
            }
        }
    }
}
