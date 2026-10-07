package io.euhedral_execution.inference.core.model.qwen4;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.model.qwen4.expert.ExpertCacheStats;
import io.euhedral_execution.inference.core.runtime.graph.GraphStorage;
import java.util.ArrayList;
import java.util.List;

/// The workspace of the graphs of one row capacity: the device buffers of a chunk of up to `rows`
/// tokens (the residual state, the activations, the logits row, the scratch of each kind of layer)
/// and the block resources of its MoE layers ([MoeLayer]). The plan serves one quantum at a
/// time and a graph is recycled only after its quantum's device work retired, so the graphs of
/// every shape of a capacity (the diagnostic variants, a test's single layers) take turns on one
/// workspace through a [Lease] instead of each holding their own.
final class Workspace {

    /// A graph's hold on the shared workspace of its capacity, in the runtime's terms. The plan
    /// owns the workspace and frees it; closing a lease only ends the graph's use.
    static final class Lease implements GraphStorage {
        private final Workspace storage;
        private boolean closed;

        Lease(Workspace storage) {
            this.storage = storage;
        }

        Workspace storage() {
            return this.storage;
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
    private final MoeLayer moe;
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
        this.moe = plan.newMoeLayer(rows);
        this.pleRowIds = new long[rows * plan.ple().rowsPerToken()];
        this.tokenUpload = gpu.allocateUploadBuffer(4L * rows);
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
            this.moeScratch = allocate(this.moe.scratchBytes(rows));
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

    MoeLayer moe() {
        return this.moe;
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
            this.moe.close();
        } finally {
            try {
                this.tokenUpload.close();
            } finally {
                releaseBuffers();
            }
        }
    }
}
