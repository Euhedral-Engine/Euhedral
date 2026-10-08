package io.euhedral_execution.inference.core.model.qwen38;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.runtime.graph.CaptureFingerprint;
import java.util.Objects;

/// Persistent GDN convolution and recurrent buffers owned by one sequence.
public final class GdnState implements AutoCloseable {
    private final ExecutionGpu gpu;
    private long convolutionStateAddress;
    private long recurrentStateAddress;
    private final long convolutionBytes;
    private final long recurrentBytes;
    private boolean closing;
    private boolean closed;

    /// Speculative verification (docs/MTP_CONTRACT.md §5, ReplaySSM): the state before the latest
    /// verification, its recorded inputs, and replay scratch. Allocated at the first verification.
    private Speculative speculative;
    /// Verified rows the next quantum must replay from the checkpoint before touching this state.
    private int pendingReplayRows;

    /// Device buffers for `capacity` verified rows.
    public record Speculative(
            int capacity,
            long recurrentCheckpoint,
            long convolutionCheckpoint,
            long queryKeyRows,
            long valueZRows,
            long alphaRows,
            long betaRows,
            long replayConvolved,
            long replayOutput) {}

    private GdnState(
            ExecutionGpu gpu,
            long convolutionStateAddress,
            long recurrentStateAddress,
            long convolutionBytes,
            long recurrentBytes) {
        this.gpu = gpu;
        this.convolutionStateAddress = convolutionStateAddress;
        this.recurrentStateAddress = recurrentStateAddress;
        this.convolutionBytes = convolutionBytes;
        this.recurrentBytes = recurrentBytes;
    }

    public long convolutionBytes() {
        return this.convolutionBytes;
    }

    public long recurrentBytes() {
        return this.recurrentBytes;
    }

    /// The speculative buffers, allocated (or grown) for `rows` verified rows of the given widths
    /// (BF16 query/key and value/z rows, FP32 alpha/beta per value head, BF16 convolved and output rows).
    public Speculative speculative(
            int rows, int queryKeyWidth, int valueZWidth, int heads, int convolvedWidth, int outputWidth) {
        ensureOpen();
        if (this.speculative != null && this.speculative.capacity() >= rows) return this.speculative;
        if (this.speculative != null) {
            if (this.pendingReplayRows > 0)
                throw new IllegalStateException("cannot grow speculative state with a pending replay");
            releaseSpeculative();
        }
        int capacity = Math.max(rows, 8);
        long[] addresses = new long[8];
        long[] sizes = {
            this.recurrentBytes,
            this.convolutionBytes,
            (long) capacity * queryKeyWidth * Short.BYTES,
            (long) capacity * valueZWidth * Short.BYTES,
            (long) capacity * heads * Float.BYTES,
            (long) capacity * heads * Float.BYTES,
            (long) capacity * convolvedWidth * Short.BYTES,
            (long) capacity * outputWidth * Short.BYTES
        };
        try {
            for (int i = 0; i < sizes.length; i++) addresses[i] = allocateRequired(this.gpu, sizes[i]);
        } catch (Throwable failure) {
            for (long address : addresses) freeAfterFailure(this.gpu, address, failure);
            throw failure;
        }
        this.speculative = new Speculative(
                capacity,
                addresses[0],
                addresses[1],
                addresses[2],
                addresses[3],
                addresses[4],
                addresses[5],
                addresses[6],
                addresses[7]);
        return this.speculative;
    }

    public Speculative speculative() {
        return this.speculative;
    }

    void fingerprint(CaptureFingerprint fingerprint) {
        fingerprint
                .add(this.gpu, this.convolutionStateAddress)
                .add(this.gpu, this.recurrentStateAddress)
                .add(this.pendingReplayRows);
        Speculative buffers = this.speculative;
        if (buffers == null) return;
        fingerprint.add(buffers.capacity());
        for (long address : new long[] {
            buffers.recurrentCheckpoint(),
            buffers.convolutionCheckpoint(),
            buffers.queryKeyRows(),
            buffers.valueZRows(),
            buffers.alphaRows(),
            buffers.betaRows(),
            buffers.replayConvolved(),
            buffers.replayOutput()
        }) fingerprint.add(this.gpu, address);
    }

    public int pendingReplayRows() {
        return this.pendingReplayRows;
    }

    /// After a verification committed `rows` of its rows: replay them from the checkpoint before the
    /// state is next used, or nothing when every row was committed (0).
    public void setPendingReplayRows(int rows) {
        if (rows < 0 || (rows > 0 && this.speculative == null)) throw new IllegalArgumentException("invalid replay");
        this.pendingReplayRows = rows;
    }

    private void releaseSpeculative() {
        Speculative s = this.speculative;
        this.speculative = null;
        Throwable failure = null;
        for (long address : new long[] {
            s.recurrentCheckpoint(),
            s.convolutionCheckpoint(),
            s.queryKeyRows(),
            s.valueZRows(),
            s.alphaRows(),
            s.betaRows(),
            s.replayConvolved(),
            s.replayOutput()
        }) {
            try {
                this.gpu.freeAsync(address);
            } catch (Throwable cleanupFailure) {
                if (failure == null) failure = cleanupFailure;
                else failure.addSuppressed(cleanupFailure);
            }
        }
        if (failure != null) throw propagate(failure);
    }

    /// Allocates zero-initialized GDN state for the supplied layer geometry.
    public static GdnState allocate(
            ExecutionGpu gpu,
            int keyHeads,
            int valueHeads,
            int keyHeadDim,
            int valueHeadDim,
            int convolutionKernelDim) {
        Objects.requireNonNull(gpu, "gpu");
        if (keyHeads <= 0
                || valueHeads <= 0
                || valueHeads % keyHeads != 0
                || keyHeadDim <= 0
                || valueHeadDim <= 0
                || convolutionKernelDim < 2) {
            throw new IllegalArgumentException("invalid GDN sequence-state dimensions");
        }

        int convolutionChannels = Math.addExact(
                Math.multiplyExact(2, Math.multiplyExact(keyHeads, keyHeadDim)),
                Math.multiplyExact(valueHeads, valueHeadDim));
        long convolutionBytes = Math.multiplyExact(
                Math.multiplyExact((long) convolutionChannels, convolutionKernelDim - 1L), Short.BYTES);
        long recurrentBytes = Math.multiplyExact(
                Math.multiplyExact(Math.multiplyExact((long) valueHeads, keyHeadDim), valueHeadDim), Float.BYTES);

        long convolutionAddress = 0;
        long recurrentAddress = 0;
        try {
            convolutionAddress = allocateRequired(gpu, convolutionBytes);
            recurrentAddress = allocateRequired(gpu, recurrentBytes);
            gpu.zeroDeviceMemory(convolutionAddress, convolutionBytes);
            gpu.zeroDeviceMemory(recurrentAddress, recurrentBytes);
            return new GdnState(gpu, convolutionAddress, recurrentAddress, convolutionBytes, recurrentBytes);
        } catch (Throwable failure) {
            freeAfterFailure(gpu, recurrentAddress, failure);
            freeAfterFailure(gpu, convolutionAddress, failure);
            throw failure;
        }
    }

    public long convolutionStateAddress() {
        ensureOpen();
        return this.convolutionStateAddress;
    }

    public long recurrentStateAddress() {
        ensureOpen();
        return this.recurrentStateAddress;
    }

    @Override
    public void close() {
        if (this.closed) return;
        this.closing = true;
        Throwable failure = null;
        if (this.speculative != null) {
            try {
                releaseSpeculative();
            } catch (Throwable cleanupFailure) {
                failure = cleanupFailure;
            }
        }
        failure = free(this.convolutionStateAddress, failure, true);
        failure = free(this.recurrentStateAddress, failure, false);
        if (failure != null) throw propagate(failure);
        this.closed = true;
    }

    private static long allocateRequired(ExecutionGpu gpu, long byteSize) {
        long address = gpu.allocateAsync(byteSize);
        if (address <= 0) throw new IllegalStateException("GPU returned an invalid GDN state address");
        return address;
    }

    private static void freeAfterFailure(ExecutionGpu gpu, long address, Throwable failure) {
        if (address == 0) return;
        try {
            gpu.freeAsync(address);
        } catch (Throwable cleanupFailure) {
            failure.addSuppressed(cleanupFailure);
        }
    }

    private Throwable free(long address, Throwable priorFailure, boolean convolution) {
        if (address == 0) return priorFailure;
        try {
            this.gpu.free(address);
            if (convolution) this.convolutionStateAddress = 0;
            else this.recurrentStateAddress = 0;
            return priorFailure;
        } catch (Throwable failure) {
            if (priorFailure != null) priorFailure.addSuppressed(failure);
            else priorFailure = failure;
            return priorFailure;
        }
    }

    private void ensureOpen() {
        if (this.closing || this.closed) throw new IllegalStateException("GDN sequence state is closed");
    }

    private static RuntimeException propagate(Throwable failure) {
        if (failure instanceof RuntimeException runtimeException) return runtimeException;
        if (failure instanceof Error error) throw error;
        return new IllegalStateException("failed to release GDN sequence state", failure);
    }
}
