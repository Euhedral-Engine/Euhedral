package io.euhedral_execution.inference.core.qwen4;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import java.util.ArrayList;
import java.util.List;

/// Everything one sequence carries from token to token: the recurrent state of every GDN layer, the key/value pages
/// and indexer keys of every sparse-attention layer, the per-layer-embedding history and n-gram context, and the
/// position. Expert residency is not here: it belongs to the model's cache, not to a sequence.
///
/// The device storage is allocated when the sequence is opened and released by [#close]; KV pages are reserved as the
/// sequence grows. [#reset] returns the sequence to its start without freeing it.
public final class Qwen4Sequence implements AutoCloseable {

    /// Key and value bytes of one token of one sparse-attention layer in the NVFP4 pages.
    private static final long KV_BYTES_PER_TOKEN = 576;

    private final ExecutionGpu gpu;
    private final Qwen4GdnLayer gdnLayer;
    private final Qwen4Ple ple;
    private final int maxTokens;
    private final Qwen4GdnLayer.State[] gdn;
    private final Qwen4QsaState[] qsa;
    private final long pleHistory;
    private final Qwen4Ple.State pleState;
    private long position;
    private boolean closed;

    Qwen4Sequence(
            ExecutionGpu gpu,
            Qwen4GdnLayer gdnLayer,
            Qwen4Ple ple,
            boolean[] sparseAttention,
            int keyValueWidth,
            int maxTokens) {
        this.gpu = gpu;
        this.gdnLayer = gdnLayer;
        this.ple = ple;
        this.maxTokens = maxTokens;
        this.gdn = new Qwen4GdnLayer.State[sparseAttention.length];
        this.qsa = new Qwen4QsaState[sparseAttention.length];
        long history = 0;
        try {
            for (int layer = 0; layer < sparseAttention.length; layer++) {
                if (sparseAttention[layer]) this.qsa[layer] = new Qwen4QsaState(gpu, keyValueWidth, maxTokens);
                else this.gdn[layer] = gdnLayer.allocateState(gpu);
            }
            history = gpu.allocate(ple.historyBytes());
            gpu.zeroDeviceMemory(history, ple.historyBytes());
        } catch (Throwable failure) {
            freeLayers();
            if (history != 0) gpu.free(history);
            throw failure;
        }
        this.pleHistory = history;
        this.pleState = ple.newState(history);
    }

    /// Tokens the sequence has consumed: the position of the next one.
    public long position() {
        return this.position;
    }

    public int maxTokens() {
        return this.maxTokens;
    }

    Qwen4GdnLayer.State gdn(int layer) {
        return this.gdn[layer];
    }

    Qwen4QsaState qsa(int layer) {
        return this.qsa[layer];
    }

    Qwen4Ple.State ple() {
        return this.pleState;
    }

    void advance(int rows) {
        this.position += rows;
    }

    /// Back to the start: every state zeroed, the position 0. The caller guarantees no step is in flight.
    public void reset(int endOfSequence) {
        ensureOpen();
        for (Qwen4GdnLayer.State state : this.gdn) if (state != null) this.gdnLayer.reset(this.gpu, state);
        for (Qwen4QsaState state : this.qsa) if (state != null) state.reset();
        this.gpu.zeroDeviceMemory(this.pleHistory, this.ple.historyBytes());
        this.ple.reset(this.pleState, endOfSequence);
        this.position = 0;
    }

    /// Device bytes held now (KV pages grow with the position).
    public long deviceBytes() {
        long bytes = this.ple.historyBytes();
        for (Qwen4GdnLayer.State state : this.gdn)
            if (state != null) bytes += this.gdnLayer.convolutionHistoryBytes() + this.gdnLayer.recurrentBytes();
        for (Qwen4QsaState state : this.qsa) {
            if (state == null) continue;
            long pageTokens = (state.length() + 255L) / 256 * 256;
            bytes += Qwen4QsaState.indexerBytes(this.maxTokens) + KV_BYTES_PER_TOKEN * pageTokens;
        }
        return bytes;
    }

    private void ensureOpen() {
        if (this.closed) throw new IllegalStateException("the sequence is closed");
    }

    private void freeLayers() {
        List<Throwable> failures = new ArrayList<>();
        for (int layer = 0; layer < this.gdn.length; layer++) {
            try {
                if (this.gdn[layer] != null) this.gdnLayer.freeState(this.gpu, this.gdn[layer]);
                if (this.qsa[layer] != null) this.qsa[layer].close();
            } catch (RuntimeException | Error failure) {
                failures.add(failure);
            }
        }
        if (!failures.isEmpty()) {
            RuntimeException combined = new IllegalStateException("releasing a sequence failed", failures.get(0));
            for (int i = 1; i < failures.size(); i++) combined.addSuppressed(failures.get(i));
            throw combined;
        }
    }

    @Override
    public void close() {
        if (this.closed) return;
        this.closed = true;
        try {
            freeLayers();
        } finally {
            this.gpu.free(this.pleHistory);
        }
    }
}
