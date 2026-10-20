package io.euhedral_execution.inference.core.state;

import io.euhedral_execution.inference.core.gpu.GpuMemory;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicIntegerArray;

/// The attention cache pages that sequences keep in pinned host memory past their device share, sized once at load:
/// `slots` sequence slots, each holding `hostPages` pages for each of `layers` attention layers in one pinned arena,
/// and one device region of `hostPages` pages per slot that the slot's layers share. A sequence keeps its first
/// `devicePages` pages of each layer on the device; before a layer's attention its host pages are copied into the
/// region, which its page table addresses, and the rows an append wrote are copied back ([AttentionKvState]).
///
/// A sequence takes a slot when its state is attached and returns it when the state closes. No slot is ever waited
/// for: [#take()] returns null when every slot is taken.
public final class KvHostPages implements AutoCloseable {
    private final GpuMemory gpu;
    private final int layers;
    private final int devicePages;
    private final int hostPages;
    private final long pageBytes;
    private final long host;
    private final long regions;
    private final AtomicIntegerArray taken;
    private boolean closed;

    private KvHostPages(
            GpuMemory gpu,
            int layers,
            int devicePages,
            int hostPages,
            long pageBytes,
            long host,
            long regions,
            int slots) {
        this.gpu = gpu;
        this.layers = layers;
        this.devicePages = devicePages;
        this.hostPages = hostPages;
        this.pageBytes = pageBytes;
        this.host = host;
        this.regions = regions;
        this.taken = new AtomicIntegerArray(slots);
    }

    /// Allocates the pinned arena and the device regions of `slots` slots: `hostPages` pages of `pageBytes` (a K
    /// plane then a V plane) for each of `layers` layers per slot, past each layer's `devicePages` device pages.
    public static KvHostPages allocate(
            GpuMemory gpu, int slots, int layers, int devicePages, int hostPages, long pageBytes) {
        Objects.requireNonNull(gpu, "gpu");
        if (slots <= 0 || layers <= 0 || devicePages <= 0 || hostPages <= 0 || pageBytes <= 0)
            throw new IllegalArgumentException("host attention pages need slots, layers, pages and page bytes");
        long region = Math.multiplyExact((long) hostPages, pageBytes);
        long host = gpu.allocateHostWeights(Math.multiplyExact(Math.multiplyExact(region, layers), slots));
        long regions = 0;
        try {
            regions = gpu.allocate(Math.multiplyExact(region, slots));
            return new KvHostPages(gpu, layers, devicePages, hostPages, pageBytes, host, regions, slots);
        } catch (RuntimeException | Error failure) {
            if (regions != 0) gpu.free(regions);
            gpu.freeHostWeights(host);
            throw failure;
        }
    }

    /// A free slot, taken, or null when every slot is taken.
    public Slot take() {
        if (this.closed) throw new IllegalStateException("host attention pages are closed");
        for (int index = 0; index < this.taken.length(); index++)
            if (this.taken.compareAndSet(index, 0, 1)) return new Slot(index);
        return null;
    }

    /// Pages each layer keeps on the device before its first host page.
    public int devicePages() {
        return this.devicePages;
    }

    /// Host pages of each layer of a slot.
    public int hostPages() {
        return this.hostPages;
    }

    /// Bytes of one page: its K plane then its V plane.
    public long pageBytes() {
        return this.pageBytes;
    }

    /// Slots in all.
    public int slots() {
        return this.taken.length();
    }

    /// Frees the arena and the regions; every sequence that used them has closed.
    @Override
    public void close() {
        if (this.closed) return;
        this.closed = true;
        this.gpu.free(this.regions);
        this.gpu.freeHostWeights(this.host);
    }

    /// One sequence's slot: its host pages of each attention layer and its device region.
    public final class Slot {
        private final int index;
        private boolean released;

        private Slot(int index) {
            this.index = index;
        }

        /// The slot's host pages of attention layer `layer` (0 for the first full-attention layer).
        public AttentionKvState.HostRun run(int layer) {
            if (layer < 0 || layer >= KvHostPages.this.layers)
                throw new IllegalArgumentException("no host pages for attention layer " + layer);
            long region = Math.multiplyExact((long) KvHostPages.this.hostPages, KvHostPages.this.pageBytes);
            long host = KvHostPages.this.host
                    + Math.multiplyExact(region, (long) this.index * KvHostPages.this.layers + layer);
            return new AttentionKvState.HostRun(
                    KvHostPages.this.devicePages,
                    KvHostPages.this.hostPages,
                    host,
                    KvHostPages.this.regions + Math.multiplyExact(region, this.index));
        }

        /// Returns the slot; its sequence's states closed.
        public void release() {
            if (this.released) return;
            this.released = true;
            KvHostPages.this.taken.set(this.index, 0);
        }
    }
}
