package io.euhedral_execution.inference.core.model_loader.qwen4.expert;

import java.util.concurrent.atomic.AtomicBoolean;

/// A claim on one resident expert of an [ExpertCache]. While the lease is open its slot is pinned: the cache
/// neither evicts nor refills it, so [#deviceAddress] and every [#projectionAddress] stay valid for kernels.
/// Closing returns the slot to the cache; the addresses must not be used afterwards, and the lease refuses to
/// reveal them once closed.
///
/// A lease may be closed from any thread. Device work that reads the slot needs no host wait before the lease is
/// closed: close it with a [DeviceFence] instead.
public final class ExpertLease implements AutoCloseable {
    private final ExpertCache cache;
    private final ExpertBank expertBank;
    private final int bank;
    private final int expert;
    private final int slot;
    private final int generation;
    private final long deviceAddress;
    private final AtomicBoolean closed = new AtomicBoolean();

    ExpertLease(
            ExpertCache cache,
            ExpertBank expertBank,
            int bank,
            int expert,
            int slot,
            int generation,
            long deviceAddress) {
        this.cache = cache;
        this.expertBank = expertBank;
        this.bank = bank;
        this.expert = expert;
        this.slot = slot;
        this.generation = generation;
        this.deviceAddress = deviceAddress;
    }

    /// The bank's ordinal.
    public int bank() {
        return this.bank;
    }

    public int expert() {
        return this.expert;
    }

    /// The bank this expert belongs to.
    public ExpertBank expertBank() {
        return this.expertBank;
    }

    /// The device address of the record, which holds [#byteSize] bytes.
    ///
    /// @throws IllegalStateException when the lease is closed
    public long deviceAddress() {
        if (this.closed.get()) throw new IllegalStateException("the lease on expert " + this.expert + " is closed");
        return this.deviceAddress;
    }

    /// The bytes of this expert's record (the slot may be larger).
    public long byteSize() {
        return this.expertBank.recordBytes(this.expert);
    }

    /// The device address of one projection: the record's address plus the projection's offset in the record.
    ///
    /// @throws IllegalArgumentException when the bank has no such projection
    public long projectionAddress(String projectionName) {
        return deviceAddress() + this.expertBank.projection(projectionName).recordOffset();
    }

    /// The projection's description in the record.
    public ExpertProjection projection(String projectionName) {
        return this.expertBank.projection(projectionName);
    }

    /// Whether the lease is open and its slot still holds this expert's record: false after [#close], and after
    /// the cache closed with the lease open.
    public boolean isValid() {
        return !this.closed.get() && this.cache.isCurrent(this.slot, this.generation);
    }

    /// Returns the slot to the cache. Idempotent.
    @Override
    public void close() {
        close(null);
    }

    /// Returns the slot to the cache, recording that the device work that read it is ordered by `fence`: the
    /// transfer that refills the slot waits for the fence on the device. The cache owns `fence` from here on,
    /// including when the lease was already closed (the fence is then released unused). A null fence is
    /// [#close()].
    public void close(DeviceFence fence) {
        if (this.closed.compareAndSet(false, true)) this.cache.release(this.slot, this.generation, fence);
        else if (fence != null) fence.release();
    }

    @Override
    public String toString() {
        return "ExpertLease[bank=" + this.bank + ", expert=" + this.expert + ", slot=" + this.slot
                + (this.closed.get() ? ", closed]" : "]");
    }
}
