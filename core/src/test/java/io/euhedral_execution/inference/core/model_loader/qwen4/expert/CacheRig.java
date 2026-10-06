package io.euhedral_execution.inference.core.model_loader.qwen4.expert;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import io.euhedral_execution.inference.core.host.TestHostFrames;
import java.io.IOException;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.LongSupplier;

/// An [ExpertCache] over a fixture file on a [HostBackedGpu], assembled from one of the host stores and one
/// of the transfers, so the same behaviour is proven on each combination.
final class CacheRig implements AutoCloseable {

    enum StoreKind {
        ARENA,
        FILE
    }

    enum TransferKind {
        /// No GPU: native-memory copies on helper threads, with injectable faults.
        SYNTHETIC,
        /// The real [GpuExpertTransfer] on a synchronous stream.
        GPU_INLINE,
        /// The real [GpuExpertTransfer] on an asynchronous stream whose boundaries fire on a driver-like
        /// thread.
        GPU_ASYNC
    }

    final ExpertFixture fixture;
    final StoreKind storeKind;
    final TransferKind transferKind;
    final HostBackedGpu gpu = new HostBackedGpu();
    final HostExpertStore store;
    final ExpertTransfer transfer;
    final SyntheticExpertTransfer synthetic;
    final ExpertCache cache;
    private volatile FakeStream asyncStream;
    private volatile Semaphore gate;
    private volatile CountDownLatch stall;

    CacheRig(ExpertFixture fixture, StoreKind storeKind, TransferKind transferKind, int slots) throws IOException {
        this(fixture, storeKind, transferKind, slots, Math.max(2, slots), System::nanoTime);
    }

    CacheRig(
            ExpertFixture fixture,
            StoreKind storeKind,
            TransferKind transferKind,
            int slots,
            int stagingSlots,
            LongSupplier clock)
            throws IOException {
        this.fixture = fixture;
        this.storeKind = storeKind;
        this.transferKind = transferKind;
        this.store = switch (storeKind) {
            case ARENA -> new ArenaExpertStore(this.gpu, fixture.file, fixture.banks, 4);
            case FILE -> new FileExpertStore(this.gpu, fixture.file, fixture.banks, stagingSlots);
        };
        switch (transferKind) {
            case SYNTHETIC -> {
                this.synthetic = new SyntheticExpertTransfer();
                this.transfer = this.synthetic;
            }
            case GPU_INLINE -> {
                this.synthetic = null;
                this.transfer = new GpuExpertTransfer(this.gpu, TestHostFrames.SHARED);
            }
            case GPU_ASYNC -> {
                this.synthetic = null;
                this.gpu.streamFactory(() -> {
                    FakeStream stream = new FakeStream();
                    this.asyncStream = stream;
                    return stream;
                });
                this.transfer = new GpuExpertTransfer(this.gpu, TestHostFrames.SHARED);
            }
            default -> throw new IllegalArgumentException();
        }
        this.cache = new ExpertCache(
                this.store,
                this.transfer,
                this.gpu,
                slots,
                fixture.slotBytes(),
                clock,
                ExpertCache.DEFAULT_CLOSE_TIMEOUT_NANOS,
                TestHostFrames.SHARED);
    }

    /// Whether `hold` can keep transfers in flight on this combination (a synchronous stream copies inside
    /// the submitting call).
    boolean holdable() {
        return this.transferKind != TransferKind.GPU_INLINE;
    }

    /// Keeps every transfer started from now on in flight until `releaseHeld`.
    void hold() {
        switch (this.transferKind) {
            case SYNTHETIC -> {
                Semaphore gate = new Semaphore(0);
                this.gate = gate;
                this.synthetic.gate(gate);
            }
            case GPU_ASYNC -> {
                CountDownLatch latch = new CountDownLatch(1);
                this.stall = latch;
                this.asyncStream.stall(latch);
            }
            default -> throw new IllegalStateException("not holdable");
        }
    }

    void releaseHeld() {
        switch (this.transferKind) {
            case SYNTHETIC -> this.gate.release(1_000_000);
            case GPU_ASYNC -> this.stall.countDown();
            default -> throw new IllegalStateException("not holdable");
        }
    }

    /// The next `count` transfers fail: by an exception from the submission on a synchronous stream, and as a
    /// device failure reported when the boundary retires otherwise.
    void failNextTransfers(int count) {
        switch (this.transferKind) {
            case SYNTHETIC -> {
                AtomicInteger remaining = new AtomicInteger(count);
                this.synthetic.failWhen(request -> remaining.getAndUpdate(n -> Math.max(0, n - 1)) > 0
                        ? new IllegalStateException("injected transfer failure")
                        : null);
            }
            case GPU_INLINE -> this.gpu.failNextHostCopies(count);
            case GPU_ASYNC -> this.asyncStream.failNextRetirements(count);
        }
    }

    /// Transfers that moved their bytes.
    int transfers() {
        return this.transferKind == TransferKind.SYNTHETIC ? this.synthetic.completed() : this.gpu.hostCopies();
    }

    FakeStream asyncStream() {
        return this.asyncStream;
    }

    /// Asserts the lease's device bytes equal the record in the file, byte for byte and by the bank's CRC-32.
    void assertLeaseBytes(ExpertLease lease) {
        byte[] device = this.gpu.readDevice(lease.deviceAddress(), lease.byteSize());
        assertArrayEquals(this.fixture.record(lease.bank(), lease.expert()), device, lease.toString());
        assertEquals(
                this.fixture.banks[lease.bank()].crc32(lease.expert()), ExpertFixture.crc32(device), "crc " + lease);
    }

    @Override
    public void close() {
        if (this.stall != null) this.stall.countDown();
        if (this.gate != null) this.gate.release(1_000_000);
        this.cache.close();
        this.gpu.assertAllReleased();
    }
}
