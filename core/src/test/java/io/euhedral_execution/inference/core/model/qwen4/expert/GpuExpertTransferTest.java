package io.euhedral_execution.inference.core.model.qwen4.expert;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.gpu.GpuStream;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/// Copies on copy streams: a stream is only an order of copies, so one takes copies from any
/// thread and any staging buffer; a copy records a marker behind itself, its retirement is a driver
/// callback that only reports a ticket, and nothing waits for it.
class GpuExpertTransferTest {

    @TempDir
    Path directory;

    private final HostBackedGpu gpu = new HostBackedGpu();
    private final List<FakeStream> streams = new ArrayList<>();
    private ExpertFixture fixture;
    private FileExpertStore store;

    @BeforeEach
    void setUp() throws Exception {
        this.fixture = ExpertFixture.standard(this.directory, 21);
        this.gpu.streamFactory(() -> {
            FakeStream stream = new FakeStream();
            synchronized (this.streams) {
                this.streams.add(stream);
            }
            return stream;
        });
        this.store = new FileExpertStore(this.gpu, this.fixture.file, this.fixture.banks, 2);
    }

    private FakeStream copyStream(int index) {
        synchronized (this.streams) {
            return this.streams.get(1 + index);
        }
    }

    private static final class Retired implements GpuStream.RetirementListener {
        final CompletableFuture<Long> ticket = new CompletableFuture<>();
        volatile boolean driver;

        @Override
        public void retired(long ticket, boolean driverThread) {
            this.driver = driverThread;
            this.ticket.complete(ticket);
        }
    }

    @Test
    void aCopyLandsInTheSlotAndItsRetirementIsADriverCallback() throws Exception {
        long slot = this.gpu.allocate(this.fixture.slotBytes());
        try (GpuExpertTransfer transfer = new GpuExpertTransfer(this.gpu, 2)) {
            assertEquals(2, transfer.streams());
            long marker = transfer.openMarker();
            Retired retired = new Retired();
            HostRecord record = this.store.open(2, 7, 0);
            transfer.stream(1, record, slot, null, marker, retired);
            long ticket = retired.ticket.get(10, TimeUnit.SECONDS);
            assertTrue(retired.driver, "the boundary reports from a driver thread, where it may only enqueue");
            assertNull(transfer.confirm(1, ticket));
            assertArrayEquals(this.fixture.record(2, 7), this.gpu.readDevice(slot, record.byteSize()));
            assertEquals(1, transfer.submittedCopies());
            transfer.closeMarker(marker);
        }
        this.gpu.free(slot);
    }

    @Test
    void theCopyDoesNotWaitForTheHostAndStreamsDoNotWaitForEachOther() throws Exception {
        long slotA = this.gpu.allocate(this.fixture.slotBytes());
        long slotB = this.gpu.allocate(this.fixture.slotBytes());
        try (GpuExpertTransfer transfer = new GpuExpertTransfer(this.gpu, 2)) {
            long markerA = transfer.openMarker();
            long markerB = transfer.openMarker();
            CountDownLatch release = new CountDownLatch(1);
            copyStream(0).stall(release);
            Retired slow = new Retired();
            Retired fast = new Retired();
            transfer.stream(0, this.store.open(2, 1, 0), slotA, null, markerA, slow);
            transfer.stream(1, this.store.open(2, 2, 1), slotB, null, markerB, fast);
            fast.ticket.get(10, TimeUnit.SECONDS);
            assertFalse(slow.ticket.isDone(), "a stalled stream leaves the other stream's copy alone");
            release.countDown();
            slow.ticket.get(10, TimeUnit.SECONDS);
            assertNull(transfer.confirm(0, slow.ticket.get()));
            assertNull(transfer.confirm(1, fast.ticket.get()));
            transfer.closeMarker(markerA);
            transfer.closeMarker(markerB);
        }
        this.gpu.free(slotA);
        this.gpu.free(slotB);
    }

    @Test
    void aFenceOrdersTheCopyStreamBehindTheKernelsThatReadTheSlot() throws Exception {
        long slot = this.gpu.allocate(this.fixture.slotBytes());
        try (GpuExpertTransfer transfer = new GpuExpertTransfer(this.gpu, 1)) {
            long marker = transfer.openMarker();
            int[] waited = {0};
            DeviceFence fence = copyStream -> {
                assertNotNull(copyStream);
                waited[0]++;
            };
            Retired retired = new Retired();
            transfer.stream(0, this.store.open(0, 0, 0), slot, fence, marker, retired);
            retired.ticket.get(10, TimeUnit.SECONDS);
            assertEquals(1, waited[0]);
            assertNull(transfer.confirm(0, retired.ticket.get()));
            transfer.closeMarker(marker);
        }
        this.gpu.free(slot);
    }

    @Test
    void aDeviceFailureIsReportedWhenTheCopyIsConfirmed() throws Exception {
        long slot = this.gpu.allocate(this.fixture.slotBytes());
        try (GpuExpertTransfer transfer = new GpuExpertTransfer(this.gpu, 1)) {
            long marker = transfer.openMarker();
            copyStream(0).failNextRetirements(1);
            Retired retired = new Retired();
            transfer.stream(0, this.store.open(0, 0, 0), slot, null, marker, retired);
            assertNotNull(transfer.confirm(0, retired.ticket.get(10, TimeUnit.SECONDS)));
            transfer.closeMarker(marker);
        }
        this.gpu.free(slot);
    }

    @Test
    void aCopyThatCannotBeSubmittedThrowsAndTheStreamIsRecovered() throws Exception {
        long slot = this.gpu.allocate(this.fixture.slotBytes());
        try (GpuExpertTransfer transfer = new GpuExpertTransfer(this.gpu, 1)) {
            long marker = transfer.openMarker();
            this.gpu.failNextHostCopies(1);
            assertThrows(
                    RuntimeException.class,
                    () -> transfer.stream(0, this.store.open(0, 0, 0), slot, null, marker, new Retired()));
            transfer.recover(0, new IllegalStateException("the copy was not queued"));
            assertEquals(1, copyStream(0).recoveries());
            transfer.closeMarker(marker);
        }
        this.gpu.free(slot);
    }

    @Test
    void oneStreamTakesTheCopiesOfManyBuffersFromManyThreads() throws Exception {
        int copies = 6;
        FileExpertStore wide = new FileExpertStore(this.gpu, this.fixture.file, this.fixture.banks, copies);
        long[] slots = new long[copies];
        for (int i = 0; i < copies; i++) slots[i] = this.gpu.allocate(this.fixture.slotBytes());
        try (GpuExpertTransfer transfer = new GpuExpertTransfer(this.gpu, 1)) {
            long[] markers = new long[copies];
            Retired[] retired = new Retired[copies];
            Thread[] submitters = new Thread[copies];
            for (int i = 0; i < copies; i++) {
                int index = i;
                markers[i] = transfer.openMarker();
                retired[i] = new Retired();
                submitters[i] = new Thread(() -> {
                    try {
                        transfer.stream(
                                0, wide.open(2, index, index), slots[index], null, markers[index], retired[index]);
                    } catch (Exception failure) {
                        throw new AssertionError(failure);
                    }
                });
                submitters[i].start();
            }
            for (Thread submitter : submitters) submitter.join();
            for (int i = 0; i < copies; i++) {
                assertNull(transfer.confirm(0, retired[i].ticket.get(10, TimeUnit.SECONDS)));
                assertArrayEquals(
                        this.fixture.record(2, i), this.gpu.readDevice(slots[i], this.fixture.banks[2].recordBytes(i)));
                transfer.closeMarker(markers[i]);
            }
            assertEquals(copies, transfer.submittedCopies());
        } finally {
            wide.close();
            for (long slot : slots) this.gpu.free(slot);
        }
    }

    @Test
    void closingClosesEveryStreamOnceAndTwiceIsHarmless() throws Exception {
        GpuExpertTransfer transfer = new GpuExpertTransfer(this.gpu, 3);
        transfer.close();
        transfer.close();
        assertEquals(this.gpu.streamsOpened(), this.gpu.streamsClosed());
        assertEquals(4, this.gpu.streamsClosed(), "three copy streams and the marker stream");
    }
}
