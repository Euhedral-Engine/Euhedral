package io.euhedral_execution.inference.core.model_loader.qwen4.expert;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class GpuExpertTransferTest {

    @TempDir
    Path directory;

    private ExpertFixture fixture;
    private HostBackedGpu gpu;
    private FileExpertStore store;
    private long slab;
    private final AtomicReference<FakeStream> copyStream = new AtomicReference<>();
    private ExecutorService pool;

    @BeforeEach
    void setUp() throws IOException {
        this.fixture = ExpertFixture.standard(this.directory, 21);
        this.gpu = new HostBackedGpu();
        this.store = new FileExpertStore(this.gpu, this.fixture.file, this.fixture.banks, 4);
        this.slab = this.gpu.allocate(8 * this.fixture.slotBytes());
        this.pool = Executors.newCachedThreadPool();
    }

    @AfterEach
    void tearDown() {
        this.pool.shutdownNow();
    }

    private void asyncStream() {
        this.gpu.streamFactory(() -> {
            FakeStream stream = new FakeStream();
            this.copyStream.set(stream);
            return stream;
        });
    }

    private long slot(int index) {
        return this.slab + index * this.fixture.slotBytes();
    }

    private static CompletableFuture<Throwable> completion() {
        return new CompletableFuture<>();
    }

    @Test
    void copiesTheRecordAndSignalsOnceOnASynchronousStream() throws Exception {
        try (GpuExpertTransfer transfer = new GpuExpertTransfer(this.gpu, 2)) {
            HostRecord record = this.store.open(1, 1);
            CompletableFuture<Throwable> done = completion();
            transfer.start(record, slot(2), null, done::complete);
            assertNull(done.get(10, TimeUnit.SECONDS));
            assertArrayEquals(
                    this.fixture.record(1, 1), this.gpu.readDevice(slot(2), this.fixture.banks[1].recordBytes(1)));
            assertEquals(4, this.store.freeSlots(), "the staging slot returned after the copy");
            assertEquals(1, this.gpu.hostCopies());
        }
        assertEquals(1, this.gpu.streamsOpened());
        assertEquals(1, this.gpu.streamsClosed());
    }

    @Test
    void completionsNeverRunOnTheDriverThread() throws Exception {
        asyncStream();
        try (GpuExpertTransfer transfer = new GpuExpertTransfer(this.gpu, 4)) {
            List<CompletableFuture<String>> threads = new ArrayList<>();
            for (int expert = 0; expert < 3; expert++) {
                CompletableFuture<String> done = new CompletableFuture<>();
                threads.add(done);
                transfer.start(this.store.open(0, expert), slot(expert), null, failure -> {
                    assertNull(failure);
                    done.complete(Thread.currentThread().getName());
                });
            }
            for (CompletableFuture<String> done : threads)
                assertEquals("expert-transfer-completion", done.get(10, TimeUnit.SECONDS));
            assertFalse(this.copyStream.get().confirmedOnWorker(), "confirmRetired runs on an ordinary thread");
            for (int expert = 0; expert < 3; expert++)
                assertArrayEquals(
                        this.fixture.record(0, expert),
                        this.gpu.readDevice(slot(expert), this.fixture.banks[0].recordBytes(expert)));
        }
    }

    @Test
    void aDeviceFailureFailsTheCompletionAndStillReleasesTheRecord() throws Exception {
        asyncStream();
        try (GpuExpertTransfer transfer = new GpuExpertTransfer(this.gpu, 2)) {
            this.copyStream.get().failNextRetirements(1);
            CompletableFuture<Throwable> done = completion();
            transfer.start(this.store.open(0, 0), slot(0), null, done::complete);
            Throwable failure = done.get(10, TimeUnit.SECONDS);
            assertInstanceOf(IllegalStateException.class, failure);
            assertEquals(4, this.store.freeSlots());
            CompletableFuture<Throwable> retry = completion();
            transfer.start(this.store.open(0, 0), slot(0), null, retry::complete);
            assertNull(retry.get(10, TimeUnit.SECONDS));
        }
    }

    @Test
    void aFailedSubmissionThrowsAfterRecoveringTheStreamAndReleasingEverything() throws Exception {
        asyncStream();
        try (GpuExpertTransfer transfer = new GpuExpertTransfer(this.gpu, 1)) {
            this.gpu.failNextHostCopies(1);
            AtomicReference<Throwable> signalled = new AtomicReference<>();
            assertThrows(
                    IllegalStateException.class,
                    () -> transfer.start(this.store.open(0, 0), slot(0), null, signalled::set));
            assertNull(signalled.get(), "a throwing start signals no completion");
            assertEquals(4, this.store.freeSlots());
            assertEquals(1, this.copyStream.get().recoveries());
            // The single in-flight permit came back.
            CompletableFuture<Throwable> done = completion();
            transfer.start(this.store.open(0, 1), slot(0), null, done::complete);
            assertNull(done.get(10, TimeUnit.SECONDS));
        }
    }

    @Test
    void aFailedBoundaryRegistrationThrowsAndReleases() throws Exception {
        asyncStream();
        try (GpuExpertTransfer transfer = new GpuExpertTransfer(this.gpu, 1)) {
            this.copyStream.get().failNextNotifications(1);
            assertThrows(
                    IllegalStateException.class,
                    () -> transfer.start(this.store.open(0, 0), slot(0), null, failure -> {
                        throw new AssertionError("no completion expected");
                    }));
            assertEquals(4, this.store.freeSlots());
            assertEquals(1, this.copyStream.get().recoveries());
            CompletableFuture<Throwable> done = completion();
            transfer.start(this.store.open(0, 1), slot(0), null, done::complete);
            assertNull(done.get(10, TimeUnit.SECONDS));
        }
    }

    @Test
    void anUnprovenCompletionKeepsTheStagedRecord() throws Exception {
        asyncStream();
        try (GpuExpertTransfer transfer = new GpuExpertTransfer(this.gpu, 2)) {
            this.gpu.completionProven(false);
            this.copyStream.get().failNextRetirements(1);
            CompletableFuture<Throwable> done = completion();
            transfer.start(this.store.open(0, 0), slot(0), null, done::complete);
            assertNotEquals(null, done.get(10, TimeUnit.SECONDS));
            assertEquals(3, this.store.freeSlots(), "a copy that may still run keeps its source pinned");
            this.gpu.completionProven(true);
        }
    }

    @Test
    void theCopyWaitsOnTheDeviceBehindTheFence() throws Exception {
        asyncStream();
        FakeStream compute = new FakeStream();
        try (GpuExpertTransfer transfer = new GpuExpertTransfer(this.gpu, 2)) {
            long old = slot(0);
            this.gpu.copyHostToDevice(
                    old,
                    java.lang.foreign.MemorySegment.ofArray(new byte[9001]),
                    9001); // the slot holds zeros: the kernels' data
            CountDownLatch kernels = new CountDownLatch(1);
            compute.stall(kernels);
            long marker = compute.openMarker();
            compute.mark(marker);
            StreamFence fence = StreamFence.owning(compute, marker);
            CompletableFuture<Throwable> done = completion();
            transfer.start(this.store.open(1, 1), old, fence, done::complete);
            assertThrows(TimeoutException.class, () -> done.get(200, TimeUnit.MILLISECONDS));
            assertArrayEquals(
                    new byte[(int) this.fixture.banks[1].recordBytes(1)],
                    this.gpu.readDevice(old, this.fixture.banks[1].recordBytes(1)),
                    "the refill did not overtake the work the fence stands for");
            kernels.countDown();
            assertNull(done.get(10, TimeUnit.SECONDS));
            assertArrayEquals(
                    this.fixture.record(1, 1), this.gpu.readDevice(old, this.fixture.banks[1].recordBytes(1)));
            assertFalse(compute.markerClosed(marker), "the transfer does not release the fence; its owner does");
            fence.release();
            assertTrue(compute.markerClosed(marker));
        } finally {
            compute.close();
        }
    }

    @Test
    void sharedStreamFencesCoverEachOtherAndCompositesReleaseEveryMember() {
        FakeStream compute = new FakeStream();
        try {
            long marker = compute.openMarker();
            long other = compute.openMarker();
            StreamFence a = new StreamFence(compute, marker);
            StreamFence b = new StreamFence(compute, marker);
            StreamFence c = StreamFence.owning(compute, other);
            assertTrue(a.covers(b));
            assertFalse(a.covers(c));
            List<DeviceFence> dropped = new ArrayList<>();
            DeviceFence merged = DeviceFences.merge(null, a, dropped);
            assertEquals(a, merged);
            merged = DeviceFences.merge(merged, b, dropped);
            assertEquals(a, merged, "an equal shared fence adds nothing");
            assertEquals(1, dropped.size());
            merged = DeviceFences.merge(merged, c, dropped);
            assertTrue(merged.covers(a) && merged.covers(c));
            merged.release();
            assertTrue(compute.markerClosed(other), "the owning member was released");
            assertFalse(compute.markerClosed(marker), "a shared marker stays with its owner");
            assertTrue(merged == DeviceFences.merge(merged, a, dropped));
        } finally {
            compute.close();
        }
    }

    @Test
    void closeWaitsForTransfersInFlight() throws Exception {
        asyncStream();
        GpuExpertTransfer transfer = new GpuExpertTransfer(this.gpu, 2);
        CountDownLatch release = new CountDownLatch(1);
        this.copyStream.get().stall(release);
        CompletableFuture<Throwable> done = completion();
        transfer.start(this.store.open(0, 0), slot(0), null, done::complete);
        Future<?> closing = this.pool.submit(transfer::close);
        assertThrows(TimeoutException.class, () -> closing.get(200, TimeUnit.MILLISECONDS));
        assertEquals(0, this.gpu.streamsClosed());
        release.countDown();
        closing.get(10, TimeUnit.SECONDS);
        assertTrue(done.isDone());
        assertEquals(1, this.gpu.streamsClosed());
        transfer.close();
        assertEquals(1, this.gpu.streamsClosed(), "closing twice closes the stream once");
        HostRecord rejected = this.store.open(0, 1);
        assertThrows(IllegalStateException.class, () -> transfer.start(rejected, slot(0), null, failure -> {}));
        assertEquals(4, this.store.freeSlots(), "a rejected record is still released");
    }

    @Test
    void inFlightTransfersAreBoundedByTheirPermits() throws Exception {
        asyncStream();
        try (GpuExpertTransfer transfer = new GpuExpertTransfer(this.gpu, 2)) {
            CountDownLatch release = new CountDownLatch(1);
            this.copyStream.get().stall(release);
            List<CompletableFuture<Throwable>> done = new ArrayList<>();
            for (int i = 0; i < 2; i++) {
                CompletableFuture<Throwable> d = completion();
                done.add(d);
                transfer.start(this.store.open(0, i), slot(i), null, d::complete);
            }
            Future<?> third = this.pool.submit(() -> {
                CompletableFuture<Throwable> d = completion();
                done.add(d);
                transfer.start(this.store.open(0, 2), slot(2), null, d::complete);
                return null;
            });
            assertThrows(TimeoutException.class, () -> third.get(200, TimeUnit.MILLISECONDS));
            release.countDown();
            third.get(10, TimeUnit.SECONDS);
            for (CompletableFuture<Throwable> d : done) assertNull(d.get(10, TimeUnit.SECONDS));
        }
    }

    @Test
    void anInterruptedStartReleasesTheRecordAndSignalsNothing() throws Exception {
        asyncStream();
        try (GpuExpertTransfer transfer = new GpuExpertTransfer(this.gpu, 1)) {
            CountDownLatch release = new CountDownLatch(1);
            this.copyStream.get().stall(release);
            CompletableFuture<Throwable> first = completion();
            transfer.start(this.store.open(0, 0), slot(0), null, first::complete);
            AtomicReference<Throwable> outcome = new AtomicReference<>();
            CountDownLatch entered = new CountDownLatch(1);
            Thread blocked = new Thread(() -> {
                entered.countDown();
                try {
                    transfer.start(this.store.open(0, 1), slot(1), null, failure -> outcome.set(failure));
                } catch (InterruptedException interrupted) {
                    outcome.set(interrupted);
                } catch (IOException ignored) {
                    // not thrown by open here
                }
            });
            blocked.start();
            assertTrue(entered.await(5, TimeUnit.SECONDS));
            while (this.store.freeSlots() > 2) Thread.sleep(1);
            Thread.sleep(50);
            blocked.interrupt();
            blocked.join(10_000);
            assertInstanceOf(InterruptedException.class, outcome.get());
            assertEquals(3, this.store.freeSlots(), "the interrupted start closed its record");
            release.countDown();
            assertNull(first.get(10, TimeUnit.SECONDS));
        }
    }
}
