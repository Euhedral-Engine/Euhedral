package io.euhedral_execution.inference.core.model_loader.qwen4.expert;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.host.TestHostFrames;
import io.euhedral_execution.inference.core.model_loader.qwen4.expert.ExpertCache.Ticket;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/// The streaming path of the cache: a miss is submitted on a lane's copy stream and its lease,
/// which carries the copy's marker, exists before the bytes arrive, so nothing on the host waits
/// for them. The slot stays unevictable until the copy retired even when its leases were closed
/// already.
class ExpertCacheStreamTest {

    @TempDir
    Path directory;

    private final HostBackedGpu gpu = new HostBackedGpu();
    private final List<FakeStream> streams = new ArrayList<>();
    private ExpertFixture fixture;
    private ExpertCache cache;

    @BeforeEach
    void setUp() throws IOException {
        this.fixture = ExpertFixture.standard(this.directory, 91);
        this.gpu.streamFactory(() -> {
            FakeStream stream = new FakeStream();
            synchronized (this.streams) {
                this.streams.add(stream);
            }
            return stream;
        });
        var store = new ArenaExpertStore(this.gpu, this.fixture.file, this.fixture.banks, 2);
        var transfer = new GpuExpertTransfer(this.gpu, TestHostFrames.SHARED, 2);
        this.cache = new ExpertCache(
                store,
                transfer,
                this.gpu,
                2,
                this.fixture.slotBytes(),
                System::nanoTime,
                ExpertCache.DEFAULT_CLOSE_TIMEOUT_NANOS,
                TestHostFrames.SHARED);
    }

    @AfterEach
    void tearDown() {
        this.cache.close();
        this.gpu.assertAllReleased();
    }

    /// Lane `lane`'s copy stream: the transfer's own stream is the first one opened.
    private FakeStream lane(int lane) {
        synchronized (this.streams) {
            return this.streams.get(1 + lane);
        }
    }

    /// A retirement listener that hands over the ticket once the copy retired.
    private static final class Retired
            implements io.euhedral_execution.inference.core.gpu.GpuStream.RetirementListener {
        final CompletableFuture<Long> ticket = new CompletableFuture<>();

        @Override
        public void retired(long ticket, boolean driverThread) {
            this.ticket.complete(ticket);
        }
    }

    private ExpertCache.Load miss(int bank, int expert) {
        Ticket ticket = new Ticket();
        this.cache.claim(bank, expert, ticket);
        assertNull(ticket.lease(), "a cold expert is a miss");
        return ticket.load();
    }

    private void finish(ExpertCache.Load load, int lane, Retired retired) throws Exception {
        load.retire(lane, retired.ticket.get(10, TimeUnit.SECONDS));
    }

    @Test
    void aMissIsLeasedBeforeItsBytesArriveAndTheLeaseCarriesTheCopysMarker() throws Exception {
        CountDownLatch release = new CountDownLatch(1);
        lane(0).stall(release);
        ExpertCache.Load load = miss(2, 5);
        Retired retired = new Retired();
        HostRecord record = load.open();
        ExpertLease lease = load.submit(0, record, retired);
        assertNotNull(lease);
        assertNotEquals(0, lease.readyMarker(), "the lease stands for bytes that are still on their way");
        assertTrue(!retired.ticket.isDone(), "the copy has not retired: the lease did not wait for it");
        release.countDown();
        finish(load, 0, retired);
        record.close();
        assertArrayEquals(this.fixture.record(2, 5), this.gpu.readDevice(lease.deviceAddress(), lease.byteSize()));
        lease.close();
        this.cache.checkQuiescent();
        assertEquals(1, this.cache.stats().snapshot().misses());
    }

    @Test
    void aSlotWhoseLeaseWasClosedStaysUnevictableUntilItsCopyRetired() throws Exception {
        CountDownLatch release = new CountDownLatch(1);
        lane(0).stall(release);
        ExpertCache.Load load = miss(0, 1);
        Retired retired = new Retired();
        HostRecord record = load.open();
        ExpertLease lease = load.submit(0, record, retired);
        // The kernels that read the slot were submitted behind the marker, so the lease closes at once.
        lease.close();
        assertEquals(0, this.cache.evictableSlots(), "the copy is still writing the slot");
        ExpertCache.Load other = miss(0, 2);
        assertThrows(IllegalStateException.class, () -> this.cache.claim(0, 3, new Ticket()), "no slot is free");
        release.countDown();
        finish(load, 0, retired);
        record.close();
        assertEquals(1, this.cache.evictableSlots(), "the retired copy's slot may be evicted now");
        other.fail(new IllegalStateException("not loaded in this test"));
        // The retired slot serves the next miss.
        ExpertCache.Load next = miss(0, 4);
        next.fail(new IllegalStateException("not loaded in this test"));
    }

    @Test
    void aRequestThatFindsTheCopySubmittedIsLeasedWithTheSameMarker() throws Exception {
        CountDownLatch release = new CountDownLatch(1);
        lane(0).stall(release);
        ExpertCache.Load load = miss(1, 1);
        Retired retired = new Retired();
        HostRecord record = load.open();
        ExpertLease first = load.submit(0, record, retired);
        Ticket again = new Ticket();
        this.cache.claim(1, 1, again);
        assertNull(again.load());
        assertEquals(first.readyMarker(), again.lease().readyMarker());
        assertEquals(1, this.cache.stats().snapshot().coalescedRequests());
        release.countDown();
        finish(load, 0, retired);
        record.close();
        first.close();
        again.lease().close();
        this.cache.checkQuiescent();
    }

    @Test
    void aResidentExpertIsAHitWithNothingToWaitFor() throws Exception {
        ExpertCache.Load load = miss(3, 0);
        Retired retired = new Retired();
        HostRecord record = load.open();
        ExpertLease lease = load.submit(0, record, retired);
        finish(load, 0, retired);
        record.close();
        lease.close();
        Ticket hit = new Ticket();
        this.cache.claim(3, 0, hit);
        assertNull(hit.load());
        assertEquals(0, hit.lease().readyMarker(), "its bytes are in the slot already");
        hit.lease().close();
    }

    @Test
    void aLoadThatNeverSubmittedReturnsItsSlot() throws Exception {
        ExpertCache.Load load = miss(2, 2);
        load.abandon(0, null, new IllegalStateException("the record could not be read"), false);
        this.cache.checkQuiescent();
        ExpertCache.Load again = miss(2, 2);
        again.fail(new IllegalStateException("not loaded in this test"));
    }

    @Test
    void aDeviceFailureAtRetirementUnmapsTheExpertButKeepsItsLeaseValid() throws Exception {
        lane(0).failNextRetirements(1);
        ExpertCache.Load load = miss(0, 6);
        Retired retired = new Retired();
        HostRecord record = load.open();
        ExpertLease lease = load.submit(0, record, retired);
        finish(load, 0, retired);
        record.close();
        assertTrue(lease.isValid(), "the holder closes its lease as usual");
        Ticket later = new Ticket();
        assertTrue(!this.cache.isResident(0, 6), "a failed copy is not served to later requests");
        lease.close();
        assertEquals(1, this.cache.stats().snapshot().failedTransfers());
        this.cache.claim(0, 6, later);
        assertNull(later.lease(), "the next request loads it again");
        later.load().fail(new IllegalStateException("not loaded in this test"));
    }
}
