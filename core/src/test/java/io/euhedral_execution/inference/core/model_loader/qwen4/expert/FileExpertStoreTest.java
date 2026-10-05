package io.euhedral_execution.inference.core.model_loader.qwen4.expert;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class FileExpertStoreTest {

    @TempDir
    Path directory;

    private ExpertFixture fixture;
    private HostBackedGpu gpu;
    private ExecutorService pool;

    @BeforeEach
    void setUp() throws IOException {
        this.fixture = ExpertFixture.standard(this.directory, 11);
        this.gpu = new HostBackedGpu();
        this.pool = Executors.newCachedThreadPool();
    }

    @AfterEach
    void tearDown() {
        this.pool.shutdownNow();
    }

    private static byte[] bytes(HostRecord record) {
        return record.segment().toArray(ValueLayout.JAVA_BYTE);
    }

    @Test
    void servesEveryRecordIdenticalToTheFileThroughOneStagingArena() throws Exception {
        long total = 0;
        try (FileExpertStore store = new FileExpertStore(this.gpu, this.fixture.file, this.fixture.banks, 3)) {
            for (int bank = 0; bank < this.fixture.banks.length; bank++) {
                for (int expert = 0; expert < this.fixture.expertCount(bank); expert++) {
                    try (HostRecord record = store.open(bank, expert)) {
                        byte[] bytes = bytes(record);
                        assertArrayEquals(this.fixture.record(bank, expert), bytes, bank + "/" + expert);
                        assertEquals(this.fixture.banks[bank].crc32(expert), ExpertFixture.crc32(bytes));
                    }
                    total += this.fixture.banks[bank].recordBytes(expert);
                }
            }
            assertEquals(total, store.bytesRead());
            assertEquals(this.fixture.totalExperts(), store.recordOpens());
            assertEquals(3, store.freeSlots());
            assertEquals(0, store.slotBytes() % 4096);
            assertTrue(store.slotBytes() >= 9001);
            assertEquals(List.of(3 * store.slotBytes()), this.gpu.hostAllocationSizes(), "one arena of N slots");
        }
        this.gpu.assertAllReleased();
    }

    @Test
    void exhaustedStagingBlocksAndResumesWhenARecordCloses() throws Exception {
        try (FileExpertStore store = new FileExpertStore(this.gpu, this.fixture.file, this.fixture.banks, 2)) {
            HostRecord first = store.open(0, 0);
            HostRecord second = store.open(0, 1);
            assertEquals(0, store.freeSlots());
            CountDownLatch started = new CountDownLatch(1);
            Future<byte[]> third = this.pool.submit(() -> {
                started.countDown();
                try (HostRecord record = store.open(0, 2)) {
                    return bytes(record);
                }
            });
            assertTrue(started.await(5, TimeUnit.SECONDS));
            assertThrows(TimeoutException.class, () -> third.get(150, TimeUnit.MILLISECONDS));
            assertEquals(1, store.blockedOpens());
            first.close();
            assertArrayEquals(this.fixture.record(0, 2), third.get(10, TimeUnit.SECONDS));
            second.close();
            assertEquals(2, store.freeSlots());
        }
    }

    @Test
    void opensTimeOutWhileStagingIsExhausted() throws Exception {
        try (FileExpertStore store = new FileExpertStore(this.gpu, this.fixture.file, this.fixture.banks, 1)) {
            HostRecord held = store.open(0, 0);
            assertThrows(TimeoutException.class, () -> store.open(0, 1, TimeUnit.MILLISECONDS.toNanos(50)));
            assertThrows(TimeoutException.class, () -> store.open(0, 1, 0));
            assertEquals(0, store.freeSlots());
            held.close();
            assertEquals(1, store.freeSlots());
            try (HostRecord record = store.open(0, 1, TimeUnit.SECONDS.toNanos(5))) {
                assertArrayEquals(this.fixture.record(0, 1), bytes(record));
            }
        }
    }

    @Test
    void anInterruptedWaiterLeavesTheSlotsIntact() throws Exception {
        try (FileExpertStore store = new FileExpertStore(this.gpu, this.fixture.file, this.fixture.banks, 1)) {
            HostRecord held = store.open(0, 0);
            AtomicInteger outcome = new AtomicInteger();
            CountDownLatch waiting = new CountDownLatch(1);
            Thread waiter = new Thread(() -> {
                waiting.countDown();
                try {
                    store.open(0, 1).close();
                } catch (InterruptedException interrupted) {
                    outcome.set(1);
                } catch (Exception other) {
                    outcome.set(2);
                }
            });
            waiter.start();
            assertTrue(waiting.await(5, TimeUnit.SECONDS));
            while (store.blockedOpens() == 0) Thread.sleep(1);
            waiter.interrupt();
            waiter.join(10_000);
            assertEquals(1, outcome.get());
            assertEquals(0, store.freeSlots());
            held.close();
            assertEquals(1, store.freeSlots());
        }
    }

    @Test
    void closingRecordsTwiceNeverFreesASlotTwice() throws Exception {
        try (FileExpertStore store = new FileExpertStore(this.gpu, this.fixture.file, this.fixture.banks, 2)) {
            HostRecord record = store.open(1, 1);
            record.close();
            record.close();
            record.close();
            assertEquals(2, store.freeSlots());
            HostRecord a = store.open(0, 0);
            HostRecord b = store.open(0, 1);
            assertTrue(a.hostAddress() != b.hostAddress(), "two open records never share a slot");
            a.close();
            b.close();
            assertEquals(2, store.freeSlots());
        }
    }

    @Test
    void aFailedReadReturnsItsSlot() throws Exception {
        RecordSource failing = new RecordSource() {
            private final FileRecordSource real = newSource();

            private FileRecordSource newSource() {
                try {
                    return new FileRecordSource(
                            FileExpertStoreTest.this.fixture.file, FileExpertStoreTest.this.fixture.banks);
                } catch (IOException failure) {
                    throw new IllegalStateException(failure);
                }
            }

            @Override
            public void read(ExpertBank bank, int expert, MemorySegment destination)
                    throws IOException, InterruptedException {
                if (bank.name().equals("layer1") && expert == 2) throw new IOException("injected read failure");
                this.real.read(bank, expert, destination);
            }

            @Override
            public long bytesRead() {
                return this.real.bytesRead();
            }

            @Override
            public void close() {
                this.real.close();
            }
        };
        try (FileExpertStore store = new FileExpertStore(this.gpu, failing, this.fixture.banks, 1)) {
            for (int attempt = 0; attempt < 3; attempt++) {
                IOException failure = assertThrows(IOException.class, () -> store.open(1, 2));
                assertEquals("injected read failure", failure.getMessage());
                assertEquals(1, store.freeSlots());
            }
            try (HostRecord record = store.open(1, 1)) {
                assertArrayEquals(this.fixture.record(1, 1), bytes(record));
            }
            assertEquals(1, store.recordOpens(), "only successful opens count");
        }
        this.gpu.assertAllReleased();
    }

    @Test
    void anInterruptedFileReadFailsAloneAndNeverClosesTheSourceForOthers() throws Exception {
        try (FileRecordSource source = new FileRecordSource(this.fixture.file, this.fixture.banks)) {
            ExpertBank bank = this.fixture.banks[0];
            byte[] bytes = new byte[(int) bank.recordBytes(1)];
            MemorySegment destination = MemorySegment.ofArray(bytes);
            Thread.currentThread().interrupt();
            assertThrows(InterruptedException.class, () -> source.read(bank, 1, destination));
            assertFalse(Thread.currentThread().isInterrupted(), "the exception consumed the interrupt");
            assertEquals(0, source.bytesRead());
            // A channel shared between reads would be closed now.
            source.read(bank, 1, destination);
            assertArrayEquals(this.fixture.record(0, 1), bytes);
            assertEquals(bytes.length, source.bytesRead());
        }
    }

    @Test
    void aSourceSeamCanServeRecordsFromAnotherTier() throws Exception {
        // A host-side record cache in front of the file: the store is indifferent to where bytes come from.
        AtomicInteger fromTier = new AtomicInteger();
        RecordSource tier = new RecordSource() {
            @Override
            public void read(ExpertBank bank, int expert, MemorySegment destination) {
                int ordinal = bank.layer();
                destination.copyFrom(MemorySegment.ofArray(FileExpertStoreTest.this.fixture.record(ordinal, expert)));
                fromTier.incrementAndGet();
            }

            @Override
            public long bytesRead() {
                return 0;
            }

            @Override
            public void close() {}
        };
        try (FileExpertStore store = new FileExpertStore(this.gpu, tier, this.fixture.banks, 2)) {
            try (HostRecord record = store.open(2, 5)) {
                assertArrayEquals(this.fixture.record(2, 5), bytes(record));
            }
            assertEquals(1, fromTier.get());
            assertEquals(0, store.bytesRead());
        }
    }

    @Test
    void closingTheStoreWakesWaitersAndFreesTheArenaAfterTheLastRecord() throws Exception {
        FileExpertStore store = new FileExpertStore(this.gpu, this.fixture.file, this.fixture.banks, 1);
        HostRecord held = store.open(0, 0);
        Future<?> waiter = this.pool.submit(() -> {
            try (HostRecord record = store.open(0, 1)) {
                return bytes(record);
            }
        });
        while (store.blockedOpens() == 0) Thread.sleep(1);
        store.close();
        store.close();
        ExecutionException failure = assertThrows(ExecutionException.class, () -> waiter.get(10, TimeUnit.SECONDS));
        assertInstanceOf(IllegalStateException.class, failure.getCause());
        assertEquals(1, this.gpu.liveHostAllocations(), "an open record keeps the staging arena");
        assertArrayEquals(this.fixture.record(0, 0), bytes(held));
        held.close();
        assertEquals(0, this.gpu.liveHostAllocations());
        assertEquals(1, this.gpu.hostFrees());
        assertThrows(IllegalStateException.class, () -> store.open(0, 0));
    }

    @Test
    void manyThreadsShareAFewSlotsWithoutMixingBytes() throws Exception {
        try (FileExpertStore store = new FileExpertStore(this.gpu, this.fixture.file, this.fixture.banks, 3)) {
            List<Future<?>> futures = new ArrayList<>();
            for (int t = 0; t < 12; t++) {
                int seed = t;
                futures.add(this.pool.submit(() -> {
                    Random random = new Random(seed);
                    for (int i = 0; i < 300; i++) {
                        int bank = random.nextInt(this.fixture.banks.length);
                        int expert = random.nextInt(this.fixture.expertCount(bank));
                        try (HostRecord record = store.open(bank, expert)) {
                            assertArrayEquals(this.fixture.record(bank, expert), bytes(record));
                        }
                    }
                    return null;
                }));
            }
            for (Future<?> future : futures) future.get(60, TimeUnit.SECONDS);
            assertEquals(3, store.freeSlots());
            assertEquals(12 * 300, store.recordOpens());
            assertTrue(store.blockedOpens() > 0, "twelve threads on three slots must have waited");
        }
        this.gpu.assertAllReleased();
    }

    @Test
    void rejectsBanksOutsideTheFileAndOutOfRangeRecords() throws Exception {
        Path shortFile = this.directory.resolve("short.bin");
        java.nio.file.Files.write(shortFile, java.util.Arrays.copyOf(this.fixture.bytes, 1000));
        assertThrows(
                IllegalArgumentException.class, () -> new FileExpertStore(this.gpu, shortFile, this.fixture.banks, 2));
        assertEquals(0, this.gpu.liveHostAllocations());
        assertThrows(
                IllegalArgumentException.class,
                () -> new FileExpertStore(this.gpu, this.fixture.file, this.fixture.banks, 0));
        try (FileExpertStore store = new FileExpertStore(this.gpu, this.fixture.file, this.fixture.banks, 2)) {
            assertThrows(IndexOutOfBoundsException.class, () -> store.open(1, 3));
            assertThrows(IndexOutOfBoundsException.class, () -> store.open(9, 0));
            assertFalse(store.freeSlots() != 2, "a rejected request takes no slot");
        }
    }
}
