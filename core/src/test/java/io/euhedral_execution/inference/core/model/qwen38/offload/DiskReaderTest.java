package io.euhedral_execution.inference.core.model.qwen38.offload;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.inference.core.runtime.graph.AsyncReads;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/// The disk reader against a fake device played by the test thread: it waits for a slot's flag (polling the reader
/// and the reads, as the lattice's workers do), checks the slot against the file, and lowers the flag. Every test runs
/// with asynchronous reads (where the machine has them) and with positional ones.
class DiskReaderTest {

    private static final long SLOT_BYTES = 64 << 10;
    private static final int FILE_BYTES = 1 << 20;

    @TempDir
    Path directory;

    private final Arena arena = Arena.ofShared();
    private byte[] contents;
    private Path file;
    private AsyncReads reads;
    private DiskReader reader;
    private DiskWindow window;

    @AfterEach
    void close() {
        if (this.reader != null) this.reader.close();
        if (this.reads != null) this.reads.close();
        this.arena.close();
    }

    private void open(boolean async) throws Exception {
        if (async) assumeTrue(AsyncReads.available(), "the machine has no asynchronous reads");
        this.contents = new byte[FILE_BYTES];
        for (int index = 0; index < FILE_BYTES; index++) this.contents[index] = (byte) (index * 31 + (index >>> 9));
        this.file = this.directory.resolve("artifact.edrl");
        Files.write(this.file, this.contents);
        MemorySegment slots = this.arena.allocate(SLOT_BYTES * DiskWindow.SLOTS, 4096);
        MemorySegment flags = this.arena.allocate(4096, 4096);
        this.window = new DiskWindow(slots, flags, SLOT_BYTES);
        this.reads = async ? AsyncReads.open() : null;
        if (this.reads != null)
            this.reads.getDelegate().addDownstream(new io.euhedral_execution.core.generics.LatticeReceiver() {
                @Override
                public void addUpstream(io.euhedral_execution.core.generics.LatticeSource upstream) {}

                @Override
                public void push(io.euhedral_execution.core.frames.AbstractFrame frame) {
                    frame.execute();
                    frame.doFinally();
                }

                @Override
                public void onComplete() {}

                @Override
                public void onError(Throwable error) {
                    throw new AssertionError(error);
                }
            });
        this.reader = new DiskReader(this.window, this.reads, this.file);
    }

    /// Lets the reader and the reads' completions run once, as a worker's poll of each source does.
    private void pump() {
        this.reader.poll();
        if (this.reads != null)
            this.reads
                    .getDelegate()
                    .pull(
                            frame -> {
                                frame.execute();
                                frame.doFinally();
                            },
                            frame -> false,
                            Long.MAX_VALUE);
    }

    /// The device's wait: pumps until `slot`'s flag is raised.
    private void awaitRaised(int slot) {
        long deadline = System.nanoTime() + 10_000_000_000L;
        while (this.window.flag(slot) != 1) {
            assertTrue(System.nanoTime() < deadline, "slot " + slot + " was never raised");
            pump();
        }
    }

    /// The device's copy of `use`: waits for its slot, checks the payload, lowers the flag.
    private void consume(DiskUse use) {
        awaitRaised(use.slot());
        assertPayload(use);
        this.window.lower(use.slot());
    }

    private void assertPayload(DiskUse use) {
        byte[] slot = MemorySegment.ofAddress(this.window.slotHost(use.slot()))
                .reinterpret(SLOT_BYTES)
                .asSlice(use.lead(), use.byteSize())
                .toArray(ValueLayout.JAVA_BYTE);
        byte[] expected = java.util.Arrays.copyOfRange(
                this.contents, (int) use.fileOffset(), (int) (use.fileOffset() + use.byteSize()));
        assertArrayEquals(expected, slot, use.name());
    }

    /// `count` uses at distinct offsets, slot `first + index` round robin.
    private static List<DiskUse> uses(String quantum, int count, int first, long offset) {
        List<DiskUse> uses = new ArrayList<>();
        for (int index = 0; index < count; index++)
            uses.add(new DiskUse(
                    quantum + "/" + index,
                    offset + 12_288L * index + 512L * (index % 3),
                    9_000 + 100 * index,
                    (first + index) % DiskWindow.SLOTS));
        return uses;
    }

    @ParameterizedTest
    @ValueSource(booleans = {true, false})
    void usesFillSlotsInOrderAndWaitForTheDeviceToFreeThem(boolean async) throws Exception {
        open(async);
        List<DiskUse> uses = uses("q1", 20, 0, 0);
        this.reader.enqueue(1, uses);
        for (DiskUse use : uses) {
            awaitRaised(use.slot());
            // A raised slot is the device's until it lowers the flag: more polls never refill it.
            for (int poll = 0; poll < 50; poll++) pump();
            assertPayload(use);
            this.window.lower(use.slot());
        }
        for (int poll = 0; poll < 10; poll++) pump();
        for (int slot = 0; slot < DiskWindow.SLOTS; slot++) assertEquals(0, this.window.flag(slot));
        assertEquals(0, this.reader.readsInFlight());
    }

    @ParameterizedTest
    @ValueSource(booleans = {true, false})
    void alignedSupersetCoversUnalignedPayloads(boolean async) throws Exception {
        open(async);
        List<DiskUse> uses = List.of(
                new DiskUse("first", 256, 10_000, 0),
                new DiskUse("second", 4096 + 512, 10_000, 1),
                new DiskUse("aligned", 8 * 4096, 4096, 2),
                // Its aligned blocks would run past the end of the file.
                new DiskUse("last", FILE_BYTES - 5_000, 5_000, 3));
        assertEquals(256, uses.get(0).lead());
        assertEquals(0, uses.get(0).alignedOffset());
        assertEquals(12_288, uses.get(0).readBytes());
        assertEquals(4096, uses.get(1).alignedOffset());
        assertEquals(0, uses.get(2).lead());
        this.reader.enqueue(1, uses);
        for (DiskUse use : uses) consume(use);
    }

    @ParameterizedTest
    @ValueSource(booleans = {true, false})
    void successiveQuantaShareTheWindowInAdmissionOrder(boolean async) throws Exception {
        open(async);
        // The draft after a decode: its uses start on slots the decode still has queued.
        List<DiskUse> first = uses("q1", 12, 0, 0);
        List<DiskUse> second = uses("q2", 9, 3, 400_000);
        this.reader.enqueue(1, first);
        this.reader.enqueue(2, second);
        for (DiskUse use : first) consume(use);
        for (DiskUse use : second) consume(use);
    }

    @ParameterizedTest
    @ValueSource(booleans = {true, false})
    void abandonDropsTheFailedQuantumsUses(boolean async) throws Exception {
        open(async);
        List<DiskUse> failed = uses("q1", 12, 0, 0);
        this.reader.enqueue(1, failed);
        for (DiskUse use : failed.subList(0, 5)) consume(use);
        this.reader.abandon(1);
        List<DiskUse> next = uses("q2", 3, 5, 600_000);
        this.reader.enqueue(2, next);
        // The failed quantum's retirement drains its remaining uses on the device, in order, without reading them.
        for (DiskUse use : failed.subList(5, 12)) {
            awaitRaised(use.slot());
            this.window.lower(use.slot());
        }
        // The next quantum's waits see its own bytes, never the failed quantum's.
        for (DiskUse use : next) consume(use);
        for (int poll = 0; poll < 10; poll++) pump();
        assertEquals(0, this.reader.readsInFlight());
    }

    @ParameterizedTest
    @ValueSource(booleans = {true, false})
    void aQuantumAbandonedBeforeItsRegistrationReadsNothing(boolean async) throws Exception {
        open(async);
        this.reader.abandon(1);
        // Uses past the end of the file: a read of any of them would fail.
        List<DiskUse> abandoned = List.of(
                new DiskUse("beyond/0", 4L * FILE_BYTES, 4096, 0), new DiskUse("beyond/1", 5L * FILE_BYTES, 4096, 1));
        AtomicReference<Throwable> failure = new AtomicReference<>();
        this.reader.enqueue(1, abandoned, failure::set);
        for (DiskUse use : abandoned) {
            awaitRaised(use.slot());
            this.window.lower(use.slot());
        }
        assertEquals(null, failure.get());
        List<DiskUse> next = uses("q2", 4, 0, 0);
        this.reader.enqueue(2, next);
        for (DiskUse use : next) consume(use);
    }

    @ParameterizedTest
    @ValueSource(booleans = {true, false})
    void aFailedReadFailsItsQuantumAndStillRaisesTheFlag(boolean async) throws Exception {
        open(async);
        AtomicReference<Throwable> failure = new AtomicReference<>();
        this.reader.enqueue(1, List.of(new DiskUse("beyond", 2L * FILE_BYTES, 4096, 0)), failure::set);
        awaitRaised(0);
        assertTrue(failure.get() != null, "the quantum heard of the failed read");
        this.window.lower(0);
        List<DiskUse> next = uses("q2", 2, 0, 0);
        this.reader.enqueue(2, next);
        for (DiskUse use : next) consume(use);
    }

    @ParameterizedTest
    @ValueSource(booleans = {true, false})
    void aClosedReaderReleasesTheWaitsTheDeviceStillHolds(boolean async) throws Exception {
        open(async);
        // The device holds four flag waits; only the first two belong to registered uses.
        int[] waits = {0, 1, 0, 2};
        this.reader.enqueue(1, List.of(new DiskUse("a", 0, 4096, 0), new DiskUse("b", 8192, 4096, 1)));
        java.util.concurrent.atomic.AtomicInteger done = new java.util.concurrent.atomic.AtomicInteger();
        Thread device = Thread.ofPlatform().daemon().start(() -> {
            for (int slot : waits) {
                while (this.window.flag(slot) != 1) Thread.onSpinWait();
                this.window.lower(slot);
                done.incrementAndGet();
            }
        });
        long deadline = System.nanoTime() + 10_000_000_000L;
        while (done.get() < 2) {
            assertTrue(System.nanoTime() < deadline, "the registered uses were never read");
            pump();
        }
        this.reader.close();
        int released = this.reader.releaseDevice(100_000_000L, 5_000_000_000L);
        device.join(5_000);
        assertEquals(4, done.get(), "the device drained every wait it held");
        assertEquals(2, released);
        assertEquals(0, this.reader.releaseDevice(20_000_000L, 1_000_000_000L), "nothing waits any more");
    }
}
