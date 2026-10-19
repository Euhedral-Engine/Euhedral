package io.euhedral_execution.inference.core.runtime.graph;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.core.frames.AbstractFrame;
import io.euhedral_execution.core.generics.LatticeReceiver;
import io.euhedral_execution.core.generics.LatticeSource;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.function.BooleanSupplier;
import org.junit.jupiter.api.Test;

/// The backend each system reads through, and what [AsyncReads] does above any backend: a stand-in backend reads
/// from memory, so these run on every machine.
class ReadBackendTest {

    private static final BooleanSupplier NOT_ON_THIS_SYSTEM = () -> {
        throw new AssertionError("Windows' I/O rings were probed on a system other than Windows");
    };

    @Test
    void linuxReadsThroughIoUringWithoutProbingWindows() {
        assertEquals(ReadBackend.Kind.IO_URING, ReadBackend.choose("Linux", NOT_ON_THIS_SYSTEM));
    }

    @Test
    void windowsReadsThroughItsRingsWhenItHasThemAndItsCompletionPortOtherwise() {
        assertEquals(ReadBackend.Kind.IORING, ReadBackend.choose("Windows 11", () -> true));
        assertEquals(ReadBackend.Kind.COMPLETION_PORT, ReadBackend.choose("Windows 10", () -> false));
        assertEquals(ReadBackend.Kind.IORING, ReadBackend.choose("Windows Server 2022", () -> true));
    }

    @Test
    void otherSystemsHaveNoBackendAndSaySo() {
        assertNull(ReadBackend.choose("Mac OS X", NOT_ON_THIS_SYSTEM));
        assertNull(ReadBackend.choose("FreeBSD", NOT_ON_THIS_SYSTEM));
        assertNull(ReadBackend.choose(null, NOT_ON_THIS_SYSTEM));
        String reason = ReadBackend.unsupported("Mac OS X");
        assertTrue(reason.contains("Linux") && reason.contains("Windows") && reason.contains("Mac OS X"), reason);
    }

    /// The class initializes on any system: nothing of another system's is looked up.
    @Test
    void thisMachineReadsThroughItsOwnSystemsBackend() {
        ReadBackend.Kind expected = ReadBackend.choose(System.getProperty("os.name"), () -> false);
        if (AsyncReads.available()) {
            assertNull(AsyncReads.unavailableReason());
            ReadBackend.Kind kind = AsyncReads.kind();
            assertNotNull(kind);
            if (expected == ReadBackend.Kind.IO_URING) assertEquals(ReadBackend.Kind.IO_URING, kind);
            assertEquals(kind.toString(), AsyncReads.backend());
        } else {
            assertNull(AsyncReads.backend());
            assertNotNull(AsyncReads.unavailableReason());
        }
    }

    // ---------------------------------------------------------------- above any backend

    /// A read the stand-in queue took: completed when the queue is reaped.
    private record Taken(int file, long address, int length, long offset, long tag) {}

    /// Reads from byte arrays. Each queue completes its reads in order, at most `limit` bytes each (a short read
    /// beyond), and may be told to refuse its next submission and retire.
    private static final class Memory implements ReadBackend {
        final List<byte[]> files = new ArrayList<>();
        final List<Queue> queues = new ArrayList<>();
        final boolean shared;
        int limit = Integer.MAX_VALUE;
        boolean closed;

        Memory(boolean shared) {
            this.shared = shared;
        }

        final class Stand implements Queue {
            final ConcurrentLinkedQueue<Taken> taken = new ConcurrentLinkedQueue<>();
            boolean refuseNext;
            /// The error the next completion reports in place of its read, when not 0.
            int failNext;
            boolean retired;
            boolean closed;

            @Override
            public boolean submit(int file, long address, int length, long offset, long tag) {
                if (this.retired) return false;
                if (this.refuseNext) {
                    this.refuseNext = false;
                    this.retired = true;
                    return false;
                }
                this.taken.add(new Taken(file, address, length, offset, tag));
                return true;
            }

            @Override
            public boolean reap(Completion into) {
                Taken read = this.taken.poll();
                if (read == null) return false;
                byte[] content = Memory.this.files.get(read.file());
                into.tag = read.tag();
                into.error = this.failNext;
                if (this.failNext != 0) {
                    this.failNext = 0;
                    into.bytes = 0;
                    return true;
                }
                if (read.offset() >= content.length) {
                    into.bytes = 0;
                    return true;
                }
                int bytes = (int) Math.min(Math.min(read.length(), Memory.this.limit), content.length - read.offset());
                MemorySegment.ofAddress(read.address())
                        .reinterpret(bytes)
                        .copyFrom(MemorySegment.ofArray(content).asSlice(read.offset(), bytes));
                into.bytes = bytes;
                return true;
            }

            @Override
            public boolean retired() {
                return this.retired;
            }

            @Override
            public void close() {
                this.closed = true;
            }
        }

        @Override
        public boolean sharedQueue() {
            return this.shared;
        }

        @Override
        public Queue queue(int tags) {
            Stand queue = new Stand();
            this.queues.add(queue);
            return queue;
        }

        @Override
        public int open(Path file, boolean direct) {
            throw new UnsupportedOperationException();
        }

        int add(byte[] content) {
            this.files.add(content);
            return this.files.size() - 1;
        }

        @Override
        public void close(int file) {}

        @Override
        public void readBlocking(int file, long address, long length, long offset) {
            throw new UnsupportedOperationException();
        }

        @Override
        public IOException failure(int error) {
            return new IOException("stand-in error " + error);
        }

        @Override
        public void close() {
            this.closed = true;
        }
    }

    private static class Done extends AsyncReads.Read {
        int reaped;

        Done(long seed) {
            super(seed);
        }

        @Override
        protected void reaped() {
            this.reaped++;
        }

        @Override
        public void execute() {}

        IOException failed() {
            return failure();
        }
    }

    /// Reads through `memory`, with a downstream as the lattice gives a registered source.
    private static AsyncReads open(Memory memory) {
        AsyncReads reads = AsyncReads.open(memory);
        reads.getDelegate().addDownstream(new LatticeReceiver() {
            @Override
            public void addUpstream(LatticeSource upstream) {}

            @Override
            public void push(AbstractFrame frame) {
                frame.execute();
            }

            @Override
            public void onComplete() {}

            @Override
            public void onError(Throwable error) {
                throw new AssertionError(error);
            }
        });
        return reads;
    }

    private static List<AbstractFrame> poll(AsyncReads reads) {
        List<AbstractFrame> emitted = new ArrayList<>();
        reads.getDelegate().pull(emitted::add, frame -> false, Long.MAX_VALUE);
        return emitted;
    }

    private static byte[] content(int length, long seed) {
        byte[] content = new byte[length];
        new SplittableRandom(seed).nextBytes(content);
        return content;
    }

    /// One queue for every thread: made once, at the open, and every submitter's reads complete through it.
    @Test
    void aSharedQueueServesEveryThread() throws Exception {
        Memory memory = new Memory(true);
        byte[] content = content(8 * 4096, 1);
        int file = memory.add(content);
        try (AsyncReads reads = open(memory);
                Arena arena = Arena.ofShared()) {
            MemorySegment destination = arena.allocate(content.length, 4096);
            Done[] done = new Done[8];
            Thread[] submitters = new Thread[4];
            for (int t = 0; t < submitters.length; t++) {
                int first = 2 * t;
                submitters[t] = new Thread(() -> {
                    for (int i = first; i < first + 2; i++) {
                        done[i] = new Done(i);
                        assertTrue(reads.submit(file, destination.address() + 4096L * i, 4096, 4096L * i, done[i]));
                    }
                });
                submitters[t].start();
            }
            for (Thread submitter : submitters) submitter.join();
            assertEquals(1, memory.queues.size());
            assertEquals(8, poll(reads).size());
            for (Done d : done) {
                assertEquals(1, d.reaped);
                assertNull(d.failed());
            }
            assertArrayEquals(content, destination.toArray(ValueLayout.JAVA_BYTE));
            assertEquals(0, reads.inFlight());
        }
        assertTrue(memory.closed);
        assertTrue(memory.queues.getFirst() instanceof Memory.Stand stand && stand.closed);
    }

    /// A queue that retires (Windows' rings, after a refused submission) hands its read back; its thread's next read
    /// goes to a new queue, and the retired one's reads in flight still complete.
    @Test
    void aRetiredQueueIsReplacedAndStillReaped() throws Exception {
        Memory memory = new Memory(false);
        byte[] content = content(3 * 4096, 2);
        int file = memory.add(content);
        try (AsyncReads reads = open(memory);
                Arena arena = Arena.ofConfined()) {
            assertEquals(1, memory.queues.size(), "the open proves a queue");
            Memory.Stand first = (Memory.Stand) memory.queues.getFirst();
            MemorySegment destination = arena.allocate(content.length, 4096);
            Done before = new Done(1);
            Done refused = new Done(2);
            Done after = new Done(3);
            assertTrue(reads.submit(file, destination.address(), 4096, 0, before));
            first.refuseNext = true;
            assertFalse(reads.submit(file, destination.address() + 4096, 4096, 4096, refused));
            assertEquals(1, reads.inFlight());
            assertEquals(1, reads.pendingReads());
            assertTrue(reads.submit(file, destination.address() + 8192, 4096, 8192, after));
            assertEquals(2, memory.queues.size());
            List<AbstractFrame> emitted = poll(reads);
            assertEquals(List.of(before, after), emitted);
            assertEquals(0, refused.reaped);
            assertEquals(0, reads.inFlight());
            assertEquals(0, reads.pendingReads());
            assertArrayEquals(
                    Arrays.copyOfRange(content, 0, 4096),
                    destination.asSlice(0, 4096).toArray(ValueLayout.JAVA_BYTE));
        }
    }

    /// A short read's rest is resubmitted under the same tag; the read completes once, when every byte is in.
    @Test
    void aShortReadCompletesOnceWhenItsRestArrived() throws Exception {
        Memory memory = new Memory(false);
        memory.limit = 1000;
        byte[] content = content(4096, 3);
        int file = memory.add(content);
        try (AsyncReads reads = open(memory);
                Arena arena = Arena.ofConfined()) {
            MemorySegment destination = arena.allocate(4096, 4096);
            Done done = new Done(1);
            assertTrue(reads.submit(file, destination.address(), 4096, 0, done));
            List<AbstractFrame> emitted = new ArrayList<>();
            for (int i = 0; i < 10 && emitted.isEmpty(); i++) emitted.addAll(poll(reads));
            assertEquals(List.of(done), emitted);
            assertEquals(1, done.reaped);
            assertNull(done.failed());
            assertArrayEquals(content, destination.toArray(ValueLayout.JAVA_BYTE));
        }
    }

    /// A read that finds the end of the file, or that the backend fails, completes as failed with the backend's
    /// description of the error.
    @Test
    void endOfFileAndErrorsFailTheRead() throws Exception {
        Memory memory = new Memory(false);
        int file = memory.add(content(100, 4));
        try (AsyncReads reads = open(memory);
                Arena arena = Arena.ofConfined()) {
            MemorySegment destination = arena.allocate(4096, 4096);
            Done past = new Done(1);
            assertTrue(reads.submit(file, destination.address(), 4096, 8192, past));
            assertEquals(List.of(past), poll(reads));
            assertNotNull(past.failed());

            ((Memory.Stand) memory.queues.getFirst()).failNext = 5;
            Done broken = new Done(2);
            assertTrue(reads.submit(file, destination.address(), 100, 0, broken));
            assertEquals(List.of(broken), poll(reads));
            assertEquals("stand-in error 5", broken.failed().getMessage());
            assertEquals(0, reads.inFlight());
        }
    }
}
