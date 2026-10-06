package io.euhedral_execution.inference.core.scheduling.graph;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.core.frames.AbstractFrame;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/// Reads that hold no worker: submitting returns at once, and a read's completion is a frame that the sink
/// emits when it is polled, as the lattice's workers poll every sink.
class AsyncReadsTest {

    @TempDir
    Path directory;

    /// A completion that records what it saw when it ran.
    private static final class Done extends AsyncReads.Read {
        boolean ran;
        java.io.IOException failure;

        Done(long seed) {
            super(seed);
        }

        @Override
        public void execute() {
            this.ran = true;
            this.failure = failure();
        }
    }

    /// Polls the sink as a worker does, running what it emits, until `count` completions ran.
    private static List<AbstractFrame> poll(AsyncReads reads, int count) {
        List<AbstractFrame> emitted = new ArrayList<>();
        long deadline = System.nanoTime() + 10_000_000_000L;
        while (emitted.size() < count) {
            if (System.nanoTime() > deadline) throw new AssertionError("the reads did not complete");
            reads.getDelegate()
                    .pull(
                            frame -> {
                                frame.execute();
                                emitted.add(frame);
                            },
                            frame -> false,
                            Long.MAX_VALUE);
        }
        return emitted;
    }

    private static AsyncReads open() {
        assumeTrue(AsyncReads.available(), "no io_uring on this machine");
        AsyncReads reads = AsyncReads.open();
        reads.getDelegate().addDownstream(new io.euhedral_execution.core.generics.LatticeReceiver() {
            @Override
            public void addUpstream(io.euhedral_execution.core.generics.LatticeSource upstream) {}

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

    @Test
    void aReadsCompletionIsAFrameEmittedWhenTheSinkIsPolledAndItsBytesArrived() throws Exception {
        byte[] content = new byte[3 * 4096 + 123];
        new SplittableRandom(3).nextBytes(content);
        Path file = this.directory.resolve("data");
        Files.write(file, content);
        try (AsyncReads reads = open();
                Arena arena = Arena.ofConfined()) {
            int fd = reads.openFile(file);
            try {
                MemorySegment first = arena.allocate(4096, 4096);
                MemorySegment rest = arena.allocate(content.length - 4096, 4096);
                Done a = new Done(1);
                Done b = new Done(2);
                assertTrue(reads.submit(fd, first.address(), first.byteSize(), 0, a));
                assertTrue(reads.submit(fd, rest.address(), rest.byteSize(), 4096, b));
                assertFalse(a.ran, "submitting runs nothing");
                List<AbstractFrame> emitted = poll(reads, 2);
                assertTrue(emitted.contains(a) && emitted.contains(b));
                assertNull(a.failure);
                assertNull(b.failure);
                assertArrayEquals(java.util.Arrays.copyOfRange(content, 0, 4096), first.toArray(ValueLayout.JAVA_BYTE));
                assertArrayEquals(
                        java.util.Arrays.copyOfRange(content, 4096, content.length),
                        rest.toArray(ValueLayout.JAVA_BYTE));
                assertEquals(0, reads.inFlight());
                assertTrue(a.readNanos() >= 0);
            } finally {
                reads.closeFile(fd);
            }
        }
    }

    @Test
    void readsFromManyThreadsEachCompleteOnce() throws Exception {
        byte[] content = new byte[64 * 4096];
        new SplittableRandom(5).nextBytes(content);
        Path file = this.directory.resolve("many");
        Files.write(file, content);
        try (AsyncReads reads = open();
                Arena arena = Arena.ofShared()) {
            int fd = reads.openFile(file);
            try {
                MemorySegment destination = arena.allocate(content.length, 4096);
                Done[] done = new Done[64];
                Thread[] submitters = new Thread[8];
                for (int t = 0; t < submitters.length; t++) {
                    int first = t * 8;
                    submitters[t] = new Thread(() -> {
                        for (int i = first; i < first + 8; i++) {
                            done[i] = new Done(i);
                            assertTrue(reads.submit(fd, destination.address() + 4096L * i, 4096, 4096L * i, done[i]));
                        }
                    });
                    submitters[t].start();
                }
                for (Thread submitter : submitters) submitter.join();
                List<AbstractFrame> emitted = poll(reads, 64);
                assertEquals(64, emitted.size());
                for (Done d : done) {
                    assertTrue(d.ran);
                    assertNull(d.failure);
                }
                assertArrayEquals(content, destination.toArray(ValueLayout.JAVA_BYTE));
            } finally {
                reads.closeFile(fd);
            }
        }
    }

    @Test
    void aDirectReadGoesPastThePageCacheAndReadsTheSameBytes() throws Exception {
        byte[] content = new byte[8 * 4096];
        new SplittableRandom(7).nextBytes(content);
        Path file = this.directory.resolve("direct");
        Files.write(file, content);
        try (AsyncReads reads = open();
                Arena arena = Arena.ofConfined()) {
            int fd;
            try {
                fd = reads.openFile(file, true);
            } catch (java.io.IOException unsupported) {
                assumeTrue(false, "the temporary directory's file system has no direct reads");
                return;
            }
            try {
                MemorySegment destination = arena.allocate(4 * 4096, 4096);
                Done done = new Done(11);
                assertTrue(reads.submit(fd, destination.address(), destination.byteSize(), 2 * 4096, done));
                poll(reads, 1);
                assertNull(done.failure);
                assertArrayEquals(
                        java.util.Arrays.copyOfRange(content, 2 * 4096, 6 * 4096),
                        destination.toArray(ValueLayout.JAVA_BYTE));
            } finally {
                reads.closeFile(fd);
            }
        }
    }

    @Test
    void aReadPastTheEndOfTheFileFails() throws Exception {
        Path file = this.directory.resolve("short");
        Files.write(file, new byte[100]);
        try (AsyncReads reads = open();
                Arena arena = Arena.ofConfined()) {
            int fd = reads.openFile(file);
            try {
                MemorySegment destination = arena.allocate(4096, 4096);
                Done done = new Done(9);
                assertTrue(reads.submit(fd, destination.address(), 4096, 8192, done));
                poll(reads, 1);
                assertNotNull(done.failure, "no bytes where the read asked for them");
            } finally {
                reads.closeFile(fd);
            }
        }
    }
}
