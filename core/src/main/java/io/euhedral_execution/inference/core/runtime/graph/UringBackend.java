package io.euhedral_execution.inference.core.runtime.graph;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.nio.file.Path;

/// Linux's reads: a ring per submitting thread (io_uring, through liburing), files opened with `open`, `O_DIRECT`
/// past the page cache. Its native functions are looked up when the class initializes, which only Linux does.
final class UringBackend implements ReadBackend {

    /// Entries of each ring's submission queue.
    private static final int ENTRIES = 1024;

    private static final Linker LINKER = Linker.nativeLinker();
    private static final MethodHandle QUEUE_INIT;
    private static final MethodHandle QUEUE_EXIT;
    private static final MethodHandle GET_SQE;
    private static final MethodHandle PREP_READ;
    private static final MethodHandle SET_DATA;
    private static final MethodHandle SUBMIT;
    private static final MethodHandle SQ_READY;
    private static final MethodHandle PREP_NOP;
    private static final MethodHandle PEEK_CQE;
    private static final MethodHandle CQE_SEEN;
    private static final MethodHandle OPEN;
    private static final MethodHandle CLOSE;
    private static final MethodHandle PREAD;
    private static final Throwable UNAVAILABLE;

    static {
        MethodHandle init = null,
                exit = null,
                sqe = null,
                prep = null,
                data = null,
                submit = null,
                ready = null,
                nop = null,
                peek = null,
                seen = null,
                open = null,
                close = null,
                pread = null;
        Throwable unavailable = null;
        try {
            SymbolLookup uring = SymbolLookup.libraryLookup("liburing-ffi.so.2", Arena.global());
            SymbolLookup libc = LINKER.defaultLookup();
            init = handle(
                    uring,
                    "io_uring_queue_init",
                    FunctionDescriptor.of(
                            ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT));
            exit = handle(uring, "io_uring_queue_exit", FunctionDescriptor.ofVoid(ValueLayout.ADDRESS));
            sqe = handle(uring, "io_uring_get_sqe", FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS));
            prep = handle(
                    uring,
                    "io_uring_prep_read",
                    FunctionDescriptor.ofVoid(
                            ValueLayout.ADDRESS,
                            ValueLayout.JAVA_INT,
                            ValueLayout.ADDRESS,
                            ValueLayout.JAVA_INT,
                            ValueLayout.JAVA_LONG));
            data = handle(
                    uring,
                    "io_uring_sqe_set_data64",
                    FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.JAVA_LONG));
            submit = handle(uring, "io_uring_submit", FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
            ready = handle(
                    uring, "io_uring_sq_ready", FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
            nop = handle(uring, "io_uring_prep_nop", FunctionDescriptor.ofVoid(ValueLayout.ADDRESS));
            peek = handle(
                    uring,
                    "io_uring_peek_cqe",
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
            seen = handle(
                    uring, "io_uring_cqe_seen", FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.ADDRESS));
            open = LINKER.downcallHandle(
                    libc.find("open").orElseThrow(),
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT),
                    Linker.Option.firstVariadicArg(2));
            close = LINKER.downcallHandle(
                    libc.find("close").orElseThrow(),
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_INT));
            pread = LINKER.downcallHandle(
                    libc.find("pread").orElseThrow(),
                    FunctionDescriptor.of(
                            ValueLayout.JAVA_LONG,
                            ValueLayout.JAVA_INT,
                            ValueLayout.ADDRESS,
                            ValueLayout.JAVA_LONG,
                            ValueLayout.JAVA_LONG));
        } catch (Throwable missing) {
            unavailable = missing;
        }
        QUEUE_INIT = init;
        QUEUE_EXIT = exit;
        GET_SQE = sqe;
        PREP_READ = prep;
        SET_DATA = data;
        SUBMIT = submit;
        SQ_READY = ready;
        PREP_NOP = nop;
        PEEK_CQE = peek;
        CQE_SEEN = seen;
        OPEN = open;
        CLOSE = close;
        PREAD = pread;
        UNAVAILABLE = unavailable;
    }

    private static MethodHandle handle(SymbolLookup library, String name, FunctionDescriptor descriptor) {
        return LINKER.downcallHandle(library.find(name).orElseThrow(), descriptor);
    }

    /// Why this machine has no io_uring reads (no liburing), or null.
    static Throwable unavailable() {
        return UNAVAILABLE;
    }

    /// `io_uring_submit` itself.
    static int submitNatively(MemorySegment ring) throws Throwable {
        return (int) SUBMIT.invokeExact(ring);
    }

    /// The tag of a prepared entry whose submission failed: a no-op the ring may still send, whose completion the
    /// reaper drops.
    private static final long DEFUSED = -1L;

    /// Linux's `O_DIRECT`: reads go from the device to the destination, past the page cache. Every read of such a
    /// descriptor must be aligned to the device's logical block (offset, length and address).
    private static final int O_DIRECT = 0x4000;

    private final AsyncReads.Submitter submitter;
    private final Arena arena = Arena.ofShared();

    UringBackend(AsyncReads.Submitter submitter) {
        this.submitter = submitter;
    }

    @Override
    public boolean sharedQueue() {
        return false;
    }

    @Override
    public Queue queue(int tags) {
        return new Ring(this.arena.allocate(1024, 64), this.arena.allocate(ValueLayout.ADDRESS));
    }

    @Override
    public int open(Path file, boolean direct) throws IOException {
        int fd;
        try (Arena path = Arena.ofConfined()) {
            fd = (int) OPEN.invokeExact(path.allocateFrom(file.toString()), direct ? O_DIRECT : 0);
        } catch (Throwable failure) {
            throw new IOException("opening " + file + " failed", failure);
        }
        if (fd < 0) throw new IOException("opening " + file + " failed");
        return fd;
    }

    @Override
    public void close(int file) {
        try {
            int ignored = (int) CLOSE.invokeExact(file);
        } catch (Throwable failure) {
            throw new IllegalStateException("closing a file failed", failure);
        }
    }

    @Override
    public void readBlocking(int file, long address, long length, long offset) throws IOException {
        long done = 0;
        while (done < length) {
            long read;
            try {
                read = (long)
                        PREAD.invokeExact(file, MemorySegment.ofAddress(address + done), length - done, offset + done);
            } catch (Throwable failure) {
                throw new IOException("pread failed", failure);
            }
            if (read < 0) throw new IOException("pread of " + length + " bytes at " + offset + " failed");
            if (read == 0) throw new IOException("the file ends before byte " + (offset + length));
            done += read;
        }
    }

    @Override
    public IOException failure(int error) {
        return new IOException("read failed with errno " + error);
    }

    @Override
    public void close() {
        this.arena.close();
    }

    /// One thread's ring: that thread is its only submitter; the sink reaps it.
    private final class Ring implements Queue {
        final MemorySegment ring;
        final MemorySegment cqe;

        Ring(MemorySegment ring, MemorySegment cqe) {
            this.ring = ring;
            this.cqe = cqe;
            int status;
            try {
                status = (int) QUEUE_INIT.invokeExact(ENTRIES, this.ring, 0);
            } catch (Throwable failure) {
                throw new IllegalStateException("io_uring setup failed", failure);
            }
            if (status < 0) throw new IllegalStateException("io_uring setup failed with errno " + -status);
        }

        /// Puts the read on the ring and submits it. True when the kernel took every entry the ring had
        /// prepared, this one included. Otherwise this entry, which the kernel takes in order and so has not
        /// taken, is turned into a no-op tagged [#DEFUSED] before returning false: the ring's next
        /// submission may still send it, and it then neither reads into memory its caller has moved on from
        /// nor completes a read it no longer owns.
        @Override
        public boolean submit(int file, long address, int length, long offset, long tag) {
            MemorySegment sqe;
            try {
                sqe = (MemorySegment) GET_SQE.invokeExact(this.ring);
            } catch (Throwable failure) {
                throw new IllegalStateException("io_uring submission failed", failure);
            }
            if (sqe.address() == 0) return false;
            boolean taken = false;
            RuntimeException thrown = null;
            try {
                PREP_READ.invokeExact(sqe, file, MemorySegment.ofAddress(address), length, offset);
                SET_DATA.invokeExact(sqe, tag);
                int prepared = (int) SQ_READY.invokeExact(this.ring);
                int submitted = UringBackend.this.submitter.submit(this.ring);
                taken = submitted >= prepared;
                return taken;
            } catch (Throwable failure) {
                thrown = new IllegalStateException("io_uring submission failed", failure);
                throw thrown;
            } finally {
                if (!taken) {
                    try {
                        defuse(sqe);
                    } catch (RuntimeException defuseFailure) {
                        // The submission's own failure stays the one reported.
                        if (thrown == null) throw defuseFailure;
                        thrown.addSuppressed(defuseFailure);
                    }
                }
            }
        }

        @Override
        public boolean reap(Completion into) {
            while (true) {
                int status;
                try {
                    status = (int) PEEK_CQE.invokeExact(this.ring, this.cqe);
                } catch (Throwable failure) {
                    throw new IllegalStateException("io_uring completion failed", failure);
                }
                if (status != 0) return false;
                MemorySegment cqe = this.cqe.get(ValueLayout.ADDRESS, 0).reinterpret(16);
                long tag = cqe.get(ValueLayout.JAVA_LONG, 0);
                int result = cqe.get(ValueLayout.JAVA_INT, 8);
                try {
                    CQE_SEEN.invokeExact(this.ring, cqe);
                } catch (Throwable failure) {
                    throw new IllegalStateException("io_uring completion failed", failure);
                }
                // A defused entry's no-op: its read was handed back when its submission failed.
                if (tag == DEFUSED) continue;
                into.tag = tag;
                into.bytes = Math.max(result, 0);
                into.error = result < 0 ? -result : 0;
                return true;
            }
        }

        @Override
        public boolean retired() {
            return false;
        }

        @Override
        public void close() {
            try {
                QUEUE_EXIT.invokeExact(this.ring);
            } catch (Throwable failure) {
                throw new IllegalStateException("io_uring teardown failed", failure);
            }
        }
    }

    /// Rewriting an entry the kernel has not taken is safe only because no ring polls its queue from a kernel
    /// thread (no SQPOLL): the kernel reads entries when a submission enters it.
    private static void defuse(MemorySegment sqe) {
        try {
            PREP_NOP.invokeExact(sqe);
            SET_DATA.invokeExact(sqe, DEFUSED);
        } catch (Throwable failure) {
            throw new IllegalStateException("io_uring submission failed", failure);
        }
    }
}
