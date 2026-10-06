package io.euhedral_execution.inference.core.scheduling.graph;

import io.euhedral_execution.core.frames.AbstractFrame;
import io.euhedral_execution.core.generics.LatticeReceiver;
import io.euhedral_execution.core.generics.LatticeSource;
import io.euhedral_execution.core.ingest.AbstractIngestSink;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.nio.file.Path;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReferenceArray;
import java.util.function.Consumer;
import java.util.function.Function;

/// File reads that do not hold a worker: a worker submits the read to the kernel (io_uring) and goes on, and
/// the read's completion is a frame, emitted by this sink when the lattice's workers poll it, like any other
/// ready work. It is the disk's counterpart of a device-completion callback that publishes a frame.
///
/// Every thread that submits has a ring of its own (it is the ring's only submitter); the sink reaps the
/// completions of every ring when a worker polls it, one poll at a time. A read is complete when all its bytes
/// arrived (a short read is resubmitted for the rest by the poll that reaped it) or it failed; its frame then
/// carries the result. Reads go through the page cache, as positional reads do.
public final class AsyncReads extends AbstractIngestSink implements AutoCloseable {

    /// The completion of a read: a frame of its own, emitted when the read ended.
    public abstract static class Read extends AbstractFrame {
        private int fd;
        private long address;
        private long remaining;
        private long offset;
        private int failure;
        long submittedAt;
        long completedAt;

        protected Read(long routingSeed) {
            super(FrameSeeds.ID_HASH);
            randomizeHash(routingSeed);
        }

        /// The read failed: the error the kernel reported. Read by the completion frame.
        protected final IOException failure() {
            return this.failure == 0 ? null : new IOException("read failed with errno " + this.failure);
        }

        /// Nanoseconds from the submission to the last of the read's bytes arriving.
        public final long readNanos() {
            return this.completedAt - this.submittedAt;
        }
    }

    private static final int ENTRIES = 1024;
    private static final int MAX_RINGS = 256;
    private static final int PENDING = 1 << 16;

    private static final Linker LINKER = Linker.nativeLinker();
    private static final MethodHandle QUEUE_INIT;
    private static final MethodHandle QUEUE_EXIT;
    private static final MethodHandle GET_SQE;
    private static final MethodHandle PREP_READ;
    private static final MethodHandle SET_DATA;
    private static final MethodHandle SUBMIT;
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

    /// One thread's ring: that thread is its only submitter; the sink reaps it.
    private static final class Ring {
        final MemorySegment ring;
        final MemorySegment cqe;
        boolean closed;

        Ring(Arena arena) {
            this.ring = arena.allocate(1024, 64);
            this.cqe = arena.allocate(ValueLayout.ADDRESS);
            int status;
            try {
                status = (int) QUEUE_INIT.invokeExact(ENTRIES, this.ring, 0);
            } catch (Throwable failure) {
                throw new IllegalStateException("io_uring setup failed", failure);
            }
            if (status < 0) throw new IllegalStateException("io_uring setup failed with errno " + -status);
        }
    }

    private final Arena arena = Arena.ofShared();
    private final AtomicReferenceArray<Ring> rings = new AtomicReferenceArray<>(MAX_RINGS);
    private final AtomicInteger ringCount = new AtomicInteger();
    private final ThreadLocal<Ring> mine = new ThreadLocal<>();
    private final AtomicReferenceArray<Read> pending = new AtomicReferenceArray<>(PENDING);
    private final AtomicLong tags = new AtomicLong();
    private final AtomicInteger inFlight = new AtomicInteger();
    private final Delegate delegate = new Delegate();
    private volatile boolean closed;

    private AsyncReads() {}

    /// Asynchronous reads, or an exception naming why this machine has none (no io_uring, or no liburing).
    public static AsyncReads open() {
        if (UNAVAILABLE != null)
            throw new UnsupportedOperationException("asynchronous reads are unavailable", UNAVAILABLE);
        AsyncReads reads = new AsyncReads();
        // A ring proves the kernel allows io_uring before anything relies on it.
        reads.ring();
        return reads;
    }

    /// Whether this machine can read asynchronously.
    public static boolean available() {
        return UNAVAILABLE == null;
    }

    /// Reads `length` bytes at `offset` of `fd` into `address`, blocking the calling thread until they are in. For
    /// threads that may block (the startup fill), never a lattice worker. A descriptor opened past the page cache
    /// needs the offset, the length and the address aligned to the device's block.
    public static void readBlocking(int fd, long address, long length, long offset) throws IOException {
        long done = 0;
        while (done < length) {
            long read;
            try {
                read = (long)
                        PREAD.invokeExact(fd, MemorySegment.ofAddress(address + done), length - done, offset + done);
            } catch (Throwable failure) {
                throw new IOException("pread failed", failure);
            }
            if (read < 0) throw new IOException("pread of " + length + " bytes at " + offset + " failed");
            if (read == 0) throw new IOException("the file ends before byte " + (offset + length));
            done += read;
        }
    }

    /// A descriptor of `file`, open for reading, for [#submit]. Closed with [#closeFile].
    public int openFile(Path file) throws IOException {
        return openFile(file, false);
    }

    /// Linux's `O_DIRECT`: reads go from the device to the destination, past the page cache. Every read of such a
    /// descriptor must be aligned to the device's logical block (offset, length and address).
    private static final int O_DIRECT = 0x4000;

    /// A descriptor of `file`, open for reading, past the page cache when `direct`.
    public int openFile(Path file, boolean direct) throws IOException {
        int fd;
        try (Arena path = Arena.ofConfined()) {
            fd = (int) OPEN.invokeExact(path.allocateFrom(file.toString()), direct ? O_DIRECT : 0);
        } catch (Throwable failure) {
            throw new IOException("opening " + file + " failed", failure);
        }
        if (fd < 0) throw new IOException("opening " + file + " failed");
        return fd;
    }

    public void closeFile(int fd) {
        try {
            int ignored = (int) CLOSE.invokeExact(fd);
        } catch (Throwable failure) {
            throw new IllegalStateException("closing a file failed", failure);
        }
    }

    /// The calling thread's ring, made on first use.
    private Ring ring() {
        Ring ring = this.mine.get();
        if (ring != null) return ring;
        int index = this.ringCount.getAndIncrement();
        if (index >= MAX_RINGS) return null;
        ring = new Ring(this.arena);
        this.mine.set(ring);
        // The reaper skips a slot whose ring is not published yet.
        this.rings.set(index, ring);
        return ring;
    }

    /// Submits the read of `length` bytes of `fd` at `offset` into `address`; `done` is emitted when they all
    /// arrived or the read failed. Returns false, having submitted nothing, when the read cannot be made
    /// asynchronously now: the caller reads synchronously instead. Any thread; never blocks.
    public boolean submit(int fd, long address, long length, long offset, Read done) {
        if (this.closed || length <= 0) return false;
        done.fd = fd;
        done.address = address;
        done.remaining = length;
        done.offset = offset;
        done.failure = 0;
        done.submittedAt = System.nanoTime();
        Ring ring = ring();
        if (ring == null) return false;
        long tag = this.tags.getAndIncrement();
        int slot = (int) (tag & (PENDING - 1));
        if (!this.pending.compareAndSet(slot, null, done)) return false;
        this.inFlight.incrementAndGet();
        if (!enqueue(ring, done, slot)) {
            this.pending.set(slot, null);
            this.inFlight.decrementAndGet();
            return false;
        }
        return true;
    }

    /// Puts the read's next piece on `ring` and submits it.
    private static boolean enqueue(Ring ring, Read read, int slot) {
        try {
            MemorySegment sqe = (MemorySegment) GET_SQE.invokeExact(ring.ring);
            if (sqe.address() == 0) return false;
            int length = (int) Math.min(read.remaining, Integer.MAX_VALUE & ~4095L);
            PREP_READ.invokeExact(sqe, read.fd, MemorySegment.ofAddress(read.address), length, read.offset);
            SET_DATA.invokeExact(sqe, (long) slot);
            int submitted = (int) SUBMIT.invokeExact(ring.ring);
            return submitted >= 1;
        } catch (Throwable failure) {
            throw new IllegalStateException("io_uring submission failed", failure);
        }
    }

    /// Reaps finished reads of every ring and hands the completions to `emit`, at most `limit` of them.
    private long reap(Consumer<AbstractFrame> emit, long limit) {
        long emitted = 0;
        int count = Math.min(this.ringCount.get(), MAX_RINGS);
        for (int index = 0; index < count && emitted < limit; index++) {
            Ring ring = this.rings.get(index);
            if (ring == null || ring.closed) continue;
            while (emitted < limit) {
                int status;
                try {
                    status = (int) PEEK_CQE.invokeExact(ring.ring, ring.cqe);
                } catch (Throwable failure) {
                    throw new IllegalStateException("io_uring completion failed", failure);
                }
                if (status != 0) break;
                MemorySegment cqe = ring.cqe.get(ValueLayout.ADDRESS, 0).reinterpret(16);
                int slot = (int) cqe.get(ValueLayout.JAVA_LONG, 0);
                int result = cqe.get(ValueLayout.JAVA_INT, 8);
                try {
                    CQE_SEEN.invokeExact(ring.ring, cqe);
                } catch (Throwable failure) {
                    throw new IllegalStateException("io_uring completion failed", failure);
                }
                Read read = this.pending.get(slot);
                if (result > 0 && result < read.remaining) {
                    // A short read: the rest is read on this thread's ring, under the same slot.
                    read.address += result;
                    read.offset += result;
                    read.remaining -= result;
                    Ring here = ring();
                    if (here != null && enqueue(here, read, slot)) continue;
                    result = -5;
                }
                this.pending.set(slot, null);
                this.inFlight.decrementAndGet();
                if (result < 0) read.failure = -result;
                else if (result == 0) read.failure = 5;
                read.completedAt = System.nanoTime();
                emit.accept(read);
                emitted++;
            }
        }
        return emitted;
    }

    /// Reads submitted and not yet emitted. Any thread.
    public int inFlight() {
        return this.inFlight.get();
    }

    @Override
    public LatticeSource getDelegate() {
        return this.delegate;
    }

    @Override
    public void complete() {
        this.delegate.complete();
    }

    @Override
    public boolean isComplete() {
        return this.delegate.isComplete();
    }

    /// Stops new reads and releases the rings. No read may be in flight.
    @Override
    public void close() {
        if (this.closed) return;
        this.closed = true;
        complete();
        int count = Math.min(this.ringCount.get(), MAX_RINGS);
        for (int index = 0; index < count; index++) {
            Ring ring = this.rings.get(index);
            if (ring == null || ring.closed) continue;
            ring.closed = true;
            try {
                QUEUE_EXIT.invokeExact(ring.ring);
            } catch (Throwable failure) {
                throw new IllegalStateException("io_uring teardown failed", failure);
            }
        }
        this.arena.close();
    }

    /// The lattice calls `request` and `pull` on a registered source one thread at a time, so the rings'
    /// completion queues are reaped by one thread at a time.
    private final class Delegate extends AbstractIngestSink.Delegate {
        @Override
        public long hookOnPull(
                Consumer<AbstractFrame> consumer, Function<AbstractFrame, Boolean> stopCondition, long demand) {
            return reap(consumer, demand);
        }

        @Override
        public void hookOnRequest(LatticeReceiver terminal, long demand) {
            long emitted = reap(terminal::push, demand);
            if (emitted > 0) addAndGetDemand(-emitted);
        }
    }
}
