package io.euhedral_execution.inference.core.runtime.graph;

import io.euhedral_execution.core.frames.AbstractFrame;
import io.euhedral_execution.core.generics.LatticeReceiver;
import io.euhedral_execution.core.generics.LatticeSource;
import io.euhedral_execution.core.ingest.AbstractIngestSink;
import io.euhedral_execution.hardware_utils.util.OSName;
import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReferenceArray;
import java.util.concurrent.atomic.LongAdder;
import java.util.function.Consumer;
import java.util.function.Function;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/// File reads that do not hold a worker: a worker submits the read to the kernel and goes on, and the read's
/// completion is a frame, emitted by this sink when the lattice's workers poll it, like any other ready work. It is
/// the disk's counterpart of a device-completion callback that publishes a frame.
///
/// The kernel's side is the system's [ReadBackend], chosen when the class initializes from the system it runs on:
/// io_uring on Linux; on Windows its I/O rings, or overlapped reads and an I/O completion port where it has no rings.
/// Nothing of another system's is looked up.
///
/// Every thread that submits has a queue of its own (it is the queue's only submitter), unless the backend has one
/// for all; the sink reaps the completions of every queue when a worker polls it, one poll at a time. A read is
/// complete when all its bytes arrived (a short read is resubmitted for the rest by the poll that reaped it) or it
/// failed; its frame then carries the result. Reads go through the page cache, as positional reads do, unless the
/// file was opened past it.
public final class AsyncReads extends AbstractIngestSink implements AutoCloseable {

    /// The completion of a read: a frame of its own, emitted when the read ended.
    public abstract static class Read extends AbstractFrame {
        private int file;
        private long address;
        private long remaining;
        private long offset;
        private IOException failure;
        long submittedAt;
        long completedAt;

        protected Read(long routingSeed) {
            super(FrameSeeds.ID_HASH);
            randomizeHash(routingSeed);
        }

        /// The read failed: the error the kernel reported. Read by the completion frame.
        protected final IOException failure() {
            return this.failure;
        }

        /// Nanoseconds from the submission to the last of the read's bytes arriving.
        public final long readNanos() {
            return this.completedAt - this.submittedAt;
        }

        /// The read ended (all its bytes arrived, or it failed: [#failure()]). Runs on the worker that
        /// reaped it, before its frame is emitted. The frame may then queue behind a worker busy with
        /// another frame, so what is awaited without the lattice (a flag the device polls) is released
        /// here. It must neither block nor throw.
        protected void reaped() {}
    }

    private static final Logger LOG = LoggerFactory.getLogger(AsyncReads.class);

    /// Entries of each queue's submission ring (the most the system allows, when that is less).
    static final int RING_ENTRIES = 1024;
    private static final int MAX_RINGS = 256;
    private static final int PENDING = 1 << 16;

    /// The alignment of a read past the page cache (offset, length and address), which its callers give it.
    public static final long DIRECT_ALIGNMENT = 4096;

    /// The longest piece of a read handed to the kernel at once: the rest follows as a short read's would.
    static final long MAX_READ = Integer.MAX_VALUE & ~(DIRECT_ALIGNMENT - 1);

    /// This system's backend, or null when it has none.
    private static final ReadBackend.Kind KIND = ReadBackend.choose(OSName.CURRENT_OS, () -> {
        try {
            return IoRingBackend.usable();
        } catch (Throwable missing) {
            return false;
        }
    });

    /// Why this machine cannot read asynchronously, or null.
    private static final Throwable UNAVAILABLE = unavailability(KIND);

    private static Throwable unavailability(ReadBackend.Kind kind) {
        if (kind == null) return new UnsupportedOperationException(ReadBackend.unsupported(OSName.CURRENT_OS));
        try {
            return switch (kind) {
                case IO_URING -> UringBackend.unavailable();
                // Its probe opened a ring.
                case IORING -> null;
                case COMPLETION_PORT -> CompletionPortBackend.unavailable();
            };
        } catch (Throwable missing) {
            return missing;
        }
    }

    /// The step that hands an io_uring ring's prepared entries to the kernel: `io_uring_submit`, or a test's
    /// stand-in. A stand-in may refuse (a negated errno, or fewer entries than were prepared) or throw only before
    /// the kernel took any entry; it must not throw after the kernel took entries, nor report more entries than
    /// were prepared.
    @FunctionalInterface
    public interface Submitter {
        /// The number of entries the kernel took, or a negated errno.
        int submit(MemorySegment ring) throws Throwable;
    }

    public static final Submitter NATIVE_SUBMIT = UringBackend::submitNatively;

    private final ReadBackend backend;
    private final AtomicReferenceArray<ReadBackend.Queue> rings = new AtomicReferenceArray<>(MAX_RINGS);
    private final AtomicInteger ringCount = new AtomicInteger();
    private final ThreadLocal<ReadBackend.Queue> mine = new ThreadLocal<>();
    /// Every thread's queue, when the backend has one for all.
    private final ReadBackend.Queue shared;
    private final AtomicReferenceArray<Read> pending = new AtomicReferenceArray<>(PENDING);
    private final AtomicLong tags = new AtomicLong();
    private final AtomicInteger inFlight = new AtomicInteger();
    /// When the reads in flight last went from none to some, and the time summed over the spans with some.
    private volatile long busySince;
    private final LongAdder busyNanos = new LongAdder();
    private final Delegate delegate = new Delegate();
    /// Filled by the reaping poll, one at a time.
    private final ReadBackend.Completion completion = new ReadBackend.Completion();
    private volatile boolean closed;

    private AsyncReads(ReadBackend backend) {
        this.backend = backend;
        if (backend.sharedQueue()) {
            this.shared = backend.queue(PENDING);
            this.rings.set(0, this.shared);
            this.ringCount.set(1);
        } else {
            this.shared = null;
        }
    }

    /// Asynchronous reads, or an exception naming why this machine has none (a system other than Linux and
    /// Windows, no liburing, or no rings or port the kernel allows).
    public static AsyncReads open() {
        if (UNAVAILABLE != null)
            throw new UnsupportedOperationException("asynchronous reads are unavailable", UNAVAILABLE);
        return open(
                switch (KIND) {
                    case IO_URING -> new UringBackend(NATIVE_SUBMIT);
                    case IORING -> new IoRingBackend();
                    case COMPLETION_PORT -> new CompletionPortBackend();
                });
    }

    /// As [#open()], with `submitter` in place of `io_uring_submit` (tests). Only io_uring takes a stand-in.
    public static AsyncReads open(Submitter submitter) {
        Objects.requireNonNull(submitter, "submitter");
        if (submitter == NATIVE_SUBMIT) return open();
        if (UNAVAILABLE != null)
            throw new UnsupportedOperationException("asynchronous reads are unavailable", UNAVAILABLE);
        if (KIND != ReadBackend.Kind.IO_URING)
            throw new UnsupportedOperationException(
                    "only io_uring submissions can be replaced; this machine reads through " + KIND);
        return open(new UringBackend(submitter));
    }

    /// Asynchronous reads through `backend` (tests).
    static AsyncReads open(ReadBackend backend) {
        AsyncReads reads;
        try {
            reads = new AsyncReads(backend);
            // A queue proves the kernel allows the backend before anything relies on it.
            reads.ring();
        } catch (RuntimeException | Error failure) {
            backend.close();
            throw failure;
        }
        return reads;
    }

    /// Whether this machine can read asynchronously.
    public static boolean available() {
        return UNAVAILABLE == null;
    }

    /// Why this machine cannot read asynchronously, or null when it can.
    public static String unavailableReason() {
        if (UNAVAILABLE == null) return null;
        return UNAVAILABLE.getMessage() != null ? UNAVAILABLE.getMessage() : UNAVAILABLE.toString();
    }

    /// What this machine reads through ("io_uring", ...), or null when it cannot read asynchronously.
    public static String backend() {
        return UNAVAILABLE == null ? KIND.toString() : null;
    }

    /// The backend this machine reads through, or null (tests).
    static ReadBackend.Kind kind() {
        return UNAVAILABLE == null ? KIND : null;
    }

    /// Reads `length` bytes at `offset` of `file` into `address`, blocking the calling thread until they are in. For
    /// threads that may block (the startup fill), never a lattice worker. A file opened past the page cache needs
    /// the offset, the length and the address aligned to [#DIRECT_ALIGNMENT].
    public void readBlocking(int file, long address, long length, long offset) throws IOException {
        this.backend.readBlocking(file, address, length, offset);
    }

    /// A file of `file`, open for reading, for [#submit]. Closed with [#closeFile].
    public int openFile(Path file) throws IOException {
        return openFile(file, false);
    }

    /// A file of `file`, open for reading, past the page cache when `direct`: every read of it must then be aligned
    /// to [#DIRECT_ALIGNMENT] (offset, length and address). Throws when the file system cannot read it so.
    public int openFile(Path file, boolean direct) throws IOException {
        return this.backend.open(file, direct);
    }

    public void closeFile(int file) {
        this.backend.close(file);
    }

    /// The calling thread's queue, made on first use (and again when its queue retired); null when there may be no
    /// more queues.
    private ReadBackend.Queue ring() {
        if (this.shared != null) return this.shared;
        ReadBackend.Queue ring = this.mine.get();
        if (ring != null && !ring.retired()) return ring;
        int index = this.ringCount.getAndIncrement();
        if (index >= MAX_RINGS) return null;
        ring = this.backend.queue(PENDING);
        this.mine.set(ring);
        // The reaper skips a slot whose queue is not published yet.
        this.rings.set(index, ring);
        return ring;
    }

    /// Submits the read of `length` bytes of `file` at `offset` into `address`; `done` is emitted when they all
    /// arrived or the read failed. Returns false, having submitted nothing, when the read cannot be made
    /// asynchronously now: the caller reads synchronously instead. Any thread; never blocks.
    public boolean submit(int file, long address, long length, long offset, Read done) {
        if (this.closed || length <= 0) return false;
        done.file = file;
        done.address = address;
        done.remaining = length;
        done.offset = offset;
        done.failure = null;
        done.submittedAt = System.nanoTime();
        ReadBackend.Queue ring = ring();
        if (ring == null) return false;
        long tag = this.tags.getAndIncrement();
        int slot = (int) (tag & (PENDING - 1));
        if (!this.pending.compareAndSet(slot, null, done)) return false;
        if (this.inFlight.incrementAndGet() == 1) this.busySince = System.nanoTime();
        boolean taken;
        try {
            taken = enqueue(ring, done, slot);
        } catch (RuntimeException | Error failure) {
            handBack(slot);
            throw failure;
        }
        if (!taken) handBack(slot);
        return taken;
    }

    /// A read that was not submitted gives its pending slot and its count in flight back.
    private void handBack(int slot) {
        this.pending.set(slot, null);
        if (this.inFlight.decrementAndGet() == 0) this.busyNanos.add(System.nanoTime() - this.busySince);
    }

    /// Reads holding a pending slot: submitted and not yet reaped (tests).
    int pendingReads() {
        int count = 0;
        for (int slot = 0; slot < PENDING; slot++) if (this.pending.get(slot) != null) count++;
        return count;
    }

    /// Hands the read's next piece to `ring` under tag `slot`. True when the kernel took it.
    private static boolean enqueue(ReadBackend.Queue ring, Read read, int slot) {
        int length = (int) Math.min(read.remaining, MAX_READ);
        return ring.submit(read.file, read.address, length, read.offset, slot);
    }

    /// Reaps finished reads of every queue and hands the completions to `emit`, at most `limit` of them.
    private long reap(Consumer<AbstractFrame> emit, long limit) {
        long emitted = 0;
        ReadBackend.Completion done = this.completion;
        int count = Math.min(this.ringCount.get(), MAX_RINGS);
        for (int index = 0; index < count && emitted < limit; index++) {
            ReadBackend.Queue ring = this.rings.get(index);
            if (ring == null || this.closed) continue;
            while (emitted < limit && ring.reap(done)) {
                int slot = (int) done.tag;
                Read read = this.pending.get(slot);
                // Never expected: a read keeps its slot until it is reaped.
                if (read == null) {
                    LOG.error(
                            "a completion (tag {}, {} bytes, error {}) belongs to no pending read",
                            done.tag,
                            done.bytes,
                            done.error);
                    continue;
                }
                IOException failure = null;
                if (done.error != 0) failure = this.backend.failure(done.error);
                else if (done.bytes == 0)
                    failure = new IOException("the file ends before byte " + (read.offset + read.remaining));
                else if (done.bytes < read.remaining) {
                    // A short read: the rest is read on this thread's queue, under the same slot. A rest that cannot
                    // be submitted ends the read as failed, so it still completes.
                    read.address += done.bytes;
                    read.offset += done.bytes;
                    read.remaining -= done.bytes;
                    ReadBackend.Queue here = ring();
                    boolean resubmitted;
                    try {
                        resubmitted = here != null && enqueue(here, read, slot);
                    } catch (RuntimeException | Error submission) {
                        LOG.error("submitting the rest of a short read failed; the read fails", submission);
                        resubmitted = false;
                    }
                    if (resubmitted) continue;
                    failure = new IOException("the rest of a short read could not be submitted");
                }
                this.pending.set(slot, null);
                read.completedAt = System.nanoTime();
                if (this.inFlight.decrementAndGet() == 0) this.busyNanos.add(read.completedAt - this.busySince);
                read.failure = failure;
                read.reaped();
                emit.accept(read);
                emitted++;
            }
        }
        return emitted;
    }

    /// Time with at least one read in flight, summed (approximate: the spans are read without a lock). Any thread.
    public long busyNanos() {
        return this.busyNanos.sum();
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

    /// Stops new reads and releases the queues. No read may be in flight.
    @Override
    public void close() {
        if (this.closed) return;
        this.closed = true;
        complete();
        try {
            int count = Math.min(this.ringCount.get(), MAX_RINGS);
            for (int index = 0; index < count; index++) {
                ReadBackend.Queue ring = this.rings.getAndSet(index, null);
                if (ring != null) ring.close();
            }
        } finally {
            this.backend.close();
        }
    }

    /// The lattice calls `request` and `pull` on a registered source one thread at a time, so the queues'
    /// completions are reaped by one thread at a time.
    private final class Delegate extends AbstractIngestSink.Delegate {
        /// The terminal's `push`, made once: a request is made on every poll, and a method reference made there
        /// allocates each time.
        private LatticeReceiver pushing;
        private Consumer<AbstractFrame> push;

        @Override
        public long hookOnPull(
                Consumer<AbstractFrame> consumer, Function<AbstractFrame, Boolean> stopCondition, long demand) {
            return reap(consumer, demand);
        }

        @Override
        public void hookOnRequest(LatticeReceiver terminal, long demand) {
            if (terminal != this.pushing) {
                this.push = terminal::push;
                this.pushing = terminal;
            }
            long emitted = reap(this.push, demand);
            if (emitted > 0) addAndGetDemand(-emitted);
        }
    }
}
