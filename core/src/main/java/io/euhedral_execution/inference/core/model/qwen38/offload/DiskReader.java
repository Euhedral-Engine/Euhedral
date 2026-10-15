package io.euhedral_execution.inference.core.model.qwen38.offload;

import io.euhedral_execution.core.frames.AbstractFrame;
import io.euhedral_execution.core.generics.LatticeReceiver;
import io.euhedral_execution.core.generics.LatticeSource;
import io.euhedral_execution.core.ingest.AbstractIngestSink;
import io.euhedral_execution.data_structures.queues.MpscQueue;
import io.euhedral_execution.hashing.HasherApi;
import io.euhedral_execution.inference.core.runtime.graph.AsyncReads;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayDeque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.LockSupport;
import java.util.function.Consumer;
import java.util.function.Function;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/// Reads the disk-backed weights of admitted quanta into the [DiskWindow]: a source of the lattice, polled one
/// worker at a time, whose bookkeeping is plain fields changed only by its poll (the reads' ends, which the workers
/// that reap them mark, aside).
///
/// Each quantum registers its disk uses at admission ([#enqueue]), in the order the device consumes them. Every
/// window slot keeps its own queue of uses, in admission order and, within a quantum, in use order: the order its
/// copies run on the device, which the staging ring's edges fix. A poll gives each slot whose flag the device lowered
/// the next use of its queue: it submits the read ([AsyncReads], io_uring). The worker that reaps the read fails the
/// use's quantum if the read failed, raises the flag, hands the slot to the use's copy ([#claim]) and marks the read
/// ended, and the next poll takes the slot back; neither the device nor the poll waits for the read's completion
/// frame. Without asynchronous reads the poll reads positionally itself, one slot at a time.
///
/// The device never waits for the host: a copy from the window is queued only once its read is in the slot
/// ([#claim]), so every stream drains on its own, and a worker that a CUDA call holds while a stream is full is
/// released as the device runs. The device lowers the slot's flag after the copy, which frees the slot for its next
/// read.
///
/// A quantum that stops before its device work consumed every use is abandoned ([#abandon]): its uses not yet read
/// are skipped, and a read of it no copy claimed is dropped by the slot's next claim. A failed read fails its quantum
/// and is still handed to its copy.
public final class DiskReader extends AbstractIngestSink implements AutoCloseable {

    private static final Logger LOG = LoggerFactory.getLogger(DiskReader.class);

    /// The longest [#close] waits for reads in flight.
    private static final long CLOSE_WAIT_NANOS = 30_000_000_000L;

    private static final int IDLE = 0;
    private static final int READING = 1;
    private static final int FILLED = 2;

    private interface Record {
        void apply();
    }

    private record Pending(long quantum, DiskUse use, Consumer<Throwable> onFailure) {}

    private final DiskWindow window;
    private final AsyncReads reads;
    private final long fileSize;
    /// A descriptor past the page cache (-1 when the file system has none) and one through it; -1 without reads.
    private final int directFd;
    private final int bufferedFd;
    /// The positional reads' channel when the machine has no asynchronous reads.
    private final FileChannel channel;
    private final MpscQueue<Record> records = new MpscQueue<>(1024, 4);
    private final AtomicInteger inFlight = new AtomicInteger();
    private final Delegate delegate = new Delegate();
    private volatile boolean closed;

    // Touched by the poll only.
    private final ArrayDeque<Pending>[] queues;
    private final int[] state = new int[DiskWindow.SLOTS];
    /// Each slot's hand-over between its reads and the device copies that consume them: null, a [Ready] read no copy
    /// claimed yet, or a [Waiter] copy whose read has not arrived. Changed by compare-and-set from any thread.
    private final java.util.concurrent.atomic.AtomicReferenceArray<Object> gates =
            new java.util.concurrent.atomic.AtomicReferenceArray<>(DiskWindow.SLOTS);

    /// A slot's read of quantum `ticket` arrived and no copy claimed it.
    private record Ready(long ticket) {}

    /// A copy of quantum `ticket` waits for its read of the slot; `release` queues it.
    private record Waiter(long ticket, Runnable release) {}

    /// Each slot's routing seed for its reads' completion frames.
    private final long[] seeds = new long[DiskWindow.SLOTS];
    /// Set by the worker that reaps a slot's read, before it hands the slot over; the poll takes it as the end of
    /// the read. The completion frame plays no part in the slot's turn, so a frame stranded behind a busy worker
    /// never holds the slot.
    private final java.util.concurrent.atomic.AtomicIntegerArray reaped =
            new java.util.concurrent.atomic.AtomicIntegerArray(DiskWindow.SLOTS);
    /// Uses of each registered quantum not yet given a slot.
    private final Map<Long, int[]> remaining = new HashMap<>();
    /// Quanta abandoned while uses of theirs may still be registered or queued.
    private final Set<Long> abandoned = new HashSet<>();
    private long latestQuantum = Long.MIN_VALUE;

    /// A reader of `artifact` into `window`; `reads` null reads positionally on the poll.
    @SuppressWarnings("unchecked")
    public DiskReader(DiskWindow window, AsyncReads reads, Path artifact) throws IOException {
        this.window = Objects.requireNonNull(window, "window");
        this.reads = reads;
        this.fileSize = java.nio.file.Files.size(artifact);
        if (reads != null) {
            int direct;
            try {
                direct = reads.openFile(artifact, true);
            } catch (IOException unsupported) {
                // The file system has no direct reads (tmpfs): every read goes through the page cache.
                direct = -1;
            }
            this.directFd = direct;
            try {
                this.bufferedFd = reads.openFile(artifact, false);
            } catch (IOException failure) {
                if (direct >= 0) reads.closeFile(direct);
                throw failure;
            }
            this.channel = null;
        } else {
            this.directFd = -1;
            this.bufferedFd = -1;
            this.channel = FileChannel.open(artifact, StandardOpenOption.READ);
        }
        this.queues = new ArrayDeque[DiskWindow.SLOTS];
        long seed = HasherApi.mix(0xd15c_0000L);
        for (int slot = 0; slot < DiskWindow.SLOTS; slot++) {
            this.queues[slot] = new ArrayDeque<>();
            this.seeds[slot] = seed + 64L * slot;
        }
    }

    // ---------------------------------------------------------------- any thread

    /// Registers `quantum`'s disk uses, in the order its device work consumes them. Called at admission, in
    /// admission order, with quantum numbers that grow with it; `onFailure` fails the quantum when a read fails. It
    /// runs on the worker that reaps the read, before the slot's flag rises, and must not block.
    public void enqueue(long quantum, List<DiskUse> uses, Consumer<Throwable> onFailure) {
        Objects.requireNonNull(onFailure, "onFailure");
        if (uses.isEmpty()) return;
        List<DiskUse> registered = List.copyOf(uses);
        post(() -> {
            this.latestQuantum = Math.max(this.latestQuantum, quantum);
            this.remaining.put(quantum, new int[] {registered.size()});
            for (DiskUse use : registered) this.queues[use.slot()].add(new Pending(quantum, use, onFailure));
        });
    }

    /// As above, for a quantum whose read failures are only logged (tests).
    public void enqueue(long quantum, List<DiskUse> uses) {
        enqueue(quantum, uses, failure -> {});
    }

    /// `quantum` stopped before its device work consumed all its uses: those not yet read are skipped, and a read of
    /// it that no copy claims is dropped when the next copy of the slot claims. Any time, before or after its
    /// registration.
    public void abandon(long quantum) {
        post(() -> {
            if (quantum > this.latestQuantum || this.remaining.containsKey(quantum)) this.abandoned.add(quantum);
        });
    }

    private void post(Record record) {
        if (this.closed) throw new IllegalStateException("the disk reader is closed");
        if (!this.records.offer(record)) throw new IllegalStateException("the disk reader refused a record");
    }

    /// Reads submitted and not yet complete.
    public int readsInFlight() {
        return this.inFlight.get();
    }

    // ---------------------------------------------------------------- the poll

    /// One poll: what the lattice's worker runs; tests call it as that worker.
    public void poll() {
        for (Record record = this.records.poll(); record != null; record = this.records.poll()) record.apply();
        if (this.closed) return;
        for (int slot = 0; slot < DiskWindow.SLOTS; slot++) {
            if (this.state[slot] == READING && this.reaped.getAndSet(slot, 0) == 1) this.state[slot] = FILLED;
            if (this.state[slot] == FILLED && this.window.flag(slot) == 0) this.state[slot] = IDLE;
            if (this.state[slot] == IDLE && !this.queues[slot].isEmpty()) dispatch(slot);
        }
    }

    /// Gives free slot `slot` the next use of its queue.
    private void dispatch(int slot) {
        Pending next = this.queues[slot].peek();
        DiskUse use = next.use();
        if (this.abandoned.contains(next.quantum())) {
            // No copy will consume it: the slot stays free for the next use.
            this.queues[slot].poll();
            given(next.quantum());
            return;
        }
        if (this.reads == null) {
            this.queues[slot].poll();
            given(next.quantum());
            readHere(slot, next, null);
            return;
        }
        // Whole blocks past the page cache, unless they would run past the end of the file.
        boolean direct = this.directFd >= 0 && use.alignedOffset() + use.readBytes() <= this.fileSize;
        int fd = direct ? this.directFd : this.bufferedFd;
        long length = direct ? use.readBytes() : use.lead() + use.byteSize();
        // A read of its own: the frame of the slot's last read may still wait in a queue.
        SlotRead read = new SlotRead(slot, this.seeds[slot], next);
        this.state[slot] = READING;
        this.inFlight.incrementAndGet();
        boolean submitted;
        Throwable refusal = null;
        try {
            submitted = this.reads.submit(fd, this.window.slotHost(slot), length, use.alignedOffset(), read);
        } catch (Throwable failure) {
            submitted = false;
            refusal = failure;
        }
        if (submitted) {
            this.queues[slot].poll();
            given(next.quantum());
            return;
        }
        // This thread cannot submit (no ring of its own, or the submission failed): the use is read here, or fails
        // its quantum; either way its copy is handed the slot.
        this.inFlight.decrementAndGet();
        this.queues[slot].poll();
        given(next.quantum());
        readHere(slot, next, refusal);
    }

    /// Reads `next` into slot `slot` on this thread, or fails its quantum with `refusal` (the submission's failure)
    /// when there is one. The slot is handed to its copy whatever the read or the quantum's failure throws: a copy
    /// left waiting would hold its graph for good.
    private void readHere(int slot, Pending next, Throwable refusal) {
        try {
            if (refusal != null) throw refusal;
            readPositionally(slot, next.use());
        } catch (Throwable failure) {
            failed(next, failure);
        } finally {
            this.state[slot] = FILLED;
            arrived(slot, next);
        }
    }

    // ---------------------------------------------------------------- the hand-over

    /// The copy of `ticket`'s next use of `slot` may be queued on the device once that use's read is in the slot:
    /// true when it is in now; otherwise `release` runs, on the thread that completes the read, once it is. Called
    /// once per use, when the copy's other inputs exist: for one slot, in the order the device consumes its uses
    /// (the staging ring's edges order a staging slot's transfers, across quanta too). Any thread.
    ///
    /// A read of another quantum still in the slot belongs to an abandoned quantum (its own copies came first): it
    /// is dropped, and the slot freed for this one's read.
    public boolean claim(int slot, long ticket, Runnable release) {
        Objects.requireNonNull(release, "release");
        Waiter waiter = new Waiter(ticket, release);
        while (true) {
            Object gate = this.gates.get(slot);
            if (gate == null) {
                if (this.gates.compareAndSet(slot, null, waiter)) return false;
            } else if (gate instanceof Ready ready) {
                if (ready.ticket() == ticket) {
                    if (this.gates.compareAndSet(slot, gate, null)) return true;
                } else if (this.gates.compareAndSet(slot, gate, waiter)) {
                    // No device work reads the dropped bytes: the poll may refill the slot.
                    this.window.lower(slot);
                    return false;
                }
            } else {
                throw new IllegalStateException("window slot " + slot + " has two copies waiting");
            }
        }
    }

    /// `read`'s use is in `slot`: the slot rises and its copy is released, or the read waits for its claim. A read
    /// of a quantum other than the waiting copy's belongs to an abandoned quantum and is dropped. Any thread; never
    /// blocks.
    private void arrived(int slot, Pending read) {
        long ticket = read.quantum();
        // Raised before a copy can be queued: the device lowers the flag after its copy, and the poll refills the
        // slot only once it is down.
        this.window.raise(slot);
        while (true) {
            Object gate = this.gates.get(slot);
            if (gate == null) {
                if (this.gates.compareAndSet(slot, null, new Ready(ticket))) return;
            } else if (gate instanceof Waiter waiter) {
                if (waiter.ticket() != ticket) {
                    this.window.lower(slot);
                    return;
                }
                if (this.gates.compareAndSet(slot, gate, null)) {
                    waiter.release().run();
                    return;
                }
            } else {
                // A slot holds one read until its copy lowers the flag, so this is a broken invariant: the read may
                // have overwritten bytes another copy still needs. Its quantum fails rather than run on them.
                LOG.error("window slot {} received a read while holding another", slot);
                failQuantum(
                        read,
                        new IllegalStateException("window slot " + slot + " received a read while holding another"));
                return;
            }
        }
    }

    /// One more use of `quantum` has its slot.
    private void given(long quantum) {
        int[] left = this.remaining.get(quantum);
        if (left == null || --left[0] > 0) return;
        this.remaining.remove(quantum);
        this.abandoned.remove(quantum);
    }

    private void readPositionally(int slot, DiskUse use) throws IOException {
        long length = use.lead() + use.byteSize();
        if (this.channel == null) {
            // io_uring is there but this thread cannot submit to it: a blocking read of the buffered descriptor.
            AsyncReads.readBlocking(this.bufferedFd, this.window.slotHost(slot), length, use.alignedOffset());
            return;
        }
        ByteBuffer target = this.window.slotSegment(slot).asSlice(0, length).asByteBuffer();
        long position = use.alignedOffset();
        while (target.hasRemaining()) {
            int read = this.channel.read(target, position);
            if (read < 0) throw new IOException("the artifact ends before byte " + (use.alignedOffset() + length));
            position += read;
        }
    }

    private static void failed(Pending pending, Throwable failure) {
        LOG.error("reading {} from the artifact failed", pending.use().name(), failure);
        failQuantum(pending, failure);
    }

    /// Fails the quantum of `pending`'s use. Runs where the read ended, a worker's poll of the reads included.
    private static void failQuantum(Pending pending, Throwable failure) {
        try {
            pending.onFailure().accept(failure);
        } catch (RuntimeException callbackFailure) {
            LOG.error("failing the quantum of a failed read failed", callbackFailure);
        }
    }

    /// Slot `slot`'s read of one use. The worker that reaps it ([#reaped]) fails the use's quantum when
    /// the read failed, hands the slot to the use's copy and marks the read ended for the poll; the
    /// completion frame only logs a failure, so neither the copy nor the poll waits for the frame.
    private final class SlotRead extends AsyncReads.Read {
        private final int slot;
        private final Pending pending;

        SlotRead(int slot, long seed, Pending pending) {
            super(seed);
            this.slot = slot;
            this.pending = pending;
        }

        /// A failed read fails its quantum before its copy is released, so the device's copy of the
        /// slot's stale bytes belongs to a quantum that cannot commit. Failing a quantum is a
        /// compare-and-set: `onFailure` must not block. The slot is handed over whatever `onFailure`
        /// throws: a copy left waiting would hold its graph for good. The read is marked ended only
        /// after the flag rose, so the poll never refills the slot before its copy ran.
        @Override
        protected void reaped() {
            try {
                IOException failure = failure();
                if (failure != null) failQuantum(this.pending, failure);
            } finally {
                try {
                    DiskReader.this.arrived(this.slot, this.pending);
                } finally {
                    DiskReader.this.reaped.set(this.slot, 1);
                    DiskReader.this.inFlight.decrementAndGet();
                }
            }
        }

        @Override
        public void execute() {
            IOException failure = failure();
            if (failure != null)
                LOG.error(
                        "reading {} from the artifact failed",
                        this.pending.use().name(),
                        failure);
        }

        /// The lattice rejected the completion without running it: the read ended all the same.
        @Override
        public void doFinallyWithError(Throwable rejection) {
            execute();
        }
    }

    // ---------------------------------------------------------------- lifecycle

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

    /// Stops reading, waits for the reads in flight (reaped while the lattice polls the reads) and closes
    /// the file. Called once every quantum retired, before the window is freed; the lake that polls it
    /// completes it. No device work waits for the window: a copy is queued only once its read is in.
    @Override
    public void close() {
        if (this.closed) return;
        this.closed = true;
        long deadline = System.nanoTime() + CLOSE_WAIT_NANOS;
        while (this.inFlight.get() > 0 && System.nanoTime() < deadline) LockSupport.parkNanos(100_000L);
        if (this.inFlight.get() > 0)
            throw new IllegalStateException(this.inFlight.get() + " artifact reads are still in flight");
        if (this.channel != null) {
            try {
                this.channel.close();
            } catch (IOException failure) {
                LOG.warn("closing the artifact failed", failure);
            }
        } else {
            if (this.directFd >= 0) this.reads.closeFile(this.directFd);
            this.reads.closeFile(this.bufferedFd);
        }
    }

    private final class Delegate extends AbstractIngestSink.Delegate {
        @Override
        public long hookOnPull(
                Consumer<AbstractFrame> consumer, Function<AbstractFrame, Boolean> stopCondition, long demand) {
            poll();
            return 0;
        }

        @Override
        public void hookOnRequest(LatticeReceiver terminal, long demand) {
            poll();
        }
    }
}
