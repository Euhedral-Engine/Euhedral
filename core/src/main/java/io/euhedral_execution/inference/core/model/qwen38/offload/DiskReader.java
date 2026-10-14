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
/// worker at a time, whose bookkeeping is plain fields changed only by its poll.
///
/// Each quantum registers its disk uses at admission ([#enqueue]), in the order the device consumes them. Every
/// window slot keeps its own queue of uses, in admission order and, within a quantum, in use order: the order its
/// copies run on the device, which the staging ring's edges fix. A poll gives each slot whose flag the device lowered
/// the next use of its queue: it submits the read ([AsyncReads], io_uring) and the read's completion frame raises
/// the flag. Without asynchronous reads the poll reads positionally itself, one slot at a time.
///
/// A quantum that stops before its device work consumed every use is abandoned ([#abandon]): its uses not yet read
/// raise their flags unread, so the device, which drains them in order, never waits on them and the next quantum's
/// uses of each slot follow. A failed read fails its quantum and still raises the flag.
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
    private final SlotRead[] completions = new SlotRead[DiskWindow.SLOTS];
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
            this.completions[slot] = new SlotRead(slot, seed + 64L * slot);
        }
    }

    // ---------------------------------------------------------------- any thread

    /// Registers `quantum`'s disk uses, in the order its device work consumes them. Called at admission, in
    /// admission order, with quantum numbers that grow with it; `onFailure` fails the quantum when a read fails.
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

    /// `quantum` stopped before its device work consumed all its uses: those not yet read are raised unread. Any
    /// time, before or after its registration.
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
            if (this.state[slot] == FILLED && this.window.flag(slot) == 0) this.state[slot] = IDLE;
            if (this.state[slot] == IDLE && !this.queues[slot].isEmpty()) dispatch(slot);
        }
    }

    /// Gives free slot `slot` the next use of its queue.
    private void dispatch(int slot) {
        Pending next = this.queues[slot].peek();
        DiskUse use = next.use();
        if (this.abandoned.contains(next.quantum())) {
            this.queues[slot].poll();
            given(next.quantum());
            this.state[slot] = FILLED;
            this.window.raise(slot);
            return;
        }
        if (this.reads == null) {
            this.queues[slot].poll();
            given(next.quantum());
            try {
                readPositionally(slot, use);
            } catch (IOException | RuntimeException failure) {
                failed(next, failure);
            }
            this.state[slot] = FILLED;
            this.window.raise(slot);
            return;
        }
        // Whole blocks past the page cache, unless they would run past the end of the file.
        boolean direct = this.directFd >= 0 && use.alignedOffset() + use.readBytes() <= this.fileSize;
        int fd = direct ? this.directFd : this.bufferedFd;
        long length = direct ? use.readBytes() : use.lead() + use.byteSize();
        SlotRead read = this.completions[slot];
        read.pending = next;
        this.state[slot] = READING;
        this.inFlight.incrementAndGet();
        boolean submitted;
        Throwable refusal = null;
        try {
            submitted = this.reads.submit(fd, this.window.slotHost(slot), length, use.alignedOffset(), read);
        } catch (RuntimeException failure) {
            submitted = false;
            refusal = failure;
        }
        if (submitted) {
            this.queues[slot].poll();
            given(next.quantum());
            return;
        }
        // This thread cannot submit (no ring of its own, or the submission failed): the use is read here, or
        // fails its quantum; either way the flag rises, so the device never waits for a read that will not come.
        this.inFlight.decrementAndGet();
        read.pending = null;
        this.queues[slot].poll();
        given(next.quantum());
        try {
            if (refusal != null) throw refusal;
            readPositionally(slot, use);
        } catch (Throwable failure) {
            failed(next, failure);
        }
        this.state[slot] = FILLED;
        this.window.raise(slot);
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
        try {
            pending.onFailure().accept(failure);
        } catch (RuntimeException callbackFailure) {
            LOG.error("failing the quantum of a failed read failed", callbackFailure);
        }
    }

    /// The completion of slot `slot`'s read: fails the quantum when the read failed, raises the flag either way.
    private final class SlotRead extends AsyncReads.Read {
        private final int slot;
        private Pending pending;

        SlotRead(int slot, long seed) {
            super(seed);
            this.slot = slot;
        }

        @Override
        public void execute() {
            Pending done = this.pending;
            // A completion the lattice runs twice (rejected after running) ends its read once.
            if (done == null) return;
            this.pending = null;
            IOException failure = failure();
            if (failure != null) failed(done, failure);
            DiskReader.this.window.raise(this.slot);
            int slot = this.slot;
            try {
                if (!DiskReader.this.closed) post(() -> DiskReader.this.state[slot] = FILLED);
            } catch (RuntimeException closing) {
                // The reader closed between the check and the post: no later use needs the slot.
            } finally {
                DiskReader.this.inFlight.decrementAndGet();
            }
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

    /// Stops reading, waits for the reads in flight (their completions run while the lattice polls the reads) and
    /// closes the file. Called once every quantum retired, before the window is freed; the lake that polls it
    /// completes it.
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
