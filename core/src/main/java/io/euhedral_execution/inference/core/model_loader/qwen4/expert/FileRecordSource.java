package io.euhedral_execution.inference.core.model_loader.qwen4.expert;

import io.euhedral_execution.inference.core.scheduling.graph.AsyncReads;
import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.nio.channels.ClosedByInterruptException;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.LongAdder;

/// Reads records straight from the artifact file with positional reads, or asynchronously through [AsyncReads]
/// when it is given them. Every record of every bank is checked against the file's size when the source opens.
///
/// Each read opens its own channel. A [FileChannel] is closed for everyone when a thread blocked in it is
/// interrupted, and records are read on the threads that ask for them, which callers may interrupt: a channel
/// shared between reads would be lost to the first interrupted acquirer. An interrupted read fails with
/// [InterruptedException] and affects nothing else; opening a channel is negligible beside a record's read.
public final class FileRecordSource implements RecordSource {
    private final Path file;
    private final LongAdder bytesRead = new LongAdder();
    private final LongAdder reads = new LongAdder();
    private final LongAdder readNanos = new LongAdder();
    private final AtomicInteger readsNow = new AtomicInteger();
    private final AtomicInteger readsHighWater = new AtomicInteger();
    private final AtomicBoolean closed = new AtomicBoolean();
    private final AsyncReads async;
    private final int fd;

    public FileRecordSource(Path file, ExpertBank[] banks) throws IOException {
        this(file, banks, null);
    }

    /// A source that reads asynchronously through `async` (null: positional reads only).
    public FileRecordSource(Path file, ExpertBank[] banks, AsyncReads async) throws IOException {
        this.file = file;
        try (FileChannel channel = FileChannel.open(file, StandardOpenOption.READ)) {
            ExpertFiles.validate(banks, channel.size());
        }
        this.async = async;
        this.fd = async == null ? -1 : async.openFile(file);
    }

    @Override
    public boolean readRangeAsync(
            ExpertBank bank, int expert, long from, MemorySegment destination, AsyncReads.Read done) {
        if (this.async == null || this.closed.get()) return false;
        long size = destination.byteSize();
        if (from < 0 || from > bank.recordBytes(expert) - size)
            throw new IllegalArgumentException(
                    "bytes " + from + ".." + (from + size) + " are outside record " + bank.recordBytes(expert));
        if (!this.async.submit(this.fd, destination.address(), size, bank.fileOffset(expert) + from, done))
            return false;
        if (from == 0) this.reads.increment();
        this.bytesRead.add(size);
        return true;
    }

    /// Adds the time of a read made with [#readRangeAsync], once it ended.
    public void asyncReadEnded(long nanos) {
        this.readNanos.add(nanos);
    }

    @Override
    public void read(ExpertBank bank, int expert, MemorySegment destination) throws IOException, InterruptedException {
        long size = bank.recordBytes(expert);
        if (destination.byteSize() != size)
            throw new IllegalArgumentException(
                    "destination holds " + destination.byteSize() + " bytes, record " + size);
        readRange(bank, expert, 0, destination);
    }

    @Override
    public boolean ranged() {
        return true;
    }

    @Override
    public void readRange(ExpertBank bank, int expert, long from, MemorySegment destination)
            throws IOException, InterruptedException {
        long size = destination.byteSize();
        if (from < 0 || from > bank.recordBytes(expert) - size)
            throw new IllegalArgumentException(
                    "bytes " + from + ".." + (from + size) + " are outside record " + bank.recordBytes(expert));
        if (this.closed.get()) throw new IllegalStateException("the record source is closed");
        int now = this.readsNow.incrementAndGet();
        this.readsHighWater.accumulateAndGet(now, Math::max);
        long begin = System.nanoTime();
        try (FileChannel channel = FileChannel.open(this.file, StandardOpenOption.READ)) {
            ExpertFiles.readFully(channel, destination, bank.fileOffset(expert) + from);
        } catch (ClosedByInterruptException interrupted) {
            // The interrupt closed this read's channel and left the flag set; the exception consumes it.
            Thread.interrupted();
            throw new InterruptedException("interrupted while reading expert " + expert + " of " + bank.name());
        } finally {
            this.readsNow.decrementAndGet();
            this.readNanos.add(System.nanoTime() - begin);
        }
        if (from == 0) this.reads.increment();
        this.bytesRead.add(size);
    }

    @Override
    public long bytesRead() {
        return this.bytesRead.sum();
    }

    /// Record reads completed.
    public long recordReads() {
        return this.reads.sum();
    }

    /// Time spent in reads, summed over reads (parallel reads add up).
    public long readNanos() {
        return this.readNanos.sum();
    }

    /// The most reads that ran at once.
    public int concurrentReadsHighWater() {
        return this.readsHighWater.get();
    }

    @Override
    public void close() {
        if (this.closed.getAndSet(true)) return;
        if (this.fd >= 0) this.async.closeFile(this.fd);
    }
}
