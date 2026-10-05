package io.euhedral_execution.inference.core.model_loader.qwen4.expert;

import io.euhedral_execution.inference.core.gpu.GpuMemory;
import java.io.IOException;
import java.io.InterruptedIOException;
import java.lang.foreign.MemorySegment;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.LongAdder;

/// Every record of every bank, loaded at construction into one pinned huge-page arena. [#open] returns an
/// address inside the arena: no copy and no staging, so closing a record does nothing.
///
/// One arena, not one allocation per record: the device's copy engines reach their full host-to-device rate
/// only from large huge-page-backed ranges, and a pinned allocation per record falls back to small pages.
/// The arena is read from the file by a bounded pool of threads issuing large positional reads, so loading
/// scales with the storage's parallelism rather than with one reader.
public final class ArenaExpertStore implements HostExpertStore {
    /// Records start on cache-line boundaries inside the arena.
    private static final long RECORD_ALIGNMENT = 64;
    /// Bytes one loading task reads before the pool picks up the next.
    private static final long LOAD_CHUNK_BYTES = 32L << 20;

    private final ExpertBank[] banks;
    private final ExpertKeys keys;
    private final long[] hostOffsets;
    private final HostArena arena;
    private final long bytesRead;
    private final LongAdder opens = new LongAdder();

    /// Loads every record of `banks` from `file` into one arena allocated with
    /// [GpuMemory#allocateHostWeights], using up to `readThreads` concurrent readers.
    ///
    /// @throws IllegalArgumentException when a record lies outside the file
    public ArenaExpertStore(GpuMemory memory, Path file, ExpertBank[] banks, int readThreads) throws IOException {
        Objects.requireNonNull(memory, "memory");
        if (readThreads < 1) throw new IllegalArgumentException("readThreads must be positive");
        this.banks = banks.clone();
        this.keys = new ExpertKeys(this.banks);
        this.hostOffsets = new long[this.keys.keyCount()];
        long total = 0;
        for (int bank = 0; bank < this.banks.length; bank++) {
            for (int expert = 0; expert < this.banks[bank].expertCount(); expert++) {
                this.hostOffsets[this.keys.key(bank, expert)] = total;
                total = Math.addExact(
                        total, ExpertFiles.alignUp(this.banks[bank].recordBytes(expert), RECORD_ALIGNMENT));
            }
        }
        try (FileChannel channel = FileChannel.open(file, StandardOpenOption.READ)) {
            ExpertFiles.validate(this.banks, channel.size());
            this.arena = new HostArena(memory, total);
            try {
                this.bytesRead = load(channel, readThreads);
            } catch (IOException | RuntimeException | Error failure) {
                this.arena.close();
                throw failure;
            }
        }
    }

    private long load(FileChannel channel, int readThreads) throws IOException {
        List<long[]> tasks = new ArrayList<>();
        for (int bank = 0; bank < this.banks.length; bank++) {
            int first = 0;
            long bytes = 0;
            for (int expert = 0; expert < this.banks[bank].expertCount(); expert++) {
                bytes += this.banks[bank].recordBytes(expert);
                if (bytes >= LOAD_CHUNK_BYTES || expert == this.banks[bank].expertCount() - 1) {
                    tasks.add(new long[] {bank, first, expert + 1});
                    first = expert + 1;
                    bytes = 0;
                }
            }
        }
        AtomicLong loaded = new AtomicLong();
        ExecutorService pool = Executors.newFixedThreadPool(Math.min(readThreads, tasks.size()), runnable -> {
            Thread thread = new Thread(runnable, "expert-arena-load");
            thread.setDaemon(true);
            return thread;
        });
        List<Future<?>> futures = new ArrayList<>(tasks.size());
        try {
            for (long[] task : tasks)
                futures.add(pool.submit(() -> {
                    ExpertBank bank = this.banks[(int) task[0]];
                    for (int expert = (int) task[1]; expert < task[2]; expert++) {
                        long size = bank.recordBytes(expert);
                        MemorySegment destination = MemorySegment.ofAddress(
                                        this.arena.address() + this.hostOffsets[this.keys.key((int) task[0], expert)])
                                .reinterpret(size);
                        ExpertFiles.readFully(channel, destination, bank.fileOffset(expert));
                        loaded.addAndGet(size);
                    }
                    return null;
                }));
            for (Future<?> future : futures) future.get();
            return loaded.get();
        } catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
            throw new InterruptedIOException("interrupted while loading the expert arena");
        } catch (ExecutionException failure) {
            Throwable cause = failure.getCause();
            if (cause instanceof IOException io) throw new IOException(io.getMessage(), io);
            if (cause instanceof RuntimeException runtime) throw runtime;
            if (cause instanceof Error error) throw error;
            throw new IOException(cause);
        } finally {
            pool.shutdownNow();
            // No reader may still write into the arena once the constructor fails and frees it.
            boolean interrupted = false;
            while (!pool.isTerminated()) {
                try {
                    pool.awaitTermination(1, TimeUnit.MINUTES);
                } catch (InterruptedException interrupt) {
                    interrupted = true;
                }
            }
            if (interrupted) Thread.currentThread().interrupt();
        }
    }

    @Override
    public ExpertBank[] banks() {
        return this.banks.clone();
    }

    @Override
    public HostRecord open(int bank, int expert, long timeoutNanos) {
        int key = this.keys.key(bank, expert);
        this.arena.borrow();
        this.opens.increment();
        return new ArenaRecord(
                this.arena, this.arena.address() + this.hostOffsets[key], this.banks[bank].recordBytes(expert));
    }

    /// Bytes read from the file: the sum of every record, once, at construction.
    @Override
    public long bytesRead() {
        return this.bytesRead;
    }

    @Override
    public long recordOpens() {
        return this.opens.sum();
    }

    /// Bytes of pinned memory the arena holds.
    public long arenaBytes() {
        return this.arena.byteSize();
    }

    /// Frees the arena once no record is open.
    @Override
    public void close() {
        this.arena.close();
    }

    private static final class ArenaRecord implements HostRecord {
        private final HostArena arena;
        private final long address;
        private final long byteSize;
        private final AtomicBoolean closed = new AtomicBoolean();

        private ArenaRecord(HostArena arena, long address, long byteSize) {
            this.arena = arena;
            this.address = address;
            this.byteSize = byteSize;
        }

        @Override
        public long hostAddress() {
            return this.address;
        }

        @Override
        public long byteSize() {
            return this.byteSize;
        }

        @Override
        public void close() {
            if (this.closed.compareAndSet(false, true)) this.arena.giveBack();
        }
    }
}
