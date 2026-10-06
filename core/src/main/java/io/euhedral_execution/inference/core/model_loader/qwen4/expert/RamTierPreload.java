package io.euhedral_execution.inference.core.model_loader.qwen4.expert;

import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;

/// The startup fill of a resident [RamTier]: a few reader threads that exist only for the load, take
/// consecutive records of one bank up to about 32 MiB at a time, and read each straight into its slot. It
/// runs before the model serves anything and ends with the call, so no thread outlives the load.
final class RamTierPreload {

    /// Bytes one reader takes before it takes the next chunk.
    static final long CHUNK_BYTES = 32L << 20;

    private RamTierPreload() {}

    /// Reads every record of `tier` from `source` with `readers` threads, and returns the bytes read.
    static long run(RamTier tier, RecordSource source, int readers) throws IOException, InterruptedException {
        if (readers < 1) throw new IllegalArgumentException("readers must be positive");
        ExpertBank[] banks = tier.banks();
        int keyCount = tier.keys().keyCount();
        int[] chunkBank = new int[keyCount + 1];
        int[] chunkFirst = new int[keyCount + 1];
        int[] chunkEnd = new int[keyCount + 1];
        int chunks = 0;
        for (int bank = 0; bank < banks.length; bank++) {
            int first = 0;
            long bytes = 0;
            int count = banks[bank].expertCount();
            for (int expert = 0; expert < count; expert++) {
                bytes += banks[bank].recordBytes(expert);
                if (bytes < CHUNK_BYTES && expert != count - 1) continue;
                chunkBank[chunks] = bank;
                chunkFirst[chunks] = first;
                chunkEnd[chunks++] = expert + 1;
                first = expert + 1;
                bytes = 0;
            }
        }
        int total = chunks;
        AtomicInteger nextChunk = new AtomicInteger();
        AtomicLong loaded = new AtomicLong();
        AtomicReference<Throwable> failure = new AtomicReference<>();
        Thread[] threads = new Thread[Math.min(readers, total)];
        for (int reader = 0; reader < threads.length; reader++) {
            threads[reader] = Thread.ofPlatform()
                    .name("expert-preload-" + reader)
                    .unstarted(() -> {
                        try {
                            for (int chunk = nextChunk.getAndIncrement();
                                    chunk < total && failure.get() == null;
                                    chunk = nextChunk.getAndIncrement())
                                loaded.addAndGet(
                                        loadChunk(tier, source, chunkBank[chunk], chunkFirst[chunk], chunkEnd[chunk]));
                        } catch (IOException | InterruptedException | RuntimeException | Error thrown) {
                            failure.compareAndSet(null, thrown);
                        }
                    });
            threads[reader].start();
        }
        try {
            for (Thread thread : threads) thread.join();
        } catch (InterruptedException interrupted) {
            failure.compareAndSet(null, interrupted);
            for (Thread thread : threads) thread.interrupt();
            for (Thread thread : threads) joinUninterruptibly(thread);
        }
        Throwable thrown = failure.get();
        if (thrown instanceof IOException io) throw io;
        if (thrown instanceof InterruptedException interrupted) throw interrupted;
        if (thrown instanceof RuntimeException runtime) throw runtime;
        if (thrown instanceof Error error) throw error;
        return loaded.get();
    }

    private static void joinUninterruptibly(Thread thread) {
        boolean interrupted = false;
        while (thread.isAlive()) {
            try {
                thread.join();
            } catch (InterruptedException again) {
                interrupted = true;
            }
        }
        if (interrupted) Thread.currentThread().interrupt();
    }

    private static long loadChunk(RamTier tier, RecordSource source, int ordinal, int first, int end)
            throws IOException, InterruptedException {
        ExpertBank bank = tier.banks()[ordinal];
        long bytes = 0;
        for (int expert = first; expert < end; expert++) {
            long size = bank.recordBytes(expert);
            long address = tier.residentAddress(tier.keys().key(ordinal, expert));
            source.read(bank, expert, MemorySegment.ofAddress(address).reinterpret(size));
            bytes += size;
        }
        return bytes;
    }
}
