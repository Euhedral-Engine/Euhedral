package io.euhedral_execution.inference.core.model_loader.qwen4.expert;

import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;

/// The startup fill of a [RamTier]: a few reader threads that exist only for the load, take runs of consecutive
/// records of one bank that sit in consecutive slots (up to about [#CHUNK_BYTES]), and read each run straight into
/// its slots, in one read where the slots line up with the file. It runs before the model serves anything and ends
/// with the call, so no thread outlives the load.
final class RamTierPreload {

    /// Bytes one reader takes before it takes the next run: large enough that the disk reads at its sequential rate,
    /// with a few runs in flight.
    static final long CHUNK_BYTES = 8L << 20;

    private RamTierPreload() {}

    /// Reads every record that has a slot in `tier` from `source` with `readers` threads, and returns the bytes read.
    static long run(RamTier tier, RecordSource source, int readers) throws IOException, InterruptedException {
        if (readers < 1) throw new IllegalArgumentException("readers must be positive");
        ExpertBank[] banks = tier.banks();
        int keyCount = tier.keys().keyCount();
        int[] chunkBank = new int[keyCount + 1];
        int[] chunkFirst = new int[keyCount + 1];
        int[] chunkEnd = new int[keyCount + 1];
        int chunks = 0;
        for (int bank = 0; bank < banks.length; bank++) {
            int count = banks[bank].expertCount();
            int first = -1;
            long bytes = 0;
            for (int expert = 0; expert <= count; expert++) {
                int slot = expert < count ? tier.slotOf(tier.keys().key(bank, expert)) : -1;
                boolean continues = first >= 0
                        && slot >= 0
                        && slot == tier.slotOf(tier.keys().key(bank, expert - 1)) + 1
                        && bytes < CHUNK_BYTES;
                if (first >= 0 && !continues) {
                    chunkBank[chunks] = bank;
                    chunkFirst[chunks] = first;
                    chunkEnd[chunks++] = expert;
                    first = -1;
                    bytes = 0;
                }
                if (slot < 0) continue;
                if (first < 0) first = expert;
                bytes += banks[bank].recordBytes(expert);
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

    /// Reads records `first` to `end - 1` of a bank into their consecutive slots: in one read when the slots are
    /// exactly as long as the records, so the run lies in memory as it lies in the file; otherwise one by one.
    private static long loadChunk(RamTier tier, RecordSource source, int ordinal, int first, int end)
            throws IOException, InterruptedException {
        ExpertBank bank = tier.banks()[ordinal];
        long address = tier.address(tier.slotOf(tier.keys().key(ordinal, first)));
        long bytes = 0;
        boolean packed = true;
        for (int expert = first; expert < end; expert++) {
            packed &= bank.recordBytes(expert) == tier.slotBytes();
            bytes += bank.recordBytes(expert);
        }
        if (packed) {
            source.readRun(
                    bank, first, end - first, MemorySegment.ofAddress(address).reinterpret(bytes));
            return bytes;
        }
        for (int expert = first; expert < end; expert++) {
            long size = bank.recordBytes(expert);
            long slot = tier.address(tier.slotOf(tier.keys().key(ordinal, expert)));
            source.read(bank, expert, MemorySegment.ofAddress(slot).reinterpret(size));
        }
        return bytes;
    }
}
