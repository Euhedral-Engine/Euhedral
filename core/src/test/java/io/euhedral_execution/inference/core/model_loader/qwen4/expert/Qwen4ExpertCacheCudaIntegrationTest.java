package io.euhedral_execution.inference.core.model_loader.qwen4.expert;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import io.euhedral_execution.inference.core.gpu.GpuStream;
import io.euhedral_execution.inference.core.model_loader.layer_weights.TensorDataType;
import io.euhedral_execution.inference.core.model_loader.layer_weights.WeightFormat;
import io.euhedral_execution.inference.core.model_loader.layer_weights.WeightLayout;
import io.euhedral_execution.inference.core.model_loader.qwen4.ComponentGroup;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.List;
import java.util.Random;
import java.util.SplittableRandom;
import java.util.zip.CRC32;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/// The expert cache on a real GPU: records of about 2.7 MB from a synthetic file of about 1 GB are
/// copied into a small slab by [GpuExpertTransfer] on a real stream, read back, and compared with
/// the file. Run with `./gradlew :core:cudaIntegrationTest --tests
/// '*Qwen4ExpertCacheCudaIntegrationTest'`.
class Qwen4ExpertCacheCudaIntegrationTest {
    private static final int BANKS = 6;
    private static final int EXPERTS = 64;
    private static final int BASE_RECORD_BYTES = 2_700_000;
    private static final int SLOTS = 8;

    private CudaGpuMemory gpu;
    private Path file;
    private ExpertBank[] banks;
    private long baseAllocated;
    private long baseHostWeights;

    @BeforeEach
    void setUp() throws IOException {
        String library = System.getProperty("euhedral.cuda.library");
        assumeTrue(library != null, "euhedral.cuda.library is not set");
        Path libraryPath = Path.of(library);
        assertTrue(Files.isRegularFile(libraryPath), "native CUDA library is missing: " + libraryPath);
        this.gpu = new CudaGpuMemory(libraryPath);
        this.baseAllocated = this.gpu.allocatedBytes();
        this.baseHostWeights = this.gpu.hostWeightBytes();
        this.file = Files.createTempFile(Path.of(System.getProperty("java.io.tmpdir")), "qwen4-experts", ".bin");
        this.banks = generate(this.file);
    }

    @AfterEach
    void tearDown() throws IOException {
        if (this.gpu != null) this.gpu.close();
        if (this.file != null) Files.deleteIfExists(this.file);
    }

    /// Writes `BANKS * EXPERTS` records of pseudo-random bytes, in a few slightly different sizes,
    /// and describes them.
    private static ExpertBank[] generate(Path file) throws IOException {
        SplittableRandom random = new SplittableRandom(20261005L);
        ExpertBank[] banks = new ExpertBank[BANKS];
        byte[] buffer = new byte[BASE_RECORD_BYTES + 3 * 16384];
        long offset = 4096;
        try (FileChannel channel = FileChannel.open(file, StandardOpenOption.WRITE, StandardOpenOption.CREATE)) {
            channel.write(ByteBuffer.wrap(new byte[4096]), 0);
            for (int bank = 0; bank < BANKS; bank++) {
                long[] offsets = new long[EXPERTS];
                long[] sizes = new long[EXPERTS];
                int[] crcs = new int[EXPERTS];
                for (int expert = 0; expert < EXPERTS; expert++) {
                    int size = BASE_RECORD_BYTES + (expert % 4) * 16384;
                    ByteBuffer bytes = ByteBuffer.wrap(buffer, 0, size);
                    for (int i = 0; i + Long.BYTES <= size; i += Long.BYTES) bytes.putLong(i, random.nextLong());
                    CRC32 crc = new CRC32();
                    crc.update(buffer, 0, size);
                    channel.write(ByteBuffer.wrap(buffer, 0, size), offset);
                    offsets[expert] = offset;
                    sizes[expert] = size;
                    crcs[expert] = (int) crc.getValue();
                    offset += (size + 63L) / 64 * 64;
                }
                banks[bank] = new ExpertBank(
                        "layer" + bank,
                        ComponentGroup.ROUTED_EXPERT,
                        bank,
                        List.of(projection("gate_up", 0, 1_800_000), projection("down", 1_800_000, 900_000)),
                        offsets,
                        sizes,
                        crcs);
            }
        }
        return banks;
    }

    private static ExpertProjection projection(String name, long offset, long size) {
        return new ExpertProjection(
                name,
                new long[] {size, 1},
                TensorDataType.UINT8,
                WeightFormat.NVFP4,
                WeightLayout.CONTIGUOUS_LE_V1,
                offset,
                size);
    }

    /// Reads the lease's slot back and compares it with the file's record and the bank's CRC-32.
    private void verify(ExpertLease lease, FileChannel channel, Arena scratch) throws IOException {
        long size = lease.byteSize();
        MemorySegment device = scratch.allocate(size, 64);
        this.gpu.copyDeviceToHost(device, lease.deviceAddress(), size);
        ByteBuffer expected = ByteBuffer.allocate((int) size);
        while (expected.hasRemaining())
            if (channel.read(expected, this.banks[lease.bank()].fileOffset(lease.expert()) + expected.position()) < 0)
                throw new IOException("short file");
        assertEquals(-1, device.mismatch(MemorySegment.ofArray(expected.array())), lease.toString());
        CRC32 crc = new CRC32();
        crc.update(device.asByteBuffer());
        assertEquals(this.banks[lease.bank()].crc32(lease.expert()), (int) crc.getValue(), "crc " + lease);
    }

    @Test
    void fileBackedCacheMovesEveryRecordIntactOverARealStream() throws Exception {
        long slotBytes = ExpertCache.slotBytesFor(this.banks);
        FileExpertStore store = new FileExpertStore(this.gpu, this.file, this.banks, 6);
        runCache("file", store, slotBytes);
    }

    @Test
    void arenaBackedCacheMovesEveryRecordIntactOverARealStream() throws Exception {
        long slotBytes = ExpertCache.slotBytesFor(this.banks);
        long start = System.nanoTime();
        ArenaExpertStore store = new ArenaExpertStore(this.gpu, this.file, this.banks, 16);
        double seconds = (System.nanoTime() - start) / 1e9;
        System.out.printf(
                "arena load: %.2f GB in %.2f s = %.2f GB/s (file, page cache)%n",
                store.arenaBytes() / 1e9, seconds, store.bytesRead() / 1e9 / seconds);
        runCache("arena", store, slotBytes);
    }

    private void runCache(String name, HostExpertStore store, long slotBytes) throws Exception {
        long deviceBefore = this.gpu.allocatedBytes();
        ExpertCache cache = new ExpertCache(store, new GpuExpertTransfer(this.gpu, 4), this.gpu, SLOTS, slotBytes, 1);
        assertEquals(deviceBefore + SLOTS * slotBytes, this.gpu.allocatedBytes(), "one slab, nothing else");
        try (FileChannel channel = FileChannel.open(this.file, StandardOpenOption.READ);
                Arena arena = Arena.ofConfined()) {
            // Sequential misses: every record of two banks, so each copy runs alone and is timed alone.
            for (int bank = 0; bank < 2; bank++) {
                for (int expert = 0; expert < EXPERTS; expert++) {
                    try (ExpertLease lease = ExpertTestSupport.acquire(cache, bank, expert)) {
                        verify(lease, channel, arena);
                    }
                }
            }
            // Random traffic with hits.
            Random random = new Random(0);
            for (int i = 0; i < 240; i++) {
                int bank = random.nextInt(BANKS);
                int expert = random.nextInt(random.nextBoolean() ? 6 : EXPERTS);
                try (ExpertLease lease = ExpertTestSupport.acquire(cache, bank, expert)) {
                    verify(lease, channel, arena);
                }
            }
            cache.checkQuiescent();

            fencedRelease(cache, channel, arena, slotBytes);
            cache.checkQuiescent();
        }
        ExpertCacheStats.Snapshot stats = cache.stats().snapshot();
        double gigabytesPerSecond = stats.transferBytesPerSecond() / 1e9;
        System.out.printf(
                "%s store: %d misses, %d hits, %d evictions, %d bytes in %.1f ms of transfers = %.2f GB/s host-to-device; "
                        + "slot waits %.1f ms, load waits %.1f ms, store read %.2f GB, peak slots %d%n",
                name,
                stats.misses(),
                stats.hits(),
                stats.evictions(),
                stats.transferBytes(),
                stats.transferNanos() / 1e6,
                gigabytesPerSecond,
                stats.slotWaitNanos() / 1e6,
                stats.loadWaitNanos() / 1e6,
                store.bytesRead() / 1e9,
                stats.peakSlotsInUse());
        assertTrue(stats.misses() >= 2 * EXPERTS);
        assertEquals(0, stats.failedTransfers());
        assertTrue(gigabytesPerSecond > 1.0, "host-to-device rate " + gigabytesPerSecond + " GB/s");

        cache.close();
        assertEquals(deviceBefore, this.gpu.allocatedBytes(), "the slab was freed");
        assertEquals(this.baseAllocated, this.gpu.allocatedBytes());
        assertEquals(this.baseHostWeights, this.gpu.hostWeightBytes(), "the host arena was freed");
    }

    /// A kernel-like read of a slot on a compute stream, closed with a stream marker: the refill of
    /// that slot is ordered behind it on the device, so the compute stream's copy still sees the
    /// old record.
    private void fencedRelease(ExpertCache cache, FileChannel channel, Arena arena, long slotBytes) throws Exception {
        long scratch = this.gpu.allocate(slotBytes);
        GpuStream compute = this.gpu.openStream();
        try {
            // Replace every resident, so 4/63 is loaded into a slot of its own and is evicted by the eighth
            // miss after it.
            for (int expert = 0; expert < SLOTS; expert++)
                ExpertTestSupport.acquire(cache, 3, 56 + expert).close();
            ExpertLease reader = ExpertTestSupport.acquire(cache, 4, 63);
            long size = reader.byteSize();
            long marker = compute.openMarker();
            compute.submit(() -> this.gpu.copyDeviceToDevice(scratch, reader.deviceAddress(), size), false);
            compute.mark(marker);
            reader.close(StreamFence.owning(compute, marker));
            for (int expert = 0; expert < SLOTS; expert++)
                ExpertTestSupport.acquire(cache, 3, 40 + expert).close();
            assertTrue(!cache.isResident(4, 63), "its slot was refilled behind the fence");
            compute.synchronize();
            MemorySegment copy = arena.allocate(size, 64);
            this.gpu.copyDeviceToHost(copy, scratch, size);
            ByteBuffer expected = ByteBuffer.allocate((int) size);
            while (expected.hasRemaining()) channel.read(expected, this.banks[4].fileOffset(63) + expected.position());
            assertEquals(
                    -1, copy.mismatch(MemorySegment.ofArray(expected.array())), "the fenced read saw the old record");
        } finally {
            compute.synchronize();
            compute.close();
            this.gpu.free(scratch);
        }
    }
}
