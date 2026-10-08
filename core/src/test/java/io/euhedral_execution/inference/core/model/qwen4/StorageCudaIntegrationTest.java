package io.euhedral_execution.inference.core.model.qwen4;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.inference.core.InferenceConfig;
import io.euhedral_execution.inference.core.artifact.TensorHandle;
import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import io.euhedral_execution.inference.core.model.qwen4.expert.ExpertBank;
import io.euhedral_execution.inference.core.model.qwen4.expert.ExpertCache;
import io.euhedral_execution.inference.core.model.qwen4.expert.ExpertLease;
import io.euhedral_execution.inference.core.model.qwen4.loader.ComponentGroup;
import io.euhedral_execution.inference.core.model.qwen4.loader.ResidencyPlan;
import io.euhedral_execution.inference.core.model.qwen4.loader.StorageClass;
import io.euhedral_execution.inference.core.model.qwen4.loader.Tensor;
import io.euhedral_execution.inference.core.testing.ModelGroup;
import io.euhedral_execution.inference.core.testing.SharedQwen38;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.BitSet;
import java.util.SplittableRandom;
import java.util.concurrent.TimeUnit;
import java.util.zip.CRC32;
import org.junit.jupiter.api.MethodOrderer;
import org.junit.jupiter.api.Order;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestMethodOrder;
import org.junit.jupiter.api.Timeout;

/// The real Flash-Next artifact through the engine's storage load on the real GPU, for several
/// maximum contexts: every component has a location, device memory stays within the plan, experts
/// move through the cache with the artifact's own bytes, rows come out of the n-gram tables, and
/// closing returns every allocation. No model runs.
@ModelGroup.FlashNext
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
@Timeout(value = 10, unit = TimeUnit.MINUTES)
class StorageCudaIntegrationTest {

    /// The two ends of the range. The plan's prefill chunk is 4,096 tokens up to a context of 32,768 and
    /// 2,048 from 131,072 on (the workspace shrinks to leave room for the KV), so one end sits on each side
    /// of that change; the contexts between only move the expert cache by a few percent.
    static final int[] CONTEXTS = {4096, 262144};

    static Path artifactPath() {
        return Path.of(System.getProperty(
                "euhedral.qwen4.artifact", "/home/brandon/models/qwen3_8_flash_next/qwen3_8_flash_next_nvfp4.edrl"));
    }

    @Test
    @Order(1)
    void loadsTheRealArtifactAtSeveralContexts() throws Exception {
        Path artifact = artifactPath();
        assumeTrue(Files.isRegularFile(artifact), "no Flash-Next artifact at " + artifact);
        Path library = Path.of(System.getProperty("euhedral.cuda.library"));
        // The model of another class must not be on the device: this loads the artifact itself, at each context.
        SharedFlashNext.release();
        for (int context : CONTEXTS) {
            BitSet cpus = new BitSet();
            cpus.set(0);
            InferenceConfig config =
                    new InferenceConfig(artifact, Path.of("unused"), library, cpus, context, Duration.ofSeconds(30));
            Storage storage = Storage.load(config);
            CudaGpuMemory gpu = (CudaGpuMemory) storage.gpu();
            try {
                verify(storage.model(), gpu, artifact, context);
            } finally {
                storage.close();
            }
            assertEquals(0, gpu.allocatedBytes(), "device allocations after closing at context " + context);
            assertEquals(0, gpu.hostWeightBytes(), "pinned host allocations after closing at context " + context);
        }
    }

    /// The MTP layer and its experts load, and their experts move through the same cache, when the
    /// mode selects MTP.
    @Test
    @Order(2)
    void selectingMtpLoadsItsLayerAndItsExpertBank() throws Exception {
        Path artifact = artifactPath();
        assumeTrue(Files.isRegularFile(artifact), "no Flash-Next artifact at " + artifact);
        SharedFlashNext.release();
        CudaGpuMemory gpu = SharedQwen38.gpu();
        long allocatedBefore = gpu.allocatedBytes();
        long hostBefore = gpu.hostWeightBytes();
        var model = SharedFlashNext.model(SharedFlashNext.MTP).model();
        assertEquals(49, model.expertBanks().length);
        assertTrue(model.tensors().containsKey("mtp/fc_embedding"));
        int mtp = model.bankOrdinal("mtp/layers/0/moe/experts");
        ExpertBank bank = model.expertBanks()[mtp];
        try (ExpertLease lease = io.euhedral_execution.inference.core.model.qwen4.expert.ExpertTestSupport.acquire(
                        model.expertCache(), mtp, 511);
                Arena arena = Arena.ofConfined()) {
            MemorySegment back = arena.allocate(lease.byteSize());
            gpu.copyDeviceToHost(back, lease.deviceAddress(), lease.byteSize());
            assertArrayEquals(
                    readFile(artifact, bank.fileOffset(511), (int) bank.recordBytes(511)),
                    back.toArray(ValueLayout.JAVA_BYTE));
        }
        SharedFlashNext.release();
        assertEquals(allocatedBefore, gpu.allocatedBytes(), "device allocations after closing the MTP model");
        assertEquals(hostBefore, gpu.hostWeightBytes(), "pinned host allocations after closing the MTP model");
    }

    /// A device with far less free memory than the GPU has: fixed objects must leave it, the output
    /// head is read in place from mapped host memory, and the load still places every object and
    /// moves every expert correctly. The two tests above check that closing returns every allocation; this one
    /// leaves the model open for the classes that follow with the same small cache.
    @Test
    @Order(3)
    void placesFixedObjectsOnTheHostWhenTheDeviceIsSmall() throws Exception {
        Path artifact = artifactPath();
        assumeTrue(Files.isRegularFile(artifact), "no Flash-Next artifact at " + artifact);
        var loaded = SharedFlashNext.model(SharedFlashNext.SMALL_CACHE);
        var model = loaded.model();
        var gpu = loaded.gpu();
        var plan = model.plan();
        assertTrue(plan.host().stagedBytes() > 0, plan.report());
        assertEquals(StorageClass.HOST_MAPPED, plan.storageOf("text/output_head"));
        assertTrue(model.staging() != null);
        long budget = plan.device().freeBytes();
        assertTrue(budget <= 5L << 30, "planned with " + budget);
        assertTrue(
                gpu.allocatedBytes() - gpu.retainedScratchBytes() <= budget,
                "allocated " + gpu.allocatedBytes() + " of " + budget);
        verify(model, gpu, artifact, 262144);
    }

    private static void verify(Qwen4Model model, CudaGpuMemory gpu, Path artifact, int context) throws Exception {
        ResidencyPlan plan = model.plan();
        assertTrue(plan.fits(), plan.report());
        assertEquals(context, plan.maxContextTokens());
        System.out.println("context " + context + "\n" + plan.report());

        // device accounting: what the load allocated is what the plan placed, within the free memory it saw
        long planned = plan.device().fixedResidentBytes()
                + plan.device().stagingRingBytes()
                + plan.device().expertCacheBytes();
        long resident = gpu.allocatedBytes() - gpu.retainedScratchBytes();
        assertTrue(resident <= planned, "allocated " + resident + " of planned " + planned);
        assertTrue(plan.device().plannedBytes() <= plan.device().freeBytes());
        assertEquals(plan.device().expertCacheBytes(), model.expertCache().capacityBytes());

        // every component has a location
        for (Tensor tensor : model.artifact().tensors()) {
            StorageClass storage_ = plan.storageOf(tensor.name());
            boolean loaded = model.tensors().containsKey(tensor.name());
            switch (storage_) {
                case DEVICE_RESIDENT, HOST_MAPPED -> assertTrue(loaded, tensor.name());
                case HOST_STAGED -> assertTrue(loaded || tensor.group() == ComponentGroup.NGRAM, tensor.name());
                case DEFERRED -> assertTrue(!loaded, tensor.name());
                default -> throw new AssertionError(tensor.name() + " is " + storage_);
            }
        }
        assertEquals(StorageClass.DEFERRED, plan.storageOf("mtp/fc_embedding"));
        assertEquals(StorageClass.DEFERRED, plan.storageOf("vision/pos_embed/weight"));

        verifyFixed(model, gpu, artifact);
        verifyExperts(model, gpu, artifact);
        verifyNgram(model, artifact);
        System.out.println(model.telemetry());
    }

    /// Every loaded fixed object holds the artifact's bytes where the plan put it (the first 64 MiB
    /// of the big ones).
    private static void verifyFixed(Qwen4Model model, CudaGpuMemory gpu, Path artifact) throws IOException {
        int checked = 0;
        for (TensorHandle handle : model.tensors().values()) {
            int length = (int) Math.min(handle.byteSize(), 64 << 20);
            byte[] expected = readFile(
                    artifact,
                    model.artifact().tensor(handle.name()).orElseThrow().dataOffset(),
                    length);
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment back = arena.allocate(length);
                if (handle.hostBacked() || handle.hostMapped())
                    MemorySegment.copy(
                            MemorySegment.ofAddress(handle.hostAddress()).reinterpret(length), 0, back, 0, length);
                else gpu.copyDeviceToHost(back, handle.deviceAddress(), length);
                assertArrayEquals(expected, back.toArray(ValueLayout.JAVA_BYTE), handle.name());
            }
            checked++;
        }
        assertTrue(checked > 100);
    }

    private static void verifyExperts(Qwen4Model model, CudaGpuMemory gpu, Path artifact) throws Exception {
        ExpertCache cache = model.expertCache();
        ExpertBank[] banks = model.expertBanks();
        int slots = cache.slotCount();
        SplittableRandom random = new SplittableRandom(42);
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment back = arena.allocate(cache.slotBytes());
            // one pass over distinct experts beyond the cache's capacity (bank-major interleaved, so no key repeats):
            // every acquisition correct, evictions forced
            long distinct = (long) banks.length * banks[0].expertCount();
            int total = (int) Math.min(slots + 64L, distinct);
            for (int i = 0; i < total; i++) {
                int bank = i % banks.length;
                int expert = (i / banks.length) % banks[bank].expertCount();
                try (ExpertLease lease =
                        io.euhedral_execution.inference.core.model.qwen4.expert.ExpertTestSupport.acquire(
                                cache, bank, expert)) {
                    gpu.copyDeviceToHost(back, lease.deviceAddress(), lease.byteSize());
                    CRC32 crc = new CRC32();
                    crc.update(back.asSlice(0, lease.byteSize()).asByteBuffer());
                    assertEquals(banks[bank].crc32(expert), (int) crc.getValue(), "bank " + bank + " expert " + expert);
                }
            }
            // random re-use: hits and misses, all checked against the file
            for (int i = 0; i < 300; i++) {
                int bank = random.nextInt(banks.length);
                int expert = random.nextInt(banks[bank].expertCount());
                try (ExpertLease lease =
                        io.euhedral_execution.inference.core.model.qwen4.expert.ExpertTestSupport.acquire(
                                cache, bank, expert)) {
                    gpu.copyDeviceToHost(back, lease.deviceAddress(), lease.byteSize());
                    byte[] expected =
                            readFile(artifact, banks[bank].fileOffset(expert), (int) banks[bank].recordBytes(expert));
                    assertArrayEquals(
                            expected, back.asSlice(0, lease.byteSize()).toArray(ValueLayout.JAVA_BYTE));
                }
            }
        }
        var stats = cache.stats().snapshot();
        assertTrue(stats.misses() > 0 && stats.transferBytes() > 0);
        if (slots < banks.length * banks[0].expertCount())
            assertTrue(stats.evictions() > 0 || slots + 64 > stats.misses());
        assertEquals(0, cache.openLeaseCount());
    }

    private static void verifyNgram(Qwen4Model model, Path artifact) throws IOException {
        var store = model.ngram();
        SplittableRandom random = new SplittableRandom(7);
        for (int i = 0; i < 200; i++) {
            int head = random.nextInt(store.heads());
            long row = store.globalRow(
                    head, random.nextLong(model.artifact().config().ngram().headsVocabSizes()[head]));
            Tensor shard = model.artifact()
                    .tensor("text/layers/1/ple/ngram/shard_" + String.format("%03d", store.shardOf(row)))
                    .orElseThrow();
            byte[] expected =
                    readFile(artifact, shard.dataOffset() + store.localRow(row) * store.rowBytes(), store.rowBytes());
            assertArrayEquals(expected, store.row(row).toArray(ValueLayout.JAVA_BYTE), "row " + row);
        }
        assertTrue(store.hostBytes() > 28_000_000_000L);
    }

    private static byte[] readFile(Path path, long offset, int length) throws IOException {
        byte[] bytes = new byte[length];
        try (RandomAccessFile file = new RandomAccessFile(path.toFile(), "r")) {
            file.seek(offset);
            file.readFully(bytes);
        }
        return bytes;
    }
}
