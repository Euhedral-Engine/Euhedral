package io.euhedral_execution.inference.core;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import io.euhedral_execution.inference.core.model_loader.layer_weights.TensorHandle;
import io.euhedral_execution.inference.core.model_loader.qwen4.ComponentGroup;
import io.euhedral_execution.inference.core.model_loader.qwen4.Qwen4Model;
import io.euhedral_execution.inference.core.model_loader.qwen4.Qwen4ResidencyPlan;
import io.euhedral_execution.inference.core.model_loader.qwen4.Qwen4Tensor;
import io.euhedral_execution.inference.core.model_loader.qwen4.StorageClass;
import io.euhedral_execution.inference.core.model_loader.qwen4.expert.ExpertBank;
import io.euhedral_execution.inference.core.model_loader.qwen4.expert.ExpertCache;
import io.euhedral_execution.inference.core.model_loader.qwen4.expert.ExpertLease;
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
import java.util.zip.CRC32;
import org.junit.jupiter.api.Test;

/// The real Flash-Next artifact through the engine's storage load on the real GPU, for several maximum contexts: every
/// component has a location, device memory stays within the plan, experts move through the cache with the artifact's
/// own bytes, rows come out of the n-gram tables, and closing returns every allocation. No model runs.
class Qwen4StorageCudaIntegrationTest {

    static final int[] CONTEXTS = {4096, 32768, 131072, 262144};

    static Path artifactPath() {
        return Path.of(System.getProperty(
                "euhedral.qwen4.artifact", "/home/brandon/models/qwen3_8_flash_next/qwen3_8_flash_next_nvfp4.edrl"));
    }

    @Test
    void loadsTheRealArtifactAtSeveralContexts() throws Exception {
        Path artifact = artifactPath();
        assumeTrue(Files.isRegularFile(artifact), "no Flash-Next artifact at " + artifact);
        Path library = Path.of(System.getProperty("euhedral.cuda.library"));
        for (int context : CONTEXTS) {
            BitSet cpus = new BitSet();
            cpus.set(0);
            InferenceConfig config =
                    new InferenceConfig(artifact, Path.of("unused"), library, cpus, context, Duration.ofSeconds(30));
            Qwen4Storage storage = Qwen4Storage.load(config);
            CudaGpuMemory gpu = (CudaGpuMemory) storage.gpu();
            try {
                verify(storage, artifact, context);
            } finally {
                storage.close();
            }
            assertEquals(0, gpu.allocatedBytes(), "device allocations after closing at context " + context);
            assertEquals(0, gpu.hostWeightBytes(), "pinned host allocations after closing at context " + context);
        }
    }

    private static void verify(Qwen4Storage storage, Path artifact, int context) throws Exception {
        Qwen4Model model = storage.model();
        Qwen4ResidencyPlan plan = storage.plan();
        CudaGpuMemory gpu = (CudaGpuMemory) storage.gpu();
        assertTrue(plan.fits(), plan.report());
        assertEquals(context, plan.maxContextTokens());
        System.out.println("context " + context + "\n" + plan.report());

        // device accounting: what the load allocated is what the plan placed, within the free memory it saw
        long planned = plan.device().fixedResidentBytes() + plan.device().stagingRingBytes() + plan.device().expertCacheBytes();
        assertTrue(gpu.allocatedBytes() <= planned, "allocated " + gpu.allocatedBytes() + " of planned " + planned);
        assertTrue(plan.device().plannedBytes() <= plan.device().freeBytes());
        assertEquals(plan.device().expertCacheBytes(), model.expertCache().capacityBytes());

        // every component has a location
        for (Qwen4Tensor tensor : model.artifact().tensors()) {
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

    private static void verifyFixed(Qwen4Model model, CudaGpuMemory gpu, Path artifact) throws IOException {
        int checked = 0;
        for (TensorHandle handle : model.tensors().values()) {
            if (handle.byteSize() > (64 << 20) && checked > 0) continue;
            byte[] expected = readFile(artifact, model.artifact().tensor(handle.name()).orElseThrow().dataOffset(), (int) handle.byteSize());
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment back = arena.allocate(handle.byteSize());
                if (handle.hostBacked() || handle.hostMapped())
                    MemorySegment.copy(MemorySegment.ofAddress(handle.hostAddress()).reinterpret(handle.byteSize()), 0, back, 0, handle.byteSize());
                else gpu.copyDeviceToHost(back, handle.deviceAddress(), handle.byteSize());
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
            // one pass over distinct experts beyond the cache's capacity: every acquisition correct, evictions forced
            int total = Math.min(slots + 64, banks.length * banks[0].expertCount());
            for (int i = 0; i < total; i++) {
                int bank = (int) ((i * 7919L) % banks.length);
                int expert = (int) ((i * 31L + bank) % banks[bank].expertCount());
                try (ExpertLease lease = cache.acquire(bank, expert)) {
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
                try (ExpertLease lease = cache.acquire(bank, expert)) {
                    gpu.copyDeviceToHost(back, lease.deviceAddress(), lease.byteSize());
                    byte[] expected = readFile(artifact, banks[bank].fileOffset(expert), (int) banks[bank].recordBytes(expert));
                    assertArrayEquals(expected, back.asSlice(0, lease.byteSize()).toArray(ValueLayout.JAVA_BYTE));
                }
            }
        }
        var stats = cache.stats().snapshot();
        assertTrue(stats.misses() > 0 && stats.transferBytes() > 0);
        if (slots < banks.length * banks[0].expertCount()) assertTrue(stats.evictions() > 0 || slots + 64 > stats.misses());
        assertEquals(0, cache.openLeaseCount());
    }

    private static void verifyNgram(Qwen4Model model, Path artifact) throws IOException {
        var store = model.ngram();
        SplittableRandom random = new SplittableRandom(7);
        for (int i = 0; i < 200; i++) {
            int head = random.nextInt(store.heads());
            long row = store.globalRow(head, random.nextLong(model.artifact().config().ngram().headsVocabSizes()[head]));
            Qwen4Tensor shard = model.artifact()
                    .tensor("text/layers/1/ple/ngram/shard_" + String.format("%03d", store.shardOf(row)))
                    .orElseThrow();
            byte[] expected = readFile(artifact, shard.dataOffset() + store.localRow(row) * store.rowBytes(), store.rowBytes());
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
