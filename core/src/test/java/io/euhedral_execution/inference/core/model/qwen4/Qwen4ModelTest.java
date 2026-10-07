package io.euhedral_execution.inference.core.model.qwen4;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.artifact.TensorHandle;
import io.euhedral_execution.inference.core.model.qwen4.expert.ExpertBank;
import io.euhedral_execution.inference.core.model.qwen4.expert.ExpertLease;
import io.euhedral_execution.inference.core.model.qwen4.expert.ExpertTestSupport;
import io.euhedral_execution.inference.core.model.qwen4.loader.Artifact;
import io.euhedral_execution.inference.core.model.qwen4.loader.HostBudget;
import io.euhedral_execution.inference.core.model.qwen4.loader.HostMemoryGpu;
import io.euhedral_execution.inference.core.model.qwen4.loader.Mode;
import io.euhedral_execution.inference.core.model.qwen4.loader.ResidencyPlan;
import io.euhedral_execution.inference.core.model.qwen4.loader.ResidencyPlanner;
import io.euhedral_execution.inference.core.model.qwen4.loader.SequenceState;
import io.euhedral_execution.inference.core.model.qwen4.loader.StorageClass;
import io.euhedral_execution.inference.core.model.qwen4.loader.Tensor;
import io.euhedral_execution.inference.core.model.qwen4.loader.TestArtifact;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;
import java.util.SplittableRandom;
import java.util.zip.CRC32;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/// The storage load end to end on the host-memory test GPU, with the mini artifact: placement,
/// device and host accounting, the expert cache's bytes against the file, and that closing returns
/// every allocation.
class Qwen4ModelTest {

    static final long GIB = 1L << 30;
    static final int CONTEXT = 2048;

    @TempDir
    Path directory;

    Path path;
    Artifact artifact;

    @BeforeEach
    void write() throws IOException {
        this.path = this.directory.resolve("mini.edrl");
        this.artifact = TestArtifact.write(this.path, TestArtifact.miniConfig(), 3);
    }

    /// Free device memory that leaves exactly `slots` expert slots beside the mini model's other
    /// needs.
    long freeFor(Mode mode, int slots) {
        var roomy = ResidencyPlanner.plan(this.artifact, mode, 64 * GIB, HostBudget.ofAvailable(64 * GIB), CONTEXT);
        // A roomy device plans a longer prefill chunk than a tight one can afford: count the tight one's.
        return roomy.device().plannedBytes()
                - roomy.device().expertCacheBytes()
                - roomy.device().workspaceBytes()
                + SequenceState.workspaceBytes(this.artifact.config())
                + ResidencyPlanner.sharedDownPaddingBytes(this.artifact.config())
                + slots * roomy.expertCache().slotBytes();
    }

    byte[] fileBytes(long offset, int length) throws IOException {
        byte[] bytes = new byte[length];
        try (RandomAccessFile file = new RandomAccessFile(this.path.toFile(), "r")) {
            file.seek(offset);
            file.readFully(bytes);
        }
        return bytes;
    }

    @Test
    void servesExpertsFromResidentRamWithTheArtifactsBytes() throws Exception {
        exercise(HostBudget.ofAvailable(64 * GIB), ResidencyPlan.ExpertStoreMode.RAM_RESIDENT);
    }

    @Test
    void servesExpertsFromABoundedRamCacheWithTheArtifactsBytes() throws Exception {
        // Room for 26 of the 32 expert records in ordinary memory: more than the 20 device slots, fewer than all.
        var reference =
                ResidencyPlanner.plan(this.artifact, Mode.TEXT, 64 * GIB, HostBudget.ofAvailable(64 * GIB), CONTEXT);
        long recordSlot =
                reference.host().expertRamBytes() / reference.expertCache().totalExperts();
        long resident = reference.host().pinnedBytes() + 26 * recordSlot;
        exercise(new HostBudget(64 * GIB, resident), ResidencyPlan.ExpertStoreMode.RAM_CACHED);
    }

    @Test
    void servesExpertsFromTheFileThroughStagingSlots() throws Exception {
        // 1 MiB holds neither the embedding nor an expert tier larger than the device cache
        exercise(new HostBudget(1 << 20), ResidencyPlan.ExpertStoreMode.FILE_BACKED);
    }

    private void exercise(HostBudget host, ResidencyPlan.ExpertStoreMode expectedStore) throws Exception {
        HostMemoryGpu gpu = new HostMemoryGpu();
        int slots = 20;
        long free = freeFor(Mode.TEXT, slots);
        long acquisitions = 0;
        try (Qwen4Model model = Qwen4Model.open(this.path, gpu, free, host, Mode.TEXT, CONTEXT)) {
            var plan = model.plan();
            assertTrue(plan.fits(), plan.report());
            assertEquals(expectedStore, plan.expertStore());
            assertEquals(slots, model.expertCache().slotCount());
            assertEquals(4, model.expertBanks().length); // the MTP bank is deferred
            assertEquals(StorageClass.DEFERRED, plan.storageOf("mtp/layers/0/moe/experts"));
            assertNull(model.staging(), "nothing moved off the device");
            assertTrue(gpu.livePinnedAllocations() > 0);

            // device accounting: the fixed tensors and one slab for the cache, within the free memory the plan saw
            assertEquals(
                    plan.device().fixedResidentBytes() + model.expertCache().capacityBytes(), gpu.liveDeviceBytes());
            assertTrue(gpu.liveDeviceBytes() <= plan.device().freeBytes());

            // every loaded fixed object holds the file's bytes where the plan put it
            for (TensorHandle handle : model.tensors().values()) {
                byte[] expected = fileBytes(
                        model.artifact().tensor(handle.name()).orElseThrow().dataOffset(), (int) handle.byteSize());
                try (Arena arena = Arena.ofConfined()) {
                    MemorySegment back = arena.allocate(handle.byteSize());
                    if (handle.hostMapped() || handle.hostBacked())
                        MemorySegment.copy(
                                MemorySegment.ofAddress(handle.hostAddress()).reinterpret(handle.byteSize()),
                                0,
                                back,
                                0,
                                handle.byteSize());
                    else gpu.copyDeviceToHost(back, handle.deviceAddress(), handle.byteSize());
                    assertArrayEquals(expected, back.toArray(ValueLayout.JAVA_BYTE), handle.name());
                }
            }
            assertEquals(StorageClass.HOST_MAPPED, plan.storageOf("text/token_embedding"));
            assertTrue(model.tensor("text/token_embedding").hostMapped());
            assertTrue(model.tensors().keySet().stream()
                    .noneMatch(name -> name.startsWith("mtp/") || name.startsWith("vision/")));
            assertTrue(model.tensors().keySet().stream().noneMatch(name -> name.contains("/ngram/")));

            // every expert of every cached bank, repeatedly and in shuffled order, with more experts than slots
            ExpertBank[] banks = model.expertBanks();
            SplittableRandom random = new SplittableRandom(5);
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment back = arena.allocate(model.expertCache().slotBytes());
                for (int round = 0; round < 40; round++) {
                    int bank = random.nextInt(banks.length);
                    int expert = random.nextInt(banks[bank].expertCount());
                    try (ExpertLease lease = ExpertTestSupport.acquire(model.expertCache(), bank, expert)) {
                        acquisitions++;
                        gpu.copyDeviceToHost(back, lease.deviceAddress(), lease.byteSize());
                        assertArrayEquals(
                                fileBytes(banks[bank].fileOffset(expert), (int) banks[bank].recordBytes(expert)),
                                back.asSlice(0, lease.byteSize()).toArray(ValueLayout.JAVA_BYTE));
                        CRC32 crc = new CRC32();
                        crc.update(back.asSlice(0, lease.byteSize()).asByteBuffer());
                        assertEquals(banks[bank].crc32(expert), (int) crc.getValue());
                    }
                }
            }
            var hierarchy = model.hierarchyStats();
            assertEquals(
                    model.expertCache().stats().snapshot().misses(),
                    hierarchy.gpu().misses());
            if (expectedStore == ResidencyPlan.ExpertStoreMode.FILE_BACKED) {
                assertNull(hierarchy.ram());
                assertEquals(hierarchy.gpu().misses(), hierarchy.artifact().recordReads());
            } else {
                assertTrue(hierarchy.ram().totalHits() + hierarchy.ram().totalMisses() > 0);
                assertEquals(hierarchy.gpu().misses(), hierarchy.staging().recordsStaged());
                if (expectedStore == ResidencyPlan.ExpertStoreMode.RAM_RESIDENT) {
                    assertEquals(32, hierarchy.ram().residentExperts());
                    assertEquals(0, hierarchy.ram().totalMisses(), "a resident tier reads nothing from the artifact");
                    assertEquals(hierarchy.gpu().misses(), hierarchy.ram().totalHits());
                } else {
                    assertTrue(hierarchy.ram().residentExperts() <= 26);
                }
            }
            var telemetry = model.telemetry();
            assertEquals(slots, telemetry.expertSlots());
            assertEquals(model.expertCache().capacityBytes(), telemetry.expertCacheBytes());
            assertEquals(acquisitions, telemetry.expertCacheHits() + telemetry.expertCacheMisses());
            assertTrue(telemetry.expertCacheMisses() >= slots);
            assertTrue(telemetry.expertEvictions() > 0, "32 experts through 20 slots must evict");
            assertTrue(telemetry.expertTransferBytes() > 0);
            assertEquals(plan.device().fixedResidentBytes(), telemetry.fixedDeviceBytes());
            assertEquals(model.ngram().hostBytes(), telemetry.ngramHostBytes());
            // the device never holds more than the plan placed
            assertEquals(
                    gpu.peakDeviceBytes(),
                    plan.device().fixedResidentBytes() + model.expertCache().capacityBytes());
        }
        assertEquals(0, gpu.liveDeviceAllocations());
        assertEquals(0, gpu.livePinnedAllocations());
    }

    @Test
    void selectingMtpLoadsItsLayerAndCachesItsExperts() throws Exception {
        HostMemoryGpu gpu = new HostMemoryGpu();
        Mode mode = new Mode(true, false);
        try (Qwen4Model model =
                Qwen4Model.open(this.path, gpu, freeFor(mode, 20), HostBudget.ofAvailable(64 * GIB), mode, CONTEXT)) {
            assertEquals(5, model.expertBanks().length);
            assertEquals(StorageClass.DEVICE_CACHED, model.plan().storageOf("mtp/layers/0/moe/experts"));
            assertTrue(model.tensors().containsKey("mtp/fc_embedding"));
            assertTrue(model.bankOrdinal("mtp/layers/0/moe/experts") >= 0);
            try (ExpertLease lease =
                    ExpertTestSupport.acquire(model.expertCache(), model.bankOrdinal("mtp/layers/0/moe/experts"), 7)) {
                assertTrue(lease.isValid());
            }
        }
        assertEquals(0, gpu.liveDeviceAllocations());
        assertEquals(0, gpu.livePinnedAllocations());
    }

    @Test
    void refusesAContextBeyondTheModelBeforeAllocatingAnything() {
        HostMemoryGpu gpu = new HostMemoryGpu();
        var failure = assertThrows(
                IOException.class,
                () -> Qwen4Model.open(this.path, gpu, 64 * GIB, HostBudget.ofAvailable(64 * GIB), Mode.TEXT, 4097));
        assertTrue(failure.getMessage().contains("4096 positions"), failure.getMessage());
        assertEquals(0, gpu.liveDeviceAllocations());
        assertEquals(0, gpu.livePinnedAllocations());
    }

    @Test
    void refusesAnImpossibleDeviceWithTheReasonAndAllocatesNothing() {
        HostMemoryGpu gpu = new HostMemoryGpu();
        var failure = assertThrows(
                IOException.class,
                () -> Qwen4Model.open(
                        this.path, gpu, 512L << 20, HostBudget.ofAvailable(64 * GIB), Mode.TEXT, CONTEXT));
        assertTrue(failure.getMessage().contains("MiB free"), failure.getMessage());
        assertEquals(0, gpu.liveDeviceAllocations());
        assertEquals(0, gpu.livePinnedAllocations());
    }

    @Test
    void aFailedLoadReleasesWhatItAllocated() throws Exception {
        // corrupt the payload of a fixed NVFP4 tensor's scale plane (a NaN scale) so the load fails after some objects
        // loaded
        Tensor victim = this.artifact
                .tensor("text/layers/3/moe/shared_expert/down_proj")
                .orElseThrow();
        long scaleOffset = victim.dataOffset()
                + io.euhedral_execution.inference.core.artifact.Nvfp4Layout.scaleOffset(
                        victim.shape()[0], victim.shape()[1]);
        try (RandomAccessFile file = new RandomAccessFile(this.path.toFile(), "rw")) {
            file.seek(scaleOffset);
            file.write(0x7f);
        }
        HostMemoryGpu gpu = new HostMemoryGpu();
        assertThrows(
                IOException.class,
                () -> Qwen4Model.open(
                        this.path, gpu, freeFor(Mode.TEXT, 20), HostBudget.ofAvailable(64 * GIB), Mode.TEXT, CONTEXT));
        assertEquals(0, gpu.liveDeviceAllocations());
        assertEquals(0, gpu.livePinnedAllocations());
    }
}
