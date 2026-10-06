package io.euhedral_execution.inference.core.model_loader.qwen4.expert;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.model_loader.qwen4.ComponentGroup;
import java.io.IOException;
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class ArenaExpertStoreTest {

    @TempDir
    Path directory;

    @Test
    void loadsEveryRecordIdenticalToTheFileIntoOneArena() throws Exception {
        ExpertFixture fixture = ExpertFixture.standard(this.directory, 1);
        HostBackedGpu gpu = new HostBackedGpu();
        long total = 0;
        try (ArenaExpertStore store = new ArenaExpertStore(gpu, fixture.file, fixture.banks, 4)) {
            for (int bank = 0; bank < fixture.banks.length; bank++) {
                for (int expert = 0; expert < fixture.expertCount(bank); expert++) {
                    {
                        HostRecord record = store.open(bank, expert, 0);
                        byte[] bytes = record.segment().toArray(ValueLayout.JAVA_BYTE);
                        assertArrayEquals(fixture.record(bank, expert), bytes, bank + "/" + expert);
                        assertEquals(fixture.banks[bank].crc32(expert), ExpertFixture.crc32(bytes));
                        assertEquals(fixture.banks[bank].recordBytes(expert), record.byteSize());
                    }
                    total += fixture.banks[bank].recordBytes(expert);
                }
            }
            assertEquals(1, gpu.hostAllocationSizes().size(), "one arena, not one allocation per record");
            assertTrue(gpu.hostAllocationSizes().get(0) >= total);
            assertEquals(total, store.bytesRead());
            assertEquals(fixture.totalExperts(), store.recordOpens());
        }
        gpu.assertAllReleased();
    }

    @Test
    void recordsOfDifferentExpertsDoNotOverlap() throws Exception {
        ExpertFixture fixture = ExpertFixture.standard(this.directory, 2);
        HostBackedGpu gpu = new HostBackedGpu();
        try (ArenaExpertStore store = new ArenaExpertStore(gpu, fixture.file, fixture.banks, 2)) {
            List<long[]> ranges = new ArrayList<>();
            for (int bank = 0; bank < fixture.banks.length; bank++) {
                for (int expert = 0; expert < fixture.expertCount(bank); expert++) {
                    {
                        HostRecord record = store.open(bank, expert, 0);
                        ranges.add(new long[] {record.hostAddress(), record.hostAddress() + record.byteSize()});
                    }
                }
            }
            ranges.sort((a, b) -> Long.compare(a[0], b[0]));
            for (int i = 1; i < ranges.size(); i++)
                assertTrue(ranges.get(i - 1)[1] <= ranges.get(i)[0], "records " + (i - 1) + " and " + i + " overlap");
        }
    }

    @Test
    void largeArenaLoadsInParallelChunks() throws Exception {
        // 40 MiB in one bank: more than one loading task.
        ExpertFixture fixture = ExpertFixture.create(
                this.directory,
                3,
                List.of(
                        ExpertFixture.Spec.uniform("big", 40, 1 << 20, 1),
                        ExpertFixture.Spec.uniform("small", 5, 7000, 1)));
        HostBackedGpu gpu = new HostBackedGpu();
        try (ArenaExpertStore store = new ArenaExpertStore(gpu, fixture.file, fixture.banks, 8)) {
            for (int bank = 0; bank < fixture.banks.length; bank++) {
                for (int expert = 0; expert < fixture.expertCount(bank); expert++) {
                    {
                        HostRecord record = store.open(bank, expert, 0);
                        assertEquals(
                                fixture.banks[bank].crc32(expert),
                                ExpertFixture.crc32(record.segment().toArray(ValueLayout.JAVA_BYTE)));
                    }
                }
            }
        }
        gpu.assertAllReleased();
    }

    @Test
    void concurrentOpensSeeTheSameBytes() throws Exception {
        ExpertFixture fixture = ExpertFixture.standard(this.directory, 4);
        HostBackedGpu gpu = new HostBackedGpu();
        ExecutorService pool = Executors.newFixedThreadPool(12);
        try (ArenaExpertStore store = new ArenaExpertStore(gpu, fixture.file, fixture.banks, 3)) {
            List<Future<?>> futures = new ArrayList<>();
            for (int t = 0; t < 12; t++) {
                int seed = t;
                futures.add(pool.submit(() -> {
                    Random random = new Random(seed);
                    for (int i = 0; i < 500; i++) {
                        int bank = random.nextInt(fixture.banks.length);
                        int expert = random.nextInt(fixture.expertCount(bank));
                        {
                            HostRecord record = store.open(bank, expert, 0);
                            assertArrayEquals(
                                    fixture.record(bank, expert),
                                    record.segment().toArray(ValueLayout.JAVA_BYTE));
                        }
                    }
                    return null;
                }));
            }
            for (Future<?> future : futures) future.get(30, TimeUnit.SECONDS);
            assertEquals(12 * 500, store.recordOpens());
        } finally {
            pool.shutdownNow();
        }
        gpu.assertAllReleased();
    }

    @Test
    void rejectsRecordsOutsideTheFileWithoutAllocating() throws Exception {
        ExpertFixture fixture = ExpertFixture.standard(this.directory, 5);
        ExpertBank beyond = new ExpertBank(
                "beyond",
                ComponentGroup.ROUTED_EXPERT,
                0,
                fixture.banks[0].projections(),
                new long[] {fixture.bytes.length - 100},
                new long[] {9000},
                new int[] {0});
        HostBackedGpu gpu = new HostBackedGpu();
        assertThrows(
                IllegalArgumentException.class,
                () -> new ArenaExpertStore(gpu, fixture.file, new ExpertBank[] {beyond}, 2));
        assertEquals(0, gpu.liveHostAllocations());
        assertEquals(0, gpu.hostAllocationSizes().size());
    }

    @Test
    void rejectsOutOfRangeRecords() throws Exception {
        ExpertFixture fixture = ExpertFixture.standard(this.directory, 6);
        HostBackedGpu gpu = new HostBackedGpu();
        try (ArenaExpertStore store = new ArenaExpertStore(gpu, fixture.file, fixture.banks, 2)) {
            assertThrows(IndexOutOfBoundsException.class, () -> store.open(0, 8, 0));
            assertThrows(IndexOutOfBoundsException.class, () -> store.open(0, -1, 0));
            assertThrows(IndexOutOfBoundsException.class, () -> store.open(5, 0, 0));
            assertThrows(IndexOutOfBoundsException.class, () -> store.open(-1, 0, 0));
            assertEquals(0, store.recordOpens());
        }
    }

    @Test
    void closingFreesTheArenaOnce() throws Exception {
        ExpertFixture fixture = ExpertFixture.standard(this.directory, 7);
        HostBackedGpu gpu = new HostBackedGpu();
        ArenaExpertStore store = new ArenaExpertStore(gpu, fixture.file, fixture.banks, 2);
        HostRecord first = store.open(0, 0, 0);
        assertArrayEquals(fixture.record(0, 0), first.segment().toArray(ValueLayout.JAVA_BYTE));
        assertEquals(1, gpu.liveHostAllocations());
        store.close();
        store.close();
        assertEquals(0, gpu.liveHostAllocations());
        assertEquals(1, gpu.hostFrees(), "the arena is freed once");
        gpu.assertAllReleased();
    }

    @Test
    void rejectsAFileShorterThanTheBanksWithoutAllocating() throws Exception {
        ExpertFixture fixture = ExpertFixture.standard(this.directory, 8);
        Path shortFile = this.directory.resolve("shorter.bin");
        java.nio.file.Files.write(shortFile, java.util.Arrays.copyOf(fixture.bytes, fixture.bytes.length - 500));
        HostBackedGpu gpu = new HostBackedGpu();
        assertThrows(IllegalArgumentException.class, () -> new ArenaExpertStore(gpu, shortFile, fixture.banks, 2));
        gpu.assertAllReleased();
    }

    @Test
    void unreadableFileIsAnIoFailure() {
        HostBackedGpu gpu = new HostBackedGpu();
        assertThrows(
                IOException.class,
                () -> new ArenaExpertStore(
                        gpu,
                        this.directory.resolve("missing.bin"),
                        ExpertFixture.standard(this.directory, 9).banks,
                        2));
        assertEquals(0, gpu.liveHostAllocations());
    }
}
