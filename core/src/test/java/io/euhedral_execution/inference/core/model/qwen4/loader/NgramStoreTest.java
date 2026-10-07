package io.euhedral_execution.inference.core.model.qwen4.loader;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class NgramStoreTest {

    @TempDir
    Path directory;

    Path path;
    Artifact artifact;

    @BeforeEach
    void write() throws IOException {
        this.path = this.directory.resolve("mini.edrl");
        this.artifact = TestArtifact.write(this.path, TestArtifact.miniConfig(), 11);
    }

    byte[] fileRow(int shard, long localRow, int rowBytes) throws IOException {
        Tensor tensor =
                this.artifact.tensor("text/layers/1/ple/ngram/shard_00" + shard).orElseThrow();
        byte[] row = new byte[rowBytes];
        try (RandomAccessFile file = new RandomAccessFile(this.path.toFile(), "r")) {
            file.seek(tensor.dataOffset() + localRow * rowBytes);
            file.readFully(row);
        }
        return row;
    }

    @Test
    void addressesRowsThroughHeadsAndShardsWithoutAliasing() throws IOException {
        try (NgramStore store = NgramStore.open(this.path, this.artifact, ResidencyPlan.NgramMode.MAPPED_FILE, null)) {
            assertEquals(2, store.heads());
            assertEquals(128, store.rowWidth());
            assertEquals(64 + 8, store.rowBytes());
            assertEquals(4 * 21, store.totalRows());
            // head 0 covers rows 0..40, head 1 rows 41..83; shard boundaries every 21 rows
            assertEquals(0, store.globalRow(0, 0));
            assertEquals(40, store.globalRow(0, 40));
            assertEquals(41, store.globalRow(1, 0));
            assertEquals(83, store.globalRow(1, 42));
            assertThrows(IndexOutOfBoundsException.class, () -> store.globalRow(0, 41));
            assertThrows(IndexOutOfBoundsException.class, () -> store.globalRow(1, 43));
            assertThrows(IndexOutOfBoundsException.class, () -> store.globalRow(2, 0));
            assertEquals(0, store.shardOf(20));
            assertEquals(1, store.shardOf(21));
            assertEquals(0, store.localRow(21));
            assertEquals(20, store.localRow(41));
            assertEquals(3, store.shardOf(83));
            assertThrows(IndexOutOfBoundsException.class, () -> store.shardOf(84));
            assertThrows(IndexOutOfBoundsException.class, () -> store.row(-1));
            // every row is distinct bytes of its own shard
            for (long row = 0; row < store.totalRows(); row++) {
                int shard = store.shardOf(row);
                assertArrayEquals(
                        fileRow(shard, store.localRow(row), store.rowBytes()),
                        store.row(row).toArray(ValueLayout.JAVA_BYTE),
                        "row " + row);
            }
            for (int shard = 0; shard < store.shardCount(); shard++) assertEquals(0.125f, store.globalScale(shard));
        }
    }

    @Test
    void gathersRowsIntoHostMemoryAndCountsThem() throws IOException {
        try (NgramStore store = NgramStore.open(this.path, this.artifact, ResidencyPlan.NgramMode.MAPPED_FILE, null);
                Arena arena = Arena.ofConfined()) {
            long[] rows = {83, 0, 41, 21, 83};
            MemorySegment out = arena.allocate((long) rows.length * store.rowBytes());
            store.gather(rows, rows.length, out);
            for (int i = 0; i < rows.length; i++)
                assertArrayEquals(
                        store.row(rows[i]).toArray(ValueLayout.JAVA_BYTE),
                        out.asSlice((long) i * store.rowBytes(), store.rowBytes())
                                .toArray(ValueLayout.JAVA_BYTE));
            assertEquals(1, store.stats().gathers());
            assertEquals(rows.length, store.stats().rowsGathered());
            assertEquals((long) rows.length * store.rowBytes(), store.stats().bytesGathered());
            assertThrows(IllegalArgumentException.class, () -> store.gather(rows, 5, arena.allocate(8)));
        }
    }

    @Test
    void pinnedArenaHoldsTheSameRowsAndFreesItself() throws IOException {
        HostMemoryGpu gpu = new HostMemoryGpu();
        try (NgramStore store = NgramStore.open(this.path, this.artifact, ResidencyPlan.NgramMode.PINNED_ARENA, gpu)) {
            assertEquals(1, gpu.livePinnedAllocations());
            for (long row = 0; row < store.totalRows(); row += 5)
                assertArrayEquals(
                        fileRow(store.shardOf(row), store.localRow(row), store.rowBytes()),
                        store.row(row).toArray(ValueLayout.JAVA_BYTE));
        }
        assertEquals(0, gpu.livePinnedAllocations());
    }

    @Test
    void stagingCopiesOnlyTheRequestedRowsToTheDevice() throws IOException {
        HostMemoryGpu gpu = new HostMemoryGpu();
        try (NgramStore store = NgramStore.open(this.path, this.artifact, ResidencyPlan.NgramMode.MAPPED_FILE, null)) {
            long[] rows = {5, 70, 41};
            long device = gpu.allocate((long) rows.length * store.rowBytes());
            try (ExecutionGpu.UploadBuffer upload = store.stage(gpu, rows, rows.length, device);
                    Arena arena = Arena.ofConfined()) {
                MemorySegment back = arena.allocate((long) rows.length * store.rowBytes());
                gpu.copyDeviceToHost(back, device, back.byteSize());
                for (int i = 0; i < rows.length; i++)
                    assertArrayEquals(
                            store.row(rows[i]).toArray(ValueLayout.JAVA_BYTE),
                            back.asSlice((long) i * store.rowBytes(), store.rowBytes())
                                    .toArray(ValueLayout.JAVA_BYTE));
            }
            gpu.free(device);
            assertEquals((long) rows.length * store.rowBytes(), store.stats().bytesStagedToDevice());
            // the whole table never reaches the device
            assertTrue(gpu.hostToDeviceBytes() < store.hostBytes());
        }
        assertEquals(0, gpu.liveDeviceBytes());
    }

    @Test
    void rangesOfOneListGatheredSideBySideFillOneBufferLikeAWholeGather() throws Exception {
        try (NgramStore store = NgramStore.open(this.path, this.artifact, ResidencyPlan.NgramMode.MAPPED_FILE, null);
                Arena arena = Arena.ofShared()) {
            long[] rows = new long[40];
            for (int i = 0; i < rows.length; i++) rows[i] = (i * 37L) % store.totalRows();
            MemorySegment whole = arena.allocate((long) rows.length * store.recordBytes());
            store.gatherRecords(rows, rows.length, whole);
            MemorySegment parts = arena.allocate((long) rows.length * store.recordBytes());
            parts.fill((byte) 0x5a);
            Thread[] threads = new Thread[4];
            for (int part = 0; part < threads.length; part++) {
                int from = rows.length * part / threads.length;
                int to = rows.length * (part + 1) / threads.length;
                threads[part] = new Thread(() -> store.gatherRecordsRange(rows, from, to, parts));
                threads[part].start();
            }
            for (Thread thread : threads) thread.join();
            assertArrayEquals(whole.toArray(ValueLayout.JAVA_BYTE), parts.toArray(ValueLayout.JAVA_BYTE));
            assertThrows(IllegalArgumentException.class, () -> store.gatherRecordsRange(rows, 30, 41, parts));
            assertThrows(IllegalArgumentException.class, () -> store.gatherRecordsRange(rows, 5, 4, parts));
        }
    }

    @Test
    void closedStoresRefuseWork() throws IOException {
        NgramStore store = NgramStore.open(this.path, this.artifact, ResidencyPlan.NgramMode.MAPPED_FILE, null);
        store.close();
        store.close();
        assertThrows(IllegalStateException.class, () -> store.gather(new long[] {0}, 1, MemorySegment.NULL));
    }
}
