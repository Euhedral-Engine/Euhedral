package io.euhedral_execution.inference.core.model_loader.qwen4.expert;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/// The cold store stages a record in the pinned staging buffer it is given: no wait, no lock.
class FileExpertStoreTest {

    @TempDir
    Path directory;

    @Test
    void aRecordIsReadIntoTheStagingBufferItIsGiven() throws Exception {
        ExpertFixture fixture = ExpertFixture.standard(this.directory, 11);
        HostBackedGpu gpu = new HostBackedGpu();
        try (FileExpertStore store = new FileExpertStore(gpu, fixture.file, fixture.banks, 3)) {
            assertEquals(3, store.stagingBuffers());
            for (int bank = 0; bank < fixture.banks.length; bank++) {
                for (int expert = 0; expert < fixture.expertCount(bank); expert++) {
                    HostRecord record = store.open(bank, expert, expert % 3);
                    byte[] bytes = record.segment().toArray(ValueLayout.JAVA_BYTE);
                    assertArrayEquals(fixture.record(bank, expert), bytes, bank + "/" + expert);
                    assertEquals(fixture.banks[bank].recordBytes(expert), record.byteSize());
                }
            }
            assertEquals(fixture.totalExperts(), store.recordOpens());
            assertTrue(store.bytesRead() > 0);
        }
        gpu.assertAllReleased();
    }

    @Test
    void buffersStageInDifferentPagesAndABufferIsReusedByItsNextOpen() throws Exception {
        ExpertFixture fixture = ExpertFixture.standard(this.directory, 12);
        HostBackedGpu gpu = new HostBackedGpu();
        try (FileExpertStore store = new FileExpertStore(gpu, fixture.file, fixture.banks, 2)) {
            HostRecord first = store.open(0, 0, 0);
            HostRecord second = store.open(0, 1, 1);
            assertNotEquals(first.hostAddress(), second.hostAddress());
            assertEquals(0, first.hostAddress() % 4096, "a buffer starts on a page");
            long address = first.hostAddress();
            // The buffer's next record replaces the previous one: its holder opens only after the copy that read it
            // retired.
            HostRecord next = store.open(0, 2, 0);
            assertEquals(address, next.hostAddress());
            assertArrayEquals(fixture.record(0, 2), next.segment().toArray(ValueLayout.JAVA_BYTE));
            assertArrayEquals(fixture.record(0, 1), second.segment().toArray(ValueLayout.JAVA_BYTE));
        }
    }

    @Test
    void rejectsOutOfRangeRecordsAndBuffers() throws Exception {
        ExpertFixture fixture = ExpertFixture.standard(this.directory, 13);
        HostBackedGpu gpu = new HostBackedGpu();
        try (FileExpertStore store = new FileExpertStore(gpu, fixture.file, fixture.banks, 2)) {
            assertThrows(IndexOutOfBoundsException.class, () -> store.open(0, 8, 0));
            assertThrows(IndexOutOfBoundsException.class, () -> store.open(5, 0, 0));
            assertThrows(IndexOutOfBoundsException.class, () -> store.open(0, 0, 2));
            assertThrows(IndexOutOfBoundsException.class, () -> store.open(0, 0, -1));
            assertEquals(0, store.recordOpens());
        }
    }

    @Test
    void aFileShorterThanTheBanksIsRejectedWithoutAllocating() throws Exception {
        ExpertFixture fixture = ExpertFixture.standard(this.directory, 14);
        Path shortFile = this.directory.resolve("shorter.bin");
        java.nio.file.Files.write(shortFile, java.util.Arrays.copyOf(fixture.bytes, fixture.bytes.length - 500));
        HostBackedGpu gpu = new HostBackedGpu();
        assertThrows(IllegalArgumentException.class, () -> new FileExpertStore(gpu, shortFile, fixture.banks, 2));
        gpu.assertAllReleased();
    }

    @Test
    void anUnreadableFileIsAnIoFailure() {
        HostBackedGpu gpu = new HostBackedGpu();
        assertThrows(
                IOException.class,
                () -> new FileExpertStore(
                        gpu,
                        this.directory.resolve("missing.bin"),
                        ExpertFixture.standard(this.directory, 15).banks,
                        2));
    }
}
