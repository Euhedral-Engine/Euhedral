package io.euhedral_execution.inference.core.scheduling;

import static org.junit.jupiter.api.Assertions.*;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class AttentionKvStateTest {
    @Test
    void pagedGrowthPreservesPayloadAddressesWithoutCopyingCache() {
        RecordingGpu gpu = new RecordingGpu();
        try (AttentionKvState state = new AttentionKvState(gpu, 1024)) {
            state.prepareAppend(0, 255);
            assertEquals(256, state.capacity());
            long page = gpu.tableEntries.get(state.keyCacheAddress())[0];
            commit(state, 255);
            state.prepareAppend(255, 2);
            assertEquals(512, state.capacity());
            assertEquals(page, gpu.tableEntries.get(state.keyCacheAddress())[0]);
            assertTrue(gpu.copySizes.isEmpty(), "growth must never copy existing KV payloads");
            assertEquals(255, state.length());
            commit(state, 2);
            assertEquals(257, state.length());
        }
        assertTrue(gpu.allocations.isEmpty());
    }

    @Test
    void sixtyFourKUsesNvfp4PayloadAndHasNoDoublingPeak() {
        RecordingGpu gpu = new RecordingGpu();
        try (AttentionKvState state = new AttentionKvState(gpu, 1024)) {
            state.prepareAppend(0, 65536);
            assertEquals(65536, state.capacity());
            // 4 heads * (128 packed bytes + 16 scale bytes) * K/V.
            assertEquals(
                    65536L * 4 * 144 * 2 + 256L * 8 * 2,
                    gpu.allocations.values().stream().mapToLong(Long::longValue).sum());
            commit(state, 65536);
            state.prepareAppend(65536, 1);
            assertEquals(65792, state.capacity());
            assertTrue(gpu.copySizes.isEmpty());
        }
        assertTrue(gpu.allocations.isEmpty());
    }

    @Test
    void failedPageAllocationRemainsClosableAndLengthUnpublished() {
        RecordingGpu gpu = new RecordingGpu();
        AttentionKvState state = new AttentionKvState(gpu, 1024);
        gpu.failAfter = 2;
        assertThrows(IllegalStateException.class, () -> state.prepareAppend(0, 1024));
        assertEquals(0, state.length());
        state.close();
        assertTrue(gpu.allocations.isEmpty());
        state.close();
    }

    @Test
    void rejectsGapsAndDoesNotAdvanceLengthBeforeAppendCommit() {
        RecordingGpu gpu = new RecordingGpu();
        try (AttentionKvState state = new AttentionKvState(gpu, 256)) {
            assertThrows(IllegalArgumentException.class, () -> state.prepareAppend(1, 1));
            state.prepareAppend(0, 2);
            assertEquals(0, state.length());
            assertEquals(0, state.submittedLength(), "reserved rows are not yet submitted");
            state.appendSubmitted(2);
            assertEquals(2, state.submittedLength(), "the submitting quantum may read its own rows");
            assertEquals(0, state.length(), "submitted rows are not committed");
            assertThrows(IllegalStateException.class, () -> state.prepareAppend(0, 1), "one pending append");
            state.commitSubmitted();
            assertEquals(2, state.length());
            assertThrows(IllegalArgumentException.class, () -> state.prepareAppend(1, 1));
            assertThrows(IllegalArgumentException.class, () -> state.appendSubmitted(257));
        }
        assertThrows(IllegalArgumentException.class, () -> new AttentionKvState(gpu, 4));
    }

    @Test
    void discardedSubmissionNeverBecomesVisible() {
        RecordingGpu gpu = new RecordingGpu();
        try (AttentionKvState state = new AttentionKvState(gpu, 256)) {
            state.prepareAppend(0, 3);
            state.appendSubmitted(3);
            state.discardSubmitted();
            assertEquals(0, state.length());
            assertEquals(0, state.submittedLength());
            state.prepareAppend(0, 1);
            commit(state, 1);
            assertEquals(1, state.length());
        }
    }

    private static void commit(AttentionKvState state, int tokens) {
        state.appendSubmitted(tokens);
        state.commitSubmitted();
    }

    @Test
    void failedTableUploadCanBeRetriedWithoutPublishingLengthOrLeaking() {
        RecordingGpu gpu = new RecordingGpu();
        try (AttentionKvState state = new AttentionKvState(gpu, 1024)) {
            gpu.failUpload = true;
            assertThrows(IllegalStateException.class, () -> state.prepareAppend(0, 257));
            assertEquals(0, state.length());
            assertEquals(0, state.capacity());
            gpu.failUpload = false;
            state.prepareAppend(0, 257);
            assertEquals(512, state.capacity());
            assertTrue(gpu.copySizes.isEmpty());
            commit(state, 257);
            assertEquals(3, gpu.allocations.size(), "two payload pages and one current table");
        }
        assertTrue(gpu.allocations.isEmpty());
    }

    @Test
    void grownTableIsUploadedFromStagingThatLivesUntilTheAppendSettles() {
        RecordingGpu gpu = new RecordingGpu();
        try (AttentionKvState state = new AttentionKvState(gpu, 1024)) {
            state.prepareAppend(0, 1);
            long first = state.keyCacheAddress();
            assertEquals(1, gpu.stagings);
            assertEquals(0, gpu.stagingReleases, "a queued upload may still read its staging");
            commit(state, 1);
            assertEquals(1, gpu.stagingReleases, "retirement releases the staging");

            state.prepareAppend(1, 256);
            long grown = state.keyCacheAddress();
            assertNotEquals(first, grown);
            assertTrue(gpu.allocations.containsKey(first), "freeing a table mid-quantum would wait on the device");
            assertEquals(1, gpu.stagingReleases);
            state.appendSubmitted(256);
            state.discardSubmitted();
            assertFalse(gpu.allocations.containsKey(first), "the outgrown table is released after retirement");
            assertEquals(2, gpu.stagingReleases);
            assertEquals(1, state.length(), "a discarded append stays invisible");
        }
        assertTrue(gpu.allocations.isEmpty());
    }

    @Test
    void unprovenCompletionRetainsStagingThatQueuedCopiesMayStillRead() {
        RecordingGpu gpu = new RecordingGpu();
        AttentionKvState state = new AttentionKvState(gpu, 1024);
        state.prepareAppend(0, 1);
        gpu.proven = false;
        state.discardSubmitted();
        assertEquals(0, gpu.stagingReleases);
        gpu.proven = true;
        state.close();
        assertEquals(1, gpu.stagingReleases);
        assertTrue(gpu.allocations.isEmpty());
    }

    @Test
    void releaseFailureRetainsOwnershipForRetryWithoutDoubleFree() {
        RecordingGpu gpu = new RecordingGpu();
        AttentionKvState state = new AttentionKvState(gpu, 1024);
        state.prepareAppend(0, 1);
        gpu.failFreeAddress = gpu.tableEntries.get(state.keyCacheAddress())[0];
        assertThrows(IllegalStateException.class, state::close);
        assertEquals(1, gpu.allocations.size());
        state.close();
        assertTrue(gpu.allocations.isEmpty());
        state.close();
        assertThrows(IllegalStateException.class, state::length);
    }

    @Test
    void decodeScratchIsStableAndOwnedAndHugeCapacityFailsBeforeAllocation() {
        RecordingGpu gpu = new RecordingGpu();
        try (AttentionKvState state = new AttentionKvState(gpu, 1024)) {
            assertThrows(ArithmeticException.class, () -> state.prepareAppend(0, Integer.MAX_VALUE));
            assertTrue(gpu.allocations.isEmpty());
            long scratch = state.decodeScratchAddress(24);
            assertEquals(scratch, state.decodeScratchAddress(24));
            assertEquals(24L * 64 * 258 * Float.BYTES, gpu.allocations.get(scratch));
            assertThrows(IllegalArgumentException.class, () -> state.decodeScratchAddress(12));
            assertThrows(IllegalArgumentException.class, () -> state.decodeScratchAddress(0));
        }
        assertTrue(gpu.allocations.isEmpty());
    }

    private static final class RecordingGpu extends ExecutionGpu {
        private long nextAddress = 4096;
        private int failAfter = Integer.MAX_VALUE;
        private boolean failUpload;
        private long failFreeAddress;
        private final Map<Long, Long> allocations = new HashMap<>();
        private final Map<Long, long[]> tableEntries = new HashMap<>();
        private final List<Long> copySizes = new ArrayList<>();
        private int stagings;
        private int stagingReleases;
        private boolean proven = true;

        @Override
        public UploadBuffer allocateUploadBuffer(long bytes) {
            stagings++;
            UploadBuffer allocated = super.allocateUploadBuffer(bytes);
            return new UploadBuffer(allocated.segment(), () -> {
                stagingReleases++;
                allocated.close();
            });
        }

        @Override
        public boolean completionProven() {
            return proven;
        }

        @Override
        public long allocate(long byteSize) {
            if (failAfter-- == 0) throw new IllegalStateException("injected allocation failure");
            long address = nextAddress;
            nextAddress += byteSize + 4096;
            allocations.put(address, byteSize);
            return address;
        }

        @Override
        public void copyHostToDevice(long destination, MemorySegment source, long byteSize) {
            if (failUpload) throw new IllegalStateException("injected upload failure");
            tableEntries.put(destination, source.asSlice(0, byteSize).toArray(java.lang.foreign.ValueLayout.JAVA_LONG));
        }

        @Override
        public void copyDeviceToHost(MemorySegment destination, long source, long byteSize) {}

        @Override
        public void free(long address) {
            if (address == failFreeAddress) {
                failFreeAddress = 0;
                throw new IllegalStateException("injected release failure");
            }
            assertNotNull(allocations.remove(address), "unknown or duplicate free");
        }

        @Override
        public void copyDeviceToDevice(long destination, long source, long byteSize) {
            copySizes.add(byteSize);
        }

        @Override
        public void embedQ3(long a, long b, long c, long d, int e, int f, int g) {}

        @Override
        public void synchronize() {}
    }
}
