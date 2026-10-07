package io.euhedral_execution.inference.core.runtime;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.gpu.GpuStream;
import io.euhedral_execution.inference.core.gpu.InlineGpuStream;
import io.euhedral_execution.inference.core.runtime.EuhedralInferenceRuntime.Lanes;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;

/// The transfer marker orders every weight-staging quantum, at admission and again at retirement on a lattice worker,
/// so reading it must take no lock.
class EuhedralInferenceRuntimeTransferMarkerTest {

    private static final long MARKER = 7;

    @Test
    void theMarkerIsReadWithoutALock() throws Exception {
        var gpu = new MarkerGpu();
        var runtime = new EuhedralInferenceRuntime(lake(), gpu, new Lanes(1, true, false));
        try {
            assertEquals(MARKER, runtime.transferMarker());
            var held = new CountDownLatch(1);
            var release = new CountDownLatch(1);
            for (String lock : List.of("laneLock", "closeLock")) {
                var field = EuhedralInferenceRuntime.class.getDeclaredField(lock);
                field.setAccessible(true);
                Object monitor = field.get(runtime);
                Thread holder = Thread.ofPlatform().start(() -> {
                    synchronized (monitor) {
                        held.countDown();
                        try {
                            release.await();
                        } catch (InterruptedException interrupted) {
                            Thread.currentThread().interrupt();
                        }
                    }
                });
                held.await();
                try {
                    var read = CompletableFuture.supplyAsync(runtime::transferMarker);
                    assertEquals(MARKER, read.get(5, TimeUnit.SECONDS), lock + " blocked the marker");
                } finally {
                    release.countDown();
                    holder.join();
                }
            }
            assertEquals(1, gpu.markersOpened.get(), "the marker opens once");
        } finally {
            runtime.close();
        }
        assertEquals(List.of(MARKER), gpu.markersClosed, "close releases the marker once");
    }

    @Test
    void aRuntimeWithoutATransferLaneOpensNothingWhenAskedForTheMarker() {
        var gpu = new MarkerGpu();
        var runtime = new EuhedralInferenceRuntime(lake(), gpu, Lanes.of(2));
        try {
            assertThrows(IllegalStateException.class, runtime::transferMarker);
            assertEquals(0, gpu.streamsOpened.get(), "asking for a missing marker opened the lanes");
        } finally {
            runtime.close();
        }
    }

    @Test
    void theMarkerOpensWithTheLanes() {
        var gpu = new MarkerGpu();
        var runtime = new EuhedralInferenceRuntime(lake(), gpu, new Lanes(2, true, false));
        try {
            assertEquals(MARKER, runtime.transferMarker());
            assertEquals(3, gpu.streamsOpened.get(), "two compute lanes and the transfer lane");
        } finally {
            runtime.close();
        }
        assertTrue(gpu.markersClosed.contains(MARKER));
    }

    private static io.euhedral_execution.inference.core.runtime.graph.InferenceLake lake() {
        return EuhedralInferenceRuntime.newLake(source -> {});
    }

    private static final class MarkerGpu extends ExecutionGpu {
        final AtomicInteger streamsOpened = new AtomicInteger();
        final AtomicInteger markersOpened = new AtomicInteger();
        final List<Long> markersClosed = new ArrayList<>();

        @Override
        public GpuStream openStream() {
            this.streamsOpened.incrementAndGet();
            return new InlineGpuStream() {
                @Override
                public long openMarker() {
                    MarkerGpu.this.markersOpened.incrementAndGet();
                    return MARKER;
                }

                @Override
                public void closeMarker(long marker) {
                    synchronized (MarkerGpu.this.markersClosed) {
                        MarkerGpu.this.markersClosed.add(marker);
                    }
                }
            };
        }

        @Override
        public long allocate(long byteSize) {
            throw new UnsupportedOperationException();
        }

        @Override
        public void free(long address) {}

        @Override
        public void copyHostToDevice(long destination, MemorySegment source, long byteSize) {}

        @Override
        public void copyDeviceToHost(MemorySegment destination, long source, long byteSize) {}

        @Override
        public void embedQ3(
                long tokenIdsAddress,
                long embeddingAddress,
                long embeddingByteSize,
                long hiddenStateAddress,
                int tokenCount,
                int vocabularySize,
                int hiddenSize) {
            throw new UnsupportedOperationException();
        }

        @Override
        public void synchronize() {}
    }
}
