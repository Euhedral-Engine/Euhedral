package io.euhedral_execution.inference.quantization;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.artifact.TensorDataType;
import io.euhedral_execution.inference.core.artifact.TensorDescriptor;
import io.euhedral_execution.inference.core.artifact.TensorHandle;
import io.euhedral_execution.inference.core.artifact.WeightFormat;
import io.euhedral_execution.inference.core.artifact.WeightLayout;
import io.euhedral_execution.inference.quantization.mixed.Ensemble;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.function.Supplier;
import org.junit.jupiter.api.Test;

/// Objects shared by candidates are loaded once and freed with the last candidate that uses them.
class WeightCacheTest {

    private static final Path BASE = Path.of("base.edrl"), OTHER = Path.of("other.edrl");

    /// A device that hands out addresses and remembers which are live; frames run where they are submitted.
    private static final class FakeDevice implements WeightCache.Backend, Ensemble.Frames {
        long next = 4096;
        final List<String> loads = new ArrayList<>();
        final Set<Long> live = new HashSet<>();
        String failOn;

        @Override
        public TensorHandle load(Path file, TensorDescriptor descriptor) throws IOException {
            if (descriptor.name().equals(failOn)) throw new IOException("unreadable " + descriptor.name());
            loads.add(file + ":" + descriptor.name());
            long address = next;
            next += 4096;
            live.add(address);
            return new TensorHandle(
                    descriptor.name(),
                    descriptor.shape(),
                    descriptor.dataType(),
                    descriptor.format(),
                    descriptor.layout(),
                    address,
                    descriptor.byteSize());
        }

        @Override
        public void free(TensorHandle handle) {
            if (!live.remove(handle.deviceAddress())) throw new IllegalStateException("double free");
        }

        @Override
        public <T> CompletableFuture<T> submit(Supplier<T> task) {
            try {
                return CompletableFuture.completedFuture(task.get());
            } catch (RuntimeException failure) {
                return CompletableFuture.failedFuture(failure);
            }
        }
    }

    private static TensorDescriptor descriptor(String name, long offset, long bytes) {
        return new TensorDescriptor(
                name,
                new long[] {bytes / 2},
                TensorDataType.BF16,
                WeightFormat.BF16,
                WeightLayout.CONTIGUOUS_LE_V1,
                offset,
                bytes);
    }

    private static WeightCache.Key key(Path file, TensorDescriptor d) {
        return WeightCache.Key.of(file, d);
    }

    @Test
    void sharedObjectsAreLoadedOnceAndFreedWithTheLastCandidateThatUsesThem() {
        FakeDevice device = new FakeDevice();
        WeightCache cache = new WeightCache(device);
        TensorDescriptor norm = descriptor("text/final_norm", 0, 64);
        TensorDescriptor up = descriptor("text/layers/0/mlp/up", 64, 128);
        TensorDescriptor down = descriptor("text/layers/0/mlp/down", 192, 256);
        // Candidate A reads up and down from the base, B reads down from the other file: norm is shared by both,
        // and so is up; down differs.
        TensorHandle normA = cache.acquire(key(BASE, norm), norm, device).join();
        TensorHandle upA = cache.acquire(key(BASE, up), up, device).join();
        TensorHandle downA = cache.acquire(key(BASE, down), down, device).join();
        TensorHandle normB = cache.acquire(key(BASE, norm), norm, device).join();
        TensorHandle upB = cache.acquire(key(BASE, up), up, device).join();
        TensorHandle downB = cache.acquire(key(OTHER, down), down, device).join();

        assertEquals(4, device.loads.size(), "norm and up once, down once per source: " + device.loads);
        assertSame(normA, normB);
        assertSame(upA, upB);
        assertFalse(downA.deviceAddress() == downB.deviceAddress());
        assertEquals(64 + 128 + 256 + 256, cache.residentBytes());

        // A ends: only its own down goes.
        cache.release(key(BASE, norm), device).join();
        cache.release(key(BASE, up), device).join();
        cache.release(key(BASE, down), device).join();
        assertEquals(3, device.live.size());
        assertTrue(device.live.contains(normB.deviceAddress()));
        assertTrue(device.live.contains(upB.deviceAddress()));
        assertTrue(device.live.contains(downB.deviceAddress()));
        assertFalse(device.live.contains(downA.deviceAddress()));
        assertFalse(cache.contains(key(BASE, down)));
        assertEquals(64 + 128 + 256, cache.residentBytes());

        // B ends: nothing is left.
        cache.release(key(BASE, norm), device).join();
        cache.release(key(BASE, up), device).join();
        cache.release(key(OTHER, down), device).join();
        assertTrue(device.live.isEmpty());
        assertEquals(0, cache.size());
        assertEquals(0, cache.residentBytes());
        assertEquals(0, cache.freesInFlight());
    }

    @Test
    void anObjectAcquiredAfterItsLastReferenceWentIsLoadedAgain() {
        FakeDevice device = new FakeDevice();
        WeightCache cache = new WeightCache(device);
        TensorDescriptor norm = descriptor("text/final_norm", 0, 64);
        cache.acquire(key(BASE, norm), norm, device).join();
        cache.release(key(BASE, norm), device).join();
        cache.acquire(key(BASE, norm), norm, device).join();
        assertEquals(2, device.loads.size());
        assertEquals(1, device.live.size());
    }

    @Test
    void aFailedLoadFailsEveryHolderAndFreesNothingElseTwice() {
        FakeDevice device = new FakeDevice();
        device.failOn = "text/final_norm";
        WeightCache cache = new WeightCache(device);
        TensorDescriptor norm = descriptor("text/final_norm", 0, 64);
        TensorDescriptor up = descriptor("text/layers/0/mlp/up", 64, 128);
        CompletableFuture<TensorHandle> first = cache.acquire(key(BASE, norm), norm, device);
        CompletableFuture<TensorHandle> second = cache.acquire(key(BASE, norm), norm, device);
        cache.acquire(key(BASE, up), up, device).join();
        assertThrows(CompletionException.class, first::join);
        assertThrows(CompletionException.class, second::join);
        cache.release(key(BASE, norm), device).join();
        cache.release(key(BASE, norm), device).join();
        cache.release(key(BASE, up), device).join();
        assertTrue(device.live.isEmpty());
        assertEquals(0, cache.size());
    }

    @Test
    void releasingWhatIsNotHeldIsAnError() {
        WeightCache cache = new WeightCache(new FakeDevice());
        assertThrows(
                IllegalStateException.class,
                () -> cache.release(key(BASE, descriptor("text/final_norm", 0, 64)), new FakeDevice()));
    }
}
