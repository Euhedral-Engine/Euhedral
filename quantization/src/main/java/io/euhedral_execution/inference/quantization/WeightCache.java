package io.euhedral_execution.inference.quantization;

import io.euhedral_execution.inference.core.artifact.TensorDescriptor;
import io.euhedral_execution.inference.core.artifact.TensorHandle;
import io.euhedral_execution.inference.quantization.mixed.Ensemble;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Path;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/// The device-resident objects of the candidates being scored together, shared: each distinct object (the file it is
/// read from, its name and its offset there) is loaded once and held for as long as any candidate using it runs.
/// A candidate [#acquire]s every object it needs and [#release]s them when it ends; the one that takes the last
/// reference frees the object. Loading and freeing are frames of their own on whatever [Ensemble.Frames] the cache is
/// given, and the counts change through the map's atomic updates, so nothing waits on a lock.
final class WeightCache {

    /// The identity of one object: the same file, name and offset are the same bytes.
    record Key(Path file, String name, long offset) {
        static Key of(Path file, TensorDescriptor descriptor) {
            return new Key(file, descriptor.name(), descriptor.dataOffset());
        }
    }

    /// Where objects come from and go: the device in production, a counting fake in tests.
    interface Backend {
        TensorHandle load(Path file, TensorDescriptor descriptor) throws IOException;

        void free(TensorHandle handle);
    }

    private record Slot(CompletableFuture<TensorHandle> handle, int references, long bytes) {}

    private final Backend backend;
    private final ConcurrentHashMap<Key, Slot> slots = new ConcurrentHashMap<>();
    private final AtomicLong resident = new AtomicLong();
    private final AtomicInteger loaded = new AtomicInteger();
    private final AtomicInteger freesInFlight = new AtomicInteger();

    WeightCache(Backend backend) {
        this.backend = backend;
    }

    /// The object's handle, once it is on the device: loaded by a frame if no candidate holds it yet, otherwise the
    /// one already there. The caller holds a reference until [#release].
    CompletableFuture<TensorHandle> acquire(Key key, TensorDescriptor descriptor, Ensemble.Frames frames) {
        AtomicBoolean created = new AtomicBoolean();
        Slot slot = slots.compute(key, (k, old) -> {
            if (old != null) return new Slot(old.handle(), old.references() + 1, old.bytes());
            created.set(true);
            return new Slot(new CompletableFuture<>(), 1, descriptor.byteSize());
        });
        if (created.get()) {
            resident.addAndGet(slot.bytes());
            frames.submit(() -> {
                        try {
                            return backend.load(key.file(), descriptor);
                        } catch (IOException failure) {
                            throw new UncheckedIOException(failure);
                        }
                    })
                    .whenComplete((handle, failure) -> {
                        if (failure != null) slot.handle().completeExceptionally(failure);
                        else {
                            loaded.incrementAndGet();
                            slot.handle().complete(handle);
                        }
                    });
        }
        return slot.handle();
    }

    /// Gives up one reference to `key`. The last one frees the object, in a frame of its own; the future completes
    /// once the device memory is free (at once if the object failed to load or is still shared).
    CompletableFuture<Void> release(Key key, Ensemble.Frames frames) {
        Slot[] last = new Slot[1];
        slots.compute(key, (k, old) -> {
            if (old == null) throw new IllegalStateException("no reference to " + key);
            if (old.references() > 1) return new Slot(old.handle(), old.references() - 1, old.bytes());
            last[0] = old;
            return null;
        });
        if (last[0] == null) return CompletableFuture.completedFuture(null);
        resident.addAndGet(-last[0].bytes());
        freesInFlight.incrementAndGet();
        CompletableFuture<Void> freed = new CompletableFuture<>();
        last[0].handle().whenComplete((handle, failure) -> {
            if (failure != null) {
                freesInFlight.decrementAndGet();
                freed.complete(null);
                return;
            }
            frames.submit(() -> {
                        backend.free(handle);
                        return null;
                    })
                    .whenComplete((ignored, freeFailure) -> {
                        freesInFlight.decrementAndGet();
                        if (freeFailure != null) freed.completeExceptionally(freeFailure);
                        else freed.complete(null);
                    });
        });
        return freed;
    }

    /// The loaded handle of `key`; the caller holds a reference and has seen [#acquire]'s future complete.
    TensorHandle handle(Key key) {
        Slot slot = slots.get(key);
        if (slot == null) throw new IllegalStateException(key + " is not held");
        return slot.handle().join();
    }

    boolean contains(Key key) {
        return slots.containsKey(key);
    }

    /// Bytes of the objects held, those still loading included.
    long residentBytes() {
        return resident.get();
    }

    /// Objects loaded so far.
    int loads() {
        return loaded.get();
    }

    /// Objects being freed: their device memory is not back yet.
    int freesInFlight() {
        return freesInFlight.get();
    }

    int size() {
        return slots.size();
    }
}
