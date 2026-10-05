package io.euhedral_execution.inference.core.model_loader.qwen4;

import io.euhedral_execution.inference.core.gpu.GpuMemory;
import io.euhedral_execution.inference.core.model_loader.QwenWeightLoadException;
import io.euhedral_execution.inference.core.model_loader.TensorLoader;
import io.euhedral_execution.inference.core.model_loader.WeightStaging;
import io.euhedral_execution.inference.core.model_loader.layer_weights.TensorHandle;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/// Loads the fixed objects of a plan: device-resident tensors onto the device, staged and mapped ones into one pinned
/// huge-page arena (allocated before any payload is read, so the pages are likelier to be huge), and the device ring
/// that staged objects pass through. Payloads are read and checked concurrently.
final class Qwen4FixedLoader {

    private Qwen4FixedLoader() {}

    /// What a load holds, to be released in the reverse order of creation.
    record Loaded(
            Map<String, TensorHandle> handles,
            long hostArena,
            long hostArenaBytes,
            WeightStaging staging,
            long residentBytes) {}

    static Loaded load(Path path, Qwen4Artifact artifact, Qwen4ResidencyPlan plan, GpuMemory gpu, int threads)
            throws IOException {
        List<Qwen4Tensor> resident = new ArrayList<>();
        List<Qwen4Tensor> hosted = new ArrayList<>();
        long largestStaged = 0;
        for (Qwen4Tensor tensor : artifact.tensors()) {
            StorageClass storage = plan.storageOf(tensor.name());
            switch (storage) {
                case DEVICE_RESIDENT -> resident.add(tensor);
                case HOST_STAGED -> {
                    if (tensor.group() == ComponentGroup.NGRAM) break; // the n-gram store owns its tables
                    hosted.add(tensor);
                    largestStaged = Math.max(largestStaged, tensor.byteSize());
                }
                case HOST_MAPPED -> hosted.add(tensor);
                case DEFERRED, DEVICE_CACHED -> {}
            }
        }
        long arenaBytes = 0;
        long[] hostOffsets = new long[hosted.size()];
        for (int i = 0; i < hostOffsets.length; i++) {
            hostOffsets[i] = arenaBytes;
            arenaBytes += alignUp(hosted.get(i).byteSize(), 4096);
        }

        Map<String, TensorHandle> handles = new LinkedHashMap<>();
        long arena = 0;
        long ring = 0;
        ExecutorService pool = Executors.newFixedThreadPool(Math.max(1, threads));
        try {
            if (arenaBytes > 0) arena = gpu.allocateHostWeights(arenaBytes);
            final long arenaBase = arena;
            List<Future<TensorHandle>> futures = new ArrayList<>();
            for (Qwen4Tensor tensor : resident)
                futures.add(pool.submit(() -> TensorLoader.load(path, tensor.descriptor(), gpu)));
            for (int i = 0; i < hosted.size(); i++) {
                final Qwen4Tensor tensor = hosted.get(i);
                final long address = arenaBase + hostOffsets[i];
                final boolean mapped = plan.storageOf(tensor.name()) == StorageClass.HOST_MAPPED;
                futures.add(pool.submit(() -> {
                    TensorHandle loaded = TensorLoader.loadToHost(path, tensor.descriptor(), address);
                    if (!mapped) return loaded;
                    return new TensorHandle(
                            loaded.name(),
                            loaded.shape(),
                            loaded.dataType(),
                            loaded.format(),
                            loaded.layout(),
                            gpu.hostWeightsDeviceAddress(address),
                            loaded.byteSize(),
                            address,
                            true);
                }));
            }
            IOException failure = null;
            List<TensorHandle> done = new ArrayList<>();
            for (Future<TensorHandle> future : futures) {
                try {
                    done.add(future.get());
                } catch (ExecutionException exception) {
                    done.add(null);
                    Throwable cause = exception.getCause();
                    if (failure == null)
                        failure = cause instanceof IOException io
                                ? io
                                : new IOException("loading a fixed object failed", cause);
                    else failure.addSuppressed(cause);
                } catch (InterruptedException interrupted) {
                    Thread.currentThread().interrupt();
                    done.add(null);
                    if (failure == null) failure = new IOException("loading was interrupted", interrupted);
                }
            }
            for (TensorHandle handle : done) if (handle != null) handles.put(handle.name(), handle);
            if (failure != null) throw failure;
            // keep plan order
            Map<String, TensorHandle> ordered = new LinkedHashMap<>();
            for (Qwen4Tensor tensor : artifact.tensors())
                if (handles.containsKey(tensor.name())) ordered.put(tensor.name(), handles.get(tensor.name()));
            handles = ordered;

            WeightStaging staging = null;
            if (largestStaged > 0) {
                long slot = WeightStaging.slotBytesFor(largestStaged);
                ring = gpu.allocate(Math.multiplyExact(slot, Qwen4ResidencyPlanner.STAGING_SLOTS));
                staging = new WeightStaging(ring, slot, Qwen4ResidencyPlanner.STAGING_SLOTS);
            }
            long residentBytes = 0;
            for (TensorHandle handle : handles.values())
                if (!handle.hostBacked() && handle.deviceAddress() != 0 && !handle.hostMapped())
                    residentBytes += handle.byteSize();
            return new Loaded(handles, arena, arenaBytes, staging, residentBytes);
        } catch (Throwable failure) {
            release(handles, arena, ring, gpu, failure);
            if (failure instanceof IOException io) throw io;
            if (failure instanceof RuntimeException runtime) throw runtime;
            if (failure instanceof Error error) throw error;
            throw new QwenWeightLoadException(String.valueOf(failure));
        } finally {
            pool.shutdownNow();
        }
    }

    /// Frees device tensors, the staging ring and the host arena.
    static void release(Map<String, TensorHandle> handles, long arena, long ring, GpuMemory gpu, Throwable failure) {
        for (TensorHandle handle : handles.values()) {
            if (handle.deviceAddress() != 0 && !handle.hostMapped() && !handle.hostBacked()) {
                try {
                    gpu.free(handle.deviceAddress());
                } catch (RuntimeException cleanup) {
                    if (failure != null) failure.addSuppressed(cleanup);
                }
            }
        }
        if (ring != 0) {
            try {
                gpu.free(ring);
            } catch (RuntimeException cleanup) {
                if (failure != null) failure.addSuppressed(cleanup);
            }
        }
        if (arena != 0) {
            try {
                gpu.freeHostWeights(arena);
            } catch (RuntimeException cleanup) {
                if (failure != null) failure.addSuppressed(cleanup);
            }
        }
    }

    private static long alignUp(long value, long alignment) {
        return (value + alignment - 1) / alignment * alignment;
    }
}
