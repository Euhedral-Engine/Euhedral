package io.euhedral_execution.inference.core.qwen4;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;

/// Blocking forms of the executor's asynchronous steps, for tests whose calling thread is not a lattice
/// worker: each starts the step and waits for its listener.
public final class Qwen4Blocking {

    private Qwen4Blocking() {}

    /// Runs one step and returns after it retired and committed.
    public static void step(
            Qwen4ExecutionPlan executor,
            Qwen4Sequence sequence,
            int[] tokens,
            int offset,
            int rows,
            Qwen4ExecutionPlan.LogitsSink sink)
            throws InterruptedException {
        CompletableFuture<Void> done = new CompletableFuture<>();
        executor.start(sequence, tokens, offset, rows, sink, failure -> {
            if (failure == null) done.complete(null);
            else done.completeExceptionally(failure);
        });
        await(done);
    }

    /// Runs one layer of a chunk on a residual state the caller supplies and returns the state after it.
    static short[] runSingleLayer(
            Qwen4ExecutionPlan plan,
            Qwen4Sequence sequence,
            int layer,
            int[] tokens,
            int offset,
            int rows,
            short[] stateIn)
            throws InterruptedException {
        short[] out = new short[stateIn.length];
        long bytes = (long) stateIn.length * Short.BYTES;
        var gpu = plan.gpu();
        try (var upload = gpu.allocateUploadBuffer(bytes)) {
            MemorySegment.copy(stateIn, 0, upload.segment(), ValueLayout.JAVA_SHORT, 0, stateIn.length);
            try (Arena arena = Arena.ofShared()) {
                var back = arena.allocate(bytes, 16);
                CompletableFuture<Void> done = new CompletableFuture<>();
                plan.startLayer(
                        sequence,
                        layer,
                        tokens,
                        offset,
                        rows,
                        new Qwen4ExecutionPlan.StateExchange() {
                            @Override
                            public void provide(io.euhedral_execution.inference.core.gpu.ExecutionGpu g, long state) {
                                // Queued on the quantum's home lane, ahead of every stage.
                                g.copyUploadToDevice(state, upload);
                            }

                            @Override
                            public void collect(io.euhedral_execution.inference.core.gpu.ExecutionGpu g, long state) {
                                g.copyDeviceToHost(back, state, bytes);
                            }
                        },
                        failure -> {
                            if (failure == null) done.complete(null);
                            else done.completeExceptionally(failure);
                        });
                await(done);
                MemorySegment.copy(back, ValueLayout.JAVA_SHORT, 0, out, 0, out.length);
            }
        }
        return out;
    }

    /// Runs one MoE block through the layer's pieces on the calling thread (not a worker) and returns once
    /// every wave is submitted (the stream is not waited for): the router and its route copy, a boundary
    /// behind them, the shared expert, the planned waves whose experts are leased through the cache's
    /// blocking acquire, and the combination.
    public static void runMoe(
            Qwen4MoeLayer moe,
            io.euhedral_execution.inference.core.gpu.GpuStream stream,
            io.euhedral_execution.inference.core.model_loader.qwen4.expert.ExpertCache cache,
            Qwen4MoeLayer.Weights weights,
            int bank,
            long input,
            int rows,
            Qwen4MoeLayer.Scratch scratch,
            long output)
            throws InterruptedException {
        stream.submit(() -> moe.submitRouting(weights, input, rows, scratch), false);
        java.util.concurrent.CountDownLatch routed = new java.util.concurrent.CountDownLatch(1);
        long ticket = stream.notifyRetired((retired, driverThread) -> routed.countDown());
        stream.submit(() -> moe.submitShared(weights, input, rows, scratch), false);
        routed.await();
        Throwable device = stream.confirmRetired(ticket);
        if (device != null) throw new IllegalStateException("device work failed", device);
        int waves = moe.plan(bank, rows);
        for (int w = 0; w < waves; w++) {
            for (int position = moe.waveStart(w); position < moe.waveStart(w + 1); position++)
                moe.hold(position, cache.acquire(bank, moe.expertAt(position)));
            int wave = w;
            stream.submit(() -> moe.submitWave(wave, stream, input, scratch), false);
        }
        stream.submit(() -> moe.submitFinish(output, rows, scratch), false);
    }

    private static void await(CompletableFuture<Void> done) throws InterruptedException {
        try {
            done.get();
        } catch (ExecutionException failure) {
            Throwable cause = failure.getCause();
            if (cause instanceof RuntimeException runtime) throw runtime;
            if (cause instanceof Error error) throw error;
            throw new IllegalStateException(cause);
        }
    }
}
