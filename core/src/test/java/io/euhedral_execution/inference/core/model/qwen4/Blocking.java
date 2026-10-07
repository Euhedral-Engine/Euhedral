package io.euhedral_execution.inference.core.model.qwen4;

import io.euhedral_execution.inference.core.runtime.graph.AbstractQuantum;
import io.euhedral_execution.inference.core.runtime.graph.FutureContinuation;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;

/// Blocking forms of the executor's asynchronous steps, for tests whose calling thread is not a
/// lattice worker: each starts the step and waits for its listener.
public final class Blocking {

    private Blocking() {}

    /// Runs one step and returns after it retired and committed.
    public static void step(
            ExecutionPlan executor,
            Sequence sequence,
            int[] tokens,
            int offset,
            int rows,
            ExecutionPlan.LogitsSink sink)
            throws InterruptedException {
        var done = new FutureContinuation<Throwable>(AbstractQuantum::terminalFailure);
        executor.start(sequence, tokens, offset, rows, sink, done);
        await(done.future());
    }

    /// Runs one layer of a chunk on a residual state the caller supplies and returns the state
    /// after it.
    static short[] runSingleLayer(
            ExecutionPlan plan, Sequence sequence, int layer, int[] tokens, int offset, int rows, short[] stateIn)
            throws InterruptedException {
        short[] out = new short[stateIn.length];
        long bytes = (long) stateIn.length * Short.BYTES;
        var gpu = plan.gpu();
        try (var upload = gpu.allocateUploadBuffer(bytes)) {
            MemorySegment.copy(stateIn, 0, upload.segment(), ValueLayout.JAVA_SHORT, 0, stateIn.length);
            try (Arena arena = Arena.ofShared()) {
                var back = arena.allocate(bytes, 16);
                var done = new FutureContinuation<Throwable>(AbstractQuantum::terminalFailure);
                plan.startLayer(
                        sequence,
                        layer,
                        tokens,
                        offset,
                        rows,
                        new ExecutionPlan.StateExchange() {
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
                        done);
                await(done.future());
                MemorySegment.copy(back, ValueLayout.JAVA_SHORT, 0, out, 0, out.length);
            }
        }
        return out;
    }

    /// Runs one MoE block through the layer's pieces on the calling thread (not a worker) and
    /// returns once everything is submitted (the stream is not waited for): the router and its
    /// route copy, a boundary behind them, the shared expert, the block's plan, every expert (leased
    /// through the test support's blocking acquire), and the ordered combine.
    public static void runMoe(
            MoeLayer moe,
            io.euhedral_execution.inference.core.gpu.GpuStream stream,
            io.euhedral_execution.inference.core.model.qwen4.expert.ExpertCache cache,
            MoeLayer.Weights weights,
            int bank,
            long input,
            int rows,
            MoeLayer.Scratch scratch,
            long output)
            throws InterruptedException {
        stream.submit(() -> moe.submitRouting(weights, input, rows, scratch), false);
        java.util.concurrent.CountDownLatch routed = new java.util.concurrent.CountDownLatch(1);
        long ticket = stream.notifyRetired((retired, driverThread) -> routed.countDown());
        stream.submit(() -> moe.submitShared(weights, input, rows, scratch), false);
        routed.await();
        Throwable device = stream.confirmRetired(ticket);
        if (device != null) throw new IllegalStateException("device work failed", device);
        int experts = moe.plan(bank, rows);
        stream.submit(moe::submitPlan, false);
        for (int index = 0; index < experts; index++) {
            moe.hold(
                    index,
                    io.euhedral_execution.inference.core.model.qwen4.expert.ExpertTestSupport.acquire(
                            cache, bank, moe.activeExpert(index)));
            int expert = index;
            stream.submit(() -> moe.submitExpert(expert, stream, 0, input, scratch), false);
        }
        stream.submit(() -> moe.submitFinish(output, rows, scratch), false);
    }

    /// Waits for a step's continuation; a step that failed throws its failure.
    private static void await(CompletableFuture<Throwable> done) throws InterruptedException {
        Throwable failed;
        try {
            failed = done.get();
        } catch (ExecutionException failure) {
            Throwable cause = failure.getCause();
            if (cause instanceof RuntimeException runtime) throw runtime;
            if (cause instanceof Error error) throw error;
            throw new IllegalStateException(cause);
        }
        if (failed instanceof RuntimeException runtime) throw runtime;
        if (failed instanceof Error error) throw error;
        if (failed != null) throw new IllegalStateException(failed);
    }
}
