package io.euhedral_execution.inference.core.model.qwen4;

import io.euhedral_execution.inference.core.gpu.GpuStream;
import java.util.concurrent.CountDownLatch;

/// Host waits on a stream, for the few points of Flash-Next execution where the host needs a device result (the
/// router's choice of experts, the logits of a step) before it can go on.
public final class Streams {

    private Streams() {}

    /// Blocks until every operation submitted to `stream` so far has completed. Uses the stream's retirement
    /// boundary, so the proof of completion is the same one every other consumer of the stream relies on.
    ///
    /// @throws IllegalStateException when the device reports a failure of the work before the boundary
    public static void awaitCompletion(GpuStream stream) throws InterruptedException {
        CountDownLatch retired = new CountDownLatch(1);
        long ticket = stream.notifyRetired((t, driverThread) -> retired.countDown());
        retired.await();
        Throwable failure = stream.confirmRetired(ticket);
        if (failure != null) throw new IllegalStateException("device work failed", failure);
    }
}
