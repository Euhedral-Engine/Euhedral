package io.euhedral_execution.inference.core.model.qwen38;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.runtime.PullingLattice;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;

/// Runs one quantum through a runtime driven by a [PullingLattice], then detaches it.
public final class TestExecution {

    private TestExecution() {}

    public static Quantum.Outcome run(
            ExecutionPlan plan,
            ExecutionGpu gpu,
            Quantum context,
            Consumer<? super Quantum> terminalConsumer,
            long timeoutSeconds)
            throws Exception {
        try (var lattice = new PullingLattice()) {
            var runtime = new Execution(lattice, plan, gpu);
            try {
                return runtime.submit(context, terminalConsumer).get(timeoutSeconds, TimeUnit.SECONDS);
            } finally {
                runtime.close();
            }
        }
    }
}
