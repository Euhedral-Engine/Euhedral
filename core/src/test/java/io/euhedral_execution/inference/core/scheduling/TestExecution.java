package io.euhedral_execution.inference.core.scheduling;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;

/// Runs one quantum through a runtime driven by a [PullingLattice], then detaches it.
public final class TestExecution {

    private TestExecution() {}

    public static QwenExecutionContext.Outcome run(
            QwenExecutionPlan plan,
            ExecutionGpu gpu,
            QwenExecutionContext context,
            Consumer<? super QwenExecutionContext> terminalConsumer,
            long timeoutSeconds)
            throws Exception {
        try (var lattice = new PullingLattice()) {
            var runtime = new EuhedralInferenceRuntime(lattice, plan, gpu);
            try {
                return runtime.submit(context, terminalConsumer).get(timeoutSeconds, TimeUnit.SECONDS);
            } finally {
                runtime.close();
            }
        }
    }
}
