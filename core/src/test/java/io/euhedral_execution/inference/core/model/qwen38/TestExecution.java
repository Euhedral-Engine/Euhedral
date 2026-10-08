package io.euhedral_execution.inference.core.model.qwen38;

import io.euhedral_execution.inference.core.InferenceConfig;
import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.runtime.EuhedralInferenceRuntime;
import io.euhedral_execution.inference.core.runtime.PullingLattice;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;

/// Runs one quantum through a runtime driven by a [PullingLattice], with a workspace that holds it, then detaches it.
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
            // The workspace is sized at load: a test's one quantum may be wider than a production chunk.
            var runtime = new Execution(
                    lattice,
                    plan,
                    gpu,
                    EuhedralInferenceRuntime.laneCount(),
                    EuhedralInferenceRuntime.CAPTURE_GRAPHS,
                    Math.max(InferenceConfig.PREFILL_CHUNK_TOKENS, context.inputTokenCount()));
            try {
                return runtime.submit(context, terminalConsumer).get(timeoutSeconds, TimeUnit.SECONDS);
            } finally {
                runtime.close();
            }
        }
    }
}
