package io.euhedral_execution.inference.core.model.qwen38;

import io.euhedral_execution.inference.core.artifact.TensorHandle;
import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.runtime.graph.StageGraph;

/// Copies a host-backed weight into its staging slot on the pool's transfer lane, so the copy engine
/// overlaps the compute lanes. Its consumer awaits it like any cross-lane predecessor.
public final class WeightTransferFrame extends InstructionFrame {
    private final long source;
    private final long destination;
    private final long byteSize;

    WeightTransferFrame(StageGraph graph, ExecutionPlan.Instruction instruction, ExecutionGpu gpu) {
        super(graph, instruction, gpu);
        TensorHandle weight = instruction.weight();
        this.source = weight.hostAddress();
        this.destination = weight.deviceAddress();
        this.byteSize = weight.byteSize();
    }

    @Override
    protected boolean transfers() {
        return true;
    }

    @Override
    protected void perform(Quantum context, ExecutionPlan.Instruction instruction) {
        gpu().copyHostWeightsToDevice(this.destination, this.source, this.byteSize);
    }
}
