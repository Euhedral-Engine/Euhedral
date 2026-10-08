package io.euhedral_execution.inference.core.model.qwen38;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.runtime.graph.GraphShape;
import io.euhedral_execution.inference.core.runtime.graph.GraphStorage;
import io.euhedral_execution.inference.core.runtime.graph.ShapeBuilder;
import io.euhedral_execution.inference.core.runtime.graph.StageFrame;
import io.euhedral_execution.inference.core.runtime.graph.StageGraph;
import io.euhedral_execution.inference.core.runtime.graph.StageTopology;
import java.util.Objects;
import java.util.TreeSet;

/// A prompt chunk that also takes a prefix checkpoint: the chunk's view, then one stage after its last stages that
/// copies the sequence's state at the chunk's end into the checkpoint's extent ([Stages.Checkpoint]). The stage
/// touches every layer's carried state, so the next chunk's updates of it follow the copy; it touches no workspace
/// buffer. Only a prompt graph uses it ([PromptShapes]).
final class CheckpointView implements GraphShape {

    private final Shape view;
    private final int checkpoint;
    private final StageTopology topology;
    private final int[] carried;

    CheckpointView(Shape view) {
        this.view = Objects.requireNonNull(view, "view");
        StageTopology base = view.topology();
        int size = base.size();
        ShapeBuilder<Integer> builder = new ShapeBuilder<>();
        for (int stage = 0; stage <= size; stage++) builder.stage(stage);
        for (int stage = 0; stage < size; stage++) {
            for (int next : base.submittedSuccessors(stage)) builder.submitted(stage, next);
            for (int next : base.retiredSuccessors(stage)) builder.retired(stage, next);
        }
        // After every stage that ends the chunk: every carried update is one of their ancestors.
        for (int stage = 0; stage < size; stage++)
            if (base.submittedSuccessors(stage).length == 0 && base.retiredSuccessors(stage).length == 0)
                builder.submitted(stage, size);
        this.checkpoint = size;
        this.topology = builder.build();
        TreeSet<Integer> keys = new TreeSet<>();
        for (int stage = 0; stage < size; stage++) for (int key : view.carriedState(stage)) keys.add(key);
        this.carried = keys.stream().mapToInt(Integer::intValue).toArray();
    }

    Shape view() {
        return this.view;
    }

    /// Whether `stage` is the checkpoint's copy.
    boolean isCheckpoint(int stage) {
        return stage == this.checkpoint;
    }

    @Override
    public StageTopology topology() {
        return this.topology;
    }

    @Override
    public StageFrame createStage(StageGraph graph, int stage, ExecutionGpu gpu) {
        throw new UnsupportedOperationException("a checkpoint view is a template of a prompt graph");
    }

    @Override
    public GraphStorage newStorage(ExecutionGpu gpu) {
        return this.view.newStorage(gpu);
    }

    @Override
    public int[] workspaceBuffers(int stage) {
        return stage == this.checkpoint ? NO_BUFFERS : this.view.workspaceBuffers(stage);
    }

    @Override
    public int workspaceBufferCount() {
        return this.view.workspaceBufferCount();
    }

    @Override
    public int[] carriedState(int stage) {
        return stage == this.checkpoint ? this.carried : this.view.carriedState(stage);
    }

    @Override
    public int carriedStateCount() {
        return this.view.carriedStateCount();
    }
}
