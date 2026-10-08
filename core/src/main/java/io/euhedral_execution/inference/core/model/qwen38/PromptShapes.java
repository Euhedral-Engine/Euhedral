package io.euhedral_execution.inference.core.model.qwen38;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.runtime.EuhedralInferenceRuntime;
import io.euhedral_execution.inference.core.runtime.graph.ChunkedShape;
import io.euhedral_execution.inference.core.runtime.graph.GraphShape;
import io.euhedral_execution.inference.core.runtime.graph.GraphStorage;
import io.euhedral_execution.inference.core.runtime.graph.StageFrame;
import io.euhedral_execution.inference.core.runtime.graph.StageGraph;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/// The prompt graphs' shapes: a prompt quantum runs a [ChunkedShape] of the view its full chunks select and the view
/// its last chunk selects. Shapes are kept per (full view, last view, chunk count), the most recently used
/// [#CAPACITY] of them; an evicted shape's pool is released to the runtime, which closes its graphs when they are
/// idle.
///
/// Confined to the workspace's owner: admission calls it, and admission runs on frames ordered on the owner's hash.
final class PromptShapes {

    /// Prompt shapes kept at once.
    static final int CAPACITY = 16;

    private record Key(Shape full, Shape last, int chunks, List<Integer> checkpointChunks) {}

    private final EuhedralInferenceRuntime runtime;
    /// The checkpointing variant of each view, built once.
    private final Map<Shape, CheckpointView> checkpointViews = new java.util.HashMap<>();
    private final Map<Key, ChunkedShape> shapes = new LinkedHashMap<>(CAPACITY, 0.75f, true) {
        @Override
        protected boolean removeEldestEntry(Map.Entry<Key, ChunkedShape> eldest) {
            if (size() <= CAPACITY) return false;
            PromptShapes.this.runtime.release(eldest.getValue());
            return true;
        }
    };

    PromptShapes(EuhedralInferenceRuntime runtime) {
        this.runtime = Objects.requireNonNull(runtime, "runtime");
    }

    /// The shape a prompt quantum of several chunks runs.
    ChunkedShape shape(Quantum prompt) {
        int chunks = prompt.chunkCount();
        Shape full = prompt.shape();
        Shape last = prompt.plan()
                .forExecution(
                        Quantum.ExecutionKind.PREFILL, prompt.chunk(chunks - 1).rows());
        List<Integer> checkpoints = prompt.checkpointChunks();
        return this.shapes.computeIfAbsent(new Key(full, last, chunks, checkpoints), key -> {
            GraphShape[] templates = new GraphShape[chunks];
            for (int chunk = 0; chunk < chunks; chunk++) {
                Shape view = chunk == chunks - 1 ? last : full;
                templates[chunk] = checkpoints.contains(chunk)
                        ? this.checkpointViews.computeIfAbsent(view, CheckpointView::new)
                        : view;
            }
            return new ChunkedShape(templates, new Frames(templates));
        });
    }

    /// Shapes kept now (tests).
    int size() {
        return this.shapes.size();
    }

    /// The frames of a prompt graph: each stage is its template's, told its chunk.
    private record Frames(GraphShape[] templates) implements ChunkedShape.Chunks {
        @Override
        public StageFrame create(StageGraph graph, int stage, int chunk, int templateStage, ExecutionGpu gpu) {
            Shape view;
            if (this.templates[chunk] instanceof CheckpointView checkpointed) {
                if (checkpointed.isCheckpoint(templateStage)) return new Stages.Checkpoint(graph, stage, chunk, gpu);
                view = checkpointed.view();
            } else view = (Shape) this.templates[chunk];
            Stages.Stage frame = Stages.create(graph, stage, view.instructions().get(templateStage), gpu, chunk);
            frame.scratchUse = view.scratchUse(templateStage);
            return frame;
        }

        @Override
        public GraphStorage newStorage(ExecutionGpu gpu) {
            return new WorkspaceStorage(gpu);
        }
    }
}
