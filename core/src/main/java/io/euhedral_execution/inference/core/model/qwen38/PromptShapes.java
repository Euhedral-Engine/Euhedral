package io.euhedral_execution.inference.core.model.qwen38;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.runtime.EuhedralInferenceRuntime;
import io.euhedral_execution.inference.core.runtime.graph.ChunkedShape;
import io.euhedral_execution.inference.core.runtime.graph.GraphStorage;
import io.euhedral_execution.inference.core.runtime.graph.StageFrame;
import io.euhedral_execution.inference.core.runtime.graph.StageGraph;
import java.util.LinkedHashMap;
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

    private record Key(Shape full, Shape last, int chunks) {}

    private final EuhedralInferenceRuntime runtime;
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
        return this.shapes.computeIfAbsent(
                new Key(full, last, chunks),
                key -> new ChunkedShape(full, last, chunks, new Frames(full, last, chunks)));
    }

    /// Shapes kept now (tests).
    int size() {
        return this.shapes.size();
    }

    /// The frames of a prompt graph: each stage is its template's, told its chunk.
    private record Frames(Shape full, Shape last, int chunks) implements ChunkedShape.Chunks {
        @Override
        public StageFrame create(StageGraph graph, int stage, int chunk, int templateStage, ExecutionGpu gpu) {
            Shape template = chunk == this.chunks - 1 ? this.last : this.full;
            Stages.Stage frame =
                    Stages.create(graph, stage, template.instructions().get(templateStage), gpu, chunk);
            frame.scratchUse = template.scratchUse(templateStage);
            return frame;
        }

        @Override
        public GraphStorage newStorage(ExecutionGpu gpu) {
            return new WorkspaceStorage(gpu);
        }
    }
}
