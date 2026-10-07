package io.euhedral_execution.inference.core.runtime.graph;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;

/// The shape of a unit of work, given to the runtime to instantiate: a static DAG of stages, how to
/// build the frame of each stage, and the storage a graph of this shape owns. The runtime builds
/// one reusable [StageGraph] per concurrent quantum of a shape; it knows nothing else about the
/// model.
public interface GraphShape {

    /// The stages and their edges.
    StageTopology topology();

    /// The frame of stage `stage`, built once when a graph of this shape is built.
    StageFrame createStage(StageGraph graph, int stage, ExecutionGpu gpu);

    /// The storage one graph of this shape owns.
    GraphStorage newStorage(ExecutionGpu gpu);
}
