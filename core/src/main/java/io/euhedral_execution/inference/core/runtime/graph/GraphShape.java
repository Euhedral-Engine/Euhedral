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

    /// The runtime workspace's buffers stage `stage` reads or writes, by index. A graph admitted after another
    /// orders its first accessors of a buffer behind the other's last accessors of it ([WorkspaceOwner]).
    default int[] workspaceBuffers(int stage) {
        return NO_BUFFERS;
    }

    /// How many workspace buffers the shape's stages may name.
    default int workspaceBufferCount() {
        return 0;
    }

    /// The sequence state stage `stage` reads or writes, by key: state one chunk of a prompt (or one graph of a
    /// sequence) hands to the next, such as a layer's recurrent state or its KV rows. A later chunk's first
    /// users of a key follow the earlier chunk's last users of it ([ChunkedShape], [CarriedState]).
    default int[] carriedState(int stage) {
        return NO_BUFFERS;
    }

    /// How many carried-state keys the shape's stages may name.
    default int carriedStateCount() {
        return 0;
    }

    int[] NO_BUFFERS = new int[0];
}
