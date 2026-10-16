package io.euhedral_execution.inference.core.model.qwen38.loader;

import io.euhedral_execution.inference.core.artifact.TensorDescriptor;
import io.euhedral_execution.inference.core.artifact.TensorHandle;
import io.euhedral_execution.inference.core.artifact.TensorLoader;
import io.euhedral_execution.inference.core.gpu.GpuMemory;
import java.io.IOException;
import java.nio.file.Path;

/// Places one device-resident object of a compact artifact. The engine's own is [#OWNED]: it reads, checks and uploads
/// the object into memory the model then owns and frees. A caller that shares objects between models supplies a
/// loader whose handles are [#borrowed]: the loader never frees them, whatever fails, and neither does the model.
@FunctionalInterface
public interface ObjectLoader {

    /// Reads `descriptor`'s payload from `path` and places it on the device.
    TensorHandle load(Path path, TensorDescriptor descriptor, GpuMemory memory) throws IOException;

    /// Whether `handle` belongs to someone else, so that a failed load must not free it.
    default boolean borrowed(TensorHandle handle) {
        return false;
    }

    /// The engine's per-object load: the model owns what it loads.
    ObjectLoader OWNED = TensorLoader::load;
}
