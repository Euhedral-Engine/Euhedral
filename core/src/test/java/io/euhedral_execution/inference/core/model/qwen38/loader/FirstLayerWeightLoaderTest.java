package io.euhedral_execution.inference.core.model.qwen38.loader;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;

import io.euhedral_execution.inference.core.gpu.GpuMemory;
import io.euhedral_execution.inference.core.model.qwen38.artifact.Artifact;
import java.lang.reflect.Method;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;

class FirstLayerWeightLoaderTest {

    @Test
    void exposesSelectiveCompactFirstLayerLoading() {
        Method method = assertDoesNotThrow(
                () -> WeightLoader.class.getMethod("loadFirstLayer", Path.class, Artifact.class, GpuMemory.class));

        assertEquals(Weights.class, method.getReturnType());
    }
}
