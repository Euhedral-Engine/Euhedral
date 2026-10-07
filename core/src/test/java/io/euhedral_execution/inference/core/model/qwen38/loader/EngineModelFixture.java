package io.euhedral_execution.inference.core.model.qwen38.loader;

import io.euhedral_execution.inference.core.gpu.GpuMemory;
import io.euhedral_execution.inference.core.model.qwen38.Qwen38Model;
import java.io.IOException;

public final class EngineModelFixture {
    public static Qwen38Model load(GpuMemory gpu, Weights weights) throws IOException {
        return Qwen38Model.load(gpu, memory -> {
            memory.allocate(128);
            return weights;
        });
    }
}
