package io.euhedral_execution.inference.core;

import io.euhedral_execution.inference.core.model.ModelRuntime;

/// Test access to an engine's model runtime and sessions from the model packages' tests.
public final class TestEngines {

    private TestEngines() {}

    public static ModelRuntime modelRuntime(InferenceEngine engine) {
        return engine.modelRuntime();
    }

    public static io.euhedral_execution.inference.core.model.qwen38.Session createSession(
            InferenceEngine engine, io.euhedral_execution.inference.core.sampling.GenerationConfig config) {
        return engine.createSession(config);
    }
}
