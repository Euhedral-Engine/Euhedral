package io.euhedral_execution.inference.core;

import io.euhedral_execution.inference.core.model.ModelRuntime;

/// Test access to an engine's model runtime from the model packages' tests.
public final class TestEngines {

    private TestEngines() {}

    public static ModelRuntime modelRuntime(InferenceEngine engine) {
        return engine.modelRuntime();
    }
}
