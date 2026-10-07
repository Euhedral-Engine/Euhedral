package io.euhedral_execution.inference.core.artifact;

import java.io.IOException;

/// Signals that a Qwen artifact cannot be assembled into the declared weight structure.
public final class WeightLoadException extends IOException {

    public WeightLoadException(String message) {
        super(message);
    }

    public WeightLoadException(String message, Throwable cause) {
        super(message, cause);
    }
}
