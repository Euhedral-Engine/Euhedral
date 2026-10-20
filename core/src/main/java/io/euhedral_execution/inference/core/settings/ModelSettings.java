package io.euhedral_execution.inference.core.settings;

import java.nio.file.Path;
import java.util.Objects;

/// The model to serve. Both are required; the artifact carries its tokenizer.
///
/// @param artifact the model artifact file
/// @param name the model name clients send as `model`
public record ModelSettings(Path artifact, String name) {

    public ModelSettings {
        Objects.requireNonNull(artifact, "model.artifact");
        Objects.requireNonNull(name, "model.name");
        if (name.isBlank()) throw new IllegalArgumentException("model.name must not be blank");
    }
}
