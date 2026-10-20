package io.euhedral_execution.inference.core.settings;

import java.nio.file.Path;
import java.util.Objects;

/// The model to serve. All three are required.
///
/// @param artifact the model artifact file
/// @param tokenizer the directory holding the tokenizer files
/// @param name the model name clients send as `model`
public record ModelSettings(Path artifact, Path tokenizer, String name) {

    public ModelSettings {
        Objects.requireNonNull(artifact, "model.artifact");
        Objects.requireNonNull(tokenizer, "model.tokenizer");
        Objects.requireNonNull(name, "model.name");
        if (name.isBlank()) throw new IllegalArgumentException("model.name must not be blank");
    }
}
