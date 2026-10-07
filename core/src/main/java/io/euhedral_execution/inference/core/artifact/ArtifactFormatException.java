package io.euhedral_execution.inference.core.artifact;

import java.io.IOException;

public final class ArtifactFormatException extends IOException {

    private static final long serialVersionUID = 1L;

    public ArtifactFormatException(String message) {
        super(message);
    }
}
