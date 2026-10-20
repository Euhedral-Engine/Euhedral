package io.euhedral_execution.inference.core.settings;

import java.util.List;

/// The settings could not be loaded; carries every problem found.
public final class SettingsException extends RuntimeException {

    private final List<String> problems;

    public SettingsException(List<String> problems) {
        super(String.join("; ", problems));
        this.problems = List.copyOf(problems);
    }

    public List<String> problems() {
        return problems;
    }
}
