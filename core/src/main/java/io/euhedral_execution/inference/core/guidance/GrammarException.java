package io.euhedral_execution.inference.core.guidance;

/// A grammar llguidance cannot compile: a syntax error, an unsupported JSON Schema keyword, or an unsatisfiable
/// schema. The message is llguidance's and describes only the grammar.
public final class GrammarException extends IllegalArgumentException {
    GrammarException(String message) {
        super(message);
    }

    /// llguidance's first line without its source position ("at 1(8): "); the rest quotes the grammar.
    static String summary(String message) {
        String first = message.lines().findFirst().orElse(message).strip();
        return first.replaceFirst("^at \\d+\\(\\d+\\): ", "");
    }
}
