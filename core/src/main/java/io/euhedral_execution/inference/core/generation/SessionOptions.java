package io.euhedral_execution.inference.core.generation;

/// Options of one session that a model honours where it can. `speculation`: decode with the artifact's drafter when
/// it has one; off for a control arm that decodes one token per quantum.
public record SessionOptions(boolean speculation) {
    public static final SessionOptions DEFAULT = new SessionOptions(true);
}
