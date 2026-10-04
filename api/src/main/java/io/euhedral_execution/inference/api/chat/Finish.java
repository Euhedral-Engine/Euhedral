package io.euhedral_execution.inference.api.chat;

/// How a generation ended. `stopSequence` is the request's stop string that ended it, for `STOP_SEQUENCE`.
public record Finish(Reason reason, String stopSequence, TokenUsage usage) {

    public enum Reason {
        /// The model ended its turn.
        END,
        /// A stop string matched.
        STOP_SEQUENCE,
        /// The completion budget ran out.
        LENGTH,
        /// The answer is tool calls.
        TOOL_CALLS
    }
}
