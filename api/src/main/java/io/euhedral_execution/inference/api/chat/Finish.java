package io.euhedral_execution.inference.api.chat;

import java.util.List;

/// How a generation ended. `stopSequence` is the request's stop string that ended it, for `STOP_SEQUENCE`. `tokenIds`
/// are the tokens the model generated, terminator included, or null when the request did not ask for them. With a
/// matched stop or `parallel_tool_calls: false` they may continue past the returned text, up to the end of the step
/// in which decoding stopped.
public record Finish(Reason reason, String stopSequence, TokenUsage usage, List<Integer> tokenIds) {

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
