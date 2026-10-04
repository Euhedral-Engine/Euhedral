package io.euhedral_execution.inference.api.chat;

import java.io.IOException;

/// Delivery side of one generation, in one API's format: a JSON body or an SSE stream.
///
/// `start`, then `reasoning`, `text` and `toolCall` in output order (all reasoning comes first), then `finish`
/// run one at a time on the backend's workers. `start` runs once generation produced its first token (or ended
/// without one), with the prompt tokens the prefix cache restored. `toolCall` receives complete calls only, with
/// `index` counting
/// calls from zero. An `IOException` means the client is gone. Exactly one of `finish` or `fail` is invoked per
/// request unless the client abandoned it; `fail` must not throw and may run on a container thread when the
/// request times out.
public interface CompletionSink {

    void start(int cachedPromptTokens) throws IOException;

    void reasoning(String delta) throws IOException;

    void text(String delta) throws IOException;

    void toolCall(int index, GeneratedCall call) throws IOException;

    void finish(Finish finish) throws IOException;

    /// Writes nothing a client reads (an SSE comment or ping) while a stream's prompt is prefilled, so a client that
    /// left is noticed by the failed write. JSON responses write nothing.
    default void keepAlive() throws IOException {}

    void fail(ApiException error);
}
