package io.euhedral_execution.inference.api.chat;

import io.euhedral_execution.inference.api.openai.OpenAiException;
import io.euhedral_execution.inference.api.openai.ToolCall;
import io.euhedral_execution.inference.api.openai.Usage;
import java.io.IOException;

/// Delivery side of one generation: a JSON body or an SSE stream.
///
/// `start`, then `text` and `toolCall` in output order, then `finish` run one at a time on the backend's
/// workers.
/// `toolCall` receives complete calls only, with `index` counting calls from zero. An `IOException` means the
/// client is gone. Exactly one of `finish` or `fail` is invoked per request unless the client abandoned it;
/// `fail` must not throw and may run on a container thread when the request times out.
interface CompletionSink {

    void start() throws IOException;

    void text(String delta) throws IOException;

    void toolCall(int index, ToolCall call) throws IOException;

    void finish(String finishReason, Usage usage) throws IOException;

    void fail(OpenAiException error);
}
