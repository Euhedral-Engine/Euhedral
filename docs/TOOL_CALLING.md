# Tool calling

`/v1/chat/completions` accepts OpenAI-style function `tools`, `tool_choice` (`auto`, `none`, `required`, or a
named function), `parallel_tool_calls`, and assistant/tool-message replay.

When functions are offered, token sampling is constrained to a JSON `tool_calls` envelope or, for `auto`, a JSON
`content` envelope for an ordinary answer. The latter is returned as normal assistant text, not as JSON on the wire.
The server validates a complete call before returning OpenAI-style `tool_calls` and `finish_reason: "tool_calls"`.

With callable tools, SSE emits tool calls and ordinary answers only after their JSON envelopes validate, so
plain-answer text may be buffered until generation ends. Without callable tools, plain text streams incrementally.

Pass each returned call ID back in a `role: "tool"` message with its result, then send the next request. Tool
results are JSON-quoted in the model prompt, not executed by the server. The model may still choose a non-tool
answer with `tool_choice: "auto"`; malformed calls fail rather than being returned as successful tool calls.

Sampling enforces JSON syntax and offered function names, not the full argument schema. Argument checking covers
top-level types, required names, and `additionalProperties: false`; it does not enforce every JSON Schema
constraint (such as nested schemas, enum values, or numeric bounds). Requests for function `strict: true` are
rejected rather than promising full JSON Schema constrained decoding.
