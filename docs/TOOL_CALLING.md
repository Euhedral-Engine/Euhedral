# Tool calling

`/v1/chat/completions` accepts OpenAI-style function `tools`, `tool_choice` (`auto`, `none`, `required`, or a
named function), `parallel_tool_calls`, function `strict`, and assistant/tool-message replay. This page uses its names.
The other APIs spell the same controls their own way and generate calls the same way: Responses (`function` tools,
`function_call` and `function_call_output` items, [RESPONSES_API.md](RESPONSES_API.md)) and Messages (`input_schema`,
`tool_choice` `any`, `disable_parallel_tool_use`, `tool_use` and `tool_result` blocks, [ANTHROPIC_API.md](ANTHROPIC_API.md)).

## How a call is generated

When functions are offered, sampling is constrained by an llguidance grammar
([STRUCTURED_OUTPUT.md](STRUCTURED_OUTPUT.md)) to one JSON envelope, spelled exactly:

```text
{"tool_calls":[{"name":"get_weather","arguments":{...}}, ...]}      a call (or several)
{"content":"..."}                                                  a direct answer (auto only)
```

`required` and a named function allow only the first form (a named function only that name);
`parallel_tool_calls: false` allows exactly one call, and decoding stops once it is complete. A direct answer is
returned as ordinary assistant text, not as JSON on the wire. The server parses the finished envelope before
returning OpenAI-style `tool_calls` and `finish_reason: "tool_calls"`; a malformed call fails rather than being
returned as a successful one.

The envelope is JSON rather than the checkpoint template's `<tool_call>` XML because JSON strings carry any argument
text unambiguously; the template's XML parameters cannot hold their own closing tags. Prompts still render the
template's tool definitions (`# Tools ... <tools>`), followed by instructions for the JSON form.

## Arguments

- **`strict: true`**: the arguments are generated under the function's `parameters` schema. Every JSON Schema keyword
  llguidance enforces is enforced (types, `properties`, `required`, `additionalProperties`, `enum`, `const`, nested
  objects and arrays, `anyOf`, disjoint `oneOf`, `allOf`, `$ref`/`$defs`, string `pattern`/`format`/lengths, numeric
  bounds, `multipleOf`, item and property counts). A schema with a keyword it cannot enforce, or one no value can
  satisfy, is refused with 400 and the reason, naming `tools[i].function.parameters`. A strict function without
  `parameters` takes exactly `{}`.
- **Not strict** (the default, except in Responses, where `strict` defaults to true as that API defines it): the
  arguments are any JSON object. After generation the server still checks
  top-level argument types, required names, and `additionalProperties: false`, and fails the request when they do not
  hold.

Each strict function's arguments are their own grammar rule, so a schema's `$ref`s resolve against its own root.

## Streaming

With callable tools, SSE emits tool calls and direct answers only after their JSON envelopes are complete, so
direct-answer text may be buffered until generation ends. Without callable tools, plain text streams incrementally.
With thinking on, the reasoning streams first, unconstrained, and the envelope follows `</think>`
([REASONING.md](REASONING.md)).

## Results

Pass each returned call ID back in a `role: "tool"` message with its result, then send the next request. Tool
results are JSON-quoted in the model prompt, in call order; the server never executes anything.

## MCP

The server does not discover, connect to, or execute MCP servers, and it will not. MCP is the application's layer:

1. An MCP-aware client lists the tools of its MCP servers. Each MCP tool has a `name`, a `description` and an
   `inputSchema` (a JSON Schema object).
2. It offers them as functions: `{"type": "function", "function": {"name": ..., "description": ..., "parameters":
   inputSchema}}`, optionally with `strict: true` when the schema uses only enforceable keywords.
3. Euhedral returns `tool_calls`.
4. The client executes each call through MCP and sends the result back as a `role: "tool"` message.
5. Euhedral continues the conversation.

Function names must match `[a-zA-Z0-9_-]{1,64}`; a client exposing MCP tools whose names contain other characters
(dots, slashes) maps them to such names and back, as it would for any OpenAI-compatible model.
