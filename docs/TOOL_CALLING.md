# Tool calling

`/v1/chat/completions` accepts OpenAI-style function `tools`, `tool_choice` (`auto`, `none`, `required`, or a
named function), `parallel_tool_calls`, function `strict`, and assistant/tool-message replay. This page uses its names.
The other APIs spell the same controls their own way and generate calls the same way: Responses (`function` tools,
`function_call` and `function_call_output` items, [RESPONSES_API.md](RESPONSES_API.md)) and Messages (`input_schema`,
`tool_choice` `any`, `disable_parallel_tool_use`, `tool_use` and `tool_result` blocks, [ANTHROPIC_API.md](ANTHROPIC_API.md)).

## How a call is generated

The prompt carries the checkpoint template's own tool section: the definitions in `# Tools ... <tools>`, followed by the
template's instructions for the call format. The model writes a call as the template renders one, and earlier calls in
the conversation replay in the same form:

```text
<tool_call>
<function=get_weather>
<parameter=city>
Paris
</parameter>
<parameter=days>
3
</parameter>
</function>
</tool_call>
```

A string argument is written verbatim and any other value as JSON. Consecutive calls are separated by a newline.

- **`auto`**: sampling is free. The server splits the output into text and complete calls as it streams, types each
  argument by the function's schema, and returns OpenAI-style `tool_calls` with `finish_reason: "tool_calls"`. A call
  that names a function that was not offered, or whose parameters do not parse, fails the request with
  `invalid_tool_call`; it is never returned as text or as a call the model did not make. After a call is parsed its
  arguments are checked: top-level types, required names and `additionalProperties: false`.
- **`required` and a named function**: an llguidance grammar ([STRUCTURED_OUTPUT.md](STRUCTURED_OUTPUT.md)) forces the
  output to be one call, or one or more calls separated by `\n` (a named function allows only that name). After
  reasoning, up to eight whitespace bytes may precede it.
- **`auto` with a JSON `response_format`**: the grammar allows calls or the response document, so a direct answer is
  the JSON document itself, returned as the message content.
- **`parallel_tool_calls: false`**: decoding stops once the first call is complete, and the grammar of a constrained
  choice allows exactly one call.
- **`none`**: the definitions are left out of the prompt, as if no tools were sent, and the output is plain text.

### The closing tag

The format cannot represent a string argument that contains `</parameter>`: the parser ends a value at the first
`</parameter>` that is followed by another parameter or `</function>`. A constrained call enforces this in the grammar,
where a string value is any text that does not contain `</parameter>` (`<`, `</paramete` and every other text are
admitted). Free generation under `auto` is not constrained, so such a value is cut or the call fails as malformed.
Parameter names cannot contain `<`, `>` or line breaks (400).

## Arguments

- **`strict: true`**: under a constrained choice the call follows the function's `parameters` schema. The
  parameters are the schema's `properties` in the schema's order; required ones are present and optional ones may be
  omitted. A property typed as a string is any text (an `enum` or `const` of strings, exactly those texts); a property
  of another type is a JSON value of its schema, with the function's `$defs`, so `$ref`s resolve; an untyped property,
  or a type that includes `string` beside others, is any text. A name outside `properties` is generated only when the
  schema does not set `additionalProperties: false` (or for a required name it does not declare). Every JSON Schema
  keyword llguidance enforces is enforced for values that are not strings (types, `enum`, `const`, nested objects
  and arrays, `anyOf`, disjoint `oneOf`, `allOf`, `$ref`/`$defs`, numeric bounds, `multipleOf`, item and property
  counts, and a string's `pattern`, `format` and lengths inside a JSON value). A schema with a keyword it cannot
  enforce, or one no value can satisfy, is refused with 400 and the reason, naming `tools[i].function.parameters`; so
  is a string parameter that uses `pattern`, `format`, `minLength` or `maxLength` where the call is constrained, since
  a text value is not a JSON string. A strict function without `parameters` takes no parameters.
- **Not strict** (the default, except in Responses, where `strict` defaults to true as that API defines it): under a
  constrained choice the parameter names are any `[a-zA-Z0-9_-]+` and the values any text without `</parameter>`.
  After generation the server checks top-level argument types, required names, and `additionalProperties: false`, and
  fails the request when they do not hold.

Under `auto` the output is not constrained, so a strict function's arguments get the same checks as a non-strict one's.

## Streaming

Text streams as it is decoded; only a possible start of `<tool_call>`, and whitespace that could precede one, is held
back until the next chunk decides it. A call is emitted once its closing tag has arrived, with its complete
arguments. With thinking on, the reasoning streams first, unconstrained, and the answer follows `</think>`
([REASONING.md](REASONING.md)).

## Results

Pass each returned call ID back in a `role: "tool"` message with its result, then send the next request. Results
render as `<tool_response>` blocks in call order, and consecutive results share one user turn; the server never
executes anything.

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
