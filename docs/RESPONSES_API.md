# OpenAI Responses API

`POST /v1/responses` serves clients of OpenAI's Responses API. A request becomes the same conversation as a Chat
Completions request and runs through the same rendering, grammar, generation and prefix cache; only the formats
differ.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:1738/v1", api_key="unused")
response = client.responses.create(model="qwen", input="Explain speculative decoding in two sentences.")
print(response.output_text)
```

## Stateless

The server keeps nothing between requests. Send the whole conversation as `input`, including the items earlier
responses returned. `previous_response_id`, `conversation` and `background: true` are refused; `store` is accepted and
has no effect (nothing is stored, and responses cannot be retrieved later).

## Request

| Field | Behavior |
|---|---|
| `model` | Must be the served model ID. |
| `input` | A string (one user message) or items: `message` (`user`, `assistant`, and leading `system`/`developer`; content a string or `input_text`/`output_text` parts), `function_call`, `function_call_output`, `reasoning`. |
| `instructions` | The system message; leading `system`/`developer` messages are appended to it. |
| `tools` | `function` tools: `name`, `description`, `parameters`, `strict`. As this API defines it, `strict` defaults to **true**: arguments follow the schema ([TOOL_CALLING.md](TOOL_CALLING.md)), and a schema that cannot be enforced is refused unless the tool sets `strict: false`. |
| `tool_choice`, `parallel_tool_calls` | `auto`, `none`, `required`, `{"type": "function", "name": ...}`. |
| `text.format` | `text`, `json_object`, `json_schema` (`name`, `schema`, `strict`) ([STRUCTURED_OUTPUT.md](STRUCTURED_OUTPUT.md)). |
| `reasoning.effort` | `none`, `low`, `medium`, `high`/`xhigh`; omitted, the template's default (thinking at `xhigh`) ([REASONING.md](REASONING.md)). |
| `reasoning.summary` | Accepted (`auto`, `concise`, `detailed`), but **no summary is generated**: the reasoning item's `summary` is empty and the reasoning itself is returned whole as `reasoning_text` content. |
| `include` | Only `reasoning.encrypted_content`: reasoning items carry an opaque token of their text to send back. It is the text in base64, not encrypted. |
| `max_output_tokens`, `temperature`, `top_p`, `stream` | As in Chat Completions. |
| `metadata`, `user`, `service_tier`, `safety_identifier`, `prompt_cache_key`, `truncation: "disabled"` | Accepted without effect. |

Refused: built-in tools (`web_search`, `file_search`, `computer_use_preview`, `code_interpreter`, `mcp`,
`image_generation`, ...), image, file and audio input, `item_reference`, `text.verbosity`, other `include` values,
`truncation: "auto"`, `max_tool_calls`, `top_logprobs`, `prompt`, `reasoning.effort: "minimal"`, and unknown fields.

A replayed `reasoning` item is read from its `reasoning_text` content, else its `encrypted_content` (only tokens this
server issued), else its summary text, and rendered as the reasoning of the assistant turn it precedes. Reasoning,
`message` and `function_call` items in a row form one assistant turn; each `function_call` needs a
`function_call_output` with its `call_id` before the conversation continues.

## Response

Output items, in order: a `reasoning` item when there was reasoning, then a `message` with one `output_text` part, or
`function_call` items (`call_id`, `name`, `arguments`). `status` is `completed`, or `incomplete` with
`incomplete_details.reason: "max_output_tokens"`. `usage` has `input_tokens_details.cached_tokens` (prompt tokens
restored from the prefix cache) and `output_tokens_details.reasoning_tokens`.

## Streaming

`stream: true` writes `response.created`, `response.in_progress`, then per item `response.output_item.added`,
`response.content_part.added`, `response.reasoning_text.delta`/`.done` or `response.output_text.delta`/`.done`,
`response.content_part.done`, `response.output_item.done` (a function call: `response.function_call_arguments.delta`
and `.done` with its whole arguments), and finally `response.completed` or `response.incomplete` with the full
response. Every event carries an increasing `sequence_number`. A failure after the stream began is an `error` event; a
client that disconnects cancels its generation before the next quantum.
