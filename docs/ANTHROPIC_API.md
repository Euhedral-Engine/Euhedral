# Anthropic Messages API

`POST /v1/messages` serves Anthropic-compatible clients. Point the client at `http://host:1738` (the SDKs append
`/v1/messages`) and use the server's model ID as `model`; any API key is accepted and none is checked.

```python
import anthropic

client = anthropic.Anthropic(base_url="http://localhost:1738", api_key="unused")
message = client.messages.create(
    model="qwen", max_tokens=1024,
    messages=[{"role": "user", "content": "Explain speculative decoding in two sentences."}],
)
print(message.content[0].text)
```

A request becomes the same conversation as a Chat Completions request and runs through the same rendering, grammar,
generation and prefix cache; only the request and response formats differ. A conversation sent through either API
renders to the same prompt, so the two share cached prefixes.

## Request

| Field | Behavior |
|---|---|
| `model` | Must be the served model ID. |
| `max_tokens` | Required; reasoning and answer together, within the context. |
| `system` | A string or text blocks (concatenated). |
| `messages` | `user` and `assistant` turns of a string or blocks: `text`, `tool_use`, `tool_result` (string or text blocks; `is_error` accepted and not rendered), `thinking` (replayed as the turn's reasoning; the `signature` is not checked). |
| `tools` | Custom tools: `name` (`[a-zA-Z0-9_-]{1,128}`), `description`, `input_schema`, `strict` ([TOOL_CALLING.md](TOOL_CALLING.md)). |
| `tool_choice` | `auto`, `any` (a call is required), `tool` with `name`, `none`; `disable_parallel_tool_use` allows one call. |
| `thinking` | Omitted or `disabled`: no thinking. `adaptive`: thinking, the model decides how long. `enabled` with `budget_tokens`: thinking, ended by `</think>` after at most `budget_tokens` tokens; must be below `max_tokens`. |
| `output_config.effort` | With thinking: `low` and `medium` select the template's levels, `high` and `max` its highest ([REASONING.md](REASONING.md)). Without thinking it is refused. |
| `output_config.format` | `{"type": "json_schema", "schema": ...}`: the text block is a JSON document of the schema ([STRUCTURED_OUTPUT.md](STRUCTURED_OUTPUT.md)). The earlier top-level `output_format` is accepted too. |
| `stop_sequences` | Up to 32 non-empty strings, matched in the answer only. |
| `temperature`, `top_p`, `top_k` | Sampling; `temperature` 0 to 1 (0 is greedy); omitted values take the checkpoint's `generation_config.json` (1.0, 0.95, 20). |
| `stream` | Server-sent events, below. |
| `metadata`, `service_tier`, `cache_control` | Accepted without effect: prefix reuse is automatic. |

Refused with an `invalid_request_error`: images, documents and other block types, server tools (`web_search`, `bash`,
...), `redacted_thinking`, a final `assistant` message (prefill: the checkpoint template has no form that continues a
turn), `mcp_servers`, `container`, `context_management`, and any unknown field.

Thinking is opt-in here, as Anthropic defines it, while Chat Completions keeps the template's default (thinking on): a
client written for this API reads `content[0]` as the answer unless it asked for thinking.

## Response

```json
{"id": "msg_...", "type": "message", "role": "assistant", "model": "qwen",
 "content": [{"type": "thinking", "thinking": "...", "signature": "euhedral-..."},
             {"type": "text", "text": "..."}],
 "stop_reason": "end_turn", "stop_sequence": null,
 "usage": {"input_tokens": 320, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 4096, "output_tokens": 17}}
```

- Blocks: a `thinking` block when there was reasoning, then a `text` block or `tool_use` blocks
  (`{"type": "tool_use", "id": "toolu_...", "name": ..., "input": {...}}`).
- `stop_reason`: `end_turn`, `max_tokens`, `stop_sequence` (with `stop_sequence` set), `tool_use`.
- `usage.input_tokens` counts the prompt tokens that were prefilled and `cache_read_input_tokens` those restored from
  the prefix cache ([PREFIX_CACHE.md](PREFIX_CACHE.md)); nothing is written to a cache on request, so
  `cache_creation_input_tokens` is 0. `output_tokens` includes thinking tokens.

## Streaming

`stream: true` writes Anthropic's events: `message_start` (written at the first generated token, with the final
`input_tokens` and `cache_read_input_tokens`), then per block `content_block_start`, `content_block_delta`
(`thinking_delta`, then a `signature_delta`; `text_delta`; `input_json_delta` carrying a call's whole arguments),
`content_block_stop`, then `message_delta` (`stop_reason`, `stop_sequence`, `usage`) and `message_stop`. A failure after
the stream began is an `error` event. Tool calls arrive once their arguments are complete.

A client that disconnects cancels its generation before the next quantum, as for Chat Completions.

## Other endpoints

- `POST /v1/messages/count_tokens`: `{"input_tokens": N}` for a request without `max_tokens`, rendered and encoded
  without generating.
- `GET /v1/models` with an `anthropic-version` header lists the model in Anthropic's format.

Errors use Anthropic's object, `{"type": "error", "error": {"type": "invalid_request_error", "message": ...}}`, with
`not_found_error`, `request_too_large`, `overloaded_error` (status 529, as Anthropic's API uses it: at capacity or shutting down) and `api_error`.
