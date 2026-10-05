# API compatibility

The server answers three request formats, all on one internal pipeline: OpenAI Chat Completions
(`POST /v1/chat/completions`), Anthropic Messages (`POST /v1/messages`, [ANTHROPIC_API.md](ANTHROPIC_API.md)) and
OpenAI Responses (`POST /v1/responses`, [RESPONSES_API.md](RESPONSES_API.md)). Also: `GET /v1/models` (OpenAI's format,
Anthropic's for clients that send `anthropic-version`), `GET /v1/models/{id}`, `POST /v1/messages/count_tokens`,
`GET /health`, `GET /metrics` ([OPERATIONS.md](OPERATIONS.md)).

Every field of a request is one of:

- **supported**: it does what the API says;
- **neutral**: accepted only with the value that asks for no behavior (anything else is refused);
- **metadata**: accepted and without effect, because it cannot change the output;
- **unsupported**: refused with 400 (`unsupported_parameter`, naming the field), or 400 `unrecognized_argument` for
  a field the API does not define.

Nothing that changes generation is accepted and ignored.

## Chat Completions

| Field | Class | Notes |
|---|---|---|
| `model` | supported | Must be the served model ID (404 `model_not_found` otherwise). |
| `messages` | supported | Roles `system`, `developer` (as system), `user`, `assistant`, `tool`. Content: a string or `text` parts. Assistant `tool_calls` and `reasoning_content` replay; `tool` messages need `tool_call_id`. |
| `max_tokens`, `max_completion_tokens` | supported | Either (they must agree if both are sent); the default is `euhedral.api.default-max-tokens` (4096), capped by the context. |
| `temperature` | supported | 0 to 2; 0 is greedy. Default from `generation_config.json` (1.0). |
| `top_p` | supported | 0 to 1; 0 is greedy. Default 0.95. |
| `top_k` | supported | Not an OpenAI field; accepted as vLLM spells it: at least 1, or 0/-1 for no limit. Default 20. |
| `seed` | supported | Seeds the sampler's random stream; greedy requests do not use it. |
| `stop` | supported | A string or up to 4 strings; matched in the answer, not in the reasoning. |
| `stream` | supported | Server-sent events, `chat.completion.chunk`, ending with `data: [DONE]`. |
| `stream_options` | supported | `include_usage`; `include_obfuscation` accepted without effect. Only with `stream`. |
| `tools` | supported | `function` tools; `strict` enforces the parameters schema ([TOOL_CALLING.md](TOOL_CALLING.md)). Other tool types are refused. |
| `tool_choice` | supported | `auto`, `none`, `required`, a named function. |
| `parallel_tool_calls` | supported | `false` allows one call and stops decoding after it. |
| `response_format` | supported | `text`, `json_object`, `json_schema` ([STRUCTURED_OUTPUT.md](STRUCTURED_OUTPUT.md)). |
| `reasoning_effort` | supported | `none`, `low`, `medium`, `high` (`xhigh` too); `minimal` refused ([REASONING.md](REASONING.md)). |
| `chat_template_kwargs` | supported | Not an OpenAI field: the template's `enable_thinking`, `reasoning_effort`, `preserve_thinking`. |
| `n` | neutral | 1. One choice per request. |
| `logprobs`, `top_logprobs` | neutral | `false`, `0`. Log probabilities are not returned. |
| `frequency_penalty`, `presence_penalty` | neutral | 0. |
| `repeat_penalty`, `repetition_penalty`, `min_p` | neutral | 1, 1, 0 (llama.cpp and vLLM extensions). |
| `logit_bias` | neutral | `{}`. |
| `modalities` | neutral | `["text"]`. |
| `functions`, `function_call` | neutral | `[]`, `"none"`. Legacy function calling: use `tools`. |
| `audio`, `prediction`, `web_search_options`, `verbosity` | unsupported | |
| `user`, `metadata`, `store`, `service_tier`, `safety_identifier`, `prompt_cache_key`, `prompt_cache_retention` | metadata | Nothing is stored; prefix reuse is automatic ([PREFIX_CACHE.md](PREFIX_CACHE.md)). |
| `messages[].name` | metadata | The checkpoint's template does not render participant names. |
| `messages[].refusal`, `messages[].audio`, `messages[].function_call` | unsupported | |
| content parts `image_url`, `input_audio`, `file` | unsupported | The server serves text only. |

The response is `chat.completion` with one choice. `message.reasoning_content` carries the reasoning when there is any;
`usage` includes `prompt_tokens_details.cached_tokens` (prompt tokens restored from the prefix cache) and, with thinking,
`completion_tokens_details.reasoning_tokens`. `finish_reason` is `stop` (end of turn or a stop string), `length`, or
`tool_calls`. Errors are OpenAI error objects; a failure after a stream began is an in-band error event.

## Unsupported on purpose

- **Several sequences at once** (`n` > 1, concurrent generations): the engine sizes device memory for one sequence and
  runs one generation at a time; others wait in a bounded queue (`euhedral.api.max-queued-generations`, 503 beyond it).
- **Log probabilities and logit bias**: sampling selects on the host from the final row; returning probabilities or
  biasing them would be features of their own, and no client this server targets needs them.
- **Penalties**: not implemented; refused rather than approximated.
- **Images, audio, files**: the artifacts are the text model.
- **Server-side state** (stored responses, `previous_response_id`, conversations): every request carries its
  conversation; the prefix cache makes resending it cheap.
- **Hosted tools and MCP**: the server returns tool calls; the client executes them ([TOOL_CALLING.md](TOOL_CALLING.md)).

## Other differences

- Speculative decoding (MTP or DFlash2, as the artifact selects) runs only for greedy requests without a grammar; sampled
  or constrained requests decode one token per step.
- Authentication is not implemented: any API key is accepted. Put the server behind a proxy that authenticates if it is
  reachable by others.
- A request body is limited to `euhedral.api.max-request-bytes` (1 MiB, 413 beyond) and a request to
  `euhedral.api.request-timeout` (30 minutes).

## Real clients

`tools/client_compatibility.py` runs the OpenAI and Anthropic Python clients and `curl` against a running server and
checks what a client relies on: response shapes, stream order, finish and stop reasons, tool-call loops, cached-token
counts, cancellation and refusals. Start the server with `EUHEDRAL_API_MAX_QUEUED_GENERATIONS=1` so the capacity
scenario can fill the queue.

Run on 2026-10-04: RTX 5070 Ti, `qwen3_8_27b_nvfp4_compressed`, openai 3.24.0, anthropic 1.11.0. The prefix-cache
scenario's first prompt was already cached from an earlier run of the script, so its time is two restored requests. The
capacity scenario waits for two long generations.

| Scenario | Client | Result | Time | Observed |
|---|---|---|---|---|
| Chat, JSON response | openai | pass | 1.1 s | finish=stop, 55 tokens: 'The three primary colors are **Red**, **Blue**, and **Yellow' |
| Chat, streaming with reasoning on (low) | openai | pass | 1.4 s | 39 reasoning chunks before 3 content chunks; reasoning_tokens=51; answer '391' |
| Chat, reasoning off | openai | pass | 0.2 s | no reasoning_content; answer '391' |
| Chat, raw SSE | curl | pass | 0.3 s | 11 chat.completion.chunk events, then [DONE] |
| Structured output, json_schema (sampled) | openai | pass | 1.2 s | valid document: {"name": "Elena Voss", "age": 29, "languages": ["Rust", "Python", "Go"]} |
| Tool call, single | openai | pass | 0.5 s | 1 call (parallel_tool_calls=false): get_weather({"city":"Paris"}) |
| Tool calls, parallel, and tool-result continuation | openai | pass | 1.6 s | parallel calls for ['Paris', 'Rome']; after the results: 'The weather in Paris is 18°C and raining. In Rome, it is 27°C and sunn' |
| Responses, JSON and streaming | openai | pass | 0.9 s | output ['reasoning', 'message'], text 'Tokyo'; stream of 11 events |
| Responses, tool call and function_call_output | openai | pass | 1.0 s | function_call {"city":"Oslo"}, then 'It is -3°C and snowing in Oslo.' |
| Messages, JSON and streaming | anthropic | pass | 0.6 s | 'Saturn'; stream of 13 text deltas |
| Messages, thinking enabled | anthropic | pass | 1.0 s | blocks ['thinking', 'text']; answer '12 × 34 = **408**' |
| Messages, tool_use and tool_result | anthropic | pass | 1.1 s | tool_use {'city': 'Lima'}, then 'It is currently 21°C and cloudy in Lima.' |
| Long conversation, prefix cache (and across APIs) | openai + anthropic | pass | 3.3 s | prompt 22925 tokens; follow-up restored 22528 of 22985 (0.91 s); the same conversation through Messages restored 22528 |
| Disconnect mid-stream | openai | pass | 1.0 s | closed after 20 chunks; next request answered in 0.18 s; cancelled +1 |
| Capacity rejection (queue of one) | openai + anthropic | pass | 142.9 s | running and queued requests answered; a third refused: openai 503 service_unavailable_error, anthropic 529 OverloadedError |
