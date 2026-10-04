# Structured output

`response_format` makes the answer a JSON document, and `strict` tools make call arguments follow their schemas
([TOOL_CALLING.md](TOOL_CALLING.md)). Both are enforced while sampling, by [llguidance](https://github.com/guidance-ai/llguidance):
at every step only tokens that keep the text a prefix of a valid document can be selected, and a generation
terminator only once the document is complete. Nothing is generated freely and repaired afterwards.

## Requests

```json
"response_format": {"type": "json_object"}
"response_format": {"type": "json_schema", "json_schema": {"name": "person", "strict": true, "schema": {...}}}
```

- `json_object`: any JSON object.
- `json_schema`: a document of `schema` (any JSON value when `schema` is omitted). `name` is required
  (`[a-zA-Z0-9_-]{1,64}`); `description` is accepted. `strict: false` is enforced exactly like `strict: true`: a
  schema that cannot be enforced is refused, never approximated.
- `text` (or omitted): free text.

Responses takes the same formats as `text.format` ([RESPONSES_API.md](RESPONSES_API.md)), and Messages a
`json_schema` format as `output_config.format` ([ANTHROPIC_API.md](ANTHROPIC_API.md)); both are enforced the same way.

The prompt is not changed: tell the model in your messages what the document should contain. A model that is not
asked for JSON still produces valid JSON, but its content suffers.

With thinking on ([REASONING.md](REASONING.md)) the reasoning is free text and the document follows `</think>`. With
`tools` and `tool_choice: "auto"`, a direct answer is the document, returned as the message `content`.

`stop` cannot be combined with a JSON format: a stop string inside the document would end it invalid. A generation
that runs out of `max_tokens` ends with `finish_reason: "length"` and the unfinished document, as OpenAI does.

## What is enforced

Schemas compile in llguidance's strict mode, so a keyword it does not enforce is refused rather than ignored.
Enforced: `type` (one or several), `properties`, `required`, `additionalProperties`, `patternProperties`,
`minProperties`/`maxProperties`, `items`, `prefixItems`, `minItems`/`maxItems`, `enum`, `const`, `anyOf`,
`oneOf` whose alternatives cannot both match, `allOf`, `$ref` with `$defs`/`definitions` (recursion included),
`minLength`/`maxLength`, `pattern`, `format` (`date-time`, `date`, `time`, `duration`, `email`, `hostname`, `ipv4`, `ipv6`, `uuid`,
`uri`), `minimum`/`maximum`/`exclusiveMinimum`/`exclusiveMaximum`, `multipleOf`.
Annotations are accepted and have no effect on validity: `title`, `description`, `default`, `examples`, `$schema`,
`$id`, `$comment`, `readOnly`, `writeOnly`, `contentMediaType`, `contentEncoding`.

Refused with 400 and llguidance's reason (`param` names the schema): any other keyword (`not`, `if`/`then`/`else`,
`dependentRequired`, `uniqueItems`, `contains`, `unevaluatedProperties`, ...), a `oneOf` whose alternatives may
overlap, an unknown `format`, a schema no value satisfies (`minimum` above `maximum`, a required property no value
satisfies), and the llguidance option key `x-guidance`.

The document is JSON as llguidance generates it: whitespace between tokens is allowed in runs of at most 40 bytes
(so a model cannot pad forever), properties appear in the schema's order, and nothing precedes or follows the
document. After a finished generation the server also parses the answer, and fails the request with
`invalid_structured_output` if it were ever not JSON.

## Cost

Measured on `q3`, sampling at temperature 1 (top-k 20, top-p 0.95), a schema of an array of objects, 600 generated
tokens: computing a token mask takes 16-18 us and committing a token 3-5 us. The sampler tests the best candidates
against the mask and needs the whole row (470 us) on about 60% of steps, where fewer than top-k tokens near the top
are allowed, against 256 us otherwise; in all, about 0.15 ms of host work per token of an 18 ms step. Compiling a
grammar takes 0.5-5 ms on a worker while the request is planned; the last 32 compiled grammars are kept, so a client
that repeats its schema or tools compiles them once.

A constrained greedy request does not draft with MTP: speculative decoding runs only for greedy, unconstrained
requests (and free reasoning before a constrained answer is still constrained in this sense).

## Rejected

- A hand-written byte-level JSON Schema recognizer, generalizing the earlier tool-call envelope automaton. llguidance
  covers far more of JSON Schema, is used by llama.cpp, vLLM and SGLang, and computes masks in microseconds.
- llguidance's default features: they add a Rayon thread pool; the library is built without it, so mask computation
  runs on the engine's worker that selects the token.
