# Operations

## Metrics

`GET /metrics` serves Prometheus text format. Point a Prometheus scrape job at `http://host:1738/metrics`; no other
management endpoint is exposed. Every series carries `application="euhedral-inference"`.

Nothing is recorded per token: request metrics are recorded once when a request is refused or ends, speculative
counts once per verification step (from the session's timing callbacks), and the engine and prefix-cache values are
read from the engine when the endpoint is scraped.

### Requests

`api` is `chat_completions`, `messages` or `responses`.

| Metric | Type | Meaning |
|---|---|---|
| `euhedral_requests_total{api,outcome}` | counter | `success`; `invalid` (refused while planning: bad or unsupported fields, over the context); `rejected` (capacity or shutdown); `cancelled` (the client left or the request timed out); `failed` (server error or an output that failed its final check) |
| `euhedral_finishes_total{api,reason}` | counter | Answered generations by `end`, `stop_sequence`, `length`, `tool_calls` |
| `euhedral_generations_active` | gauge | 1 while a generation runs |
| `euhedral_generations_queued` | gauge | Requests waiting for the generation slot |
| `euhedral_request_duration_seconds{api}` | histogram | Arrival to the last token |
| `euhedral_queue_wait_seconds{api}` | histogram | Arrival to the start of generation |
| `euhedral_time_to_first_token_seconds{api}` | histogram | Start of generation (prefix restore, prefill) to the first token |
| `euhedral_prefill_tokens_per_second` | histogram | Prefilled prompt tokens over the time to the first token |
| `euhedral_decode_tokens_per_second` | histogram | Tokens after the first over the time after it |
| `euhedral_prompt_tokens_total` | counter | Prompt tokens |
| `euhedral_prompt_cached_tokens_total` | counter | Prompt tokens restored from the prefix cache |
| `euhedral_prompt_prefilled_tokens_total` | counter | Prompt tokens prefilled |
| `euhedral_completion_tokens_total` | counter | Generated tokens, reasoning and terminator included |
| `euhedral_reasoning_tokens_total` | counter | Generated reasoning tokens |
| `euhedral_tool_calls_total{api}` | counter | Tool calls returned |
| `euhedral_schema_rejections_total{api}` | counter | Requests refused because a JSON Schema or grammar cannot be enforced |
| `euhedral_constrained_failures_total{reason}` | counter | Finished outputs that failed their final check (`invalid_tool_call`, `invalid_structured_output`) |

### Speculative decoding

| Metric | Type | Meaning |
|---|---|---|
| `euhedral_speculative_verifications_total` | counter | Verification steps |
| `euhedral_speculative_accepted_total{drafts}` | counter | Steps by the number of drafted tokens accepted (`0` to the artifact's depth): the acceptance histogram |
| `euhedral_speculative_accepted_tokens_total` | counter | Drafted tokens accepted; divided by the verifications, the mean accepted per step |

Only greedy, unconstrained requests speculate ([MTP_SPECULATIVE.md](MTP_SPECULATIVE.md)).

### Prefix cache

| Metric | Type | Meaning |
|---|---|---|
| `euhedral_prefix_lookups_total`, `euhedral_prefix_hits_total` | counter | Lookups, and those that restored a prefix |
| `euhedral_prefix_restored_tokens_total` | counter | Prompt tokens restored instead of prefilled |
| `euhedral_prefix_restores_total`, `euhedral_prefix_restore_seconds_total` | counter | Restores and the time they took |
| `euhedral_prefix_captures_total`, `euhedral_prefix_capture_seconds_total` | counter | Checkpoints stored and the time they took |
| `euhedral_prefix_skipped_total`, `euhedral_prefix_failed_total` | counter | Checkpoints not stored for lack of room, and those whose copies failed |
| `euhedral_prefix_evictions_total` | counter | Checkpoints evicted |
| `euhedral_prefix_nodes` | gauge | Checkpoints held |
| `euhedral_prefix_used_bytes`, `euhedral_prefix_capacity_bytes` | gauge | Pinned arena bytes in use, and its size |

Absent when the cache is off ([PREFIX_CACHE.md](PREFIX_CACHE.md)).

### Engine

| Metric | Type | Meaning |
|---|---|---|
| `euhedral_engine_info{model,artifact,speculative_depth}` | gauge | 1; the served model ID, the artifact (`q3`, `q3-compressed`, `nvfp4`, `nvfp4-compressed`) and its draft depth |
| `euhedral_engine_context_tokens` | gauge | Longest prompt plus completion |
| `euhedral_engine_workers` | gauge | Worker CPUs |
| `euhedral_engine_device_allocated_bytes` | gauge | Device memory the engine allocated: weights, KV cache, sequence state, workspaces |
| `euhedral_engine_device_free_bytes` | gauge | Free device memory the driver reports |
| `euhedral_engine_host_backed_weight_bytes` | gauge | Weights held in pinned host memory and streamed in |

The JVM's and the process's metrics (`jvm_*`, `process_*`, `process_uptime_seconds`), Tomcat's and HTTP server
request timings (`http_server_requests_seconds`) come from Micrometer.

### Examples

```promql
# Prefix-cache hit rate and the share of prompt tokens it saved
rate(euhedral_prefix_hits_total[5m]) / rate(euhedral_prefix_lookups_total[5m])
rate(euhedral_prompt_cached_tokens_total[5m]) / rate(euhedral_prompt_tokens_total[5m])

# Median time to first token
histogram_quantile(0.5, sum by (le) (rate(euhedral_time_to_first_token_seconds_bucket[5m])))

# Mean drafted tokens accepted per verification
rate(euhedral_speculative_accepted_tokens_total[5m]) / rate(euhedral_speculative_verifications_total[5m])
```
