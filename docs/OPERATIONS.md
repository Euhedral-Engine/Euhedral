# Operations

## Settings

Each setting is a property or its environment variable: `euhedral.inference.artifact-path` is
`EUHEDRAL_INFERENCE_ARTIFACT_PATH`. `PORT` sets the port (1738). The container and `mise run deploy` read them from a
`.env` file ([.env.example](../.env.example)); there the image fixes the artifact, checkpoint and CUDA library paths,
and `.env` gives the host paths mounted on them (`EUHEDRAL_ARTIFACT_FILE`, `EUHEDRAL_CHECKPOINT_DIR`).

| Setting | Default | Meaning |
|---|---|---|
| `euhedral.inference.artifact-path` | required | The `.edrl` artifact to serve. |
| `euhedral.inference.tokenizer-directory` | required | The checkpoint directory (`tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, `generation_config.json`). |
| `euhedral.inference.cuda-library-path` | required | `libeuhedral_cuda.so` (`euhedral_cuda.dll` on Windows). `libllguidance.so` sits beside it and the kernel sources in `../share/euhedral_cuda/`, as the build lays them out. |
| `euhedral.inference.worker-cpus` | required | Processor IDs or ranges for the engine's worker threads, for example `2-5,8`. |
| `euhedral.inference.model-id` | required | The name clients send as `model`. |
| `euhedral.inference.max-context-tokens` | 32768 | The longest prompt plus completion. The engine keeps device memory for that much KV cache and holds weights in pinned host memory when both do not fit. At most the model's 262144 positions. |
| `euhedral.inference.prefix-cache-bytes` | 4294967296 | Pinned host memory for the prefix cache ([PREFIX_CACHE.md](PREFIX_CACHE.md)); 0 turns it off. |
| `euhedral.inference.prefix-cache-checkpoint-tokens` | 2048 | Prompt tokens between stored checkpoints, a multiple of 512. |
| `euhedral.inference.shutdown-timeout` | 10s | How long closing the engine waits for its workers to stop. |
| `euhedral.api.default-max-tokens` | 4096 | Completion length when a request sets none. |
| `euhedral.api.max-queued-generations` | 16 | Requests waiting behind the running generation; more are refused with 503. |
| `euhedral.api.max-request-bytes` | 1048576 | Request body limit (413 beyond). |
| `euhedral.api.request-timeout` | 30m | Per-request limit. |
| `spring.lifecycle.timeout-per-shutdown-phase` | 30s | How long a shutdown waits for requests in flight (below). |

`EUHEDRAL_CUDA_INCLUDE_DIR` points at the CUDA headers the kernels compile against at startup (else `CUDA_HOME/include`,
`CUDA_PATH/include`, `/usr/local/cuda/include`); `LD_LIBRARY_PATH` must reach the CUDA runtime and NVRTC libraries. The
image sets both.

## Starting

Before anything loads, the server checks the files the settings name: the artifact, the checkpoint's tokenizer files,
the CUDA library with llguidance and the kernel sources beside it, and the CUDA headers. It reports every problem at
once, each with the setting to change, and exits within about a second:

```text
***************************
APPLICATION FAILED TO START
***************************

Description:

The server's settings point at files that are missing:

    euhedral.inference.artifact-path (EUHEDRAL_INFERENCE_ARTIFACT_PATH): no file at /models/nope.edrl
    euhedral.inference.cuda-library-path (EUHEDRAL_INFERENCE_CUDA_LIBRARY_PATH): the constrained-decoding library belongs at /opt/euhedral/lib/libllguidance.so, beside the CUDA library, and is missing

Action:

Correct the settings above. The settings and their environment variables are listed in docs/OPERATIONS.md.
```

What fails later, while the engine loads, is reported the same way, with the messages of its causes instead of the
chain of Spring beans around them: worker CPUs this process may not run on, a context longer than the model's positions
or than the GPU can hold with every weight in host memory, a GPU that is missing or not a Blackwell, a CUDA runtime
the native library cannot load. A port in use is reported by Spring Boot, after the weights have loaded.

A successful start logs the context and how it fits (RTX 5070 Ti, `qwen3_8_27b_q3`, the default context):

```text
Context 32768 tokens: 13854 MiB of the 15100 MiB free on the GPU, 0 MiB of weights in host memory
Prefix cache: 4096 MiB pinned, a checkpoint every 2048 tokens
Started EuhedralInferenceApplication in 7.761 seconds (process running for 7.951)
```

With `qwen3_8_27b_nvfp4` and a context of 131072 the same card keeps 3485 MiB of weights in host memory.

The port opens only after the engine has loaded, so a server that answers is ready.

## Health

`GET /health` answers `200 {"status":"up","engine":"ready","model":"..."}`, and `503` with `"engine":"closed"` once the
engine has closed. The image's Docker health check calls it.

## Shutting down

On `SIGTERM` or `SIGINT` the server stops accepting connections and waits up to
`spring.lifecycle.timeout-per-shutdown-phase` (30 s) for the requests in flight, queued ones included, to finish. What is
left after that is answered before the server stops: a request still queued gets 503 (`The server is shutting down.`),
the running generation is stopped at its next quantum and gets an error (in a stream, an `error` event), and the engine
then closes its sessions, its workers and the GPU. A second signal exits at once.

Give the process longer than the window to exit: Docker's default is 10 s, so run the container with
`--stop-timeout 45` (Kubernetes: `terminationGracePeriodSeconds: 45`), or shorten the window.

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
