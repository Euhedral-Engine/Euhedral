"""The fixed request set of `api_diff.py`, as data.

`requests_for(model, body_limit, context_tokens)` returns the list of requests, each a dict:

  name         the file name and the row label of the request
  method, path the HTTP request line
  json         the body as a JSON value, or
  raw          the body as bytes (a body that is not JSON, or too large)
  headers      extra request headers (a JSON body gets `content-type: application/json`)
  stream       true when the reply is a server-sent event stream
  chunked      send the body with `transfer-encoding: chunked`
  disconnect_after  read this many events of the stream, then close the connection
  refusal      true when the Rust server is expected to refuse the request with `unsupported_parameter` for now

Every request is deterministic: greedy (temperature 0), or sampled with a seed (the chat surface alone takes one).
Replies are short (small token limits) so that the set runs within one arbiter grant.
"""

from __future__ import annotations

PROMPT = "Say hello in one short sentence."
WEATHER_PROPERTIES = {"city": {"type": "string", "description": "The city."}}


def _request(name, method, path, body=None, **fields):
    entry = {"name": name, "method": method, "path": path, "headers": {}, "stream": False}
    if body is not None:
        entry["json"] = body
    entry.update(fields)
    return entry


def requests_for(model: str, body_limit: int, context_tokens: int) -> list[dict]:
    out: list[dict] = []

    def post(name, path, body, **fields):
        out.append(_request(name, "POST", path, body, **fields))

    def get(name, path, **fields):
        out.append(_request(name, "GET", path, **fields))

    # -- service routes
    get("health", "/health")
    get("models_list", "/v1/models")
    get("models_get", f"/v1/models/{model}")
    get("models_get_unknown", "/v1/models/no-such-model")

    # -- chat completions
    chat = "/v1/chat/completions"
    hello = [{"role": "user", "content": PROMPT}]
    base = {"model": model, "messages": hello, "temperature": 0, "max_tokens": 24, "reasoning_effort": "none"}

    def chat_body(**extra):
        return {**base, **extra}

    post("chat_greedy", chat, chat_body())
    post("chat_greedy_stream", chat, chat_body(stream=True), stream=True)
    post("chat_stream_usage", chat, chat_body(stream=True, stream_options={"include_usage": True}), stream=True)
    post("chat_stream_usage_off", chat, chat_body(stream=True, stream_options={"include_usage": False}), stream=True)
    sampled = {"temperature": 0.8, "top_p": 0.9, "seed": 1234}
    post("chat_sampled_seed", chat, chat_body(**sampled))
    post("chat_sampled_seed_stream", chat, chat_body(stream=True, **sampled), stream=True)
    post("chat_sampled_other_seed", chat, chat_body(**{**sampled, "seed": 99}))
    post("chat_system_message", chat, chat_body(messages=[{"role": "system", "content": "Answer in French."}, *hello]))
    post("chat_content_parts", chat, chat_body(messages=[{"role": "user", "content": [{"type": "text", "text": PROMPT}]}]))
    post("chat_stop_string", chat, chat_body(stop=[","], max_tokens=32,
                                             messages=[{"role": "user", "content": "Count from one to ten, with commas."}]))
    post("chat_stop_list_stream", chat, chat_body(stop=["3", "five"], max_tokens=32, stream=True,
                                                  messages=[{"role": "user", "content": "Count from one to ten, with commas."}]), stream=True)
    post("chat_max_tokens_hit", chat, chat_body(max_tokens=3))
    post("chat_max_tokens_hit_stream", chat, chat_body(max_tokens=3, stream=True), stream=True)
    post("chat_max_completion_tokens", chat, {k: v for k, v in chat_body(max_completion_tokens=5).items() if k != "max_tokens"})
    post("chat_reasoning_on", chat, chat_body(reasoning_effort="low", max_tokens=48))
    post("chat_reasoning_on_stream", chat, chat_body(reasoning_effort="low", max_tokens=48, stream=True), stream=True)
    post("chat_reasoning_off_template_kwargs", chat, {k: v for k, v in chat_body(chat_template_kwargs={"enable_thinking": False}).items() if k != "reasoning_effort"})
    weather = {"type": "function", "function": {"name": "get_weather", "description": "Weather of a city.",
                                                "parameters": {"type": "object", "properties": WEATHER_PROPERTIES, "required": ["city"]}}}
    ask = [{"role": "user", "content": "What is the weather in Oslo? Use the tool."}]
    post("chat_tools_auto", chat, chat_body(tools=[weather], messages=ask, max_tokens=64))
    post("chat_tools_auto_stream", chat, chat_body(tools=[weather], messages=ask, max_tokens=64, stream=True), stream=True)
    call = {"id": "call_diff1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\":\"Oslo\"}"}}
    post("chat_tool_result_turn", chat, chat_body(tools=[weather], max_tokens=32, messages=[
        *ask, {"role": "assistant", "content": None, "tool_calls": [call]},
        {"role": "tool", "tool_call_id": "call_diff1", "content": "-3C and clear"}]))
    post("chat_tool_choice_none", chat, chat_body(tools=[weather], tool_choice="none", messages=ask, max_tokens=24))
    post("chat_tool_choice_required", chat, chat_body(tools=[weather], tool_choice="required", messages=ask, max_tokens=48), refusal=True)
    post("chat_tool_choice_named", chat, chat_body(
        tools=[weather], tool_choice={"type": "function", "function": {"name": "get_weather"}}, messages=ask, max_tokens=48), refusal=True)
    schema = {"type": "json_schema", "json_schema": {"name": "greeting", "strict": True, "schema": {
        "type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"], "additionalProperties": False}}}
    post("chat_json_schema", chat, chat_body(response_format=schema, max_tokens=48), refusal=True)
    post("chat_json_object", chat, chat_body(response_format={"type": "json_object"}, max_tokens=48), refusal=True)
    post("chat_n_two", chat, chat_body(n=2, temperature=0.8, seed=5), refusal=True)

    # -- chat completions errors
    post("chat_err_no_messages", chat, {"model": model, "max_tokens": 8})
    post("chat_err_empty_messages", chat, {"model": model, "messages": [], "max_tokens": 8})
    post("chat_err_bad_role", chat, chat_body(messages=[{"role": "wizard", "content": "hi"}]))
    post("chat_err_unknown_model", chat, chat_body(model="no-such-model"))
    post("chat_err_no_model", chat, {k: v for k, v in chat_body().items() if k != "model"})
    post("chat_err_temperature_high", chat, chat_body(temperature=3))
    post("chat_err_temperature_type", chat, chat_body(temperature="hot"))
    post("chat_err_top_p_zero", chat, chat_body(top_p=0))
    post("chat_err_max_tokens_zero", chat, chat_body(max_tokens=0))
    post("chat_err_max_tokens_negative", chat, chat_body(max_tokens=-5))
    post("chat_err_stop_too_many", chat, chat_body(stop=["a", "b", "c", "d", "e"]))
    post("chat_err_unknown_field", chat, chat_body(no_such_field=1))
    post("chat_err_reasoning_effort", chat, chat_body(reasoning_effort="extreme"))
    post("chat_err_context_too_long", chat, chat_body(max_tokens=16, messages=[
        {"role": "user", "content": "alpha beta gamma delta " * (context_tokens // 2)}]))
    post("chat_err_context_too_long_stream", chat, chat_body(max_tokens=16, stream=True, messages=[
        {"role": "user", "content": "alpha beta gamma delta " * (context_tokens // 2)}]), stream=True)
    post("chat_err_max_tokens_over_context", chat, chat_body(max_tokens=context_tokens * 4))

    # -- HTTP-level errors
    out.append(_request("http_bad_json", "POST", chat, raw=b'{"model": "x", "messages": [', headers={"content-type": "application/json"}))
    out.append(_request("http_empty_body", "POST", chat, raw=b"", headers={"content-type": "application/json"}))
    out.append(_request("http_json_array", "POST", chat, raw=b"[1, 2]", headers={"content-type": "application/json"}))
    out.append(_request("http_wrong_content_type", "POST", chat, raw=b"model=x", headers={"content-type": "text/plain"}))
    out.append(_request("http_no_content_type", "POST", chat, raw=b"{}"))
    get("http_unknown_route", "/v1/nothing")
    get("http_unknown_root", "/")
    get("http_get_on_post_route", chat)
    post("http_post_on_get_route", "/v1/models", {})
    out.append(_request("http_options_route", "OPTIONS", chat))
    out.append(_request("http_head_health", "HEAD", "/health"))
    out.append(_request("http_body_over_limit", "POST", chat, raw=(
        b'{"model":"x","messages":[{"role":"user","content":"' + b"a" * (body_limit + 4096) + b'"}]}'),
        headers={"content-type": "application/json"}))
    chunked = chat_body()
    out.append(_request("http_chunked_body", "POST", chat, chunked, chunked=True))
    out.append(_request("http_chunked_body_over_limit", "POST", chat, raw=(
        b'{"model":"x","messages":[{"role":"user","content":"' + b"a" * (body_limit + 4096) + b'"}]}'),
        headers={"content-type": "application/json"}, chunked=True))
    post("chat_disconnect_mid_stream", chat, chat_body(stream=True, max_tokens=64), stream=True, disconnect_after=2)
    post("chat_after_disconnect", chat, chat_body())

    # -- Anthropic messages
    messages = "/v1/messages"
    version = {"anthropic-version": "2023-06-01"}
    hi = [{"role": "user", "content": PROMPT}]
    anth = {"model": model, "max_tokens": 24, "temperature": 0, "messages": hi, "thinking": {"type": "disabled"}}

    def anth_body(**extra):
        return {**anth, **extra}

    def amsg(name, body, **fields):
        post(name, messages, body, headers=dict(version), **fields)

    amsg("messages_greedy", anth_body())
    amsg("messages_stream", anth_body(stream=True), stream=True)
    amsg("messages_system", anth_body(system="Answer in French."))
    amsg("messages_system_blocks", anth_body(system=[{"type": "text", "text": "Answer in French."}]))
    amsg("messages_content_blocks", anth_body(messages=[{"role": "user", "content": [{"type": "text", "text": PROMPT}]}]))
    amsg("messages_stop_sequences", anth_body(stop_sequences=[","], max_tokens=32,
                                              messages=[{"role": "user", "content": "Count from one to ten, with commas."}]))
    amsg("messages_stop_sequences_stream", anth_body(stop_sequences=["3"], max_tokens=32, stream=True,
                                                     messages=[{"role": "user", "content": "Count from one to ten, with commas."}]), stream=True)
    amsg("messages_max_tokens_hit", anth_body(max_tokens=3))
    amsg("messages_max_tokens_hit_stream", anth_body(max_tokens=3, stream=True), stream=True)
    amsg("messages_thinking_adaptive", anth_body(thinking={"type": "adaptive"}, max_tokens=48))
    amsg("messages_thinking_adaptive_stream", anth_body(thinking={"type": "adaptive"}, max_tokens=48, stream=True), stream=True)
    amsg("messages_thinking_budget", anth_body(thinking={"type": "enabled", "budget_tokens": 1024}, max_tokens=2048), refusal=True)
    a_weather = {"name": "get_weather", "description": "Weather of a city.",
                 "input_schema": {"type": "object", "properties": WEATHER_PROPERTIES, "required": ["city"]}}
    a_ask = [{"role": "user", "content": "What is the weather in Oslo? Use the tool."}]
    amsg("messages_tools_auto", anth_body(tools=[a_weather], messages=a_ask, max_tokens=64))
    amsg("messages_tools_auto_stream", anth_body(tools=[a_weather], messages=a_ask, max_tokens=64, stream=True), stream=True)
    use = {"type": "tool_use", "id": "toolu_diff1", "name": "get_weather", "input": {"city": "Oslo"}}
    amsg("messages_tool_result_turn", anth_body(tools=[a_weather], max_tokens=32, messages=[
        *a_ask, {"role": "assistant", "content": [use]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_diff1", "content": "-3C and clear"}]}]))
    amsg("messages_tool_choice_any", anth_body(tools=[a_weather], tool_choice={"type": "any"}, messages=a_ask, max_tokens=48), refusal=True)
    amsg("messages_tool_choice_named", anth_body(
        tools=[a_weather], tool_choice={"type": "tool", "name": "get_weather"}, messages=a_ask, max_tokens=48), refusal=True)
    amsg("messages_json_output", anth_body(output_config={"format": {"type": "json_schema", "schema": {
        "type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"], "additionalProperties": False}}},
        max_tokens=48), refusal=True)
    post("messages_count_tokens", "/v1/messages/count_tokens", {"model": model, "messages": hi}, headers=dict(version))
    post("messages_count_tokens_tools", "/v1/messages/count_tokens",
         {"model": model, "messages": a_ask, "tools": [a_weather], "system": "Be brief."}, headers=dict(version))
    amsg("messages_err_no_max_tokens", {"model": model, "messages": hi})
    amsg("messages_err_no_messages", {"model": model, "max_tokens": 8})
    amsg("messages_err_bad_role", anth_body(messages=[{"role": "system", "content": "hi"}]))
    amsg("messages_err_unknown_model", anth_body(model="no-such-model"))
    amsg("messages_err_temperature_high", anth_body(temperature=3))
    amsg("messages_err_max_tokens_zero", anth_body(max_tokens=0))
    amsg("messages_err_context_too_long", anth_body(max_tokens=16, messages=[
        {"role": "user", "content": "alpha beta gamma delta " * (context_tokens // 2)}]))
    out.append(_request("messages_err_bad_json", "POST", messages, raw=b"{", headers={**version, "content-type": "application/json"}))
    post("messages_no_version_header", messages, anth_body())

    # -- Responses
    responses = "/v1/responses"
    resp = {"model": model, "input": PROMPT, "temperature": 0, "max_output_tokens": 24, "reasoning": {"effort": "none"}}

    def resp_body(**extra):
        return {**resp, **extra}

    post("responses_greedy", responses, resp_body())
    post("responses_stream", responses, resp_body(stream=True), stream=True)
    post("responses_instructions", responses, resp_body(instructions="Answer in French."))
    post("responses_input_items", responses, resp_body(input=[
        {"role": "user", "content": [{"type": "input_text", "text": PROMPT}]}]))
    post("responses_input_messages", responses, resp_body(input=[{"role": "user", "content": PROMPT}]))
    post("responses_max_tokens_hit", responses, resp_body(max_output_tokens=3))
    post("responses_max_tokens_hit_stream", responses, resp_body(max_output_tokens=3, stream=True), stream=True)
    post("responses_reasoning_on", responses, resp_body(reasoning={"effort": "low"}, max_output_tokens=48))
    post("responses_reasoning_on_stream", responses, resp_body(reasoning={"effort": "low"}, max_output_tokens=48, stream=True), stream=True)
    r_weather = {"type": "function", "name": "get_weather", "description": "Weather of a city.",
                 "parameters": {"type": "object", "properties": WEATHER_PROPERTIES, "required": ["city"]}}
    post("responses_tools_auto", responses, resp_body(tools=[r_weather], input="What is the weather in Oslo? Use the tool.", max_output_tokens=64))
    post("responses_tools_auto_stream", responses, resp_body(
        tools=[r_weather], input="What is the weather in Oslo? Use the tool.", max_output_tokens=64, stream=True), stream=True)
    post("responses_tool_result_turn", responses, resp_body(tools=[r_weather], max_output_tokens=32, input=[
        {"role": "user", "content": "What is the weather in Oslo? Use the tool."},
        {"type": "function_call", "call_id": "call_diff1", "name": "get_weather", "arguments": "{\"city\":\"Oslo\"}"},
        {"type": "function_call_output", "call_id": "call_diff1", "output": "-3C and clear"}]))
    post("responses_tool_choice_required", responses, resp_body(tools=[r_weather], tool_choice="required", max_output_tokens=48), refusal=True)
    post("responses_json_schema", responses, resp_body(text={"format": {
        "type": "json_schema", "name": "greeting", "strict": True,
        "schema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"], "additionalProperties": False}}},
        max_output_tokens=48), refusal=True)
    post("responses_err_no_input", responses, {"model": model, "max_output_tokens": 8})
    post("responses_err_unknown_model", responses, resp_body(model="no-such-model"))
    post("responses_err_unknown_previous", responses, resp_body(previous_response_id="resp_does_not_exist"))
    post("responses_err_temperature_high", responses, resp_body(temperature=3))
    post("responses_err_max_tokens_zero", responses, resp_body(max_output_tokens=0))
    post("responses_err_context_too_long", responses, resp_body(
        input="alpha beta gamma delta " * (context_tokens // 2), max_output_tokens=16))
    out.append(_request("responses_err_bad_json", "POST", responses, raw=b"{", headers={"content-type": "application/json"}))
    return out
