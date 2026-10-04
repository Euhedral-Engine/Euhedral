#!/usr/bin/env python3
"""Runs real clients against a running server and prints a compatibility matrix.

Needs the `openai` and `anthropic` packages and `curl`. Start the server with
EUHEDRAL_API_MAX_QUEUED_GENERATIONS=1 so the capacity scenario can fill the queue:

    python3 tools/client_compatibility.py --base http://localhost:1738 --model qwen

Each scenario checks what a client relies on, not exact model text. The output is a Markdown table.
"""

import argparse
import json
import subprocess
import sys
import threading
import time
import traceback
import urllib.request

import anthropic
import openai

NO_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}
WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}
WEATHER_ANTHROPIC = {
    "name": "get_weather",
    "description": "Current weather for a city.",
    "input_schema": WEATHER["function"]["parameters"],
}


class Matrix:
    def __init__(self):
        self.rows = []

    def run(self, scenario, client, function):
        started = time.monotonic()
        try:
            detail = function()
            outcome = "pass"
        except Exception as failure:  # noqa: BLE001 - every failure is a row of the matrix
            detail = f"{type(failure).__name__}: {failure}"
            outcome = "FAIL"
            traceback.print_exc(file=sys.stderr)
        seconds = time.monotonic() - started
        self.rows.append((scenario, client, outcome, f"{seconds:.1f} s", str(detail).replace("|", "\\|").replace("\n", " ")))
        print(f"{outcome:4} {scenario} ({client}): {detail}", file=sys.stderr, flush=True)

    def markdown(self):
        lines = ["| Scenario | Client | Result | Time | Observed |", "|---|---|---|---|---|"]
        lines += [f"| {' | '.join(row)} |" for row in self.rows]
        return "\n".join(lines)


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def metric(base, name, labels=""):
    text = urllib.request.urlopen(base + "/metrics", timeout=10).read().decode()
    total = 0.0
    for line in text.splitlines():
        if line.startswith(name + "{") and labels in line:
            total += float(line.rsplit(" ", 1)[1])
    return total


def long_document(words):
    sentences = [
        "The survey team recorded the river's depth at each marker before noon.",
        "Every station logged temperature, wind and the colour of the water.",
        "Later the samples were sealed, labelled and sent to the laboratory.",
        "The laboratory compared them with the readings of the previous year.",
    ]
    text = []
    while len(text) < words:
        for index, sentence in enumerate(sentences):
            text.append(f"Entry {len(text)}: {sentence}")
    return " ".join(text[:words])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://localhost:1738")
    parser.add_argument("--model", default="qwen")
    args = parser.parse_args()
    base, model = args.base, args.model
    oai = openai.OpenAI(base_url=base + "/v1", api_key="unused", max_retries=0, timeout=600)
    claude = anthropic.Anthropic(base_url=base, api_key="unused", max_retries=0, timeout=600)
    matrix = Matrix()

    def chat_json():
        reply = oai.chat.completions.create(
            model=model, messages=[{"role": "user", "content": "Name three primary colours."}],
            max_tokens=100, temperature=0, extra_body=NO_THINKING)
        text = reply.choices[0].message.content
        check(text and reply.choices[0].finish_reason == "stop", reply)
        return f"finish=stop, {reply.usage.completion_tokens} tokens: {text[:60]!r}"

    matrix.run("Chat, JSON response", "openai", chat_json)

    def chat_stream():
        reasoning, content, usage = [], [], None
        stream = oai.chat.completions.create(
            model=model, messages=[{"role": "user", "content": "What is 17 * 23? Answer with the number."}],
            max_tokens=2000, stream=True, stream_options={"include_usage": True}, reasoning_effort="low")
        for chunk in stream:
            if chunk.usage:
                usage = chunk.usage
            for choice in chunk.choices:
                delta = choice.delta
                if getattr(delta, "reasoning_content", None):
                    check(not content, "reasoning after content")
                    reasoning.append(delta.reasoning_content)
                if delta.content:
                    content.append(delta.content)
        answer = "".join(content)
        check("391" in answer, answer)
        check(reasoning and usage and usage.completion_tokens_details.reasoning_tokens > 0, usage)
        return (f"{len(reasoning)} reasoning chunks before {len(content)} content chunks; "
                f"reasoning_tokens={usage.completion_tokens_details.reasoning_tokens}; answer {answer.strip()!r}")

    matrix.run("Chat, streaming with reasoning on (low)", "openai", chat_stream)

    def reasoning_off():
        reply = oai.chat.completions.create(
            model=model, messages=[{"role": "user", "content": "What is 17 * 23? Answer with the number."}],
            max_tokens=50, temperature=0, reasoning_effort="none")
        message = reply.choices[0].message
        check(getattr(message, "reasoning_content", None) is None, message)
        check("391" in message.content, message.content)
        return f"no reasoning_content; answer {message.content.strip()!r}"

    matrix.run("Chat, reasoning off", "openai", reasoning_off)

    def raw_sse():
        body = json.dumps({"model": model, "messages": [{"role": "user", "content": "Say hello."}],
                           "max_tokens": 20, "stream": True, **NO_THINKING})
        output = subprocess.run(
            ["curl", "-sN", base + "/v1/chat/completions", "-H", "Content-Type: application/json", "-d", body],
            capture_output=True, text=True, timeout=120, check=True).stdout
        events = [line for line in output.splitlines() if line.startswith("data:")]
        check(events and events[-1].strip() == "data: [DONE]" or events[-1].replace(" ", "") == "data:[DONE]", events[-3:])
        chunks = [json.loads(line[5:]) for line in events[:-1]]
        check(all(chunk["object"] == "chat.completion.chunk" for chunk in chunks), chunks[0])
        return f"{len(chunks)} chat.completion.chunk events, then [DONE]"

    matrix.run("Chat, raw SSE", "curl", raw_sse)

    def structured():
        schema = {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"},
                                                   "languages": {"type": "array", "items": {"type": "string"}}},
                  "required": ["name", "age", "languages"], "additionalProperties": False}
        reply = oai.chat.completions.create(
            model=model, max_tokens=300, temperature=0.7, extra_body=NO_THINKING,
            messages=[{"role": "user", "content": "Invent a programmer. Reply as JSON with name, age and languages."}],
            response_format={"type": "json_schema", "json_schema": {"name": "person", "strict": True, "schema": schema}})
        document = json.loads(reply.choices[0].message.content)
        check(set(document) == {"name", "age", "languages"} and isinstance(document["age"], int), document)
        return f"valid document: {json.dumps(document)[:80]}"

    matrix.run("Structured output, json_schema (sampled)", "openai", structured)

    def single_call():
        reply = oai.chat.completions.create(
            model=model, max_tokens=200, temperature=0, tools=[WEATHER], parallel_tool_calls=False, extra_body=NO_THINKING,
            messages=[{"role": "user", "content": "What is the weather in Paris and in Rome?"}])
        calls = reply.choices[0].message.tool_calls
        check(reply.choices[0].finish_reason == "tool_calls" and len(calls) == 1, reply)
        return f"1 call (parallel_tool_calls=false): {calls[0].function.name}({calls[0].function.arguments})"

    matrix.run("Tool call, single", "openai", single_call)

    def agent_loop():
        messages = [{"role": "user", "content": "What is the weather in Paris and in Rome? Use the tool for each."}]
        reply = oai.chat.completions.create(model=model, max_tokens=300, temperature=0, tools=[WEATHER],
                                            messages=messages, extra_body=NO_THINKING)
        calls = reply.choices[0].message.tool_calls
        check(len(calls) == 2, reply)
        cities = sorted(json.loads(call.function.arguments)["city"] for call in calls)
        messages.append(reply.choices[0].message.model_dump(exclude_none=True))
        weather = {"Paris": "18 C and raining", "Rome": "27 C and sunny"}
        for call in calls:
            city = json.loads(call.function.arguments)["city"]
            messages.append({"role": "tool", "tool_call_id": call.id, "content": weather.get(city, "unknown")})
        final = oai.chat.completions.create(model=model, max_tokens=300, temperature=0, tools=[WEATHER],
                                            messages=messages, extra_body=NO_THINKING)
        answer = final.choices[0].message.content
        check(final.choices[0].finish_reason == "stop" and "18" in answer and "27" in answer, answer)
        return f"parallel calls for {cities}; after the results: {answer[:70]!r}"

    matrix.run("Tool calls, parallel, and tool-result continuation", "openai", agent_loop)

    def responses_api():
        reply = oai.responses.create(model=model, input="What is the capital of Japan? One word.",
                                     max_output_tokens=500, reasoning={"effort": "low"})
        kinds = [item.type for item in reply.output]
        check(reply.status == "completed" and "Tokyo" in reply.output_text, reply)
        events = [event.type for event in oai.responses.create(
            model=model, input="Say hi.", max_output_tokens=50, reasoning={"effort": "none"}, stream=True)]
        check(events[0] == "response.created" and events[-1] == "response.completed", events)
        return f"output {kinds}, text {reply.output_text.strip()!r}; stream of {len(events)} events"

    matrix.run("Responses, JSON and streaming", "openai", responses_api)

    def responses_tools():
        tool = {"type": "function", "name": "get_weather", "description": "Current weather for a city.",
                "parameters": WEATHER["function"]["parameters"]}
        first = oai.responses.create(model=model, input="What is the weather in Oslo?", tools=[tool],
                                     max_output_tokens=300, reasoning={"effort": "none"}, temperature=0)
        calls = [item for item in first.output if item.type == "function_call"]
        check(len(calls) == 1, first.output)
        second = oai.responses.create(
            model=model, tools=[tool], max_output_tokens=300, reasoning={"effort": "none"}, temperature=0,
            input=[{"role": "user", "content": "What is the weather in Oslo?"}, calls[0].model_dump(exclude_none=True),
                   {"type": "function_call_output", "call_id": calls[0].call_id, "output": "-3 C and snowing"}])
        check("-3" in second.output_text or "snow" in second.output_text.lower(), second.output_text)
        return f"function_call {calls[0].arguments}, then {second.output_text[:60]!r}"

    matrix.run("Responses, tool call and function_call_output", "openai", responses_tools)

    def messages_text():
        message = claude.messages.create(model=model, max_tokens=300,
                                         messages=[{"role": "user", "content": "Name a planet with rings. One word."}])
        check(message.stop_reason == "end_turn" and message.content[0].type == "text", message)
        text_events = 0
        with claude.messages.stream(model=model, max_tokens=50,
                                    messages=[{"role": "user", "content": "Count to five."}]) as stream:
            for _ in stream.text_stream:
                text_events += 1
            final = stream.get_final_message()
        check(final.stop_reason == "end_turn", final)
        return f"{message.content[0].text.strip()[:50]!r}; stream of {text_events} text deltas"

    matrix.run("Messages, JSON and streaming", "anthropic", messages_text)

    def messages_thinking():
        message = claude.messages.create(model=model, max_tokens=3000, thinking={"type": "enabled", "budget_tokens": 1500},
                                         messages=[{"role": "user", "content": "What is 12 * 34?"}])
        kinds = [block.type for block in message.content]
        check(kinds[0] == "thinking" and kinds[-1] == "text" and "408" in message.content[-1].text, message)
        return f"blocks {kinds}; answer {message.content[-1].text.strip()[:40]!r}"

    matrix.run("Messages, thinking enabled", "anthropic", messages_thinking)

    def messages_tools():
        question = {"role": "user", "content": "What is the weather in Lima?"}
        first = claude.messages.create(model=model, max_tokens=300, tools=[WEATHER_ANTHROPIC], messages=[question])
        uses = [block for block in first.content if block.type == "tool_use"]
        check(first.stop_reason == "tool_use" and len(uses) == 1, first)
        second = claude.messages.create(
            model=model, max_tokens=300, tools=[WEATHER_ANTHROPIC],
            messages=[question, {"role": "assistant", "content": first.content},
                      {"role": "user", "content": [{"type": "tool_result", "tool_use_id": uses[0].id,
                                                    "content": "21 C and cloudy"}]}])
        text = second.content[-1].text
        check(second.stop_reason == "end_turn" and "21" in text, second)
        return f"tool_use {uses[0].input}, then {text[:60]!r}"

    matrix.run("Messages, tool_use and tool_result", "anthropic", messages_tools)

    def prefix_cache():
        document = long_document(1200)
        messages = [{"role": "system", "content": "You answer questions about the survey log."},
                    {"role": "user", "content": document + "\n\nHow many entries mention the laboratory? Estimate."}]
        first = oai.chat.completions.create(model=model, messages=messages, max_tokens=40, temperature=0,
                                            extra_body=NO_THINKING)
        messages += [first.choices[0].message.model_dump(exclude_none=True),
                     {"role": "user", "content": "And which entry comes first?"}]
        started = time.monotonic()
        second = oai.chat.completions.create(model=model, messages=messages, max_tokens=40, temperature=0,
                                             extra_body=NO_THINKING)
        follow_up = time.monotonic() - started
        cached = second.usage.prompt_tokens_details.cached_tokens
        check(cached >= 4096, second.usage)
        message = claude.messages.create(model=model, max_tokens=40, system=messages[0]["content"],
                                         messages=messages[1:2] + [{"role": "assistant", "content": first.choices[0].message.content},
                                                                    {"role": "user", "content": "And which entry comes first?"}])
        check(message.usage.cache_read_input_tokens >= 4096, message.usage)
        return (f"prompt {first.usage.prompt_tokens} tokens; follow-up restored {cached} of "
                f"{second.usage.prompt_tokens} ({follow_up:.2f} s); the same conversation through Messages restored "
                f"{message.usage.cache_read_input_tokens}")

    matrix.run("Long conversation, prefix cache (and across APIs)", "openai + anthropic", prefix_cache)

    def disconnect():
        before = metric(base, "euhedral_requests_total", 'outcome="cancelled"')
        stream = oai.chat.completions.create(
            model=model, messages=[{"role": "user", "content": "Count from 1 to 2000, one number per line."}],
            max_tokens=6000, temperature=0, stream=True, extra_body=NO_THINKING)
        received = 0
        for _ in stream:
            received += 1
            if received == 20:
                break
        stream.close()
        started = time.monotonic()
        reply = oai.chat.completions.create(model=model, messages=[{"role": "user", "content": "Say OK."}],
                                            max_tokens=5, temperature=0, extra_body=NO_THINKING)
        next_request = time.monotonic() - started
        time.sleep(0.5)
        cancelled = metric(base, "euhedral_requests_total", 'outcome="cancelled"') - before
        check(cancelled == 1 and reply.choices[0].message.content, (cancelled, reply))
        return f"closed after 20 chunks; next request answered in {next_request:.2f} s; cancelled +1"

    matrix.run("Disconnect mid-stream", "openai", disconnect)

    def capacity():
        long_body = {"model": model, "messages": [{"role": "user", "content": "Count from 1 to 3000, one per line."}],
                     "max_tokens": 8000, "temperature": 0, "extra_body": NO_THINKING}
        holders = []

        def hold():
            try:
                holders.append(oai.chat.completions.create(**long_body, timeout=600))
            except Exception as error:  # noqa: BLE001 - reported by the check below
                holders.append(error)

        threads = [threading.Thread(target=hold) for _ in range(2)]
        threads[0].start()
        time.sleep(1.5)
        threads[1].start()
        time.sleep(1.5)
        try:
            oai.chat.completions.create(model=model, messages=[{"role": "user", "content": "Hi"}], max_tokens=5,
                                        extra_body=NO_THINKING)
            refused = None
        except openai.APIStatusError as error:
            refused = error
        try:
            claude.messages.create(model=model, max_tokens=5, messages=[{"role": "user", "content": "Hi"}])
            anthropic_refused = None
        except anthropic.APIStatusError as error:
            anthropic_refused = error
        for thread in threads:
            thread.join()
        check(refused is not None and refused.status_code == 503, refused)
        check(isinstance(anthropic_refused, anthropic.OverloadedError), anthropic_refused)
        check(all(not isinstance(holder, Exception) for holder in holders), holders)
        return (f"running and queued requests answered; a third refused: openai {refused.status_code} "
                f"{refused.type}, "
                f"anthropic {anthropic_refused.status_code} {type(anthropic_refused).__name__}")

    matrix.run("Capacity rejection (queue of one)", "openai + anthropic", capacity)

    print(matrix.markdown())
    return 0 if all(row[2] == "pass" for row in matrix.rows) else 1


if __name__ == "__main__":
    sys.exit(main())
