#!/usr/bin/env python3
"""Generate the T4 documents: the reference model's own answers, thinking included (docs/QUALITY.md, holdout T4).

Usage: t4_generate.py SERVER_URL PROMPTS.jsonl OUT.jsonl [--target-tokens N] [--parallel N] [--max-tokens N]

SERVER_URL is a llama.cpp `llama-server` running the reference checkpoint with --jinja. Each prompt is rendered with
the model's own chat template (`/apply-template`, tools included), completed greedily (`/completion`, temperature 0)
and stored as one document: the rendered prompt followed by the raw completion, template and thinking tokens
included, so that the scored text is what the deployed model reads and writes. New prompts stop being sent once the
documents written reach --target-tokens tokens; prompts already in OUT.jsonl are skipped, so a stopped run resumes.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import threading
import time
import urllib.request


def post(url: str, body: dict, timeout: float) -> dict:
    request = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("server")
    parser.add_argument("prompts")
    parser.add_argument("out")
    parser.add_argument("--target-tokens", type=int, default=70000)
    parser.add_argument("--parallel", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=2048)
    args = parser.parse_args()
    server = args.server.rstrip("/")
    prompts = [json.loads(line) for line in open(args.prompts, encoding="utf-8")]
    done, written = set(), 0
    try:
        for line in open(args.out, encoding="utf-8"):
            record = json.loads(line)
            done.add(record["id"])
            written += record["tokens"]
    except FileNotFoundError:
        pass
    lock, started = threading.Lock(), time.time()
    state = {"written": written, "generated": 0}

    def generate(prompt: dict) -> None:
        with lock:
            if state["written"] >= args.target_tokens:
                return
        body = {"messages": prompt["messages"]}
        if prompt.get("tools"):
            body["tools"] = prompt["tools"]
        rendered = post(f"{server}/apply-template", body, 300)["prompt"]
        result = post(f"{server}/completion", {"prompt": rendered, "n_predict": args.max_tokens, "temperature": 0,
                                               "cache_prompt": False}, 86400)
        tokens = int(result["tokens_evaluated"]) + int(result["tokens_predicted"])
        record = {"id": prompt["id"], "source": prompt["source"], "tokens": tokens,
                  "completion_tokens": int(result["tokens_predicted"]), "stopped": result.get("stop_type"),
                  "text": rendered + result["content"]}
        with lock:
            with open(args.out, "a", encoding="utf-8") as out:
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
            state["written"] += tokens
            state["generated"] += record["completion_tokens"]
            rate = state["generated"] / (time.time() - started)
            print(f"{prompt['id']}: {tokens} tokens ({record['completion_tokens']} generated, {record['stopped']}); "
                  f"total {state['written']}/{args.target_tokens}, {rate:.1f} generated tok/s", flush=True)

    pending = [p for p in prompts if p["id"] not in done]
    with concurrent.futures.ThreadPoolExecutor(args.parallel) as pool:
        for future in [pool.submit(generate, prompt) for prompt in pending]:
            future.result()
    print(f"done: {state['written']} tokens in {sum(1 for _ in open(args.out, encoding='utf-8'))} documents")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
