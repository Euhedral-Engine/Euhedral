"""Prompts for the MTP drafter's training corpus (`quantization generate`): chat, code, tool calls, answers after tool
results, coding-agent turns over real repositories, multilingual and math prompts, from training splits.

Usage (in tools/): python mtp_corpus_prompts.py OUT.jsonl --t4 T4_PROMPTS.jsonl --exclude-store STORE ...
    [--cache DIR] [--scale 1.0]

Each line is `{"id", "domain", "messages", "tools"?}` (OpenAI messages; assistant turns of a history may carry
`tool_calls`). The answer the model gives is the training text, so a prompt ends where the model's turn begins:

- chat: an UltraChat 200k `train_sft` first user turn; chat-turn: its first exchange and second user turn;
- code: a Magicoder-OSS-Instruct-75K problem;
- tool-call: a Hermes function-calling (`func_calling`) first user turn with its tools;
- tool-result: the same conversation through the dataset's own tool calls and tool responses;
- agent: a coding-agent task over a pallets project at a pinned tag, after the agent listed the package and read one
  file (`list_directory`, `read_file`);
- multilingual: an Aya `train` prompt in Spanish, Chinese, Japanese, Arabic or Hindi;
- math: a GSM8K `train` question.

No prompt shares a user message with T4 or with the stores named by `--exclude-store` (the held-out set, the
calibration set). The drafting measurement prompts (tools/mtp_draft_prompts.py) are written from this repository's own
documents and sources, which no domain here reads.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from gptq import conversations as source

SEED = 20261019
COUNTS = {"chat": 700, "chat-turn": 300, "code": 800, "tool-call": 450, "tool-result": 450, "agent": 500,
          "multilingual": 400, "math": 400}
PROJECTS = [("pallets/itsdangerous", "2.2.0"), ("pallets/markupsafe", "3.0.2"), ("pallets/click", "8.1.7"),
            ("pallets/flask", "3.0.3"), ("pallets/jinja", "3.1.4"), ("pallets/werkzeug", "3.0.3")]
MAX_FILE_BYTES = 8000  # one read file stays well inside the generator's 4096-token prompt


def user(text: str) -> dict:
    return {"role": "user", "content": text}


def call(identifier: str, name: str, arguments: dict) -> dict:
    return {"role": "assistant", "content": "",
            "tool_calls": [{"id": identifier, "type": "function", "function": {"name": name, "arguments": arguments}}]}


def result(identifier: str, content: str) -> dict:
    return {"role": "tool", "tool_call_id": identifier, "content": content}


def chat(rng, count, turns, excluded):
    single, turn = [], []
    for row in source.rows("HuggingFaceH4/ultrachat_200k", "default", "train_sft", rng, 2 * (count + turns)):
        messages = row["messages"]
        first = messages[0]["content"].strip()
        if first in excluded or len(first) > 3000:
            continue
        if len(turn) < turns and len(messages) >= 3 and messages[1]["role"] == "assistant":
            second = messages[2]["content"].strip()
            if second not in excluded and len(messages[1]["content"]) < 6000:
                turn.append({"id": f"ultrachat-turn-{row['prompt_id'][:12]}", "domain": "chat-turn", "messages": [
                    user(first), {"role": "assistant", "content": messages[1]["content"].strip()}, user(second)]})
                continue
        if len(single) < count:
            single.append({"id": f"ultrachat-{row['prompt_id'][:12]}", "domain": "chat", "messages": [user(first)]})
        if len(single) == count and len(turn) == turns:
            break
    return single + turn


def code(rng, count, excluded):
    out = []
    for index, row in enumerate(source.rows("ise-uiuc/Magicoder-OSS-Instruct-75K", "default", "train", rng, 2 * count)):
        problem = row["problem"].strip()
        if problem in excluded or len(problem) > 3000:
            continue
        out.append({"id": f"magicoder-{source.hashlib.sha256(problem.encode()).hexdigest()[:12]}", "domain": "code",
                    "messages": [user(problem)]})
        if len(out) == count:
            break
    return out


def tools(rng, count, excluded):
    calls, results = [], []
    for row in source.rows("NousResearch/hermes-function-calling-v1", "func_calling", "train", rng, 3 * count):
        declared = json.loads(row["tools"]) if isinstance(row["tools"], str) else row["tools"]
        if not declared:
            continue
        turns = row["conversations"]
        humans = [t for t in turns if t["from"] == "human"]
        if not humans or humans[0]["value"].strip() in excluded:
            continue
        first = humans[0]["value"].strip()
        if len(calls) < count:
            calls.append({"id": f"hermes-call-{row['id']}", "domain": "tool-call", "messages": [user(first)],
                          "tools": declared})
        # The dataset's first tool calls and their responses, then the model's answer.
        try:
            gpt = next(t for t in turns if t["from"] == "gpt")
            response = next(t for t in turns if t["from"] == "tool")
            made = [json.loads(c) for c in source.TOOL_CALL.findall(gpt["value"])]
            answered = [json.loads(r) for r in source.TOOL_RESPONSE.findall(response["value"])]
        except (StopIteration, ValueError):
            made, answered = [], []
        if len(results) < count and made and len(made) == len(answered):
            ids = [f"call_{index}" for index in range(len(made))]
            messages = [user(first), {"role": "assistant", "content": "", "tool_calls": [
                {"id": i, "type": "function", "function": {"name": c["name"], "arguments": c.get("arguments") or {}}}
                for i, c in zip(ids, made)]}]
            messages += [result(i, json.dumps(r.get("content", r), ensure_ascii=False)) for i, r in zip(ids, answered)]
            results.append({"id": f"hermes-result-{row['id']}", "domain": "tool-result", "messages": messages,
                            "tools": declared})
        if len(calls) == count and len(results) == count:
            break
    return calls + results


def agents(rng, count):
    source.MAX_FILE_BYTES = MAX_FILE_BYTES
    projects = [(repo, tag, source.project_files(repo, tag)) for repo, tag in PROJECTS]
    pairs = [(task, repo, tag, files, path) for task in source.AGENT_TASKS for repo, tag, files in projects
             for path in files if path.startswith("src/") and path.endswith(".py")]
    rng.shuffle(pairs)
    out = []
    for task, repo, tag, files, path in pairs[:count]:
        package = path.rsplit("/", 1)[0]
        inside = [p[len(package) + 1:] for p in files if p.startswith(package + "/")]
        listing = sorted({p.split("/")[0] + ("/" if "/" in p else "") for p in inside})
        name = f"{repo.split('/')[1]}-{tag}-{path}"
        out.append({"id": f"agent-{source.hashlib.sha256((task[0] + name).encode()).hexdigest()[:12]}",
                    "domain": "agent", "tools": source.FILE_TOOLS, "messages": [
                        {"role": "system", "content": source.AGENT_SYSTEM}, user(task[0]),
                        call("call_0", "list_directory", {"path": package}), result("call_0", "\n".join(listing)),
                        call("call_1", "read_file", {"path": path}), result("call_1", files[path])]})
    return out


def multilingual(rng, count, excluded):
    rows = source.multilingual_domain(rng, count + len(source.CODE_SWITCHING), excluded, written=[])
    return [{"id": f"aya-{source.hashlib.sha256(r['turns'][0].encode()).hexdigest()[:12]}", "domain": "multilingual",
             "messages": [user(r["turns"][0])]} for r in rows[:count]]


def math(rng, count, excluded):
    out = []
    for row in source.rows("openai/gsm8k", "main", "train", rng, 2 * count):
        question = row["question"].strip()
        if question in excluded:
            continue
        out.append({"id": f"gsm8k-{source.hashlib.sha256(question.encode()).hexdigest()[:12]}", "domain": "math",
                    "messages": [user(question)]})
        if len(out) == count:
            break
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", type=Path)
    parser.add_argument("--t4", type=Path, required=True)
    parser.add_argument("--exclude-store", type=Path, action="append", default=[])
    parser.add_argument("--cache", type=Path, help="fetched dataset pages, so a rerun reads the same rows")
    parser.add_argument("--scale", type=float, default=1.0, help="multiplies every domain's count")
    args = parser.parse_args(argv)
    if args.cache:
        args.cache.mkdir(parents=True, exist_ok=True)
        source.CACHE = args.cache
    excluded = source.held_out_texts(args.t4)
    for store in args.exclude_store:
        excluded |= source.store_texts(store)
    counts = {domain: max(1, round(n * args.scale)) for domain, n in COUNTS.items()}
    rng = random.Random(SEED)
    chosen = (chat(rng, counts["chat"], counts["chat-turn"], excluded) + code(rng, counts["code"], excluded)
              + tools(rng, counts["tool-call"], excluded) + agents(rng, counts["agent"])
              + multilingual(rng, counts["multilingual"], excluded) + math(rng, counts["math"], excluded))
    ids = set()
    with args.out.open("w", encoding="utf-8") as handle:
        for prompt in chosen:
            if prompt["id"] in ids:
                continue
            ids.add(prompt["id"])
            handle.write(json.dumps(prompt, ensure_ascii=False) + "\n")
    by_domain = {}
    for prompt in chosen:
        by_domain[prompt["domain"]] = by_domain.get(prompt["domain"], 0) + 1
    print(f"{len(ids)} prompts: {by_domain}; {len(excluded)} excluded user messages")
    return 0


if __name__ == "__main__":
    sys.exit(main())
