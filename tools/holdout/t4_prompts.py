#!/usr/bin/env python3
"""Build the T4 prompt set: a fixed sample of public training-split prompts (docs/QUALITY.md, holdout T4).

Usage: t4_prompts.py OUT.jsonl

Draws, with a fixed seed from whole pages of rows, 60 grade-school math questions (openai/gsm8k train, MIT), 50 programming tasks
(google-research-datasets/mbpp train, CC-BY-4.0) and 40 single-turn function-calling requests with their tools
(NousResearch/hermes-function-calling-v1, func_calling_singleturn, Apache-2.0), interleaved, one JSON object per
line: id, source, messages and, for function calling, tools. Training splits only, so the GSM8K test set stays free
for downstream gates. Rows come from the Hugging Face datasets-server API.
"""
from __future__ import annotations

import json
import random
import sys
import time
import urllib.parse
import urllib.error
import urllib.request

ROWS = "https://datasets-server.huggingface.co/rows"
SEED = 20261011


def page(dataset: str, config: str, split: str, offset: int) -> list[dict]:
    query = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": split, "offset": offset, "length": 100})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(f"{ROWS}?{query}", timeout=60) as response:
                return [entry["row"] for entry in json.load(response)["rows"]]
        except (urllib.error.URLError, TimeoutError):
            if attempt == 5:
                raise
            time.sleep(5 * 2 ** attempt)
    return []


def rows(dataset: str, config: str, split: str, total: int, count: int, rng: random.Random) -> list[dict]:
    """`count` rows drawn without replacement from whole pages of 100 at random offsets (few requests: the
    datasets-server rate-limits single-row fetches)."""
    offsets = sorted(rng.sample(range(0, total, 100), min(len(range(0, total, 100)), (count + 49) // 50)))
    pool = [row for offset in offsets for row in page(dataset, config, split, offset)]
    return rng.sample(pool, count)


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    rng = random.Random(SEED)
    math = [{"id": f"gsm8k-train-{i}", "source": "gsm8k",
             "messages": [{"role": "user", "content": row["question"]}]}
            for i, row in enumerate(rows("openai/gsm8k", "main", "train", 7473, 60, rng))]
    code = [{"id": f"mbpp-train-{row['task_id']}", "source": "mbpp",
             "messages": [{"role": "user", "content": f"{row['text']}\nYour code should pass these tests:\n"
                           + "\n".join(row["test_list"])}]}
            for row in rows("google-research-datasets/mbpp", "full", "train", 374, 50, rng)]
    tools = []
    for row in rows("NousResearch/hermes-function-calling-v1", "func_calling_singleturn", "train", 1893, 40, rng):
        human = next(turn["value"] for turn in row["conversations"] if turn["from"] == "human")
        declared = json.loads(row["tools"]) if isinstance(row["tools"], str) else row["tools"]
        tools.append({"id": f"hermes-fc-{row['id']}", "source": "function-calling",
                      "messages": [{"role": "user", "content": human}], "tools": declared})
    prompts = []
    for i in range(max(len(math), len(code), len(tools))):
        for group in (math, code, tools):
            if i < len(group):
                prompts.append(group[i])
    with open(argv[0], "w", encoding="utf-8") as out:
        for prompt in prompts:
            out.write(json.dumps(prompt, ensure_ascii=False) + "\n")
    print(f"{len(prompts)} prompts: {len(math)} math, {len(code)} code, {len(tools)} function calling")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
