"""Build the GPTQ calibration set: token sequences of the model's own answers, prose and code, disjoint from the holdout.

Usage (in tools/, with `tokenizers` and `jinja2` importable):
  python -m gptq.calibration prompts OUT/prompts.jsonl T4_PROMPTS.jsonl [--count N]
  python -m gptq.calibration generate SERVER_URL OUT/prompts.jsonl OUT/generations.jsonl [--parallel N] [--max-tokens N]
  python -m gptq.calibration tokens CHECKPOINT OUT [--length 2048] [--generated N] [--prose N] [--code N]

`prompts` samples training-split prompts (GSM8K, MBPP, Hermes function calling; tools/holdout/t4_prompts.py) with a
seed of its own, leaving out every T4 prompt. `generate` asks an OpenAI-compatible server (the engine serving
Swift-1.5) for greedy answers with thinking, several requests at a time, resuming a stopped run. `tokens` renders each
answered prompt with the checkpoint's chat template (prompt, thinking, answer), tokenizes it with the checkpoint's
tokenizer (the same ids llama.cpp produces), adds wikitext-2 train and source files other than T3's, and writes
`tokens.npy` (int32 [sequences, length]): each kind of text concatenated and cut into windows of `length` tokens.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import io
import json
import random
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "holdout"))
import t4_prompts  # noqa: E402

SEED = 20261012
WIKITEXT = "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip"
RAW = "https://raw.githubusercontent.com"
# Same projects and tags as T3 (tools/holdout/texts.py), other files.
CODE = [
    ("python/cpython", "v3.12.0", ["Lib/argparse.py", "Lib/collections/__init__.py", "Lib/typing.py", "Lib/pathlib.py",
                                   "Lib/functools.py", "Lib/dataclasses.py", "Lib/enum.py", "Lib/inspect.py"]),
    ("golang/go", "go1.22.0", ["src/net/http/server.go", "src/strings/strings.go", "src/sort/sort.go",
                               "src/bufio/bufio.go", "src/fmt/print.go"]),
    ("serde-rs/json", "v1.0.120", ["src/value/mod.rs", "src/value/de.rs", "src/map.rs", "src/number.rs"]),
    ("DaveGamble/cJSON", "v1.7.18", ["tests/parse_object.c", "tests/print_value.c", "tests/misc_tests.c"]),
]


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()


def held_out_prompts(path: Path) -> set[str]:
    """The user messages of the T4 prompts (their GSM8K ids are positions, so prompts are compared by text)."""
    return {json.loads(line)["messages"][0]["content"] for line in path.open()}


def prompts(out: Path, count: int, t4: Path) -> None:
    rng = random.Random(SEED)
    held_out = held_out_prompts(t4)
    per = count // 3 + 1
    math = [{"id": f"gsm8k-train-{hashlib.sha256(row['question'].encode()).hexdigest()[:12]}", "messages": [{"role": "user", "content": row["question"]}]}
            for row in t4_prompts.rows("openai/gsm8k", "main", "train", 7473, per, rng)]
    code = [{"id": f"mbpp-train-{row['task_id']}", "messages": [{"role": "user", "content": f"{row['text']}\nYour code "
             "should pass these tests:\n" + "\n".join(row["test_list"])}]}
            for row in t4_prompts.rows("google-research-datasets/mbpp", "full", "train", 374, per, rng)]
    tools = []
    for row in t4_prompts.rows("NousResearch/hermes-function-calling-v1", "func_calling_singleturn", "train", 1893, per, rng):
        human = next(turn["value"] for turn in row["conversations"] if turn["from"] == "human")
        declared = json.loads(row["tools"]) if isinstance(row["tools"], str) else row["tools"]
        tools.append({"id": f"hermes-fc-{row['id']}", "messages": [{"role": "user", "content": human}], "tools": declared})
    chosen = [p for group in zip(math, code, tools) for p in group
              if p["messages"][0]["content"] not in held_out][:count]
    with out.open("w", encoding="utf-8") as handle:
        for prompt in chosen:
            handle.write(json.dumps(prompt, ensure_ascii=False) + "\n")
    print(f"{len(chosen)} prompts, none of the {len(held_out)} T4 prompts")


def generate(server: str, prompts_path: Path, out: Path, parallel: int, max_tokens: int) -> None:
    done = {json.loads(line)["id"] for line in out.open()} if out.exists() else set()
    todo = [json.loads(line) for line in prompts_path.open() if json.loads(line)["id"] not in done]

    def ask(prompt: dict) -> dict:
        body = {"model": "default", "messages": prompt["messages"], "temperature": 0, "max_tokens": max_tokens}
        if "tools" in prompt:
            body["tools"] = prompt["tools"]
        request = urllib.request.Request(f"{server}/v1/chat/completions", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=3600) as response:
            message = json.load(response)["choices"][0]["message"]
        return {**prompt, "answer": message}

    with out.open("a", encoding="utf-8") as handle, concurrent.futures.ThreadPoolExecutor(parallel) as pool:
        for count, result in enumerate(pool.map(ask, todo), start=1):
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"{count}/{len(todo)} {result['id']}", flush=True)


def render(template, record: dict) -> str:
    """The prompt and the answer as the chat template lays out a finished turn, thinking included."""
    answer = record["answer"]
    assistant = {"role": "assistant", "content": answer.get("content") or ""}
    if answer.get("reasoning_content"):
        assistant["reasoning_content"] = answer["reasoning_content"]
    if answer.get("tool_calls"):
        assistant["tool_calls"] = [{**call, "function": {**call["function"],
                                                        "arguments": json.loads(call["function"]["arguments"])}}
                                   if isinstance(call["function"].get("arguments"), str) else call
                                   for call in answer["tool_calls"]]
    return template.render(messages=record["messages"] + [assistant], tools=record.get("tools"),
                           add_generation_prompt=False, enable_thinking=True)


def windows(ids: list[int], length: int, count: int, rng: random.Random) -> list[np.ndarray]:
    stream = np.asarray(ids, dtype=np.int32)
    starts = list(range(0, len(stream) - length + 1, length))
    rng.shuffle(starts)
    return [stream[s:s + length] for s in starts[:count]]


def tokens(checkpoint: Path, out: Path, length: int, generated: int, prose: int, code: int) -> None:
    import jinja2
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(checkpoint / "tokenizer.json"))
    environment = jinja2.Environment(loader=jinja2.FileSystemLoader(str(checkpoint)), trim_blocks=False,
                                     extensions=["jinja2.ext.loopcontrols"])
    environment.filters["tojson"] = lambda value, **kw: json.dumps(value, ensure_ascii=False, **kw)
    template = environment.get_template("chat_template.jinja")
    rng = random.Random(SEED)
    records = [json.loads(line) for line in (out / "generations.jsonl").open()]
    documents = [render(template, record) for record in records]
    gen_ids = [i for doc in documents for i in tokenizer.encode(doc, add_special_tokens=False).ids]
    archive = zipfile.ZipFile(io.BytesIO(fetch(WIKITEXT)))
    wiki = archive.read("wikitext-2-raw/wiki.train.raw").decode("utf-8")
    sources = []
    with concurrent.futures.ThreadPoolExecutor(16) as pool:
        urls = [f"{RAW}/{repo}/{tag}/{name}" for repo, tag, names in CODE for name in names]
        for url, data in zip(urls, pool.map(fetch, urls)):
            sources.append(f"// {url}\n" + data.decode("utf-8", errors="replace"))
    code_text = "\n".join(sources)
    parts = (windows(gen_ids, length, generated, rng)
             + windows(tokenizer.encode(wiki, add_special_tokens=False).ids, length, prose, rng)
             + windows(tokenizer.encode(code_text, add_special_tokens=False).ids, length, code, rng))
    array = np.stack(parts)
    np.save(out / "tokens.npy", array)
    print(f"tokens.npy {array.shape}: generated {len(gen_ids)} tokens over {len(documents)} answers")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prompts"); p.add_argument("out", type=Path); p.add_argument("t4", type=Path)
    p.add_argument("--count", type=int, default=240)
    g = sub.add_parser("generate"); g.add_argument("server"); g.add_argument("prompts", type=Path); g.add_argument("out", type=Path)
    g.add_argument("--parallel", type=int, default=4); g.add_argument("--max-tokens", type=int, default=1800)
    t = sub.add_parser("tokens"); t.add_argument("checkpoint", type=Path); t.add_argument("out", type=Path)
    t.add_argument("--length", type=int, default=2048); t.add_argument("--generated", type=int, default=128)
    t.add_argument("--prose", type=int, default=64); t.add_argument("--code", type=int, default=64)
    args = parser.parse_args(argv)
    if args.command == "prompts":
        prompts(args.out, args.count, args.t4)
    elif args.command == "generate":
        generate(args.server, args.prompts, args.out, args.parallel, args.max_tokens)
    else:
        tokens(args.checkpoint, args.out, args.length, args.generated, args.prose, args.code)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
