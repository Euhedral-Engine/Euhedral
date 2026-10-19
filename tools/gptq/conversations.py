"""Build a calibration store of scripted conversations, in the format `t5.generate` runs (tools/t5).

Usage (in tools/): python -m gptq.conversations OUT_STORE --t4 T4_PROMPTS.jsonl [--per-domain 40] [--seed N]

The calibration text is generated the way the held-out evaluation set is: BF16 Swift-1.5 served by the engine, the
checkpoint's native tool calls answered from fixtures, multi-turn, sampled at the model's published settings for half
of the conversation, coding and multilingual domains (records.plan_decoding). The prompts come from training splits,
so the calibration shares no prompt with T4 or the held-out set (check with `python -m t5.overlap`):

- conversation: UltraChat 200k `train_sft`, the user turns of a multi-turn conversation (up to three);
- coding: Magicoder-OSS-Instruct-75K problems, and coding-agent sessions over small files of pallets projects at
  pinned tags (the `read_file`, `list_directory` and `write_file` builtins), tasks written here;
- tool_use: Hermes function-calling (`func_calling`) conversations; the dataset's own tool responses become the
  fixture (exact arguments match, else the tool's first response);
- multilingual: Aya `train` prompts in Spanish, Chinese, Japanese, Arabic and Hindi, and code-switching prompts written
  here.

Rows are read through the Hugging Face dataset viewer API (standard library only), from pages chosen with a fixed
seed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROWS = "https://datasets-server.huggingface.co/rows"
RAW = "https://raw.githubusercontent.com"
TREES = "https://api.github.com/repos/{repo}/git/trees/{tag}?recursive=1"
SEED = 20261018
AYA_LANGUAGES = {"es": "spa", "zh": "zho", "ja": "jpn", "ar": "arb", "hi": "hin"}
PROJECTS = [("pallets/itsdangerous", "2.2.0"), ("pallets/markupsafe", "3.0.2"), ("pallets/click", "8.1.7")]
MAX_FILE_BYTES = 4000  # the generation context is 6144 tokens: the agent reads small files only
FILE_TOOLS = [
    {"type": "function", "function": {
        "name": "list_directory", "description": "Show what a folder of the repository contains (folders end in '/').",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Folder relative to the repository root ('.' is the root)"}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "read_file", "description": "Return the full text of one file in the repository.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write_file", "description": "Create a file in the repository or overwrite it with new text.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                       "required": ["path", "content"]}}},
]
AGENT_SYSTEM = ("You maintain an open-source Python library. You can browse and edit its repository with the tools "
                "given; read the relevant code first, keep changes focused, and explain them briefly at the end.")
# Left out after `t5.overlap` against T1-T4 and the held-out set (seed 20261018).
DROPPED = {"cal-u008": "a T4 prompt's tool definition is 83% contained in it"}
AGENT_TASKS = [
    ["Look through the project and pick the module that is least covered by the tests. Explain why you chose it.",
     "Write a test file for it that covers its edge cases, and save it next to the existing tests."],
    ["Add type hints to the functions of the smallest module in the package, without changing behaviour.",
     "Explain any place where the right type was not obvious."],
    ["Find one function whose docstring is missing or does not match what it does, and fix the docstring.",
     "Now add an example to that docstring that a reader could run."],
    ["Read the package's public entry point and summarize its API for a new contributor.",
     "Write that summary into a new file docs/API.md."],
    ["Find a place where an error message could be more helpful to the caller and improve it.",
     "Add a test that checks the new message."],
    ["Look for duplicated logic between two modules and propose how to remove it.",
     "Make the change, keeping every public name working."],
    ["Pick a function with branching logic and walk me through what it does on each branch.",
     "Add a small test for the branch you think is least obvious."],
    ["Find a constant or magic value that should be named, and name it.",
     "Check that nothing else in the project depends on the old spelling."],
    ["Add a module-level docstring to every module of the package that lacks one.",
     "List the files you changed and what each docstring says."],
    ["Read the tests and tell me which behaviour of the package they do not check at all.",
     "Write one test for the most important unchecked behaviour."],
]
CODE_SWITCHING = [
    ("mixed", "Necesito help con mi resume. I'm applying for a data analyst job pero no tengo mucha experiencia, "
              "solo un internship de seis meses. ¿Cómo lo presento para que se vea strong?"),
    ("mixed", "Yaar, mujhe ek budget plan banana hai. My salary is 45k per month, rent 15k hai, aur main har mahine "
              "kuch savings karna chahta hoon. Can you make a simple plan?"),
    ("mixed", "我在写一个 Python script 来处理 CSV 文件, but the dates are in different formats. 有什么好方法 to "
              "normalize them?"),
    ("mixed", "Je dois préparer une presentation for my team demain, about our Q3 results. Comment je structure ça "
              "pour que ce soit clair et pas trop long?"),
]


PAGE = 100  # rows per viewer request, its maximum


CACHE: Path | None = None  # fetched pages, so a rerun reads the same rows without asking again


def get_json(url: str, attempts: int = 8):
    cached = CACHE / (hashlib.sha256(url.encode()).hexdigest() + ".json") if CACHE else None
    if cached and cached.is_file():
        return json.loads(cached.read_text(encoding="utf-8"))
    value = fetch_json(url, attempts)
    if cached:
        cached.write_text(json.dumps(value), encoding="utf-8")
    return value


def fetch_json(url: str, attempts: int):
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                return json.loads(response.read())
        except (OSError, ValueError):
            if attempt == attempts - 1:
                raise
            time.sleep(5 * 2 ** attempt)  # the viewer rate-limits (429): back off for minutes, not seconds


def rows(dataset: str, config: str, split: str, rng: random.Random, count: int) -> list[dict]:
    """About `count` rows from random pages of the split, in random order."""
    total = size(dataset, config, split)
    pages = rng.sample(range(max(1, total // PAGE)), min(max(1, total // PAGE), -(-count // PAGE) + 1))
    out = []
    for page in pages:
        query = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": split,
                                        "offset": page * PAGE, "length": PAGE})
        out.extend(r["row"] for r in get_json(f"{ROWS}?{query}")["rows"])
        time.sleep(1)
    rng.shuffle(out)
    return out


def size(dataset: str, config: str, split: str) -> int:
    query = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": split, "offset": 0, "length": 1})
    return get_json(f"{ROWS}?{query}")["num_rows_total"]


def conversation_domain(rng: random.Random, count: int, excluded: set[str]) -> list[dict]:
    out = []
    for row in rows("HuggingFaceH4/ultrachat_200k", "default", "train_sft", rng, count * 2):
        turns = [m["content"].strip() for m in row["messages"] if m["role"] == "user"][:3]
        if len(turns) < 2 or any(t in excluded for t in turns) or any(len(t) > 2000 for t in turns):
            continue
        out.append({"turns": turns, "source_url": "https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k"})
        if len(out) == count:
            break
    return out


def project_files(repo: str, tag: str) -> dict[str, str]:
    tree = get_json(TREES.format(repo=repo, tag=tag))["tree"]
    files = {}
    for entry in tree:
        path = entry["path"]
        if entry["type"] != "blob" or not path.endswith((".py", ".md", ".toml", ".rst")):
            continue
        if not (path.startswith(("src/", "tests/")) or "/" not in path) or entry.get("size", 0) > MAX_FILE_BYTES:
            continue
        with urllib.request.urlopen(f"{RAW}/{repo}/{tag}/{path}", timeout=120) as response:
            files[path] = response.read().decode("utf-8")
    return files


def coding_domain(rng: random.Random, count: int, store: Path, excluded: set[str], agent_tasks=AGENT_TASKS) -> list[dict]:
    agents = []
    projects = [(repo, tag, project_files(repo, tag)) for repo, tag in PROJECTS] if agent_tasks else []
    for index, turns in enumerate(agent_tasks):
        repo, tag, files = projects[index % len(projects)]
        fixture = f"fixtures/cal-k-g{index + 1:02d}.json"
        tools = {t["function"]["name"]: {"builtin": t["function"]["name"]} for t in FILE_TOOLS}
        (store / fixture).write_text(json.dumps({"tools": tools, "files": files}, ensure_ascii=False, indent=1))
        agents.append({"turns": turns, "system": AGENT_SYSTEM, "tools": FILE_TOOLS, "fixture": fixture,
                       "source_url": f"https://github.com/{repo}/tree/{tag}"})
    problems = []
    for row in rows("ise-uiuc/Magicoder-OSS-Instruct-75K", "default", "train", rng, count * 2):
        problem = row["problem"].strip()
        if problem in excluded or len(problem) > 3000:
            continue
        problems.append({"turns": [problem], "source_url":
                         "https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K"})
        if len(problems) + len(agents) == count:
            break
    return agents + problems


TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
TOOL_RESPONSE = re.compile(r"<tool_response>\s*(\{.*?\})\s*</tool_response>", re.S)


def hermes_conversation(row: dict) -> tuple[list[str], list[dict], dict] | None:
    """The user turns, the OpenAI tools and a fixture from one Hermes `func_calling` conversation."""
    tools = json.loads(row["tools"]) if isinstance(row["tools"], str) else row["tools"]
    if not tools:
        return None
    turns, calls, rules, defaults = [], [], {}, {}
    for message in row["conversations"]:
        if message["from"] == "human":
            turns.append(message["value"].strip())
        elif message["from"] == "gpt":
            calls = [json.loads(c) for c in TOOL_CALL.findall(message["value"])]
        elif message["from"] == "tool":
            for call, response in zip(calls, (json.loads(r) for r in TOOL_RESPONSE.findall(message["value"]))):
                name, content = call.get("name"), response.get("content")
                if not name or content is None:
                    continue
                rules.setdefault(name, []).append({"match": call.get("arguments") or {}, "result": content})
                defaults.setdefault(name, content)
    names = [t["function"]["name"] for t in tools if isinstance(t, dict) and "function" in t]
    if not turns or not names:
        return None
    spec = {"tools": {n: {"rules": rules.get(n, []), "default": defaults.get(n, {"error": "no result available"})}
                      for n in names}, "files": {}}
    return turns[:3], tools, spec


def tool_domain(rng: random.Random, count: int, store: Path, excluded: set[str]) -> list[dict]:
    out = []
    for row in rows("NousResearch/hermes-function-calling-v1", "func_calling", "train", rng, count * 3):
        try:
            parsed = hermes_conversation(row)
        except (ValueError, TypeError, KeyError):
            continue
        if parsed is None or any(t in excluded for t in parsed[0]):
            continue
        turns, tools, spec = parsed
        fixture = f"fixtures/cal-u-{len(out) + 1:02d}.json"
        (store / fixture).write_text(json.dumps(spec, ensure_ascii=False, indent=1))
        out.append({"turns": turns, "tools": tools, "fixture": fixture,
                    "source_url": "https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1"})
        if len(out) == count:
            break
    return out


def multilingual_domain(rng: random.Random, count: int, excluded: set[str], written=CODE_SWITCHING) -> list[dict]:
    per_language = (count - len(written)) // len(AYA_LANGUAGES)
    chosen: dict[str, list[str]] = {language: [] for language in AYA_LANGUAGES}
    languages = {code: language for language, code in AYA_LANGUAGES.items()}
    # Aya mixes some 65 languages: scan random pages until each of the five has its share.
    pages = list(range(size("CohereLabs/aya_dataset", "default", "train") // PAGE))
    rng.shuffle(pages)
    for page in pages:
        if all(len(texts) >= per_language for texts in chosen.values()):
            break
        query = urllib.parse.urlencode({"dataset": "CohereLabs/aya_dataset", "config": "default", "split": "train",
                                        "offset": page * PAGE, "length": PAGE})
        for row in (r["row"] for r in get_json(f"{ROWS}?{query}")["rows"]):
            language, text = languages.get(row["language_code"]), row["inputs"].strip()
            if language and len(chosen[language]) < per_language and 40 <= len(text) <= 2000 and text not in excluded:
                chosen[language].append(text)
        time.sleep(1)
    out = []
    for language, texts in chosen.items():
        out.extend({"turns": [text], "language": language,
                    "source_url": "https://huggingface.co/datasets/CohereLabs/aya_dataset"} for text in texts)
    out.extend({"turns": [text], "language": language, "source_url": None} for language, text in written)
    return out


def held_out_texts(t4: Path) -> set[str]:
    """User messages of the T4 prompts, compared by text before the n-gram check."""
    return {m["content"].strip() for line in t4.open() for m in json.loads(line)["messages"] if m["role"] == "user"}


def store_texts(store: Path) -> set[str]:
    """The user turns of another store's conversations (the calibration set, the release test)."""
    return {t.strip() for line in (store / "conversations.jsonl").open() if line.strip()
            for t in json.loads(line)["turns"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", type=Path)
    parser.add_argument("--t4", type=Path, required=True, help="the T4 prompts JSONL (tools/holdout/t4_prompts.py)")
    parser.add_argument("--per-domain", type=int, default=40)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--exclude-store", type=Path, action="append", default=[],
                        help="leave out every prompt of this store (repeatable)")
    parser.add_argument("--no-written-prompts", action="store_true",
                        help="leave out the prompts written here (coding-agent tasks, code-switching), which another store holds")
    parser.add_argument("--prefix", default="cal", help="conversation ID prefix")
    parser.add_argument("--drop", nargs="*", default=[], help="conversation IDs to leave out (flagged by t5.overlap)")
    args = parser.parse_args(argv)
    global CACHE
    rng = random.Random(args.seed)
    (args.out / "fixtures").mkdir(parents=True, exist_ok=True)
    CACHE = args.out / "cache"
    CACHE.mkdir(exist_ok=True)
    excluded = held_out_texts(args.t4)
    for store in args.exclude_store:
        excluded |= store_texts(store)
    dropped = (DROPPED if args.seed == SEED and args.prefix == "cal" else {}) | {cid: "flagged" for cid in args.drop}
    domains = {
        "conversation": conversation_domain(rng, args.per_domain, excluded),
        "coding": coding_domain(rng, args.per_domain, args.out, excluded, [] if args.no_written_prompts else AGENT_TASKS),
        "tool_use": tool_domain(rng, args.per_domain, args.out, excluded),
        "multilingual": multilingual_domain(rng, args.per_domain, excluded, [] if args.no_written_prompts else CODE_SWITCHING),
    }
    prefix = {domain: f"{args.prefix}-{letter}" for domain, letter in
              (("conversation", "c"), ("coding", "k"), ("tool_use", "u"), ("multilingual", "m"))}
    with (args.out / "conversations.jsonl").open("w", encoding="utf-8") as out:
        for domain, items in domains.items():
            for order, item in enumerate(items):
                cid = f"{prefix[domain]}{order + 1:03d}"
                if cid in dropped:
                    continue
                record = {"id": cid, "domain": domain, "order": order,
                          "language": item.get("language", "en"), "system": item.get("system"),
                          "turns": item["turns"], "tools": item.get("tools", []), "fixture": item.get("fixture"),
                          "source": {"kind": "authored"}, "dataset": item.get("source_url")}
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"{domain}: {sum(1 for o in range(len(items)) if f'{prefix[domain]}{o + 1:03d}' not in dropped)} conversations")
    return 0


if __name__ == "__main__":
    sys.exit(main())
