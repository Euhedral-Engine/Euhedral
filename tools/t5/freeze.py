"""Records the frozen evaluation set: every input, output and setting it depends on, with hashes.

Usage (from tools/): python3 -m t5.freeze STORE --model GGUF --tokenizer DIR --llama-cpp DIR --command TEXT
                                          [--version 1]

Writes STORE/frozen/manifest.json: the sha256 and size of the corpus (chunks.jsonl, tokens.bin, text.txt from
`t5.chunks`), of the reference logits (reference.kld), of the BF16 model file the references come from, of the
tokenizer and chat template, of the generation record (progress.json, summary.json, server.env) and of the prompt,
document, fixture and check files; the llama.cpp commit and command line that wrote the references; the repository
commit of the scorer; and the scoring protocol. Any change to one of them makes a new version. `t5.run` checks the
manifest before every run.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

CHUNK = 2560
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json",
                   "config.json")
STORE_FILES = ("conversations.jsonl", "documents.jsonl", "documents/manifest.jsonl", "generation/progress.json",
               "generation/summary.json", "generation/server.env")
STORE_DIRECTORIES = ("checks", "fixtures", "prompts")
FROZEN_FILES = ("chunks.jsonl", "tokens.bin", "text.txt", "reference.kld")


def digest(path: Path) -> dict:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 24):
            sha.update(block)
    return {"sha256": sha.hexdigest(), "bytes": path.stat().st_size}


def commit(directory: Path) -> str | None:
    result = subprocess.run(["git", "-C", str(directory), "rev-parse", "HEAD"], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def files(store: Path) -> dict:
    """The hashed files under the store, by path relative to it (the frozen ones first)."""
    out = {f"frozen/{name}": digest(store / "frozen" / name) for name in FROZEN_FILES}
    out.update({name: digest(store / name) for name in STORE_FILES})
    for directory in STORE_DIRECTORIES:
        for path in sorted((store / directory).rglob("*")):
            if path.is_file():
                out[path.relative_to(store).as_posix()] = digest(path)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("store", type=Path)
    parser.add_argument("--model", type=Path, required=True, help="the BF16 GGUF the references were computed from")
    parser.add_argument("--tokenizer", type=Path, required=True, help="the checkpoint directory of the tokenizer")
    parser.add_argument("--llama-cpp", type=Path, required=True, help="the llama.cpp checkout that was built")
    parser.add_argument("--command", required=True, help="the llama-perplexity command line that wrote reference.kld")
    parser.add_argument("--version", default="1")
    args = parser.parse_args(argv)
    chunks = [json.loads(line) for line in (args.store / "frozen" / "chunks.jsonl").read_text().splitlines()]
    domains: dict[str, int] = {}
    for c in chunks:
        domains[c["domain"]] = domains.get(c["domain"], 0) + 1
    manifest = {
        "version": args.version,
        "frozen": datetime.date.today().isoformat(),
        "chunks": len(chunks),
        "chunk_tokens": CHUNK,
        "domains": domains,
        "protocol": {
            "scored": f"the second half of every chunk: positions {CHUNK // 2 + 1}..{CHUNK - 1}, each predicted from "
                      "the chunk's tokens before it on a fresh sequence",
            "metrics": ["mean KL(reference || candidate)", "top-1 agreement", "KL p95", "KL p99",
                        "dNLL (candidate NLL - reference NLL)",
                        "paired per-domain KL difference against the baseline, bootstrap over chunks"],
            "scorer": "quantization score --values (row-exact forced verification, 8 rows per step)",
            "output": "aggregate and per-domain numbers only",
        },
        "model": {"file": args.model.name, **digest(args.model)},
        "tokenizer": {name: digest(args.tokenizer / name) for name in TOKENIZER_FILES
                      if (args.tokenizer / name).is_file()},
        "references": {"llama_cpp_commit": commit(args.llama_cpp), "command": args.command},
        "scorer_commit": commit(Path(__file__).resolve().parent),
        "files": files(args.store),
    }
    out = args.store / "frozen" / "manifest.json"
    out.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(f"{out}: version {args.version}, {len(chunks)} chunks, sha256 {digest(out)['sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
