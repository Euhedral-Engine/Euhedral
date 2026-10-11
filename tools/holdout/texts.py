#!/usr/bin/env python3
"""Assemble the T2 and T3 holdout texts (docs/QUALITY.md).

Usage: texts.py OUT_DIR

T2 is wikitext-2's `wiki.test.raw` (the archive llama.cpp's scripts/get-wikitext-2.sh downloads). T3 is source code
from four permissively licensed projects outside this repository, at pinned tags, each file preceded by a comment
naming its path: Python (CPython, PSF), Go (the Go standard library, BSD-3-Clause), Rust (serde_json, MIT or
Apache-2.0) and C (cJSON, MIT), each language cut at a line boundary after LANGUAGE_BYTES (about 16K tokens), so
that the first 25 chunks of 2560 tokens, which a reference scores, cover all four. Writes t2.txt, t3.txt and
sources.json (URL and SHA-256 of every input and output).
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

WIKITEXT = "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip"
RAW = "https://raw.githubusercontent.com"
LANGUAGE_BYTES = 56000
CODE = {
    "python": ("python/cpython", "v3.12.0", ["Lib/json/decoder.py", "Lib/json/encoder.py", "Lib/json/scanner.py",
                                              "Lib/textwrap.py", "Lib/difflib.py"]),
    "go": ("golang/go", "go1.22.0", ["src/encoding/json/decode.go", "src/encoding/json/encode.go",
                                     "src/encoding/json/scanner.go"]),
    "rust": ("serde-rs/json", "v1.0.120", ["src/de.rs", "src/ser.rs", "src/read.rs"]),
    "c": ("DaveGamble/cJSON", "v1.7.18", ["cJSON.c", "cJSON_Utils.c"]),
}


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    out = Path(argv[0])
    out.mkdir(parents=True, exist_ok=True)
    sources = {"t2": {}, "t3": {}}
    archive = fetch(WIKITEXT)
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        t2 = zipped.read("wikitext-2-raw/wiki.test.raw")
    (out / "t2.txt").write_bytes(t2)
    sources["t2"] = {"url": WIKITEXT, "archive_sha256": sha256(archive), "member": "wikitext-2-raw/wiki.test.raw",
                     "sha256": sha256(t2)}
    parts, inputs = [], []
    for language, (repo, tag, files) in CODE.items():
        text = b""
        for name in files:
            url = f"{RAW}/{repo}/{tag}/{name}"
            data = fetch(url)
            inputs.append({"language": language, "url": url, "sha256": sha256(data)})
            text += f"// {repo} {tag} {name}\n".encode() + data + b"\n"
            if len(text) >= LANGUAGE_BYTES:
                break
        parts.append(text[:text.rfind(b"\n", 0, LANGUAGE_BYTES) + 1])
    t3 = b"".join(parts)
    (out / "t3.txt").write_bytes(t3)
    sources["t3"] = {"inputs": inputs, "sha256": sha256(t3)}
    (out / "sources.json").write_text(json.dumps(sources, indent=2) + "\n")
    print(f"t2: {len(t2)} bytes; t3: {len(t3)} bytes from {len(inputs)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
