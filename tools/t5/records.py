"""Conversation and document records for the evaluation set, with validation.

Usage (from tools/): python3 -m t5.records validate STORE [--base-seed N]

STORE holds conversations.jsonl, documents.jsonl, the fixtures and the document files they name.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime
import hashlib
import json
import sys
from pathlib import Path
from urllib.parse import urlparse

try:
    from t5 import fixtures
except ImportError:  # run as a script
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from t5 import fixtures

CUTOFF = datetime.date(2026, 9, 21)
CHUNKS = {"conversation": 24, "coding": 24, "tool_use": 16, "multilingual": 16}
DOCUMENT_CHUNKS = {"prose": 16, "code": 16}
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0}
SAMPLED_DOMAINS = frozenset({"conversation", "coding", "multilingual"})
LANGUAGES = frozenset({"es", "zh", "ja", "ar", "hi", "mixed"})
CODE_LANGUAGES = frozenset({"java", "rust", "python", "sql", "config", "data"})
EXCLUDED_PATH = Path(__file__).with_name("excluded_sources.json")


@dataclasses.dataclass(frozen=True)
class Source:
    kind: str  # "authored" | "human"
    url: str | None = None
    published: datetime.date | None = None
    license: str | None = None
    checked: str | None = None  # how the date was verified

    @staticmethod
    def from_json(value: dict | None) -> "Source":
        value = value or {"kind": "authored"}
        published = value.get("published")
        return Source(value.get("kind", "authored"), value.get("url"),
                      datetime.date.fromisoformat(published) if published else None,
                      value.get("license"), value.get("checked"))


@dataclasses.dataclass(frozen=True)
class Conversation:
    id: str
    domain: str
    order: int  # pre-declared selection order within the domain (spares last)
    language: str  # BCP-47, "mixed" for code-switching
    system: str | None
    turns: tuple[str, ...]  # scripted user turns
    tools: tuple[dict, ...]  # OpenAI function tools
    fixture: str | None  # path relative to the store
    source: Source


@dataclasses.dataclass(frozen=True)
class Document:
    id: str
    kind: str  # "prose" | "code"
    language: str  # code: java, rust, python, sql, config, data; prose: BCP-47
    path: str  # relative to the store
    sha256: str
    source: Source


@dataclasses.dataclass(frozen=True)
class Decoding:
    mode: str  # "greedy" | "sampled"
    seed: int | None

    def request_fields(self) -> dict:
        if self.mode == "greedy":
            return {"temperature": 0}
        return {**SAMPLING, "seed": self.seed}


def _lines(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_conversations(path: Path) -> list[Conversation]:
    return [Conversation(r["id"], r["domain"], r["order"], r["language"], r.get("system"),
                         tuple(r.get("turns", ())), tuple(r.get("tools", ())), r.get("fixture"),
                         Source.from_json(r.get("source")))
            for r in _lines(path)]


def load_documents(path: Path) -> list[Document]:
    return [Document(r["id"], r["kind"], r["language"], r["path"], r["sha256"], Source.from_json(r.get("source")))
            for r in _lines(path)]


def load_excluded(path: Path = EXCLUDED_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def plan_decoding(conversations: list[Conversation], base_seed: int) -> dict[str, Decoding]:
    """Selection takes a prefix by `order` (spares last), so the greedy/sampled alternation is fixed by `order`.

    Within each sampled domain, in (order, id) order, even positions decode greedily and odd ones sampled."""
    plan: dict[str, Decoding] = {}
    for domain in sorted({c.domain for c in conversations}):
        members = sorted((c for c in conversations if c.domain == domain), key=lambda c: (c.order, c.id))
        for index, conversation in enumerate(members):
            if domain in SAMPLED_DOMAINS and index % 2 == 1:
                digest = hashlib.sha256(f"{base_seed}:{conversation.id}".encode()).hexdigest()
                plan[conversation.id] = Decoding("sampled", int(digest[:15], 16))
            else:
                plan[conversation.id] = Decoding("greedy", None)
    return plan


def excluded_reason(url: str | None, excluded: dict) -> str | None:
    """Names the excluded repository or dataset a URL points into, else None."""
    if not url:
        return None
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    parts = [p for p in parsed.path.split("/") if p]
    if host in ("github.com", "raw.githubusercontent.com", "api.github.com"):
        if host == "api.github.com" and parts[:1] == ["repos"]:
            parts = parts[1:]
        name = "/".join(parts[:2]).lower()
        for repo in excluded.get("repositories", []):
            if repo.lower() == name:
                return f"excluded repository {repo}"
    for site in excluded.get("sites", []):
        if host == site or host.endswith("." + site):
            return f"excluded site {site}"
    if host.endswith("huggingface.co"):
        if parts[:1] == ["datasets"]:
            parts = parts[1:]
        name = "/".join(parts[:2]).lower()
        query = parsed.query.lower()
        for dataset in excluded.get("datasets", []):
            if dataset.lower() == name or f"dataset={dataset.lower()}" in query.replace("%2f", "/"):
                return f"excluded dataset {dataset}"
    return None


def _check_human(label: str, source: Source, excluded: dict, errors: list[str]) -> None:
    if source.kind not in ("authored", "human"):
        errors.append(f"{label}: unknown source kind {source.kind!r}")
        return
    if source.kind == "authored":
        return
    if not source.url:
        errors.append(f"{label}: human source needs a url")
    if source.published is None:
        errors.append(f"{label}: human source needs a publication date")
    elif source.published <= CUTOFF:
        errors.append(f"{label}: published {source.published} is not after {CUTOFF}")
    if not source.license:
        errors.append(f"{label}: human source needs a license")
    if not source.checked:
        errors.append(f"{label}: human source needs a record of how the date was checked")
    reason = excluded_reason(source.url, excluded)
    if reason:
        errors.append(f"{label}: {reason}")


def _inside(store: Path, relative: str) -> bool:
    """Whether `relative` stays within the store (no absolute paths, no `..`)."""
    try:
        (store / relative).resolve().relative_to(store.resolve())
    except ValueError:
        return False
    return not Path(relative).is_absolute()


def validate_conversations(conversations: list[Conversation], store: Path, excluded: dict,
                           chunks: dict | None = None, languages=LANGUAGES,
                           base_seed: int = 0) -> list[str]:
    chunks = CHUNKS if chunks is None else chunks
    errors: list[str] = []
    seen: set[str] = set()
    for c in conversations:
        if c.id in seen:
            errors.append(f"{c.id}: duplicate id")
        seen.add(c.id)
        if c.domain not in chunks:
            errors.append(f"{c.id}: unknown domain {c.domain!r}")
        if not isinstance(c.order, int) or isinstance(c.order, bool):
            errors.append(f"{c.id}: order must be an integer")
        if not c.turns or not all(isinstance(t, str) and t.strip() for t in c.turns):
            errors.append(f"{c.id}: turns must be non-empty strings")
        if c.domain == "tool_use":
            if not c.tools:
                errors.append(f"{c.id}: tool_use needs tools")
            if not c.fixture or not _inside(store, c.fixture) or not (store / c.fixture).is_file():
                errors.append(f"{c.id}: missing fixture {c.fixture}")
        if c.tools and c.fixture and _inside(store, c.fixture) and (store / c.fixture).is_file():
            names = [t.get("function", {}).get("name") for t in c.tools]
            try:
                spec = json.loads((store / c.fixture).read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                errors.append(f"{c.id}: fixture is not valid JSON")
            else:
                for error in fixtures.validate_spec(spec, [n for n in names if n]):
                    errors.append(f"{c.id}: fixture: {error}")
            if not all(names):
                errors.append(f"{c.id}: every tool needs a function name")
        _check_human(c.id, c.source, excluded, errors)
    for domain, count in chunks.items():
        members = [c for c in conversations if c.domain == domain]
        if len(members) < count:
            errors.append(f"domain {domain}: {len(members)} conversations, need at least {count}")
        orders = [c.order for c in members]
        if len(set(orders)) != len(orders):
            errors.append(f"domain {domain}: order values are not unique")
    spoken = {c.language.split("-")[0].lower() for c in conversations if c.domain == "multilingual"}
    if "multilingual" in chunks and languages - spoken:
        errors.append(f"multilingual: missing languages {sorted(languages - spoken)}")
    if not all(isinstance(c.order, int) for c in conversations):
        return errors
    for conversation_id, decoding in plan_decoding(conversations, base_seed).items():
        domain = next(c.domain for c in conversations if c.id == conversation_id)
        if domain == "tool_use" and decoding.mode != "greedy":
            errors.append(f"{conversation_id}: sampled tool_use")
    return errors


def validate_documents(documents: list[Document], store: Path, excluded: dict,
                       chunks: dict | None = None, code_languages=CODE_LANGUAGES) -> list[str]:
    chunks = DOCUMENT_CHUNKS if chunks is None else chunks
    errors: list[str] = []
    seen: set[str] = set()
    for d in documents:
        if d.id in seen:
            errors.append(f"{d.id}: duplicate id")
        seen.add(d.id)
        if d.kind not in chunks:
            errors.append(f"{d.id}: unknown kind {d.kind!r}")
        file = store / d.path
        if not _inside(store, d.path):
            errors.append(f"{d.id}: path {d.path} leaves the store")
        elif not file.is_file():
            errors.append(f"{d.id}: missing file {d.path}")
        elif hashlib.sha256(file.read_bytes()).hexdigest() != d.sha256:
            errors.append(f"{d.id}: sha256 mismatch for {d.path}")
        if d.source.kind != "human":
            errors.append(f"{d.id}: documents must be human-authored")
        _check_human(d.id, d.source, excluded, errors)
    for kind, count in chunks.items():
        have = sum(1 for d in documents if d.kind == kind)
        if have < count:
            errors.append(f"documents {kind}: {have}, need at least {count}")
    if "code" in chunks:
        covered = {d.language for d in documents if d.kind == "code"}
        if code_languages - covered:
            errors.append(f"code documents: missing languages {sorted(code_languages - covered)}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="t5.records")
    parser.add_argument("command", choices=["validate"])
    parser.add_argument("store", type=Path)
    parser.add_argument("--base-seed", type=int, default=0)
    args = parser.parse_args(argv)
    excluded = load_excluded()
    conversations = load_conversations(args.store / "conversations.jsonl")
    documents = load_documents(args.store / "documents.jsonl")
    errors = validate_conversations(conversations, args.store, excluded, base_seed=args.base_seed)
    errors += validate_documents(documents, args.store, excluded)
    for error in errors:
        print(error)
    print(f"{len(conversations)} conversations, {len(documents)} documents, {len(errors)} errors")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
