"""Publication-date checks for human sources hosted on GitHub.

Usage (from tools/): python3 -m t5.sources check STORE

Re-derives the publication date of every human source with a github.com URL (an issue or pull request: its
`created_at`; a file: the date of the oldest commit touching it on the linked ref, following renames and copies
back through the history, so a file moved in after the cutoff from an earlier path carries the earlier date; the
repository may be older than the cutoff, since the material itself is what must be new) and writes STORE/checks/dates.json. GITHUB_TOKEN,
when set, is sent as a bearer token and never printed.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sys
import urllib.request
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

try:
    from t5 import records
except ImportError:  # run as a script
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from t5 import records

API = "https://api.github.com"
MAX_FOLLOW = 10  # renames followed back through a path's history
FILES_PER_PAGE = 300  # the commit endpoint caps its file list per page
MAX_FILE_PAGES = 10


def urlopen_json(url: str):
    """Returns (parsed JSON, response headers with lowercase names)."""
    request = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json",
                                                   "User-Agent": "t5-source-check"})
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response), {k.lower(): v for k, v in response.headers.items()}


def _date(stamp: str) -> datetime.date:
    return datetime.datetime.fromisoformat(stamp.replace("Z", "+00:00")).date()


def github_issue_created(owner: str, repo: str, number: int, fetch=urlopen_json) -> datetime.date:
    data, _ = fetch(f"{API}/repos/{owner}/{repo}/issues/{number}")
    return _date(data["created_at"])


def github_repo_created(owner: str, repo: str, fetch=urlopen_json) -> datetime.date:
    data, _ = fetch(f"{API}/repos/{owner}/{repo}")
    return _date(data["created_at"])


class Unverified(Exception):
    """The publication date cannot be established with certainty; the source counts as failed."""


def _commit_files(owner: str, repo: str, sha: str, path: str, fetch) -> tuple[dict, dict | None]:
    """(commit detail, the file entry for `path`). The commit's file list is capped per page, so the following
    pages are read until the path turns up or a short page ends the list; a path that never appears leaves the
    commit unverified."""
    try:
        detail, _ = fetch(f"{API}/repos/{owner}/{repo}/commits/{sha}?per_page={FILES_PER_PAGE}")
        page = 1
        while True:
            files = detail.get("files", [])
            for entry in files:
                if entry.get("filename") == path:
                    return detail, entry
            if len(files) < FILES_PER_PAGE or page >= MAX_FILE_PAGES:
                break
            page += 1
            more, _ = fetch(f"{API}/repos/{owner}/{repo}/commits/{sha}?per_page={FILES_PER_PAGE}&page={page}")
            detail = {**detail, "files": more.get("files", [])}
    except (OSError, KeyError, TypeError) as e:  # network failure, HTTP error or malformed answer
        raise Unverified(f"commit {sha[:10]}: {type(e).__name__}: {e}") from e
    raise Unverified(f"commit {sha[:10]} does not list {path} among its files")


def github_path_first_commit(owner: str, repo: str, path: str, fetch=urlopen_json,
                             ref: str | None = None, _depth: int = 0) -> datetime.date:
    """Date of the oldest commit touching `path` (on `ref`, a branch, tag or commit, when given). Commits come
    newest first, so the last page holds it. When that commit created the path by renaming or copying another
    one (the commit's file status says so), the search continues from its parent on the earlier path and the
    earlier date wins. Anything that leaves the answer uncertain raises Unverified: the path missing from the
    commit's file list, a failing commit lookup, or more than MAX_FOLLOW hops. A copy that git does not report
    as one shows as a plain addition."""
    sha = f"sha={quote(ref, safe='')}&" if ref else ""
    data, headers = fetch(f"{API}/repos/{owner}/{repo}/commits?{sha}path={quote(path)}&per_page=100")
    last = re.search(r'<([^>]+)>;\s*rel="last"', headers.get("link", ""))
    if last:
        data, _ = fetch(last.group(1))
    if not data:
        raise ValueError(f"no commits touch {owner}/{repo}:{path}")
    found = min(_date(c["commit"][who]["date"]) for c in data for who in ("author", "committer"))
    oldest = data[-1]
    if not oldest.get("sha"):
        return found
    detail, entry = _commit_files(owner, repo, oldest["sha"], path, fetch)
    if entry.get("status") not in ("renamed", "copied") or not entry.get("previous_filename"):
        return found
    if _depth >= MAX_FOLLOW:
        raise Unverified(f"{path} was renamed or copied more than {MAX_FOLLOW} times")
    parents = detail.get("parents") or oldest.get("parents") or []
    if not parents:
        raise Unverified(f"commit {oldest['sha'][:10]} renamed {path} but has no parent")
    try:
        earlier = github_path_first_commit(owner, repo, entry["previous_filename"], fetch,
                                           parents[0]["sha"], _depth + 1)
    except ValueError as e:  # the earlier path has no history on the parent
        raise Unverified(str(e)) from e
    return min(found, earlier)


def published_date(url: str, fetch=urlopen_json) -> tuple[datetime.date, datetime.date | None] | None:
    """(publication date, repository creation date for file sources else None) of a github.com URL, or None when
    its form is not recognised. A file URL `/owner/repo/blob/REF/PATH` is ambiguous when REF holds `/`: the
    longest-ref-first split that yields commits wins."""
    parsed = urlparse(url)
    parts = [unquote(p) for p in parsed.path.split("/") if p]
    if parsed.hostname != "github.com" or len(parts) < 4:
        return None
    owner, repo, kind = parts[:3]
    if kind in ("issues", "pull") and parts[3].isdigit():
        return github_issue_created(owner, repo, int(parts[3]), fetch), None
    if kind in ("blob", "tree") and len(parts) >= 5:
        rest = parts[3:]
        for split in range(len(rest) - 1, 0, -1):
            try:
                found = github_path_first_commit(owner, repo, "/".join(rest[split:]), fetch, "/".join(rest[:split]))
            except (ValueError, OSError):
                continue
            return found, github_repo_created(owner, repo, fetch)
        raise ValueError("no ref and path split of the url has commits")
    return None


def check(store: Path, fetch=urlopen_json) -> tuple[dict, bool]:
    items = []
    conversations = store / "conversations.jsonl"
    if conversations.exists():
        items += [(c.id, c.source) for c in records.load_conversations(conversations)]
    documents = store / "documents.jsonl"
    if documents.exists():
        items += [(d.id, d.source) for d in records.load_documents(documents)]
    result: dict = {}
    failed = False
    for item_id, source in items:
        if source.kind != "human" or not source.url or urlparse(source.url).hostname != "github.com":
            continue
        entry = {"url": source.url, "recorded": str(source.published)}
        try:
            dated = published_date(source.url, fetch)
            found, created = dated if dated else (None, None)
        except Exception as e:  # network or API failure counts as unverified
            entry["error"] = f"{type(e).__name__}: {e}"
            failed = True
        else:
            if found is None:
                entry["error"] = "url form not recognised"
                failed = True
            else:
                entry["verified"] = str(found)
                if created is not None:
                    entry["repository_created"] = str(created)
                    entry["note"] = "oldest commit at the path, following renames and copies"
                if found <= records.CUTOFF or found != source.published:
                    entry["error"] = "date is not after the cutoff" if found <= records.CUTOFF \
                        else "date differs from the record"
                    failed = True
        result[item_id] = entry
    return result, failed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="t5.sources")
    parser.add_argument("command", choices=["check"])
    parser.add_argument("store", type=Path)
    args = parser.parse_args(argv)
    result, failed = check(args.store)
    out = args.store / "checks"
    out.mkdir(exist_ok=True)
    (out / "dates.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    for item_id, entry in result.items():
        if "error" in entry:
            print(f"{item_id}: {entry['error']}")
    print(f"{len(result)} sources checked, {'failures' if failed else 'all verified'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
