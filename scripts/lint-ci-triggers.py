#!/usr/bin/env python3
"""Lint the triggers of ci.yaml: it runs once per pull request, when the label `land` is added, and tests nothing on a push.

A workflow file fails when:
  - its `on:` has a `pull_request` without `types: [labeled]`, or with a `paths` or `paths-ignore` filter (a filtered pull
    request never posts its required contexts, so it could never land);
  - its `on:` has a `push` to any branch but `rust/next*` (a push to main or an integration branch is not a second test of
    the tree the land run tested);
  - a job other than `land-guard` has no `if:` that mentions the added label `land`;
  - it has no `land-guard` job, or its concurrency group does not tell a land run from another label's run.
Plain text is read, like scripts/lint-workflows.py: no YAML library.

usage: lint-ci-triggers.py [FILE]   (default .gitea/workflows/ci.yaml; exit 0: clean, 1: a violation)
"""

import re
import sys
from pathlib import Path

LAND = "added_labels.*.name, 'land'"
GUARD = "land-guard"


def top_block(lines, key):
    """The lines under the top-level `key:` up to the next top-level key."""
    for start, line in enumerate(lines):
        if line.startswith(f"{key}:"):
            end = start + 1
            while end < len(lines) and (not lines[end].strip() or lines[end].startswith((" ", "#"))):
                end += 1
            return lines[start + 1:end]
    return None


def sub_block(block, key):
    """The lines under `  key:` (two-space indent) in a top-level block, or None."""
    for start, line in enumerate(block):
        if re.match(rf"^  {re.escape(key)}:", line):
            end = start + 1
            while end < len(block) and (not block[end].strip() or block[end].startswith("    ") or block[end].lstrip().startswith("#")):
                end += 1
            return [line] + block[start + 1:end]
    return None


def lint(text):
    lines = text.splitlines()
    problems = []
    triggers = top_block(lines, "on")
    if triggers is None:
        return ["no `on:` block"]
    pull = sub_block(triggers, "pull_request")
    if pull is not None:
        body = "\n".join(pull)
        if not re.search(r"types:\s*\[\s*labeled\s*\]", body):
            problems.append("pull_request must have `types: [labeled]`")
        if re.search(r"^\s+paths(-ignore)?:", body, re.M):
            problems.append("pull_request must have no path filter")
    push = sub_block(triggers, "push")
    if push is not None:
        branches = re.findall(r"branches:\s*\[(.*?)\]", "\n".join(push))
        names = [name.strip().strip("\"'") for name in ",".join(branches).split(",") if name.strip()]
        if not names or any(name != "rust/next*" for name in names):
            problems.append(f"push may run only for rust/next*, not for {names or 'every branch'}")
    jobs = top_block(lines, "jobs") or []
    names = [m.group(1) for line in jobs if (m := re.match(r"^  ([\w-]+):\s*$", line))]
    if GUARD not in names:
        problems.append(f"no {GUARD} job")
    for name in names:
        block = sub_block(jobs, name)
        condition = next((line for line in block if line.startswith("    if:")), "")
        if name != GUARD and LAND not in condition:
            problems.append(f"job {name} has no `if:` on the added label land")
    group = "\n".join(top_block(lines, "concurrency") or [])
    if "'land'" not in group or "'other'" not in group:
        problems.append("the concurrency group must tell a land run from another label's run")
    return problems


def main(argv):
    path = Path(argv[0] if argv else ".gitea/workflows/ci.yaml")
    problems = lint(path.read_text())
    for problem in problems:
        print(f"{path}: {problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
