#!/usr/bin/env python3
"""Lint the triggers of ci.yaml: it runs once per pull request, when the label `land` is added, and tests nothing on a push.

A workflow file fails when:
  - its `on:` has a `pull_request` without `types: [labeled]`, or with a `paths` or `paths-ignore` filter (a filtered pull
    request never posts its required contexts, so it could never land);
  - its `on:` has a `push` to any branch but `rust/next*` (a push to main or an integration branch is not a second test of
    the tree the land run tested);
  - a job other than `land-guard` has no `if:` with the added label `land` as a top-level `&&` conjunct (or the whole condition);
    the conjunct may be `(github.event_name != 'pull_request' || <the label term>)`, which lets a push and a dispatch run;
    a job that runs only on a push (`github.event_name == 'push'` as such a conjunct) needs no label;
  - it has no `land-guard` job, or its concurrency group does not tell a land run from another label's run, or the whole
    `concurrency:` block is over 200 characters (Gitea keeps it in a varchar(255); a longer one fails the insert of the run).
Plain text is read, like scripts/lint-workflows.py: no YAML library.

usage: lint-ci-triggers.py [FILE]   (default .gitea/workflows/ci.yaml; exit 0: clean, 1: a violation)
"""

import re
import sys
from pathlib import Path

LAND = "contains(github.event.changes.added_labels.*.name, 'land')"
PUSH = "github.event_name == 'push'"
OTHER_EVENT = f"(github.event_name != 'pull_request' || {LAND})"
GUARD = "land-guard"
MAX_CONCURRENCY = 200


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


def conjuncts(condition):
    """The parts of an `if:` condition joined by top-level `&&`, or None when a top-level `||` makes it an alternative."""
    parts, depth, quote, start, index = [], 0, None, 0, 0
    while index < len(condition):
        char = condition[index]
        pair = condition[index:index + 2]
        if quote:
            quote = None if char == quote else quote
        elif char in "'\"":
            quote = char
        elif char in "([":
            depth += 1
        elif char in ")]":
            depth -= 1
        elif depth == 0 and pair == "||":
            return None
        elif depth == 0 and pair == "&&":
            parts.append(condition[start:index].strip())
            start = index + 2
            index += 1
        index += 1
    parts.append(condition[start:].strip())
    return parts


def gated(line):
    """Whether the `if:` line requires the label land or a push as a top-level conjunct."""
    condition = line.partition("if:")[2].strip()
    if condition.startswith("${{") and condition.endswith("}}"):
        condition = condition[3:-2].strip()
    parts = conjuncts(condition)
    return parts is not None and any(part in (LAND, PUSH, OTHER_EVENT) for part in parts)


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
        if name == GUARD:
            if condition or any(line.startswith("    needs:") for line in block):
                problems.append(f"{GUARD} must have no `needs:` and no `if:`: it runs on every run")
        elif not gated(condition):
            problems.append(f"job {name} has no `if:` with the added label land (or a push) as a top-level && conjunct")
    block = [line for line in (top_block(lines, "concurrency") or []) if line.strip() and not line.lstrip().startswith("#")]
    if LAND not in "\n".join(block):
        problems.append("the concurrency group must tell a land run from another label's run")
    size = len("concurrency:\n" + "\n".join(block))
    if size > MAX_CONCURRENCY:
        problems.append(f"the concurrency block is {size} characters, over {MAX_CONCURRENCY}: Gitea stores it in a varchar(255)")
    return problems


def main(argv):
    path = Path(argv[0] if argv else ".gitea/workflows/ci.yaml")
    problems = lint(path.read_text())
    for problem in problems:
        print(f"{path}: {problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
