#!/usr/bin/env python3
"""Lint the arbiter requests of the workflows: no job may ask for more than 32 GiB, or for CPU.

Every `ARBITER_ARGS` value and every `system-run` command line in the given workflow files (or directories) is read.
A request fails when it asks for more than --mem 32G (also through `--spec mem=...` or `--res mem=...`), for no memory
cap (`--mem 0`, `--mem none`), for the whole machine (`--system`), for any cpu form, or when its value cannot be read
(a `${{ }}` expression, a size with no unit). The machine ran out of memory at 54G once, and the arbiter's cpu forms are
refused by the runner, so the workflow is the cheapest place to stop both.

usage: lint-workflows.py PATH...   (exit 0: clean, 1: a violation, 2: nothing to read)
"""

import re
import shlex
import sys
from pathlib import Path

MAX_BYTES = 32 * 1024**3
UNITS = {"": None, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
SIZE = re.compile(r"^(\d+(?:\.\d+)?)([KMGT]?)(?:i?B)?$", re.IGNORECASE)


def size_bytes(text, bare_unit=None):
    """Bytes of a size such as 32G or 512M; a bare number counts in `bare_unit` (None: refused)."""
    if text.lower() in ("0", "none"):
        raise ValueError("no memory cap")
    match = SIZE.match(text)
    if not match:
        raise ValueError(f"cannot read the size {text!r}")
    number, unit = float(match.group(1)), match.group(2).upper()
    factor = UNITS[unit] if unit else bare_unit
    if factor is None:
        raise ValueError(f"the size {text!r} has no unit")
    return number * factor


def check_mem(text, bare_unit=None):
    value = size_bytes(text, bare_unit)
    if value > MAX_BYTES:
        raise ValueError(f"asks for {text}, more than 32G")


def check_item(item):
    """One NAME=MODE[:AMT] resource of --spec or --res."""
    name, _, mode = item.partition("=")
    name = name.strip().lower()
    if name == "cpu":
        raise ValueError("a cpu form is refused")
    if name == "mem":
        check_mem(mode.rsplit(":", 1)[-1], bare_unit=1024**3)


def check_args(args):
    """Raise ValueError for the first violation in an arbiter argument string."""
    tokens = shlex.split(args)
    i = 0
    while i < len(tokens):
        token = tokens[i]
        name, eq, inline = token.partition("=")
        if name == "--system":
            raise ValueError("--system asks for the whole machine")
        if name == "--cpu" or token.startswith("--cpu"):
            raise ValueError("a cpu form is refused")
        if name in ("--mem", "--spec", "--res"):
            if eq:
                value = inline
            else:
                i += 1
                if i >= len(tokens):
                    raise ValueError(f"{name} has no value")
                value = tokens[i]
            if name == "--mem":
                check_mem(value)
            else:
                for item in re.split(r"[,\s]+", value.strip()):
                    if item:
                        check_item(item)
        i += 1


def plain_scalar(value):
    """The text of a one-line plain or fully quoted scalar; anything else cannot be checked and is refused."""
    if value[:1] in "\"'":
        match = re.fullmatch(r"(['\"])(.*)\1\s*(?:#.*)?", value)
        if not match:
            raise ValueError("ARBITER_ARGS is not one fully quoted value")
        value = match.group(2)
    else:
        value = re.sub(r"\s+#.*$", "", value)
        if not value or value[0] in ">|{[&*!%@`" or re.search(r"[{}]|,\s*\w+\s*:", value):
            raise ValueError("ARBITER_ARGS is not a one-line plain or quoted value")
    if "${{" in value:
        raise ValueError("ARBITER_ARGS is an expression, which cannot be checked")
    return value


def requests(line):
    """The argument strings a workflow line gives the arbiter."""
    code = line.strip()
    if code.startswith("#"):
        return
    match = re.search(r"\bARBITER_ARGS\b\s*[:=]\s*(.*)$", code)
    if match:
        yield plain_scalar(match.group(1).strip())
    match = re.search(r"\bsystem-run\b(.*?)(?:\s--(?:\s|$)|$)", code)
    if match and "ARBITER_ARGS" not in code:
        yield match.group(1)


def lint(path):
    problems = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        try:
            for args in requests(line):
                check_args(args)
        except ValueError as error:
            problems.append(f"{path}:{number}: {error}")
    return problems


def main(argv):
    files = []
    for name in argv:
        path = Path(name)
        if path.is_dir():
            files += sorted(p for p in path.rglob("*") if p.suffix in (".yaml", ".yml"))
        elif path.is_file():
            files.append(path)
    if not files:
        print("lint-workflows: no workflow files to read", file=sys.stderr)
        return 2
    problems = [problem for path in files for problem in lint(path)]
    for problem in problems:
        print(problem, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
