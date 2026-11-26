#!/usr/bin/env python3
"""Lint the arbiter requests of the workflows: no job may ask for more than 32 GiB, or for CPU.

Every `ARBITER_ARGS` value and every `system-run` / `gpu-run` command line in the given workflow files (or directories)
is read, shell continuation lines joined. A request fails when it:
  - uses any `--` option but --mem --est --spec --res --gpu --disk --headset --threads --who --background --tool-timeout
    --no-learned-cap (exact names, also as `--name=value`; the arbiter's argument parser accepts abbreviations such as
    `--me 64G`, so those are refused), or any single-dash option;
  - asks for more than 32G through `--mem`, or through `mem=` in `--spec` / `--res`, or asks for no cap
    (0, none, off, unlimited, inf, infinity, max), or gives a size it cannot read (no unit, a variable);
  - has an item `system` (the whole machine) or anything containing `cpu` in `--spec` / `--res`;
  - sets ARBITER_ARGS (plain, quoted or quoted-key, in a block or a shell line) to anything but a one-line plain or fully
    quoted value with no `$`, `\\` or `${{ }}` in it: a block or flow scalar, a folded plain scalar continued on a more
    indented line, a tag, an anchor, or an expression is refused, not parsed.
The machine ran out of memory at 54G once, and the arbiter's cpu forms are refused by the runner, so the workflow is the
cheapest place to stop both. This file is the canonical copy: the same bytes are used by every repository.

usage: lint-workflows.py PATH...   (exit 0: clean, 1: a violation, 2: nothing to read)
"""

import re
import shlex
import sys
from pathlib import Path

MAX_BYTES = 32 * 1024**3
UNITS = {"": None, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
SIZE = re.compile(r"^(\d+(?:\.\d+)?)([KMGT]?)(?:i?B)?$", re.IGNORECASE)
NO_CAP = ("0", "none", "off", "unlimited", "inf", "infinity", "max")
# Option -> does it take a value? (--gpu and --disk may have none.)
OPTIONS = {
    "--mem": True, "--est": True, "--spec": True, "--res": True, "--threads": True, "--who": True,
    "--tool-timeout": True, "--gpu": None, "--disk": None, "--headset": False, "--background": False,
    "--no-learned-cap": False,
}
ARBITER_ARGS_REFERENCE = re.compile(r"^\$(?:ARBITER_ARGS|\{ARBITER_ARGS\})$")


def size_bytes(text, bare_unit=None):
    """Bytes of a size such as 32G or 512M; a bare number counts in `bare_unit` (None: refused)."""
    if text.lower() in NO_CAP:
        raise ValueError("no memory cap")
    match = SIZE.match(text)
    if not match:
        raise ValueError(f"cannot read the size {text!r}")
    number, unit = float(match.group(1)), match.group(2).upper()
    factor = UNITS[unit] if unit else bare_unit
    if factor is None:
        raise ValueError(f"the size {text!r} has no unit")
    if number == 0:
        raise ValueError("no memory cap")
    return number * factor


def check_mem(text, bare_unit=None):
    if size_bytes(text, bare_unit) > MAX_BYTES:
        raise ValueError(f"asks for {text}, more than 32G")


def check_item(item):
    """One NAME=MODE[:AMT] resource of --spec or --res."""
    lowered = item.lower()
    if "cpu" in lowered:
        raise ValueError("a cpu form is refused")
    if lowered == "system":
        raise ValueError("system asks for the whole machine")
    name, _, mode = lowered.partition("=")
    if name.strip() == "mem":
        check_mem(mode.rsplit(":", 1)[-1], bare_unit=1024**3)


def check_args(args):
    """Raise ValueError for the first violation in an arbiter argument string."""
    tokens = shlex.split(args)
    i = 0
    while i < len(tokens):
        token = tokens[i]
        i += 1
        if ARBITER_ARGS_REFERENCE.match(token):
            continue
        if not token.startswith("-"):
            if "$" in token or "`" in token:
                raise ValueError(f"cannot read {token!r}")
            continue
        name, eq, inline = token.partition("=")
        if name not in OPTIONS:
            raise ValueError(f"the option {name} is not one of the allowed ones (abbreviations are refused)")
        takes_value = OPTIONS[name]
        if takes_value is False:
            continue
        if eq:
            value = inline
        elif takes_value is None:
            if i < len(tokens) and not tokens[i].startswith("-") and not ARBITER_ARGS_REFERENCE.match(tokens[i]):
                value = tokens[i]
                i += 1
            else:
                continue
        else:
            if i >= len(tokens):
                raise ValueError(f"{name} has no value")
            value = tokens[i]
            i += 1
        if "$" in value or "`" in value:
            raise ValueError(f"cannot read the value of {name}")
        if name == "--mem":
            check_mem(value)
        elif name in ("--spec", "--res"):
            for item in re.split(r"[,\s]+", value.strip()):
                if item:
                    check_item(item)
        elif name == "--gpu" and "cpu" in value.lower():
            raise ValueError("a cpu form is refused")


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
    if "\\" in value or "$" in value or "`" in value:
        raise ValueError("ARBITER_ARGS has an escape or a variable, which cannot be checked")
    return value


def indent_of(line):
    stripped = line.lstrip(" ")
    width = len(line) - len(stripped)
    return width + 2 if stripped.startswith("- ") else width


def logical_lines(text):
    """(number, joined text, raw lines index) with backslash-continued shell lines joined."""
    raw = text.splitlines()
    out, i = [], 0
    while i < len(raw):
        start, joined = i, raw[i]
        while joined.rstrip().endswith("\\") and i + 1 < len(raw):
            i += 1
            joined = joined.rstrip()[:-1] + " " + raw[i].strip()
        out.append((start, joined))
        i += 1
    return raw, out


def requests(raw, index, line):
    """The argument strings a workflow line gives the arbiter."""
    code = line.strip()
    if code.startswith("#"):
        return
    match = re.search(r"""["']?\bARBITER_ARGS\b["']?\s*([:=])\s*(.*)$""", code)
    if match:
        yield plain_scalar(match.group(2).strip())
        if match.group(1) == ":":
            for following in raw[index + 1:]:
                if following.strip() and not following.lstrip().startswith("#"):
                    if indent_of(following) > indent_of(raw[index]):
                        raise ValueError("ARBITER_ARGS is continued on the next line, which cannot be checked")
                    break
    for found in re.finditer(r"\b(?:system-run|gpu-run)\b(.*?)(?:\s--(?:\s|$)|$)", code):
        yield found.group(1)


def lint(path):
    problems = []
    raw, lines = logical_lines(path.read_text())
    for index, line in lines:
        try:
            for args in requests(raw, index, line):
                check_args(args)
        except ValueError as error:
            problems.append(f"{path}:{index + 1}: {error}")
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
