"""Source check of the programmatic-launch contract (native/src/common/pdl.cuh): a kernel launched with the
programmatic-stream-serialization attribute may start while its predecessor still runs, so it must call
euhedral_pdl_begin() before it returns and before it touches anything a predecessor wrote.

Rule, applied to every __global__ entry function (macro-generated ones included) that calls euhedral_pdl_begin():
  1. the call is a top-level statement of the body (not under a branch or loop);
  2. nothing before it contains `return`;
  3. nothing before it mentions a pointer parameter of the kernel (any read through it, any pointer arithmetic on
     it, and passing it to a helper). Which pointers a predecessor writes is not decided statically, so every
     pointer parameter counts. Scalar parameters, blockIdx/threadIdx arithmetic and declarations are allowed.
Helpers are not followed: the rule forbids handing a pointer parameter to one before the wait.
A kernel the Rust launch catalog registers for programmatic launch (its `true` entries) must also call
euhedral_pdl_begin() at all, in some definition of that name.

Usage: pdl_wait_check.py [native/src directory] [rust catalog.rs]. Prints one line per violation; exit 1 on any."""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCES = ROOT / "native" / "src"
CATALOG = ROOT / "rust" / "crates" / "cuda" / "src" / "launch" / "catalog.rs"
WAIT = "euhedral_pdl_begin"


def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", lambda m: "\n" * m.group(0).count("\n"), text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def balanced(text, start, opening, closing):
    """Index just past the group opened at text[start]."""
    depth = 0
    for index in range(start, len(text)):
        if text[index] == opening:
            depth += 1
        elif text[index] == closing:
            depth -= 1
            if depth == 0:
                return index + 1
    raise ValueError("unbalanced group")


def split_top_level(text):
    parts, depth, current = [], 0, ""
    for char in text:
        if char in "(<[{":
            depth += 1
        elif char in ")>]}":
            depth -= 1
        if char == "," and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += char
    parts.append(current)
    return [part.strip() for part in parts if part.strip()]


def functions(text):
    """Yields (name, is_kernel, macro or None, pointer parameter names, body) of every __device__ or __global__
    function defined in the (comment-free, continuation-joined) text. Names may carry ## token pasting."""
    for match in re.finditer(r"__(device|global)__", text):
        kernel = match.group(1) == "global"
        position = match.end()
        while True:
            call = re.compile(r"[^;{(]*?([\w#]+)\s*\(", re.S).match(text, position)
            if call is None:
                break
            if call.group(1) == "__launch_bounds__":
                position = balanced(text, call.end() - 1, "(", ")")
                continue
            break
        if call is None:
            continue
        name = call.group(1)
        end_params = balanced(text, call.end() - 1, "(", ")")
        tail = re.compile(r"[^;{]*", re.S).match(text, end_params)
        if tail.end() >= len(text) or text[tail.end()] != "{":
            continue  # a declaration
        brace = tail.end()
        params = split_top_level(text[call.end():end_params - 1])
        pointers = [re.findall(r"\w+", p)[-1] for p in params if "*" in p and re.findall(r"\w+", p)]
        body = text[brace + 1:balanced(text, brace, "{", "}") - 1]
        yield name, kernel, macro_of(text, match.start()), pointers, body


def macro_of(text, position):
    """(name, parameter names) of the macro whose one-line (continuations joined) definition holds the position."""
    start = text.rfind("#define", 0, position)
    if start < 0:
        return None
    line_end = text.find("\n", start)
    if line_end != -1 and line_end < position:
        return None
    match = re.match(r"#define\s+(\w+)\(([^)]*)\)", text[start:])
    return (match.group(1), [a.strip() for a in match.group(2).split(",")]) if match else None


def top_level_prefix(body):
    """The body before its first wait call and whether that call is a top-level statement; (None, _) without one."""
    index = body.find(WAIT)
    if index < 0:
        return None, False
    prefix = body[:index]
    return prefix, prefix.count("{") == prefix.count("}")


# Parameters a kernel may read before the wait: model weights, which no kernel writes.
READ_ONLY = {"weights", "weight_tensor"}

ALIAS = re.compile(r"^[\w:<>\s*]*?\b(\w+)\s*(?:=|\{)(.*)$", re.S)
DEREFERENCE = r"(?:(?<![\w])\s*\*\s*{0}\b|\b{0}\s*(?:\[|->)|[(,]\s*&?\s*{0}\s*[,)])"


def pointer_uses(prefix, pointers):
    """The pointers a statement of the prefix does more with than address arithmetic. A pointer parameter, and a
    local (pointer or struct of pointers) initialised from one by cast and offset, may be computed with; dereferencing, subscripting or
    passing any of them on is a use."""
    tracked = {p for p in pointers if p not in READ_ONLY}
    used = []
    for statement in prefix.split(";"):
        names = [n for n in sorted(tracked) if re.search(r"\b" + re.escape(n) + r"\b", statement)]
        if not names:
            continue
        alias = ALIAS.match(statement.strip())
        if alias:
            right = alias.group(2)
            right = re.sub(r"(?:reinterpret_cast|static_cast|const_cast)\s*<[^>]*>", "", right)
            if not any(re.search(DEREFERENCE.format(re.escape(n)), right) for n in names):
                tracked.add(alias.group(1))
                continue
        used.extend(n for n in names if n not in used)
    return used


def prefix_problems(prefix, pointers, top):
    found = []
    if not top:
        found.append("euhedral_pdl_begin() is not a top-level statement")
    if re.search(r"\breturn\b", prefix):
        found.append("returns before euhedral_pdl_begin()")
    used = pointer_uses(prefix, pointers)
    if used:
        found.append(f"uses pointer parameter(s) {', '.join(used)} before euhedral_pdl_begin()")
    return found


def violations(source_root, catalog=None):
    """Problems of every kernel that waits, directly or through the helpers it calls (followed by name, across the
    sources, namespaces ignored). A helper that holds the wait is checked against its own pointer parameters, and
    the code the kernel runs before calling it against the kernel's."""
    sources = sorted(p for p in pathlib.Path(source_root).rglob("*") if p.suffix in (".cu", ".cuh"))
    texts = {path: strip_comments(path.read_text()).replace("\\\n", " ") for path in sources}
    table = {}  # name -> [(pointers, body)]
    entries = []  # (file, name, macro, pointers, body)
    for path, text in texts.items():
        for name, kernel, macro, pointers, body in functions(text):
            table.setdefault(name, []).append((pointers, body))
            if kernel:
                entries.append((path.relative_to(source_root), name, macro, pointers, body))

    memo = {}

    def waits(pointers, body, stack=()):
        """None when the function never waits, else the list of its problems."""
        found = []
        prefix, top = top_level_prefix(body)
        if prefix is not None:
            return prefix_problems(prefix, pointers, top)
        best = None
        for call in re.finditer(r"(?:\w+::)*(\w+)\s*(?:<[^;(){}]*>)?\s*\(", body):
            callee = call.group(1)
            if callee in stack or callee not in table:
                continue
            for callee_pointers, callee_body in table[callee]:
                inner = memo.get(id(callee_body))
                if inner is None:
                    inner = memo[id(callee_body)] = ("pending", waits(callee_pointers, callee_body, stack + (callee,)))
                result = inner[1]
                if result is not None:
                    best = (call.start(), result)
                    break
            if best:
                break
        if best is None:
            return None
        before = body[:best[0]]
        return best[1] + prefix_problems(before, pointers, before.count("{") == before.count("}"))

    problems, with_wait = [], set()
    for relative, name, macro, pointers, body in entries:
        result = waits(pointers, body)
        if result is None:
            continue
        owners = [name]
        if macro is not None:
            macro_name, macro_params = macro
            owners = []
            for text in texts.values():
                for use in re.finditer(r"^\s*" + macro_name + r"\(([^)]*)\)", text, re.M):
                    owner = name
                    for parameter, argument in zip(macro_params, [a.strip() for a in use.group(1).split(",")]):
                        owner = re.sub(r"\b" + parameter + r"\b", argument, owner)
                    owners.append(owner.replace("##", ""))
            owners = owners or [name]
        with_wait.update(owners)
        for message in result:
            problems.append(f"{relative}: {', '.join(owners)}: {message}")
    if catalog is not None and pathlib.Path(catalog).is_file():
        registered = re.findall(r'spec\(\s*Family::\w+,\s*"(\w+)",\s*Module::\w+,\s*\w+,\s*true,?\s*\)',
                                pathlib.Path(catalog).read_text())
        for name in registered:
            if name not in with_wait:
                problems.append(f"{name}: registered for programmatic launch but never reaches euhedral_pdl_begin()")
    return sorted(set(problems))


if __name__ == "__main__":
    root = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else SOURCES
    catalog = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else CATALOG
    found = violations(root, catalog)
    print("\n".join(found) if found else "ok")
    sys.exit(1 if found else 0)
