"""Source check of the programmatic-launch contract (native/src/common/pdl.cuh): a kernel launched with the
programmatic-stream-serialization attribute may start while its predecessor still runs, so it must call
euhedral_pdl_begin() before it returns and before it reads or writes anything a predecessor touched.

Why not libclang: the Python binding is not installed on the CI hosts, and parsing these sources as CUDA needs the CUDA
toolkit headers and a preprocessor run over macro-generated kernels, which the host-only native tests do not have. The
check is a small structural parser instead; its known gaps are listed below.

The rule, for every __global__ function (macro-generated ones included) that reaches euhedral_pdl_begin(), directly or
through helpers it calls:
  1. The wait is a top-level statement of the function: braces around it balance, and the statement before it ended
     (so `if (c) euhedral_pdl_begin();` does not count).
  2. No `return` comes before it.
  3. Before it, no pointer parameter is used, read or written, except to compute an address. Which buffers a
     predecessor writes is not decided statically, so every pointer parameter counts. Allowed before the wait: scalars,
     blockIdx/threadIdx arithmetic, and declaring a local (pointer or struct of pointers) from a pointer by cast and
     offset arithmetic only; that local is then tracked as well. Not allowed: dereferencing (`*p`, `*(p + 1)`,
     `*reinterpret_cast<T*>(p)`), subscripting, a call with the pointer inside its arguments (`__ldg(p + 1)`),
     writing through it, and any statement using a tracked struct other than the declaring one.
  4. Exception: a `const`-qualified pointer parameter named in READ_ONLY (model weights) may be read before the wait.
     The condition is that no kernel writes a weight buffer while the model runs; a writable pointer of the same name
     is not exempt.
  5. A wait inside a helper is found by following calls. A callee is looked up in the kernel's own translation unit (its
     .cu file and the quoted includes), honouring namespaces; more than one definition matching a call is a failure. The
     helper is checked against its own pointer parameters. In the kernel, everything up to and including the call's
     arguments is checked against the kernel's: a pointer may be passed bare, but `*position` in the arguments is a use.
  6. Every kernel the Rust launch catalog registers (`true` entries) must reach the wait.

Known gaps: a type with a __device__ constructor that is declared from a pointer by `T x = T(p)` or in a member
initialiser is not seen (only `T x(p)` and `T x{p}` are); a pointer reaching a helper through a struct member or a
reference parameter is not bound; asm blocks other than exit/trap and calls that never return are not early exits;
helpers are followed only by name (no overload resolution, no function pointers, no `using namespace`);
pointers reaching a kernel through __constant__ globals or struct parameters are not tracked; arithmetic arguments like
`f(p + 1)` to a call are seen as a use, but a pointer hidden in a macro argument is not expanded; a kernel that never calls
euhedral_pdl_begin() is checked only if the catalog registers it.

Usage: pdl_wait_check.py [native/src directory] [rust catalog.rs]. Prints one line per violation; exit 1 on any."""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCES = ROOT / "native" / "src"
CATALOG = ROOT / "rust" / "crates" / "cuda" / "src" / "launch" / "catalog.rs"
WAIT = "euhedral_pdl_begin"
# Const pointer parameters that may be read before the wait: model weights, which no kernel writes.
READ_ONLY = {"weights", "weight_tensor"}
KEYWORDS = {"if", "for", "while", "switch", "return", "sizeof", "static_cast", "reinterpret_cast", "const_cast",
            "decltype", "alignof", "defined"}
TYPE_WORDS = (r"(?:const|unsigned|signed|int|long|short|char|float|double|bool|size_t|u?int(?:8|16|32|64)_t|"
              r"__nv_bfloat16|__half|void|\*|\s)+")


def read(path):
    return pathlib.Path(path).read_text(encoding="utf-8")


def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", lambda m: "\n" * m.group(0).count("\n"), text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text).replace("\\\n", " ")


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


class Function:
    def __init__(self, path, name, namespace, kernel, macro, pointers, body, params=()):
        self.path, self.name, self.namespace, self.kernel = path, name, namespace, kernel
        self.macro, self.pointers, self.body, self.params = macro, pointers, body, list(params)


def namespaces(text):
    """[(start, end, name)] of every named namespace block."""
    return [(m.start(), balanced(text, m.end() - 1, "{", "}"), m.group(1))
            for m in re.finditer(r"\bnamespace\s+(\w+)\s*\{", text)]


def macro_of(text, position):
    """(name, parameter names) of the macro whose definition (continuations joined) holds the position."""
    start = text.rfind("#define", 0, position)
    if start < 0:
        return None
    line_end = text.find("\n", start)
    if line_end != -1 and line_end < position:
        return None
    match = re.match(r"#define\s+(\w+)\(([^)]*)\)", text[start:])
    return (match.group(1), [a.strip() for a in match.group(2).split(",")]) if match else None


def parse(path, text):
    """Every __device__ or __global__ function defined in the (comment-free) text."""
    blocks = namespaces(text)
    found = []
    for match in re.finditer(r"__(device|global)__", text):
        position = match.end()
        while True:
            call = re.compile(r"[^;{(]*?([\w#]+)\s*\(", re.S).match(text, position)
            if call is None or call.group(1) != "__launch_bounds__":
                break
            position = balanced(text, call.end() - 1, "(", ")")
        if call is None:
            continue
        end_params = balanced(text, call.end() - 1, "(", ")")
        tail = re.compile(r"[^;{]*", re.S).match(text, end_params)
        if tail.end() >= len(text) or text[tail.end()] != "{":
            continue  # a declaration
        brace = tail.end()
        pointers, params = [], []
        for param in split_top_level(text[call.end():end_params - 1]):
            param = re.sub(r"\s*=.*$", "", re.sub(r"\[[^\]]*\]", "[]", param), flags=re.S)
            words = re.findall(r"\w+", param)
            params.append(words[-1] if words else "")
            if ("*" in param or "[]" in param) and words:
                cut = param.rindex("*") if "*" in param else param.index("[")
                pointers.append((words[-1], bool(re.search(r"\bconst\b", param[:cut]))))
        body = text[brace + 1:balanced(text, brace, "{", "}") - 1]
        namespace = tuple(name for start, end, name in blocks if start < match.start() < end)
        found.append(Function(path, call.group(1), namespace, match.group(1) == "global",
                              macro_of(text, match.start()), pointers, body, params))
    return found


class Tree:
    """The sources under a root: per-file functions, and the translation units (a .cu with its quoted includes)."""

    def __init__(self, root):
        self.root = pathlib.Path(root).resolve()
        self.paths = sorted(p for p in self.root.rglob("*") if p.suffix in (".cu", ".cuh"))
        self.texts = {p: strip_comments(read(p)) for p in self.paths}
        self.functions = {p: parse(p, t) for p, t in self.texts.items()}
        included = set()
        self.units = {}
        for path in self.paths:
            if path.suffix == ".cu":
                self.units[path] = self.closure(path)
                included |= self.units[path]
        for path in self.paths:
            if path not in included:
                self.units[path] = self.closure(path)

    def closure(self, path, seen=None):
        seen = set() if seen is None else seen
        if path in seen:
            return seen
        seen.add(path)
        for include in re.findall(r'#include\s+"([^"]+)"', self.texts[path]):
            for base in (path.parent, self.root):
                if (base / include).resolve() in self.texts:
                    self.closure((base / include).resolve(), seen)
                    break
        return seen

    def macro_bindings(self, name, parameters, active=()):
        """Every way the macro is instantiated, as parameter -> argument text, following a use that sits inside
        another macro's definition back to that macro's own uses."""
        found = []
        for text in self.texts.values():
            for use in re.finditer(r"\b" + name + r"\(([^()]*)\)", text):
                if re.search(r"#define\s+$", text[:use.start()]):
                    continue
                arguments = [a.strip() for a in use.group(1).split(",")]
                outer = macro_of(text, use.start())
                if outer is None:
                    found.append(dict(zip(parameters, arguments)))
                elif outer[0] != name and outer[0] not in active:
                    for binding in self.macro_bindings(outer[0], outer[1], active + (name,)):
                        found.append(dict(zip(parameters, [binding.get(a, a) for a in arguments])))
        return found

    def macro_owners(self, function):
        """The kernels a macro-generated function stands for, with ## pasted names spelled out. A macro used by
        another macro is expanded through the outer one's arguments."""
        if function.macro is None:
            return [function.name]
        owners = []
        for binding in self.macro_bindings(*function.macro):
            owner = function.name
            for parameter, argument in binding.items():
                owner = re.sub(r"\b" + parameter + r"\b", argument, owner)
            owners.append(owner.replace("##", ""))
        return owners or [function.name]


def init_braces(text):
    """Braces that open an initialiser (after a name, `=` or `>`), not a block, rewritten as parentheses so a
    declaration such as `const View v{input, n};` stays one statement."""
    out, stack = [], []
    for index, char in enumerate(text):
        if char == "{":
            previous = text[:index].rstrip()
            initialiser = bool(previous) and (previous[-1] in "=>" or (re.search(r"[\w]$", previous) is not None
                                              and not re.search(r"\b(?:else|do|try)$", previous)))
            stack.append(initialiser)
            out.append("(" if initialiser else "{")
        elif char == "}":
            out.append(")" if stack and stack.pop() else "}")
        else:
            out.append(char)
    return "".join(out)


def top_level(before):
    """Whether the wait that follows `before` is a statement of the function's own block."""
    if before.count("{") != before.count("}"):
        return False
    last = max(before.rfind(";"), before.rfind("{"), before.rfind("}"))
    return before[last + 1:].strip() == ""


DECL_TYPE = re.compile(r"[A-Za-z_]\w*")


def constructor_types(tree, unit):
    """Names of the types with a __device__ constructor in the translation unit."""
    names = set()
    for path in unit:
        names.update(re.findall(r"__device__\s+(?:__forceinline__\s+|inline\s+|constexpr\s+|explicit\s+)*(\w+)\s*\(",
                                tree.texts[path]))
    return names


def pointer_uses(text, pointers, constructors=()):
    """The tracked pointers that statements of `text` use beyond address arithmetic. Tracked: the pointer parameters
    (except const ones in READ_ONLY) and locals declared from tracked names by cast and offset arithmetic."""
    tracked = {name for name, const in pointers if not (const and name in READ_ONLY)}
    used = []
    for statement in re.split(r"[;{}]", init_braces(text)):
        statement = statement.strip()
        names = [n for n in sorted(tracked) if re.search(r"\b" + re.escape(n) + r"\b", statement)]
        if not names:
            continue
        split = re.search(r"(?<![=!<>])=(?!=)|\(", statement)
        if split:
            declared = re.sub(r"<[^>]*>", "", statement[:split.start()])
            tokens = re.findall(r"[\w:]+|\*", declared)
            plain = not re.search(r"[\[\(.]|->", declared)
            if (plain and len(tokens) >= 2 and DECL_TYPE.fullmatch(tokens[-1])
                    and not KEYWORDS.intersection(tokens) and any(DECL_TYPE.match(t) for t in tokens[:-1])
                    and tokens[-1] not in tracked):
                right = statement[split.end():] if statement[split.start()] == "=" else statement[split.start():]
                if (statement[split.start()] == "(" and "*" not in declared and len(tokens) >= 2
                        and tokens[-2] in constructors):
                    used.extend(n for n in names if n not in used)  # the constructor may read through the pointer
                    continue
                if _address_only(right, names, tracked):
                    tracked.add(tokens[-1])
                    continue
        used.extend(n for n in names if n not in used)
    return used


def _address_only(right, names, tracked):
    right = re.sub(r"(?:reinterpret_cast|static_cast|const_cast)\s*<[^>]*>", "", right)
    right = re.sub(r"\(\s*" + TYPE_WORDS + r"\)", " ", right)  # C casts of arithmetic and pointer types
    if "[" in right or "->" in right:
        return False
    if re.search(r"(?:^|[-+*/%(,<>=&|!?:])\s*\*", right):  # a unary star
        return False
    for call in re.finditer(r"([A-Za-z_]\w*)\s*\(", right):
        if call.group(1) in KEYWORDS:
            continue
        arguments = right[call.end():balanced(right, call.end() - 1, "(", ")") - 1]
        if any(re.search(r"\b" + re.escape(n) + r"\b", arguments) for n in tracked):
            return False
    return True


def own_problems(before, upto, pointers, constructors=()):
    found = []
    if not top_level(before):
        found.append("euhedral_pdl_begin() is not a top-level statement of the kernel")
    if re.search(r"\breturn\b|__trap\s*\(|\basm\b(?:\s+volatile)?\s*\(\s*\"\s*(?:exit|trap)\s*;?\s*\"", upto):
        found.append("returns before euhedral_pdl_begin()")
    used = pointer_uses(upto, pointers, constructors)
    if used:
        found.append(f"uses pointer parameter(s) {', '.join(used)} before euhedral_pdl_begin()")
    return found


def candidates(tree, unit, caller, callee):
    segments = callee.split("::")
    found = [f for path in unit for f in tree.functions[path] if f.name == segments[-1]]
    if len(segments) > 1:
        return [f for f in found if (f.namespace + (f.name,))[-len(segments):] == tuple(segments)]
    for depth in range(len(caller.namespace), -1, -1):
        level = [f for f in found if f.namespace == caller.namespace[:depth]]
        if level:
            return level
    return found


def analyze(tree, unit, function, stack=(), pointers=None):
    """None when the function never reaches the wait, else its problems. `pointers` are the (name, const) pairs
    tracked in it: its pointer parameters plus, for a helper, the parameters bound to tracked arguments of the call."""
    pointers = function.pointers if pointers is None else pointers
    constructors = constructor_types(tree, unit)
    body = function.body
    index = body.find(WAIT)
    if index >= 0:
        return own_problems(body[:index], body[:index], pointers, constructors)
    tracked = {n for n, c in pointers if not (c and n in READ_ONLY)}
    for call in re.finditer(r"((?:\w+::)*[A-Za-z_]\w*)\s*(?:<[^;(){}]*>)?\s*\(", body):
        callee = call.group(1)
        if callee.split("::")[-1] in KEYWORDS or callee in stack:
            continue
        open_paren = body.index("(", call.end() - 1)
        close = balanced(body, open_paren, "(", ")")
        arguments = split_top_level(body[open_paren + 1:close - 1])
        found = [c for c in candidates(tree, unit, function, callee) if c is not function]
        results = []
        for c in found:
            effective = list(c.pointers)
            known = {n for n, _ in effective}
            for position, argument in enumerate(arguments[:len(c.params)]):
                name = c.params[position]
                if name and name not in known and any(re.search(r"\b" + re.escape(t) + r"\b", argument) for t in tracked):
                    effective.append((name, False))
            results.append((c, analyze(tree, unit, c, stack + (callee,), effective)))
        waiting = [(c, r) for c, r in results if r is not None]
        if not waiting:
            continue
        if len(found) > 1:
            return [f"call of {callee}: {len(found)} definitions match in the translation unit"]
        problems = waiting[0][1]
        rest = ";".join(a for a in arguments if a not in tracked)
        return problems + own_problems(body[:call.start()], body[:call.start()] + ";" + rest, pointers, constructors)
    return None


def registered_names(catalog):
    path = pathlib.Path(catalog)
    if not path.is_file():
        raise FileNotFoundError(f"the launch catalog {path} is missing")
    text = read(path)
    entries = re.findall(r'spec\(\s*Family::\w+,\s*"(\w+)",\s*Module::\w+,\s*\w+,\s*(true|false),?\s*\)', text)
    calls = len(re.findall(r"^\s+spec\(\s*Family::", text, re.M))
    if len(entries) != calls:
        raise ValueError(f"parsed {len(entries)} catalog entries of {calls}")
    return [name for name, flag in entries if flag == "true"]


def violations(source_root, catalog=None):
    tree = Tree(source_root)
    problems, with_wait, seen = [], set(), set()
    for unit_path, unit in tree.units.items():
        for path in sorted(unit):
            for function in tree.functions[path]:
                if not function.kernel or (path, function.name, function.body) in seen:
                    continue
                result = analyze(tree, unit, function)
                if result is None:
                    continue
                seen.add((path, function.name, function.body))
                owners = tree.macro_owners(function)
                with_wait.update(owners)
                for message in result:
                    problems.append(f"{path.relative_to(tree.root)}: {', '.join(owners)}: {message}")
    wanted = set()
    if catalog is not None:
        wanted |= set(registered_names(catalog))
    for name in sorted(wanted - with_wait):
        problems.append(f"{name}: registered for programmatic launch but never reaches euhedral_pdl_begin()")
    return sorted(set(problems))


if __name__ == "__main__":
    root = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else SOURCES
    catalog = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else CATALOG
    found = violations(root, catalog)
    print("\n".join(found) if found else "ok")
    sys.exit(1 if found else 0)
