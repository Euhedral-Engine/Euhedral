#!/usr/bin/env bash
# The layer check: dependencies between the workspace's crates point one way, a few external crates are confined to
# the crates that own them, and the no-wait clippy rules apply to every library crate.
#
#   scripts/check-layers.sh              check the workspace (exit 1 on a violation)
#   scripts/check-layers.sh --self-test  check the workspace, and prove the check fails on planted violations
#
# Part of the merge gate (mise task `rust-layers`, the `Layers` step of rust.yaml). Needs python3; reads
# `cargo metadata` (no build). Crates in the table below that do not exist yet are ignored, so a rule is in force
# before its crate lands; a workspace crate that is not in the table fails the check until it is placed.
set -euo pipefail
cd "$(dirname "$0")/.."

if command -v cargo >/dev/null 2>&1; then
    metadata="$(cargo metadata --format-version 1 --no-deps)"
else
    metadata="$(mise exec -- cargo metadata --format-version 1 --no-deps)"
fi

python3 -I - "$@" "$metadata" <<'PY'
import copy
import fnmatch
import json
import os
import sys

SETTINGS = "euhedral-inference-settings"
EDRL = "euhedral-inference-edrl"
CUDA = "euhedral-inference-cuda"
HOST_GEMV = "euhedral-inference-host-gemv"
TOKENIZER = "euhedral-inference-tokenizer"
ENGINE = "euhedral-inference-engine"
BENCH = "euhedral-inference-bench"
TEST_SUPPORT = "test-support"
XTASK = "xtask"

# Which dependencies count: normal and build dependencies, in every target section. Dev-dependencies are ignored by
# every rule below, so a test may use any crate (test-support is a dev-dependency only, which this makes a rule).
#
# crate -> workspace crates it may depend on.
ALLOWED = {
    SETTINGS: set(),
    EDRL: set(),
    HOST_GEMV: set(),
    TEST_SUPPORT: set(),
    TOKENIZER: {EDRL},
    CUDA: set(),
    ENGINE: {SETTINGS, EDRL, CUDA, HOST_GEMV, TOKENIZER},
    BENCH: {ENGINE, SETTINGS, EDRL, CUDA, HOST_GEMV, TOKENIZER},
    XTASK: set(),
}
# Binaries: their main thread may block, so they do not carry the no-wait clippy.toml. Every other crate does.
BINARIES = {BENCH, XTASK}
# Crates that no crate may depend on except as a dev-dependency.
DEV_ONLY = {TEST_SUPPORT}

# external crate -> crates that may depend on it.
RUSTEE = {"euhedral-core", "euhedral-graph"}
RUSTEE_OWNERS = {ENGINE, BENCH, CUDA}
# Configuration is parsed in one place.
YAML = "serde_yaml_ng"
YAML_OWNERS = {SETTINGS}
# crate -> the only external crates it may depend on.
EXTERNAL_ONLY = {CUDA: {"euhedral-graph", "libloading", "thiserror"}}
# External crates no crate may depend on (glob patterns), with the rule each protects.
DENIED = {
    "level-zero*": "hardware agnosticism: no Intel iGPU-specific code; integrated-GPU compute goes through a "
                   "vendor-neutral API chosen at run time",
    "opencl3": "hardware agnosticism: no OpenCL; compute goes through CUDA or a vendor-neutral API",
    "ocl": "hardware agnosticism: no OpenCL; compute goes through CUDA or a vendor-neutral API",
    "core_affinity": "worker CPUs are never pinned or restricted; RustEE places the threads",
}

CANONICAL = "clippy-no-wait.toml"


def check(meta, clippy_texts):
    errors = []
    names = {p["name"] for p in meta["packages"]}
    for p in meta["packages"]:
        name = p["name"]
        if name not in ALLOWED:
            errors.append(f"{name} is not in the layer table of scripts/check-layers.sh: place it")
            continue
        normal = {d["name"] for d in p["dependencies"] if d.get("kind") in (None, "build")}
        local = {n for n in normal if n in names}
        external = {n for n in normal if n not in names}

        bad = local - ALLOWED[name]
        if bad:
            errors.append(f"{name} may not depend on {sorted(bad)} (allowed: {sorted(ALLOWED[name]) or 'none'})")
        for dev_only in DEV_ONLY & local:
            errors.append(f"{name} depends on {dev_only}, which is a dev-dependency only")

        if YAML in normal and name not in YAML_OWNERS:
            errors.append(f"{name} may not depend on {YAML}: configuration is parsed in {SETTINGS} only")
        if (RUSTEE & external) and name not in RUSTEE_OWNERS:
            errors.append(f"{name} may not depend on {sorted(RUSTEE & external)}: only {sorted(RUSTEE_OWNERS)} do")
        for pattern, why in DENIED.items():
            for denied in sorted(n for n in external if fnmatch.fnmatchcase(n, pattern)):
                errors.append(f"{name} may not depend on {denied}: {why}")
        if name in EXTERNAL_ONLY and external - EXTERNAL_ONLY[name]:
            errors.append(f"{name} may depend externally only on {sorted(EXTERNAL_ONLY[name])}, "
                          f"has {sorted(external - EXTERNAL_ONLY[name])}")

        if name not in BINARIES:
            text = clippy_texts.get(name)
            if text is None:
                errors.append(f"{name} has no clippy.toml (the no-wait rules)")
            elif text != clippy_texts["__canonical__"]:
                errors.append(f"{name}'s clippy.toml differs from {CANONICAL}")
    return errors


def clippy_for(meta):
    texts = {"__canonical__": open(CANONICAL).read()}
    for p in meta["packages"]:
        f = os.path.join(os.path.dirname(p["manifest_path"]), "clippy.toml")
        if os.path.exists(f):
            texts[p["name"]] = open(f).read()
    return texts


def add_dependency(meta, crate, dependency, kind=None, target=None):
    for p in meta["packages"]:
        if p["name"] == crate:
            p["dependencies"].append({"name": dependency, "kind": kind, "target": target})
            return
    raise SystemExit(f"self-test: {crate} is missing from the workspace")


def self_test(meta, texts):
    # (description, edited metadata, edited clippy texts, a phrase the errors must contain)
    cases = []

    def plant(description, phrase, edge=None, text=None):
        planted, planted_texts = copy.deepcopy(meta), dict(texts)
        if edge:
            add_dependency(planted, *edge)
        if text:
            crate, value = text
            planted_texts[crate] = value
            if value is None:
                del planted_texts[crate]
        cases.append((description, planted, planted_texts, phrase))

    plant("a build dependency outside the table", f"{EDRL} may not depend on ['{SETTINGS}']",
          edge=(EDRL, SETTINGS, "build"))
    plant("a target-specific dependency outside the table", f"{EDRL} may not depend on ['{CUDA}']",
          edge=(EDRL, CUDA, None, "cfg(windows)"))
    plant("a denied crate (level-zero)", "may not depend on level-zero-sys", edge=(EDRL, "level-zero-sys"))
    plant("a denied crate (core_affinity)", "may not depend on core_affinity", edge=(SETTINGS, "core_affinity"))
    plant("a workspace edge outside the table", f"{EDRL} may not depend on", edge=(EDRL, CUDA))
    plant("a dependency on the cuda crate from settings", f"{SETTINGS} may not depend on", edge=(SETTINGS, CUDA))
    plant("test-support as a normal dependency", "dev-dependency only", edge=(EDRL, TEST_SUPPORT))
    plant("serde_yaml_ng outside settings", f"may not depend on {YAML}", edge=(EDRL, YAML))
    plant("euhedral-graph outside engine, bench and cuda", "may not depend on ['euhedral-graph']",
          edge=(SETTINGS, "euhedral-graph"))
    plant("an extra external crate in cuda", f"{CUDA} may depend externally only on", edge=(CUDA, "serde_json"))
    plant("a drifted clippy.toml", f"{EDRL}'s clippy.toml differs", text=(EDRL, texts["__canonical__"] + "\n# drift\n"))
    plant("a missing clippy.toml", f"{SETTINGS} has no clippy.toml", text=(SETTINGS, None))
    unplaced = copy.deepcopy(meta)
    unplaced["packages"].append({"name": "euhedral-inference-unplaced", "dependencies": [],
                                 "manifest_path": "/nonexistent/Cargo.toml"})
    cases.append(("a crate missing from the table", unplaced, dict(texts), "is not in the layer table"))

    ok = True
    for description, planted, planted_texts, phrase in cases:
        found = check(planted, planted_texts)
        if not any(phrase in e for e in found):
            print(f"self-test: NOT caught: {description} (expected {phrase!r}, got {found})", file=sys.stderr)
            ok = False
    # Dev-dependencies are ignored: a test may use any crate, and test-support is a dev-dependency only.
    for crate, dependency in ((EDRL, CUDA), (EDRL, TEST_SUPPORT), (SETTINGS, "serde_json"), (EDRL, "opencl3")):
        planted = copy.deepcopy(meta)
        add_dependency(planted, crate, dependency, "dev")
        if check(planted, texts) != check(meta, texts):
            print(f"self-test: a dev-dependency {crate} -> {dependency} changed the result", file=sys.stderr)
            ok = False
    # The unmodified workspace passes, and a table entry for an absent crate is ignored.
    if check(meta, texts):
        print("self-test: the unmodified workspace fails", file=sys.stderr)
        ok = False
    print("self-test:", f"all {len(cases)} planted violations are caught" if ok else "FAILED", file=sys.stderr)
    return ok


meta = json.loads(sys.argv[-1])
texts = clippy_for(meta)
errors = check(meta, texts)
for e in errors:
    print("layer check:", e, file=sys.stderr)

if "--self-test" in sys.argv[1:-1] and not self_test(meta, texts):
    sys.exit(2)

if errors:
    sys.exit(1)
print("layer check: ok", file=sys.stderr)
PY
