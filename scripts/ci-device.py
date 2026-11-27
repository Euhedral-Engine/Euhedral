#!/usr/bin/env python3
"""The device tests of the Rust workspace, built once and run in groups.

  ci-device.py pack CARGO_JSON... --out ARCHIVE --native DIR [--root DIR]
      Reads the `cargo ... --no-run --message-format=json` output of the device build, and writes the test executables of
      the device groups, the server binary that one server test starts, the native library directory (DIR: lib and share) as `device-native`, and a manifest, into one tar archive. Paths in
      the archive are relative to the workspace root; the root's own path goes into the manifest.
  ci-device.py run GROUP [--root DIR] [--archive FILE]
      Runs one group from the unpacked archive in DIR (the workspace root, which must be where the build ran: the test
      binaries carry the build's paths). Every binary runs with `--ignored --nocapture`, from its package directory as
      cargo does, and the run stops at the first failing binary. The engine binary runs the tests of the group's modules
      by exact name. With --archive FILE, first unpacks it into the root and sets the test environment: the
      library is device-native/lib/libeuhedral_cuda.so in the root, EUHEDRAL_NVRTC_DIR (required) is the directory holding libnvrtc, and
      EUHEDRAL_TEST_QWEN_RCO_ARTIFACT, EUHEDRAL_TEST_RUST_GOLDENS and EUHEDRAL_TEST_QUANTIZATION_CUDA_LIBRARIES are
      passed on when set and not empty.
  ci-device.py check [--root DIR]
      Fails unless every ignored test of every device binary is in exactly one group, and prints the count of each.

A group is a set of (package, test target, modules); no modules means the whole binary.
"""

import argparse
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

ENGINE = ("engine", "device")

# The tests of the engine binary that use the card without an engine of the model (the rig tests, with a reduced model),
# and the modules that start engines, by the settings they start with.
RIG_MODULES = [
    "batched", "capture", "capture_exact", "flash", "formats", "formats_oracle", "prefix_copy", "retire", "stages",
    "state",
]
SMALL_ENGINE_MODULES = ["constraint", "hidden", "reserve", "sampled_mtp"]
BIG_ENGINE_MODULES = [
    "cancel", "capture_fault", "composite", "generation", "loader", "mtp", "prefix_mtp", "prefix_restore",
]

GROUPS = {
    "kernels": [
        ("cuda", "gpu", None),
        ("cuda", "kernels_gpu", None),
        ("cuda", "math_gpu", None),
        ("quantization", "kernels_gpu", None),
    ],
    "servers": [
        ("api-server", "engine_constraint_device", None),
        ("api-server", "engine_device", None),
        ("bench", "prompt_sizes", None),
        ("server", "engine_device", None),
        ("tokenizer", "real_vocabulary", None),
    ],
    "engine-rig": [
        ("engine", "staging_plan", None),
        (*ENGINE, RIG_MODULES),
    ],
    "engine-small": [(*ENGINE, SMALL_ENGINE_MODULES)],
    "engine-big": [(*ENGINE, BIG_ENGINE_MODULES)],
}
SERVER_BINARY = ("server", "euhedral-server")


def wanted():
    return {(package, target) for group in GROUPS.values() for package, target, *_ in group}


def pack(args):
    root = Path(args.root).resolve()
    found = {}
    binaries = {}
    for path in args.json:
        for line in Path(path).read_text().splitlines():
            if not line.startswith("{"):
                continue
            message = json.loads(line)
            if message.get("reason") != "compiler-artifact" or not message.get("executable"):
                continue
            # A package is named by its directory under rust/crates.
            package_name = Path(message["manifest_path"]).parent.name
            target = message["target"]["name"]
            entry = {
                "executable": message["executable"],
                "directory": str(Path(message["manifest_path"]).parent),
            }
            if "test" in message["target"]["kind"] and message["profile"]["test"]:
                found[(package_name, target)] = entry
            elif "bin" in message["target"]["kind"]:
                binaries[(package_name, target)] = entry
    manifest = {"root": str(root), "tests": [], "binaries": []}
    chosen = []
    for key in sorted(wanted()):
        if key not in found:
            sys.exit(f"ci-device.py: no test binary {key[0]}/{key[1]} in the build output")
        chosen.append((key, found[key], "tests"))
    if SERVER_BINARY not in binaries:
        sys.exit(f"ci-device.py: no binary {SERVER_BINARY[1]} in the build output")
    chosen.append((SERVER_BINARY, binaries[SERVER_BINARY], "binaries"))
    with tarfile.open(args.out, "w") as archive:
        for (package, target), entry, kind in chosen:
            executable = Path(entry["executable"])
            directory = Path(entry["directory"])
            try:
                relative = executable.relative_to(root)
                package_directory = directory.relative_to(root)
            except ValueError:
                sys.exit(f"ci-device.py: {executable} or {directory} is outside the workspace {root}")
            archive.add(executable, arcname=str(relative))
            manifest[kind].append(
                {
                    "package": package,
                    "target": target,
                    "executable": str(relative),
                    "directory": str(package_directory),
                }
            )
        archive.add(args.native, arcname="device-native")
        data = json.dumps(manifest, indent=1).encode()
        info = tarfile.TarInfo("ci-device-manifest.json")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    print(f"packed {len(manifest['tests'])} test binaries and the server binary into {args.out}")


def load(root):
    manifest = json.loads((root / "ci-device-manifest.json").read_text())
    if Path(manifest["root"]).resolve() != root.resolve() and not os.environ.get("CI_DEVICE_ANY_ROOT"):
        sys.exit(
            f"ci-device.py: the binaries were built in {manifest['root']} and carry its paths; this workspace is {root}"
        )
    return manifest


def binary(manifest, package, target):
    for entry in manifest["tests"]:
        if (entry["package"], entry["target"]) == (package, target):
            return entry
    sys.exit(f"ci-device.py: {package}/{target} is not in the archive")


def ignored_tests(root, entry):
    out = subprocess.run(
        [str(root / entry["executable"]), "--ignored", "--list"],
        cwd=root / entry["directory"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [line[: -len(": test")] for line in out.splitlines() if line.endswith(": test")]


def selected(root, entry, modules):
    names = ignored_tests(root, entry)
    if modules is None:
        return names
    return [name for name in names if name.split("::")[0] in modules]


def unpack_and_set_environment(root, archive):
    nvrtc = os.environ.get("EUHEDRAL_NVRTC_DIR", "")
    if not nvrtc:
        sys.exit("ci-device.py: the Actions variable EUHEDRAL_NVRTC_DIR is not set")
    with tarfile.open(archive) as tests:
        tests.extractall(root, filter="data")
    library = root / "device-native/lib/libeuhedral_cuda.so"
    if not library.is_file():
        sys.exit(f"ci-device.py: {library} is missing from the archive")
    os.environ["EUHEDRAL_TEST_CUDA_LIBRARY"] = str(library)
    os.environ["EUHEDRAL_NVRTC"] = f"{nvrtc}/libnvrtc.so.13"
    os.environ["LD_LIBRARY_PATH"] = nvrtc
    for variable in ("QWEN_RCO_ARTIFACT", "RUST_GOLDENS", "QUANTIZATION_CUDA_LIBRARIES"):
        value = os.environ.get(f"EUHEDRAL_TEST_{variable}", "")
        if not value:
            os.environ.pop(f"EUHEDRAL_TEST_{variable}", None)


def run(args):
    root = Path(args.root).resolve()
    if args.archive:
        unpack_and_set_environment(root, Path(args.archive).resolve())
    manifest = load(root)
    if args.group not in GROUPS:
        sys.exit(f"ci-device.py: no group {args.group}; groups: {', '.join(GROUPS)}")
    for entry in manifest["tests"]:
        # The tests that write files put them in the build's `tmp` directory next to the profile directory.
        (root / entry["executable"]).parent.parent.parent.joinpath("tmp").mkdir(exist_ok=True)
    for package, target, modules in GROUPS[args.group]:
        entry = binary(manifest, package, target)
        command = [str(root / entry["executable"]), "--ignored", "--nocapture"]
        if modules is not None:
            names = selected(root, entry, modules)
            if not names:
                sys.exit(f"ci-device.py: no test of {modules} in {package}/{target}")
            command += ["--exact", *names]
            print(f"== {package}/{target}: {len(names)} tests of {', '.join(modules)}", flush=True)
        else:
            print(f"== {package}/{target}", flush=True)
        status = subprocess.run(command, cwd=root / entry["directory"]).returncode
        if status != 0:
            sys.exit(f"ci-device.py: {package}/{target} failed (exit {status})")


def check(args):
    root = Path(args.root).resolve()
    manifest = load(root)
    claimed = {}
    for name, group in GROUPS.items():
        total = 0
        for package, target, modules in group:
            for test in selected(root, binary(manifest, package, target), modules):
                key = (package, target, test)
                if key in claimed:
                    sys.exit(f"ci-device.py: {key} is in the groups {claimed[key]} and {name}")
                claimed[key] = name
                total += 1
        print(f"{name}: {total}")
    for entry in manifest["tests"]:
        every = ignored_tests(root, entry)
        missing = [t for t in every if (entry["package"], entry["target"], t) not in claimed]
        if missing:
            sys.exit(f"ci-device.py: tests in no group: {missing[:5]}")
    print(f"all {len(claimed)} ignored tests are in exactly one group")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    pack_parser = commands.add_parser("pack")
    pack_parser.add_argument("json", nargs="+")
    pack_parser.add_argument("--out", required=True)
    pack_parser.add_argument("--root", default=".")
    pack_parser.add_argument("--native", required=True)
    pack_parser.set_defaults(handler=pack)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("group")
    run_parser.add_argument("--root", default=".")
    run_parser.add_argument("--archive")
    run_parser.set_defaults(handler=run)
    check_parser = commands.add_parser("check")
    check_parser.add_argument("--root", default=".")
    check_parser.set_defaults(handler=check)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
