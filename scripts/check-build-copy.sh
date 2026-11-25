#!/usr/bin/env bash
# The copy of the euhedral_build module in bazel/euhedral_build must equal the archive of the version MODULE.bazel declares, byte
# for byte, as the private registry holds it. Run in a container job of the bazel label, where the registry is at /registry
# (read-only); REGISTRY names another place. Fails, never skips, when the version, the archive or the copy is missing.
set -euo pipefail
cd "$(dirname "$0")/.."
registry=${REGISTRY:-/registry}
copy=bazel/euhedral_build

version=$(sed -n 's/^bazel_dep(name = "euhedral_build", version = "\([^"]*\)".*/\1/p' MODULE.bazel)
[ -n "$version" ] || { echo "MODULE.bazel has no bazel_dep on euhedral_build"; exit 1; }
archive=$registry/modules/euhedral_build/$version/source.tar.gz
[ -f "$archive" ] || { echo "the registry holds no archive of euhedral_build $version ($archive)"; exit 1; }
[ -d "$copy" ] || { echo "$copy is missing"; exit 1; }

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
tar -xzf "$archive" --no-same-owner --no-same-permissions -C "$work"
if ! diff -r --no-dereference "$work/euhedral_build" "$copy"; then
  echo "$copy differs from the archive of euhedral_build $version: replace it with the archive's contents, never edit it"
  exit 1
fi
# diff does not see the executable bit, which git keeps.
exec_files() { (cd "$1" && find . -type f -perm /111 | LC_ALL=C sort); }
if [ "$(exec_files "$work/euhedral_build")" != "$(exec_files "$copy")" ]; then
  echo "$copy differs from the archive of euhedral_build $version in its executable files"
  exit 1
fi
echo "$copy equals the archive of euhedral_build $version"
