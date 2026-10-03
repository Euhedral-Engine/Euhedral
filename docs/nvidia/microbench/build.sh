#!/usr/bin/env bash
# Builds ./nvbench with zig cc against the repo's pinned CUDA 13.1 headers and NVRTC.
# Usage: bash build.sh            (from anywhere; output lands next to this script)
#        CUDA_DEV=/path/to/cuda-dev/linux-x64 bash build.sh
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../../.." && pwd)"
cuda="${CUDA_DEV:-$repo/build/cuda-dev/linux-x64}"
zig="${ZIG:-zig}"
"$zig" cc -O2 -std=gnu11 -Wall -Wno-unused-function \
    -I"$cuda/include" -I"$here" \
    "$here"/main.c "$here"/common.c "$here"/attrs.c "$here"/mem.c "$here"/mma.c \
    "$here"/alu.c "$here"/tex.c "$here"/pcie.c "$here"/l2.c "$here"/misc.c \
    -L"$cuda/lib" -L"$cuda/runtime" -Wl,-rpath,"$cuda/runtime" \
    -lcuda -lnvrtc -lm -lpthread -o "$here/nvbench"
echo "built $here/nvbench"
echo "run:  CUDA_INCLUDE_DIR=$cuda/include MB_ROOT=$here $here/nvbench <command>"
