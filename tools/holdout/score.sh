#!/bin/bash
# Score one artifact on one holdout text against its BF16 reference (docs/QUALITY.md).
#
# Usage: tools/holdout/score.sh NAME ARTIFACT REFERENCE.kld RESULTS.txt [HOST_MIB]
#
# Teacher-forces every chunk of REFERENCE.kld on ARTIFACT (TeacherForcedQualityCudaIntegrationTest with
# euhedral.quality.tokens), appends the `compare_reference.py --detail` summary to RESULTS.txt and deletes the chunk
# reports (about 0.6 GB each). Run from the repository root with the GPU free; HOST_MIB defaults to 2304, enough for
# the NVFP4 artifacts on a 16 GB card.
set -euo pipefail
name=$1 artifact=$2 reference=$3 results=$4 host=${5:-2304}
work=$(mktemp -d "${TMPDIR:-/tmp}/holdout-$name.XXXX")
python=${PYTHON:-python3}
trap 'rm -rf "$work"' EXIT
EUHEDRAL_HOST_MEMORY_MIB=${EUHEDRAL_HOST_MEMORY_MIB:-12000} ./gradlew -q :core:cudaIntegrationTest \
  --tests '*TeacherForcedQualityCudaIntegrationTest' --rerun \
  -Peuhedral.quality.artifact="$artifact" -Peuhedral.quality.tokens="$reference" \
  -Peuhedral.quality.report="$work/$name" -Peuhedral.quality.host-mib="$host"
{
  echo "# $name $(date -u +%FT%TZ) artifact $artifact reference $reference"
  "$python" tools/compare_reference.py --detail "$reference" "$work/$name"
} | tee -a "$results"
