#!/bin/bash
# Weight-only KL of an artifact (docs/QUALITY.md): its dequantized weights in a BF16 GGUF, scored by llama.cpp.
#
# Usage: tools/holdout/weight_only.sh NAME ARTIFACT TEMPLATE-BF16.gguf REFERENCE.kld RESULTS.txt [KEEP_REGEX]
#
# Writes the artifact's weights into a copy of TEMPLATE (dequantize_gguf.py; tensors matching KEEP_REGEX stay BF16),
# scores it against REFERENCE with `llama-perplexity --kl-divergence` through LLAMA_PERPLEXITY (default: the CUDA
# build wrapper, weights streamed from host memory, F16 KV cache), appends llama.cpp's summary to RESULTS and deletes
# the GGUF. Run from the repository root; WORK (default /tmp) must hold one BF16 GGUF (about 55 GB).
set -euo pipefail
name=$1 artifact=$2 template=$3 reference=$4 results=$5 keep=${6:-}
work=${WORK:-/tmp}
python=${PYTHON:-python3}
perplexity=${LLAMA_PERPLEXITY:-/mnt/shared/qwen38-quant/quality/bin/llama-gpu llama-perplexity}
gguf=$work/$name.gguf
trap 'rm -f "$gguf"' EXIT
"$python" tools/holdout/dequantize_gguf.py "$artifact" "$template" "$gguf" ${keep:+--keep "$keep"} > "$work/$name.dequantize.log"
# The CUDA wrapper mounts /mnt/shared/qwen38-quant at /q and the current directory at /w.
in_container() { case $1 in /mnt/shared/qwen38-quant/*) echo "/q/${1#/mnt/shared/qwen38-quant/}" ;; *) echo "$1" ;; esac; }
(cd "$work" && $perplexity -m "$(in_container "$gguf")" --kl-divergence-base "$(in_container "$reference")" --kl-divergence \
   -c 2560 -b 2560 -ub 2560 -ngl 0 -t 16 > "$work/$name.perplexity.log" 2>&1)
{
  echo "# weight-only $name $(date -u +%FT%TZ) artifact $artifact keep '${keep}' reference $reference"
  grep -E 'Mean PPL\(Q\)|Mean PPL\(base\)|Mean ln\(PPL|Mean    KLD|Maximum KLD|99\.0%   KLD|Median  KLD|Same top p|RMS Δp' "$work/$name.perplexity.log"
} | tee -a "$results"
