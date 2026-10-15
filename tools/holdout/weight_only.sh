#!/bin/bash
# Weight-only KL of an artifact (docs/QUALITY.md): its dequantized weights in a BF16 GGUF, scored by llama.cpp.
#
# Usage: tools/holdout/weight_only.sh NAME ARTIFACT TEMPLATE-BF16.gguf REFERENCE.kld RESULTS.txt [KEEP_REGEX]
#
# Writes the artifact's weights into a copy of TEMPLATE (dequantize_gguf.py; tensors matching KEEP_REGEX stay BF16),
# scores it against REFERENCE with `llama-perplexity --kl-divergence` through LLAMA_PERPLEXITY (default: the CUDA
# build run natively by the llama-gpu wrapper, weights streamed from host memory, F16 KV cache), appends llama.cpp's
# summary to RESULTS and deletes the GGUF. Run from the repository root. WORK (default /tmp) must hold one BF16 GGUF
# (about 55 GB) on a local Linux filesystem: llama.cpp memory-maps the model, and TEMPLATE belongs there too.
set -euo pipefail
name=$1 artifact=$2 template=$3 reference=$4 results=$5 keep=${6:-}
work=${WORK:-/tmp}
python=${PYTHON:-python3}
perplexity=${LLAMA_PERPLEXITY:?set LLAMA_PERPLEXITY to the llama-perplexity command}
gguf=$work/$name.gguf
trap 'rm -f "$gguf"' EXIT
"$python" tools/holdout/dequantize_gguf.py "$artifact" "$template" "$gguf" ${keep:+--keep "$keep"} > "$work/$name.dequantize.log"
(cd "$work" && $perplexity -m "$gguf" --kl-divergence-base "$reference" --kl-divergence \
   -c 2560 -b 2560 -ub 2560 -ngl 0 -t 16 > "$work/$name.perplexity.log" 2>&1)
{
  echo "# weight-only $name $(date -u +%FT%TZ) artifact $artifact keep '${keep}' reference $reference"
  grep -E 'Mean PPL\(Q\)|Mean PPL\(base\)|Mean ln\(PPL|Mean    KLD|Maximum KLD|99\.0%   KLD|Median  KLD|Same top p|RMS Δp' "$work/$name.perplexity.log"
} | tee -a "$results"
