# Mixed NVFP4 and trellis artifacts

A mixed artifact stores each fused projection object of the text layers in its own format: NVFP4 (the GPTQ
codes) or trellis at 2.5, 3.0, 3.5 or 4.0 bits per weight. The engine routes every object by its own format, so a
mixed artifact runs the same kernels, row-exact verification, MTP, offload and prefix cache as a pure one; it
stores the MTP draft depth it was measured with (`ArtifactProfile` names it `mixed` and refuses MTP without one).

## Sources and assembly

Every option comes from a pure artifact converted once from its calibration (trellis: `calibrate --format trellis
--bits B`, one rate for every linear; NVFP4: the GPTQ artifact). `quantization assemble --allocation FILE --source
OPTION=ARTIFACT... --base OPTION --out ARTIFACT` takes each projection the allocation names from its option's source,
with its companions, and everything else from the base: the token embedding, the heads, the MTP drafter and the
tokenizer. The
objects taken straight from the checkpoint must be bit for bit equal in every source; `assemble` refuses a source
that differs. In process the same assembly is a composite view whose objects are read from their source files, which
is how a search loads candidates without writing them. An allocation that names nothing assembles to a file byte for
byte equal to its base.

| Source | File |
|---|---|
| NVFP4 (GPTQ) | 14.51 GiB |
| trellis 2.5 | 8.84 GiB |
| trellis 3.0 | 10.26 GiB |
| trellis 3.5 | 11.67 GiB |
| trellis 4.0 | 13.09 GiB |

## Cost model

`quantization cost` times one launch of each option's real payload for every distinct projection shape on the
device alone (the stream waits on a host flag while the launches queue, as in captured decode graphs), with the
weights rotated past L2. Microseconds per launch at 1 row / per 512-row prefill chunk (trellis prefill on block-scaled FP8,
[TRELLIS.md](TRELLIS.md)), at +150 MHz core,
+1500 MHz memory and 350 W:

| Shape (in × out) | Objects | nvfp4 | trellis-2.5 | trellis-3.0 | trellis-3.5 | trellis-4.0 |
|---|---|---|---|---|---|---|
| 5120 × 4096 | GDN query/key | 15.9 / 71 | 16.4 / 136 | 16.3 / 136 | 17.9 / 137 | 18.6 / 134 |
| 5120 × 12288 | GDN value/z | 43.1 / 201 | 35.6 / 362 | 36.8 / 361 | 42.4 / 366 | 46.1 / 360 |
| 6144 × 5120 | GDN and attention output | 24.3 / 119 | 19.7 / 200 | 20.0 / 200 | 21.8 / 203 | 23.5 / 199 |
| 5120 × 7168 | attention query/key and gate/value | 25.9 / 133 | 22.1 / 239 | 22.6 / 238 | 24.6 / 241 | 26.6 / 237 |
| 5120 × 34816 | MLP gate/up | 116.5 / 519 | 85.1 / 1053 | 88.4 / 1046 | 99.2 / 1076 | 111.2 / 1085 |
| 17408 × 5120 | MLP down | 63.2 / 338 | 44.9 / 560 | 46.3 / 556 | 51.0 / 560 | 56.8 / 553 |

An allocation's prefill time per token is the sum of its projections' 512-row times over 512 plus 103.5 µs; its
5-row verification step the sum of their 5-row times plus 5.45 ms; its one-row step the sum of their one-row times
plus 2.74 ms. The fixed terms are the non-linear work measured end to end (prefill of 4096 tokens, decode and MTP
verification) on the trellis 2.5, 3.0 and 3.5 sources. The model's prefill matches the engine's within 4%: 390 against
376 µs per token for the trellis 3.0 source and 393 against 387 for `NVFP4-Trellis-RCO`, at 4K tokens.

## Search reference

Candidates are scored against BF16 logits the engine computes itself: `quantization reference` runs the BF16
artifact (weights split over the device, pinned host memory and disk by the residency planner) over a text of 8
chunks of 2560 tokens and writes the logits in llama.cpp's format. The text is built like the held-out set
([RELEASE_TEST.md](RELEASE_TEST.md)) and disjoint from it: six conversations BF16 Swift-1.5 generated from their own
prompts (two conversation, two coding, one tool use, one multilingual) and two human-written documents the held-out
set holds as unscored spares (one prose, one code), cut and ordered by `tools/t5/chunks.py`.

## Family swaps

Each family of projections moved together, across all 64 layers, from the trellis 3.0 base to another option, at a
131,072-token context (search KL against the reference above; bytes are the artifact's; prefill and step from the
cost model; "does not fit": not every weight stays on the 16 GB card at that context):

| Projections moved to | Option | Search KL | Bytes | Prefill µs/token | 5-row step ms |
|---|---|---|---|---|---|
| every projection | nvfp4 | 0.0641 | 14.49 GiB (does not fit) | 259 | 22.60 |
| every projection | trellis-2.5 | 0.1451 | 8.84 GiB | 392 | 18.33 |
| every projection | trellis-3.0 | 0.1126 | 10.25 GiB | 390 | 18.78 |
| every projection | trellis-3.5 | 0.0865 | 11.67 GiB (does not fit) | 396 | 20.52 |
| every projection | trellis-4.0 | 0.0598 | 13.09 GiB (does not fit) | 394 | 21.88 |
| `attention/gate_value` | nvfp4 | 0.1038 | 10.36 GiB | 387 | 18.87 |
| `attention/gate_value` | trellis-2.5 | 0.1042 | 10.22 GiB | 390 | 18.78 |
| `attention/gate_value` | trellis-3.5 | 0.1061 | 10.29 GiB | 390 | 18.82 |
| `attention/gate_value` | trellis-4.0 | 0.1047 | 10.32 GiB | 390 | 18.85 |
| `attention/output` | nvfp4 | 0.0973 | 10.34 GiB | 388 | 18.86 |
| `attention/output` | trellis-2.5 | 0.1066 | 10.23 GiB | 390 | 18.78 |
| `attention/output` | trellis-3.5 | 0.0994 | 10.28 GiB | 390 | 18.81 |
| `attention/output` | trellis-4.0 | 0.0993 | 10.31 GiB | 390 | 18.84 |
| `attention/query_key` | nvfp4 | 0.1084 | 10.36 GiB | 387 | 18.87 |
| `attention/query_key` | trellis-2.5 | 0.1158 | 10.22 GiB | 390 | 18.78 |
| `attention/query_key` | trellis-3.5 | 0.1101 | 10.29 GiB | 390 | 18.82 |
| `attention/query_key` | trellis-4.0 | 0.1096 | 10.32 GiB | 390 | 18.85 |
| `gdn/output` | nvfp4 | 0.1014 | 10.52 GiB | 383 | 19.01 |
| `gdn/output` | trellis-2.5 | 0.1115 | 10.17 GiB | 390 | 18.77 |
| `gdn/output` | trellis-3.5 | 0.0964 | 10.34 GiB | 390 | 18.87 |
| `gdn/output` | trellis-4.0 | 0.1067 | 10.43 GiB | 390 | 18.95 |
| `gdn/query_key` | nvfp4 | 0.1057 | 10.43 GiB | 384 | 18.81 |
| `gdn/query_key` | trellis-2.5 | 0.1107 | 10.20 GiB | 390 | 18.78 |
| `gdn/query_key` | trellis-3.5 | 0.1083 | 10.31 GiB | 390 | 18.86 |
| `gdn/query_key` | trellis-4.0 | 0.1024 | 10.37 GiB | 390 | 18.90 |
| `gdn/value_z` | nvfp4 | 0.0964 | 10.78 GiB | 375 | 19.13 |
| `gdn/value_z` | trellis-2.5 | 0.1055 | 10.08 GiB | 390 | 18.73 |
| `gdn/value_z` | trellis-3.5 | 0.1038 | 10.43 GiB | 391 | 19.07 |
| `gdn/value_z` | trellis-4.0 | 0.1004 | 10.61 GiB | 390 | 19.24 |
| `mlp/down` | nvfp4 | 0.1019 | 11.25 GiB (does not fit) | 363 | 19.97 |
| `mlp/down` | trellis-2.5 | 0.1211 | 9.92 GiB | 391 | 18.69 |
| `mlp/down` | trellis-3.5 | 0.1039 | 10.59 GiB | 391 | 19.08 |
| `mlp/down` | trellis-4.0 | 0.1050 | 10.92 GiB | 390 | 19.44 |
| `mlp/gate_up` | nvfp4 | 0.0862 | 12.24 GiB (does not fit) | 324 | 20.56 |
| `mlp/gate_up` | trellis-2.5 | 0.1144 | 9.59 GiB | 391 | 18.51 |
| `mlp/gate_up` | trellis-3.5 | 0.0982 | 10.92 GiB | 394 | 19.67 |
| `mlp/gate_up` | trellis-4.0 | 0.0853 | 11.58 GiB (does not fit) | 395 | 20.30 |

At that context the pure NVFP4, trellis 3.5 and trellis 4.0 options do not fit, nor NVFP4 for either MLP family or
trellis 4.0 for MLP gate/up.

## Search

`quantization allocate` searches for allocations on the Pareto front of search KL, prefill time and 5-row
verification time, among those whose every weight stays on the device at a target context (`--context`; the
residency planner decides). Each round:

1. fits a bootstrap ensemble of a fidelity model to the KL measured so far: per-object prior errors (the trellis
   proxy error, NVFP4's weighted rounding error) weighted per family and layer band, pairwise terms within a layer and
   between neighbouring bands, and a bend `kl = s·x^γ` for compounding, with non-negative weights;
2. runs Riemannian Constrained Optimization (RCO; Helcig and Alistarh, [arXiv:2605.00649](https://arxiv.org/abs/2605.00649),
   [IST-DASLab/RCO](https://github.com/IST-DASLab/RCO)) on the ensemble mean plus `--lambdas`-weighted prefill and
   verification times, under the byte budget, once per weight pair and budget fraction;
3. gives part of the batch (`explore`) to feasible proposals the ensemble members disagree on most, and to single
   moves of the front's points;
4. assembles each proposal as a composite view, checks it with the residency planner and scores it in the engine
   against the search reference (row-exact verification, several candidates on the device at once over one cache of
   their shared weights).

Every measurement goes into one resume file; a stopped search resumes from it, and `--import-state` starts a search
from another's measurements, refitted to its own budget and context. Held-out error is the mean absolute log ratio of
the ensemble's predicted KL to the KL measured for the round's proposals, before they are fitted (0.06: predictions
about 6% off).

| Context | Rounds | Measured | Front | Held-out error, last round | Lowest search KL on the front |
|---|---|---|---|---|---|
| 131,072 | 30 | 277 | 75 | 0.133 | 0.0798, 11.16 GiB |

## NVFP4-Trellis-RCO

The lowest-KL point of the 131,072-token front, assembled to a file and measured like the README's artifacts (MTP
depth 4; +150 MHz core, +1500 MHz memory, 350 W; 2026-10-19). Every projection is trellis: 114 of 320 at 4.0 bits, 88
at 3.5, 45 at 3.0 and 73 at 2.5, with the NVFP4 heads and Q3 token embedding of the trellis 3.0 base; 11.17 GiB.

| | 4K | 32K | 64K | 128K |
|---|---|---|---|---|
| prefill tok/s | 2,586 | 2,039 | 1,663 | 1,203 |
| decode tok/s (MTP) | 147.2 | 114.6 | 93.1 | 78.3 |
| time to first token | 1.54 s | 16.1 s | 38.2 s | 107 s |
| device memory in use | 11.78 GiB | 12.31 GiB | 12.87 GiB | 14.05 GiB |

No weight is held in host memory at any of these contexts. Decode without speculation at 4K: 60.9 tok/s. The MTP
depth was chosen from 3 to 7: depth 5 decodes fastest after a 4K prompt (143.5 against 140.4 tok/s at depth 4, mean
of six prompts) but depth 4 is faster at long context (93.1 against 87.2 tok/s at 64K, 78.3 against 71.2 at 128K).

On the held-out set ([RELEASE_TEST.md](RELEASE_TEST.md)): dNLL +0.051 ± 0.006, KL 0.072, top-1 92.9%.

Measured and not kept: the fastest-prefill point of the front within 1.25 times its search KL (search KL 0.0966,
11.14 GiB): prefill 2,881 tok/s at 4K and 1,263 at 128K, held-out dNLL +0.075 ± 0.008,
KL 0.089, top-1 92.0%.
