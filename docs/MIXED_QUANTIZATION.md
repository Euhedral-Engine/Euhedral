# Mixed NVFP4 and trellis artifacts

A mixed artifact stores each fused projection object of the text layers in its own format: NVFP4 (the GPTQ
codes) or trellis at 2.5, 3.0, 3.5 or 4.0 bits per weight. The engine routes every object by its own format, so a
mixed artifact runs the same kernels, row-exact verification, MTP, offload and prefix cache as a pure one; it
stores the MTP draft depth it was measured with (`ArtifactProfile` names it `mixed` and refuses MTP without one).

## Sources and assembly

Every option comes from a pure artifact converted once from its calibration (trellis: `calibrate --format trellis
--bits B`, one rate for every linear; NVFP4: the GPTQ artifact). `quantization assemble --allocation FILE --source
OPTION=ARTIFACT... --base OPTION --out ARTIFACT` takes each projection the allocation names from its option's source,
with its companions, and everything else from the base: the token embedding, the heads and the MTP drafter. The
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

Calibration of the trellis 2.5 and 4.0 sources took 66 and 33 minutes on the RTX 5070 Ti.

## Cost model

`quantization cost` times one launch of each option's real payload for every distinct projection shape on the
device alone (the stream waits on a host flag while the launches queue, as in captured decode graphs), with the
weights rotated past L2. Microseconds per launch at 1 row / per 512-row prefill chunk, at +150 MHz core,
+1500 MHz memory and 350 W:

| Shape (in × out) | Objects | nvfp4 | trellis-2.5 | trellis-3.0 | trellis-3.5 | trellis-4.0 |
|---|---|---|---|---|---|---|
| 5120 × 4096 | GDN query/key | 16.2 / 71 | 16.4 / 324 | 16.3 / 323 | 17.9 / 324 | 18.7 / 326 |
| 5120 × 12288 | GDN value/z | 43.0 / 201 | 35.6 / 794 | 37.1 / 793 | 42.4 / 795 | 46.0 / 801 |
| 6144 × 5120 | GDN and attention output | 24.6 / 120 | 19.7 / 460 | 20.0 / 458 | 21.8 / 458 | 23.7 / 463 |
| 5120 × 7168 | attention query/key and gate/value | 25.9 / 133 | 22.1 / 527 | 22.5 / 525 | 24.5 / 525 | 26.8 / 530 |
| 5120 × 34816 | MLP gate/up | 117.1 / 521 | 85.2 / 2214 | 88.7 / 2214 | 100.0 / 2213 | 112.1 / 2231 |
| 17408 × 5120 | MLP down | 63.6 / 339 | 45.0 / 1234 | 46.4 / 1224 | 51.4 / 1222 | 56.7 / 1236 |

An allocation's prefill time per token is the sum of its projections' 512-row times over 512 plus 103.5 µs; its
5-row verification step the sum of their 5-row times plus 5.45 ms; its one-row step the sum of their one-row times
plus 2.74 ms. The fixed terms are the non-linear work measured end to end (prefill of 4096 tokens, decode and MTP
verification) on the trellis 2.5, 3.0 and 3.5 sources; with them the model reproduces each source's measured
prefill within 2%, its verification step within 1% and its one-row step within 3%.

## Search reference

Candidates are scored against BF16 logits the engine computes itself: `quantization reference` runs the BF16
artifact (weights split over the device, pinned host memory and disk by the residency planner) over a text of 8
chunks of 2560 tokens (wikitext-2 test text past the positions T2 scores, the model's own generations past the
positions T4 scores, and Python and Go sources T3 does not hold) and writes the logits in llama.cpp's format. It
took 1045 s. Scoring is deterministic: the trellis 3.0 source scores KL 0.0895 on it twice, bit for bit.

## Family swaps

Each family of projections moved together, across all 64 layers, from the trellis 3.0 base to another option, at a
32,768-token context (Search KL against the reference above; bytes are the artifact's; prefill and step from the
cost model):

| Projections moved to | Option | Search KL | Bytes | Prefill µs/token | 5-row step ms |
|---|---|---|---|---|---|
| every projection | nvfp4 | 0.0536 | 14.49 GiB (does not fit) | 260 | 22.62 |
| every projection | trellis-2.5 | 0.1564 | 8.84 GiB | 730 | 18.37 |
| every projection | trellis-3.0 | 0.0895 | 10.26 GiB | 728 | 18.82 |
| every projection | trellis-3.5 | 0.0571 | 11.67 GiB | 728 | 20.63 |
| every projection | trellis-4.0 | 0.0446 | 13.09 GiB (does not fit) | 734 | 22.09 |
| `attention/query_key` | nvfp4 | 0.0857 | 10.36 GiB | 716 | 18.90 |
| `attention/query_key` | trellis-2.5 | 0.0916 | 10.22 GiB | 728 | 18.81 |
| `attention/query_key` | trellis-3.5 | 0.0901 | 10.29 GiB | 728 | 18.86 |
| `attention/query_key` | trellis-4.0 | 0.0853 | 10.32 GiB | 728 | 18.89 |
| `attention/gate_value` | nvfp4 | 0.0896 | 10.36 GiB | 716 | 18.90 |
| `attention/gate_value` | trellis-2.5 | 0.0985 | 10.22 GiB | 728 | 18.81 |
| `attention/gate_value` | trellis-3.5 | 0.0930 | 10.29 GiB | 728 | 18.86 |
| `attention/gate_value` | trellis-4.0 | 0.0889 | 10.32 GiB | 728 | 18.89 |
| `attention/output` | nvfp4 | 0.0906 | 10.34 GiB | 717 | 18.90 |
| `attention/output` | trellis-2.5 | 0.0921 | 10.23 GiB | 728 | 18.82 |
| `attention/output` | trellis-3.5 | 0.0953 | 10.28 GiB | 728 | 18.85 |
| `attention/output` | trellis-4.0 | 0.0893 | 10.31 GiB | 728 | 18.88 |
| `gdn/query_key` | nvfp4 | 0.0910 | 10.43 GiB | 704 | 18.85 |
| `gdn/query_key` | trellis-2.5 | 0.0960 | 10.20 GiB | 728 | 18.81 |
| `gdn/query_key` | trellis-3.5 | 0.0874 | 10.31 GiB | 728 | 18.90 |
| `gdn/query_key` | trellis-4.0 | 0.0910 | 10.37 GiB | 728 | 18.93 |
| `gdn/value_z` | nvfp4 | 0.0815 | 10.78 GiB | 672 | 19.17 |
| `gdn/value_z` | trellis-2.5 | 0.0919 | 10.08 GiB | 728 | 18.77 |
| `gdn/value_z` | trellis-3.5 | 0.0911 | 10.43 GiB | 728 | 19.13 |
| `gdn/value_z` | trellis-4.0 | 0.0873 | 10.61 GiB | 729 | 19.29 |
| `gdn/output` | nvfp4 | 0.0898 | 10.52 GiB | 696 | 19.05 |
| `gdn/output` | trellis-2.5 | 0.1004 | 10.17 GiB | 728 | 18.81 |
| `gdn/output` | trellis-3.5 | 0.0862 | 10.34 GiB | 728 | 18.91 |
| `gdn/output` | trellis-4.0 | 0.0856 | 10.43 GiB | 728 | 19.00 |
| `mlp/gate_up` | nvfp4 | 0.0850 | 12.24 GiB | 516 | 20.61 |
| `mlp/gate_up` | trellis-2.5 | 0.1169 | 9.59 GiB | 728 | 18.55 |
| `mlp/gate_up` | trellis-3.5 | 0.0734 | 10.92 GiB | 728 | 19.74 |
| `mlp/gate_up` | trellis-4.0 | 0.0696 | 11.58 GiB | 730 | 20.47 |
| `mlp/down` | nvfp4 | 0.0848 | 11.25 GiB | 617 | 19.99 |
| `mlp/down` | trellis-2.5 | 0.1147 | 9.92 GiB | 729 | 18.73 |
| `mlp/down` | trellis-3.5 | 0.0877 | 10.59 GiB | 728 | 19.14 |
| `mlp/down` | trellis-4.0 | 0.0858 | 10.92 GiB | 729 | 19.49 |

Every allocation above but the pure NVFP4 and trellis 4.0 ones fits on the 16 GB card at that context with every
weight resident.

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
| 32,768 | 30 | 277 | 80 | 0.125 | 0.0438, 12.75 GiB |
| 131,072 | 30 | 517 (277 imported) | 57 | 0.060 | 0.0659, 11.01 GiB |

At 131,072 tokens 48 of the 277 imported allocations fit.

## 128K finalists

Two points of the 131,072-token front, `NVFP4-TA` and `NVFP4-TB`, assembled to files and measured like the README's
artifacts (MTP depth 4, chosen from 3, 4 and 5 by 4K decode; +150 MHz core, +1500 MHz memory, 350 W; 2026-10-16):

| Finalist | File | Formats per family (objects) |
|---|---|---|
| `NVFP4-TA`, most accurate | 11.02 GiB | MLP down mostly trellis 3.5 (51 of 64), MLP gate/up mostly 3.0 (44), GDN output 3.5 and 4.0 (24, 16), GDN query/key 3.0 and 4.0 (23, 19) with 5 NVFP4, GDN value/z 2.5 and 3.0 (14, 34), attention mostly 3.0 and 4.0 |
| `NVFP4-TB`, fastest prefill | 10.96 GiB | MLP down mostly NVFP4 (36 of 64), GDN output 20 NVFP4, MLP gate/up mostly 3.0 (42) with 6 NVFP4, GDN value/z almost all 2.5 (46 of 48), attention 2.5 to 4.0 with 4 NVFP4 |

| Finalist | | 4K | 32K | 64K | 128K |
|---|---|---|---|---|---|
| `NVFP4-TA`, most accurate | prefill tok/s | 1,376 | 1,204 | 1,062 | 853 |
| | decode tok/s (MTP) | 137.8 | 142.8 | 111.3 | 79.0 |
| | time to first token | 2.89 s | 27.1 s | 59.8 s | 150 s |
| | device memory in use | 11.63 GiB | 12.16 GiB | 12.72 GiB | 13.90 GiB |
| `NVFP4-TB`, fastest prefill | prefill tok/s | 1,589 | 1,364 | 1,185 | 931 |
| | decode tok/s (MTP) | 132.9 | 125.5 | 106.6 | 82.3 |
| | time to first token | 2.50 s | 24.0 s | 53.6 s | 138 s |
| | device memory in use | 11.58 GiB | 12.10 GiB | 12.66 GiB | 13.84 GiB |

Neither holds any weight in host memory at any of these contexts. Decode without speculation at 4K: 62.3 and
60.8 tok/s.

KL against the four references (T1 the README's quality document, T2 wikitext-2, T3 code, T4 the model's own
generations; [docs/QUALITY.md](QUALITY.md)):

| Finalist | T1 | T2 | T3 | T4 | T1 top-1 |
|---|---|---|---|---|---|
| `NVFP4-TA`, most accurate | 0.0382 | 0.0450 | 0.0893 | 0.1935 | 90.3% |
| `NVFP4-TB`, fastest prefill | 0.0546 | 0.0637 | 0.1188 | 0.2246 | 86.7% |

Two more front points were measured and not kept: one at search KL 0.0748 (T1 0.0440, T2 0.0525, T3 0.0948, T4 0.1935;
prefill 1,214 tok/s at 32K, decode 134.3) and the knee at 0.0783 (T1 0.0546, T2 0.0567, T3 0.1126, T4 0.2359; prefill
1,329 tok/s at 32K, decode 120.1); `NVFP4-TB` prefills faster than the knee at the same T1 KL.
