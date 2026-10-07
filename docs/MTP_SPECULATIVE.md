# MTP speculative decode

Greedy MTP speculative decoding (contract in [MTP_CONTRACT.md](MTP_CONTRACT.md)) on the RTX 5070 Ti (16 GB). It covers what runs,
how exactness is proven, and what it measured. The depth, the exact multi-row kernels and the draft costs are in
[MTP_VERIFIER.md](MTP_VERIFIER.md).

## 1. What runs

Each artifact gets one policy, read from its tensors (`ArtifactProfile`): MTP is on when the artifact carries the MTP layer and
the draft head (all four artifacts do), at depth 2 for Q3 and 3 for NVFP4. There is no option for it.

One speculative step is a chain of quanta on the existing runtime, with no central dispatcher and no host loop over layers (shown
for depth 3):

1. **VERIFY** over `[t₀, d₁, d₂, d₃]`. This is the decode topology with ALL_TOKENS logits and a device
   argmax per row; only the four token IDs cross to the host. Acceptance (`SpeculativeAcceptance`)
   resolves on retirement. The base KV commits `a + 1` rows (`AttentionKvState.commitSubmitted`).
   GDN checkpoints its state and records its per-row inputs for a replay.
2. **DRAFT catch-up**: the MTP layer over the committed outputs, seeded by the verified rows'
   post-final-norm hiddens. Its last row drafts d′₁.
3. **Two recursive DRAFT rows**, each seeded by the previous MTP hidden.

The MTP layer is a plan view (`ExecutionPlan.mtpDraft`):
- embedding, then `MTP_STEM` (the two norms and the `fc` projection);
- one full-attention layer with its own KV slot;
- `mtp.norm`, then the 131,072-row draft head.

The prompt prefills in chunks, each followed by its MTP catch-up (in pieces of at most 128 rows). Speculative decoding applies to
greedy, unconstrained generations from a fresh sequence (`Session.enableSpeculativeDecoding`).

**GDN rollback is ReplaySSM.** VERIFY checkpoints the recurrent and convolution state and records each
row's convolution inputs and α/β. On a partial accept, the next quantum restores the checkpoint and
replays the committed rows through the same row-exact kernels. Cost at depth 3: about 157 MiB per sequence. Measured per
verification (1024-token prompt): 221 MiB of device-to-device copies, 0.47 ms of copy-engine time, partly overlapping kernels.

## 2. Exactness

The oracle is ordinary one-row greedy decode, and the requirement is identical token IDs and
identical state, bitwise.

- **Verifier (row-exact).** Every row goes through the one-row arithmetic (MTP_CONTRACT §6):
  - attention, norms and the GDN recurrence run row by row;
  - Q3, Q4 and Q5 linears use multi-row twins of the one-row GEMVs that repeat its FMA sequence per row; NVFP4 linears use the
    tensor-core decode kernels, whose token rows are independent MMA columns ([NVFP4_NATIVE.md](NVFP4_NATIVE.md)).
- **Tests:**
  - `SpeculativeVerifyCudaIntegrationTest`: logits, GDN state and KV after a VERIFY quantum equal
    sequential decode, bitwise, across the 2048-key attention kernel switch; partial commits with replay.
  - `SpeculativeDecodeCudaIntegrationTest`: full MTP generation equals greedy decode in tokens and
    final state, for three prompts (a chat answer, 600 tokens of prose, a Java chat prompt).
  - Native: `test_nvfp4.py` (decode_rows bitwise), `test_nvfp4_native.py` (skinny rows independent
    and deterministic), `test_q3_kernels.py` and `test_q45_kernels.py` (Q3/Q4/Q5 twins bitwise one-row), `test_q3_p2e2.py`
    (P2E2 decode bitwise the contiguous kernel).
- **Every benchmark row records `outputSha256`.** In every gate a speculative arm's hash equals its non-speculative arm's for every
  prompt.

## 3. Memory

MTP is never host-backed: `HostWeightSelection` only takes `text/layers/N/...` projections and the token embedding. Sizes of the
`nvfp4` artifact's objects ([NVFP4_RESIDENCY.md](NVFP4_RESIDENCY.md) for how the engine fits them):

| Item | MiB |
|---|---|
| Base layers + output head + embedding (artifact) | 14,275 |
| MTP layer (resident) | 228 |
| Draft head (131,072 rows) + token ids (resident) | 360.5 |
| Speculative GDN (checkpoint, records, replay scratch) | about 157 |

A Q3 artifact with MTP needs no host-backed weights at 4096 tokens: 12.25 GiB peak allocated and 2635 MiB free after the
iteration (measured with the earlier Q3 MTP layer at depth 3; the NVFP4 MTP layer adds about 50 MB).

## 4. Measurements

Measurement setup:
- **Prompts:** `promptCorpus: "chat"`, four chat requests with thinking disabled. A 4096-token prompt
  quotes 3964-3970 tokens of a frozen document; a 64-token prompt is a standalone 29-36 token request.
- **Generation:** 256 tokens; warmup 1, then 4 iterations (one per prompt).
- **Forks:** 6 paired forks; each fork runs every arm in a fresh JVM, rotating the starting arm.
- **Rate:** per fork, committed decode tokens over decode time across the four prompts. The
  speculative decode window starts at first-token selection, so it includes the prompt's last MTP
  catch-up.

**Acceptance and step anatomy** (NVFP4, depth 3, all forks; histogram = verifications that accepted 0/1/2/3 drafts):

| Context | Histogram | Mean accepted | Tokens per verification | Verifications per token | Verify ms | Draft ms | Verifications/s | TTFT ms |
|---|---|---|---|---|---|---|---|---|
| 64 | 354/552/312/936 | 1.85 | 2.84 | 0.352 | 28.6 | 2.9 | 31.9 | 38 |
| 4096 | 396/414/438/906 | 1.86 | 2.84 | 0.352 | 31.1 | 3.8 | 28.8 | 1865 |

- **Draft time** is wall time between verifications: catch-up, two recursive rows and host gaps.
- **TTFT** includes the per-chunk catch-up, which adds +1.6 to +4.5 ms at 64 tokens and +53 to +111 ms at 4096.
- **Commit and rollback:** base KV commit and MTP KV truncation only move frontiers. GDN checkpoints
  and records copy 221 MiB per verification (0.47 ms of copy engine, overlapping kernels). The
  replay of up to three committed rows is part of the next verification's time.

Decode rates for the shipped configurations at 1K to 32K contexts are in [MTP_VERIFIER.md](MTP_VERIFIER.md).

## 5. Dependency map

**Dependency map of one step:**

```
VERIFY(P..P+3) ─▶ accept a ─▶ DRAFT catch-up(P..P+a) ─▶ DRAFT(P+a+1) ─▶ DRAFT(P+a+2) ─▶ VERIFY(P+a+1..)
   │ base KV/GDN commit a+1 rows       │ needs the verified hiddens h_P..h_{P+a}
   └ GDN replay (a<3) runs in the next VERIFY's first frames
```

The chain is strictly serial:
- every draft needs the verified hidden states and the accepted count;
- the next verification needs all three drafts;
- inside a verification the four rows are independent only in the linears, and stay causal in
  attention and GDN.

The one independent piece of work MTP exposes is in the prompt. The catch-up for prefill chunk i
depends only on chunk i's hiddens, so it could overlap the base prefill of chunk i + 1. Today it runs
between them (+53 to +111 ms TTFT at 4096 tokens).

A one-row MTP draft costs about 0.74 ms of GEMV. Most of it is the 360 MiB draft head.

## 6. Rejected and open

**Rejected:**
- **`decode_rows` loop interchange.** Decode each weight value once, then FMA it into all M rows,
  with packed activations. Bitwise equal, but slower with cold weights: 1.30-1.53× one row's time,
  against 1.13-1.48× for the shipped kernel.
- **`decode_rows<M ≥ 4>` register prefetch of the next slice.** Bitwise equal. With cold weights,
  1.00-1.23× one row's time against 1.06-1.47×, but in-model NVFP4 + MTP3 was −1% in 5 of 6 forks
  at both contexts (verification 28.8 against 28.6 ms). Not kept.
- **Random-word benchmark prompts for MTP.** The model repeats itself and 31 of 34 steps accept all
  drafts, which overstated MTP3 at 119 tok/s. Replaced by the chat corpus.
- **Per-row Q3 linears in verification** (each row through its own one-row launch): Q3 with MTP3 decoded at 42.5 tok/s at 64 tokens
  and 41.9 at 4096. Replaced by the multi-row Q3 twins.

**Open:**
- **Verification itself is the bottleneck.** Exact multi-row twins of decode attention that share KV reads across the rows
  (per-row arithmetic unchanged) are the next exact-mode candidate, together with the Q3 linear twins
  ([MTP_VERIFIER.md](MTP_VERIFIER.md)).
- **Overlap the prompt's MTP catch-up** with the next prefill chunk (TTFT only).
