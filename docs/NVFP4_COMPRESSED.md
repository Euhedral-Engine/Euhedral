# NVFP4 at 64K: compressed scales and host-mapped embedding

The `nvfp4-compressed` artifact makes NVFP4 usable at 64K context on the 16 GB card. Two mechanisms cover the 1.9 GiB that 64K
needs beyond the free device memory; staging covers the rest ([NVFP4_RESIDENCY.md](NVFP4_RESIDENCY.md)):

- **The token embedding is read from host memory in place** (0.48 GiB).
- **SD4, a 4.25-bit NVFP4 layout for the base model** (0.75 GiB). Each 8-bit block scale becomes a 4-bit index into a per-tensor
  table of 16 E4M3 codes. Its teacher-forced NLL equals plain NVFP4's within noise. The drafting stack (MTP layer, draft head)
  stays plain NVFP4.

**Result:** the compressed artifact runs 64K-token contexts with MTP3 speculative decoding at 47-48 tok/s, with prefill of a
63K-token prompt in about 50 s. Measured end to end, 256 generated tokens, MTP3, chat corpus, one fork per context, with the host
memory each context needed set by hand (the engine now plans it, see below):

| Context | Decode | Verification | Time to first token |
|---|---|---|---|
| 16K | 84.3 tok/s | 29.8 ms | 8.3 s |
| 32K | 68.0 tok/s | 36.5 ms | 20.2 s |
| 64K | 47.8 tok/s | 52.2 ms | 53.1 s |

With the embedding read in place, plain NVFP4 stages 1.65 GiB of projections per step at 64K and verification is long enough to
hide it, so SD4's value at 64K is memory: about 0.9 GiB less host memory for the same context. That margin matters, because
desktop GPU memory drifts.

Measured on 2026-10-02 on an RTX 5070 Ti (16 GB) with an i9-14900K.

## Memory budget at 64K

Measured at 63.4K-token prompts, MTP3:

| Item | Size |
|---|---|
| Device total | 15.56 GiB |
| Desktop (Xorg, browser) and the CUDA context before loading | 0.75-0.95 GiB |
| NVFP4 weights loaded for MTP (text, LM head, MTP layer, draft head, Q3 embedding) | 14.52 GiB |
| Fixed per-sequence state and workspaces (GDN state 0.15 GiB, prefill, verify, scratch) | 0.51 GiB |
| KV cache (NVFP4 pages, 16 attention layers) | 18 KiB per token: about 1.1 GiB at 63.6K |
| Kernel modules and pools after the first runs | 0.07-0.3 GiB |

That is about 17.2 GiB at 64K, so about 1.9 GiB has to leave the device, including a margin for desktop memory, which moved between
470 and 705 MiB during this work. With the `nvfp4` artifact at 14.52 GiB the 64K context needs the host-backed selection of
`ResidencyPlanner`; the `nvfp4-compressed` artifact is 13.77 GiB.

## How far NVFP4 compresses

**Losslessly, very little:**
- An NVFP4 tensor is 4-bit E2M1 codes plus one 8-bit E4M3 scale per 16 values: 4.5 bits per weight.
- The codes carry 3.90 of their 4 bits (byte entropy agrees), because the non-uniform FP4 grid
  already equalizes code use.
- The scales carry 3.63 of their 8 bits.
- A tensor's 16 most frequent scales cover 98.7% of its blocks and the top 32 cover 99.98%, but a
  tensor uses 47-68 distinct scales. A fixed-width lossless index needs 6 bits (0.39 GiB saved), and
  anything shorter needs an escape path in every kernel.

| Family | GiB | Scale entropy (bits) | Top-16 coverage | Top-8 coverage | Code entropy (bits) |
|---|---|---|---|---|---|
| mlp gate_up (x64) | 5.98 | 3.58 | 99.2% | 83.2% | 3.899 |
| mlp down (x64) | 2.99 | 3.62 | 98.8% | 82.3% | 3.899 |
| gdn value_z (x48) | 1.58 | 3.65 | 98.8% | 81.3% | 3.899 |
| gdn output (x48) | 0.79 | 3.68 | 98.3% | 80.9% | 3.899 |
| LM head | 0.67 | 3.73 | 98.0% | 78.8% | 3.899 |
| all NVFP4 tensors | 13.98 | 3.63 | 98.7% | 81.8% | 3.899 |

**With re-quantization, the scale plane halves at no measurable cost (SD4).** The blocks' scales are
chosen again from BF16, from a 16-entry table per tensor. That saves 1/18 of every tensor, 0.78 GiB in
all.

Going further would cut code bits, which is where the information is: Q3 is 3.4 bits per weight and
costs +0.21 nats of NLL. 64K did not need it.

## SD4 layout (`row-split-k128-sd4-v1`)

Rows of K values, K padded to 128:

| Plane | Offset | Size |
|---|---|---|
| E2M1 codes, two per byte, even K in the low nibble (as plain NVFP4) | 0 | rows * K / 2 |
| scale indices, two per byte, even block in the low nibble | align256(rows * K / 2) | rows * K / 32 |
| table: 16 E4M3 codes | align256(index plane end) | 16 |
| FP32 global scale, little endian | table + 16 | 4 |

A weight is e2m1(code) * e4m3(table[index]) * global.

The NaN check at load time (`Nvfp4Layout.validate`) needs only the table, since every index selects a
table entry.

## Converter

```bash
python3 tools/convert_checkpoint.py --model /mnt/shared/qwen38-quant/source/qwen --quantization nvfp4 --compressed \
    --draft-ids-from /mnt/shared/qwen38-quant/artifacts/qwen3_8_27b_q3.edrl \
    --out /mnt/shared/qwen38-quant/artifacts/qwen3_8_27b_nvfp4_compressed.edrl
```

See `tools/README.md`. It runs on the GPU and takes about 14 minutes. SD4's error pass costs about 60 times plain NVFP4's rounding.
`--jobs` 16 saturates the GPU and fills swap; the default of 4 on CUDA is enough.

**The drafting stack stays plain NVFP4.** Drafts only propose, and plain drafting objects cost 33 MiB. Base-model objects are
identical either way (779 of 785 byte for byte), so both versions generate the same tokens, and the difference below is the
drafts' quantization alone. MTP3, chat corpus, paired forks:

| Context | SD4 drafting stack (rejected) | Plain drafting stack |
|---|---|---|
| 4K | 2.674 tokens/step, 92.3 tok/s | 2.702, 93.4 tok/s |
| 16K | 2.776, 83.9 tok/s | 2.807, 84.5 tok/s |

The plain stack was ahead in 3 of 3 forks at 4K (+1.7 / +1.4 / +0.4%) and at 16K (+0.7%).

**The table minimizes the tensor's total squared error.**
- One GPU pass groups blocks by their round-to-nearest scale code o.
- It sums their squared error under every code from o to o+40. Further above, the values all round to
  zero, so the cost is the blocks' energy. Codes below o are not allowed.
- A dynamic program over the 127 E4M3 codes then picks the 16 codes minimizing the total. It costs each
  o at its bracketing table entries, which is exact because a block's error falls to its best scale
  and then rises.
- Each block takes its least-error table scale, never below its round-to-nearest one, so it clips no
  more than plain NVFP4. Its codes round to nearest under that scale.

**What it does to the weights:**
- Squared error is 15-25% below plain NVFP4: plain rounds amax/6, but SD4 measures the error.
- Each table reaches the largest scales its tensor needs.

**Rejected: the 16 most frequent scales.**
- That table stopped a median of 23 codes (three octaves) below each tensor's largest scale.
- Every tensor clipped its largest blocks, by up to 8x, which the averaged weight error hid.
- NLL 4.32 against 2.23: see Quality.

**Rejected: allowing scales 2 codes below round-to-nearest** (`SD4_BELOW=2`).
- It lowered the squared error further and the NLL by 0.003, within noise.
- It clipped the largest weights by up to 14%, which one document cannot show is harmless.

## Quality

`TeacherForcedQualityCudaIntegrationTest` decodes the benchmark corpus document teacher-forced:
2048 tokens after a 512-token prefix, recording every step's logits. `tools/compare_teacher_forced.py`
compares the artifacts on the same tokens:

| Artifact | Mean NLL (nats) | Paired dNLL vs NVFP4 | KL(NVFP4 ‖ artifact) | Top-1 agreement |
|---|---|---|---|---|
| `nvfp4` | 2.2269 | | | |
| **`nvfp4-compressed` (SD4)** | 2.2281 | +0.0011 ± 0.0087 | 0.058 | 86.7% |
| SD4 allowing 2 codes below (rejected) | 2.2240 | -0.0029 ± 0.0090 | 0.063 | 87.4% |
| SD4 with a frequency table (rejected) | 4.3188 | +2.09 | 2.22 | 33.9% |
| `q3` | 2.4320 | +0.2051 ± 0.0193 | 0.317 | 70.9% |

**SD4 and NVFP4 differ from each other about as much as two equally good quantizations do.** Their
KL is a fifth of Q3's. Greedy generation answers the factual prompt identically (" Paris.").

```bash
./gradlew :core:cudaIntegrationTest --tests '*TeacherForcedQualityCudaIntegrationTest' --rerun \
    -Peuhedral.quality.report=/tmp/sd4.bin -Peuhedral.quality.artifact=ARTIFACT -Peuhedral.quality.host-mib=512
python3 tools/compare_teacher_forced.py /tmp/nvfp4.bin /tmp/sd4.bin
```

## Kernels

**Every NVFP4 kernel has an `_sd4` twin:**
- decode for 1 row and for 2-8 rows (`euhedral_nvfp4_decode_sd4`, `euhedral_nvfp4_decode_rows<M>_sd4`);
- the native OMMA linear, gate/up and skinny kernels (`euhedral_nvfp4n_*_sd4`);
- the scalar reference (`euhedral_nvfp4_reference_sd4`), which exact numerics run.

**Correctness: each twin computes, bit for bit, the plain kernel on the tensor expanded to plain
NVFP4.** `native/tests/test_nvfp4.py` and `test_nvfp4_native.py` check this, with every table entry in
use. Host dispatch (`nvfp4_linear.c`) tells the layouts apart by byte size: the two sizes never coincide.

**How each kernel family reads the table:**
- **Decode kernels** decode the table into 16 shared floats once per CTA, beside the E2M1 table they
  already keep, and index it with the lane's scale nibbles.
- **Native kernels** stage the index bytes with `cp.async`, half of plain's scale bytes. Each B scale
  register (4 E4M3 codes) is looked up in registers just before its MMAs: two `prmt` over the table's
  halves, merged on each index's bit 3. Looking scales up in registers instead of shared memory was
  1.20x in the synthetic.

**Decode GEMVs at the model's shapes, cold weights,** weighted by launches per token (synthetic: one kernel launched back to
back): 18.4-19.1 ms at 1 row, 18.6-18.9 ms at 2 rows, 19.4 ms at 3 rows and 28.5 ms at 4 rows (MTP3 verification). In the model,
SD4 verifies a 4-row step in 25.8 ms at 4K and drafts in 3.0 ms; the synthetic 4-row figure does not appear in the model. A trace
of SD4 verification at 16K puts the 4-row GEMVs at 22.8 ms of a 31.3 ms step (73%), streaming weights at about 600 GB/s.

**Rejected: loading scales four slices at a time** (one 256-byte request per row, 128 for SD4, shared out by shuffles; bit for
bit unchanged).
- The synthetic was 9% faster at 4 rows for plain and 26% faster for SD4. Skipping the scale loads altogether had shown them to
  cost about a fifth of a 4-row decode. The register drop also gave 3 CTAs per SM: 168 registers instead of 212-223.
- In the model it was slower in every fork: `nvfp4` at 4K 98.5 -> 81.9 tok/s (-15.0 / -17.2 / -17.0%), SD4 at 4K 92.7 -> 87.5 tok/s
  (-5.6 / -6.1 / -5.3%), SD4 at 16K 84.0 -> 79.7 tok/s (-5.1%). The model launches kernels with programmatic dependent launch and
  interleaves them; the synthetic does neither.

## Host-mapped embedding

The token embedding is the first object `HostWeightSelection` takes when the device cannot hold everything:

- **The embedding is mapped, not staged.** The host-weight arena is pinned with `cudaHostRegisterMapped`, and the embedding
  kernels read the rows they need in place over PCIe: a few KiB per decode token, a few MiB per prefill chunk.
- **It frees 0.48 GiB at no measurable cost.** On the `q3` artifact, which fits whole, 3 paired forks (embedding resident against
  mapped, no projections staged) put decode at 4K, prefill at 2048 tokens and time to first token within 0.2%.
- **Layer projections follow as before,** staged through the ring on each use ([NVFP4_RESIDENCY.md](NVFP4_RESIDENCY.md)).
- **Accounting:** `TensorHandle.hostMapped` marks the embedding as resident to execution, while its
  memory belongs to the host arena. The staging ring ignores it.

## Results

In-process screen: one process per configuration, four chat-corpus prompts at the given length, 256 generated tokens, one warm-up round
and one measured round. MTP3 exact verification. Decode tok/s counts committed tokens over verification, catch-up and draft time
per step. The host memory column is what these runs were given by hand.

| Context | Artifact | Host memory | Staged projections | Lowest free | Verify | Decode | Prefill |
|---|---|---|---|---|---|---|---|
| 16K | `nvfp4-compressed` | 512 MiB | none (embedding only) | 0.29 GiB | 30.0 ms | 83.2 tok/s | 7.6 s |
| 16K | `nvfp4` | 1088 MiB | 0.58 GiB | 0.08 GiB | 32.0 ms | 77.9 tok/s | 8.0 s |
| 32K | `nvfp4-compressed` | 896 MiB | 0.40 GiB | 0.17 GiB | 37.0 ms | 67.1 tok/s | 18.7 s |
| 32K | `nvfp4` | 1376 MiB | 0.86 GiB | 0.09 GiB | 43.2 ms | 58.4 tok/s | 21.1 s |
| 64K | `nvfp4-compressed` | 1408 MiB | 0.90 GiB | 0.35 GiB | 52.9 ms | 47.2 tok/s | 48.3 s |
| 64K | `nvfp4` | 2176 MiB | 1.65 GiB | 0.33 GiB | 54.8 ms | 46.0 tok/s | 49.2 s |

Prefill excludes the MTP layer's pass over the prompt: 0.4 s at 16K, 1.2 s at 32K, 4.0 s at 64K.

**At 64K, verification is long enough to hide most staging.** One verification takes 55-62 ms: attention over 64K keys plus the
4-row linears, so plain NVFP4 stages 1.65 GiB per step at little extra cost.

**Cross-process noise is large at these lengths.** Two plain NVFP4 64K runs differed by 12% in verify time (62.2 ms at 1920 MiB,
54.8 ms at 2176 MiB). Host-transfer bandwidth varies between pinned-arena allocations, by up to 16% in the MTP verifier campaign.
Single runs at 16K-64K do not separate the artifacts' speeds; their memory does.

**SD4's value is margin and less staging.** It fits each context with 0.6-0.9 GiB less host memory, which matters as desktop
memory moves: a plain NVFP4 run at 1920 MiB ran out of memory once the desktop grew from 0.5 to 0.7 GiB.

**Serving 64K.** Set `euhedral.inference.max-context-tokens: 65536`. The engine reads the artifact's quantization and layout and
`ResidencyPlanner` chooses the host-backed weights (embedding first, then layer projections, through a 4-slot device ring) that
leave room for that much KV cache plus the per-sequence state; startup fails with a message to lower the maximum when even the
largest selection does not fit. MTP3 is the policy for NVFP4 artifacts.

## Measurement notes

- **`QwenSpeculativeDecoder.Statistics.catchUpNanos` excludes the MTP layer's catch-up over the prompt.** That catch-up (45 ms per
  step at 63K-token prompts) is reported in `promptCatchUpNanos`.

## Not done

- **Faster 4-row verification GEMVs.** In the model they run at about 600 GB/s, two-thirds of peak.
  The synthetic does not predict in-model changes here, so candidates need in-model gates.
- **Code-bit reduction** (below 4 bits per value), the only route to large further savings.
- **The 0.51 GiB of fixed per-sequence workspaces**, which could hold about 28K more tokens of KV.
