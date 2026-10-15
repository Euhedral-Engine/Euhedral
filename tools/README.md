# Artifact converter

`convert_checkpoint.py` converts a Qwen3.8-27B Hugging Face checkpoint directory (BF16 safetensors,
`config.json`, `tokenizer_config.json`) to one of the four artifacts the engine ships. Two choices select
the artifact: the quantization and the representation.

| Artifact | Quantization | Representation | Weights |
|---|---|---|---|
| `q3` | `--quantization q3` | uncompressed | Q3 text weights (Q4/Q5 for the attention and GDN input projections), NVFP4 MTP layer, Q3 draft head |
| `q3-compressed` | `--quantization q3 --compressed` | compressed | a Q3 artifact losslessly transcoded to P2E2, MLP down projections kept row-split: every tensor decodes to the values of its Q3 form |
| `nvfp4` | `--quantization nvfp4` | uncompressed | NVFP4 text weights, MTP layer and draft head; Q3 token embedding |
| `nvfp4-compressed` | `--quantization nvfp4 --compressed` | compressed | NVFP4 text weights at 4.25 bits per weight; plain NVFP4 MTP layer and draft head; Q3 token embedding |

All four artifacts hold the text model and the MTP draft head (785 objects) and no vision tower.

## Commands

The draft head's token shortlist is chosen from a token frequency ranking (`--ranking FILE`, int64 little
endian counts, one per vocabulary entry) or taken from an existing artifact (`--draft-ids-from ARTIFACT`).
Pass exactly one. Quantization runs on the GPU when PyTorch finds one (`--device cuda|cpu` overrides) and
needs NumPy; the GPU path needs PyTorch.

```bash
CHECKPOINT=/mnt/shared/qwen38-quant/source/qwen
OUT=/mnt/shared/qwen38-quant/artifacts
RANKING=/path/to/token-ranking.i64

# q3
python3 tools/convert_checkpoint.py --model $CHECKPOINT --quantization q3 \
    --ranking $RANKING --out $OUT/qwen3_8_27b_q3.edrl

# q3-compressed
python3 tools/convert_checkpoint.py --model $CHECKPOINT --quantization q3 --compressed \
    --draft-ids-from $OUT/qwen3_8_27b_q3.edrl --out $OUT/qwen3_8_27b_q3_compressed.edrl

# nvfp4
python3 tools/convert_checkpoint.py --model $CHECKPOINT --quantization nvfp4 \
    --draft-ids-from $OUT/qwen3_8_27b_q3.edrl --out $OUT/qwen3_8_27b_nvfp4.edrl

# nvfp4-compressed
python3 tools/convert_checkpoint.py --model $CHECKPOINT --quantization nvfp4 --compressed \
    --draft-ids-from $OUT/qwen3_8_27b_q3.edrl --out $OUT/qwen3_8_27b_nvfp4_compressed.edrl
```

`--jobs N` sets the number of parallel worker processes (default 4 on cuda, one per CPU on cpu), and
`--force` replaces an existing artifact. Each command writes the artifact and `ARTIFACT.manifest.json`
(object, format and layout counts, payload bytes, SHA-256) through temporary files renamed into place.
For `q3-compressed` the uncompressed Q3 form is built beside the output, transcoded, and removed; every
transcoded tensor is decoded back and compared byte for byte with its source before the output is published.

## Calibrated rounding

`--imatrix FILE` rounds every quantized object by calibrated scale search instead of taking each group's scale
from its largest value. FILE is a llama.cpp importance matrix (GGUF, as `llama-imatrix` writes): the mean square
of each linear layer's input columns over a calibration text. Each group's scale minimizes its squared error
weighted by those column importances:

- Q3, Q4 and Q5 try scales that map the group's largest magnitude to either end of the code range, refit each by
  weighted least squares, and keep the best after rounding to binary16. Scales are signed: a group whose largest
  magnitude is positive reaches the most negative code (-4 for Q3) through a negative scale, which the kernels apply as
  they apply any scale;
- NVFP4 tries the block scale codes from 6 below to 2 above the round-to-nearest code;
- SD4 chooses its table and each block's entry under the weighted error.

Objects the matrix does not cover (the token embedding, the LM and draft heads, the MTP layer) are searched
with uniform weights. Formats and layouts are unchanged; the manifest records the matrix and its SHA-256.
Calibrated rounding needs PyTorch. Compressed q3 limits its Q3 tensors to the codes -3..3, which its P2E2 layout stores,
except the MLP down projections, which it keeps row-split with the full range (docs/COMPRESSED_Q3.md).

```bash
python3 tools/convert_checkpoint.py --model $CHECKPOINT --quantization q3 \
    --draft-ids-from $OUT/qwen3_8_27b_q3.edrl --imatrix imatrix-qwen3.8-27b.gguf --out $OUT/qwen3_8_27b_q3.edrl
```

Use the importance matrix of the checkpoint being converted. Two public ones are measured:

- Qwen3.8-27B: `imatrix-qwen3.8-27b.gguf` from ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF (1000 chunks of 4096 tokens);
- Swift-1.5: `imatrix-swift15-v1mix.gguf` from ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF (309 chunks of 4096 tokens of web,
  reasoning, multilingual and code text), used for all four artifacts, which are converted from Swift-1.5.

## Pre-rounded codes and fake quantization

`--codes-from DIR` writes the quantized objects DIR holds files for from those files instead of rounding them, and
rounds the rest as usual (with `--imatrix` if given). A file is one object's stored rounding
(`euhedral_artifacts/codes.py`, `text.layers.10.mlp.down.npz` for `text/layers/10/mlp/down`): the integer codes and
FP16 group scales of a Q3/Q4/Q5 object, or the E2M1 codes, E4M3 block scales (SD4: table indices and the table) and
FP32 tensor scale of an NVFP4 object. It must match the object's format, layout and shape in the chosen artifact;
the manifest records DIR and how many objects it supplied.

`euhedral_artifacts/fake_quant.py` holds the same rounding as PyTorch modules for training: built from an object's
weights with the converter's own scale selection, a module's codes and scales are the converter's byte for byte, its
forward pass returns exactly the weights they decode to, and gradients reach the master weights and the scales
through straight-through rounding. `state()` exports a file for `--codes-from`. `tools/fake_quant_parity.py` rebuilds
every quantized object of an artifact as a module and compares its export with the artifact's bytes and its forward
pass with the decoded weights:

```bash
python3 tools/fake_quant_parity.py --model $CHECKPOINT --artifact $OUT/qwen3_8_27b_q3.edrl --quantization q3 \
    --imatrix imatrix-swift15-v1mix.gguf [--objects REGEX] [--export DIR]
```

On the calibrated Swift-1.5 `q3` (with its LM head in Q3), `nvfp4` and `nvfp4-compressed` artifacts every quantized
object, 328 per artifact, matches byte for byte and every module's forward pass equals its decoded weights exactly; a parity run takes
5 to 11 minutes on the GPU. Converting `q3` and `nvfp4-compressed` again with the exported codes of layers 10 and 11, the
LM head and the MTP input projection (`--codes-from`) reproduces each artifact's SHA-256.

The fake-quant model and the engine running the artifact agree closely: with the modules' decoded weights in a BF16
GGUF run by llama.cpp as the reference, the engine's teacher-forced KL over T2 (docs/QUALITY.md) is 0.0076 for that `q3`
(dNLL +0.005, top-1 agreement 96.5%) and 0.0081 for `nvfp4-compressed` (dNLL +0.001, 96.7%).

## GPTQ

`gptq/calibration.py` builds the GPTQ calibration set; the `quantization` module (quantization/README.md) rounds an
artifact's decoder layers and heads by GPTQ on the engine's kernels and writes the codes for `--codes-from`
(docs/QUALITY.md#gptq). The calibration set needs `tokenizers` and `jinja2`:

```bash
cd tools
# the calibration set: prompts disjoint from T4, the served model's answers, prose and code
python -m gptq.calibration prompts $C/prompts.jsonl $HOLDOUT/t4_prompts.jsonl
python -m gptq.calibration generate http://localhost:8080 $C/prompts.jsonl $C/generations.jsonl --parallel 2
python -m gptq.calibration tokens $CHECKPOINT $C            # writes $C/tokens.npy
# the codes, one layer at a time on the GPU; then the artifact
(cd .. && ./gradlew :quantization:run --args="calibrate --model $CHECKPOINT --quantization nvfp4 --compressed \
    --tokens $C/tokens.npy --draft-ids-from $OUT/qwen3_8_27b_q3.edrl --out $CODES --act-order")
python3 convert_checkpoint.py --model $CHECKPOINT --quantization nvfp4 --compressed \
    --draft-ids-from $OUT/qwen3_8_27b_q3.edrl --imatrix imatrix-swift15-v1mix.gguf --codes-from $CODES \
    --out $OUT/qwen3_8_27b_nvfp4_compressed.edrl
```

`calibrate` holds the hidden states of all sequences in pinned host memory (5.4 GiB for 218 sequences of 2048
tokens); `--sequences N` and `--layers N` make a short run.

## DFlash2

`--dflash2 DIR` adds the DFlash2 drafter (`z-lab/Qwen3.8-27B-DFlash2`: `config.json`, `model.safetensors`) to a conversion, and
`--extend ARTIFACT --dflash2 DIR` appends it to an existing artifact, copying its objects byte for byte. An artifact with the drafter
speculates with DFlash2 (docs/DFLASH2.md). `--dflash2-projections` stores the drafter's attention and MLP projections as `bf16`
(the default) or `nvfp4`; `nvfp4-all` also stores the feature fusion and the convolutions' kernel projections as NVFP4.

```bash
python3 tools/convert_checkpoint.py --extend $OUT/qwen3_8_27b_nvfp4_compressed.edrl \
    --dflash2 /mnt/shared/Qwen3.8-27B-DFlash2 --dflash2-projections nvfp4 \
    --out $OUT/qwen3_8_27b_nvfp4_compressed_dflash2.edrl
```

## Layout

- `convert_checkpoint.py`: the command line.
- `euhedral_artifacts/`: the importable modules.
  - `recipes.py`: what each of the four artifacts stores, per object.
  - `inventory.py`: the 785 objects of the checkpoint and their writers.
  - `pipeline.py`: checkpoint directory to artifact file.
  - `grouped.py`, `nvfp4.py`: the quantizers (Q3/Q4/Q5; NVFP4 and the table-scale NVFP4 layout of
    `docs/NVFP4_COMPRESSED.md`).
  - `q3_p2e2.py`: the lossless Q3 transcode (`docs/COMPRESSED_Q3.md`).
  - `edrl.py`, `sources.py`, `device.py`: container format, safetensors access, CPU/CUDA selection.
- `test_*.py`: unit tests, which need NumPy and no GPU or checkpoint (`python3 -m unittest discover -s tools`;
  three CUDA-versus-CPU byte-equality tests run when PyTorch has a CUDA device).
- `nvfp4z.py`, `nvfp4z/`: lossless compression of NVFP4 safetensors checkpoints to about 90% of their size, on the GPU (PyTorch with
  CUDA; docs/NVFP4_LOSSLESS.md). `test_nvfp4z.py` runs on any machine; its GPU tests need a CUDA device.
- `convert_flash_next.py`, `flash_next_inventory.py`, `euhedral_artifacts/qwen4_source.py`, `qwen4_edrl.py`: the Qwen3.8-Flash-Next
  (`qwen4_exp`) NVFP4 checkpoint to an EDRL version 3 artifact, restartable and byte-verified, and its tensor inventory
  (`docs/FLASH_NEXT_ARTIFACT.md`). `test_flash_next_converter.py` converts a miniature checkpoint and needs NumPy only.
- `trellis/make_fixture.py`: golden fixtures for the `quantization` module's trellis port, generated by exllamav3 1.6.1's own
  code (PyTorch with CUDA, exllamav3 on `PYTHONPATH`): 64 tiles quantized and packed at every rate (2 to 6, 2.5, 3.5),
  one linear quantized end to end with its weight, Hessian and outputs, and the bit allocations of a checkpoint at
  2.5 to 3.5 bits (`python -m trellis.make_fixture OUT_DIR CHECKPOINT`, run from `tools`).
- `compare_teacher_forced.py`: compares teacher-forced logits reports with one another (NumPy).
- `dflash2_reference.py`: runs the upstream DFlash2 draft model (z-lab/dflash, unmodified; needs PyTorch, transformers 5.15 and the
  upstream source tree) over fixed inputs and writes every intermediate as fixtures (docs/DFLASH2.md).
- `dflash2_compare.py`: per-tensor relative error between two DFlash2 fixture directories (NumPy).
- `dflash2_quality.py`: compares two drafters' quality records (DFlash2QualityCudaIntegrationTest).
- `render_qwen_chat_template_golden.py`: renders chat-template golden prompts (jinja2).

## Conversation generation

`python3 -m t5.generate SERVER_URL STORE --model ID --base-seed N --parallel N [--domains D ...] [--limit N]
[--max-tokens N] [--target-tokens N] [--artifact-manifest PATH] [--allow-new-commit]` (run from `tools/`) drives the scripted
conversations in `STORE/conversations.jsonl` through the server's chat API, executing tool calls against each
conversation's fixture. A script that ends before `--target-tokens` continues with follow-up turns from `t5/follow_ups.py` (depth, examples, edge cases; per domain and language). Greedy or sampled decoding comes from `t5.records.plan_decoding`. Token ids are taken from
the server (`return_token_ids`), not re-tokenized. `--parallel` is required and must match the server's maximum number of
sequences. The server must run with `euhedral.api.max-queued-generations` at least `--parallel` and with
`euhedral.api.request-timeout` above the longest request (its 30m default is counted from arrival including queue
wait, and BF16 requests take 45+ minutes; use for example 6h). A request whose prompt plus `--max-tokens` exceeds the server's context is sent again with `max_tokens` cut to the room left. A timeout answer is recorded as an error and not
retried; a full queue or restarting server (502/503/504) is retried with backoff for about 30 minutes. Ctrl-C stops
the run: workers stop before their next request and an interrupted conversation is not recorded; a request in
flight is not abandoned, so the process exits when it returns (a second Ctrl-C exits at once). Finished
conversations stay in the file.

All output is one file, `STORE/generation/progress.json`, rewritten atomically after every finished conversation.
Rerunning the same command resumes: finished conversations are skipped, an interrupted one restarts from its
beginning, and a changed server, model, seed, limit, artifact, input file or git commit is refused. `--allow-new-commit`
resumes at a new commit and keeps both commits in the file.
