# Artifact converter

`convert_checkpoint.py` converts a Qwen3.8-27B Hugging Face checkpoint directory (BF16 safetensors,
`config.json`, `tokenizer_config.json`) to one of the four artifacts the engine ships. Two choices select
the artifact: the quantization and the representation.

| Artifact | Quantization | Representation | Weights |
|---|---|---|---|
| `q3` | `--quantization q3` | uncompressed | Q3 text weights (Q4/Q5 for the attention and GDN input projections), NVFP4 MTP layer, Q3 draft head |
| `q3-compressed` | `--quantization q3 --compressed` | compressed | the `q3` artifact, losslessly transcoded: every tensor decodes to the same values |
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

- Q3, Q4 and Q5 try scales that map the group's largest magnitude to either end of the code range (so the most
  negative code is used), refit each by weighted least squares, and keep the best after rounding to binary16;
- NVFP4 tries the block scale codes from 6 below to 2 above the round-to-nearest code;
- SD4 chooses its table and each block's entry under the weighted error.

Objects the matrix does not cover (the token embedding, the LM and draft heads, the MTP layer) are searched
with uniform weights. Formats and layouts are unchanged; the manifest records the matrix and its SHA-256.
Calibrated rounding needs PyTorch.

```bash
python3 tools/convert_checkpoint.py --model $CHECKPOINT --quantization q3 \
    --draft-ids-from $OUT/qwen3_8_27b_q3.edrl --imatrix imatrix-qwen3.8-27b.gguf --out $OUT/qwen3_8_27b_q3.edrl
```

The importance matrix measured so far is the public one published with ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF
(`imatrix-qwen3.8-27b.gguf`, 1000 chunks of 4096 tokens).

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
- `compare_teacher_forced.py`: compares teacher-forced logits reports with one another (NumPy).
- `compare_reference.py`: compares teacher-forced logits reports with a BF16 reference written by llama.cpp (NumPy).
- `dflash2_reference.py`: runs the upstream DFlash2 draft model (z-lab/dflash, unmodified; needs PyTorch, transformers 5.15 and the
  upstream source tree) over fixed inputs and writes every intermediate as fixtures (docs/DFLASH2.md).
- `dflash2_compare.py`: per-tensor relative error between two DFlash2 fixture directories (NumPy).
- `dflash2_quality.py`: compares two drafters' quality records (DFlash2QualityCudaIntegrationTest).
- `render_qwen_chat_template_golden.py`: renders chat-template golden prompts (jinja2).
