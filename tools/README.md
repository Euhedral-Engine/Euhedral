# Artifact converter

`convert_checkpoint.py` converts a Qwen3.5-27B Hugging Face checkpoint directory (BF16 safetensors,
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
    --ranking $RANKING --out $OUT/qwen3_5_27b_q3.edrl

# q3-compressed
python3 tools/convert_checkpoint.py --model $CHECKPOINT --quantization q3 --compressed \
    --draft-ids-from $OUT/qwen3_5_27b_q3.edrl --out $OUT/qwen3_5_27b_q3_compressed.edrl

# nvfp4
python3 tools/convert_checkpoint.py --model $CHECKPOINT --quantization nvfp4 \
    --draft-ids-from $OUT/qwen3_5_27b_q3.edrl --out $OUT/qwen3_5_27b_nvfp4.edrl

# nvfp4-compressed
python3 tools/convert_checkpoint.py --model $CHECKPOINT --quantization nvfp4 --compressed \
    --draft-ids-from $OUT/qwen3_5_27b_q3.edrl --out $OUT/qwen3_5_27b_nvfp4_compressed.edrl
```

`--jobs N` sets the number of parallel worker processes (default 4 on cuda, one per CPU on cpu), and
`--force` replaces an existing artifact. Each command writes the artifact and `ARTIFACT.manifest.json`
(object, format and layout counts, payload bytes, SHA-256) through temporary files renamed into place.
For `q3-compressed` the uncompressed Q3 form is built beside the output, transcoded, and removed; every
transcoded tensor is decoded back and compared byte for byte with its source before the output is published.

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
- `compare_teacher_forced.py`: compares teacher-forced logits reports (NumPy).
- `compare_compact_edrl_reference.py`: compares an artifact's objects and payloads with an NInfer v2 artifact.
- `render_qwen_chat_template_golden.py`: renders chat-template golden prompts (jinja2).
