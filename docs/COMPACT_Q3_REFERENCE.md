# Q3 artifact format

This is the contract of the `q3` artifact (`tools/convert_checkpoint.py --quantization q3`, see `tools/README.md`) and of the
Q3G64_F16S, Q4G64_F16S and Q5G64_F16S weight formats the engine executes. The `q3-compressed` artifact stores the same tensors in
the lossless P2E2 layout ([COMPRESSED_Q3.md](COMPRESSED_Q3.md)).

## Inventory

The artifact holds the text model and the MTP draft head, 785 objects and no vision tower:

- 360 BF16 objects (norms, the GDN convolution and A/B projections), `contiguous-le-v1`
- 96 FP32 objects (GDN `a_log` and `dt_bias`), `contiguous-le-v1`
- 1 I32 object (`text/draft_head_token_ids`), `contiguous-le-v1`
- 195 Q3G64_F16S objects, `row-split-k128-v1`: the token embedding, the LM head, the draft head, and per layer the attention
  and GDN output projections and the FFN `gate_up` and `down`
- 64 Q4G64_F16S objects, `row-split-k128-v1`: `attention/query_key` and `gdn/query_key`
- 64 Q5G64_F16S objects, `row-split-k128-v1`: `attention/gate_value` and `gdn/value_z`
- 5 NVFP4 objects, `row-split-k128-v1`: the MTP layer's projections (`mtp/input_projection` and the four projections of
  `mtp/layer`). NVFP4 drafts accept more tokens per verification than Q3 ones ([MTP_VERIFIER.md](MTP_VERIFIER.md)).

The container is EDRL version 2, the only version the reader accepts: each tensor descriptor carries the source dtype, the
persistent storage format and the persistent layout as separate fields.

Canonical fused text objects:

- `text/layers/<n>/attention/query_key` (Q4) and `attention/gate_value` (Q5)
- `text/layers/<n>/gdn/query_key` (Q4) and `gdn/value_z` (Q5)
- `text/layers/<n>/mlp/gate_up` and `mlp/down`
- `mtp/layer/attention/query_key_gate_value`

## Layouts

`row-split-k128-v1`: a row of K values is padded to a multiple of 128 and split into groups of 64 values. Each plane starts on a
256-byte boundary.

| Format | Code range | Base plane per group | High plane per group | Scale plane |
|---|---|---|---|---|
| Q3G64_F16S | -4..3 | 24 bytes, three bits per code | none | one little-endian binary16 scale per group |
| Q4G64_F16S | -8..7 | 32 bytes, one nibble per code | none | same |
| Q5G64_F16S | -16..15 | 32 bytes of low nibbles | 8 bytes of fifth bits | same |

The planes are stored in the order base, high, scales. A scale is `fp16(max-abs / qmax)` and codes are quantized after the scale
is rounded to binary16. Rows are padded on K to 128.

The Java loader validates descriptor metadata, reads the exact payload range, allocates the exact byte count, uploads the bytes
and keeps fused objects as fused `TensorHandle` instances. It does not quantize, dequantize, split or repack at load time.

Layout sources: `native/src/q3/layout.cuh` and `native/src/q45/layout.cuh` (device), `tools/euhedral_artifacts/grouped.py`
(converter).

## Execution

Quantized linears run by row count. One to eight rows run the decode kernels: the one-row contiguous kernel and its 2 to 8 row
twins, whose every row equals the one-row result bit for bit (`native/src/q3/contiguous.cuh`, `native/src/q45/contiguous.cuh`).
Nine or more rows run the block-scaled MXFP8 route ([PREFILL_MX.md](PREFILL_MX.md)). The scalar references in
`native/src/reference/kernels.cu` are the numerical oracle: exact numerics (`CudaGpuMemory.selectExactNumerics`, used by tests
and never selected in production) run them, and so does any shape no kernel takes.

## P2E2

The Q3 tensors of the `q3-compressed` artifact use the smaller, lossless `row-split-p2e2-v1` layout (EDRL layout ordinal 2,
`WeightLayout.ROW_SPLIT_P2E2_V1`), produced by `tools/convert_checkpoint.py --quantization q3 --compressed`. The loader checks
each P2E2 tensor's internal consistency before uploading it. See [COMPRESSED_Q3.md](COMPRESSED_Q3.md).
