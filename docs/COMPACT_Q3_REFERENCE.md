# Compact Q3 EDRL reference contract

The reference is the NInfer artifact `qwen3_8_27b_compact_v2.ninfer`, inspected directly from the local model directory.
The binary directory was inspected directly; it contains 1,124 persistent objects:

- 6 frontend resources (not represented as model weight descriptors in EDRL)
- 1,118 tensors
- 12,858,941,344 persistent tensor bytes
- 773 `text/` tensors, 12 `mtp/` tensors, and 333 `vision/` tensors

The compact tensor inventory is:

- BF16: 582 objects, `contiguous-le-v1`
- FP32: 96 objects, `contiguous-le-v1`
- I32: 1 object, `contiguous-le-v1`
- Q3G64_F16S: 199 objects, `row-split-k128-v1`
- Q4G64_F16S: 118 objects, `row-split-k128-v1`
- Q5G64_F16S: 118 objects, `row-split-k128-v1`
- Q6G64_F16S: 1 object, `row-split-k128-v1`
- W8G32_F16S: 3 objects, `row-split-k128-v1`

EDRL version 2 carries source dtype, persistent storage format, and persistent
layout as separate descriptor fields. Version 1 remains unchanged for the raw
BF16 artifact.

Canonical fused text objects include:

- `text/layers/<n>/attention/query_key`
- `text/layers/<n>/attention/gate_value`
- `text/layers/<n>/gdn/query_key`
- `text/layers/<n>/gdn/value_z`
- `text/layers/<n>/mlp/gate_up`
- `text/layers/<n>/mlp/down`
- `mtp/layer/attention/query_key_gate_value`

Q3 objects use signed grouped codes with group size 64 and binary16 scales.
Codes are quantized after binary16 scale rounding, with qmin=-4 and qmax=3.
The row-split payload stores a packed low-bit base plane, optional high-bit
plane, then the little-endian binary16 scale plane. Rows are padded on K to 128,
and planes are aligned to 256 bytes. The converter implements the reference
host scale oracle and three-bit low-plane packing from NInfer.

The Java loader is intentionally opaque: it validates descriptor metadata,
reads the exact payload range, allocates the exact byte count, uploads the
bytes, and retains fused objects as fused `TensorHandle` instances. It does
not quantize, dequantize, split, or repack at model-load time.

The Q3 tensors may also use the smaller, lossless `row-split-p2e2-v1` layout (EDRL layout ordinal 2,
`WeightLayout.ROW_SPLIT_P2E2_V1`), produced from a compact artifact by
`tools/convert_compact_edrl_to_p2e2.py`. The loader checks each P2E2 tensor's internal consistency
before uploading it. See [COMPRESSED_Q3.md](COMPRESSED_Q3.md).
