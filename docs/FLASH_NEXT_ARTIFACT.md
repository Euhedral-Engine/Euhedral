# Qwen3.8-Flash-Next artifact (EDRL version 3)

Qwen3.8-Flash-Next (`qwen4_exp`) is not a variant of the dense Qwen3.8-27B: it has 512 routed experts per layer, hyper-connections
instead of layer norms, sparse attention with an indexer, and an n-gram embedding of 51 billion parameters. The engine therefore
carries it as its own architecture (`ModelArchitecture.QWEN4_EXP`) with its own container version, loader, residency planner and
storage classes; the dense path (EDRL version 2) is unchanged.

The artifact holds everything the runtime needs (configuration, tokenizer-independent weights, layouts). Nothing reads
`config.json`, safetensors metadata, Hugging Face code or Python after conversion.

## Source checkpoint

`/mnt/shared/qwen38-flash-next/nvfp4`: 131 safetensors shards, 2,732 tensors, 104,810,594,748 stored bytes. It is already NVFP4
(E2M1 codes, E4M3 block scales of 16, FP32 global scales; routed experts carry one global scale per expert). The converter
copies every code, scale and global scale unchanged; nothing is quantized again.

### Inventory

`tools/flash_next_inventory.py CHECKPOINT` classifies every tensor from the safetensors headers. Parameters are logical
(NVFP4: rows x K); stored bytes are the checkpoint's; artifact bytes are those of the runtime objects.

| group | objects | parameters | stored bytes | artifact bytes | storage |
|---|---|---|---|---|---|
| routed experts (48 text banks) | 48 banks, 24,576 experts | 120,795,955,200 | 67,947,921,408 | 68,048,388,096 | NVFP4 |
| n-gram embedding | 128 shards | 51,200,245,760 | 28,800,138,752 | 28,800,156,160 | NVFP4 |
| MTP (1 layer, with its 512 experts) | 29 + 1 bank | 2,607,150,848 | 1,499,213,348 | 1,501,306,404 | NVFP4 and BF16 |
| vision | 333 | 448,931,056 | 897,862,112 | 897,862,112 | BF16 |
| hyper-connections | 387 | 640,624,640 | 1,281,249,280 | 1,281,249,280 | BF16 |
| token embedding | 1 | 635,699,200 | 1,271,398,400 | 1,271,398,400 | BF16 |
| output head | 1 | 635,699,200 | 1,271,398,400 | 1,271,398,400 | BF16 |
| GDN (36 layers) | 216 | 2,086,502,400 | 1,188,495,792 | 1,188,495,792 | NVFP4 projections, BF16 rest |
| QSA and indexer (12 layers) | 60 | 617,349,120 | 375,521,472 | 375,521,472 | NVFP4 projections, BF16 indexer |
| shared experts | 192 | 236,052,480 | 132,956,736 | 132,956,736 | NVFP4, BF16 gate |
| routers | 48 | 62,914,560 | 125,829,120 | 125,829,120 | BF16 |
| fixed text (n-gram projections) | 3 | 32,808,960 | 18,513,928 | 18,513,928 | NVFP4, BF16 |
| norms and small state | 159 | 48,000 | 96,000 | 96,000 | BF16 |
| **total** | 1,606 | 179,999,981,424 | 104,810,594,748 | 104,913,171,900 | |

Per layer: routed experts 1,417,674,752 bytes; GDN 33,013,772; QSA 31,293,456; hyper-connections 26,419,200; shared
expert 2,769,932; router 2,621,440. Per expert: 2,764,808 bytes stored, 2,768,896 in its record (676 x 4096).

Shape families (rows x K): routed expert gate_up 1280 x 2560 and down 2560 x 640 (all 49 banks, all 512 experts, equal); GDN
`in_proj_qkv` 10240 x 2560, `in_proj_z` 6144 x 2560, `out_proj` 2560 x 6144; QSA `q_proj` 12288 x 2560 (queries and output gate),
`k_proj`/`v_proj` 512 x 2560, `o_proj` 2560 x 6144, indexer 640 x 2560; shared expert 640 x 2560 and 2560 x 640;
n-gram shard 2,500,012 rows x 160.

## Container

All integers in the header, metadata and tables are big-endian.

```
header (64) | metadata | tables | zero padding to 4096 | payloads
```

Header: `int32 magic 0x5157454E, int32 version 3, int32 architecture (1 = qwen4_exp), int32 reserved 0, int64 metadataOffset (64),
metadataSize, tablesOffset, tablesSize, dataOffset (multiple of 4096), fileSize`. A conversion writes the header last, so a
partial file is never valid.

Metadata: typed key/value pairs sorted by key (`int32 count`, then key, type byte, value): int64, binary64, boolean, string,
int64 array, string array. The reader requires exactly the keys it knows: topology (`text.*`, `attention.*`, `gdn.*`, `qsa.*`,
`moe.*`, `hc.*`), n-gram (`ngram.*`, `ple.*`: shard geometry, per-head offsets and vocabulary sizes, hash multipliers), MTP
(`mtp.*`), vision (`vision.*`), quantization (`quant.*`) and the maximum trained context (`text.max_position_embeddings`).

Tensor table: for each fixed object `name, rank, shape (int64 each), dtype, format, layout, group, offset, size, CRC-32`. Expert
bank table: for each bank `name, group, layer, expertCount, projections (name, shape, dtype, format, layout, offsetInRecord,
size)`, then one `(offset, size, CRC-32)` per expert. Every payload starts on a 256-byte boundary; banks and n-gram shards on 4096.
The CRC-32 is IEEE (`zlib.crc32`, `java.util.zip.CRC32`) over the object's bytes.

Dense artifacts stay version 2; a v2 reader rejects a version 3 file, and `ModelArchitecture.detect` tells them apart from the
first twelve bytes.

### Objects

Names follow the dense convention (`text/layers/N/...`): `text/token_embedding`, `text/output_head`,
`text/hyper_connection_mixer/{hc_norm,input_mix_down,input_mix_up}`; per layer `attn_hc` and `mlp_hc` (`hc_norm`,
`input_mix_down`, `input_mix_up`, `block_inject`), `gdn/{a_log,dt_bias,conv1d,in_proj_a,in_proj_b,in_proj_qkv,in_proj_z,out_proj,norm}`
or `attention/{q_proj,k_proj,v_proj,o_proj,q_norm,k_norm,indexer/{index_qk_proj,q_layernorm,k_layernorm}}`,
`moe/{router,shared_expert/{gate_proj,up_proj,down_proj},shared_expert_gate}` and the bank `moe/experts`; layer 1 also holds
`ple/{key_proj,value_proj,conv1d,norm_*}` and `ple/ngram/shard_000..127` (the checkpoint's `ple_layer_ids` of `[2]` are
numbered from 1). `mtp/...` mirrors a layer plus `fc_embedding`, `fc_hidden` and the pre-fusion norms; `vision/...` keeps the
tower's names. `ExpectedInventory.expected(config)` derives the whole list from the configuration, and the validator accepts an
artifact only when its tables are exactly that: no missing, extra, misshapen or mis-grouped object.

Every fixed NVFP4 tensor is a `row-split-k128-v1` tensor of the dense format (`Nvfp4Layout`): codes, scales at the next 256-byte
boundary, the global scale at the next. The group of each object is stored in its table entry, so placement never depends on
parsing names.

### Expert banks

One bank per layer, one record per expert, each record a contiguous range holding both projections as complete NVFP4 tensors:

```
record (2,768,896 bytes, 4096-aligned):
  gate_up  [1280 x 2560]   offset 0          1,843,204 bytes (codes 1,638,400 | scales 204,800 | global scale 4)
  down     [2560 x 640]    offset 1,843,456    921,604 bytes (codes   819,200 | scales 102,400 | global scale 4)
  zero padding to the stride
```

A cache miss is one copy of one range. `ExpertBank` keeps the record table in primitive arrays (offset, size, CRC-32 per
expert; no per-expert objects), so a bank may have unequal records: the slot size is the bank's largest record, and the reader
does not assume a fixed stride. This model's 49 banks have 512 equal records.

### N-gram shards

`row-interleaved-nvfp4-v1` (`NgramLayout`): rows of 160 values are 90 bytes (80 bytes of codes, then 10 block scales), no padding
between rows, then the FP32 global scale at the next 256-byte boundary. A row is one contiguous range. The table is 128 shards of
2,500,012 rows; the 16 hash heads address 20,000,003 to 20,000,171 rows each from `heads_offsets`, 320,001,446 of the 320,001,536
rows.

## Converting

```bash
python3 tools/convert_flash_next.py convert /mnt/shared/qwen38-flash-next/nvfp4 ARTIFACT.edrl [--jobs 4] [--force]
python3 tools/convert_flash_next.py verify  ARTIFACT.edrl --checkpoint /mnt/shared/qwen38-flash-next/nvfp4
```

`convert` plans the objects from the checkpoint (rejecting unknown tensors, incomplete NVFP4 triples, inconsistent shapes, or
n-gram layers that disagree with the config), preallocates `ARTIFACT.edrl.partial`, writes objects in parallel worker processes
(experts in chunks of 64) and journals each completed chunk with its CRC-32s in `ARTIFACT.edrl.progress`. Run again after an
interruption and only the missing chunks are written; the finished bytes equal an uninterrupted conversion's. It then writes the
tables and the header, renames the file into place and writes `ARTIFACT.edrl.manifest.json` (artifact size, object counts,
bytes per group, format and layout counts, source checkpoint, SHA-256). `verify` checks that no ranges overlap, every range's
CRC-32, and with `--checkpoint` every byte of every object against the checkpoint.

## Artifact

`/home/brandon/models/qwen3_8_flash_next/qwen3_8_flash_next_nvfp4.edrl`: 104,913,957,888 bytes (97.71 GiB), 1,606 objects (1,557
tensors, 49 expert banks of 512 records), SHA-256 `c52ef26dee424671a1f852a762eca12ad6b35c357c3aed54c65a9a726a0f46ec`.

| | bytes |
|---|---|
| fixed text weights (without shared experts) | 5,532,502,392 |
| routed experts (text) | 68,048,388,096 |
| shared experts | 132,956,736 |
| n-gram | 28,800,156,160 |
| MTP | 1,501,306,404 |
| vision | 897,862,112 |

Layout counts: 1,118 contiguous BF16, 311 row-split NVFP4, 128 interleaved NVFP4 shards, 25,088 expert records. Conversion on
this machine (checkpoint on exFAT, artifact on ext4, 4 workers) takes about 75 seconds of writing; `verify --checkpoint`
confirmed all 104,913,171,900 payload bytes equal the checkpoint's, and `Validator.verifyChecksums` checks every CRC-32
in the Java reader.
