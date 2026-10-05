# Lossless NVFP4 checkpoints

`tools/nvfp4z.py` stores a checkpoint of NVFP4 safetensors shards in about 90% of its size and rebuilds every shard
byte for byte. It is an archive and transfer format for the checkpoint files, not an inference artifact: the engine
reads the artifacts of [tools/README.md](../tools/README.md).

```bash
python3 tools/nvfp4z.py compress   CHECKPOINT_DIR  COMPRESSED_DIR   # one .safetensors.nvfp4z per shard, other files copied
python3 tools/nvfp4z.py decompress COMPRESSED_DIR  CHECKPOINT_DIR   # exact shards, each checked against its SHA-256
python3 tools/nvfp4z.py info       COMPRESSED_DIR                   # sizes and tensor kinds per file
```

It needs PyTorch with CUDA (the kernels are compiled with NVRTC on first use and cached in `~/.cache/nvfp4z`) and NumPy.
`--help` lists the options: reader, writer and GPU thread counts, the page-locked memory budget (default a quarter of
RAM, at most 16 GiB; one buffer holds one whole shard), `--no-verify`, `--no-fsync`, `--direct-io`, `--force`.
Existing outputs are skipped, so an interrupted run resumes.

## What is coded

| Tensor | Treatment |
|---|---|
| NVFP4 codes (`U8`, last dimension K/2) with an `F8_E4M3` scale tensor (last dimension K/16) | entropy coded together; found as `X_packed` + `X_scale` or `X` + `X_scale` |
| `BF16`, at least 65536 elements | the high byte (sign and 7 exponent bits) entropy coded, the low byte stored |
| everything else (`F32` global scales, small tensors, other dtypes) | stored |

A tensor whose scale tensor does not have the NVFP4 shape, or that has no partner, is stored. Segments that
would not shrink, or whose scale bytes are not valid positive E4M3 values (above 127), are stored too, so a
container is never larger than its source plus about 4 KiB.

## Model

A block is 16 weights: 8 bytes of E2M1 codes and one E4M3 scale byte. Probabilities are static, counted per segment
and stored in the container.

- **Scale byte:** one table per group, a group being one expert of a rank-3 tensor (one table for rank 2). The scale
  of a block is about 3.6 bits of noise in the block maximum; the per-group table captures the group's global scale
  and its spread.
- **Magnitude (3 bits):** the context is the scale bucket (scale >> 3, 4 or 7 by segment size), whether a magnitude-7
  code (6.0) has already occurred in the block, and the position in the block. Every block has a 7-code, so the last
  positions are often determined.
- **Sign:** coded only for nonzero magnitudes, as an exact 1-bit symbol. A segment that contains negative zeros codes
  every sign (so they survive).

Measured entropy of the Qwen3.8-Flash-Next NVFP4 weights, bits per weight (raw 4.5): E2M1 codes 3.86 to 3.87, scales 0.22,
total 4.09 for the routed experts and 4.08 for the n-gram embeddings. Tried and not kept: contexts from the left or
upper neighbouring scale (no gain once the scale table is per expert), a sign context from the previous sign (no gain),
magnitude tables per expert (the table cost exceeds the gain).

## Stream

Segments of up to 2^27 blocks are coded independently by up to 65536 lanes; lane l codes blocks l, l+L, l+2L, ...
(the last block repeats to fill the final row). Each lane is a rANS stream with a 64-bit state in [2^32, 2^48), 16
probability bits and 16-bit renormalisation words, stored as [words in emission order][state low, middle, high] and
read from the back by the decoder. The encoder runs backwards, so decoding runs in block order, one thread per lane.
Lane streams are stored one after the other; their lengths (u16 or u32) are stored per segment. Per-lane lengths and
states cost about 0.1% of a segment.

`kernels.cu` has the kernels: histogram (with the largest scale byte and a negative-zero flag), encode, decode, and a
copy that concatenates the lanes. The encoder divides in double precision (exact for a 48-bit state and a 16-bit
frequency). Every encoded segment is decoded on the GPU and compared before it is stored, unless `--no-verify`.

## File format

```
"NVFP4Z\0\1" | u64 header length | JSON header, space padded so the block is a multiple of 4096 bytes | payload
```

The payload offsets in the header are relative to the payload start. The header holds the source safetensors header
verbatim, the source size and SHA-256, and per tensor its kind and pieces (`[offset, count, "u8"|"u16"|"u32"]`). Decompression
rebuilds the exact file and compares its SHA-256 with the recorded one; a mismatch deletes the output and fails.
Shards whose tensors do not tile the data region exactly are rejected at compression.

## Pipeline

Reader threads fill page-locked buffers (hashing as they read), GPU threads (one stream each) compress or decompress
whole shards, writer threads write the outputs through a temporary file, `fsync` and rename. Only raw tensors are
copied on the host. Direct I/O (`--direct-io auto`) is used on exFAT, where it writes faster than the page cache; on
ext4 it was slower (1.6 GB/s against 4.9 GB/s with four writers), so it stays off there.

## Measured (RTX 5070 Ti, i9-14900K, 2 NVMe drives)

Qwen3.8-Flash-Next NVFP4, 131 shards, 104.81 GB, compressed to 94.25 GB.

| | |
|---|---|
| GPU encode of one expert tensor (0.94 GB, 512 experts), counting, coding and the copy to pinned memory | 130 ms (7.3 GB/s) |
| GPU decode of the same tensor, with table build and upload | 66 ms (14.2 GB/s) |
| `compress`, whole checkpoint, read and written on one exFAT NVMe (cold) | 105 s, limited by 0.9 GB/s sustained writes |
| `compress`, 48 shards (36.5 GB) read from exFAT, written to an ext4 NVMe | 18.9 s |
| `decompress`, whole checkpoint to exFAT, each shard SHA-256 checked (six batches) | 110 s |
| Raw read throughput of the exFAT drive, cold | 2.9 GB/s |

Each of these is a single run on one machine; the storage dominates.
