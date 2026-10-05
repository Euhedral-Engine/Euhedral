"""Reading Euhedral's EDRL v3 artifact (`docs/FLASH_NEXT_ARTIFACT.md`) for the reference harness."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import struct
import zlib

import numpy as np
import torch

from euhedral_artifacts.qwen4_edrl import read_artifact
from euhedral_artifacts.nvfp4 import nvfp4_offsets

from .nvfp4 import PackedNvfp4, expand_nvfp4

DEFAULT_ARTIFACT = Path("/home/brandon/models/qwen3_8_flash_next/qwen3_8_flash_next_nvfp4.edrl")
DEFAULT_HF_CONFIG = Path("/mnt/shared/qwen38-flash-next/nvfp4/config.json")
DEFAULT_TOKENIZER = Path("/mnt/shared/qwen38-flash-next/nvfp4/tokenizer.json")
FORMAT_BF16, FORMAT_NVFP4 = 0, 7
LAYOUT_ROW_SPLIT, LAYOUT_NGRAM = 1, 4
NGRAM_ROW_VALUES = 160
NGRAM_ROW_BYTES = NGRAM_ROW_VALUES // 2 + NGRAM_ROW_VALUES // 16  # 90


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


class ExpertRecord:
    """One routed expert (gate_up [1280, 2560] and down [2560, 640]) kept packed on the GPU."""

    def __init__(self, gate_up: PackedNvfp4, down: PackedNvfp4):
        self.gate_up, self.down = gate_up, down

    @property
    def nbytes(self) -> int:
        return self.gate_up.nbytes + self.down.nbytes

    def expand(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.gate_up.expand(), self.down.expand()


class Artifact:
    def __init__(self, path: Path | str = DEFAULT_ARTIFACT, expert_cache_bytes: int = 768 << 20):
        self.path = Path(path)
        parsed = read_artifact(self.path)
        self.header = parsed["header"]
        self.metadata: dict = parsed["metadata"]
        self.tensors = {t["name"]: t for t in parsed["tensors"]}
        self.banks = {b["layer"]: b for b in parsed["banks"] if b["name"].startswith("text/")}
        self._map = np.memmap(self.path, dtype=np.uint8, mode="r")
        self._cache: OrderedDict[tuple[int, int], ExpertRecord] = OrderedDict()
        self._cache_bytes = 0
        self.expert_cache_limit = expert_cache_bytes
        self.expert_reads = 0
        self.expert_hits = 0

    # --- raw access ------------------------------------------------------------------------------------------

    def view(self, offset: int, size: int) -> np.ndarray:
        return self._map[offset: offset + size]

    def crc32(self, offset: int, size: int, chunk: int = 1 << 26) -> int:
        value, position = 0, 0
        while position < size:
            piece = self._map[offset + position: offset + min(size, position + chunk)]
            value = zlib.crc32(piece, value)
            position += len(piece)
        return value

    # --- fixed tensors ---------------------------------------------------------------------------------------

    def entry(self, name: str) -> dict:
        try:
            return self.tensors[name]
        except KeyError:
            raise KeyError(f"artifact has no object {name}") from None

    def has(self, name: str) -> bool:
        return name in self.tensors

    def bf16(self, name: str, device="cpu") -> torch.Tensor:
        entry = self.entry(name)
        if entry["format"] != FORMAT_BF16:
            raise ValueError(f"{name} is not stored as BF16")
        data = torch.from_numpy(np.array(self.view(entry["offset"], entry["bytes"])))
        return data.view(torch.bfloat16).reshape(entry["shape"]).to(device)

    def bf16_rows(self, name: str, rows: np.ndarray, device="cpu") -> torch.Tensor:
        """Some rows of a BF16 [rows, K] tensor."""
        entry = self.entry(name)
        row_bytes = int(entry["shape"][1]) * 2
        starts = entry["offset"] + rows.astype(np.int64) * row_bytes
        data = np.array(self._map[starts[:, None] + np.arange(row_bytes)[None, :]])
        return torch.from_numpy(data).view(torch.bfloat16).reshape(len(rows), -1).to(device)

    def nvfp4(self, name: str, device) -> PackedNvfp4:
        entry = self.entry(name)
        if entry["format"] != FORMAT_NVFP4 or entry["layout"] != LAYOUT_ROW_SPLIT:
            raise ValueError(f"{name} is not a row-split NVFP4 tensor")
        return PackedNvfp4.parse(self.view(entry["offset"], entry["bytes"]), tuple(entry["shape"]), device)

    def verify_tensor(self, name: str) -> None:
        entry = self.entry(name)
        value = self.crc32(entry["offset"], entry["bytes"])
        if value != entry["crc"]:
            raise RuntimeError(f"{name}: CRC-32 {value:08x} != table {entry['crc']:08x}")

    # --- expert banks ----------------------------------------------------------------------------------------

    def bank(self, layer: int) -> dict:
        return self.banks[layer]

    def expert_bytes(self, layer: int, expert: int) -> np.ndarray:
        offset, size, _ = self.banks[layer]["index"][expert]
        return self.view(offset, size)

    def verify_expert(self, layer: int, expert: int) -> None:
        offset, size, crc = self.banks[layer]["index"][expert]
        value = self.crc32(offset, size)
        if value != crc:
            raise RuntimeError(f"layer {layer} expert {expert}: CRC-32 {value:08x} != table {crc:08x}")

    def expert(self, layer: int, expert: int, device) -> ExpertRecord:
        """The packed record of an expert on `device`, through a small LRU of records."""
        key = (layer, expert)
        record = self._cache.get(key)
        if record is not None:
            self._cache.move_to_end(key)
            self.expert_hits += 1
            return record
        self.expert_reads += 1
        bank = self.banks[layer]
        raw = self.expert_bytes(layer, expert)
        parts = {}
        for projection in bank["projections"]:
            _, _, size = nvfp4_offsets(tuple(projection["shape"]))
            parts[projection["name"]] = PackedNvfp4.parse(
                raw[projection["offset"]: projection["offset"] + size], tuple(projection["shape"]), device)
        record = ExpertRecord(parts["gate_up"], parts["down"])
        self._cache[key] = record
        self._cache_bytes += record.nbytes
        while self._cache_bytes > self.expert_cache_limit and len(self._cache) > 1:
            _, evicted = self._cache.popitem(last=False)
            self._cache_bytes -= evicted.nbytes
        return record

    # --- n-gram shards ---------------------------------------------------------------------------------------

    def ngram_shards(self, layer: int) -> list[dict]:
        prefix = f"text/layers/{layer}/ple/ngram/shard_"
        shards = sorted((n for n in self.tensors if n.startswith(prefix)), key=lambda n: int(n[len(prefix):]))
        return [self.tensors[n] for n in shards]

    def ngram_rows(self, layer: int, rows: np.ndarray, device) -> torch.Tensor:
        """Global n-gram table rows `rows` (int64 array) dequantized to BF16 [n, 160]: e2m1 * e4m3 * global in fp32."""
        shards = self.ngram_shards(layer)
        shard_rows = self.metadata["ngram.shard_rows"]
        shard = rows // shard_rows
        local = rows % shard_rows
        if rows.size and (rows.min() < 0 or shard.max() >= len(shards)):
            raise IndexError("n-gram row out of range")
        base = np.array([s["offset"] for s in shards], dtype=np.int64)[shard]
        starts = base + local * NGRAM_ROW_BYTES
        data = np.asarray(self._map[starts[:, None] + np.arange(NGRAM_ROW_BYTES)[None, :]])
        trailer = np.array([s["offset"] + align_up(s["shape"][0] * NGRAM_ROW_BYTES, 256) for s in shards],
                           dtype=np.int64)[shard]
        scale_bytes = np.asarray(self._map[trailer[:, None] + np.arange(4)[None, :]])
        global_scale = np.ascontiguousarray(scale_bytes).view("<f4").reshape(-1)
        codes = torch.from_numpy(np.array(data[:, :80])).to(device)
        scales = torch.from_numpy(np.array(data[:, 80:])).to(device)
        g = torch.from_numpy(global_scale.copy()).to(device)
        return expand_nvfp4(codes, scales, g).to(torch.bfloat16)

    def close(self) -> None:
        self._map = None
