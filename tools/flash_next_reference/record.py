"""Fixture writer: manifest.json + raw little-endian tensor files (the format of tools/dflash2_reference.py).

Names are relative to the case directory; every chunk or step is the prefix `c{k}/`. A tensor whose bytes equal an
earlier tensor's is written once and listed under both names (the manifest names the file explicitly).
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
import re
import shutil

import numpy as np
import torch

DTYPES = {torch.bfloat16: "bf16", torch.float32: "f32", torch.int32: "i32", torch.int64: "i64", torch.uint8: "u8"}
_DIR_PART = re.compile(r"^(c\d+|s\d+|oneshot)$")


def tensor_to_bytes(tensor: torch.Tensor) -> tuple[str, bytes]:
    tensor = tensor.detach().to("cpu").contiguous()
    if tensor.dtype == torch.bool:
        tensor = tensor.to(torch.uint8)
    if tensor.dtype not in DTYPES:
        raise TypeError(f"unsupported fixture dtype {tensor.dtype}")
    raw = tensor.view(torch.int16) if tensor.dtype == torch.bfloat16 else tensor
    return DTYPES[tensor.dtype], raw.numpy().tobytes()


class Recorder:
    def __init__(self, directory: Path, metadata: dict | None = None):
        self.directory = Path(directory)
        if self.directory.exists():
            shutil.rmtree(self.directory)
        self.directory.mkdir(parents=True)
        self.metadata: dict = dict(metadata or {})
        self.manifest: dict[str, dict] = {}
        self.prefix = ""
        self.detail = True
        self.light: set[str] | None = None  # names (after the prefix) kept when detail is off
        self.skip: set[str] = set()  # names (after the prefix) never written in this chunk
        self.muted = 0
        self.bytes = 0
        self._digests: dict[tuple, str] = {}

    # --- context -------------------------------------------------------------------------------------------

    def begin(self, prefix: str, detail: bool = True) -> None:
        self.prefix = prefix.rstrip("/") + "/" if prefix else ""
        self.detail = detail
        self.skip = set()

    @contextlib.contextmanager
    def mute(self):
        """Recording is off inside (the reference re-run of a recording module)."""
        self.muted += 1
        try:
            yield
        finally:
            self.muted -= 1

    def wants(self, name: str) -> bool:
        if self.muted or name in self.skip:
            return False
        return self.detail or (self.light is not None and name in self.light)

    # --- writing ---------------------------------------------------------------------------------------------

    def _file(self, full: str) -> Path:
        parts = full.split("/")
        dirs = []
        while len(parts) > 1 and _DIR_PART.match(parts[0]):
            dirs.append(parts.pop(0))
        return Path(*dirs, ".".join(parts) + ".bin")

    def add(self, name: str, tensor: torch.Tensor, force: bool = False) -> bool:
        if self.muted or not (force or self.wants(name)):
            return False
        full = self.prefix + name
        if full in self.manifest:
            raise RuntimeError(f"tensor {full} recorded twice")
        dtype, raw = tensor_to_bytes(tensor)
        key = (dtype, tuple(tensor.shape), hash(raw), len(raw))
        shared = self._digests.get(key)
        if shared is not None and (self.directory / shared).read_bytes() == raw:
            file = shared
        else:
            file = str(self._file(full))
            path = self.directory / file
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
            self._digests[key] = file
            self.bytes += len(raw)
        self.manifest[full] = {"dtype": dtype, "shape": list(tensor.shape), "file": file}
        return True

    def has(self, name: str) -> bool:
        return (self.prefix + name) in self.manifest

    def alias(self, name: str, other: str) -> None:
        """Lists `name` as the same tensor (file) as the already recorded `other` (names relative to the prefix)."""
        if not self.wants(name):
            return
        full, source = self.prefix + name, self.prefix + other
        if full in self.manifest:
            raise RuntimeError(f"tensor {full} recorded twice")
        self.manifest[full] = dict(self.manifest[source])

    def finish(self, extra: dict | None = None) -> int:
        metadata = dict(self.metadata)
        if extra:
            metadata.update(extra)
        unique = {entry["file"] for entry in self.manifest.values()}
        total = sum((self.directory / f).stat().st_size for f in unique)
        metadata["tensor_count"] = len(self.manifest)
        metadata["file_count"] = len(unique)
        metadata["total_bytes"] = total
        (self.directory / "manifest.json").write_text(
            json.dumps({"metadata": metadata, "tensors": self.manifest}, indent=1) + "\n")
        return total


def read_manifest(directory: Path) -> dict:
    return json.loads((Path(directory) / "manifest.json").read_text())


_NUMPY = {"bf16": None, "f32": np.float32, "i32": np.int32, "i64": np.int64, "u8": np.uint8}


def load_tensor(directory: Path, manifest: dict, name: str) -> torch.Tensor:
    entry = manifest["tensors"][name]
    raw = (Path(directory) / entry["file"]).read_bytes()
    shape = entry["shape"]
    if entry["dtype"] == "bf16":
        return torch.frombuffer(bytearray(raw), dtype=torch.int16).view(torch.bfloat16).reshape(shape)
    return torch.from_numpy(np.frombuffer(raw, dtype=_NUMPY[entry["dtype"]]).copy()).reshape(shape)
