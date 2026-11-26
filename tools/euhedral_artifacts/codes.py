"""Pre-rounded codes: the stored rounding of a quantized object, saved one file per object, and packed into the
artifact's planes exactly as the converter's quantizers lay them out (the converter's --codes-from entry).
fake_quant.py exports these from trained modules."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from euhedral_artifacts.edrl import align_up, fail
from euhedral_artifacts.grouped import QUANT, pack_codes, row_split_size
from euhedral_artifacts.nvfp4 import NVFP4_BLOCK, SD4_TABLE, nvfp4_offsets, nvfp4_sd4_offsets
from euhedral_artifacts.recipes import LAYOUT_SD4, NVFP4_FORMAT


@dataclass
class QuantState:
    """The stored rounding of one quantized object of shape (N, K), K padded to K_pad = K rounded up to 128.

    Grouped formats: `codes` int8 [N, K_pad] signed codes, `scales` float16 [N, K_pad / 64].
    NVFP4: `codes` uint8 [N, K_pad] E2M1 codes (magnitude | sign << 3), `scales` uint8 [N, K_pad / 16] E4M3 codes,
    `global_scale` float32. NVFP4-SD4: `scales` holds uint8 indices into `table` (16 E4M3 codes) instead."""
    format_name: str
    layout: str
    shape: tuple[int, int]
    codes: np.ndarray
    scales: np.ndarray
    global_scale: np.float32 = np.float32(0)
    table: np.ndarray | None = None

    def save(self, path: Path) -> None:
        arrays = {"codes": self.codes, "scales": self.scales, "global_scale": np.float32(self.global_scale)}
        if self.table is not None:
            arrays["table"] = self.table
        np.savez(path, format_name=self.format_name, layout=self.layout, shape=np.asarray(self.shape, np.int64), **arrays)

    @staticmethod
    def load(path: Path) -> "QuantState":
        with np.load(path) as data:
            return QuantState(str(data["format_name"]), str(data["layout"]), tuple(int(v) for v in data["shape"]),
                              data["codes"], data["scales"], np.float32(data["global_scale"]),
                              data["table"] if "table" in data else None)


def state_file(directory: Path, name: str) -> Path:
    """The file of object `name` in a codes directory."""
    return directory / (name.replace("/", ".") + ".npz")


def state_size(state: QuantState) -> int:
    if state.layout == LAYOUT_SD4:
        return nvfp4_sd4_offsets(state.shape)[2]
    if state.format_name == NVFP4_FORMAT:
        return nvfp4_offsets(state.shape)[2]
    return row_split_size(state.shape, state.format_name)


def check_state(state: QuantState, format_name: str, layout: str, shape: tuple[int, ...]) -> None:
    """Fails unless `state` stores an object of `shape` as `format_name` / `layout`, with arrays of matching shapes."""
    if (state.format_name, state.layout, tuple(state.shape)) != (format_name, layout, tuple(shape)):
        fail(f"codes are {state.format_name}/{state.layout} {state.shape}, the recipe stores "
             f"{format_name}/{layout} {tuple(shape)}")
    n, k = shape
    k_pad = align_up(k, 128)
    per_scale = NVFP4_BLOCK if format_name == NVFP4_FORMAT else QUANT[format_name][1]
    if state.codes.shape != (n, k_pad) or state.scales.shape != (n, k_pad // per_scale):
        fail(f"codes {state.codes.shape} and scales {state.scales.shape} do not match shape {tuple(shape)}")
    if layout == LAYOUT_SD4 and (state.table is None or state.table.shape != (SD4_TABLE,)):
        fail("NVFP4-SD4 codes need a 16-entry scale table")


def write_state(output, base_offset: int, state: QuantState) -> None:
    """Writes `state` into its object's planes at `base_offset`, as the converter's quantizers lay them out."""
    n, k = state.shape
    k_pad = align_up(k, 128)
    if state.format_name == NVFP4_FORMAT:
        codes = state.codes.astype(np.uint8)
        output.seek(base_offset)
        output.write((codes[:, 0::2] | (codes[:, 1::2] << 4)).astype(np.uint8).tobytes())
        if state.layout == LAYOUT_SD4:
            index_offset, table_offset, _ = nvfp4_sd4_offsets(state.shape)
            indices = state.scales.astype(np.uint8)
            output.seek(base_offset + index_offset)
            output.write((indices[:, 0::2] | (indices[:, 1::2] << 4)).astype(np.uint8).tobytes())
            output.seek(base_offset + table_offset)
            output.write(state.table.astype(np.uint8).tobytes() + np.float32(state.global_scale).astype("<f4").tobytes())
        else:
            scale_offset, global_offset, _ = nvfp4_offsets(state.shape)
            output.seek(base_offset + scale_offset)
            output.write(state.scales.astype(np.uint8).tobytes())
            output.seek(base_offset + global_offset)
            output.write(np.float32(state.global_scale).astype("<f4").tobytes())
        return
    bits, group_size, _, _ = QUANT[state.format_name]
    groups = k_pad // group_size
    base, high = pack_codes(state.codes.reshape(-1, group_size), bits)
    high_offset = align_up(n * groups * (24 if bits == 3 else 32), 256)
    scale_offset = high_offset + align_up(n * groups * (8 if bits == 5 else 0), 256)
    output.seek(base_offset)
    output.write(base)
    if high:
        output.seek(base_offset + high_offset)
        output.write(high)
    output.seek(base_offset + scale_offset)
    output.write(np.ascontiguousarray(state.scales.astype("<f2")).tobytes())


MANIFEST_FILE = "calibration.json"
PRODUCER_FAKE_QUANT = "fake_quant"

# The manifest of a codes directory (MANIFEST_FILE) says which files are the producer's and whether they are whole.
# `quantization calibrate` writes {"key", "layers_done", "layers_total", "heads_done", "mtp_done", "files"}: the state
# files it saved (file names), and how far it got. A producer that rounds the objects it chooses (fake_quant's export)
# writes {"producer", "key", "objects", "complete"}: the run it exports (its key), the object names it saved, and
# whether the export is finished; an export into a directory of another key is refused, so two runs never mix. The
# converter takes a directory only when its manifest says it is whole and every file in it is listed, and refuses
# listed files that are missing and listed names that are not plain file names of the directory.

# Characters a listed file name may not hold: the separators of every platform, and Windows' stream and drive mark.
_NOT_IN_NAMES = ("/", "\\", ":", "\0")


def _read_manifest(directory: Path) -> dict:
    path = directory / MANIFEST_FILE
    if not path.is_file():
        return {}
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        fail(f"{path} is unreadable: {error}")
    if not isinstance(manifest, dict):
        fail(f"{path} is unreadable: not an object")
    return manifest


def _write_manifest(directory: Path, manifest: dict) -> None:
    path = directory / MANIFEST_FILE
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _plain(name) -> bool:
    """Whether `name` is a file name directly in the directory, on every platform."""
    return (isinstance(name, str) and name not in ("", ".", "..", MANIFEST_FILE)
            and not any(mark in name for mark in _NOT_IN_NAMES))


def export_key(*parts) -> str:
    """The key of an export: what names the run whose objects it holds (the source model, the artifact, the recipe,
    the importance matrix), joined as calibrate joins its key. The objects chosen are not part of it: passes over other
    objects of the same run add to one export."""
    return "|".join(str(part) for part in parts)


def _own_export(directory: Path, key: str, producer: str) -> dict:
    """The manifest of `directory` when it is `producer`'s export of `key` or there is none; fails otherwise."""
    manifest = _read_manifest(directory)
    if manifest and manifest.get("producer") != producer:
        fail(f"{directory} holds the {MANIFEST_FILE} of another producer ({manifest.get('producer', 'calibrate')}): "
             "export to another directory")
    if manifest and manifest.get("key") != key:
        fail(f"{directory} holds the export of another run (key {manifest.get('key')!r}, not {key!r}): export to "
             "another directory or empty this one")
    return manifest


def export_state(directory: Path, name: str, state: QuantState, key: str, producer: str = PRODUCER_FAKE_QUANT) -> None:
    """Saves `state` as the file of object `name` in `directory` and lists the object in the manifest as one of the
    export of `key` by `producer`, not yet complete: finish_export says the export is whole. Refuses, before it writes
    anything, a directory whose manifest is another producer's (calibrate's included) or another key's, so an export
    never overwrites another run's output or mixes with it."""
    if not _plain(state_file(directory, name).name):
        fail(f"object {name!r} has no plain file name")
    directory.mkdir(parents=True, exist_ok=True)
    manifest = _own_export(directory, key, producer)
    state.save(state_file(directory, name))
    objects = list(manifest.get("objects", []))
    if name not in objects:
        objects.append(name)
    _write_manifest(directory, {"producer": producer, "key": key, "objects": sorted(objects), "complete": False})


def finish_export(directory: Path, key: str, producer: str = PRODUCER_FAKE_QUANT) -> None:
    """Marks the export of `key` in `directory` complete: call it last, after every export_state."""
    manifest = _own_export(directory, key, producer)
    if not manifest:
        fail(f"{directory} has no {MANIFEST_FILE} of {producer}: nothing was exported")
    _write_manifest(directory, {**manifest, "complete": True})


def _strict(manifest: dict, key: str, kind: type, path: Path):
    value = manifest.get(key)
    if type(value) is not kind:
        fail(f"{path} is unreadable: {key} is {value!r}, not {kind.__name__}")
    return value


def require_complete_codes(directory: Path, layers: int | None = None) -> None:
    """Fails unless `directory` holds a manifest that says its codes are whole (see above): calibrate's with every
    layer (of the model's `layers`), both heads and the MTP block written, or a finished export. Every file the
    manifest lists must be there and every state file there must be listed: the converter rounds any object without a
    file itself, so a stopped run would otherwise be taken for a whole one."""
    path = directory / MANIFEST_FILE
    if not path.is_file():
        fail(f"{directory} has no {MANIFEST_FILE}: it is not the output of a finished calibrate run or of an export")
    manifest = _read_manifest(directory)
    if "producer" in manifest:
        if manifest["producer"] != PRODUCER_FAKE_QUANT:
            fail(f"{path} is from producer {manifest['producer']!r}, which is not a known producer")
        _strict(manifest, "key", str, path)
        if not _strict(manifest, "complete", bool, path):
            fail(f"{directory} is not complete: the export did not finish")
        objects = _strict(manifest, "objects", list, path)
        named = [name for name in objects if not isinstance(name, str) or not _plain(state_file(directory, name).name)]
        if named:
            fail(f"{path} is unreadable: it lists objects that are not plain names: {named!r}")
        listed = {state_file(directory, name).name for name in objects}
    else:
        layers_done = _strict(manifest, "layers_done", int, path)
        layers_total = _strict(manifest, "layers_total", int, path)
        heads_done = _strict(manifest, "heads_done", bool, path)
        mtp_done = _strict(manifest, "mtp_done", bool, path)
        files = _strict(manifest, "files", list, path)
        named = [name for name in files if not _plain(name)]
        if named:
            fail(f"{path} is unreadable: it lists names that are not file names of the directory: {named!r}")
        listed = set(files)
        if layers is not None and layers_total != layers:
            fail(f"{directory} is calibrate output of a model of {layers_total} layers, not {layers}")
        missing = []
        if layers_done < layers_total:
            missing.append(f"layers {layers_done} of {layers_total}")
        if not heads_done:
            missing.append("the heads")
        if not mtp_done:
            missing.append("the MTP block")
        if missing:
            fail(f"{directory} is not complete, calibrate wrote only: missing {', '.join(missing)}")
    absent = sorted(name for name in listed if not (directory / name).is_file())
    if absent:
        fail(f"{directory} lists files that are not there: {', '.join(absent)}")
    unlisted = sorted(entry.name for entry in directory.glob("*.npz") if entry.name not in listed)
    if unlisted:
        fail(f"{directory} holds state files its {MANIFEST_FILE} does not list: {', '.join(unlisted)}")
