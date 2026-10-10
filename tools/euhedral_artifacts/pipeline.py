"""Checkpoint directory -> artifact file."""

from __future__ import annotations

import contextlib
import json
import multiprocessing
import os
from pathlib import Path
import struct
import tempfile
from typing import Any

import numpy as np

from euhedral_artifacts import dflash2, q3_p2e2
from euhedral_artifacts.edrl import (FORMAT_ORDINAL, HEADER_SIZE, LAYOUT_ORDINAL, MAGIC, VERSION, ObjectPlan,
                                     assign_offsets, encode_table, fail, read_table, sha256)
from euhedral_artifacts.importance import ImportanceMatrix
from euhedral_artifacts.inventory import (VOCAB_SIZE, build_plans, draft_token_ids, encode_metadata, shortlist)
from euhedral_artifacts.recipes import Recipe
from euhedral_artifacts.sources import SourceStore, read_json

# Objects are independent and their offsets fixed, so worker processes quantize them in parallel and
# write their own regions; forked workers inherit the plans and the memory-mapped source.
_PARALLEL_PLANS: list[ObjectPlan] = []
_PARALLEL_OUTPUT = ""


def _write_object(index: int) -> int:
    plan = _PARALLEL_PLANS[index]
    with open(_PARALLEL_OUTPUT, "r+b") as output:
        plan.writer(output, plan.offset)
    return index


def write_objects(output_path: Path, output, plans: list[ObjectPlan], jobs: int) -> None:
    global _PARALLEL_PLANS, _PARALLEL_OUTPUT
    done = 0

    def report(index: int) -> None:
        nonlocal done
        done += 1
        if done == 1 or done == len(plans) or done % 32 == 0:
            print(f"converted {done}/{len(plans)} {plans[index].name}", flush=True)

    if jobs <= 1:
        for index, plan in enumerate(plans):
            plan.writer(output, plan.offset)
            report(index)
        return
    output.flush()
    _PARALLEL_PLANS, _PARALLEL_OUTPUT = plans, str(output_path)
    order = sorted(range(len(plans)), key=lambda i: -plans[i].byte_size)
    with multiprocessing.get_context("fork").Pool(jobs) as pool:
        for index in pool.imap_unordered(_write_object, order):
            report(index)


def check_checkpoint(config: dict[str, Any]) -> None:
    text = config.get("text_config")
    if config.get("architectures") != ["Qwen3_5ForConditionalGeneration"] or not isinstance(text, dict):
        fail("source is not the supported Qwen3_5ForConditionalGeneration checkpoint")
    if text.get("num_hidden_layers") != 64 or text.get("vocab_size") != VOCAB_SIZE:
        fail("source topology does not match the supported Qwen3.8-27B checkpoint")


def write_artifact(model: Path, config: dict[str, Any], output_path: Path, recipe: Recipe, selected: np.ndarray,
                   jobs: int, draft: Path | None = None, draft_projections: str = "bf16",
                   importance: ImportanceMatrix | None = None) -> None:
    """Quantizes the checkpoint into the artifact `recipe` names (stored as is: for compressed q3, the
    uncompressed form), at `output_path`, through a temporary file renamed into place. With `draft` the
    DFlash2 drafter checkpoint in that directory is added as `dflash2/` objects; with `importance` the
    weights are rounded by calibrated scale search."""
    with SourceStore(model) as store, contextlib.ExitStack() as drafts:
        plans = build_plans(store, selected, recipe, importance)
        if draft is not None:
            draft_store = drafts.enter_context(SourceStore(draft))
            plans += dflash2.build_plans(draft_store, dflash2.check_draft(read_json(draft / "config.json")),
                                         draft_projections)
        metadata = encode_metadata(config)
        table_size = len(encode_table(plans))
        data_base = HEADER_SIZE + len(metadata) + table_size
        file_size = assign_offsets(plans, data_base)
        table = encode_table(plans)
        header = struct.pack(">iiqqqiiq", MAGIC, VERSION, HEADER_SIZE, len(metadata), HEADER_SIZE + len(metadata),
                             len(plans), 0, data_base)
        if len(header) != HEADER_SIZE:
            fail("internal EDRL header size mismatch")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(prefix=f".{output_path.name}.", suffix=".partial", dir=output_path.parent)
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            with temporary.open("w+b") as output:
                output.truncate(file_size)
                output.seek(0)
                output.write(header)
                output.write(metadata)
                output.write(table)
                write_objects(temporary, output, plans, jobs)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, output_path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise


def convert(model: Path, output_path: Path, recipe: Recipe, ranking_path: Path | None = None,
            draft_ids_from: Path | None = None, jobs: int = 1, force: bool = False, draft: Path | None = None,
            draft_projections: str = "bf16", imatrix: Path | None = None) -> dict[str, Any]:
    """Converts the checkpoint directory `model` to the artifact `recipe` names. Exactly one of
    `ranking_path` (token frequency counts, to choose the draft-head shortlist) and `draft_ids_from`
    (an existing artifact whose shortlist is reused) is required. `draft` adds the DFlash2 drafter.
    `imatrix` (a llama.cpp importance matrix) selects calibrated rounding."""
    if output_path.exists() and not force:
        fail(f"output already exists; pass --force: {output_path}")
    if (ranking_path is None) == (draft_ids_from is None):
        fail("pass exactly one of --ranking and --draft-ids-from")
    config = read_json(model / "config.json")
    check_checkpoint(config)
    selected = shortlist(model, ranking_path) if ranking_path is not None else draft_token_ids(draft_ids_from)
    importance = ImportanceMatrix(imatrix) if imatrix is not None else None

    # Compressed q3 is the q3 artifact transcoded: the q3 form is built beside the output and removed.
    transcode = recipe.quantization == "q3" and recipe.compressed
    if transcode:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fd, compact_name = tempfile.mkstemp(prefix=f".{output_path.name}.", suffix=".q3", dir=output_path.parent)
        os.close(fd)
        built = Path(compact_name)
    else:
        built = output_path
    try:
        write_artifact(model, config, built, recipe, selected, jobs, draft, draft_projections, importance)
        stats = q3_p2e2.transcode(built, output_path, force=True) if transcode else {}
    finally:
        if transcode:
            built.unlink(missing_ok=True)

    manifest = describe(output_path)
    manifest.update({"artifact": recipe.name, "source_model": str(model)})
    if draft is not None:
        manifest.update({"draft_model": str(draft), "draft_projections": draft_projections})
    if imatrix is not None:
        manifest.update({"imatrix": str(imatrix), "imatrix_sha256": sha256(imatrix)})
    manifest.update(stats)
    with Path(f"{output_path}.manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest


def extend(artifact: Path, draft: Path, output_path: Path, draft_projections: str = "bf16",
           force: bool = False) -> dict[str, Any]:
    """Adds the DFlash2 drafter in `draft` to the existing artifact `artifact`, whose objects are copied byte
    for byte, as `output_path`."""
    dflash2.extend(artifact, draft, output_path, draft_projections, force)
    manifest = describe(output_path)
    manifest.update({"extends": str(artifact), "draft_model": str(draft), "draft_projections": draft_projections})
    with Path(f"{output_path}.manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest


def describe(path: Path) -> dict[str, Any]:
    """Object, format and layout counts, payload bytes and SHA-256 of the artifact file `path`."""
    with path.open("rb") as handle:
        _, _, objects = read_table(handle)
    formats = {ordinal: name for name, ordinal in FORMAT_ORDINAL.items()}
    layouts = {ordinal: name for name, ordinal in LAYOUT_ORDINAL.items()}
    format_counts: dict[str, int] = {}
    layout_counts: dict[str, int] = {}
    for obj in objects:
        format_counts[formats[obj["format"]]] = format_counts.get(formats[obj["format"]], 0) + 1
        layout_counts[layouts[obj["layout"]]] = layout_counts.get(layouts[obj["layout"]], 0) + 1
    return {
        "object_count": len(objects),
        "payload_bytes": sum(obj["bytes"] for obj in objects),
        "file_bytes": path.stat().st_size,
        "sha256": sha256(path),
        "format_counts": format_counts,
        "layout_counts": layout_counts,
    }
