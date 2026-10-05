#!/usr/bin/env python3
"""Inventory of the Qwen3.8-Flash-Next NVFP4 checkpoint, read from its safetensors headers.

Every tensor is classified into a logical group (fixed text weights, routed experts, shared experts, routers, GDN, QSA
indexer, hyper-connections, token embedding, output head, norms and small state, n-gram embeddings, MTP, vision) and
each group reports its tensor count, parameter count, stored bytes, dtypes, shape families and per-layer bytes.
Nothing here is estimated from documentation: sizes are the checkpoint's own, and the artifact bytes are those of the
runtime objects the converter would write.

    python3 tools/flash_next_inventory.py CHECKPOINT_DIR [--json OUT.json] [--markdown OUT.md]
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from euhedral_artifacts.qwen4_source import GROUPS, SourceCheckpoint, build_plan  # noqa: E402


def inventory(directory: Path) -> dict:
    source = SourceCheckpoint(directory)
    plan = build_plan(source)
    groups: dict[str, dict] = {name: {"tensors": 0, "params": 0, "stored_bytes": 0, "artifact_bytes": 0,
                                       "formats": set(), "shapes": defaultdict(int), "layers": defaultdict(int),
                                       "source_tensors": 0} for name in GROUPS}

    def stored(parts: dict[str, str]) -> int:
        return sum(source.tensors[name].size for name in parts.values())

    def layer_of(name: str) -> int | None:
        pieces = name.split("/")
        for index, piece in enumerate(pieces):
            if piece == "layers" and index + 1 < len(pieces) and pieces[index + 1].isdigit():
                return int(pieces[index + 1])
        return None

    for tensor in plan.tensors:
        group = groups[tensor.group]
        elements = 1
        for dimension in tensor.shape:
            elements *= dimension
        group["tensors"] += 1
        group["params"] += elements
        group["stored_bytes"] += stored(tensor.parts)
        group["artifact_bytes"] += tensor.size
        group["source_tensors"] += len(tensor.parts)
        group["formats"].add(f"{tensor.fmt}/{tensor.layout}")
        group["shapes"][f"{tensor.kind} {list(tensor.shape)}"] += 1
        layer = layer_of(tensor.name)
        if layer is not None:
            group["layers"][layer] += tensor.size
    expert_info = []
    for bank in plan.banks:
        group = groups[bank.group]
        params = 0
        stored_bytes = 0
        for projection in bank.projections:
            params += projection.shape[0] * projection.shape[1] * bank.expert_count
            stored_bytes += stored(bank.parts[projection.name])
            group["shapes"][f"expert {projection.name} {list(projection.shape)} x{bank.expert_count}"] += 1
        group["tensors"] += 1
        group["params"] += params
        group["stored_bytes"] += stored_bytes
        group["artifact_bytes"] += bank.record_bytes * bank.expert_count
        group["source_tensors"] += 3 * len(bank.projections)
        group["formats"].add("NVFP4/expert-record")
        group["layers"][bank.layer] += bank.record_bytes * bank.expert_count
        expert_info.append({
            "bank": bank.name, "layer": bank.layer, "experts": bank.expert_count,
            "projections": {p.name: {"shape": list(p.shape), "bytes": p.size, "offset": p.offset}
                            for p in bank.projections},
            "record_bytes": bank.record_bytes,
            "stored_bytes_per_expert": stored_bytes // bank.expert_count,
        })
    result = {"source": str(directory), "groups": {}, "banks": expert_info, "metadata": plan.metadata,
              "source_tensor_count": len(source.tensors)}
    for name, group in groups.items():
        layers = group["layers"]
        result["groups"][name] = {
            "tensors": group["tensors"],
            "source_tensors": group["source_tensors"],
            "params": group["params"],
            "stored_bytes": group["stored_bytes"],
            "artifact_bytes": group["artifact_bytes"],
            "formats": sorted(group["formats"]),
            "shape_families": dict(sorted(group["shapes"].items(), key=lambda item: -item[1])),
            "layers_present": len(layers),
            "bytes_per_layer": (min(layers.values()), max(layers.values()),
                                sum(layers.values()) // len(layers)) if layers else None,
        }
    source.close()
    return result


def markdown(result: dict) -> str:
    gib = 1 << 30
    lines = ["| group | tensors | params | stored bytes | artifact bytes | layers | bytes per layer (min/max) | "
             "formats |", "|---|---|---|---|---|---|---|---|"]
    totals = [0, 0, 0, 0]
    for name in GROUPS:
        group = result["groups"][name]
        if group["tensors"] == 0:
            continue
        per_layer = group["bytes_per_layer"]
        lines.append("| {} | {} | {:,} | {:,} ({:.3f} GiB) | {:,} | {} | {} | {} |".format(
            name, group["tensors"], group["params"], group["stored_bytes"], group["stored_bytes"] / gib,
            group["artifact_bytes"], group["layers_present"],
            f"{per_layer[0]:,} / {per_layer[1]:,}" if per_layer else "", ", ".join(group["formats"])))
        totals[0] += group["tensors"]
        totals[1] += group["params"]
        totals[2] += group["stored_bytes"]
        totals[3] += group["artifact_bytes"]
    lines.append(f"| **total** | {totals[0]} | {totals[1]:,} | {totals[2]:,} ({totals[2] / gib:.3f} GiB) | "
                 f"{totals[3]:,} | | | |")
    lines.append("")
    for name in GROUPS:
        group = result["groups"][name]
        if group["tensors"] == 0:
            continue
        lines.append(f"**{name}** shape families:")
        for family, count in list(group["shape_families"].items())[:12]:
            lines.append(f"- {family} x{count}")
        if len(group["shape_families"]) > 12:
            lines.append(f"- ... {len(group['shape_families']) - 12} more")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    result = inventory(args.checkpoint)
    text = markdown(result)
    if args.json:
        args.json.write_text(json.dumps(result, indent=1, default=list))
    if args.markdown:
        args.markdown.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
