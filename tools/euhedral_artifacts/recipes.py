"""The four artifacts the engine ships, and what each stores per object.

    q3                Q3G64/Q4G64/Q5G64 text weights (grouped, row-split); NVFP4 LM head and MTP layer; Q3 draft head
    bf16              every object as the checkpoint's BF16 values (contiguous): the unquantized model
    q3 compressed     the q3 artifact with its Q3 tensors but the MLP down projections transcoded losslessly to P2E2
    nvfp4             NVFP4 text weights, MTP layer and draft head; Q3 token embedding
    nvfp4 compressed  NVFP4-SD4 text weights (4.25 bits per weight); plain NVFP4 MTP layer and draft head

No artifact holds the vision tower.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from euhedral_artifacts.edrl import fail

QUANTIZATIONS = ("q3", "nvfp4", "bf16")

LAYOUT_ROW_SPLIT = "row-split-k128-v1"
LAYOUT_SD4 = "row-split-k128-sd4-v1"
TOKEN_EMBEDDING = "text/token_embedding"
Q3_FORMAT = "Q3G64_F16S"
NVFP4_FORMAT = "NVFP4"
BF16_FORMAT = "BF16"
LAYOUT_CONTIGUOUS = "contiguous-le-v1"


@dataclass(frozen=True)
class Recipe:
    quantization: str
    compressed: bool = False

    def __post_init__(self) -> None:
        if self.quantization not in QUANTIZATIONS:
            fail(f"unknown quantization {self.quantization!r}; expected one of {', '.join(QUANTIZATIONS)}")
        if self.quantization == "bf16" and self.compressed:
            fail("the bf16 recipe has no compressed form")

    @property
    def name(self) -> str:
        return self.quantization + ("-compressed" if self.compressed else "")


# The four shipped artifacts; bf16 is the unquantized model, built on request.
RECIPES = tuple(Recipe(quantization, compressed) for quantization in ("q3", "nvfp4") for compressed in (False, True))


def stays_row_split(name: str) -> bool:
    """Q3 objects compressed q3 keeps in row-split Q3 with the full code range instead of P2E2 (which holds -3..3): the
    MLP down projections, where the restriction costs the most KL per byte it saves (docs/COMPRESSED_Q3.md)."""
    return re.fullmatch(r"text/layers/\d+/mlp/down", name) is not None


def is_drafting(name: str) -> bool:
    """The MTP layer and the draft head: they propose tokens the base model verifies."""
    return name.startswith("mtp/") or name == "text/draft_head"


def storage(recipe: Recipe, name: str, q3_format: str) -> tuple[str, str]:
    """(format, layout) of the quantized object `name`. `q3_format` is its format in the q3 artifact."""
    if recipe.quantization == "bf16":
        return BF16_FORMAT, LAYOUT_CONTIGUOUS
    if recipe.quantization == "q3" or name == TOKEN_EMBEDDING:
        return q3_format, LAYOUT_ROW_SPLIT
    if recipe.compressed and not is_drafting(name):
        return NVFP4_FORMAT, LAYOUT_SD4
    return NVFP4_FORMAT, LAYOUT_ROW_SPLIT
