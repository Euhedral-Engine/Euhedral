"""The four artifacts the engine ships, and what each stores per object.

    q3                Q3G64/Q4G64/Q5G64 text weights (grouped, row-split); NVFP4 MTP layer; Q3 draft head
    q3 compressed     the q3 artifact with its Q3 tensors transcoded losslessly to P2E2
    nvfp4             NVFP4 text weights, MTP layer and draft head; Q3 token embedding
    nvfp4 compressed  NVFP4-SD4 text weights (4.25 bits per weight); plain NVFP4 MTP layer and draft head

No artifact holds the vision tower.
"""

from __future__ import annotations

from dataclasses import dataclass

from euhedral_artifacts.edrl import fail

QUANTIZATIONS = ("q3", "nvfp4")

LAYOUT_ROW_SPLIT = "row-split-k128-v1"
LAYOUT_SD4 = "row-split-k128-sd4-v1"
TOKEN_EMBEDDING = "text/token_embedding"
Q3_FORMAT = "Q3G64_F16S"
NVFP4_FORMAT = "NVFP4"


@dataclass(frozen=True)
class Recipe:
    quantization: str
    compressed: bool = False

    def __post_init__(self) -> None:
        if self.quantization not in QUANTIZATIONS:
            fail(f"unknown quantization {self.quantization!r}; expected one of {', '.join(QUANTIZATIONS)}")

    @property
    def name(self) -> str:
        return self.quantization + ("-compressed" if self.compressed else "")


RECIPES = tuple(Recipe(quantization, compressed) for quantization in QUANTIZATIONS for compressed in (False, True))


def is_drafting(name: str) -> bool:
    """The MTP layer and the draft head: they propose tokens the base model verifies."""
    return name.startswith("mtp/") or name == "text/draft_head"


def storage(recipe: Recipe, name: str, q3_format: str) -> tuple[str, str]:
    """(format, layout) of the quantized object `name`. `q3_format` is its format in the q3 artifact."""
    if recipe.quantization == "q3" or name == TOKEN_EMBEDDING:
        return q3_format, LAYOUT_ROW_SPLIT
    if recipe.compressed and not is_drafting(name):
        return NVFP4_FORMAT, LAYOUT_SD4
    return NVFP4_FORMAT, LAYOUT_ROW_SPLIT
