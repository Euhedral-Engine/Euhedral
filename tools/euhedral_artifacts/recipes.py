"""The four artifacts the engine ships, and what each stores per object.

    q3                Q3G64/Q4G64/Q5G64 text weights (grouped, row-split); NVFP4 LM head and MTP layer; Q3 draft head
    bf16              every object as the checkpoint's BF16 values (contiguous): the unquantized model
    q3 compressed     the q3 artifact with its Q3 tensors but the MLP down projections transcoded losslessly to P2E2
    nvfp4             NVFP4 text weights, MTP layer and draft head; Q3 token embedding
    nvfp4 compressed  NVFP4-SD4 text weights (4.25 bits per weight); plain NVFP4 MTP layer and draft head
    trellis           the linears of an EXL3 checkpoint (--trellis-from), the MTP layer too when it carries one
                      (else BF16); NVFP4 LM head and draft head from pre-rounded codes; Q3 token embedding

Every recipe stores the token embedding in Q3 unless Recipe.token_embedding names another format. No artifact holds
the vision tower.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from euhedral_artifacts.edrl import fail

QUANTIZATIONS = ("q3", "nvfp4", "bf16", "trellis")

LAYOUT_ROW_SPLIT = "row-split-k128-v1"
LAYOUT_SD4 = "row-split-k128-sd4-v1"
TOKEN_EMBEDDING = "text/token_embedding"
Q3_FORMAT = "Q3G64_F16S"
NVFP4_FORMAT = "NVFP4"
BF16_FORMAT = "BF16"
LAYOUT_CONTIGUOUS = "contiguous-le-v1"
TRELLIS_FORMAT = "TRELLIS"
LAYOUT_TRELLIS = "trellis-tile16-v1"


HEAD_FORMATS = ("NVFP4", "Q3G64_F16S", "Q4G64_F16S", "Q5G64_F16S", "BF16")
# The formats the engine looks token embeddings up in.
EMBEDDING_FORMATS = ("Q3G64_F16S", "NVFP4", "BF16")


@dataclass(frozen=True)
class Recipe:
    quantization: str
    compressed: bool = False
    output_head: str | None = None
    """The LM head's format (row-split) in place of the recipe's."""
    token_embedding: str | None = None
    """The token embedding's format in place of Q3 (row-split, or contiguous for BF16)."""

    def __post_init__(self) -> None:
        if self.quantization not in QUANTIZATIONS:
            fail(f"unknown quantization {self.quantization!r}; expected one of {', '.join(QUANTIZATIONS)}")
        if self.output_head is not None and self.output_head not in HEAD_FORMATS:
            fail(f"unknown LM head format {self.output_head!r}; expected one of {', '.join(HEAD_FORMATS)}")
        if self.token_embedding is not None and self.token_embedding not in EMBEDDING_FORMATS:
            fail(f"unknown token embedding format {self.token_embedding!r}; "
                 f"expected one of {', '.join(EMBEDDING_FORMATS)}")
        if self.quantization in ("bf16", "trellis") and self.compressed:
            fail(f"the {self.quantization} recipe has no compressed form")

    @property
    def name(self) -> str:
        return self.quantization + ("-compressed" if self.compressed else "")


# The four shipped artifacts; bf16 is the unquantized model, built on request.
RECIPES = tuple(Recipe(quantization, compressed) for quantization in ("q3", "nvfp4") for compressed in (False, True))


def stays_row_split(name: str) -> bool:
    """Q3 objects compressed q3 keeps in row-split Q3 with the full code range instead of P2E2 (which holds -3..3): the
    MLP down projections, where the restriction costs the most KL per byte it saves (docs/quantization/COMPRESSED_Q3.md)."""
    return re.fullmatch(r"text/layers/\d+/mlp/down", name) is not None


def is_drafting(name: str) -> bool:
    """The MTP layer and the draft head: they propose tokens the base model verifies."""
    return name.startswith("mtp/") or name == "text/draft_head"


def storage(recipe: Recipe, name: str, q3_format: str) -> tuple[str, str]:
    """(format, layout) of the quantized object `name`. `q3_format` is its format in the q3 artifact."""
    if name == TOKEN_EMBEDDING and recipe.token_embedding is not None:
        layout = LAYOUT_CONTIGUOUS if recipe.token_embedding == BF16_FORMAT else LAYOUT_ROW_SPLIT
        return recipe.token_embedding, layout
    if name == "text/output_head" and recipe.output_head == BF16_FORMAT:
        return BF16_FORMAT, LAYOUT_CONTIGUOUS
    if name == "text/output_head" and recipe.output_head is not None:
        return recipe.output_head, LAYOUT_ROW_SPLIT
    if recipe.quantization == "trellis":
        if name == TOKEN_EMBEDDING:
            return Q3_FORMAT, LAYOUT_ROW_SPLIT
        if name in ("text/output_head", "text/draft_head"):
            return NVFP4_FORMAT, LAYOUT_ROW_SPLIT
        return TRELLIS_FORMAT, LAYOUT_TRELLIS
    if recipe.quantization == "bf16":
        return BF16_FORMAT, LAYOUT_CONTIGUOUS
    if recipe.quantization == "q3" or name == TOKEN_EMBEDDING:
        return q3_format, LAYOUT_ROW_SPLIT
    if recipe.compressed and not is_drafting(name):
        return NVFP4_FORMAT, LAYOUT_SD4
    return NVFP4_FORMAT, LAYOUT_ROW_SPLIT
