"""Python reference harness for Qwen3.8-Flash-Next (`qwen4_exp`): fixtures from the unmodified upstream
transformers implementation over the weights of Euhedral's NVFP4 EDRL v3 artifact (docs/FLASH_NEXT_REFERENCE.md)."""

import sys
from pathlib import Path

# `euhedral_artifacts` lives next to this package in tools/.
_TOOLS = str(Path(__file__).resolve().parent.parent)
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

UPSTREAM_REVISION = "f5ab85619d989359ef47b5efed8a91a15045627b"
