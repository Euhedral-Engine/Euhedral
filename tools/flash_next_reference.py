#!/usr/bin/env python3
"""Python reference harness for Qwen3.8-Flash-Next: fixtures from the unmodified upstream transformers `qwen4_exp`
implementation over the weights of the NVFP4 EDRL v3 artifact. See docs/FLASH_NEXT_REFERENCE.md.

  PYTHONPATH=tools ~/.cache/flash-next-ref-venv/bin/python tools/flash_next_reference.py list-cases
  ... check-artifact
  ... fixtures --out /home/brandon/fixtures/flash-next --case short
  ... generate --prompt-ids 785 6722 --steps 4
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from flash_next_reference.cli import main  # noqa: E402  (the package next to this script)

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
