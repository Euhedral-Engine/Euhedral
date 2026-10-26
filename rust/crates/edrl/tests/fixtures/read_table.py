"""Prints the tensor table of an artifact as JSON, read by the converter's own reader (tools/euhedral_artifacts/edrl.py).

    python3 -I read_table.py TOOLS_DIR ARTIFACT

Only the header, metadata and tensor table are read. The ignored `real_artifact` test compares this output with the
Rust parser's. (tools/euhedral_artifacts/inventory.py is a module, not a command, and lists a checkpoint's objects,
not an artifact's.)
"""

import json
import sys

sys.path.insert(0, sys.argv[1])

from euhedral_artifacts.edrl import read_table  # noqa: E402

with open(sys.argv[2], "rb") as handle:
    _, _, objects = read_table(handle)
json.dump([{**obj, "shape": list(obj["shape"])} for obj in objects], sys.stdout)
