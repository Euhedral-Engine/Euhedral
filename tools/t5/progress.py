"""The generation progress file: one JSON document rewritten in place after every finished conversation."""
from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path


class Progress:
    """One JSON file, rewritten in place (temporary file in the same directory, fsync, os.replace) after every
    finished conversation: {"settings": {...}, "conversations": {id: record}}. Finished conversations are skipped
    on resume; an unfinished one (or one that ended in error) restarts, since generation is deterministic."""

    def __init__(self, path: Path, settings: dict, conversations: dict) -> None:
        self.path = path
        self.settings = settings
        self.conversations = conversations
        self._lock = threading.Lock()

    @classmethod
    def open(cls, path: Path, settings: dict) -> "Progress":
        settings = json.loads(json.dumps(settings))
        path = Path(path)
        if path.exists():
            stored = json.loads(path.read_text(encoding="utf-8"))
            if stored.get("settings") != settings:
                changed = sorted(k for k in set(stored.get("settings", {})) | set(settings)
                                 if stored.get("settings", {}).get(k) != settings.get(k))
                raise ValueError(f"{path} was written with different settings: {', '.join(changed)}")
            return cls(path, settings, stored.get("conversations", {}))
        path.parent.mkdir(parents=True, exist_ok=True)
        progress = cls(path, settings, {})
        progress._save()
        return progress

    def done(self) -> set[str]:
        with self._lock:
            return {k for k, v in self.conversations.items() if v.get("ended") != "error"}

    def record(self, conversation_id: str, record: dict) -> None:
        with self._lock:
            self.conversations[conversation_id] = record
            self._save()

    def _save(self) -> None:
        text = json.dumps({"settings": self.settings, "conversations": self.conversations},
                          ensure_ascii=False, sort_keys=True)
        handle, temporary = tempfile.mkstemp(dir=self.path.parent, prefix=self.path.name + ".")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as out:
                out.write(text)
                out.flush()
                os.fsync(out.fileno())
            os.replace(temporary, self.path)
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise
        directory = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
