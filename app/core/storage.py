from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any


class JsonStore:
    def __init__(self) -> None:
        self._lock = threading.RLock()

    def load(self, path: Path, default: Any) -> Any:
        with self._lock:
            if not path.exists():
                return default
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)

    def save(self, path: Path, payload: Any) -> None:
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = path.with_suffix(path.suffix + ".tmp")
            with temp_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            temp_path.replace(path)


STORE = JsonStore()
