"""Small JSONL logger used by sensor and shadow modes."""

import json
import math
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np


class JsonlLogger:
    def __init__(self, path: Optional[str] = None):
        self.path = Path(path) if path else None
        self.handle = None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.handle = self.path.open("a", encoding="utf-8")

    def write(self, record: Mapping[str, Any]) -> None:
        line = json.dumps(_jsonable(record), ensure_ascii=False, allow_nan=False)
        if self.handle:
            self.handle.write(line + "\n")
            self.handle.flush()
        print(line)

    def close(self) -> None:
        if self.handle:
            self.handle.close()
            self.handle = None


def _jsonable(value):
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
