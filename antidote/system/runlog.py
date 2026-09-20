"""JSON Lines run logging. Owner: Person 3."""

import dataclasses
import json
import math
import os


def _json_safe(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _json_safe(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    raise TypeError(f"cannot write {type(value).__name__} to the run log")


class RunLog:
    """Write one strict JSON object per line and flush each record."""

    def __init__(self, path):
        path = os.fspath(path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._f = open(path, "w", encoding="utf-8")

    def write(self, record):
        if self._f is None:
            raise ValueError("run log is closed")
        safe = _json_safe(record)
        self._f.write(json.dumps(safe, allow_nan=False, sort_keys=True) + "\n")
        self._f.flush()

    def close(self):
        if self._f is not None:
            self._f.close()
            self._f = None
