"""STUB for Person 3. Writes runs/<name>.jsonl, one JSON object per line."""

import json
import os


class RunLog:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._f = open(path, "w")

    def write(self, record):
        self._f.write(json.dumps(record) + "\n")
        self._f.flush()

    def close(self):
        self._f.close()
