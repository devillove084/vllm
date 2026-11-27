# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Local storage for validated, versioned serving capacity measurements."""

import fcntl
import hashlib
import json
import os
import tempfile
from pathlib import Path


def fingerprint(payload: object) -> str:
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ProfileStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def read(self) -> list[dict]:
        try:
            document = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return []
        if document["format"] != 1 or not isinstance(document["profiles"], list):
            raise ValueError("unsupported Plait mode state")
        return document["profiles"]

    def save(self, profile: dict, now_ms: int) -> dict:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_name(f"{self.path.name}.lock")
        with lock_path.open("a", encoding="utf-8") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            records = []
            identity = ("context", "versions", "binding", "manifest")
            expires = profile["slo_profile"]["valid_until_ms"]
            for other in self.read():
                other_expires = other["slo_profile"]["valid_until_ms"]
                if other_expires <= now_ms:
                    continue
                if all(other[key] == profile[key] for key in identity):
                    if other_expires > expires:
                        return other
                else:
                    records.append(other)
            with tempfile.NamedTemporaryFile(
                "w",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                encoding="utf-8",
                delete=False,
            ) as temporary:
                pending = Path(temporary.name)
                try:
                    json.dump(
                        {"format": 1, "profiles": [*records, profile]},
                        temporary,
                        allow_nan=False,
                    )
                    temporary.flush()
                    os.fsync(temporary.fileno())
                    temporary.close()
                    pending.replace(self.path)
                    directory = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
                    try:
                        os.fsync(directory)
                    finally:
                        os.close(directory)
                finally:
                    pending.unlink(missing_ok=True)
        return profile
