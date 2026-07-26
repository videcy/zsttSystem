"""Track file change status via SHA256 hashes for incremental pipeline runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


class FileManifest:
    """Persistent file-hash manifest for detecting added/changed/removed files."""

    def __init__(self, manifest_path: str | Path = "outputs/.file_manifest.json"):
        self.path = Path(manifest_path)
        self._data: dict[str, str] = {}
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._data = {}

    @staticmethod
    def file_hash(path: str | Path) -> str:
        """Return the SHA256 hex digest of a file."""
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            while True:
                block = fh.read(65536)
                if not block:
                    break
                h.update(block)
        return h.hexdigest()

    def check(
        self,
        files: list[Path],
    ) -> tuple[list[Path], list[Path], list[str]]:
        """Return (added, changed, removed_relative_paths) compared to manifest."""
        current: dict[str, str] = {}

        for f in files:
            if f.is_file():
                rel = str(f.as_posix())
                current[rel] = self.file_hash(f)

        previous_keys: set[str] = set(self._data.keys())
        current_keys: set[str] = set(current.keys())

        added_keys = current_keys - previous_keys
        removed_keys = previous_keys - current_keys
        common_keys = current_keys & previous_keys

        added = [f for f in files if str(f.as_posix()) in added_keys]
        changed = [
            f for f in files
            if str(f.as_posix()) in common_keys
            and current.get(str(f.as_posix())) != self._data.get(str(f.as_posix()))
        ]
        removed_relative = sorted(removed_keys)

        return added, changed, removed_relative

    def update(self, files: list[Path]) -> None:
        """Record current hashes for the given files."""
        for f in files:
            if f.is_file():
                self._data[str(f.as_posix())] = self.file_hash(f)

    def remove(self, relative_paths: list[str]) -> None:
        """Remove entries for deleted files."""
        for p in relative_paths:
            self._data.pop(p, None)

    def save(self) -> None:
        """Persist to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
