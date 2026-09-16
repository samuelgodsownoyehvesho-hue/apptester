"""Artifact storage on the local filesystem.

Screenshots, videos, HAR files and console logs are the evidence a verdict
cites, so they need to be addressable and durable, but they are bulky and may
contain captured application data.

Two properties matter:

* **Content-addressed.** The key includes a hash of the bytes, so re-running a
  test that produces an identical screenshot writes nothing new. This makes
  storage grow with distinct evidence rather than with run count.
* **Bounded.** A video per execution adds up quickly. Callers get a size report
  so a run can refuse to keep going rather than filling the disk.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from crucible.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    """A stored artifact, addressable without reading it back."""

    key: str
    kind: str
    run_id: str
    size_bytes: int
    sha256: str
    created_at: datetime

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "kind": self.kind,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


class ArtifactStore:
    """Writes artifacts under a root directory, one subtree per run."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    def _key_for(self, run_id: str, kind: str, digest: str, suffix: str) -> str:
        safe_kind = "".join(c for c in kind if c.isalnum() or c in "-_") or "misc"
        return f"{run_id}/{safe_kind}/{digest[:16]}{suffix}"

    def save_bytes(
        self,
        run_id: str,
        kind: str,
        data: bytes,
        *,
        suffix: str = ".bin",
    ) -> ArtifactRef:
        """Store raw bytes, returning a reference. Idempotent for equal content."""
        digest = hashlib.sha256(data).hexdigest()
        key = self._key_for(run_id, kind, digest, suffix)
        target = self._root / key

        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            # Write to a temporary name then rename, so a crash mid-write
            # cannot leave a truncated file that looks like valid evidence.
            temporary = target.with_suffix(target.suffix + ".part")
            temporary.write_bytes(data)
            temporary.replace(target)

        return ArtifactRef(
            key=key,
            kind=kind,
            run_id=run_id,
            size_bytes=len(data),
            sha256=digest,
            created_at=datetime.now(UTC),
        )

    def save_text(self, run_id: str, kind: str, text: str, *, suffix: str = ".txt") -> ArtifactRef:
        return self.save_bytes(run_id, kind, text.encode("utf-8"), suffix=suffix)

    def save_json(self, run_id: str, kind: str, payload: Any) -> ArtifactRef:
        serialised = json.dumps(payload, indent=2, default=str)
        return self.save_text(run_id, kind, serialised, suffix=".json")

    def path_for(self, key: str) -> Path:
        """Resolve an artifact key to a path, refusing to escape the root.

        A key is derived from target-controlled input in places, so traversal
        must be rejected here rather than trusted upstream.
        """
        root = self._root.resolve()
        candidate = (self._root / key).resolve()
        if not candidate.is_relative_to(root):
            raise ValueError(f"Artifact key escapes the store root: {key!r}")
        return candidate

    def read_bytes(self, key: str) -> bytes:
        return self.path_for(key).read_bytes()

    def read_json(self, key: str) -> Any:
        return json.loads(self.path_for(key).read_text(encoding="utf-8"))

    def list_run(self, run_id: str) -> list[ArtifactRef]:
        """Enumerate stored artifacts for a run, oldest directory order."""
        run_dir = self._root / run_id
        if not run_dir.is_dir():
            return []

        refs: list[ArtifactRef] = []
        for path in sorted(run_dir.rglob("*")):
            if not path.is_file() or path.suffix == ".part":
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            stat = path.stat()
            kind = path.parent.name
            refs.append(
                ArtifactRef(
                    key=str(path.relative_to(self._root)).replace("\\", "/"),
                    kind=kind,
                    run_id=run_id,
                    size_bytes=stat.st_size,
                    sha256=digest,
                    created_at=datetime.fromtimestamp(stat.st_mtime, tz=UTC),
                )
            )
        return refs

    def total_bytes(self, run_id: str) -> int:
        return sum(ref.size_bytes for ref in self.list_run(run_id))

    def delete_run(self, run_id: str) -> int:
        """Remove every artifact for a run. Returns the number of files removed."""
        run_dir = self._root / run_id
        if not run_dir.is_dir():
            return 0

        removed = 0
        for path in sorted(run_dir.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
                removed += 1
            elif path.is_dir():
                path.rmdir()
        run_dir.rmdir()
        logger.info("artifacts_deleted run=%s files=%d", run_id, removed)
        return removed
