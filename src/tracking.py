"""Durable local measurements with checkpoint-bound recovery and TensorBoard views."""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .runs import canonical_hash, fsync_directory


class MetricLog:
    """JSONL is authoritative; CSV is regenerated from the committed JSONL prefix."""

    def __init__(
        self,
        directory: Path,
        committed: dict[str, int] | None = None,
        expected_hashes: dict[str, str] | None = None,
    ) -> None:
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.rows: dict[str, list[dict[str, Any]]] = {}
        self.handles = {}
        committed = committed or {"updates": 0, "epochs": 0}
        for kind in ("updates", "epochs"):
            path = directory / f"{kind}.jsonl"
            retained = []
            if path.exists():
                with path.open(encoding="utf-8") as handle:
                    for index, line in enumerate(handle):
                        if index >= committed[kind]:
                            break
                        retained.append(json.loads(line))
            if len(retained) != committed[kind]:
                raise ValueError(
                    f"Committed {kind} measurements are missing; refusing invented recovery"
                )
            if expected_hashes is not None and canonical_hash(retained) != expected_hashes[kind]:
                raise ValueError(f"Committed {kind} measurements changed")
            self.rows[kind] = retained
            temporary = path.with_suffix(".recovery.tmp")
            with temporary.open("w", encoding="utf-8") as handle:
                for row in retained:
                    handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            self.handles[kind] = path.open("a", encoding="utf-8")
        self.flush()

    @property
    def counts(self) -> dict[str, int]:
        return {kind: len(rows) for kind, rows in self.rows.items()}

    @property
    def hashes(self) -> dict[str, str]:
        return {kind: canonical_hash(rows) for kind, rows in self.rows.items()}

    def append(self, kind: str, row: dict[str, Any]) -> None:
        if kind not in self.handles:
            raise ValueError(kind)
        serialized = json.dumps(row, sort_keys=True, allow_nan=False)
        self.handles[kind].write(serialized + "\n")
        self.rows[kind].append(json.loads(serialized))

    def flush(self) -> None:
        for kind, handle in self.handles.items():
            handle.flush()
            os.fsync(handle.fileno())
            rows = self.rows[kind]
            path = self.directory / f"{kind}.csv"
            temporary = path.with_suffix(".tmp")
            with temporary.open("w", encoding="utf-8", newline="") as output:
                if rows:
                    columns = sorted({key for row in rows for key in row})
                    writer = csv.DictWriter(output, fieldnames=columns)
                    writer.writeheader()
                    writer.writerows(rows)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
        fsync_directory(self.directory)

    def close(self) -> None:
        for handle in self.handles.values():
            if not handle.closed:
                handle.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def tensorboard_writer(directory: Path, committed_update: int):
    from torch.utils.tensorboard import SummaryWriter

    return SummaryWriter(
        log_dir=str(directory), purge_step=committed_update + 1, flush_secs=30, max_queue=100
    )


def append_log(directory: Path, message: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with (directory / "training.log").open("a", encoding="utf-8") as handle:
        handle.write(f"{timestamp} {message}\n")
        handle.flush()
