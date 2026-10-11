"""Explicit numbered runs, immutable identities and atomic local persistence."""

from __future__ import annotations

import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import sys
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .config import ExperimentConfig

_LOCK_GUARD = threading.Lock()
_LOCKS: dict[str, tuple[threading.RLock, dict[str, Any]]] = {}
RUN_DIRECTORIES = (
    "config",
    "environment",
    "source_snapshot",
    "logs",
    "tensorboard",
    "checkpoints",
    "evaluation",
    "benchmarks",
    "sessions",
    "figures",
    "report",
)


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path | str) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_json(path: Path | str, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def exclusive_lock(path: Path) -> Iterator[None]:
    """Nonblocking, process-safe and same-thread-reentrant advisory lock."""
    path = path.resolve()
    key = str(path)
    with _LOCK_GUARD:
        mutex, state = _LOCKS.setdefault(key, (threading.RLock(), {"depth": 0}))
    if not mutex.acquire(blocking=False):
        raise RuntimeError(f"Another thread owns {path.name}")
    try:
        if state["depth"] == 0:
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = path.open("a+")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                handle.close()
                raise RuntimeError(f"Another process owns {path.name}") from error
            state["handle"] = handle
        state["depth"] += 1
        try:
            yield
        finally:
            state["depth"] -= 1
            if state["depth"] == 0:
                handle = state.pop("handle")
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                handle.close()
    finally:
        mutex.release()


def environment_identity() -> dict[str, Any]:
    packages = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            packages[name.lower().replace("_", "-")] = distribution.version
    return {
        "python": platform.python_version(),
        "executable": str(Path(sys.executable).absolute()),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": dict(sorted(packages.items())),
    }


def source_files(root: Path) -> dict[str, str]:
    files = list((root / "src").rglob("*.py"))
    files.extend((root / "tests").rglob("*.py"))
    for name in ("requirements.in", "requirements.txt", "pyproject.toml", "uv.lock"):
        if (root / name).is_file():
            files.append(root / name)
    if not any(path.name == "config.py" for path in files):
        raise ValueError("Project root lacks src/config.py")
    return {str(path.relative_to(root)): sha256_file(path) for path in sorted(files)}


@dataclass(frozen=True)
class Run:
    root: Path
    path: Path
    config: ExperimentConfig
    identity: dict[str, Any]

    def verify(self) -> None:
        if read_json(self.path / "config" / "identity.json") != self.identity:
            raise ValueError("Run identity changed")
        expected_parent = _runs_directory(self.root, self.identity.get("runs_directory", "runs"))
        if self.path.resolve().parent != expected_parent or self.path.name != self.identity["run"]:
            raise ValueError("Run path differs from its registered directory/name")
        config = read_json(self.path / "config" / "experiment.json")
        if canonical_hash(config) != self.identity["config_hash"]:
            raise ValueError("Saved configuration changed")
        if canonical_hash(self.config.to_dict()) != self.identity["config_hash"]:
            raise ValueError("Requested configuration differs from this run")
        manifest = read_json(self.path / "source_snapshot" / "manifest.json")
        if canonical_hash(manifest) != self.identity["source_hash"]:
            raise ValueError("Source manifest changed")
        if source_files(self.root) != manifest:
            raise ValueError("Source or dependency files changed; create a fresh run")
        for relative, expected in manifest.items():
            if sha256_file(self.path / "source_snapshot" / relative) != expected:
                raise ValueError(f"Saved source changed: {relative}")
        notebook_manifest = read_json(self.path / "source_snapshot" / "notebook_manifest.json")
        if canonical_hash(notebook_manifest) != self.identity["notebook_snapshot_hash"]:
            raise ValueError("Notebook snapshot manifest changed")
        for relative, expected in notebook_manifest.items():
            if sha256_file(self.path / "source_snapshot" / relative) != expected:
                raise ValueError(f"Saved notebook changed: {relative}")
        saved_environment = read_json(self.path / "environment" / "identity.json")
        if canonical_hash(saved_environment) != self.identity["environment_hash"]:
            raise ValueError("Saved environment identity changed")
        if canonical_hash(environment_identity()) != self.identity["environment_hash"]:
            raise ValueError(
                "Environment changed; use its original environment or create a fresh run"
            )

    def bind(self, name: str, identity: Any) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            raise ValueError("Binding names use lowercase letters, digits and underscores")
        with self.writer_lock():
            path = self.path / "config" / "bindings.json"
            bindings = read_json(path)
            if name in bindings and canonical_hash(bindings[name]) != canonical_hash(identity):
                raise ValueError(f"Immutable binding differs: {name}")
            if name not in bindings:
                bindings[name] = identity
                write_json(path, bindings)

    def writer_lock(self):
        return exclusive_lock(self.path / ".writer.lock")


def _runs_directory(root: Path, runs_directory: Path | str | None) -> Path:
    selected = Path("runs") if runs_directory is None else Path(runs_directory).expanduser()
    path = (selected if selected.is_absolute() else root / selected).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError("runs_directory must be a bounded subdirectory of this project")
    return path


def create_run(
    root: Path | str, config: ExperimentConfig, *, runs_directory: Path | str | None = None
) -> Run:
    root = Path(root).expanduser().resolve()
    manifest, environment = source_files(root), environment_identity()
    runs = _runs_directory(root, runs_directory)
    runs.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(runs / ".allocation.lock"):
        existing = [int(p.name[4:]) for p in runs.iterdir() if re.fullmatch(r"run_\d{3,}", p.name)]
        number = max(existing, default=0) + 1
        while True:
            path = runs / f"run_{number:03d}"
            try:
                path.mkdir()
                break
            except FileExistsError:
                number += 1
        # A partially initialized run remains visible and is never silently reused.
        for name in RUN_DIRECTORIES:
            (path / name).mkdir()
        identity = {
            "schema": 1,
            "run": path.name,
            "runs_directory": str(runs.relative_to(root)),
            "config_hash": canonical_hash(config.to_dict()),
            "source_hash": canonical_hash(manifest),
            "environment_hash": canonical_hash(environment),
        }
        write_json(path / "config" / "experiment.json", config.to_dict())
        write_json(path / "config" / "bindings.json", {})
        write_json(path / "environment" / "identity.json", environment)
        for relative in manifest:
            destination = path / "source_snapshot" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / relative, destination)
        # Record the creation-time notebook without binding mutable live outputs.
        # Pipeline modules, tests and dependency specifications remain strictly bound.
        notebooks = {}
        for notebook in sorted((root / "notebooks").glob("*.ipynb")):
            relative = str(notebook.relative_to(root))
            destination = path / "source_snapshot" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(notebook, destination)
            notebooks[relative] = sha256_file(destination)
        identity["notebook_snapshot_hash"] = canonical_hash(notebooks)
        write_json(path / "source_snapshot" / "notebook_manifest.json", notebooks)
        write_json(path / "source_snapshot" / "manifest.json", manifest)
        write_json(path / "config" / "identity.json", identity)
    run = Run(root, path, config, identity)
    run.verify()
    return run


def open_run(root: Path | str, name: str, *, runs_directory: Path | str | None = None) -> Run:
    if not re.fullmatch(r"run_\d{3,}", name):
        raise ValueError("Select an explicit run_XXX name")
    root = Path(root).expanduser().resolve()
    path = _runs_directory(root, runs_directory) / name
    run = Run(
        root,
        path,
        ExperimentConfig.from_dict(read_json(path / "config" / "experiment.json")),
        read_json(path / "config" / "identity.json"),
    )
    run.verify()
    return run
