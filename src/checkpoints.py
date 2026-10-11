"""Atomic update recovery and independent architecture/weight/tokenizer exports."""

from __future__ import annotations

import os
import shutil
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from .runs import canonical_hash, fsync_directory, read_json, sha256_file, write_json
from .runtime import capture_rng, require_runtime_validation, runtime_identity, select_device


def cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    return value


def training_identity(run, encoded) -> dict[str, Any]:
    return {"run": run.identity, "encoded": encoded.identity, "runtime": runtime_identity()}


def save_recovery(
    run,
    encoded,
    model,
    optimizer,
    progress: dict[str, Any],
    metrics: dict[str, int],
    *,
    is_best: bool = False,
    metric_hashes: dict[str, str] | None = None,
    scheduler_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Caller holds writer/device leases and flushes metrics before this commit."""
    directory = run.path / "checkpoints" / "recovery"
    directory.mkdir(parents=True, exist_ok=True)
    pointer_path = run.path / "checkpoints" / "committed.json"
    previous = read_json(pointer_path) if pointer_path.exists() else None
    name = f"update_{progress['updates']:08d}_{uuid.uuid4().hex}.pt"
    path = directory / name
    state = dict(progress)
    if scheduler_state is not None and scheduler_state.get("completed_updates") != state["updates"]:
        raise ValueError("Scheduler cursor differs from committed optimizer updates")
    if is_best:
        state["best_checkpoint"] = name
    payload = {
        "schema": 1,
        "identity": training_identity(run, encoded),
        "architecture": asdict(run.config.model),
        "progress": state,
        "metrics": metrics,
        "metric_hashes": metric_hashes,
        "scheduler_state": scheduler_state,
        "model": cpu_tree(model.state_dict()),
        "optimizer": cpu_tree(optimizer.state_dict()),
        "rng": capture_rng(),
    }
    # A uniquely named immutable snapshot is made durable before the pointer changes.
    with path.open("xb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    checksum = sha256_file(path)
    fsync_directory(directory)
    best_name = state.get("best_checkpoint")
    best_hash = checksum if best_name == name else (previous or {}).get("best_sha256")
    pointer = {
        "schema": 1,
        "checkpoint": name,
        "sha256": checksum,
        "best_checkpoint": best_name,
        "best_sha256": best_hash,
        "previous_checkpoint": previous["checkpoint"] if previous else None,
        "identity": payload["identity"],
        "progress": state,
        "metrics": metrics,
        "metric_hashes": metric_hashes,
        "scheduler_state": scheduler_state,
    }
    pointer["pointer_hash"] = canonical_hash(pointer)
    write_json(pointer_path, pointer)
    write_json(run.path / "checkpoints" / "status.json", state)
    keep = {name, pointer["previous_checkpoint"], best_name}
    # All removals are generated recovery files inside this run, after a durable commit.
    for old in directory.glob("update_*.pt"):
        if old.name not in keep:
            old.unlink()
    progress.clear()
    progress.update(state)
    return pointer


def read_commit(run, encoded=None) -> tuple[dict[str, Any], dict[str, Any]]:
    pointer = read_json(run.path / "checkpoints" / "committed.json")
    expected = pointer.get("pointer_hash")
    if expected != canonical_hash(
        {key: value for key, value in pointer.items() if key != "pointer_hash"}
    ):
        raise ValueError("Recovery commit pointer changed")
    if encoded is not None and pointer["identity"] != training_identity(run, encoded):
        raise ValueError("Recovery source/config/environment/data/backend identity differs")
    name = pointer["checkpoint"]
    if Path(name).name != name:
        raise ValueError("Invalid checkpoint path")
    path = run.path / "checkpoints" / "recovery" / name
    if sha256_file(path) != pointer["sha256"]:
        raise ValueError("Committed checkpoint is missing or changed")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("identity", "progress", "metrics", "metric_hashes", "scheduler_state"):
        if canonical_hash(payload[key]) != canonical_hash(pointer[key]):
            raise ValueError(f"Recovery payload/pointer mismatch: {key}")
    if payload["metrics"] != {
        "updates": payload["progress"]["updates"],
        "epochs": payload["progress"]["epoch"],
    }:
        raise ValueError("Metric cursor differs from committed training progress")
    if payload["schema"] != 1 or payload["architecture"] != asdict(run.config.model):
        raise ValueError("Recovery architecture/schema mismatch")
    best_name = pointer.get("best_checkpoint")
    if best_name:
        if (
            Path(best_name).name != best_name
            or sha256_file(path.parent / best_name) != pointer["best_sha256"]
        ):
            raise ValueError("Best recovery checkpoint changed")
    return pointer, payload


def _export(directory: Path, payload: dict[str, Any], run, tokenizer) -> dict[str, Any]:
    from safetensors.torch import save_file

    directory.mkdir()
    weights = {
        name: tensor.detach().cpu().contiguous().clone()
        for name, tensor in payload["model"].items()
    }
    save_file(
        weights, str(directory / "model.safetensors"), metadata={"format": "pt", "schema": "1"}
    )
    architecture = {
        "schema": 1,
        "class": "src.model.CodeTransformer",
        "config": payload["architecture"],
        "tied_embeddings": True,
        "parameter_count": sum(tensor.numel() for tensor in weights.values()),
        "dtype": "float32",
        "source_hash": run.identity["source_hash"],
        "tokenizer_identity": tokenizer.identity,
    }
    write_json(directory / "architecture.json", architecture)
    source = Path(tokenizer.path)
    if source.is_file():
        source = source.parent
    for name in ("tokenizer.json", "metadata.json"):
        if not (source / name).is_file():
            raise ValueError(f"Tokenizer bundle lacks {name}")
        shutil.copyfile(source / name, directory / name)
    for path in directory.iterdir():
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
    files = {path.name: sha256_file(path) for path in sorted(directory.iterdir()) if path.is_file()}
    identity = {
        "schema": 1,
        "architecture": canonical_hash(architecture),
        "weights_sha256": files["model.safetensors"],
        "tokenizer": tokenizer.identity,
        "run_identity": run.identity,
    }
    descriptor = {
        "directory": str(directory.relative_to(run.path)),
        "files": files,
        "identity": identity,
        "epoch": payload["progress"]["epoch"],
        "updates": payload["progress"]["updates"],
    }
    write_json(directory / "manifest.json", descriptor)
    fsync_directory(directory)
    return descriptor


def verify_frozen(run) -> dict[str, Any]:
    frozen = read_json(run.path / "evaluation" / "frozen.json")
    if frozen.get("receipt_hash") != canonical_hash(
        {key: value for key, value in frozen.items() if key != "receipt_hash"}
    ):
        raise ValueError("Frozen selection receipt changed")
    if frozen["run_identity"] != run.identity:
        raise ValueError("Frozen selection belongs to a different run")
    for kind in ("best", "final"):
        export = frozen["exports"][kind]
        directory = run.path / export["directory"]
        if not directory.resolve().is_relative_to(run.path.resolve()):
            raise ValueError("Frozen artifact path escapes the run")
        if read_json(directory / "manifest.json") != export:
            raise ValueError("Frozen export manifest changed")
        for name, checksum in export["files"].items():
            if Path(name).name != name or sha256_file(directory / name) != checksum:
                raise ValueError(f"Frozen export changed: {kind}/{name}")
    return frozen


def freeze_checkpoint(run, encoded) -> dict[str, Any]:
    """Freeze validation selection before held-out test evaluation, without inference."""
    from .evaluate import evaluation_protocol

    run.verify()
    require_runtime_validation(run, encoded)
    with run.writer_lock():
        run.bind("encoded", encoded.identity)
        protocol = evaluation_protocol(run)
        run.bind("evaluation_protocol", protocol)
        frozen_path = run.path / "evaluation" / "frozen.json"
        if frozen_path.exists():
            frozen = verify_frozen(run)
            if frozen["encoded_identity"] != encoded.identity:
                raise ValueError("Encoded dataset differs from frozen selection")
            if frozen["evaluation_protocol"] != protocol:
                raise ValueError("Frozen evaluation protocol differs")
            return frozen
        pointer, final = read_commit(run, encoded)
        if final["progress"]["status"] != "complete" or not pointer["best_checkpoint"]:
            raise RuntimeError(
                "Training must finish and select a validation checkpoint before freezing"
            )
        best_path = run.path / "checkpoints" / "recovery" / pointer["best_checkpoint"]
        best = torch.load(best_path, map_location="cpu", weights_only=True)
        if best["identity"] != final["identity"]:
            raise ValueError("Best checkpoint identity mismatch")
        # A unique export batch makes a crash before selection commit harmless and reviewable.
        export_root = run.path / "checkpoints" / f"export_{uuid.uuid4().hex}"
        export_root.mkdir()
        exports = {
            "best": _export(export_root / "best", best, run, encoded.tokenizer),
            "final": _export(export_root / "final", final, run, encoded.tokenizer),
        }
        frozen = {
            "schema": 1,
            "run_identity": run.identity,
            "encoded_identity": encoded.identity,
            "artifact_root": str(export_root.relative_to(run.path)),
            "exports": exports,
            "selected": "best",
            "selection_metric": "validation token-weighted negative log likelihood",
            "best_epoch": final["progress"]["best_epoch"],
            "best_validation_nll": final["progress"]["best_validation_nll"],
            "training_stop_reason": final["progress"]["stop_reason"],
            "evaluation_protocol": protocol,
        }
        frozen["receipt_hash"] = canonical_hash(frozen)
        write_json(frozen_path, frozen)
        return frozen


def load_inference(run, device=None):
    """Reconstruct selected weights and tokenizer; never load a pickled whole model."""
    from safetensors.torch import load_file

    from .config import ModelConfig
    from .model import CodeTransformer
    from .runtime import preserve_rng
    from .tokenizer import load_tokenizer

    run.verify()
    frozen = verify_frozen(run)
    export = frozen["exports"][frozen["selected"]]
    directory = run.path / export["directory"]
    architecture = read_json(directory / "architecture.json")
    tokenizer = load_tokenizer(directory)
    if tokenizer.identity != export["identity"]["tokenizer"]:
        raise ValueError("Frozen tokenizer identity differs")
    chosen = select_device() if device is None else torch.device(device)
    with preserve_rng():
        with torch.device("cpu"):
            model = CodeTransformer(ModelConfig(**architecture["config"])).float()
        model.load_state_dict(
            load_file(str(directory / "model.safetensors"), device="cpu"), strict=True
        )
        model.to(chosen).eval()
    return model, tokenizer, export["identity"]
