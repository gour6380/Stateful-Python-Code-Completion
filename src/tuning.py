"""Explicit, disposable training trials; importing this module starts no work.

The controller must set the MPS environment before importing this module and run
each job in a fresh process. Only ``run_job`` loads data or constructs a model.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import plistlib
import statistics
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import psutil
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from tqdm.auto import tqdm

from .bootstrap import validate_environment
from .config import ExperimentConfig
from .model import CodeTransformer
from .runs import canonical_hash, create_run, read_json, sha256_file, write_json
from .runtime import (
    accelerator_lease,
    capture_rng,
    configure_execution,
    memory_snapshot,
    release_resources,
    restore_rng,
    runtime_identity,
    seed_everything,
    select_device,
    synchronize,
    validate_runtime,
)
from .tokenizer import load_encoded_data
from .tracking import MetricLog, tensorboard_writer
from .train import (
    epoch_order,
    learning_rate_for_update,
    make_loader,
    optimizer_groups,
    train_model,
    train_update,
)

_MIB = 1024**2
_TOLERANCE = {"rtol": 3e-4, "atol": 5e-4}


class TuningGuardError(RuntimeError):
    """Stop a disposable trial instead of changing its memory/precision policy."""


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _inside(path: str | Path, directory: Path) -> Path:
    path = Path(path).expanduser().resolve()
    if not path.is_relative_to(directory) or path == directory:
        raise ValueError(f"Trial output must be inside {directory}: {path}")
    return path


def _performance_statistics(value) -> list[dict]:
    found = []
    if isinstance(value, dict):
        statistics_value = value.get("PerformanceStatistics")
        if isinstance(statistics_value, dict):
            found.append(statistics_value)
        for child in value.values():
            found.extend(_performance_statistics(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_performance_statistics(child))
    return found


def _mac_metrics() -> dict:
    result = {"device_utilization_percent": None, "memory_pressure_level": None}
    if sys.platform != "darwin":
        return result
    try:
        completed = subprocess.run(
            ["/usr/sbin/ioreg", "-r", "-c", "AGXAccelerator", "-a"],
            capture_output=True,
            check=True,
            timeout=1.0,
        )
        candidates = []
        for record in _performance_statistics(plistlib.loads(completed.stdout)):
            for key, value in record.items():
                normalized = "".join(character for character in key.lower() if character.isalnum())
                if normalized in {"deviceutilization", "deviceutilizationpercent"}:
                    if isinstance(value, (int, float)) and math.isfinite(value):
                        candidates.append(float(value))
        if candidates:
            result["device_utilization_percent"] = max(candidates)
    except (
        OSError,
        ValueError,
        plistlib.InvalidFileException,
        subprocess.SubprocessError,
    ) as error:
        result["utilization_query_error"] = f"{type(error).__name__}: {error}"
    try:
        completed = subprocess.run(
            ["/usr/sbin/sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
            capture_output=True,
            text=True,
            check=True,
            timeout=1.0,
        )
        result["memory_pressure_level"] = int(completed.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        result["pressure_query_error"] = f"{type(error).__name__}: {error}"
    return result


class TelemetrySampler:
    """Half-second system samples; the hot-path guard only reads cached state.

    GPU activity is for the whole device, including other applications. Allocation
    maxima are sampled maxima, never claimed as allocator peaks. Unified-memory
    RSS and device counters overlap and must not be added together.
    """

    def __init__(self, output: Path, device, interval: float = 0.5):
        self.path = output
        self.device = device
        self.interval = interval
        self.rows: list[dict] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._handle = None
        self._phase = "setup"
        self._reason = None
        self._critical_count = 0
        self._baseline_swap = psutil.swap_memory().used
        self._started = time.perf_counter()
        self._last_sample_started = self._started

    def start(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("x", encoding="utf-8")
        self._sample()
        self.guard()
        self._thread = threading.Thread(target=self._loop, name="tuning-telemetry", daemon=True)
        self._thread.start()
        return self

    def phase(self, phase: str):
        with self._lock:
            self._phase = phase

    def _sample(self):
        self._last_sample_started = time.perf_counter()
        available = psutil.virtual_memory().available
        swap_used = psutil.swap_memory().used
        counters = memory_snapshot(self.device)
        mac = _mac_metrics()
        with self._lock:
            phase = self._phase
        row = {
            "timestamp": _utc(),
            "elapsed_seconds": time.perf_counter() - self._started,
            "phase": phase,
            "available_bytes": available,
            "swap_used_bytes": swap_used,
            "swap_growth_bytes": max(0, swap_used - self._baseline_swap),
            **counters,
            **mac,
        }
        with self._lock:
            self._critical_count = (
                self._critical_count + 1 if mac["memory_pressure_level"] == 4 else 0
            )
            reason = None
            if mac["memory_pressure_level"] is None:
                reason = "Required critical-memory-pressure counter is unavailable"
            elif swap_used - self._baseline_swap > 512 * _MIB:
                reason = "System swap use grew by more than 512 MiB"
            elif self._critical_count >= 3:
                reason = "Critical system memory pressure persisted for three samples"
            if reason and self._reason is None:
                self._reason = reason
            row["guard_reason"] = self._reason
            self.rows.append(row)
        self._handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        self._handle.flush()

    def _loop(self):
        # Target a start-to-start cadence; slow OS queries can lengthen it.
        while not self._stop.wait(
            max(0.0, self._last_sample_started + self.interval - time.perf_counter())
        ):
            try:
                self._sample()
            except Exception as error:
                with self._lock:
                    self._reason = (
                        f"Required memory telemetry failed: {type(error).__name__}: {error}"
                    )
                return

    def guard(self):
        with self._lock:
            reason = self._reason
        if reason:
            raise TuningGuardError(reason)

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                raise RuntimeError("Telemetry thread did not stop")
        if self._handle is not None and not self._handle.closed:
            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._handle.close()

    def summary(self) -> dict:
        with self._lock:
            rows = list(self.rows)
            reason = self._reason
        measured = [
            row
            for row in rows
            if row["phase"] in {"screen_measured", "validation_measured", "epoch_training"}
        ]
        utilization = [
            row["device_utilization_percent"]
            for row in measured
            if row["device_utilization_percent"] is not None
        ]
        result = {
            "path": str(self.path),
            "sample_interval_seconds": self.interval,
            "sampling_interval_note": "Target start-to-start interval; slow OS queries can lengthen it",
            "sample_count": len(rows),
            "measured_sample_count": len(measured),
            "utilization_scope": "whole device; includes unrelated application activity",
            "maxima_scope": "sampled maxima across the complete job; not allocator peaks",
            "unified_memory_note": "RSS and MPS counters overlap; never sum them",
            "mean_device_utilization_percent": statistics.fmean(utilization)
            if utilization
            else None,
            "sampled_max_device_utilization_percent": max(utilization) if utilization else None,
            "guard_reason": reason,
            "guard_policy": {
                "available_memory_floor_bytes": None,
                "available_memory_note": "Observation only; owner removed the available-RAM stop",
                "maximum_swap_growth_bytes": 512 * _MIB,
                "critical_pressure_consecutive_samples": 3,
                "stop_on_oom_or_nonfinite": True,
            },
        }
        for key in ("rss_bytes", "allocated_bytes", "driver_allocated_bytes", "swap_growth_bytes"):
            values = [row[key] for row in rows if row.get(key) is not None]
            result[f"sampled_max_{key}"] = max(values) if values else None
        result["sampled_min_available_bytes"] = min(
            (row["available_bytes"] for row in rows), default=None
        )
        result["pressure_query_available"] = any(
            row["memory_pressure_level"] is not None for row in rows
        )
        sample_gaps = [
            later["elapsed_seconds"] - earlier["elapsed_seconds"]
            for earlier, later in zip(rows, rows[1:])
        ]
        result["observed_mean_sample_interval_seconds"] = (
            statistics.fmean(sample_gaps) if sample_gaps else None
        )
        result["observed_max_sample_interval_seconds"] = max(sample_gaps, default=None)
        return result


def _weights_hash(model) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _new_model(config, device, receipt, *, dropout=None):
    seed_everything(config.training.seed)
    model_config = config.model if dropout is None else replace(config.model, dropout=dropout)
    with torch.device("cpu"):
        model = CodeTransformer(model_config).float()
    receipt["initial_weights_sha256"] = _weights_hash(model)
    return model.to(device)


def _new_optimizer(model, settings):
    return torch.optim.AdamW(
        optimizer_groups(model, settings.weight_decay),
        lr=settings.learning_rate,
        betas=settings.betas,
        eps=settings.eps,
        weight_decay=0.0,
    )


def _loader(dataset, config, device, indices, *, batch_size=None):
    return make_loader(
        dataset,
        batch_size=batch_size or config.training.batch_size,
        device=device,
        indices=indices,
        seed=config.training.seed,
        num_workers=config.training.num_workers,
    )


def _close_iterator(iterator):
    # PyTorch exposes no public early-close DataLoader API. A disposable screen
    # may stop before exhaustion; explicitly join its private worker iterator.
    close = getattr(iterator, "_shutdown_workers", None)
    if close is not None:
        close()


def _group(iterator, count, guard):
    batches = []
    for _ in range(count):
        guard()
        batches.append(next(iterator))
    return batches


def _indices(dataset, count: int, seed: int) -> list[int]:
    if len(dataset) < count:
        raise ValueError(f"Trial requires {count} windows; dataset has {len(dataset)}")
    return epoch_order(len(dataset), seed, 0)[:count]


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _screen(job, config, encoded, device, sampler, receipt, output):
    settings = config.training
    repetitions, warmups = int(job.get("repetitions", 12)), int(job.get("warmups", 3))
    if min(repetitions, warmups) < 1 or repetitions >= settings.checkpoint_interval:
        raise ValueError("Screen needs positive warmup/repetition counts below checkpoint cadence")
    dataset = encoded.dataset("train")
    indices = _indices(dataset, 32 * max(repetitions, warmups), settings.seed)
    model = _new_model(config, device, receipt)
    initial = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    rng = capture_rng()
    optimizer = _new_optimizer(model, settings)
    total_updates = math.ceil(len(dataset) / 32) * settings.max_epochs
    sampler.phase("screen_warmup")
    iterator = iter(_loader(dataset, config, device, indices[: 32 * warmups]))
    try:
        for update in range(warmups):
            for group in optimizer.param_groups:
                group["lr"] = learning_rate_for_update(settings, update, total_updates)
            batches = _group(iterator, settings.accumulation_steps, sampler.guard)
            train_update(
                model, optimizer, batches, device, settings.gradient_clip, guard=sampler.guard
            )
        synchronize(device)
    finally:
        _close_iterator(iterator)
    del optimizer, batches, iterator
    model.load_state_dict(initial, strict=True)
    model.zero_grad(set_to_none=True)
    del initial
    gc.collect()
    optimizer = _new_optimizer(model, settings)
    restore_rng(rng)
    synchronize(device)
    sampler.guard()
    trial_directory = output.with_suffix("")
    trial_directory.mkdir(parents=True, exist_ok=True)
    rows = []
    iterator = None
    with MetricLog(trial_directory / "logs") as metrics:
        writer = tensorboard_writer(trial_directory / "tensorboard", 0)
        bar = tqdm(total=repetitions, desc=f"Screen block {job.get('block', 0)}", unit="update")
        try:
            sampler.phase("screen_measured")
            started = time.perf_counter()
            iterator = iter(_loader(dataset, config, device, indices[: 32 * repetitions]))
            for update in range(repetitions):
                update_started = time.perf_counter()
                sampler.guard()
                input_started = time.perf_counter()
                batches = _group(iterator, settings.accumulation_steps, sampler.guard)
                input_wait = time.perf_counter() - input_started
                lr = learning_rate_for_update(settings, update, total_updates)
                for parameter_group in optimizer.param_groups:
                    parameter_group["lr"] = lr
                compute_started = time.perf_counter()
                row = train_update(
                    model, optimizer, batches, device, settings.gradient_clip, guard=sampler.guard
                )
                synchronize(device)
                compute_seconds = time.perf_counter() - compute_started
                row.update(
                    update=update + 1,
                    learning_rate=lr,
                    input_wait_seconds=input_wait,
                    compute_seconds=compute_seconds,
                )
                metrics.append("updates", row)
                writer.add_scalar("train/update_nll", row["nll"], update + 1)
                writer.add_scalar("train/learning_rate", lr, update + 1)
                writer.add_scalar(
                    "train/gradient_norm_before_clip", row["gradient_norm_before_clip"], update + 1
                )
                bar.update(1)
                bar.set_postfix(nll=f"{row['nll']:.4f}", lr=f"{lr:.2g}")
                sampler.guard()
                row["wall_seconds"] = time.perf_counter() - update_started
                rows.append(row)
            writer.flush()
            metrics.flush()
            synchronize(device)
            elapsed = time.perf_counter() - started
            sampler.guard()
        finally:
            sampler.phase("screen_cleanup")
            bar.close()
            writer.close()
            if iterator is not None:
                _close_iterator(iterator)
    tokens = sum(row["target_tokens"] for row in rows)
    del model, optimizer
    return {
        "updates": len(rows),
        "target_tokens": tokens,
        "wall_seconds": elapsed,
        "tokens_per_second": tokens / elapsed,
        "median_update_seconds": statistics.median(row["wall_seconds"] for row in rows),
        "p95_update_seconds": _percentile([row["wall_seconds"] for row in rows], 0.95),
        "input_wait_seconds": sum(row["input_wait_seconds"] for row in rows),
        "checkpoint_seconds": 0.0,
        "checkpoint_note": "Normal cadence is not reached by this short screen",
        "measured_updates": rows,
        "window_indices_sha256": canonical_hash(indices[: 32 * repetitions]),
        "warmups": warmups,
        "weights_rng_optimizer_reset_after_warmup": True,
        "timing_boundary": "Loader creation through all updates, telemetry guards, per-update logging and final log flush",
        "schedule_total_updates": total_updates,
        "schedule_max_epochs": settings.max_epochs,
    }


def _reference(job, receipt, result):
    if not job.get("reference"):
        return None
    path = Path(job["reference"]).expanduser().resolve()
    reference = read_json(path)
    if reference.get("status") != "passed" or reference.get("kind") != receipt["kind"]:
        raise ValueError("Reference is not a passed receipt for this trial kind")
    for key in ("encoded_identity", "initial_weights_sha256"):
        if reference.get(key) != receipt.get(key):
            raise ValueError(f"Reference {key} differs")
    if reference["result"]["window_indices_sha256"] != result["window_indices_sha256"]:
        raise ValueError("Reference windows differ")
    result["reference_sha256"] = sha256_file(path)
    result["reference_path"] = str(path)
    return reference


def _parity(job, config, encoded, device, sampler, receipt, output):
    sampler.phase("parity")
    model = _new_model(config, device, receipt, dropout=0.0).train()
    dataset = encoded.dataset("train")
    indices = _indices(dataset, 32, config.training.seed)
    iterator = iter(_loader(dataset, config, device, indices))
    try:
        batches = _group(iterator, config.training.accumulation_steps, sampler.guard)
    finally:
        _close_iterator(iterator)
    target_count = sum(int(batch["labels"].ne(-100).sum()) for batch in batches)
    if not target_count:
        raise ValueError("Parity sample has no scored targets")
    model.zero_grad(set_to_none=True)
    total_loss = torch.zeros((), device=device)
    for batch in batches:
        sampler.guard()
        inputs, labels = batch["input_ids"].to(device), batch["labels"].to(device)
        logits = model(inputs)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite parity loss")
        (loss / target_count).backward()
        total_loss += loss.detach()
    synchronize(device)
    sampler.guard()
    gradients = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
            raise FloatingPointError(f"Missing or nonfinite parity gradient: {name}")
        gradients[name] = parameter.grad.detach().cpu().contiguous().clone()
    path = Path(job["workspace"]).expanduser().resolve() / "parity" / f"{output.stem}.safetensors"
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(gradients, str(path))
    result = {
        "passed": True,
        "gradient_path": str(path),
        "gradient_sha256": sha256_file(path),
        "reference_sha256": None,
        "loss_nll": float(total_loss.cpu()) / target_count,
        "target_tokens": target_count,
        "window_indices_sha256": canonical_hash(indices),
        "dropout": 0.0,
        "optimizer_step": False,
        "tolerance": dict(_TOLERANCE),
        "max_abs_gradient_error": 0.0,
        "relative_gradient_l2_error": 0.0,
    }
    receipt["result"] = result
    reference = _reference(job, receipt, result)
    if reference is not None:
        reference_result = reference["result"]
        reference_path = Path(reference_result["gradient_path"])
        if sha256_file(reference_path) != reference_result["gradient_sha256"]:
            raise ValueError("Reference gradient artifact changed")
        expected = load_file(str(reference_path), device="cpu")
        if set(gradients) != set(expected):
            raise ValueError("Reference parameter names differ")
        max_error, squared_error, squared_reference = 0.0, 0.0, 0.0
        failures = []
        for name, actual in gradients.items():
            difference = actual.double() - expected[name].double()
            max_error = max(max_error, float(difference.abs().max()))
            squared_error += float(difference.square().sum())
            squared_reference += float(expected[name].double().square().sum())
            try:
                torch.testing.assert_close(actual, expected[name], **_TOLERANCE)
            except AssertionError as error:
                failures.append({"parameter": name, "message": str(error)})
        result["max_abs_gradient_error"] = max_error
        result["relative_gradient_l2_error"] = math.sqrt(
            squared_error / max(squared_reference, 1e-30)
        )
        try:
            torch.testing.assert_close(
                result["loss_nll"], reference_result["loss_nll"], **_TOLERANCE
            )
        except AssertionError as error:
            failures.append({"parameter": "loss_nll", "message": str(error)})
        result["passed"] = not failures
        result["failures"] = failures
        if failures:
            raise AssertionError(f"Gradient/loss parity failed for {len(failures)} comparisons")
    del model, gradients
    return result


@torch.inference_mode()
def _validation(job, config, encoded, device, sampler, receipt, output):
    model = _new_model(config, device, receipt).eval()
    dataset = encoded.dataset("validation")
    if len(dataset) < 128:
        raise ValueError("Validation timing requires 128 validation windows")
    indices = list(range(128))
    batch_size = int(
        job.get("validation_batch_size")
        or config.training.validation_batch_size
        or config.training.batch_size
    )
    warmups = int(job.get("warmups", 3))
    if batch_size < 1 or warmups < 1 or warmups * batch_size > 128:
        raise ValueError("Invalid validation warmup size")

    def batch_loss(batch):
        sampler.guard()
        labels = batch["labels"].to(device)
        logits = model(batch["input_ids"].to(device))
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite validation loss")
        return float(loss.cpu()), int(batch["labels"].ne(-100).sum())

    sampler.phase("validation_warmup")
    iterator = iter(
        _loader(dataset, config, device, indices[: warmups * batch_size], batch_size=batch_size)
    )
    try:
        for batch in iterator:
            batch_loss(batch)
    finally:
        _close_iterator(iterator)
    synchronize(device)
    sampler.guard()
    sampler.phase("validation_measured")
    started = time.perf_counter()
    total_loss, total_tokens, input_wait = 0.0, 0, 0.0
    iterator = iter(_loader(dataset, config, device, indices, batch_size=batch_size))
    try:
        while True:
            input_started = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                break
            input_wait += time.perf_counter() - input_started
            loss, tokens = batch_loss(batch)
            total_loss += loss
            total_tokens += tokens
        synchronize(device)
        elapsed = time.perf_counter() - started
        sampler.guard()
    finally:
        sampler.phase("validation_cleanup")
        _close_iterator(iterator)
    if not total_tokens:
        raise ValueError("Validation sample has no scored targets")
    result = {
        "loss_nll": total_loss / total_tokens,
        "loss_sum": total_loss,
        "target_tokens": total_tokens,
        "wall_seconds": elapsed,
        "tokens_per_second": total_tokens / elapsed,
        "batch_size": batch_size,
        "windows": 128,
        "warmups": warmups,
        "input_wait_seconds": input_wait,
        "window_indices_sha256": canonical_hash(indices),
        "reference_sha256": None,
        "tolerance": dict(_TOLERANCE),
        "passed": True,
    }
    receipt["result"] = result
    reference = _reference(job, receipt, result)
    if reference is not None:
        expected = reference["result"]["loss_nll"]
        result["nll_abs_error"] = abs(result["loss_nll"] - expected)
        try:
            torch.testing.assert_close(result["loss_nll"], expected, **_TOLERANCE)
        except AssertionError:
            result["passed"] = False
            raise
    return result


def _isolated_run(job, config):
    run = create_run(job["root"], config, runs_directory=Path(job["workspace"]) / "trials")
    bindings = read_json(job["bindings_path"])
    # Runtime/previous-run bindings cannot be copied into a new immutable run.
    for name in ("raw_data", "prepared_data", "tokenizer", "encoded"):
        if name in bindings:
            run.bind(name, bindings[name])
    return run


def _runtime(job, config, encoded, device, sampler, receipt, output):
    run = _isolated_run(job, config)
    receipt["result"] = {"run_path": str(run.path)}
    sampler.guard()
    model = _new_model(config, device, receipt)
    sampler.phase("runtime_gate")
    result = validate_runtime(run, model, encoded)
    sampler.guard()
    del model
    return {"run_path": str(run.path), "runtime_validation": result}


def _epoch(job, config, encoded, device, sampler, receipt, output):
    run = _isolated_run(job, config)
    receipt["result"] = {"run_path": str(run.path)}
    sampler.guard()
    model = _new_model(config, device, receipt)
    sampler.phase("epoch_runtime_gate")
    validate_runtime(run, model, encoded)
    sampler.guard()
    sampler.phase("epoch_training")
    started = time.perf_counter()
    progress = train_model(
        run, encoded, model=model, resume=False, stop_after_epoch=1, guard=sampler.guard
    )
    synchronize(device)
    elapsed = time.perf_counter() - started
    sampler.phase("epoch_cleanup")
    result = {"run_path": str(run.path), "progress": progress, "wall_seconds": elapsed}
    performance_path = progress.get("performance_receipt")
    if performance_path:
        path = _inside(run.path / performance_path, run.path)
        result["performance_path"] = str(path)
        result["performance"] = read_json(path)
    else:
        raise RuntimeError("One-epoch training did not return its performance receipt")
    del model
    return result


def run_job(job: dict) -> dict:
    """Run one explicitly dispatched trial and write a JSON success/failure receipt.

    ``reference`` is a successful same-kind receipt path for parity/validation.
    Outputs and raw telemetry are confined to ``workspace``. Failures return a
    failed receipt instead of silently lowering precision, context, or batch size.
    """
    workspace = Path(job["workspace"]).expanduser().resolve()
    root = Path(job["root"]).expanduser().resolve()
    if not workspace.is_relative_to(root) or workspace == root:
        raise ValueError("Tuning workspace must be a subdirectory of the project")
    output = _inside(job["output"], workspace)
    reserved = [output, output.with_suffix(".telemetry.jsonl"), output.with_suffix("")]
    if job.get("kind") == "parity":
        reserved.append(workspace / "parity" / f"{output.stem}.safetensors")
    if any(path.exists() for path in reserved):
        raise FileExistsError("A trial artifact already exists; select a fresh output/job name")
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    receipt = {
        "schema": 1,
        "kind": job.get("kind"),
        "status": "running",
        "block": job.get("block", 0),
        "config": job.get("config"),
        "started_at": _utc(),
        "output": str(output),
    }
    sampler = None
    try:
        config = ExperimentConfig.from_dict(job["config"])
        if config.training.batch_size * config.training.accumulation_steps != 32:
            raise ValueError("Every tuning trial must preserve effective batch size 32")
        if config.training.seed != 42:
            raise ValueError("The paired tuning protocol uses seed 42")
        if torch.get_default_dtype() != torch.float32:
            raise ValueError("Tuning requires the registered FP32 default dtype")
        validate_environment(config.training.prefer_metal)
        receipt["execution"] = configure_execution(config.training)
        device = select_device()
        if device.type != "mps":
            raise RuntimeError("This tuning protocol requires MPS; CPU/CUDA timings are prohibited")
        receipt["device"] = str(device)
        receipt["runtime"] = runtime_identity()
        sampler = TelemetrySampler(output.with_suffix(".telemetry.jsonl"), device)
        sampler.start()
        bindings_path = Path(job["bindings_path"]).expanduser().resolve()
        context_path = output.with_suffix("") / "context"
        write_json(context_path / "config" / "bindings.json", read_json(bindings_path))
        context = SimpleNamespace(root=root, config=config, path=context_path)
        encoded = load_encoded_data(context)
        receipt["encoded_identity"] = encoded.identity
        sampler.guard()
        kind = job["kind"]
        if kind in {"epoch", "runtime"}:
            # validate_runtime/train_model each own the project's accelerator lease.
            handler = _epoch if kind == "epoch" else _runtime
            receipt["result"] = handler(job, config, encoded, device, sampler, receipt, output)
        else:
            handlers = {"screen": _screen, "parity": _parity, "validation": _validation}
            if kind not in handlers:
                raise ValueError(f"Unknown tuning job kind: {kind}")
            with accelerator_lease(root):
                receipt["result"] = handlers[kind](
                    job, config, encoded, device, sampler, receipt, output
                )
        sampler.guard()
        receipt["status"] = "passed"
    except BaseException as error:
        message = str(error)
        memory_pressure = isinstance(error, TuningGuardError) or "out of memory" in message.lower()
        receipt["status"] = "guard_stopped" if memory_pressure else "failed"
        receipt["error"] = {
            "type": type(error).__name__,
            "message": message,
            "memory_pressure": memory_pressure,
            "traceback": traceback.format_exc(),
        }
    finally:
        if sampler is not None:
            try:
                sampler.close()
            except Exception as error:
                receipt["status"] = "failed"
                receipt["cleanup_error"] = f"{type(error).__name__}: {error}"
            receipt["telemetry"] = sampler.summary()
        # No live model or optimizer escapes a handler. Reclaim the released
        # allocations only after the telemetry thread has stopped querying them.
        try:
            release_resources()
        except Exception as error:
            receipt["cleanup_error"] = f"{type(error).__name__}: {error}"
            receipt["status"] = "failed"
        receipt["wall_seconds"] = time.perf_counter() - started
        receipt["finished_at"] = _utc()
        write_json(output, receipt)
    return receipt
