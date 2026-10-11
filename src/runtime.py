"""Explicit device selection and synthetic gates; no runtime work at import."""

from __future__ import annotations

import gc
import os
import platform
import random
import subprocess
import sys
import traceback
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

from .bootstrap import validate_environment
from .runs import canonical_hash, exclusive_lock, read_json, write_json


def configure_execution(settings) -> dict[str, Any]:
    """Apply an explicit CPU thread profile before tensor work; validate MPS flags."""
    flags = validate_environment(settings.prefer_metal)
    if settings.cpu_threads is not None:
        torch.set_num_threads(settings.cpu_threads)
    return {
        "requested_cpu_threads": settings.cpu_threads,
        "actual_cpu_threads": torch.get_num_threads(),
        "actual_interop_threads": torch.get_num_interop_threads(),
        "num_workers": settings.num_workers,
        "worker_cpu_threads": 1 if settings.num_workers else None,
        "validation_batch_size": settings.validation_batch_size or settings.batch_size,
        **flags,
    }


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
            raise RuntimeError("Disable implicit MPS CPU fallback before recording this experiment")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def synchronize(device: torch.device | str) -> None:
    device = torch.device(device)
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def memory_snapshot(device: torch.device | str) -> dict[str, Any]:
    import psutil

    device = torch.device(device)
    result = {"rss_bytes": psutil.Process().memory_info().rss, "device": str(device)}
    if device.type == "mps":
        result.update(
            allocated_bytes=torch.mps.current_allocated_memory(),
            driver_allocated_bytes=torch.mps.driver_allocated_memory(),
        )
    elif device.type == "cuda":
        result.update(
            allocated_bytes=torch.cuda.memory_allocated(device),
            reserved_bytes=torch.cuda.memory_reserved(device),
        )
    else:
        result["allocated_bytes"] = None
    result["measurement"] = "instantaneous; overlapping unified-memory counters must not be summed"
    return result


def runtime_identity() -> dict[str, Any]:
    device = select_device()
    return {
        "device": str(device),
        "torch": str(torch.__version__),
        "machine": platform.machine(),
        "device_name": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else platform.processor(),
        "default_dtype": str(torch.get_default_dtype()),
        "mps_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0"),
        "mps_prefer_metal": os.environ.get("PYTORCH_MPS_PREFER_METAL", "0"),
        "mps_fast_math": os.environ.get("PYTORCH_MPS_FAST_MATH", "0"),
        "cpu_threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "cuda_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def inspect_runtime(run) -> dict[str, Any]:
    run.verify()
    execution = configure_execution(run.config.training)
    result = runtime_identity()
    result["execution"] = execution
    result["memory"] = memory_snapshot(result["device"])
    return result


def capture_rng() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    result = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
        "numpy": [
            numpy_state[0],
            numpy_state[1].tolist(),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4],
        ],
    }
    if torch.backends.mps.is_available():
        result["mps"] = torch.mps.get_rng_state().cpu()
    if torch.cuda.is_available():
        result["cuda"] = [value.cpu() for value in torch.cuda.get_rng_state_all()]
    return result


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    name, keys, position, has_gaussian, cached_gaussian = state["numpy"]
    np.random.set_state(
        (name, np.asarray(keys, dtype=np.uint32), position, has_gaussian, cached_gaussian)
    )
    torch.set_rng_state(state["torch_cpu"].cpu())
    if "mps" in state:
        if not torch.backends.mps.is_available():
            raise RuntimeError("Saved MPS RNG requires MPS")
        torch.mps.set_rng_state(state["mps"].cpu())
    if "cuda" in state:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise RuntimeError("Saved CUDA RNG requires the original CUDA device set")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextmanager
def preserve_rng() -> Iterator[None]:
    state = capture_rng()
    try:
        yield
    finally:
        restore_rng(state)


def validation_identity(run, encoded) -> dict[str, Any]:
    return {
        "run": run.identity,
        "encoded": encoded.identity,
        "runtime": runtime_identity(),
        "execution": {
            "requested_cpu_threads": run.config.training.cpu_threads,
            "num_workers": run.config.training.num_workers,
            "validation_batch_size": run.config.training.validation_batch_size
            or run.config.training.batch_size,
            "prefer_metal": run.config.training.prefer_metal,
        },
    }


def require_runtime_validation(run, encoded) -> dict[str, Any]:
    path = run.path / "environment" / "runtime_validation.json"
    if not path.is_file():
        raise RuntimeError("Execute the explicit synthetic runtime-validation notebook cell first")
    receipt = read_json(path)
    digest = receipt.pop("receipt_hash", None)
    if digest != canonical_hash(receipt) or receipt.get("status") != "passed":
        raise ValueError("Invalid runtime-validation receipt")
    if receipt.get("identity") != validation_identity(run, encoded):
        raise ValueError(
            "Runtime receipt does not match this source/configuration/environment/data/device"
        )
    receipt["receipt_hash"] = digest
    return receipt


def validate_runtime(run, model, encoded) -> dict[str, Any]:
    """Run synthetic numerical checks only when explicitly called by the owner."""
    from .model import CodeTransformer

    run.verify()
    configure_execution(run.config.training)
    if model.config != run.config.model:
        raise ValueError("Runtime gate model configuration differs from the run")
    if (
        next(model.parameters()).dtype != torch.float32
        or torch.get_default_dtype() != torch.float32
    ):
        raise ValueError("The registered runtime gate requires float32 weights and default dtype")
    device = select_device()
    original_device = next(model.parameters()).device
    original_mode = model.training
    tolerance = (
        {"rtol": 3e-4, "atol": 5e-4} if device.type != "cpu" else {"rtol": 2e-5, "atol": 2e-5}
    )
    checks: dict[str, Any] = {}
    with accelerator_lease(run.root), run.writer_lock(), preserve_rng():
        try:
            model.to(device).eval()
            tokens = [1, 17, 23, 31, 47, 59, 61, 2][: run.config.model.context_length]
            tokens = [min(value, run.config.model.vocab_size - 1) for value in tokens]
            split = max(1, len(tokens) // 2)
            inputs = torch.tensor([tokens], device=device, dtype=torch.long)
            with torch.inference_mode():
                reference = model(inputs, reference=True)
                logits = model(inputs)
                torch.testing.assert_close(logits, reference, **tolerance)
                prefix_logits, cache = model.prefill(tokens[:split])
                torch.testing.assert_close(prefix_logits, reference[0, split - 1], **tolerance)
                for index, token in enumerate(tokens[split:], start=split):
                    cached_logits, cache = model.step(token, cache)
                    torch.testing.assert_close(cached_logits, reference[0, index], **tolerance)
                altered = inputs.clone()
                altered[:, split:] = 3
                torch.testing.assert_close(
                    model(altered)[:, :split], logits[:, :split], **tolerance
                )
                checks["configured_model"] = {
                    "reference_max_abs_error": float((logits - reference).abs().max().cpu()),
                    "full_prefill_incremental": "passed",
                    "causal_future_isolation": "passed",
                    "greedy_disagreements": int(
                        (logits.argmax(-1) != reference.argmax(-1)).sum().cpu()
                    ),
                    "note": "Numerical synthetic gate; trained greedy parity has its own later gate",
                }
            tiny_config = replace(
                run.config.model,
                width=32,
                heads=4,
                layers=2,
                ff_width=64,
                context_length=16,
                dropout=0.0,
            )
            seed_everything(917)
            with torch.device("cpu"):
                tiny = CodeTransformer(tiny_config).float().eval()
            cpu_inputs = torch.tensor([tokens], dtype=torch.long)
            cpu_logits = tiny(cpu_inputs, reference=True).detach()
            tiny.to(device)
            output = tiny(cpu_inputs.to(device), reference=True)
            torch.testing.assert_close(output.detach().cpu(), cpu_logits, **tolerance)
            loss = torch.nn.functional.cross_entropy(
                output[:, :-1].reshape(-1, tiny_config.vocab_size),
                cpu_inputs[:, 1:].to(device).reshape(-1),
            )
            loss.backward()
            gradients = [
                parameter.grad for parameter in tiny.parameters() if parameter.requires_grad
            ]
            if not bool(torch.isfinite(loss)) or any(
                gradient is None or not bool(torch.isfinite(gradient).all())
                for gradient in gradients
            ):
                raise FloatingPointError("Nonfinite or missing synthetic gradients")
            optimizer = torch.optim.AdamW(tiny.parameters(), lr=1e-3)
            optimizer.step()
            if any(not bool(torch.isfinite(parameter).all()) for parameter in tiny.parameters()):
                raise FloatingPointError("Nonfinite synthetic optimizer update")
            synchronize(device)
            checks["tiny_cpu_device_forward_backward_update"] = "passed"
            identity = validation_identity(run, encoded)
            run.bind("training_runtime", identity["runtime"])
            run.bind("training_encoded", encoded.identity)
            receipt = {
                "schema": 1,
                "status": "passed",
                "identity": identity,
                "checks": checks,
                "tolerance": tolerance,
                "scope": "Synthetic correctness only; no learned quality or 16GB feasibility conclusion",
            }
            receipt["receipt_hash"] = canonical_hash(receipt)
            write_json(run.path / "environment" / "runtime_validation.json", receipt)
        except BaseException as error:
            failure = {
                "schema": 1,
                "status": "failed",
                "identity": validation_identity(run, encoded),
                "checks_completed": checks,
                "exception": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            }
            write_json(run.path / "environment" / "runtime_validation_failure.json", failure)
            # A failed rerun must not leave a previously passed gate usable.
            write_json(run.path / "environment" / "runtime_validation.json", failure)
            raise
        finally:
            model.to(original_device).train(original_mode)
    return receipt


def accelerator_lease(root: Path | str):
    """One training/service/benchmark owner for this project's accelerator at a time."""
    return exclusive_lock(Path(root) / ".accelerator.lock")


def release_resources(*objects: Any) -> None:
    """Release unreferenced temporaries; callers must first drop their own live references."""
    del objects
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_acceptance_tests(run) -> dict[str, Any]:
    """Future explicit notebook action; authoring/importing never invokes pytest."""
    run.verify()
    output = run.path / "logs" / "acceptance_tests.log"
    junit = run.path / "environment" / "acceptance_tests.xml"
    receipt_path = run.path / "environment" / "acceptance_tests.json"
    identity = {"run": run.identity, "interpreter": sys.executable}
    command = [sys.executable, "-m", "pytest", "tests", "-q", f"--junitxml={junit}"]
    with accelerator_lease(run.root), run.writer_lock():
        write_json(receipt_path, {"status": "running", "identity": identity, "command": command})
        try:
            with output.open("w", encoding="utf-8") as handle:
                completed = subprocess.run(
                    command,
                    cwd=run.root,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    check=False,
                    text=True,
                )
                handle.flush()
                os.fsync(handle.fileno())
            run.verify()
            counts = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
            if junit.is_file():
                tree = ET.parse(junit)
                suites = list(tree.getroot().iter("testsuite"))
                for key in counts:
                    counts[key] = sum(int(suite.attrib.get(key, 0)) for suite in suites)
            passed = (
                completed.returncode == 0
                and counts["tests"] > 0
                and not counts["failures"]
                and not counts["errors"]
            )
            receipt = {
                "schema": 1,
                "status": "passed" if passed else "failed",
                "identity": identity,
                "returncode": completed.returncode,
                "counts": counts,
                "console_log": str(output.relative_to(run.path)),
                "junit": str(junit.relative_to(run.path)),
            }
            receipt["receipt_hash"] = canonical_hash(receipt)
            write_json(receipt_path, receipt)
            return receipt
        except BaseException as error:
            write_json(
                receipt_path,
                {
                    "status": "failed",
                    "identity": identity,
                    "exception": type(error).__name__,
                    "message": str(error),
                },
            )
            raise
