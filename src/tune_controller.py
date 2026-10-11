"""Explicit local tuning orchestration; importing does not execute experiments.

Call execute_tuning(project_root) only after the prepared-data bindings and
acceptance checks exist. Each candidate uses a fresh native Python process.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path


def _read(path):
    return json.loads(Path(path).read_text())


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _changed(config, **settings):
    result = copy.deepcopy(config)
    result["training"].update(settings)
    assert result["training"]["batch_size"] * result["training"]["accumulation_steps"] == 32
    return result


class TuningController:
    def __init__(self, root, config):
        self.root = Path(root).resolve()
        self.workspace = self.root / "benchmarks" / "training-tuning"
        self.baseline = config
        self.bindings = self.workspace / "preparation" / "config" / "bindings.json"
        if not self.bindings.is_file() or "encoded" not in _read(self.bindings):
            raise RuntimeError("Complete the registered data preparation before tuning")
        from .bootstrap import validate_environment
        from .config import DataConfig
        from .runs import environment_identity, source_files

        bindings = _read(self.bindings)
        try:
            cache_id = bindings["prepared_data"]["cache_id"]
            metadata = _read(self.root / "data" / "prepared" / cache_id / "metadata.json")
            if metadata["cache_id"] != cache_id:
                raise ValueError("Prepared cache identity differs from its binding")
            # Historical preparations retained every eligible validation file.
            # A missing field must not inherit today's capped validation default.
            prepared_config = DataConfig(**{"validation_files": None, **metadata["config"]})
            requested_config = DataConfig(**{"validation_files": None, **config["data"]})
        except (OSError, KeyError, TypeError, ValueError) as error:
            raise RuntimeError(
                "Cannot verify the bound prepared-data configuration before tuning"
            ) from error
        if prepared_config != requested_config:
            raise RuntimeError(
                "Prepared-data configuration does not match the requested DataConfig; "
                "prepare and bind the requested population before starting tuning"
            )
        validate_environment(config["training"]["prefer_metal"])
        self.source_hashes = source_files(self.root)
        self.environment = environment_identity()
        self.session = {
            "source": self.source_hashes,
            "data": bindings,
            "baseline": config,
            "environment": self.environment,
        }
        self.session_hash = _digest(self.session)
        self.failures = []
        self.groups = []
        self.baseline_parity = None
        self.baseline_validation = None

    def job(self, kind, config, label, block=0, reference=None):
        case = {
            "kind": kind,
            "root": str(self.root),
            "workspace": str(self.workspace),
            "bindings_path": str(self.bindings),
            "config": config,
            "block": block,
            "warmups": 3,
            "repetitions": 12,
            "reference": reference,
            "session_hash": self.session_hash,
        }
        name = f"{label}-{kind}-{block}-{_digest(case)[:12]}"
        output = self.workspace / "results" / f"{name}.json"
        jobfile = self.workspace / "jobs" / f"{name}.json"
        case["output"] = str(output)
        _write(jobfile, case)
        if output.exists():
            receipt = _read(output)
            if receipt.get("job_identity") == _digest(case):
                return receipt, str(output)
            raise RuntimeError(f"Existing tuning receipt has a different identity: {output}")
        env = dict(os.environ)
        env["PYTORCH_MPS_PREFER_METAL"] = "1" if config["training"]["prefer_metal"] else "0"
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
        env["PYTORCH_MPS_FAST_MATH"] = "0"
        env["TOKENIZERS_PARALLELISM"] = "false"
        if config["training"]["cpu_threads"] is not None:
            env["OMP_NUM_THREADS"] = str(config["training"]["cpu_threads"])
            env["MKL_NUM_THREADS"] = str(config["training"]["cpu_threads"])
        else:
            env.pop("OMP_NUM_THREADS", None)
            env.pop("MKL_NUM_THREADS", None)
        log = self.workspace / "logs" / f"{name}.log"
        log.parent.mkdir(exist_ok=True)
        print(
            json.dumps(
                {
                    "event": "start",
                    "kind": kind,
                    "label": label,
                    "block": block,
                    "batch": config["training"]["batch_size"],
                    "threads": config["training"]["cpu_threads"],
                    "metal": config["training"]["prefer_metal"],
                }
            ),
            flush=True,
        )
        code = (
            "import json,sys; from src.tuning import run_job; run_job(json.load(open(sys.argv[1])))"
        )
        with log.open("w") as handle:
            process = subprocess.run(
                [sys.executable, "-u", "-c", code, str(jobfile)],
                cwd=self.root,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        if process.returncode != 0 or not output.exists():
            receipt = {
                "status": "failed",
                "kind": kind,
                "config": config,
                "error": {"type": "WorkerFailure", "message": log.read_text()[-4000:]},
                "returncode": process.returncode,
            }
        else:
            receipt = _read(output)
        receipt["job_identity"] = _digest(case)
        receipt["source_hashes"] = self.source_hashes
        _write(output, receipt)
        print(
            json.dumps(
                {
                    "event": "finished",
                    "kind": kind,
                    "label": label,
                    "block": block,
                    "status": receipt["status"],
                    "result": receipt.get("result", {}),
                    "error": receipt.get("error"),
                }
            ),
            flush=True,
        )
        if receipt["status"] != "passed":
            self.failures.append(
                {
                    "label": label,
                    "kind": kind,
                    "receipt": str(output),
                    "error": receipt.get("error"),
                    "status": receipt["status"],
                }
            )
        return receipt, str(output)

    @staticmethod
    def memory_failure(receipt):
        error = str(receipt.get("error", {})).lower()
        return (
            receipt["status"] == "guard_stopped"
            or receipt.get("returncode") not in (None, 0)
            or bool(receipt.get("error", {}).get("memory_pressure"))
            or any(
                token in error for token in ("out of memory", "memory pressure", "swap", "2 gib")
            )
        )

    def qualify(self, config, label):
        runtime_receipt, runtime_path = self.job("runtime", config, label)
        if runtime_receipt["status"] != "passed":
            return runtime_receipt, runtime_path
        receipt, path = self.job("parity", config, label, reference=self.baseline_parity)
        if self.baseline_parity is None and receipt["status"] == "passed":
            self.baseline_parity = path
        return receipt, path

    def screen_group(self, name, candidates, *, memory_ladder=False):
        allowed = []
        for label, cfg in candidates:
            receipt, _ = self.qualify(cfg, f"{name}-{label}")
            if receipt["status"] == "passed":
                allowed.append((label, cfg))
            elif memory_ladder and self.memory_failure(receipt):
                break
        if not allowed:
            raise RuntimeError(f"No qualified candidates in {name}")
        measurements = {label: [] for label, _ in allowed}
        blocked = set()
        for block in range(3):
            ordered = allowed[block % len(allowed) :] + allowed[: block % len(allowed)]
            for label, cfg in ordered:
                if label in blocked:
                    continue
                receipt, path = self.job("screen", cfg, f"{name}-{label}", block)
                if receipt["status"] == "passed":
                    measurements[label].append(
                        {
                            "receipt": path,
                            **receipt["result"],
                            "telemetry": receipt.get("telemetry", {}),
                        }
                    )
                else:
                    blocked.add(label)
                    if memory_ladder and self.memory_failure(receipt):
                        size = cfg["training"]["batch_size"]
                        blocked.update(
                            candidate_label
                            for candidate_label, candidate_config in allowed
                            if candidate_config["training"]["batch_size"] >= size
                        )
        eligible = [(label, cfg) for label, cfg in allowed if len(measurements[label]) == 3]
        if not eligible:
            raise RuntimeError(f"All candidates failed measured screening in {name}")
        scores = {
            label: statistics.median(row["tokens_per_second"] for row in measurements[label])
            for label, _ in eligible
        }
        winner, selected = max(eligible, key=lambda item: scores[item[0]])
        self.groups.append(
            {
                "group": name,
                "winner": winner,
                "config": selected,
                "median_tokens_per_second": scores,
                "measurements": measurements,
            }
        )
        _write(
            self.workspace / "screening_progress.json",
            {"groups": self.groups, "failures": self.failures},
        )
        return selected, measurements[winner]

    def validation_group(self, config):
        candidates = []
        for size in (2, 4, 8, 16):
            cfg = _changed(config, validation_batch_size=size)
            receipt, path = self.job(
                "validation", cfg, f"validation-{size}-qualify", reference=self.baseline_validation
            )
            if receipt["status"] == "passed":
                if self.baseline_validation is None:
                    self.baseline_validation = path
                candidates.append((size, cfg))
            elif self.memory_failure(receipt):
                break
        if not candidates:
            raise RuntimeError("No qualified validation-batch candidate")
        results = {size: [] for size, _ in candidates}
        blocked = set()
        for block in range(3):
            for size, cfg in (
                candidates[block % len(candidates) :] + candidates[: block % len(candidates)]
            ):
                if size in blocked:
                    continue
                receipt, path = self.job(
                    "validation", cfg, f"validation-{size}", block, self.baseline_validation
                )
                if receipt["status"] == "passed":
                    results[size].append({"receipt": path, **receipt["result"]})
                else:
                    blocked.add(size)
                    if self.memory_failure(receipt):
                        blocked.update(n for n, _ in candidates if n >= size)
        eligible = [(size, cfg) for size, cfg in candidates if len(results[size]) == 3]
        if not eligible:
            raise RuntimeError("No valid validation-batch candidate")
        size, selected = max(
            eligible,
            key=lambda item: statistics.median(
                row["tokens_per_second"] for row in results[item[0]]
            ),
        )
        self.groups.append(
            {
                "group": "validation_batch",
                "winner": size,
                "config": selected,
                "measurements": results,
            }
        )
        return selected

    def execute(self):
        _write(
            self.workspace / "protocol.json",
            {
                "session": self.session,
                "session_hash": self.session_hash,
                "screen_warmups": 3,
                "screen_updates": 12,
                "blocks": 3,
                "adoption_minimum_speedup_fraction": 0.05,
                "scientific_epochs": self.baseline["training"]["max_epochs"],
                "full_confirmation_epochs_per_condition": 1,
                "effective_windows_per_update": 32,
                "available_memory_floor_bytes": None,
                "maximum_swap_growth_bytes": 512 * 1024**2,
                "critical_pressure_consecutive_samples": 3,
                "protocol_amendment": "Owner removed the 2 GiB available-memory stop; other guards retained",
            },
        )
        selected, _ = self.screen_group(
            "batch",
            [
                (
                    f"b{batch}",
                    _changed(self.baseline, batch_size=batch, accumulation_steps=32 // batch),
                )
                for batch in (2, 4, 8, 16)
            ],
            memory_ladder=True,
        )
        hardware = _read(self.workspace / "hardware.json")
        default_threads = hardware["default_cpu_threads"]
        threads = list(dict.fromkeys((default_threads, 1, 4, 6)))
        baseline_validation, baseline_validation_path = self.job(
            "validation", self.baseline, "baseline-validation-reference"
        )
        if baseline_validation["status"] != "passed":
            raise RuntimeError("Baseline validation equivalence reference failed")
        self.baseline_validation = baseline_validation_path
        selected, _ = self.screen_group(
            "threads", [(f"t{n}", _changed(selected, cpu_threads=n)) for n in threads]
        )
        selected, winning_rows = self.screen_group(
            "metal",
            [
                ("default", _changed(selected, prefer_metal=False)),
                ("preferred", _changed(selected, prefer_metal=True)),
            ],
        )
        input_fraction = sum(row["input_wait_seconds"] for row in winning_rows) / sum(
            row["wall_seconds"] for row in winning_rows
        )
        if input_fraction > 0.10:
            selected, _ = self.screen_group(
                "workers", [(f"w{n}", _changed(selected, num_workers=n)) for n in (0, 1, 2)]
            )
        else:
            self.groups.append(
                {
                    "group": "workers",
                    "winner": 0,
                    "input_wait_fraction": input_fraction,
                    "reason": "Input wait did not exceed 10%; retain workers0",
                }
            )
        selected = self.validation_group(selected)
        _write(self.workspace / "screened_candidate.json", selected)
        baseline_receipt, baseline_path = self.job("epoch", self.baseline, "baseline-full-epoch")
        if baseline_receipt["status"] != "passed":
            raise RuntimeError("Baseline full epoch failed; no speed claim or default change")
        candidate_receipt, candidate_path = self.job("epoch", selected, "candidate-full-epoch")
        result = {
            "baseline": baseline_receipt,
            "candidate": candidate_receipt,
            "baseline_receipt": baseline_path,
            "candidate_receipt": candidate_path,
            "baseline_config": self.baseline,
            "candidate_config": selected,
            "groups": self.groups,
            "failures": self.failures,
            "source_hashes": self.source_hashes,
            "data_bindings": _read(self.bindings),
        }
        _write(self.workspace / "comparison.json", result)
        return result


def execute_tuning(root):
    from .config import ExperimentConfig

    root = Path(root).resolve()
    baseline = ExperimentConfig().to_dict()
    baseline = _changed(
        baseline,
        batch_size=2,
        accumulation_steps=16,
        validation_batch_size=2,
        cpu_threads=None,
        num_workers=0,
        prefer_metal=False,
    )
    return TuningController(root, baseline).execute()
