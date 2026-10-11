"""CPU-only tuning safety checks; all device/system probes are mocked."""

import json
import plistlib
import subprocess
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src import tune_controller, tuning
from src.config import ExperimentConfig, TrainingConfig


@pytest.fixture(autouse=True)
def prohibit_model_or_accelerator_work(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("A tuning unit test attempted model/device/data work")

    monkeypatch.setattr(tuning, "_new_model", forbidden)
    monkeypatch.setattr(tuning, "load_encoded_data", forbidden)
    monkeypatch.setattr(tuning, "release_resources", lambda: None)
    for flag in (
        "PYTORCH_MPS_PREFER_METAL",
        "PYTORCH_ENABLE_MPS_FALLBACK",
        "PYTORCH_MPS_FAST_MATH",
    ):
        monkeypatch.setenv(flag, "0")


@pytest.fixture
def sampled_hardware(tmp_path, monkeypatch):
    state = {
        "available": 8 * 1024**3,
        "swap": 123 * 1024**2,
        "pressure": 1,
        "utilization": 30.0,
        "queries": 0,
    }
    monkeypatch.setattr(
        tuning.psutil, "virtual_memory", lambda: SimpleNamespace(available=state["available"])
    )
    monkeypatch.setattr(tuning.psutil, "swap_memory", lambda: SimpleNamespace(used=state["swap"]))
    monkeypatch.setattr(
        tuning,
        "memory_snapshot",
        lambda device: {
            "rss_bytes": 100,
            "allocated_bytes": 60,
            "driver_allocated_bytes": 80,
        },
    )

    def metrics():
        state["queries"] += 1
        return {
            "device_utilization_percent": state["utilization"],
            "memory_pressure_level": state["pressure"],
        }

    monkeypatch.setattr(tuning, "_mac_metrics", metrics)
    sampler = tuning.TelemetrySampler(tmp_path / "telemetry.jsonl", torch.device("cpu"))
    # Synchronous samples make consecutive-pressure checks deterministic and
    # avoid starting a real sampler thread in the test process.
    sampler._handle = sampler.path.open("x", encoding="utf-8")
    try:
        yield sampler, state
    finally:
        sampler.close()


@pytest.mark.parametrize(
    ("available", "swap_growth", "stopped"),
    [
        (2 * 1024**3, 512 * 1024**2, False),
        (2 * 1024**3 - 1, 0, False),
        (256 * 1024**2, 0, False),
        (8 * 1024**3, 512 * 1024**2 + 1, True),
    ],
)
def test_available_memory_is_observation_only_and_swap_boundary_is_exact(
    sampled_hardware, available, swap_growth, stopped
):
    sampler, state = sampled_hardware
    state.update(available=available, swap=state["swap"] + swap_growth)
    sampler._sample()
    if stopped:
        with pytest.raises(tuning.TuningGuardError):
            sampler.guard()
    else:
        sampler.guard()
    assert sampler.rows[-1]["swap_growth_bytes"] == swap_growth


def test_critical_pressure_requires_three_consecutive_samples(sampled_hardware):
    sampler, state = sampled_hardware
    for pressure in (4, 4, 1, 4, 4):
        state["pressure"] = pressure
        sampler._sample()
        sampler.guard()
    sampler._sample()
    with pytest.raises(tuning.TuningGuardError, match="three samples"):
        sampler.guard()


def test_missing_pressure_counter_stops_before_model_work(sampled_hardware):
    sampler, state = sampled_hardware
    state["pressure"] = None
    sampler._sample()
    with pytest.raises(tuning.TuningGuardError, match="unavailable"):
        sampler.guard()
    assert sampler.summary()["pressure_query_available"] is False


def test_guard_is_cached_and_failure_stays_latched(sampled_hardware):
    sampler, state = sampled_hardware
    state["swap"] += 513 * 1024**2
    sampler._sample()
    state["swap"] -= 513 * 1024**2
    sampler._sample()
    queries = state["queries"]
    for _ in range(5):
        with pytest.raises(tuning.TuningGuardError, match="512 MiB"):
            sampler.guard()
    assert state["queries"] == queries


def test_utilization_excludes_warmup_and_labels_sampled_memory(sampled_hardware):
    sampler, state = sampled_hardware
    state["utilization"] = 100.0
    sampler.phase("screen_warmup")
    sampler._sample()
    sampler.phase("screen_measured")
    for utilization in (20.0, 40.0):
        state["utilization"] = utilization
        sampler._sample()
    summary = sampler.summary()
    assert summary["guard_policy"]["available_memory_floor_bytes"] is None
    assert summary["guard_policy"]["maximum_swap_growth_bytes"] == 512 * 1024**2
    assert summary["sample_count"] == 3
    assert summary["measured_sample_count"] == 2
    assert summary["mean_device_utilization_percent"] == 30.0
    assert summary["sampled_max_device_utilization_percent"] == 40.0
    assert summary["sampled_max_rss_bytes"] == 100
    assert summary["sampled_max_allocated_bytes"] == 60
    assert "whole device" in summary["utilization_scope"]
    assert "not allocator peaks" in summary["maxima_scope"]
    assert "never sum" in summary["unified_memory_note"]


@pytest.mark.parametrize("pressure_available", [True, False])
def test_mac_probe_parses_nested_plist_and_reports_missing_counter(monkeypatch, pressure_available):
    monkeypatch.setattr(tuning.sys, "platform", "darwin")
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command[0].endswith("ioreg"):
            payload = [{"children": [{"PerformanceStatistics": {"Device Utilization %": 73}}]}]
            return SimpleNamespace(stdout=plistlib.dumps(payload))
        if not pressure_available:
            raise subprocess.CalledProcessError(1, command)
        return SimpleNamespace(stdout="4\n")

    monkeypatch.setattr(tuning.subprocess, "run", run)
    result = tuning._mac_metrics()
    assert result["device_utilization_percent"] == 73.0
    assert result["memory_pressure_level"] == (4 if pressure_available else None)
    assert ("pressure_query_error" in result) is not pressure_available
    assert len(calls) == 2


def worker_job(tmp_path):
    root = tmp_path / "project"
    workspace = root / "benchmarks" / "training-tuning"
    return {
        "kind": "screen",
        "root": str(root),
        "workspace": str(workspace),
        "output": str(workspace / "results" / "case.json"),
        "bindings_path": str(workspace / "preparation" / "config" / "bindings.json"),
        "config": ExperimentConfig(training=TrainingConfig(prefer_metal=False)).to_dict(),
    }


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_worker_refuses_non_mps_before_data_or_model_work(tmp_path, monkeypatch, device):
    job = worker_job(tmp_path)
    monkeypatch.setattr(tuning, "validate_environment", lambda settings: {})
    monkeypatch.setattr(tuning, "configure_execution", lambda settings: {})
    monkeypatch.setattr(tuning, "select_device", lambda: torch.device(device))
    receipt = tuning.run_job(job)
    assert receipt["status"] == "failed"
    assert "requires MPS" in receipt["error"]["message"]
    assert json.loads(Path(job["output"]).read_text())["status"] == "failed"
    assert not Path(job["output"]).with_suffix(".telemetry.jsonl").exists()


def test_worker_never_overwrites_an_existing_receipt(tmp_path):
    job = worker_job(tmp_path)
    output = Path(job["output"])
    output.parent.mkdir(parents=True)
    original = b'{"status":"passed","reference":"immutable"}\n'
    output.write_bytes(original)
    with pytest.raises(FileExistsError):
        tuning.run_job(job)
    assert output.read_bytes() == original


@pytest.fixture
def controller(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "src").mkdir()
    (root / "src" / "config.py").write_text("# Inert source-identity fixture.\n")
    bindings = root / "benchmarks" / "training-tuning" / "preparation" / "config" / "bindings.json"
    bindings.parent.mkdir(parents=True)
    config = ExperimentConfig(training=TrainingConfig(prefer_metal=False)).to_dict()
    bindings.write_text(
        json.dumps({"encoded": {"cache_id": "fixture"}, "prepared_data": {"cache_id": "fixture"}})
    )
    metadata = root / "data" / "prepared" / "fixture" / "metadata.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(json.dumps({"cache_id": "fixture", "config": config["data"]}))
    return tune_controller.TuningController(root, config)


@pytest.mark.parametrize("legacy_training_files", [20_000, 5_000])
def test_controller_rejects_preserved_population_before_any_worker(
    controller, monkeypatch, legacy_training_files
):
    metadata_path = controller.root / "data" / "prepared" / "fixture" / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["config"]["training_files"] = legacy_training_files
    metadata["config"].pop("validation_files", None)
    metadata_path.write_text(json.dumps(metadata))

    def forbidden(*args, **kwargs):
        pytest.fail("Mismatched prepared population must fail before launching any worker")

    monkeypatch.setattr(tune_controller.TuningController, "job", forbidden)
    monkeypatch.setattr(tune_controller.subprocess, "run", forbidden)
    with pytest.raises(RuntimeError, match="Prepared-data configuration does not match"):
        tune_controller.execute_tuning(controller.root)
    assert not (controller.workspace / "jobs").exists()
    assert not (controller.workspace / "protocol.json").exists()


def test_legacy_preparation_missing_validation_count_matches_explicit_all_validation(controller):
    metadata_path = controller.root / "data" / "prepared" / "fixture" / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["config"].pop("validation_files", None)
    metadata_path.write_text(json.dumps(metadata))
    requested = deepcopy(controller.baseline)
    requested["data"]["validation_files"] = None
    restored = tune_controller.TuningController(controller.root, requested)
    assert restored.baseline["data"]["validation_files"] is None


@pytest.mark.parametrize("epochs", [15, 7])
def test_tuning_protocol_records_configured_schedule_without_dispatching_jobs(
    controller, monkeypatch, epochs
):
    config = deepcopy(controller.baseline)
    config["training"]["max_epochs"] = epochs
    controller = tune_controller.TuningController(controller.root, config)

    def stop_before_screen(*args, **kwargs):
        raise RuntimeError("Stop mocked screen before any job")

    monkeypatch.setattr(controller, "screen_group", stop_before_screen)
    with pytest.raises(RuntimeError, match="Stop mocked screen"):
        controller.execute()
    protocol = json.loads((controller.workspace / "protocol.json").read_text())
    assert protocol["scientific_epochs"] == epochs
    assert protocol["session"]["baseline"]["training"]["max_epochs"] == epochs
    assert protocol["full_confirmation_epochs_per_condition"] == 1
    assert not (controller.workspace / "jobs").exists()


def candidates(controller, sizes=(2, 4, 8, 16)):
    return [
        (
            f"b{size}",
            tune_controller._changed(
                controller.baseline, batch_size=size, accumulation_steps=32 // size
            ),
        )
        for size in sizes
    ]


def passed(size):
    return {
        "status": "passed",
        "result": {
            "tokens_per_second": float(size),
            "wall_seconds": 1.0,
            "input_wait_seconds": 0.0,
        },
        "telemetry": {"mean_device_utilization_percent": 100.0 / size},
    }


def test_controller_qualifies_runtime_before_parity_and_rotates_measurement_order(
    controller, monkeypatch
):
    calls = []

    def job(kind, config, label, block=0, reference=None):
        size = config["training"]["batch_size"]
        calls.append((kind, size, block, reference))
        return passed(size), f"/{kind}-b{size}-{block}.json"

    monkeypatch.setattr(controller, "job", job)
    selected, rows = controller.screen_group(
        "batch", candidates(controller, (2, 4, 8)), memory_ladder=True
    )
    assert [(kind, size) for kind, size, _, _ in calls[:6]] == [
        ("runtime", 2),
        ("parity", 2),
        ("runtime", 4),
        ("parity", 4),
        ("runtime", 8),
        ("parity", 8),
    ]
    assert calls[1][3] is None
    assert calls[3][3] == calls[5][3] == "/parity-b2-0.json"
    assert [(size, block) for kind, size, block, _ in calls if kind == "screen"] == [
        (2, 0),
        (4, 0),
        (8, 0),
        (4, 1),
        (8, 1),
        (2, 1),
        (8, 2),
        (2, 2),
        (4, 2),
    ]
    # The fastest candidate wins even though its mocked utilization is lower.
    assert selected["training"]["batch_size"] == 8
    assert len(rows) == 3


@pytest.mark.parametrize("failure_stage", ["runtime", "parity"])
def test_memory_ladder_does_not_qualify_larger_sizes_after_failure(
    controller, monkeypatch, failure_stage
):
    calls = []

    def job(kind, config, label, block=0, reference=None):
        size = config["training"]["batch_size"]
        calls.append((kind, size))
        receipt = passed(size)
        if size == 4 and kind == failure_stage:
            receipt = {"status": "guard_stopped", "error": {"memory_pressure": True}}
        return receipt, f"/{kind}-{size}.json"

    monkeypatch.setattr(controller, "job", job)
    selected, _ = controller.screen_group("batch", candidates(controller), memory_ladder=True)
    assert selected["training"]["batch_size"] == 2
    failure = calls.index((failure_stage, 4))
    assert all(size == 2 for _, size in calls[failure + 1 :])
    assert not any(size > 4 for _, size in calls)


def test_memory_failure_during_screen_blocks_same_and_larger_batches(controller, monkeypatch):
    calls = []

    def job(kind, config, label, block=0, reference=None):
        size = config["training"]["batch_size"]
        calls.append((kind, size, block))
        receipt = passed(size)
        if kind == "screen" and size == 4:
            receipt = {"status": "failed", "error": {"message": "MPS backend out of memory"}}
        return receipt, f"/{kind}-{size}-{block}.json"

    monkeypatch.setattr(controller, "job", job)
    selected, _ = controller.screen_group("batch", candidates(controller), memory_ladder=True)
    failure = calls.index(("screen", 4, 0))
    assert all(size < 4 for _, size, _ in calls[failure + 1 :])
    assert selected["training"]["batch_size"] == 2


def test_validation_memory_ladder_stops_at_first_unsafe_size(controller, monkeypatch):
    calls = []

    def job(kind, config, label, block=0, reference=None):
        size = config["training"]["validation_batch_size"]
        calls.append((size, reference))
        if size == 4:
            return {"status": "guard_stopped", "error": {"memory_pressure": True}}, "/unsafe.json"
        return passed(size), f"/validation-{size}.json"

    monkeypatch.setattr(controller, "job", job)
    selected = controller.validation_group(controller.baseline)
    assert [size for size, _ in calls] == [2, 4, 2, 2, 2]
    assert calls[0][1] is None
    assert all(reference == "/validation-2.json" for _, reference in calls[1:])
    assert selected["training"]["validation_batch_size"] == 2


def test_explicit_memory_failure_flag_is_authoritative():
    assert tune_controller.TuningController.memory_failure(
        {
            "status": "failed",
            "error": {"memory_pressure": True, "message": "OS telemetry unavailable"},
        }
    )


def test_controller_refuses_to_relabel_stale_passed_receipt_after_worker_failure(
    controller, monkeypatch
):
    calls = []

    def run(command, **kwargs):
        job = json.loads(Path(command[-1]).read_text())
        calls.append(job)
        if len(calls) == 1:
            Path(job["output"]).parent.mkdir(parents=True, exist_ok=True)
            Path(job["output"]).write_text(json.dumps(passed(2)))
            return SimpleNamespace(returncode=0)
        # The worker refuses an existing artifact and leaves it untouched.
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(tune_controller.subprocess, "run", run)
    _, output = controller.job("runtime", controller.baseline, "fixture")
    stale = json.loads(Path(output).read_text())
    stale["job_identity"] = "unrelated-previous-job"
    Path(output).write_text(json.dumps(stale))
    try:
        receipt, _ = controller.job("runtime", controller.baseline, "fixture")
    except (FileExistsError, ValueError, RuntimeError):
        pass
    else:
        assert receipt["status"] != "passed", (
            "A failed worker must not bless a stale passed receipt"
        )
    assert json.loads(Path(output).read_text())["job_identity"] == "unrelated-previous-job"


@pytest.mark.parametrize("returncode", [-9, -11, 1])
def test_abnormal_worker_exit_prevents_larger_trials_without_claiming_oom(returncode):
    receipt = {
        "status": "failed",
        "returncode": returncode,
        "error": {"type": "WorkerFailure", "message": ""},
    }
    assert tune_controller.TuningController.memory_failure(receipt)
    assert "memory_pressure" not in receipt["error"]
