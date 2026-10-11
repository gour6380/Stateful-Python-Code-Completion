"""Selected execution-profile propagation without a model, GPU or live server."""

import os
import queue
import threading
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from src import checkpoints, runs, runtime, service
from src.config import ExperimentConfig


@pytest.fixture(autouse=True)
def controlled_environment(monkeypatch):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setenv("PYTORCH_MPS_PREFER_METAL", "1")
    monkeypatch.setenv("PYTORCH_MPS_FAST_MATH", "0")
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")

    def forbidden(*args, **kwargs):
        pytest.fail("Service-profile tests must not execute accelerator operations")

    monkeypatch.setattr(runtime, "synchronize", forbidden)
    monkeypatch.setattr(torch.mps, "synchronize", forbidden)
    monkeypatch.setattr(torch.cuda, "synchronize", forbidden)
    yield
    torch.set_num_threads(previous_threads)


@pytest.fixture
def run(tmp_path):
    config = ExperimentConfig()
    return SimpleNamespace(
        root=tmp_path,
        path=tmp_path / "runs" / "run_001",
        config=replace(config, training=replace(config.training, prefer_metal=True, cpu_threads=3)),
        verify=lambda: None,
    )


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("PYTORCH_MPS_PREFER_METAL", "0"),
        ("PYTORCH_MPS_FAST_MATH", "1"),
        ("PYTORCH_ENABLE_MPS_FALLBACK", "1"),
        ("PYTORCH_MPS_PREFER_METAL", "invalid"),
    ],
)
def test_start_service_rejects_wrong_flags_before_spawn_without_changing_them(
    run, monkeypatch, flag, value
):
    monkeypatch.setenv(flag, value)

    def forbid_spawn(*args, **kwargs):
        pytest.fail("Invalid execution settings must fail before process creation")

    monkeypatch.setattr(service.mp, "get_context", forbid_spawn)
    with pytest.raises((RuntimeError, ValueError)):
        service.start_service(run)
    assert os.environ[flag] == value
    assert torch.get_num_threads() == 1


def test_parent_spawns_with_matching_environment_without_changing_parent_threads(run, monkeypatch):
    events = []
    ready = queue.Queue()
    ready.put({"url": "http://127.0.0.1:8000", "startup_seconds": 0.1, "identity": {}})
    stop = threading.Event()

    class FakeProcess:
        def start(self):
            events.append(("start", os.environ["PYTORCH_MPS_PREFER_METAL"]))

    process = FakeProcess()

    def make_process(*, target, args):
        assert target is service._serve
        assert args == (str(run.root), run.path.name, ready, stop)
        return process

    context = SimpleNamespace(Queue=lambda: ready, Event=lambda: stop, Process=make_process)
    monkeypatch.setattr(
        service.mp, "get_context", lambda method: context if method == "spawn" else None
    )
    handle = service.start_service(run)
    assert handle.process is process
    assert events == [("start", "1")]
    assert torch.get_num_threads() == 1


def install_child_stubs(run, monkeypatch):
    monkeypatch.setattr(runs, "open_run", lambda root, name: run)
    monkeypatch.setattr(runtime, "accelerator_lease", lambda root: nullcontext())
    return queue.Queue(), threading.Event()


def test_child_applies_configured_threads_before_inference_loading(run, monkeypatch):
    ready, stop = install_child_stubs(run, monkeypatch)
    observed = []

    def stop_at_model_boundary(actual_run):
        observed.append(
            (actual_run, torch.get_num_threads(), os.environ["PYTORCH_MPS_PREFER_METAL"])
        )
        raise RuntimeError("Stop fixture before any model work")

    monkeypatch.setattr(checkpoints, "load_inference", stop_at_model_boundary)
    service._serve(str(run.root), run.path.name, ready, stop)
    assert observed == [(run, 3, "1")]
    assert "Stop fixture before any model work" in ready.get_nowait()["error"]


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("PYTORCH_MPS_PREFER_METAL", "0"),
        ("PYTORCH_MPS_FAST_MATH", "1"),
        ("PYTORCH_ENABLE_MPS_FALLBACK", "1"),
    ],
)
def test_child_rechecks_environment_before_thread_change_or_model_work(
    run, monkeypatch, flag, value
):
    ready, stop = install_child_stubs(run, monkeypatch)
    monkeypatch.setenv(flag, value)

    def forbidden(*args, **kwargs):
        pytest.fail("Mismatched child settings must fail before inference loading")

    monkeypatch.setattr(checkpoints, "load_inference", forbidden)
    service._serve(str(run.root), run.path.name, ready, stop)
    error = ready.get_nowait()["error"]
    assert "restart" in error or "no override" in error
    assert os.environ[flag] == value
    assert torch.get_num_threads() == 1
