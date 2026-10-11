"""CPU correctness tests for training, recovery, execution profiles, and run integrity."""

from __future__ import annotations

import copy
import json
import os
import random
import subprocess
import sys
import threading
from dataclasses import replace
from functools import wraps
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src import bootstrap, checkpoints, runs, runtime, tokenizer, train
from src.config import DataConfig, ExperimentConfig, ModelConfig, ServiceConfig, TrainingConfig
from src.model import CodeTransformer
from src.runs import create_run, open_run, read_json, write_json
from src.tracking import MetricLog
from src.train import (
    _new_progress,
    epoch_order,
    learning_rate_for_update,
    make_loader,
    optimizer_groups,
    register_validation,
    scheduler_state,
    train_update,
    training_status,
    validate_scheduler_state,
    validation_nll,
)


@pytest.fixture(autouse=True)
def cpu_only_execution(monkeypatch):
    """These unit tests must never initialize the laptop accelerator."""
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)

    @wraps(torch.backends.mps.is_available)
    def no_mps():
        return False

    monkeypatch.setattr(torch.backends.mps, "is_available", no_mps)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    for name in (
        "PYTORCH_MPS_FAST_MATH",
        "PYTORCH_ENABLE_MPS_FALLBACK",
        "PYTORCH_MPS_PREFER_METAL",
    ):
        monkeypatch.delenv(name, raising=False)
    yield
    torch.set_num_threads(previous_threads)


@pytest.fixture
def small_config():
    return ExperimentConfig(
        data=DataConfig(training_files=2, validation_files=2, vocab_size=260),
        model=ModelConfig(
            vocab_size=260, layers=1, width=16, heads=2, ff_width=32, context_length=16, dropout=0.2
        ),
        training=TrainingConfig(
            max_epochs=5,
            min_epochs=2,
            batch_size=2,
            accumulation_steps=2,
            validation_batch_size=None,
            cpu_threads=1,
            prefer_metal=False,
        ),
        service=ServiceConfig(budgets=(8, 16), sink_tokens=2),
    )


@pytest.fixture
def project(tmp_path, monkeypatch):
    root = tmp_path / "project"
    (root / "src").mkdir(parents=True)
    (root / "src" / "config.py").write_text("VERSION = 1\n")
    (root / "requirements.txt").write_text("example==1.0\n")
    monkeypatch.setattr(runs, "environment_identity", lambda: {"python": "test", "packages": {}})
    monkeypatch.setattr(checkpoints, "runtime_identity", lambda: {"device": "cpu"})
    return root


def encoded_stub(tmp_path):
    directory = tmp_path / "tokenizer"
    directory.mkdir(exist_ok=True)
    write_json(directory / "tokenizer.json", {"test_fixture": True})
    identity = {"vocab_size": 260, "sha256": runs.sha256_file(directory / "tokenizer.json")}
    write_json(directory / "metadata.json", {"identity": identity})
    return SimpleNamespace(
        identity={"encoded": "fixed-test-fixture"},
        tokenizer=SimpleNamespace(path=directory, identity=identity),
    )


def batch(start=4, count=2):
    values = torch.arange(start, start + count * 6).reshape(count, 6)
    return {"input_ids": values, "labels": values.roll(-1, 1)}


def test_config_roundtrip_and_frozen_fields(small_config):
    restored = ExperimentConfig.from_dict(json.loads(json.dumps(small_config.to_dict())))
    assert restored == small_config
    with pytest.raises(AttributeError):
        restored.training.seed = 4


def test_pilot_defaults_preserve_selected_execution_profile_and_optimizer_recipe():
    data = DataConfig()
    assert (data.training_files, data.validation_files) == (5000, 1000)
    config = TrainingConfig()
    assert (config.batch_size, config.accumulation_steps) == (4, 8)
    assert config.batch_size * config.accumulation_steps == 32
    assert config.validation_batch_size == 16
    assert config.cpu_threads == 6 and config.num_workers == 0 and config.prefer_metal is True
    assert (config.seed, config.min_epochs, config.max_epochs, config.patience) == (42, 5, 15, 3)
    assert config.learning_rate == 3e-4 and config.min_learning_rate == 3e-5
    assert config.warmup_fraction == 0.05 and config.betas == (0.9, 0.95)
    assert config.weight_decay == 0.1 and config.gradient_clip == 1.0


@pytest.mark.parametrize(
    "kwargs", [{"width": 15, "heads": 2}, {"dropout": 1.0}, {"context_length": 0}]
)
def test_invalid_model_semantics_rejected(kwargs):
    with pytest.raises(ValueError):
        ModelConfig(**kwargs)


def test_config_rejects_vocabulary_and_context_disagreement(small_config):
    with pytest.raises(ValueError):
        replace(small_config, data=replace(small_config.data, vocab_size=261))
    with pytest.raises(ValueError):
        replace(small_config, service=replace(small_config.service, budgets=(32,)))


def test_numbered_runs_do_not_reuse_gaps_or_overwrite(project, small_config):
    first = create_run(project, small_config)
    (project / "runs" / "run_009").mkdir()
    second = create_run(project, small_config)
    assert first.path.name == "run_001"
    assert second.path.name == "run_010"
    assert open_run(project, first.path.name).identity == first.identity


def test_bindings_are_immutable_and_persist(project, small_config):
    run = create_run(project, small_config)
    run.bind("encoded", {"hash": "a"})
    run.bind("encoded", {"hash": "a"})
    with pytest.raises(ValueError, match="Immutable"):
        run.bind("encoded", {"hash": "b"})
    assert read_json(run.path / "config" / "bindings.json")["encoded"] == {"hash": "a"}


def test_current_source_changes_block_run_reuse(project, small_config):
    run = create_run(project, small_config)
    (project / "src" / "config.py").write_text("VERSION = 2\n")
    with pytest.raises(ValueError, match="Source or dependency"):
        run.verify()


def test_snapshot_changes_are_detected(project, small_config):
    run = create_run(project, small_config)
    (run.path / "source_snapshot" / "src" / "config.py").write_text("modified\n")
    with pytest.raises(ValueError, match="Saved source"):
        run.verify()


def test_live_notebook_outputs_can_change_but_snapshot_cannot(project, small_config):
    notebooks = project / "notebooks"
    notebooks.mkdir()
    notebook = notebooks / "study.ipynb"
    write_json(notebook, {"cells": [{"source": ["x = 1"], "outputs": []}]})
    run = create_run(project, small_config)
    write_json(notebook, {"cells": [{"source": ["x = 1"], "outputs": [{"text": "1"}]}]})
    run.verify()
    write_json(run.path / "source_snapshot" / "notebooks" / notebook.name, {"changed": True})
    with pytest.raises(ValueError, match="Saved notebook"):
        run.verify()


def test_environment_changes_block_reuse(project, small_config, monkeypatch):
    run = create_run(project, small_config)
    monkeypatch.setattr(runs, "environment_identity", lambda: {"different": True})
    with pytest.raises(ValueError, match="Environment changed"):
        run.verify()


def test_writer_lock_is_reentrant_but_excludes_other_threads(project, small_config):
    run = create_run(project, small_config)
    results = []

    def contender():
        try:
            with run.writer_lock():
                results.append("entered")
        except RuntimeError:
            results.append("excluded")

    with run.writer_lock(), run.writer_lock():
        run.bind("nested", {"value": 1})
        worker = threading.Thread(target=contender)
        worker.start()
        worker.join(timeout=5)
        assert not worker.is_alive()
    assert results == ["excluded"]
    contender()
    assert results[-1] == "entered"


def test_accelerator_lease_shared_across_callers(project):
    outcomes = []
    with runtime.accelerator_lease(project):

        def contender():
            try:
                with runtime.accelerator_lease(project):
                    outcomes.append("entered")
            except RuntimeError:
                outcomes.append("excluded")

        worker = threading.Thread(target=contender)
        worker.start()
        worker.join(timeout=5)
    assert outcomes == ["excluded"]


def test_rng_roundtrip_python_numpy_torch():
    runtime.seed_everything(73)
    state = runtime.capture_rng()
    expected = (random.random(), np.random.rand(4), torch.rand(4))
    runtime.restore_rng(state)
    assert random.random() == expected[0]
    np.testing.assert_array_equal(np.random.rand(4), expected[1])
    torch.testing.assert_close(torch.rand(4), expected[2], rtol=0, atol=0)


def test_loader_and_epoch_order_do_not_consume_dropout_rng():
    before = torch.get_rng_state().clone()
    order = epoch_order(5, 42, 3)
    loader = make_loader(
        [batch(count=1) for _ in range(5)], batch_size=2, device="cpu", indices=order
    )
    list(loader)
    torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
    assert order == epoch_order(5, 42, 3)


def test_learning_rate_schedule_uses_update_cursor(small_config):
    config = replace(small_config.training, warmup_fraction=0.2)
    rates = [learning_rate_for_update(config, update, 10) for update in range(10)]
    assert rates[0] == config.learning_rate / 2
    assert rates[1] == config.learning_rate
    assert rates[-1] == config.min_learning_rate
    assert rates[4:] == sorted(rates[4:], reverse=True)


def test_adamw_matrix_decay_excludes_normalization_even_with_zero_gradients(small_config):
    model = CodeTransformer(small_config.model)
    groups = optimizer_groups(model, 0.1)
    parameters = list(model.parameters())
    assert len({id(parameter) for group in groups for parameter in group["params"]}) == len(
        parameters
    )
    assert sum(len(group["params"]) for group in groups) == len(parameters)
    assert all(parameter.ndim >= 2 for parameter in groups[0]["params"])
    assert all(parameter.ndim < 2 for parameter in groups[1]["params"])
    assert any("norm" in name for name in groups[1]["param_names"])
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    optimizer = torch.optim.AdamW(groups, lr=0.01, weight_decay=0.0)
    for parameter in parameters:
        parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    for name, parameter in model.named_parameters():
        if parameter.ndim < 2:
            torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
        else:
            torch.testing.assert_close(parameter, before[name] * 0.999, rtol=1e-6, atol=1e-8)
            assert not torch.equal(parameter, before[name])


def test_scheduler_resume_rejects_budget_cursor_formula_and_config_changes(small_config):
    config = small_config.training
    saved = scheduler_state(config, completed_updates=3, total_updates=10)
    validate_scheduler_state(saved, config, 3, 10)
    assert saved["last_learning_rate"] == learning_rate_for_update(config, 2, 10)
    assert saved["next_learning_rate"] == learning_rate_for_update(config, 3, 10)
    for candidate, active, cursor, budget in [
        (saved, config, 4, 10),
        (saved, config, 3, 11),
        (saved, replace(config, learning_rate=0.0004), 3, 10),
        (saved | {"schedule": "changed"}, config, 3, 10),
    ]:
        with pytest.raises(ValueError, match="scheduler"):
            validate_scheduler_state(candidate, active, cursor, budget)
    assert scheduler_state(config, 10, 10)["next_learning_rate"] is None


def test_validation_checkpoint_ties_and_inclusive_patience_threshold(small_config):
    config = replace(small_config.training, min_delta=0.25)
    progress = _new_progress()
    assert register_validation(progress, 2.0, 1, config)
    assert not register_validation(progress, 2.0, 2, config)
    assert progress["best_epoch"] == 1 and progress["bad_epochs"] == 1
    assert register_validation(progress, 1.9, 3, config)
    assert progress["best_epoch"] == 3 and progress["bad_epochs"] == 2
    # Exactly min_delta relative to the patience anchor resets patience.
    assert register_validation(progress, 1.75, 4, config)
    assert progress["bad_epochs"] == 0
    assert progress["patience_anchor_nll"] == 1.75


def test_zero_delta_ties_still_consume_patience_and_minimum_epochs_hold():
    config = TrainingConfig(min_delta=0.0, min_epochs=5, patience=3, max_epochs=20)
    progress = _new_progress()
    register_validation(progress, 1.0, 1, config)
    for epoch in (2, 3, 4):
        register_validation(progress, 1.0, epoch, config)
        assert progress["status"] == "running"
    register_validation(progress, 1.0, 5, config)
    assert progress["status"] == "complete"
    assert progress["stop_reason"] == "early_stopping"
    assert progress["best_epoch"] == 1


def test_scheduler_state_is_bound_inside_recovery_payload(project, small_config, tmp_path):
    run = create_run(project, small_config)
    encoded = encoded_stub(tmp_path)
    model = CodeTransformer(small_config.model)
    optimizer = torch.optim.AdamW(optimizer_groups(model, 0.1), lr=0.001, weight_decay=0.0)
    progress = _new_progress() | {"updates": 2, "next_batch": 4}
    schedule = scheduler_state(small_config.training, 2, 10)
    checkpoints.save_recovery(
        run,
        encoded,
        model,
        optimizer,
        progress,
        {"updates": 2, "epochs": 0},
        scheduler_state=schedule,
    )
    pointer, payload = checkpoints.read_commit(run, encoded)
    assert payload["scheduler_state"] == pointer["scheduler_state"] == schedule
    with pytest.raises(ValueError, match="Scheduler cursor"):
        checkpoints.save_recovery(
            run,
            encoded,
            model,
            optimizer,
            progress,
            {"updates": 2, "epochs": 0},
            scheduler_state=scheduler_state(small_config.training, 3, 10),
        )


class TokenLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(260, 7)
        self.head = torch.nn.Linear(7, 260)

    def forward(self, ids):
        return self.head(self.embedding(ids))


def test_unequal_microbatches_use_token_denominator_and_one_update():
    torch.manual_seed(11)
    whole = TokenLinear()
    accumulated = copy.deepcopy(whole)
    first, second = batch(4, 2), batch(20, 1)
    first["labels"][0, 3:] = -100
    second["labels"][0, 1:] = -100
    combined = {key: torch.cat((first[key], second[key])) for key in first}
    a = torch.optim.SGD(whole.parameters(), lr=0.01)
    b = torch.optim.SGD(accumulated.parameters(), lr=0.01)
    full = train_update(whole, a, [combined], torch.device("cpu"), 10.0)
    parts = train_update(accumulated, b, [first, second], torch.device("cpu"), 10.0)
    assert full["target_tokens"] == parts["target_tokens"] == 10
    for left, right in zip(whole.parameters(), accumulated.parameters(), strict=True):
        torch.testing.assert_close(left, right, rtol=1e-6, atol=1e-7)


def test_nonfinite_loss_fails_before_optimizer_step(monkeypatch):
    model = TokenLinear()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    with torch.no_grad():
        model.head.weight.fill_(float("nan"))
    monkeypatch.setattr(optimizer, "step", lambda: pytest.fail("Optimizer must not run"))
    with pytest.raises(FloatingPointError, match="loss"):
        train_update(model, optimizer, [batch()], torch.device("cpu"), 1.0)


def test_metric_recovery_truncates_only_uncommitted_suffix(tmp_path):
    with MetricLog(tmp_path) as log:
        log.append("updates", {"update": 1, "nll": 2.0})
        log.flush()
        log.append("updates", {"update": 2, "nll": 1.0})
    with (tmp_path / "updates.jsonl").open("a") as handle:
        handle.write('{"partial":')
    with MetricLog(tmp_path, {"updates": 1, "epochs": 0}) as restored:
        assert restored.counts == {"updates": 1, "epochs": 0}
        assert restored.rows["updates"] == [{"update": 1, "nll": 2.0}]
    assert len((tmp_path / "updates.jsonl").read_text().splitlines()) == 1
    assert "2,1.0" not in (tmp_path / "updates.csv").read_text()


def test_missing_committed_metrics_are_not_invented(tmp_path):
    with pytest.raises(ValueError, match="missing"):
        MetricLog(tmp_path, {"updates": 1, "epochs": 0})


def test_committed_metric_value_changes_are_detected(tmp_path):
    with MetricLog(tmp_path) as log:
        log.append("updates", {"update": 1, "nll": 2.0})
        log.flush()
        hashes = log.hashes
    (tmp_path / "updates.jsonl").write_text('{"update": 1, "nll": 0.1}\n')
    with pytest.raises(ValueError, match="measurements changed"):
        MetricLog(tmp_path, {"updates": 1, "epochs": 0}, hashes)


def test_checkpoint_pointer_failure_preserves_previous_commit(
    project, small_config, tmp_path, monkeypatch
):
    run = create_run(project, small_config)
    encoded = encoded_stub(tmp_path)
    model = CodeTransformer(small_config.model)
    optimizer = torch.optim.AdamW(model.parameters())
    progress = _new_progress()
    first = checkpoints.save_recovery(
        run, encoded, model, optimizer, progress, {"updates": 0, "epochs": 0}
    )
    original_write = checkpoints.write_json

    def fail_pointer(path, value):
        if Path(path).name == "committed.json":
            raise OSError("simulated pointer publication failure")
        original_write(path, value)

    monkeypatch.setattr(checkpoints, "write_json", fail_pointer)
    progress["updates"] = 1
    with pytest.raises(OSError):
        checkpoints.save_recovery(
            run, encoded, model, optimizer, progress, {"updates": 1, "epochs": 0}
        )
    pointer, payload = checkpoints.read_commit(run, encoded)
    assert pointer == first
    assert payload["progress"]["updates"] == 0


def test_committed_weight_tampering_is_detected(project, small_config, tmp_path):
    run = create_run(project, small_config)
    encoded = encoded_stub(tmp_path)
    model = CodeTransformer(small_config.model)
    pointer = checkpoints.save_recovery(
        run,
        encoded,
        model,
        torch.optim.AdamW(model.parameters()),
        _new_progress(),
        {"updates": 0, "epochs": 0},
    )
    path = run.path / "checkpoints" / "recovery" / pointer["checkpoint"]
    with path.open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="changed"):
        checkpoints.read_commit(run, encoded)


def test_cpu_update_boundary_resume_reproduces_dropout_and_optimizer(
    project, small_config, tmp_path
):
    run = create_run(project, small_config)
    encoded = encoded_stub(tmp_path)
    runtime.seed_everything(81)
    uninterrupted = CodeTransformer(small_config.model)
    optimizer = torch.optim.AdamW(uninterrupted.parameters(), lr=0.001)
    samples = [batch(4), batch(22), batch(40), batch(60)]
    for sample in samples[:2]:
        train_update(uninterrupted, optimizer, [sample], torch.device("cpu"), 1.0)
    progress = _new_progress() | {"updates": 2, "next_batch": 4}
    checkpoints.save_recovery(
        run, encoded, uninterrupted, optimizer, progress, {"updates": 2, "epochs": 0}
    )
    for sample in samples[2:]:
        train_update(uninterrupted, optimizer, [sample], torch.device("cpu"), 1.0)
    _, payload = checkpoints.read_commit(run, encoded)
    resumed = CodeTransformer(small_config.model)
    resumed.load_state_dict(payload["model"])
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=0.001)
    resumed_optimizer.load_state_dict(payload["optimizer"])
    runtime.restore_rng(payload["rng"])
    for sample in samples[2:]:
        train_update(resumed, resumed_optimizer, [sample], torch.device("cpu"), 1.0)
    for expected, actual in zip(uninterrupted.parameters(), resumed.parameters(), strict=True):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    assert resumed_optimizer.state_dict()["param_groups"] == optimizer.state_dict()["param_groups"]


def test_freeze_requires_completion_and_exports_reconstructible_weights(
    project, small_config, tmp_path, monkeypatch
):
    run = create_run(project, small_config)
    encoded = encoded_stub(tmp_path)
    monkeypatch.setattr(checkpoints, "require_runtime_validation", lambda *_: {"status": "passed"})
    model = CodeTransformer(small_config.model)
    optimizer = torch.optim.AdamW(model.parameters())
    progress = _new_progress()
    checkpoints.save_recovery(run, encoded, model, optimizer, progress, {"updates": 0, "epochs": 0})
    with pytest.raises(RuntimeError, match="finish"):
        checkpoints.freeze_checkpoint(run, encoded)
    progress.update(
        epoch=5,
        updates=7,
        status="complete",
        best_epoch=5,
        best_validation_nll=1.2,
        stop_reason="max_epochs",
    )
    checkpoints.save_recovery(
        run, encoded, model, optimizer, progress, {"updates": 7, "epochs": 5}, is_best=True
    )
    frozen = checkpoints.freeze_checkpoint(run, encoded)
    assert frozen["evaluation_protocol"]
    assert frozen["selected"] == "best"
    assert checkpoints.freeze_checkpoint(run, encoded) == frozen
    monkeypatch.setattr(tokenizer, "load_tokenizer", lambda _: encoded.tokenizer)
    restored, _, identity = checkpoints.load_inference(run, device="cpu")
    assert identity == frozen["exports"]["best"]["identity"]
    model.eval()
    inputs = batch()["input_ids"]
    with torch.inference_mode():
        torch.testing.assert_close(restored(inputs), model(inputs), rtol=0, atol=0)
    export = run.path / frozen["exports"]["best"]["directory"]
    assert (export / "model.safetensors").is_file()
    assert not list(export.glob("*.pt"))


def test_frozen_export_tampering_is_detected(project, small_config, tmp_path, monkeypatch):
    run = create_run(project, small_config)
    encoded = encoded_stub(tmp_path)
    monkeypatch.setattr(checkpoints, "require_runtime_validation", lambda *_: {})
    model = CodeTransformer(small_config.model)
    progress = _new_progress() | {
        "epoch": 5,
        "updates": 7,
        "status": "complete",
        "best_epoch": 5,
        "best_validation_nll": 1.0,
        "stop_reason": "max_epochs",
    }
    checkpoints.save_recovery(
        run,
        encoded,
        model,
        torch.optim.AdamW(model.parameters()),
        progress,
        {"updates": 7, "epochs": 5},
        is_best=True,
    )
    frozen = checkpoints.freeze_checkpoint(run, encoded)
    path = run.path / frozen["exports"]["best"]["directory"] / "architecture.json"
    write_json(path, {"changed": True})
    with pytest.raises(ValueError, match="export changed"):
        checkpoints.verify_frozen(run)


def test_status_remains_bound_to_committed_progress(project, small_config, tmp_path):
    run = create_run(project, small_config)
    assert training_status(run)["status"] == "not_started"
    encoded = encoded_stub(tmp_path)
    model = CodeTransformer(small_config.model)
    checkpoints.save_recovery(
        run,
        encoded,
        model,
        torch.optim.AdamW(model.parameters()),
        _new_progress(),
        {"updates": 0, "epochs": 0},
    )
    write_json(run.path / "checkpoints" / "status.json", {"updates": 99, "status": "running"})
    assert training_status(run)["updates"] == 0


def test_runtime_gate_receipt_cannot_be_fabricated_by_status_only(project, small_config, tmp_path):
    run = create_run(project, small_config)
    encoded = encoded_stub(tmp_path)
    write_json(run.path / "environment" / "runtime_validation.json", {"status": "passed"})
    with pytest.raises(ValueError, match="receipt"):
        runtime.require_runtime_validation(run, encoded)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"validation_batch_size": 0},
        {"validation_batch_size": True},
        {"cpu_threads": -1},
        {"cpu_threads": 1.5},
        {"num_workers": -1},
        {"num_workers": True},
        {"prefer_metal": 1},
    ],
)
def test_execution_profile_rejects_invalid_options(kwargs):
    with pytest.raises(ValueError):
        TrainingConfig(**kwargs)


def test_execution_profile_roundtrip_and_runtime_identity(small_config):
    config = replace(
        small_config,
        training=replace(
            small_config.training, validation_batch_size=7, cpu_threads=1, num_workers=2
        ),
    )
    assert ExperimentConfig.from_dict(config.to_dict()) == config
    actual = runtime.configure_execution(config.training)
    assert actual["actual_cpu_threads"] == 1
    assert actual["validation_batch_size"] == 7
    assert actual["num_workers"] == 2 and actual["worker_cpu_threads"] == 1
    identity = runtime.runtime_identity()
    assert identity["cpu_threads"] == 1 and identity["mps_prefer_metal"] == "0"
    fallback = TrainingConfig(
        batch_size=2, validation_batch_size=None, cpu_threads=1, prefer_metal=False
    )
    assert runtime.configure_execution(fallback)["validation_batch_size"] == 2


def test_bootstrap_is_standard_library_only_and_configures_before_torch():
    script = "from src.bootstrap import configure_environment; import sys; assert 'torch' not in sys.modules; assert configure_environment(True)['prefer_metal']; assert 'torch' not in sys.modules"
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        env=dict(os.environ),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("flag", ["PYTORCH_MPS_FAST_MATH", "PYTORCH_ENABLE_MPS_FALLBACK"])
def test_unsafe_environment_is_rejected_without_silent_override(monkeypatch, flag):
    monkeypatch.setenv(flag, "1")
    with pytest.raises(RuntimeError, match="no override"):
        bootstrap.configure_environment(False)
    with pytest.raises(RuntimeError, match="no override"):
        runtime.configure_execution(TrainingConfig(cpu_threads=1))
    assert os.environ[flag] == "1"


def test_metal_preference_requires_restart_if_torch_already_imported(monkeypatch):
    assert bootstrap.configure_environment(False)["prefer_metal"] is False
    with pytest.raises(RuntimeError, match="restart"):
        bootstrap.configure_environment(True)
    with pytest.raises(RuntimeError, match="restart"):
        runtime.configure_execution(TrainingConfig(prefer_metal=True))
    assert os.environ["PYTORCH_MPS_PREFER_METAL"] == "0"


def test_custom_runs_directory_is_bounded_and_identity_distinct(project, small_config, tmp_path):
    normal = create_run(project, small_config)
    benchmark = create_run(project, small_config, runs_directory="benchmarks/profile/runs")
    assert normal.path.name == benchmark.path.name == "run_001"
    assert normal.identity != benchmark.identity
    assert open_run(project, "run_001", runs_directory="benchmarks/profile/runs") == benchmark
    for outside in (project, "..", tmp_path / "outside"):
        with pytest.raises(ValueError, match="bounded"):
            create_run(project, small_config, runs_directory=outside)
    (project / "escape").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="bounded"):
        open_run(project, "run_001", runs_directory="escape")


def test_loader_workers_are_cpu_single_thread_and_preserve_sample_order():
    loader = make_loader(list(range(5)), batch_size=2, device="cpu", num_workers=1)
    loader.timeout = 15
    assert loader.num_workers == 1
    assert loader.worker_init_fn is train._initialize_worker
    assert loader.multiprocessing_context.get_start_method() == "spawn"
    assert torch.cat(list(loader)).tolist() == list(range(5))
    train._initialize_worker(0)
    assert torch.get_num_threads() == 1


def test_update_guard_checks_each_microbatch_and_before_optimizer_step(monkeypatch):
    model = TokenLinear()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    checks = []
    original_step = optimizer.step

    def step():
        assert len(checks) == 3
        return original_step()

    monkeypatch.setattr(optimizer, "step", step)
    train_update(
        model, optimizer, [batch(4), batch(20)], "cpu", 1.0, guard=lambda: checks.append("checked")
    )
    assert len(checks) == 3


def test_update_guard_can_cancel_before_any_optimizer_mutation(monkeypatch):
    model = TokenLinear()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    before = copy.deepcopy(model.state_dict())
    calls = 0

    def guard():
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("memory limit")

    monkeypatch.setattr(
        optimizer, "step", lambda: pytest.fail("No optimizer step after guard failure")
    )
    with pytest.raises(RuntimeError, match="memory limit"):
        train_update(model, optimizer, [batch(4), batch(20)], "cpu", 1.0, guard=guard)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def _training_encoded(tmp_path):
    encoded = encoded_stub(tmp_path)
    rows = [
        {key: value.squeeze(0) for key, value in batch(4 + index * 6, count=1).items()}
        for index in range(5)
    ]
    encoded.dataset = lambda split: rows
    return encoded


def test_validation_batch_option_and_guard_cover_full_split(tmp_path):
    encoded = _training_encoded(tmp_path)
    checks = []
    expected = validation_nll(
        TokenLinear(),
        encoded.dataset("validation"),
        batch_size=3,
        device="cpu",
        guard=lambda: checks.append(True),
    )
    assert len(checks) == 2 and expected["target_tokens"] == 30
    assert expected["input_wait_seconds"] >= 0


def test_epoch_limit_resumes_original_schedule_and_records_inclusive_timing(
    project, small_config, tmp_path, monkeypatch
):
    config = replace(
        small_config,
        training=replace(
            small_config.training,
            max_epochs=15,
            min_epochs=5,
            patience=20,
            cpu_threads=1,
            validation_batch_size=3,
            checkpoint_interval=1,
        ),
    )
    encoded = _training_encoded(tmp_path)
    monkeypatch.setattr(train, "require_runtime_validation", lambda *_: {})
    run = create_run(project, config, runs_directory="benchmarks/test/runs")
    checks = []
    first = train.train_model(run, encoded, stop_after_epoch=1, guard=lambda: checks.append(True))
    assert first["status"] == "running" and first["epoch"] == 1
    assert first["stop_reason"] is None and first["invocation_stop_reason"] == "epoch_limit"
    assert len(checks) == 7  # Three microbatches, two update boundaries, two validation batches.
    _, first_payload = checkpoints.read_commit(run, encoded)
    assert first_payload["scheduler_state"]["total_updates"] == 30
    assert first_payload["scheduler_state"]["completed_updates"] == 2
    receipt = read_json(run.path / first["performance_receipt"])
    epoch = receipt["epochs"][0]
    assert epoch["completed_epoch"] and epoch["epoch_wall_seconds"] > 0
    assert receipt["initial_checkpoint_seconds"] > 0
    components = sum(
        epoch[key]
        for key in (
            "training_input_wait_seconds",
            "training_update_seconds",
            "validation_seconds",
            "checkpoint_seconds",
        )
    )
    assert epoch["epoch_wall_seconds"] >= components
    assert epoch["checkpoint_seconds"] > 0 and epoch["validation_seconds"] > 0
    assert (
        epoch["input_wait_seconds"]
        == epoch["training_input_wait_seconds"] + epoch["validation_input_wait_seconds"]
    )
    assert (
        receipt["invocation_wall_seconds"]
        >= epoch["epoch_wall_seconds"] + receipt["initial_checkpoint_seconds"]
    )
    with MetricLog(run.path / "logs", first_payload["metrics"], first_payload["metric_hashes"]):
        pass
    unchanged = train.train_model(run, encoded, stop_after_epoch=1)
    assert unchanged["invocation_stop_reason"] == "epoch_limit_already_reached"
    assert unchanged["checkpoint_sha256"] == first["checkpoint_sha256"]
    second = train.train_model(run, encoded, stop_after_epoch=2)
    assert second["status"] == "running" and second["epoch"] == 2
    _, resumed = checkpoints.read_commit(run, encoded)
    assert resumed["scheduler_state"]["total_updates"] == 30
    uninterrupted = create_run(project, config, runs_directory="benchmarks/comparison/runs")
    train.train_model(uninterrupted, encoded, stop_after_epoch=2)
    _, expected = checkpoints.read_commit(uninterrupted, encoded)
    for name, value in expected["model"].items():
        torch.testing.assert_close(resumed["model"][name], value, rtol=0, atol=0)


def test_guard_interruption_records_partial_timing_and_keeps_durable_cursor(
    project, small_config, tmp_path, monkeypatch
):
    run = create_run(
        project, replace(small_config, training=replace(small_config.training, cpu_threads=1))
    )
    encoded = _training_encoded(tmp_path)
    monkeypatch.setattr(train, "require_runtime_validation", lambda *_: {})

    def stop():
        raise RuntimeError("memory limit")

    with pytest.raises(RuntimeError, match="memory limit"):
        train.train_model(run, encoded, stop_after_epoch=1, guard=stop)
    assert training_status(run)["updates"] == 0
    receipt = read_json(next((run.path / "logs" / "performance").glob("*.json")))
    assert receipt["invocation_stop_reason"] == "interrupted"
    assert receipt["epochs"][0]["epoch_wall_seconds"] is None
    assert not receipt["epochs"][0]["completed_epoch"]
    assert receipt["epochs"][0]["training_update_seconds"] > 0
