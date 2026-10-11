"""Owner-invoked FP32 training with token-weighted accumulation and exact cursors."""

from __future__ import annotations

import math
import time
import uuid
from contextlib import contextmanager
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .checkpoints import read_commit, save_recovery
from .runs import canonical_hash, read_json, write_json
from .runtime import (
    accelerator_lease,
    configure_execution,
    memory_snapshot,
    require_runtime_validation,
    restore_rng,
    seed_everything,
    select_device,
    synchronize,
)
from .tracking import MetricLog, append_log, tensorboard_writer


def learning_rate_for_update(config, update: int, total_updates: int) -> float:
    """Stateless warmup/cosine schedule; cursor stores the next zero-based update."""
    if not 0 <= update < total_updates:
        raise ValueError("Update cursor outside the registered schedule")
    warmup = max(1, int(total_updates * config.warmup_fraction)) if config.warmup_fraction else 0
    if update < warmup:
        return config.learning_rate * (update + 1) / warmup
    denominator = max(1, total_updates - warmup - 1)
    phase = min(1.0, (update - warmup) / denominator)
    return config.min_learning_rate + 0.5 * (config.learning_rate - config.min_learning_rate) * (
        1 + math.cos(math.pi * phase)
    )


def optimizer_groups(model, weight_decay: float) -> list[dict[str, Any]]:
    """Decay matrix parameters only; normalization vectors and biases stay undecayed."""
    decayed, undecayed, decayed_names, undecayed_names = [], [], [], []
    seen = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in seen:
            raise ValueError("Optimizer parameter appears more than once")
        seen.add(id(parameter))
        if parameter.ndim >= 2:
            decayed.append(parameter)
            decayed_names.append(name)
        else:
            undecayed.append(parameter)
            undecayed_names.append(name)
    return [
        {"params": decayed, "param_names": decayed_names, "weight_decay": weight_decay},
        {"params": undecayed, "param_names": undecayed_names, "weight_decay": 0.0},
    ]


def scheduler_state(config, completed_updates: int, total_updates: int) -> dict[str, Any]:
    """Serializable update schedule, including its planned budget and committed cursor."""
    if total_updates < 1 or not 0 <= completed_updates <= total_updates:
        raise ValueError("Invalid scheduler update budget or committed cursor")
    warmup = max(1, int(total_updates * config.warmup_fraction)) if config.warmup_fraction else 0
    return {
        "schema": 1,
        "schedule": "linear_warmup_cosine_v1",
        "total_updates": total_updates,
        "completed_updates": completed_updates,
        "warmup_updates": warmup,
        "formula": {
            "warmup": "lr_max * (zero_based_update + 1) / warmup_updates",
            "cosine": "lr_min + 0.5*(lr_max-lr_min)*(1+cos(pi*min(1,(update-warmup)/max(1,total-warmup-1))))",
        },
        "config": {
            "learning_rate": config.learning_rate,
            "min_learning_rate": config.min_learning_rate,
            "warmup_fraction": config.warmup_fraction,
            "max_epochs": config.max_epochs,
            "batch_size": config.batch_size,
            "accumulation_steps": config.accumulation_steps,
        },
        "last_learning_rate": learning_rate_for_update(config, completed_updates - 1, total_updates)
        if completed_updates
        else None,
        "next_learning_rate": learning_rate_for_update(config, completed_updates, total_updates)
        if completed_updates < total_updates
        else None,
    }


def validate_scheduler_state(
    saved: dict[str, Any] | None, config, completed_updates: int, total_updates: int
) -> None:
    if saved != scheduler_state(config, completed_updates, total_updates):
        raise ValueError("Recovery scheduler budget/configuration/formula/cursor differs")


def register_validation(progress: dict[str, Any], nll: float, epoch: int, config) -> bool:
    """Keep the earliest exact minimum; min_delta controls only patience resets."""
    if not math.isfinite(nll) or not 1 <= epoch <= config.max_epochs:
        raise ValueError("Invalid validation loss or completed epoch")
    is_best = progress["best_validation_nll"] is None or nll < progress["best_validation_nll"]
    if is_best:
        progress["best_validation_nll"], progress["best_epoch"] = nll, epoch
    anchor = progress["patience_anchor_nll"]
    if anchor is None or (nll < anchor and nll <= anchor - config.min_delta):
        progress["patience_anchor_nll"], progress["bad_epochs"] = nll, 0
    else:
        progress["bad_epochs"] += 1
    if epoch >= config.max_epochs:
        progress.update(status="complete", stop_reason="max_epochs")
    elif epoch >= config.min_epochs and progress["bad_epochs"] >= config.patience:
        progress.update(status="complete", stop_reason="early_stopping")
    return is_best


def epoch_order(length: int, seed: int, epoch: int) -> list[int]:
    generator = torch.Generator(device="cpu").manual_seed(seed + epoch)
    return torch.randperm(length, generator=generator).tolist()


def _initialize_worker(_worker_id: int) -> None:
    # Workers only read CPU data. Never initialize an accelerator in a worker.
    torch.set_num_threads(1)


def make_loader(
    dataset, *, batch_size: int, device, indices=None, seed: int = 0, num_workers: int = 0
):
    # Even DataLoader iterator creation receives its own generator and cannot consume
    # model/dropout RNG. Dataset reads and collation must be deterministic.
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=indices,
        num_workers=num_workers,
        worker_init_fn=_initialize_worker if num_workers else None,
        multiprocessing_context="spawn" if num_workers else None,
        pin_memory=torch.device(device).type == "cuda",
        drop_last=False,
        generator=torch.Generator().manual_seed(seed),
    )


def train_update(
    model,
    optimizer,
    batches: list[dict[str, torch.Tensor]],
    device,
    gradient_clip: float,
    *,
    guard: Callable[[], None] | None = None,
) -> dict[str, float | int]:
    """One optimizer update across microbatches, weighted by nonpadding tokens."""
    target_count = sum(int(batch["labels"].ne(-100).sum()) for batch in batches)
    if target_count < 1:
        raise ValueError("An optimizer update must contain scored targets")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_sum = torch.zeros((), device=device, dtype=torch.float32)
    for batch in batches:
        if guard is not None:
            guard()
        inputs = batch["input_ids"].to(device, non_blocking=torch.device(device).type == "cuda")
        labels = batch["labels"].to(device, non_blocking=torch.device(device).type == "cuda")
        logits = model(inputs)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite training loss")
        (loss / target_count).backward()
        loss_sum += loss.detach()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), gradient_clip, error_if_nonfinite=True
    )
    if guard is not None:
        guard()
    optimizer.step()
    finite = [torch.isfinite(parameter).all() for parameter in model.parameters()]
    for state in optimizer.state.values():
        finite.extend(
            torch.isfinite(value).all().to(device)
            for value in state.values()
            if isinstance(value, torch.Tensor)
        )
    if not bool(torch.stack(finite).all()):
        raise FloatingPointError("Nonfinite parameter or optimizer state after update")
    return {
        "loss_sum": float(loss_sum.detach().cpu()),
        "target_tokens": target_count,
        "nll": float(loss_sum.detach().cpu()) / target_count,
        "gradient_norm_before_clip": float(gradient_norm.detach().cpu()),
    }


@torch.inference_mode()
def validation_nll(
    model,
    dataset,
    *,
    batch_size: int,
    device,
    num_workers: int = 0,
    guard: Callable[[], None] | None = None,
) -> dict[str, Any]:
    model.eval()
    total_loss, total_tokens = 0.0, 0
    started = time.perf_counter()
    loader = make_loader(dataset, batch_size=batch_size, device=device, num_workers=num_workers)
    iterator = iter(loader)
    input_wait = time.perf_counter() - started
    for _ in tqdm(range(len(loader)), desc="Full validation", leave=False):
        started = time.perf_counter()
        batch = next(iterator)
        input_wait += time.perf_counter() - started
        if guard is not None:
            guard()
        labels = batch["labels"].to(device)
        logits = model(batch["input_ids"].to(device))
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        count = int(labels.ne(-100).sum().cpu())
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite validation loss")
        total_loss += float(loss.cpu())
        total_tokens += count
    if not total_tokens:
        raise ValueError("Validation split has no scored tokens")
    return {
        "nll": total_loss / total_tokens,
        "loss_sum": total_loss,
        "target_tokens": total_tokens,
        "input_wait_seconds": input_wait,
    }


def training_status(run) -> dict[str, Any]:
    path = run.path / "checkpoints" / "committed.json"
    if not path.exists():
        return {"status": "not_started", "epoch": 0, "updates": 0}
    pointer = read_json(path)
    if pointer.get("pointer_hash") != canonical_hash(
        {key: value for key, value in pointer.items() if key != "pointer_hash"}
    ):
        raise ValueError("Training commit pointer changed")
    if pointer["identity"]["run"] != run.identity:
        raise ValueError("Training status belongs to a different run")
    return dict(pointer["progress"]) | {
        "checkpoint": pointer["checkpoint"],
        "checkpoint_sha256": pointer["sha256"],
    }


def _new_progress() -> dict[str, Any]:
    return {
        "status": "running",
        "epoch": 0,
        "next_batch": 0,
        "updates": 0,
        "epoch_loss_sum": 0.0,
        "epoch_target_tokens": 0,
        "epoch_training_seconds": 0.0,
        "best_epoch": None,
        "best_validation_nll": None,
        "best_checkpoint": None,
        "patience_anchor_nll": None,
        "bad_epochs": 0,
        "stop_reason": None,
    }


@contextmanager
def _measure(record: dict[str, Any], key: str):
    started = time.perf_counter()
    try:
        yield
    finally:
        record[key] += time.perf_counter() - started


def _finish_epoch_timing(timing, started, progress, *, completed: bool) -> None:
    wall = time.perf_counter() - started
    timing.update(
        end_batch=progress["next_batch"],
        end_updates=progress["updates"],
        completed_epoch=completed,
        observed_epoch_segment_wall_seconds=wall,
        epoch_wall_seconds=wall if completed and timing["start_batch"] == 0 else None,
    )
    timing["input_wait_seconds"] = (
        timing["training_input_wait_seconds"] + timing["validation_input_wait_seconds"]
    )
    timing["other_overhead_seconds"] = max(
        0.0,
        wall
        - sum(
            timing[key]
            for key in (
                "training_input_wait_seconds",
                "training_update_seconds",
                "validation_seconds",
                "checkpoint_seconds",
            )
        ),
    )


def train_model(
    run,
    encoded,
    model=None,
    resume: bool = True,
    *,
    stop_after_epoch: int | None = None,
    guard: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Train explicitly. No notebook import, test or run construction starts training."""
    from .model import CodeTransformer

    invocation_started = time.perf_counter()
    run.verify()
    settings = run.config.training
    if stop_after_epoch is not None and (
        type(stop_after_epoch) is not int or not 1 <= stop_after_epoch <= settings.max_epochs
    ):
        raise ValueError("stop_after_epoch must be an absolute epoch in the configured schedule")
    execution = configure_execution(settings)
    require_runtime_validation(run, encoded)
    if (run.path / "evaluation" / "frozen.json").exists():
        raise RuntimeError("Evaluation is frozen; further training in this run is prohibited")
    device = select_device()
    train_data, valid_data = encoded.dataset("train"), encoded.dataset("validation")
    if not len(train_data) or not len(valid_data):
        raise ValueError("Nonempty training and full validation splits are required")
    batches_per_epoch = math.ceil(len(train_data) / settings.batch_size)
    updates_per_epoch = math.ceil(batches_per_epoch / settings.accumulation_steps)
    total_updates = updates_per_epoch * settings.max_epochs
    performance_path = run.path / "logs" / "performance" / f"{uuid.uuid4().hex}.json"
    performance = {
        "schema": 1,
        "run": run.identity,
        "execution": execution,
        "stop_after_epoch": stop_after_epoch,
        "planned_total_updates": total_updates,
        "initial_checkpoint_seconds": 0.0,
        "epochs": [],
        "timing_note": "Wall time includes input, updates, validation, logging, and durable checkpoints. Validation input wait is a subset of validation_seconds. Interrupted/resumed partial epochs have no full epoch_wall_seconds.",
    }

    def finish(reason, *, error=None):
        performance.update(
            invocation_stop_reason=reason,
            invocation_wall_seconds=time.perf_counter() - invocation_started,
        )
        if error is not None:
            performance["error"] = {"type": type(error).__name__, "message": str(error)}
        write_json(performance_path, performance)
        return training_status(run) | {
            "invocation_stop_reason": reason,
            "performance_receipt": str(performance_path.relative_to(run.path)),
        }

    with accelerator_lease(run.root), run.writer_lock():
        run.verify()
        run.bind("encoded", encoded.identity)
        pointer_path = run.path / "checkpoints" / "committed.json"
        if pointer_path.exists() and not resume:
            raise RuntimeError("Run already contains training; choose resume or create a fresh run")
        if model is not None and (
            model.config != run.config.model or next(model.parameters()).dtype != torch.float32
        ):
            raise ValueError("Supplied model must match the run's FP32 architecture")
        seed_everything(settings.seed)
        if torch.get_default_dtype() != torch.float32:
            raise ValueError("The registered scratch initialization requires default float32")
        if model is None:
            with torch.device("cpu"):
                model = CodeTransformer(run.config.model).float()
        elif not pointer_path.exists():
            # The inspection model may have been constructed before seeding. Always
            # install the registered scratch initialization for a new experiment.
            with torch.device("cpu"):
                fresh = CodeTransformer(run.config.model).float()
            model.load_state_dict(fresh.state_dict(), strict=True)
            del fresh
        model.to(device)
        optimizer = torch.optim.AdamW(
            optimizer_groups(model, settings.weight_decay),
            lr=settings.learning_rate,
            betas=settings.betas,
            eps=settings.eps,
            weight_decay=0.0,
        )
        if pointer_path.exists():
            pointer, payload = read_commit(run, encoded)
            validate_scheduler_state(
                payload["scheduler_state"], settings, payload["progress"]["updates"], total_updates
            )
            if payload["progress"]["status"] == "complete":
                return finish("training_complete")
            if stop_after_epoch is not None and payload["progress"]["epoch"] >= stop_after_epoch:
                return finish("epoch_limit_already_reached")
            model.load_state_dict(payload["model"], strict=True)
            optimizer.load_state_dict(payload["optimizer"])
            progress, counts = payload["progress"], payload["metrics"]
            metric_hashes = payload["metric_hashes"]
            if metric_hashes is None:
                raise ValueError("Training recovery requires committed metric integrity hashes")
            resumed_rng = payload["rng"]
            del payload
        else:
            progress, counts = _new_progress(), {"updates": 0, "epochs": 0}
            resumed_rng = None
            metric_hashes = None
        if (
            not 0 <= progress["epoch"] < settings.max_epochs
            or not 0 <= progress["next_batch"] < batches_per_epoch
        ):
            raise ValueError("Invalid committed epoch/minibatch cursor")
        if progress["updates"] != progress["epoch"] * updates_per_epoch + math.ceil(
            progress["next_batch"] / settings.accumulation_steps
        ):
            raise ValueError("Committed update cursor disagrees with the minibatch cursor")
        if progress["next_batch"] % settings.accumulation_steps:
            raise ValueError("Recovery cursor is not an optimizer-update boundary")
        with MetricLog(run.path / "logs", counts, metric_hashes) as metrics:
            writer = tensorboard_writer(run.path / "tensorboard", progress["updates"])
            active_timing = None
            try:
                if resumed_rng is not None:
                    restore_rng(resumed_rng)
                append_log(
                    run.path / "logs",
                    f"Training on {device}; resume={pointer_path.exists()}; committed update={progress['updates']}; epoch={progress['epoch']}; next batch={progress['next_batch']}",
                )
                if not pointer_path.exists():
                    with _measure(performance, "initial_checkpoint_seconds"):
                        metrics.flush()
                        save_recovery(
                            run,
                            encoded,
                            model,
                            optimizer,
                            progress,
                            metrics.counts,
                            metric_hashes=metrics.hashes,
                            scheduler_state=scheduler_state(
                                settings, progress["updates"], total_updates
                            ),
                        )
                while progress["epoch"] < settings.max_epochs:
                    epoch_index = progress["epoch"]
                    epoch_started = time.perf_counter()
                    active_timing = {
                        "epoch": epoch_index + 1,
                        "start_batch": progress["next_batch"],
                        "start_updates": progress["updates"],
                        "training_input_wait_seconds": 0.0,
                        "training_update_seconds": 0.0,
                        "validation_seconds": 0.0,
                        "validation_input_wait_seconds": 0.0,
                        "checkpoint_seconds": 0.0,
                    }
                    performance["epochs"].append(active_timing)
                    order = epoch_order(len(train_data), settings.seed, epoch_index)
                    remaining = order[progress["next_batch"] * settings.batch_size :]
                    with _measure(active_timing, "training_input_wait_seconds"):
                        loader = make_loader(
                            train_data,
                            batch_size=settings.batch_size,
                            device=device,
                            indices=remaining,
                            seed=settings.seed + epoch_index,
                            num_workers=settings.num_workers,
                        )
                        iterator = iter(loader)
                    bar = tqdm(
                        total=batches_per_epoch,
                        initial=progress["next_batch"],
                        desc=f"Epoch {epoch_index + 1}/{settings.max_epochs}",
                        unit="batch",
                    )
                    try:
                        while progress["next_batch"] < batches_per_epoch:
                            group = []
                            for _ in range(
                                min(
                                    settings.accumulation_steps,
                                    batches_per_epoch - progress["next_batch"],
                                )
                            ):
                                with _measure(active_timing, "training_input_wait_seconds"):
                                    group.append(next(iterator))
                            lr = learning_rate_for_update(
                                settings, progress["updates"], total_updates
                            )
                            for parameter_group in optimizer.param_groups:
                                parameter_group["lr"] = lr
                            synchronize(device)
                            started = time.perf_counter()
                            with _measure(active_timing, "training_update_seconds"):
                                row = train_update(
                                    model,
                                    optimizer,
                                    group,
                                    device,
                                    settings.gradient_clip,
                                    guard=guard,
                                )
                                synchronize(device)
                            elapsed = time.perf_counter() - started
                            progress["next_batch"] += len(group)
                            progress["updates"] += 1
                            progress["epoch_loss_sum"] += row["loss_sum"]
                            progress["epoch_target_tokens"] += row["target_tokens"]
                            progress["epoch_training_seconds"] += elapsed
                            metrics.append(
                                "updates",
                                {
                                    "update": progress["updates"],
                                    "epoch": epoch_index + 1,
                                    "next_batch": progress["next_batch"],
                                    "learning_rate": lr,
                                    "seconds": elapsed,
                                    **row,
                                },
                            )
                            writer.add_scalar("train/update_nll", row["nll"], progress["updates"])
                            writer.add_scalar("train/learning_rate", lr, progress["updates"])
                            writer.add_scalar(
                                "train/gradient_norm_before_clip",
                                row["gradient_norm_before_clip"],
                                progress["updates"],
                            )
                            bar.update(len(group))
                            bar.set_postfix(nll=f"{row['nll']:.4f}", lr=f"{lr:.2g}")
                            # The last update in an epoch is committed together with
                            # validation and selection, so it is never ambiguously half-finalized.
                            if (
                                progress["updates"] % settings.checkpoint_interval == 0
                                and progress["next_batch"] < batches_per_epoch
                            ):
                                with _measure(active_timing, "checkpoint_seconds"):
                                    metrics.flush()
                                    writer.flush()
                                    save_recovery(
                                        run,
                                        encoded,
                                        model,
                                        optimizer,
                                        progress,
                                        metrics.counts,
                                        metric_hashes=metrics.hashes,
                                        scheduler_state=scheduler_state(
                                            settings, progress["updates"], total_updates
                                        ),
                                    )
                                append_log(
                                    run.path / "logs",
                                    f"Committed update {progress['updates']}; training NLL={row['nll']:.6f}",
                                )
                            del group
                    finally:
                        bar.close()
                    synchronize(device)
                    with _measure(active_timing, "validation_seconds"):
                        validation = validation_nll(
                            model,
                            valid_data,
                            batch_size=settings.validation_batch_size or settings.batch_size,
                            device=device,
                            num_workers=settings.num_workers,
                            guard=guard,
                        )
                        synchronize(device)
                    validation_seconds = active_timing["validation_seconds"]
                    active_timing["validation_input_wait_seconds"] = validation[
                        "input_wait_seconds"
                    ]
                    nll = validation["nll"]
                    completed_epoch = epoch_index + 1
                    is_best = register_validation(progress, nll, completed_epoch, settings)
                    memory = memory_snapshot(device)
                    epoch_row = {
                        "epoch": completed_epoch,
                        "updates": progress["updates"],
                        "training_nll": progress["epoch_loss_sum"]
                        / progress["epoch_target_tokens"],
                        "training_target_tokens": progress["epoch_target_tokens"],
                        "training_update_seconds": progress["epoch_training_seconds"],
                        "training_tokens_per_update_second": progress["epoch_target_tokens"]
                        / progress["epoch_training_seconds"],
                        "validation_nll": nll,
                        "validation_target_tokens": validation["target_tokens"],
                        "validation_seconds": validation_seconds,
                        "best_epoch": progress["best_epoch"],
                        "bad_epochs": progress["bad_epochs"],
                        "rss_bytes": memory["rss_bytes"],
                        "accelerator_allocated_bytes": memory["allocated_bytes"],
                    }
                    metrics.append("epochs", epoch_row)
                    writer.add_scalar(
                        "epoch/train_nll", epoch_row["training_nll"], progress["updates"]
                    )
                    writer.add_scalar("epoch/validation_nll", nll, progress["updates"])
                    progress.update(
                        epoch=completed_epoch,
                        next_batch=0,
                        epoch_loss_sum=0.0,
                        epoch_target_tokens=0,
                        epoch_training_seconds=0.0,
                    )
                    with _measure(active_timing, "checkpoint_seconds"):
                        metrics.flush()
                        writer.flush()
                        save_recovery(
                            run,
                            encoded,
                            model,
                            optimizer,
                            progress,
                            metrics.counts,
                            is_best=is_best,
                            metric_hashes=metrics.hashes,
                            scheduler_state=scheduler_state(
                                settings, progress["updates"], total_updates
                            ),
                        )
                    append_log(
                        run.path / "logs",
                        f"Epoch {completed_epoch} committed; train NLL={epoch_row['training_nll']:.6f}; validation NLL={nll:.6f}; best epoch={progress['best_epoch']}; status={progress['status']}",
                    )
                    _finish_epoch_timing(active_timing, epoch_started, progress, completed=True)
                    active_timing = None
                    write_json(performance_path, performance)
                    if progress["status"] == "complete" or (
                        stop_after_epoch is not None and completed_epoch >= stop_after_epoch
                    ):
                        break
                run.verify()
                return finish(
                    "training_complete" if progress["status"] == "complete" else "epoch_limit"
                )
            except BaseException as error:
                if active_timing is not None:
                    _finish_epoch_timing(active_timing, epoch_started, progress, completed=False)
                finish("interrupted", error=error)
                # Never commit a possibly partial optimizer step. Resume only the
                # last durable update boundary, replaying uncommitted work.
                write_json(
                    run.path / "logs" / "last_interruption.json",
                    {
                        "exception": type(error).__name__,
                        "message": str(error),
                        "recovery": "last committed update boundary",
                        "attempted_updates": progress["updates"],
                    },
                )
                append_log(
                    run.path / "logs",
                    f"Interrupted: {type(error).__name__}: {error}. Resume uses the last durable commit.",
                )
                raise
            finally:
                writer.close()
