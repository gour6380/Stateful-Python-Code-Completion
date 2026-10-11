"""Frozen, prediction-only evaluation. Generated Python is never executed."""

from __future__ import annotations

import csv
import json
import os
import traceback
import uuid
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .cache import CachePolicy
from .checkpoints import load_inference
from .data import stable_hash
from .inference import generate_completion
from .metrics import TokenMetrics, completion_scores, failure_labels
from .runs import canonical_hash, read_json, write_json
from .runtime import accelerator_lease

EVALUATION_PROTOCOL = {
    "version": "prediction-only-v1",
    "split": "test",
    "line_limit": 5000,
    "line_min_prefix_tokens": 64,
    "line_max_target_tokens": 64,
    "generation_max_tokens": 64,
    "stream_file_limit": 128,
    "stream_min_tokens": 1026,
    "stream_max_tokens": 4096,
    "stream_score_stride": 32,
    "unigram_smoothing": 1.0,
    "line_scoring": "literal physical line including existing newline and indentation",
    "generated_source_execution": False,
}


def evaluation_protocol(run: Any) -> dict[str, Any]:
    """Bound by freeze_checkpoint before any held-out generation or scoring."""
    return {
        **EVALUATION_PROTOCOL,
        "budgets": list(run.config.service.budgets),
        "sink_tokens": run.config.service.sink_tokens,
        "seed": run.config.training.seed,
        "context_length": run.config.model.context_length,
    }


@contextmanager
def _evaluation(run: Any, encoded: Any, stage: str) -> Iterator[tuple[Any, Any, dict[str, Any]]]:
    run.verify()
    bindings = read_json(run.path / "config" / "bindings.json")
    if bindings.get("encoded") != encoded.identity:
        raise ValueError("Evaluation encoded-data identity differs from run")
    frozen = read_json(run.path / "evaluation" / "frozen.json")
    if frozen.get("encoded_identity") != encoded.identity:
        raise ValueError("Frozen encoded-data identity differs from evaluation data")
    if frozen.get("evaluation_protocol") != evaluation_protocol(run):
        raise ValueError("Frozen evaluation protocol differs from implementation")
    try:
        with accelerator_lease(run.root):
            model, tokenizer, identity = load_inference(run)
            if tokenizer.identity != encoded.tokenizer.identity:
                raise ValueError("Frozen tokenizer differs from encoded evaluation data")
            model.eval()
            with torch.inference_mode():
                yield model, tokenizer, identity
    except BaseException as error:
        write_json(
            run.path / "evaluation" / f"{stage}_failure.json",
            {
                "stage": stage,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "encoded": encoded.identity,
                "policy": "Operation stopped; no automatic device/precision/budget change.",
            },
        )
        raise


class _Rows:
    def __init__(self, directory: Path, name: str):
        directory.mkdir(parents=True, exist_ok=True)
        self.directory, self.name = directory, name
        self.batch_id = uuid.uuid4().hex
        self.json_path = directory / f".{name}.{self.batch_id}.jsonl.partial"
        self.csv_path = directory / f".{name}.{self.batch_id}.csv.partial"
        self.json_stream = self.json_path.open("x", encoding="utf-8")
        self.csv_stream = self.csv_path.open("x", encoding="utf-8", newline="")
        self.writer: csv.DictWriter | None = None

    def append(self, row: dict[str, Any]) -> None:
        self.json_stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        if self.writer is None:
            self.writer = csv.DictWriter(self.csv_stream, fieldnames=list(row))
            self.writer.writeheader()
        self.writer.writerow(
            {
                key: json.dumps(value, ensure_ascii=False)
                if isinstance(value, (dict, list))
                else value
                for key, value in row.items()
            }
        )

    def __enter__(self) -> _Rows:
        return self

    def __exit__(self, error_type: Any, *_: Any) -> None:
        self.json_stream.close()
        self.csv_stream.close()
        if error_type is None:
            os.replace(self.json_path, self.directory / f"{self.name}.jsonl")
            os.replace(self.csv_path, self.directory / f"{self.name}.csv")
        # On failure .partial files are intentionally retained beside the diagnostic.


def _device(model: Any) -> torch.device:
    return next(model.parameters()).device


def evaluate_model(run: Any, encoded: Any) -> dict[str, Any]:
    """All held-out file windows, weighted by nonpadding target tokens."""
    with _evaluation(run, encoded, "tokens") as (model, _, identity):
        totals, baseline = TokenMetrics(), TokenMetrics()
        counts = (
            encoded.train_token_counts.astype(np.float64) + EVALUATION_PROTOCOL["unigram_smoothing"]
        )
        probabilities = counts / counts.sum()
        log_probs = torch.tensor(np.log(probabilities), dtype=torch.float64)
        frequent = int(probabilities.argmax())
        dataset = encoded.dataset("test")
        loader = DataLoader(
            dataset, batch_size=run.config.training.batch_size, shuffle=False, num_workers=0
        )
        with _Rows(run.path / "evaluation", "token_rows") as rows:
            for batch_id, batch in enumerate(tqdm(loader, desc="Held-out token prediction")):
                inputs, labels = (
                    batch["input_ids"].to(_device(model)),
                    batch["labels"].to(_device(model)),
                )
                logits = model(inputs)
                losses = F.cross_entropy(
                    logits.transpose(1, 2), labels, reduction="none", ignore_index=-100
                )
                valid = labels != -100
                correct = ((logits.argmax(-1) == labels) & valid).sum(1).cpu().tolist()
                loss_sums = losses.sum(1).cpu().tolist()
                token_counts = valid.sum(1).cpu().tolist()
                for index in range(len(inputs)):
                    targets = batch["labels"][index]
                    targets = targets[targets != -100]
                    base_loss = float(-log_probs[targets].sum())
                    base_correct = int((targets == frequent).sum())
                    totals.add(loss_sums[index], correct[index], token_counts[index])
                    baseline.add(base_loss, base_correct, token_counts[index])
                    rows.append(
                        {
                            "window": batch_id * run.config.training.batch_size + index,
                            "tokens": token_counts[index],
                            "loss_sum": loss_sums[index],
                            "correct": correct[index],
                            "baseline_loss_sum": base_loss,
                            "baseline_correct": base_correct,
                        }
                    )
        result = {
            "model": totals.result(),
            "token_frequency_baseline": baseline.result(),
            "windows": len(dataset),
            "checkpoint": identity,
            "protocol": EVALUATION_PROTOCOL,
            "encoded": encoded.identity,
        }
        write_json(run.path / "evaluation" / "token_metrics.json", result)
        return result


def line_cases(data: Any, tokenizer: Any, seed: int, limit: int = 5000) -> list[dict[str, Any]]:
    """Select at most one eligible physical line per file before any generation."""
    cases = []
    records = sorted(
        data.records_for("test"), key=lambda row: stable_hash(seed, "file:" + row["id"])
    )
    for record in tqdm(records, desc="Select line cases"):
        text = data.text(record)
        pieces = text.split("\n")
        lines = [piece + "\n" for piece in pieces[:-1]]
        if pieces[-1]:
            lines.append(pieces[-1])
        starts = np.cumsum([0, *map(len, lines[:-1])]).tolist()
        candidate_indices = sorted(
            range(len(lines)), key=lambda index: stable_hash(seed, f"line:{record['id']}:{index}")
        )
        for index in candidate_indices:
            target = lines[index]
            if not target.strip() or target.lstrip().startswith("#"):
                continue
            if len(tokenizer.encode(target)) > EVALUATION_PROTOCOL["line_max_target_tokens"]:
                continue
            prefix = text[: starts[index]]
            if len(tokenizer.encode(prefix)) < EVALUATION_PROTOCOL["line_min_prefix_tokens"]:
                continue
            cases.append(
                {
                    "id": record["id"],
                    "line_number": index + 1,
                    "prefix": prefix,
                    "target": target,
                    "previous_line": lines[index - 1] if index else "",
                }
            )
            break
        if len(cases) == limit:
            break
    return cases


def _save_cases(run: Any, name: str, cases: list[dict[str, Any]], identity: dict[str, Any]) -> None:
    content = {"cases": cases, "checkpoint": identity, "protocol": EVALUATION_PROTOCOL}
    path = run.path / "evaluation" / f"{name}_cases.json"
    if path.exists() and read_json(path) != content:
        raise ValueError(f"Previously frozen {name} panel differs; use a fresh run")
    write_json(path, content)


def evaluate_lines(run: Any, encoded: Any) -> dict[str, Any]:
    with _evaluation(run, encoded, "lines") as (model, tokenizer, identity):
        cases = line_cases(
            encoded.data, tokenizer, run.config.training.seed, EVALUATION_PROTOCOL["line_limit"]
        )
        _save_cases(run, "line", cases, identity)
        sums = Counter()
        failures = Counter()
        with _Rows(run.path / "evaluation", "line_rows") as rows:
            for case in tqdm(cases, desc="Next-line completion"):
                policy = CachePolicy(
                    model,
                    "exact",
                    run.config.model.context_length,
                    sink_tokens=run.config.service.sink_tokens,
                )
                policy.set_tokens([tokenizer.bos_id, *tokenizer.encode(case["prefix"])])
                done = None
                for event in generate_completion(
                    policy, tokenizer, max_new_tokens=64, stop_at_newline=True
                ):
                    if event["type"] == "done":
                        done = event
                if done is None:
                    raise RuntimeError("Generation ended without a completion receipt")
                prediction = done["text"]
                scores = completion_scores(prediction, case["target"])
                base = completion_scores(case["previous_line"], case["target"])
                labels = failure_labels(prediction, case["target"], done["stop_reason"])
                sums.update(scores)
                sums.update({"baseline_" + key: value for key, value in base.items()})
                failures.update(labels)
                rows.append(
                    {
                        "id": case["id"],
                        "line_number": case["line_number"],
                        "target": case["target"],
                        "prediction": prediction,
                        "previous_line": case["previous_line"],
                        **scores,
                        **{"baseline_" + k: v for k, v in base.items()},
                        "generated_tokens": len(done["token_ids"]),
                        "stop_reason": done["stop_reason"],
                        "failures": labels,
                    }
                )
        result = {
            "cases": len(cases),
            "requested_cases": EVALUATION_PROTOCOL["line_limit"],
            "available_test_files": len(encoded.data.records_for("test")),
            "metrics": {
                key: sums[key] / len(cases) if cases else None
                for key in (
                    "exact_match",
                    "edit_similarity",
                    "baseline_exact_match",
                    "baseline_edit_similarity",
                )
            },
            "failure_counts": dict(failures),
            "case_panel_sha256": canonical_hash(cases),
            "checkpoint": identity,
            "protocol": EVALUATION_PROTOCOL,
            "limitation": "Text continuation scores do not establish functional correctness; no generated code was executed.",
        }
        write_json(run.path / "evaluation" / "line_metrics.json", result)
        return result


def streaming_cases(data: Any, tokenizer: Any, seed: int) -> list[dict[str, Any]]:
    records = sorted(
        data.records_for("test"), key=lambda row: stable_hash(seed, "stream:" + row["id"])
    )
    cases = []
    for row in records:
        ids = [tokenizer.bos_id, *tokenizer.encode(data.text(row)), tokenizer.eos_id]
        if len(ids) < EVALUATION_PROTOCOL["stream_min_tokens"]:
            continue
        cases.append(
            {
                "id": row["id"],
                "tokens": ids[: EVALUATION_PROTOCOL["stream_max_tokens"]],
                "original_tokens": len(ids),
            }
        )
        if len(cases) == EVALUATION_PROTOCOL["stream_file_limit"]:
            break
    return cases


def evaluate_streaming(run: Any, encoded: Any) -> dict[str, Any]:
    with _evaluation(run, encoded, "streaming") as (model, tokenizer, identity):
        cases = streaming_cases(encoded.data, tokenizer, run.config.training.seed)
        _save_cases(run, "streaming", cases, identity)
        totals: dict[tuple[str, int, str], TokenMetrics] = defaultdict(TokenMetrics)
        deltas: dict[tuple[int, str], list[float]] = defaultdict(list)
        with _Rows(run.path / "evaluation", "streaming_rows") as rows:
            for case in tqdm(cases, desc="Cache-quality replay"):
                ids = case["tokens"]
                baseline_losses: dict[tuple[int, int], float] = {}
                for budget in run.config.service.budgets:
                    for policy_name in ("stateless", "exact", "sink"):
                        policy = CachePolicy(
                            model, policy_name, budget, sink_tokens=run.config.service.sink_tokens
                        )
                        for position in range(
                            EVALUATION_PROTOCOL["stream_score_stride"],
                            len(ids),
                            EVALUATION_PROTOCOL["stream_score_stride"],
                        ):
                            logits = policy.set_tokens(ids[:position])
                            label = ids[position]
                            loss = float(-F.log_softmax(logits.float(), dim=-1)[label].item())
                            correct = int(logits.argmax().item() == label)
                            phase = "before_capacity" if position <= budget else "after_capacity"
                            totals[(policy_name, budget, phase)].add(loss, correct, 1)
                            if policy_name == "stateless":
                                baseline_losses[(budget, position)] = loss
                            delta = loss - baseline_losses[(budget, position)]
                            if policy_name == "sink":
                                deltas[(budget, phase)].append(delta)
                            rows.append(
                                {
                                    "id": case["id"],
                                    "policy": policy_name,
                                    "budget": budget,
                                    "position": position,
                                    "phase": phase,
                                    "target": label,
                                    "loss": loss,
                                    "correct": correct,
                                    "loss_delta_from_stateless": delta,
                                    "context_positions": min(budget, position),
                                    "resident_kv_positions": policy.cache_size,
                                }
                            )
        result = {
            "files": len(cases),
            "requested_files": EVALUATION_PROTOCOL["stream_file_limit"],
            "groups": [
                {
                    "policy": policy,
                    "budget": budget,
                    "phase": phase,
                    **totals[(policy, budget, phase)].result(),
                }
                for budget in run.config.service.budgets
                for policy in ("stateless", "exact", "sink")
                for phase in ("before_capacity", "after_capacity")
            ],
            "sink_loss_deltas": [
                {
                    "budget": budget,
                    "phase": phase,
                    "positions": len(values),
                    "mean_delta_from_stateless": float(np.mean(values)),
                }
                for (budget, phase), values in deltas.items()
            ],
            "case_panel_sha256": canonical_hash(cases),
            "checkpoint": identity,
            "protocol": EVALUATION_PROTOCOL,
            "limitation": "Long-file subset; sink reuse is approximate and measures historical influence, not retained full source attention.",
        }
        write_json(run.path / "evaluation" / "streaming_metrics.json", result)
        return result


def failure_examples(run: Any, limit: int = 20) -> list[dict[str, Any]]:
    path = run.path / "evaluation" / "line_rows.jsonl"
    if not path.exists():
        raise FileNotFoundError("Evaluate next-line completions before inspecting failures")
    examples = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if not row["exact_match"]:
                examples.append(row)
                if len(examples) == limit:
                    break
    return examples
