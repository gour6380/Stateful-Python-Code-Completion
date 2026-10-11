"""Assemble observed results and pending gates without inventing measurements."""

from __future__ import annotations

import json

import pandas as pd

from .runs import read_json, sha256_file, write_json

RESULTS = {
    "Data audit": "evaluation/data_audit.json",
    "Authored acceptance suite": "environment/acceptance_tests.json",
    "Synthetic runtime validation": "environment/runtime_validation.json",
    "Held-out token prediction": "evaluation/token_metrics.json",
    "Next-line completion": "evaluation/line_metrics.json",
    "Streaming cache quality": "evaluation/streaming_metrics.json",
    "Trained cache parity": "benchmarks/parity.json",
    "Direct-engine timings": "benchmarks/engine.json",
    "HTTP/session timings": "benchmarks/http.json",
    "Restart replay": "benchmarks/restart.json",
}


def generate_report(run):
    """Read completed stages only; missing stages remain explicitly pending."""
    run.verify()
    directory = run.path / "report"
    directory.mkdir(exist_ok=True)
    sections = [
        f"# Stateful Python Code Completion — {run.path.name}",
        "## Question and scope",
        "How do exact cache reuse and bounded attention-sink caching affect prediction quality, "
        "latency and session correctness as Python source grows and changes?",
        "One scratch-trained model and one seed. Prediction scores do not establish functional "
        "correctness. Generated Python is never executed. A weak model, parity failure or cache "
        "slowdown remains a reportable outcome.",
        "## Protocol",
        f"Py150-derived repository/verified-clone split; exactly {run.config.data.training_files:,} selected training files. "
        "Tokenizer sees selected training data only. Checkpoints use validation cross-entropy. "
        "Held-out cases follow frozen rules. Sink states preserve historical influence and are "
        "approximate; they do not provide attention to discarded source.",
        "Budgets count every retained position. Stateless and exact modes rebase the retained "
        f"window. Sink mode retains the first {run.config.service.sink_tokens} positions (including BOS) and the newest W−{run.config.service.sink_tokens} "
        "positions, with sequential eviction and raw-key RoPE repositioning.",
        "Timings require fixed-panel trained-weight parity. MPS absolute tolerance is 3e-4; "
        "CPU/CUDA use 3e-5, with zero relative tolerance and matching greedy decisions. "
        "Failed policies remain in the gate report and are excluded from qualified timing claims.",
        "Warmups and measured requests begin from equal accepted histories. Engine forced-token "
        "work and natural API generation have different timing boundaries. Memory snapshots "
        "are not peak memory. Device allocations may overlap process RSS and must not be summed.",
    ]
    inventory = []
    for title, relative in RESULTS.items():
        path = run.path / relative
        sections.append(f"## {title}")
        if not path.exists():
            sections.append("PENDING — this stage has no saved result.")
            inventory.append({"stage": title, "status": "pending", "path": relative})
            continue
        result = read_json(path)
        inventory.append(
            {
                "stage": title,
                "status": "artifact_present_runtime_not_independently_verified",
                "path": relative,
                "sha256": sha256_file(path),
            }
        )
        sections.append(f"Saved evidence: [{relative}](../{relative}).")
        compact = {
            key: value
            for key, value in result.items()
            if key
            not in {
                "rows",
                "identity",
                "checkpoint",
                "encoded",
                "protocol",
                "startup_rows",
                "repository_distribution",
            }
        }
        if result.get("startup_rows"):
            startup = pd.DataFrame(result["startup_rows"])
            startup[["block", "repeat", "load_seconds", "startup_seconds"]].to_csv(
                directory / "startup_measurements.csv", index=False
            )
            compact["startup_measured_requests"] = len(startup)
            compact["loading_p50_seconds"] = float(startup["load_seconds"].median())
            compact["startup_p50_seconds"] = float(startup["startup_seconds"].median())
            compact["startup_p95_seconds"] = float(startup["startup_seconds"].quantile(0.95))
            sections.append(
                "[Shared-model loading/startup measurements](startup_measurements.csv)."
            )
        if isinstance(result.get("summary"), list) and result["summary"]:
            table = pd.DataFrame(result["summary"])
            table.to_csv(directory / f"{path.stem}_summary.csv", index=False)
            compact.pop("summary", None)
            sections.append(f"[Per-block summary table]({path.stem}_summary.csv).")
        sections.append("```json\n" + json.dumps(compact, indent=2, ensure_ascii=False) + "\n```")
    sections.extend(
        [
            "## Interpretation and limitations",
            "Inspect per-file failures and before/after-capacity groups alongside averages. "
            "Exact-mode recomputation after a window shift can remove the benefit of reuse. "
            "A sink advantage in latency does not establish equal prediction quality. "
            "License/parse filters and approximate candidate discovery alter the retained population; "
            "report the data audit, exclusions and unavailable groups.",
            "This is an adaptation motivated by RoFormer and StreamingLLM, not a reproduction. "
            "No pretrained model, hyperparameter search, functional execution, retrieval, suffix "
            "infilling or deployment-scale claim is included. CPU correctness and MPS behavior "
            "require their own executed validation receipts.",
            "## Reproduction artifacts",
            "See saved config/, environment/, source_snapshot/, checkpoints/, evaluation/, "
            "benchmarks/, logs/, tensorboard/ and figures/. Reopening a run uses saved configuration; "
            "identity changes require a fresh run. Restart reconstructs state from accepted source.",
        ]
    )
    (directory / "report.md").write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    write_json(directory / "stage_inventory.json", inventory)
    checksums = {}
    for path in sorted(run.path.rglob("*")):
        if path.is_file() and path.name != "artifact_checksums.json" and path.suffix != ".lock":
            checksums[path.relative_to(run.path).as_posix()] = sha256_file(path)
    write_json(directory / "artifact_checksums.json", checksums)
    return {
        "report": str(directory / "report.md"),
        "stages": inventory,
        "checksummed_files": len(checksums),
    }
