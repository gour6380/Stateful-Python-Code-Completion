"""One explicit saved-artifact chart per notebook call; no experiment execution."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd

from .runs import read_json

COLORS = {"stateless": "#3975b9", "exact": "#db8d24", "sink": "#548b50"}


def _save(run, fig, name):
    directory = run.path / "figures"
    directory.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(directory / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(directory / f"{name}.svg", bbox_inches="tight")
    return fig


def plot_learning_curves(run):
    path = run.path / "logs" / "epochs.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("No committed epoch metrics are available")
    frame = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 4))
    for field, label in (("training_nll", "Training"), ("validation_nll", "Validation")):
        if field not in frame:
            raise ValueError(f"Missing {field} in saved epoch metrics")
        ax.plot(frame["epoch"], frame[field], marker="o", label=label)
    ax.set(
        xlabel="Completed epoch",
        ylabel="Token-weighted cross-entropy (nats)",
        title="Learning curves — committed epochs",
    )
    ax.legend()
    return _save(run, fig, "learning_curves")


def plot_prediction_quality(run):
    result = read_json(run.path / "evaluation" / "token_metrics.json")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    systems = ("model", "token_frequency_baseline")
    for ax, metric, title in zip(
        axes, ("cross_entropy", "accuracy"), ("Cross-entropy (nats)", "Token accuracy"), strict=True
    ):
        values = [result[name][metric] for name in systems]
        ax.bar(["Transformer", "Training frequency"], values, color=["#3975b9", "#999999"])
        ax.set_title(title)
    fig.suptitle("Held-out token prediction")
    return _save(run, fig, "prediction_quality")


def plot_completion_quality(run):
    result = read_json(run.path / "evaluation" / "line_metrics.json")
    if not result["cases"]:
        raise ValueError("No eligible next-line cases; report the unavailable result")
    metrics = result["metrics"]
    fig, ax = plt.subplots(figsize=(8, 4))
    for offset, prefix, name, color in (
        (-0.18, "", "Transformer", "#3975b9"),
        (0.18, "baseline_", "Previous line", "#999999"),
    ):
        ax.bar(
            [offset, 1 + offset],
            [metrics[prefix + key] for key in ("exact_match", "edit_similarity")],
            width=0.36,
            label=name,
            color=color,
        )
    ax.set(
        xticks=[0, 1],
        xticklabels=["Exact match", "Character edit similarity"],
        ylim=(0, 1),
        title=f"Next-line text quality — {result['cases']} cases",
    )
    ax.legend()
    return _save(run, fig, "completion_quality")


def plot_streaming_quality(run):
    result = read_json(run.path / "evaluation" / "streaming_metrics.json")
    frame = pd.DataFrame(result["groups"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, phase in zip(axes, ("before_capacity", "after_capacity"), strict=True):
        for policy, color in COLORS.items():
            values = frame[(frame["policy"] == policy) & (frame["phase"] == phase)]
            ax.plot(
                values["budget"], values["cross_entropy"], marker="o", label=policy, color=color
            )
        ax.set(title=phase.replace("_", " "), xlabel="Total retained positions")
        ax.legend()
    axes[0].set_ylabel("Cross-entropy (nats)")
    fig.suptitle(f"Streaming prediction — {result['files']} long test files")
    return _save(run, fig, "streaming_quality")


def plot_engine_latency(run):
    result = read_json(run.path / "benchmarks" / "engine.json")
    frame = pd.DataFrame(result["summary"])
    frame = frame[frame["workload"] == "fixed_32_tokens"]
    if frame.empty:
        raise ValueError("No qualified fixed-work timings")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, regime in zip(axes, ("before_capacity", "at_capacity", "after_capacity"), strict=True):
        for policy, color in COLORS.items():
            rows = frame[(frame["policy"] == policy) & (frame["regime"] == regime)]
            for block, values in rows.groupby("block"):
                ax.plot(
                    values["budget"],
                    values["p50_seconds"] * 1000,
                    marker="o",
                    color=color,
                    alpha=0.55,
                    label=policy if block == rows["block"].min() else None,
                )
        ax.set(title=regime.replace("_", " "), xlabel="Cache budget")
        ax.legend()
    axes[0].set_ylabel("32-token operation p50 (ms), each paired block")
    return _save(run, fig, "engine_latency")


def plot_http_latency(run):
    rows = read_json(run.path / "benchmarks" / "http.json")["rows"]
    frame = pd.DataFrame([row for row in rows if row.get("workload") == "natural_next_line"])
    if frame.empty:
        raise ValueError("No qualified HTTP timings")
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey="row")
    regimes = ("before_capacity", "at_capacity", "after_capacity")
    for row, (column, title) in enumerate(
        (("first_text_seconds", "First visible text"), ("seconds", "Total request"))
    ):
        for ax, regime in zip(axes[row], regimes, strict=True):
            for policy, color in COLORS.items():
                selected = frame[(frame["policy"] == policy) & (frame["regime"] == regime)]
                for block, samples in selected.groupby("block"):
                    values = samples.groupby("budget")[column].median()
                    ax.plot(
                        values.index,
                        values.values * 1000,
                        marker="o",
                        color=color,
                        alpha=0.55,
                        label=policy if block == selected["block"].min() else None,
                    )
            ax.set(
                title=regime.replace("_", " "),
                xlabel="Total cache positions",
                ylabel=f"{title} median (ms)",
            )
            ax.legend()
    fig.suptitle("Natural next-line HTTP requests — each paired block, variable output length")
    return _save(run, fig, "http_latency")


def plot_memory(run):
    rows = read_json(run.path / "benchmarks" / "engine.json")["rows"]
    values = []
    for row in rows:
        if row.get("status") == "measured":
            memory = row.get("memory", {})
            rss = memory.get("process_rss_bytes", memory.get("rss_bytes"))
            if rss is not None:
                values.append({"policy": row["policy"], "budget": row["budget"], "rss": rss})
    if not values:
        raise ValueError("No process RSS measurements were available")
    frame = pd.DataFrame(values)
    fig, ax = plt.subplots(figsize=(8, 4))
    for policy, color in COLORS.items():
        points = frame[frame["policy"] == policy].groupby("budget")["rss"].median()
        ax.plot(points.index, points.values / 2**20, marker="o", color=color, label=policy)
    ax.set(
        xlabel="Retained positions",
        ylabel="Process RSS (MiB)",
        title="Post-request process snapshots — not peak or KV-only memory",
    )
    ax.legend()
    return _save(run, fig, "memory_snapshots")


def plot_kv_memory(run):
    rows = read_json(run.path / "benchmarks" / "engine.json")["rows"]
    frame = pd.DataFrame([row for row in rows if row.get("status") == "measured"])
    if frame.empty or "kv_bytes" not in frame:
        raise ValueError("No retained KV measurements are available")
    fig, ax = plt.subplots(figsize=(8, 4))
    for policy, color in COLORS.items():
        values = frame[frame["policy"] == policy].groupby("budget")["kv_bytes"].max()
        ax.plot(values.index, values.values / 2**20, marker="o", color=color, label=policy)
    ax.set(
        xlabel="Total cache budget",
        ylabel="Retained KV tensor bytes (MiB)",
        title="Retained KV only — excludes source text, proposals and temporary attention",
    )
    ax.legend()
    return _save(run, fig, "retained_kv_memory")
