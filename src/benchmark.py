"""Explicit, paired benchmarks. No missing training or evaluation is launched here."""

from __future__ import annotations

import hashlib
import json
import time
from functools import wraps

import numpy as np
from tqdm.auto import tqdm

from .cache import CachePolicy, sink_reference
from .checkpoints import load_inference
from .client import CompletionClient
from .inference import greedy_token
from .runs import canonical_hash, read_json, write_json
from .runtime import accelerator_lease, memory_snapshot, synchronize

POLICIES = ("stateless", "exact", "sink")


def _record_failure(function):
    @wraps(function)
    def operation(run, *args, **kwargs):
        try:
            return function(run, *args, **kwargs)
        except BaseException as error:
            write_json(
                run.path / "benchmarks" / f"{function.__name__}_failure.json",
                {
                    "operation": function.__name__,
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "status": "failed_no_recipe_fallback",
                },
            )
            raise

    return operation


def _panel(run, encoded):
    tokenizer = encoded.tokenizer
    rows = []
    records = sorted(
        encoded.data.records_for("validation"),
        key=lambda row: hashlib.sha256(row["id"].encode()).hexdigest(),
    )
    for budget in run.config.service.budgets:
        used = 0
        for record in records:
            text = encoded.data.text(record)
            tokens = [tokenizer.bos_id, *tokenizer.encode(text)]
            if len(tokens) < budget + 65:
                continue
            for regime, length in (
                ("before_capacity", 64),
                ("at_capacity", budget),
                ("after_capacity", budget + 32),
            ):
                # Skip byte/BPE boundaries that cannot represent exactly the same
                # canonical editor text. Never silently replace broken Unicode.
                prefix = tokens[:length]
                try:
                    source = tokenizer.decode(prefix[1:])
                except UnicodeDecodeError:
                    continue
                if [tokenizer.bos_id, *tokenizer.encode(source)] != prefix:
                    continue
                rows.append(
                    {
                        "id": f"{record['id']}:{budget}:{length}",
                        "file_id": record["id"],
                        "budget": budget,
                        "regime": regime,
                        "prefix_ids": prefix,
                        "continuation_ids": tokens[length : length + 32],
                        "source": source,
                    }
                )
            used += 1
            if used == 4:
                break
    if not rows:
        raise ValueError("No eligible validation file supports the registered benchmark panel")
    return rows


def prepare_benchmark_panel(run, encoded) -> dict:
    """Freeze data-selected validation traces independently of model outputs."""
    run.verify()
    bindings = read_json(run.path / "config" / "bindings.json")
    frozen = read_json(run.path / "evaluation" / "frozen.json")
    if (
        bindings.get("encoded") != encoded.identity
        or frozen.get("encoded_identity") != encoded.identity
    ):
        raise ValueError("Benchmark inputs do not match the frozen encoded-data identity")
    rows = _panel(run, encoded)
    result = {
        "rows": rows,
        "hash": canonical_hash(rows),
        "partition": "validation",
        "encoded_identity": encoded.identity,
        "tokenizer_identity": encoded.tokenizer.identity,
        "unavailable_budgets": [
            w for w in run.config.service.budgets if not any(row["budget"] == w for row in rows)
        ],
        "unavailable_groups": [
            {"budget": w, "regime": regime}
            for w in run.config.service.budgets
            for regime in ("before_capacity", "at_capacity", "after_capacity")
            if not any(row["budget"] == w and row["regime"] == regime for row in rows)
        ],
    }
    path = run.path / "benchmarks" / "panel.json"
    if path.exists() and read_json(path) != result:
        raise ValueError("A different benchmark panel is already registered")
    write_json(path, result)
    return {
        "cases": len(rows),
        "hash": result["hash"],
        "unavailable_budgets": result["unavailable_budgets"],
        "unavailable_groups": result["unavailable_groups"],
    }


@_record_failure
def check_cache_parity(run, encoded) -> dict:
    """Trained-weight gates on fixed validation traces; writes failures as evidence."""
    panel = read_json(run.path / "benchmarks" / "panel.json")
    with accelerator_lease(run.root):
        model, tokenizer, identity = load_inference(run)
        if (
            panel["tokenizer_identity"] != tokenizer.identity
            or panel["encoded_identity"] != encoded.identity
        ):
            raise ValueError("Benchmark panel tokenizer or data identity differs")
        device = next(model.parameters()).device
        tolerance = 3e-4 if device.type == "mps" else 3e-5
        rows = []
        for case in tqdm(panel["rows"], desc="Trained cache parity"):
            reference = CachePolicy(model, "stateless", case["budget"])
            exact = CachePolicy(model, "exact", case["budget"])
            reference.set_tokens(case["prefix_ids"])
            exact.set_tokens(case["prefix_ids"])
            maximum, disagreements = 0.0, 0
            for token in case["continuation_ids"]:
                delta = float((reference.logits - exact.logits).abs().max().item())
                maximum = max(maximum, delta)
                disagreements += int(
                    greedy_token(reference.logits, tokenizer)
                    != greedy_token(exact.logits, tokenizer)
                )
                reference.append(token)
                exact.append(token)
            # Free-running decisions are a separate gate from teacher-prefix logits.
            reference.set_tokens(case["prefix_ids"])
            exact.set_tokens(case["prefix_ids"])
            greedy_disagreements = 0
            for _ in range(32):
                left, right = (
                    greedy_token(reference.logits, tokenizer),
                    greedy_token(exact.logits, tokenizer),
                )
                greedy_disagreements += int(left != right)
                reference.append(left)
                exact.append(right)
            sink = CachePolicy(model, "sink", case["budget"], run.config.service.sink_tokens)
            sink.set_tokens(case["prefix_ids"])
            expected = sink_reference(
                model, case["prefix_ids"], case["budget"], run.config.service.sink_tokens
            )
            sink_delta = float((sink.logits - expected).abs().max().item())
            sink_greedy = int(
                greedy_token(sink.logits, tokenizer) != greedy_token(expected, tokenizer)
            )
            rows.append(
                {
                    "case_id": case["id"],
                    "budget": case["budget"],
                    "regime": case["regime"],
                    "max_abs_delta": maximum,
                    "teacher_prefix_disagreements": disagreements,
                    "greedy_disagreements": greedy_disagreements,
                    "sink_reference_max_abs_delta": sink_delta,
                    "sink_reference_greedy_disagreements": sink_greedy,
                    "exact_pass": maximum <= tolerance
                    and disagreements == 0
                    and greedy_disagreements == 0,
                    "sink_pass": sink_delta <= tolerance and sink_greedy == 0,
                }
            )
        receipt = {
            "identity": identity,
            "panel_hash": panel["hash"],
            "atol": tolerance,
            "rtol": 0,
            "rows": rows,
            "qualified": {
                "stateless": True,
                "exact": all(row["exact_pass"] for row in rows),
                "sink": all(row["sink_pass"] for row in rows),
            },
            "note": "Sink is checked against its own semantic reference, not last-W logits.",
        }
        write_json(run.path / "benchmarks" / "parity.json", receipt)
        return receipt


def _require_gate(run, identity: dict | None = None):
    run.verify()
    panel = read_json(run.path / "benchmarks" / "panel.json")
    gate = read_json(run.path / "benchmarks" / "parity.json")
    frozen = read_json(run.path / "evaluation" / "frozen.json")
    if (
        panel["encoded_identity"] != frozen["encoded_identity"]
        or panel["tokenizer_identity"] != gate["identity"]["tokenizer"]
    ):
        raise ValueError("Benchmark data/tokenizer identity no longer matches frozen artifacts")
    if gate["panel_hash"] != panel["hash"] or canonical_hash(panel["rows"]) != panel["hash"]:
        raise ValueError("Benchmark panel/parity receipt identity mismatch")
    if identity is not None and canonical_hash(gate["identity"]) != canonical_hash(identity):
        raise ValueError("Parity receipt belongs to different model artifacts")
    return panel, gate


def _summary(rows):
    groups = {}
    for row in rows:
        if row.get("status") != "measured":
            continue
        key = (row["policy"], row["budget"], row["regime"], row["workload"], row["block"])
        groups.setdefault(key, []).append(row)
    output = []
    for (policy, budget, regime, workload, block), values in groups.items():
        seconds = np.array([row["seconds"] for row in values])
        output.append(
            {
                "policy": policy,
                "budget": budget,
                "regime": regime,
                "workload": workload,
                "block": block,
                "requests": len(values),
                "p50_seconds": float(np.median(seconds)),
                "p95_seconds": float(np.percentile(seconds, 95)),
                "tokens_per_second": sum(row.get("tokens", 0) for row in values)
                / float(seconds.sum()),
            }
        )
    return output


def _save_rows(run, filename, rows, metadata):
    path = run.path / "benchmarks" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    result = {**metadata, "rows": rows, "summary": _summary(rows)}
    write_json(path.with_suffix(".json"), result)
    return result


@_record_failure
def benchmark_engine(run, encoded, *, warmups=20, repetitions=100, blocks=3) -> dict:
    """One model owner. Each measured request starts with identical canonical history."""
    with accelerator_lease(run.root):
        model, tokenizer, identity = load_inference(run)
        panel, gate = _require_gate(run, identity)
        device = next(model.parameters()).device
        rows = []
        for block in range(blocks):
            order = POLICIES[block % 3 :] + POLICIES[: block % 3]
            for policy in order:
                if not gate["qualified"][policy]:
                    rows.append({"policy": policy, "block": block, "status": "blocked_parity"})
                    continue
                for budget in run.config.service.budgets:
                    for regime in ("before_capacity", "at_capacity", "after_capacity"):
                        cases = [
                            c
                            for c in panel["rows"]
                            if c["budget"] == budget and c["regime"] == regime
                        ]
                        if not cases:
                            continue
                        for workload in ("prefill", "fixed_32_tokens", "edit_rebuild", "replay"):
                            for repeat in tqdm(
                                range(warmups + repetitions),
                                desc=f"{policy}/{budget}/{regime}/{workload}",
                                leave=False,
                            ):
                                case = cases[repeat % len(cases)]
                                engine = CachePolicy(
                                    model, policy, budget, run.config.service.sink_tokens
                                )
                                if workload not in {"prefill", "replay"}:
                                    engine.set_tokens(case["prefix_ids"])
                                edited = [*case["prefix_ids"]]
                                index = len(edited) // 2
                                edited[index] = next(
                                    t for t in tokenizer.encode("x y z") if t != edited[index]
                                )
                                synchronize(device)
                                started = time.perf_counter()
                                if workload in {"prefill", "replay"}:
                                    engine.set_tokens(case["prefix_ids"])
                                elif workload == "fixed_32_tokens":
                                    for token in case["continuation_ids"]:
                                        engine.append(token)
                                        _ = int(engine.logits.argmax().item())
                                else:
                                    engine.set_tokens(edited)
                                synchronize(device)
                                duration = time.perf_counter() - started
                                if repeat >= warmups:
                                    rows.append(
                                        {
                                            "status": "measured",
                                            "policy": policy,
                                            "budget": budget,
                                            "regime": regime,
                                            "workload": workload,
                                            "block": block,
                                            "repeat": repeat - warmups,
                                            "case_id": case["id"],
                                            "seconds": duration,
                                            "tokens": 32 if workload == "fixed_32_tokens" else 0,
                                            "kv_bytes": engine.diagnostics()["cache_bytes"],
                                            "canonical_token_count": len(engine.tokens),
                                            "memory": memory_snapshot(device),
                                        }
                                    )
        return _save_rows(
            run,
            "engine",
            rows,
            {
                "identity": identity,
                "panel_hash": panel["hash"],
                "warmups": warmups,
                "repetitions": repetitions,
                "blocks": blocks,
                "boundary": "Synchronized engine operation; setup excluded; fixed work includes argmax materialization.",
                "memory_definition": "Post-request snapshots, not peaks; RSS and device allocations overlap.",
            },
        )


@_record_failure
def benchmark_http(run, handle, *, warmups=20, repetitions=100, blocks=3) -> dict:
    """Request latency includes client HTTP, server queueing/tokenization and output transfer."""
    client = CompletionClient(handle.url)
    panel, gate = _require_gate(run, client.health()["identity"]["model"])
    rows = []
    try:
        for block in range(blocks):
            for policy in POLICIES[block % 3 :] + POLICIES[: block % 3]:
                if not gate["qualified"][policy]:
                    rows.append({"policy": policy, "block": block, "status": "blocked_parity"})
                    continue
                for budget in run.config.service.budgets:
                    for regime in ("before_capacity", "at_capacity", "after_capacity"):
                        cases = [
                            row
                            for row in panel["rows"]
                            if row["budget"] == budget and row["regime"] == regime
                        ]
                        if not cases:
                            continue
                        for repeat in tqdm(
                            range(warmups + repetitions), desc=f"HTTP {policy}/{budget}/{regime}"
                        ):
                            case = cases[repeat % len(cases)]
                            session = client.create(policy, budget)
                            sid = session["session_id"]
                            try:
                                source = client.update_source(
                                    sid, case["source"], session["revision"]
                                )
                                started, first, first_text, done = (
                                    time.perf_counter(),
                                    None,
                                    None,
                                    None,
                                )
                                for event in client.stream(sid, source["revision"]):
                                    if event.get("type") == "token" and first is None:
                                        first = time.perf_counter() - started
                                    if (
                                        event.get("type") == "token"
                                        and event.get("text")
                                        and first_text is None
                                    ):
                                        first_text = time.perf_counter() - started
                                    if event.get("type") == "done":
                                        done = event
                                elapsed = time.perf_counter() - started
                                if done is None:
                                    raise RuntimeError("HTTP stream ended without a done event")
                                if done.get("committable"):
                                    client.reject(sid, done["proposal_id"])
                                # Identical operation trace, with generated content/length
                                # retained as measured covariates, then canonical reset.
                                accepted = client.suggest(sid, source["revision"])
                                accept_seconds = None
                                if accepted.get("committable"):
                                    accept_started = time.perf_counter()
                                    source = client.accept(
                                        sid,
                                        accepted["proposal_id"],
                                        source["revision"],
                                        characters=1,
                                    )
                                    accept_seconds = time.perf_counter() - accept_started
                                source = client.update_source(
                                    sid, case["source"], source["revision"]
                                )
                                # Same edit transition in every mode, independent of generated content.
                                edited = "# fixed prefix edit\n" + case["source"]
                                edit_started = time.perf_counter()
                                source = client.update_source(sid, edited, source["revision"])
                                edit_seconds = time.perf_counter() - edit_started
                                if repeat >= warmups:
                                    common = {
                                        "status": "measured",
                                        "policy": policy,
                                        "budget": budget,
                                        "regime": regime,
                                        "block": block,
                                        "case_id": case["id"],
                                        "repeat": repeat - warmups,
                                    }
                                    rows.append(
                                        {
                                            **common,
                                            "workload": "natural_next_line",
                                            "seconds": elapsed,
                                            "first_token_seconds": first,
                                            "first_text_seconds": first_text,
                                            "tokens": len(done.get("token_ids", [])),
                                            "stop_reason": done.get("stop_reason"),
                                        }
                                    )
                                    rows.append(
                                        {
                                            **common,
                                            "workload": "source_edit",
                                            "seconds": edit_seconds,
                                            "tokens": 0,
                                            "kv_bytes": source["cache"]["cache_bytes"],
                                            "accepted_source_utf8_bytes": len(
                                                source["text"].encode("utf-8")
                                            ),
                                        }
                                    )
                                    rows.append(
                                        {
                                            **common,
                                            "workload": "partial_accept",
                                            "status": "measured"
                                            if accept_seconds is not None
                                            else "unavailable_empty_suggestion",
                                            "seconds": accept_seconds,
                                            "tokens": 0,
                                            "accepted_text": accepted["text"][:1]
                                            if accept_seconds is not None
                                            else "",
                                        }
                                    )
                            finally:
                                client.delete(sid)
        return _save_rows(
            run,
            "http",
            rows,
            {
                "identity": gate["identity"],
                "panel_hash": panel["hash"],
                "warmups": warmups,
                "repetitions": repetitions,
                "blocks": blocks,
                "boundary": "Completion includes HTTP/queue/model/transfer; session creation and initial source setup excluded.",
                "limitation": "Natural output lengths differ; use fixed-work engine results for equal-work speed claims.",
            },
        )
    finally:
        client.close()


@_record_failure
def benchmark_restart(run, *, warmups=20, repetitions=100, blocks=3) -> dict:
    """Real process restarts, then paired accepted-source replay on equal histories.

    Each block has 20 warm and 100 measured restarts. Each restart restores one
    session per qualified policy, cycling the fixed panel. Loading is shared-model
    work and reported once per restart, separately from per-policy replay latency.
    """
    from .service import start_service, stop_service

    panel, gate = _require_gate(run)
    rows, starts = [], []
    for block in range(blocks):
        order = POLICIES[block % 3 :] + POLICIES[: block % 3]
        for repeat in tqdm(range(warmups + repetitions), desc=f"Restart block {block + 1}"):
            handle = start_service(run)
            client = CompletionClient(handle.url)
            snapshots = []
            try:
                case = panel["rows"][repeat % len(panel["rows"])]
                for policy in order:
                    if not gate["qualified"][policy]:
                        continue
                    session = client.create(policy, case["budget"])
                    sid = session["session_id"]
                    source = client.update_source(sid, case["source"], session["revision"])
                    before = client.suggest(sid, source["revision"])
                    if before.get("committable"):
                        client.reject(sid, before["proposal_id"])
                    snapshots.append((policy, sid, before))
                client.close()
                stop_service(handle)
                handle = start_service(run)
                client = CompletionClient(handle.url)
                _require_gate(run, client.health()["identity"]["model"])
                if repeat >= warmups:
                    starts.append({"block": block, "repeat": repeat - warmups, **handle.startup})
                for policy, sid, before in snapshots:
                    started = time.perf_counter()
                    restored = client.restore(sid)
                    replay_seconds = time.perf_counter() - started
                    after = client.suggest(sid, restored["revision"])
                    equal = before["token_ids"] == after["token_ids"]
                    if repeat >= warmups or not equal:
                        rows.append(
                            {
                                "status": "measured" if equal else "failed_replay",
                                "policy": policy,
                                "block": block,
                                "repeat": repeat - warmups,
                                "workload": "restart_replay",
                                "tokens": 0,
                                "seconds": replay_seconds,
                                "same_greedy_tokens": equal,
                                "case_id": case["id"],
                                "budget": case["budget"],
                                "regime": case["regime"],
                                "source_revision": restored["revision"],
                                "kv_bytes": restored["cache"]["cache_bytes"],
                                "accepted_source_utf8_bytes": len(restored["text"].encode("utf-8")),
                            }
                        )
                    if not equal:
                        _save_rows(
                            run,
                            "restart",
                            rows,
                            {
                                "passed": False,
                                "identity": gate["identity"],
                                "panel_hash": panel["hash"],
                            },
                        )
                        raise RuntimeError("Restart replay changed greedy output; receipt retained")
            finally:
                for _, sid, _ in snapshots:
                    try:
                        client.delete(sid)
                    except Exception:
                        pass
                client.close()
                stop_service(handle)
    return _save_rows(
        run,
        "restart",
        rows,
        {
            "passed": bool(rows) and all(row["same_greedy_tokens"] for row in rows),
            "startup_rows": starts,
            "identity": gate["identity"],
            "panel_hash": panel["hash"],
            "warmups": warmups,
            "repetitions": repetitions,
            "blocks": blocks,
            "sampling": "100 measured replays per qualified policy/block, cycling fixed panel; actual budget/regime counts reported",
            "cold_definition": "Fresh worker/model process; operating-system file cache is not cleared.",
        },
    )
