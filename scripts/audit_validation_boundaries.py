"""Bounded validation-only diagnosis; never changes scientific source or run artifacts.

Run from the project root with .venv/bin/python -B scripts/audit_validation_boundaries.py.
The output directory must be new. Selection and protocol are saved before model inference.
Generated source is treated only as text and is never executed.
"""

from __future__ import annotations

import argparse
import codecs
import hashlib
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_state(path):
    return {
        str(item.relative_to(path)): [item.stat().st_size, item.stat().st_mtime_ns]
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }


def common_length(left, right):
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return min(len(left), len(right))


def containing_token(tokenizer, ids, boundary):
    """Return the first full-source token with any bytes after the prefix boundary."""
    consumed = 0
    for index, token_id in enumerate(ids):
        raw = tokenizer.token_bytes(token_id)
        if consumed + len(raw) > boundary:
            return index, consumed, raw
        consumed += len(raw)
    raise ValueError("Eligible target has no source token after its prefix")


def decoded_sample(tokenizer, ids):
    """Independent strict decoder for all sampled bytes, including post-newline bytes."""
    raw = b"".join(tokenizer.token_bytes(token) for token in ids)
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    text = decoder.decode(raw, final=False)
    return text, len(decoder.getstate()[0])


def audit(root: Path, output: Path):
    root, output = root.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"Audit output already exists; choose a new --output: {output}")
    if not output.is_relative_to(root / "docs" / "validation_audit"):
        raise ValueError("Audit outputs must remain under docs/validation_audit")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root))
    # These flags must precede the first torch import, including indirect imports.
    from src.bootstrap import configure_environment

    configure_environment(prefer_metal=True)
    import torch

    from src.cache import CachePolicy
    from src.checkpoints import load_inference, verify_frozen
    from src.data import PreparedData, stable_hash
    from src.inference import generate_completion, greedy_token
    from src.runs import canonical_hash, open_run, read_json, write_json
    from src.runtime import accelerator_lease, configure_execution, runtime_identity, synchronize
    from src.tokenizer import load_tokenizer

    if not torch.backends.mps.is_available():
        raise RuntimeError("This audit requires MPS access; do not substitute another device")
    run = open_run(root, "run_001")
    if run.config.training.cpu_threads != 6:
        raise ValueError("This audit requires the recorded six-thread execution profile")
    before = tree_state(run.path)
    notebook_paths = sorted(root.glob("*.ipynb")) + sorted((root / "notebooks").glob("*.ipynb"))
    notebook_before = {str(path.relative_to(root)): checksum(path) for path in notebook_paths}
    frozen = verify_frozen(run)
    if frozen["selected"] != "best":
        raise ValueError("The audit is defined for the selected best-validation checkpoint")
    export = frozen["exports"]["best"]
    tokenizer = load_tokenizer(run.path / export["directory"])
    bindings = read_json(run.path / "config" / "bindings.json")
    identity = bindings["prepared_data"]
    prepared_path = root / "data" / "prepared" / identity["cache_id"]

    def manifest_at(path, expected):
        inventory = read_json(path / "inventory.json")
        if canonical_hash(inventory) != expected["inventory"]:
            raise ValueError("Prepared inventory identity mismatch")
        # Verify the manifests and only selected validation source contents below.
        # Do not read train/test source text or rebuild/rehash the raw corpus.
        for name in ("metadata.json", "records.json"):
            if checksum(path / name) != inventory[name]["sha256"]:
                raise ValueError(f"Prepared {name} checksum mismatch")
        return read_json(path / "metadata.json"), inventory

    metadata, inventory = manifest_at(prepared_path, identity)
    if metadata["cache_id"] != identity["cache_id"]:
        raise ValueError("Prepared manifest identity mismatch")
    text_root = prepared_path
    if "parent" in identity:
        if metadata["parent"] != identity["parent"]:
            raise ValueError("Prepared parent identity mismatch")
        text_root = prepared_path.parent / identity["parent"]["cache_id"]
        parent_metadata, parent_inventory = manifest_at(text_root, identity["parent"])
        if parent_metadata["cache_id"] != identity["parent"]["cache_id"]:
            raise ValueError("Prepared parent cache mismatch")
        inventory = parent_inventory
    records = [
        row for row in read_json(prepared_path / "records.json") if row["split"] == "validation"
    ]
    data = PreparedData(prepared_path, records, identity, text_root=text_root)
    protocol = {
        "schema": 1,
        "name": "validation-boundary-blank-eos-audit-v1",
        "split": "validation",
        "seed": 42,
        "case_limit": 64,
        "required_device": "mps",
        "selection": "SHA256(seed:file:<id>) file order; SHA256(seed:line:<id>:<zero-based-index>) line order; first eligible line per file; stop at 64 cases, before any model inference",
        "eligibility": "Nonblank, non-comment physical line; independently encoded prefix >=64 tokens; independently encoded target <=64 tokens; preserve whitespace and newline",
        "canonical_generation": {
            "policy": "exact",
            "budget": 1024,
            "max_new_tokens": 64,
            "stop_at_newline": True,
        },
        "diagnostics": [
            "UTF-8 round trips and unchanged-token prefix against full-source and prefix+target tokenization",
            "Whether a full-source BPE token crosses the byte boundary between prefix and target",
            "Canonical next-token distribution: EOS, newline-containing mass, independently encoded target first ID (not an aligned gold label if retokenization occurs)",
            "Teacher-forced full-source token context ending at/before the character boundary and its next source token; diagnostic only, not a changed completion protocol",
            "Existing greedy generation plus independent decoding to verify EOS/newline handling and discarded bytes inside final token",
        ],
        "no_training_or_parameter_search": True,
        "no_test_source_read_or_scoring": True,
        "no_generated_source_execution": True,
        "interpretation": "Post-hoc descriptive validation diagnosis, not a new held-out score or a causal fix experiment; checkpoint selection already used validation loss",
        "run_identity": run.identity,
        "checkpoint_identity": export["identity"],
        "prepared_identity": identity,
        "script_sha256": checksum(__file__),
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "audit_script.py").write_bytes(Path(__file__).read_bytes())
    write_json(output / "protocol.json", protocol)
    cases = []
    source_cache = {}
    for row in sorted(records, key=lambda item: stable_hash(42, "file:" + item["id"])):
        if inventory[row["text_path"]]["sha256"] != row["sha256"]:
            raise ValueError("Validation source differs from parent inventory")
        text = data.text(row)
        pieces = text.split("\n")
        lines = [piece + "\n" for piece in pieces[:-1]]
        if pieces[-1]:
            lines.append(pieces[-1])
        starts, position = [], 0
        for line in lines:
            starts.append(position)
            position += len(line)
        for index in sorted(
            range(len(lines)), key=lambda i: stable_hash(42, f"line:{row['id']}:{i}")
        ):
            target, prefix = lines[index], text[: starts[index]]
            if not target.strip() or target.lstrip().startswith("#"):
                continue
            if len(tokenizer.encode(target)) > 64 or len(tokenizer.encode(prefix)) < 64:
                continue
            cases.append(
                {
                    "id": row["id"],
                    "split": "validation",
                    "line_number": index + 1,
                    "source_sha256": row["sha256"],
                    "text_path": row["text_path"],
                    "prefix": prefix,
                    "target": target,
                }
            )
            source_cache[row["id"]] = text
            break
        if len(cases) == 64:
            break
    if len(cases) != 64:
        raise ValueError(f"Expected 64 eligible validation cases, found {len(cases)}")
    write_json(output / "cases.json", {"protocol_hash": canonical_hash(protocol), "cases": cases})
    print(f"Frozen {len(cases)} validation cases before model inference", flush=True)
    rows = []
    started = time.perf_counter()
    with accelerator_lease(root):
        execution = configure_execution(run.config.training)
        model, frozen_tokenizer, checkpoint_identity = load_inference(run)
        if (
            frozen_tokenizer.identity != tokenizer.identity
            or checkpoint_identity != export["identity"]
        ):
            raise ValueError("Audit model/tokenizer identities differ from the protocol")
        device = next(model.parameters()).device
        if device.type != "mps":
            raise RuntimeError("Audit device differs from its required MPS protocol")
        newline_ids = [
            i for i in range(4, tokenizer.vocab_size) if b"\n" in tokenizer.token_bytes(i)
        ]

        def token_info(token_id):
            return {
                "id": token_id,
                "bytes_hex": tokenizer.token_bytes(token_id).hex(),
                "text": tokenizer.token_bytes(token_id).decode("utf-8", errors="backslashreplace"),
                "special": token_id < 4,
            }

        def distribution(logits, target_id):
            values = logits.float().cpu()
            probabilities = values.softmax(-1)
            top = int(values.argmax())
            allowed = greedy_token(logits, tokenizer)
            return {
                "greedy": token_info(allowed),
                "unmasked_top_id": top,
                "target": token_info(target_id),
                "target_probability": float(probabilities[target_id]),
                "target_rank": 1 + int((values > values[target_id]).sum()),
                "target_nll": float(-values.log_softmax(-1)[target_id]),
                "eos_probability": float(probabilities[tokenizer.eos_id]),
                "newline_containing_probability": float(probabilities[newline_ids].sum()),
            }

        with torch.inference_mode(), (output / "rows.jsonl").open("x", encoding="utf-8") as stream:
            for number, case in enumerate(cases, 1):
                prefix, target, source = case["prefix"], case["target"], source_cache[case["id"]]
                prefix_ids = tokenizer.encode(prefix)
                target_ids = tokenizer.encode(target)
                full_ids = tokenizer.encode(source)
                joined_ids = tokenizer.encode(prefix + target)
                if tokenizer.decode(prefix_ids) != prefix or tokenizer.decode(full_ids) != source:
                    raise ValueError("Tokenizer failed a validation byte/text round trip")
                boundary = len(prefix.encode("utf-8"))
                source_index, source_start, source_raw = containing_token(
                    tokenizer, full_ids, boundary
                )
                exact_boundary = source_start == boundary
                history = [tokenizer.bos_id, *prefix_ids]
                engine = CachePolicy(model, "exact", 1024)
                logits = engine.set_tokens(history)
                prompt_distribution = distribution(logits, target_ids[0])
                teacher_context = [tokenizer.bos_id, *full_ids[:source_index]]
                if teacher_context == history:
                    teacher_logits = logits
                else:
                    teacher_logits, _ = model.prefill(teacher_context[-1024:])
                teacher_distribution = distribution(teacher_logits, full_ids[source_index])
                done = next(
                    event
                    for event in generate_completion(engine, tokenizer, 64, True)
                    if event["type"] == "done"
                )
                raw_decoded, pending = decoded_sample(tokenizer, done["token_ids"])
                expected = (
                    raw_decoded.split("\n", 1)[0] + "\n"
                    if done["stop_reason"] == "newline"
                    else raw_decoded
                )
                stop_consistent = (
                    done["text"] == expected and pending == done["incomplete_utf8_bytes"]
                )
                if done["stop_reason"] == "eos":
                    stop_consistent = (
                        stop_consistent
                        and greedy_token(engine.logits, tokenizer) == tokenizer.eos_id
                    )
                if not stop_consistent:
                    raise AssertionError("Independent generation stop/decoder check failed")
                row = {
                    "id": case["id"],
                    "line_number": case["line_number"],
                    "source_sha256": case["source_sha256"],
                    "prefix_tokens": len(prefix_ids),
                    "prefix_bytes": boundary,
                    "target": target,
                    "prefix_last_80_characters": prefix[-80:],
                    "prefix_tail_tokens": [token_info(i) for i in prefix_ids[-4:]],
                    "target_leading_whitespace": target[: len(target) - len(target.lstrip(" \t"))],
                    "full_source_prefix_common_tokens": common_length(prefix_ids, full_ids),
                    "prefix_tokens_replaced_by_full_source": len(prefix_ids)
                    - common_length(prefix_ids, full_ids),
                    "prefix_target_common_tokens": common_length(prefix_ids, joined_ids),
                    "full_source_has_exact_token_boundary": exact_boundary,
                    "full_source_first_target_token_index": source_index,
                    "full_source_first_target_token": token_info(full_ids[source_index]),
                    "crossing_token_prefix_bytes": boundary - source_start,
                    "crossing_token_target_bytes": len(source_raw) - (boundary - source_start),
                    "prompt_next_distribution": prompt_distribution,
                    "teacher_forced_source_distribution": teacher_distribution,
                    "teacher_context_matches_prompt": teacher_context == history,
                    "prediction": done["text"],
                    "stop_reason": done["stop_reason"],
                    "sampled_ids": done["token_ids"],
                    "raw_sampled_text": raw_decoded,
                    "bytes_discarded_after_newline": len(
                        raw_decoded[len(done["text"]) :].encode("utf-8")
                    ),
                    "stop_and_incremental_decode_consistent": stop_consistent,
                    "exact_line_match": done["text"] == target,
                }
                rows.append(row)
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                if number % 8 == 0:
                    print(f"Audited {number}/64 validation cases", flush=True)
        synchronize(device)
        runtime = runtime_identity()
    run.verify()
    if tree_state(run.path) != before:
        raise AssertionError("Original run file size/timestamp inventory changed during audit")
    if {str(path.relative_to(root)): checksum(path) for path in notebook_paths} != notebook_before:
        raise AssertionError("Notebook changed during audit")

    def group_summary(group):
        count = len(group)
        return {
            "cases": count,
            "empty_eos": sum(
                row["prediction"] == "" and row["stop_reason"] == "eos" for row in group
            ),
            "newline_only": sum(row["prediction"] == "\n" for row in group),
            "whitespace_only_nonempty": sum(
                bool(row["prediction"]) and not row["prediction"].strip() for row in group
            ),
            "exact_line_matches": sum(row["exact_line_match"] for row in group),
            "mean_prompt_eos_probability": sum(
                row["prompt_next_distribution"]["eos_probability"] for row in group
            )
            / count
            if count
            else None,
            "mean_teacher_first_target_probability": sum(
                row["teacher_forced_source_distribution"]["target_probability"] for row in group
            )
            / count
            if count
            else None,
            "teacher_first_token_correct": sum(
                row["teacher_forced_source_distribution"]["greedy"]["id"]
                == row["teacher_forced_source_distribution"]["target"]["id"]
                for row in group
            ),
        }

    summary = {
        "status": "complete",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_hash": canonical_hash(protocol),
        "case_manifest_sha256": checksum(output / "cases.json"),
        "rows_sha256": checksum(output / "rows.jsonl"),
        "script_sha256": checksum(__file__),
        "runtime": runtime,
        "execution": execution,
        "wall_seconds_including_model_load": time.perf_counter() - started,
        "validation_files_available": len(records),
        "distinct_validation_files": len({row["id"] for row in rows}),
        "overall": group_summary(rows),
        "stop_reasons": dict(Counter(row["stop_reason"] for row in rows)),
        "exact_boundary": group_summary(
            [row for row in rows if row["full_source_has_exact_token_boundary"]]
        ),
        "crossing_boundary": group_summary(
            [row for row in rows if not row["full_source_has_exact_token_boundary"]]
        ),
        "prompt_tail_retokenized_cases": sum(
            row["prefix_tokens_replaced_by_full_source"] > 0 for row in rows
        ),
        "teacher_context_identical_cases": sum(
            row["teacher_context_matches_prompt"] for row in rows
        ),
        "sampled_newline_token_had_discarded_suffix_cases": sum(
            row["bytes_discarded_after_newline"] > 0 for row in rows
        ),
        "newline_only_with_discarded_suffix_cases": sum(
            row["prediction"] == "\n" and row["bytes_discarded_after_newline"] > 0 for row in rows
        ),
        "stopping_decoder_checks_passed": sum(
            row["stop_and_incremental_decode_consistent"] for row in rows
        ),
        "tokenizer_roundtrip_checks_passed": 2 * len(rows),
        "run_identity": run.identity,
        "checkpoint_identity": checkpoint_identity,
        "prepared_identity": identity,
        "preservation": {
            "run_file_count": len(before),
            "run_size_mtime_inventory_unchanged": True,
            "run_source_environment_and_frozen_export_checks_passed": True,
            "notebook_sha256_unchanged": notebook_before,
            "no_dataset_or_run_write_functions_called": True,
        },
        "limitations": [
            "Only 64 deterministic validation cases; not a new held-out estimate or a significance test",
            "Teacher-forced and deployment contexts can differ at BPE boundaries; their probabilities are descriptive, not a controlled decoding fix",
            "Validation data already contributed to checkpoint selection",
            "No settings changed and no generated source executed; this audit alone cannot establish a quality remedy",
            "Project accelerator lease excludes cooperating project jobs, not unrelated applications",
        ],
    }
    write_json(output / "summary.json", summary)
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(output),
                "overall": summary["overall"],
                "crossing_boundary": summary["crossing_boundary"],
                "stop_reasons": summary["stop_reasons"],
            },
            indent=2,
        ),
        flush=True,
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("docs/validation_audit"))
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    audit(project, args.output if args.output.is_absolute() else project / args.output)
