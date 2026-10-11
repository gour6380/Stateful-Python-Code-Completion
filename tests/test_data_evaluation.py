"""Synthetic acceptance cases for data, tokenization and metric behavior."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import src.data as data_module
from src.config import DataConfig, ExperimentConfig, ModelConfig, ServiceConfig
from src.data import (
    MANIFEST_REVISION,
    DisjointSet,
    PreparedData,
    _inventory,
    lexical_shingles,
    load_prepared_data,
    normalize_source,
    partition_group,
    prepare_data,
    safe_extract,
    stable_hash,
    verify_inventory,
)
from src.evaluate import line_cases, streaming_cases
from src.metrics import TokenMetrics, completion_scores, failure_labels
from src.runs import canonical_hash, read_json, write_json
from src.tokenizer import (
    WindowDataset,
    build_encoded_windows,
    load_tokenizer,
    train_tokenizer,
    verify_roundtrips,
)


class FakeRun:
    def __init__(self, root: Path, config=None, name="run_001"):
        self.root = root
        self.path = root / "runs" / name
        self.config = config or ExperimentConfig()
        self.bound = {}
        (self.path / "evaluation").mkdir(parents=True)
        write_json(self.path / "config" / "bindings.json", {})

    def bind(self, name, identity):
        if name in self.bound:
            assert self.bound[name] == identity
        self.bound[name] = identity
        write_json(self.path / "config" / "bindings.json", self.bound)


def prepared_fixture(root: Path, texts: dict[str, str]) -> PreparedData:
    root.mkdir(parents=True)
    records = []
    for index, (split, text) in enumerate(texts.items()):
        file_id = f"file-{index}"
        path = root / f"{file_id}.py"
        path.write_text(text, encoding="utf-8")
        records.append(
            {
                "id": file_id,
                "repository": file_id,
                "split": split,
                "text_path": path.name,
                "sha256": hashlib.sha256(text.encode()).hexdigest(),
                "license": "mit",
            }
        )
    return PreparedData(root, records, {"test_fixture": True})


def small_configuration():
    return ExperimentConfig(
        data=DataConfig(training_files=1, validation_files=None, vocab_size=260),
        model=ModelConfig(vocab_size=260, context_length=8),
        service=ServiceConfig(budgets=(8,)),
    )


def test_source_encoding_and_newlines_preserved():
    source = b"# coding: latin-1\r\nname = 'caf\xe9'\r\n\t# comment\r\n"
    assert normalize_source(source) == "# coding: latin-1\nname = 'café'\n\t# comment\n"


def test_python2_source_is_excluded_instead_of_translated():
    with pytest.raises(SyntaxError):
        normalize_source(b"print 'legacy'\n")


def test_shingles_ignore_comments_and_formatting():
    assert lexical_shingles("x = 1\ny = x + 2\n") == lexical_shingles("x=1 # text\n\ny=x+2\n")
    assert lexical_shingles("x = 1\n") != lexical_shingles("x = 2\n")


def test_repository_and_clone_transitive_grouping():
    groups = DisjointSet()
    groups.union("owner/z", "owner/b")
    groups.union("owner/b", "owner/a")
    assert {groups.find(repo) for repo in ("owner/z", "owner/b", "owner/a")} == {"owner/a"}
    assert (
        len({partition_group(groups.find(repo), 42) for repo in ("owner/z", "owner/b", "owner/a")})
        == 1
    )


@pytest.mark.parametrize(
    "name,kind",
    [
        ("../escape.py", "file"),
        ("/absolute.py", "file"),
        ("link.py", "symlink"),
        ("link.py", "hardlink"),
    ],
)
def test_archive_rejects_escaping_and_links(tmp_path, name, kind):
    archive = tmp_path / "bad.tar"
    with tarfile.open(archive, "w") as tar:
        member = tarfile.TarInfo(name)
        if kind == "symlink":
            member.type, member.linkname = tarfile.SYMTYPE, "/tmp/escape"
        elif kind == "hardlink":
            member.type, member.linkname = tarfile.LNKTYPE, "elsewhere"
        else:
            member.size = 1
        tar.addfile(member, io.BytesIO(b"x") if kind == "file" else None)
    with pytest.raises(ValueError):
        safe_extract(archive, tmp_path / "out")


def test_archive_cannot_overwrite_existing_file(tmp_path):
    archive = tmp_path / "one.tar"
    with tarfile.open(archive, "w") as tar:
        member = tarfile.TarInfo("one.py")
        member.size = 1
        tar.addfile(member, io.BytesIO(b"x"))
    destination = tmp_path / "out"
    safe_extract(archive, destination)
    with pytest.raises(FileExistsError):
        safe_extract(archive, destination)
    assert (destination / "one.py").read_bytes() == b"x"


def test_inventory_detects_corruption_and_unexpected_files(tmp_path):
    (tmp_path / "a").write_text("original")
    write_json(tmp_path / "inventory.json", _inventory(tmp_path))
    assert verify_inventory(tmp_path)
    (tmp_path / "a").write_text("tampered")
    with pytest.raises(ValueError, match="verification"):
        verify_inventory(tmp_path)
    (tmp_path / "a").write_text("original")
    (tmp_path / "extra").write_text("unexpected")
    with pytest.raises(ValueError, match="Unexpected"):
        verify_inventory(tmp_path)


def raw_fixture(run, sources):
    raw = run.root / "data" / "raw" / MANIFEST_REVISION
    (raw / "source" / "data").mkdir(parents=True)
    manifest = []
    for path, text, license_name in sources:
        target = raw / "source" / "data" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
        manifest.append({"filepath": "data/" + path, "license": license_name})
    for name, content in (("train", manifest), ("dev", []), ("eval", [])):
        write_json(raw / f"{name}__manifest.json", content)
    write_json(raw / "inventory.json", _inventory(raw))


def repo_for(split):
    return next(
        f"owner/repo{i}" for i in range(1000) if partition_group(f"owner/repo{i}", 42) == split
    )


def split_sources(counts=(("train", 6), ("validation", 5), ("test", 3))):
    return [
        (
            f"{repo_for(split)}/{index}.py",
            f"unique_{split}_{index} = {100 * split_index + index}\n",
            "mit",
        )
        for split_index, (split, count) in enumerate(counts)
        for index in range(count)
    ]


def forbid_cold_preparation(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("A compatible prepared parent must avoid full preparation")

    for name in ("normalize_source", "lexical_shingles", "MinHash", "MinHashLSH"):
        monkeypatch.setattr(data_module, name, forbidden)


@pytest.fixture
def full_parent(tmp_path):
    config = replace(
        small_configuration(),
        data=DataConfig(training_files=4, validation_files=None, vocab_size=260),
    )
    run = FakeRun(tmp_path, config)
    raw_fixture(run, split_sources())
    return run, prepare_data(run)


def pilot_run_for(parent_run, *, name="run_002", **changes):
    values = {"training_files": 2, "validation_files": 2, **changes}
    config = replace(parent_run.config, data=replace(parent_run.config.data, **values))
    return FakeRun(parent_run.root, config, name=name)


def rewrite_full_parent_metadata(parent_run, parent, transform):
    """Model a previously produced, internally consistent full cache."""
    metadata = read_json(parent.path / "metadata.json")
    transform(metadata)
    raw = parent_run.root / "data/raw" / MANIFEST_REVISION
    metadata["cache_id"] = canonical_hash(
        {
            "protocol": data_module.DATA_PROTOCOL,
            "raw": verify_inventory(raw),
            "config": metadata["config"],
            "implementation": metadata["implementation"],
        }
    )
    write_json(parent.path / "metadata.json", metadata)
    write_json(parent.path / "inventory.json", _inventory(parent.path))
    destination = parent.path.parent / metadata["cache_id"]
    parent.path.rename(destination)
    return data_module._load_prepared(destination)


def test_prepare_honors_license_split_and_exact_training_count(tmp_path):
    run = FakeRun(
        tmp_path,
        replace(ExperimentConfig(), data=DataConfig(training_files=1, validation_files=None)),
    )
    repositories = {split: repo_for(split) for split in ("train", "validation", "test")}
    sources = [
        (f"{repo}/file.py", f"unique_{index} = {index}\n", "mit")
        for index, repo in enumerate(repositories.values())
    ]
    sources += [("owner/restricted/file.py", "restricted = 99\n", "gpl-3.0")]
    raw_fixture(run, sources)
    data = prepare_data(run)
    assert len(data.records_for("train")) == 1
    assert all(record["license"] == "mit" for record in data.records)
    assert {record["repository"] for record in data.records} == set(repositories.values())
    assert read_json(data.path / "audit.json")["exclusions"]["license"] == 1
    assert prepare_data(run).identity == data.identity


def test_prepare_stops_with_count_report_when_subset_too_small(tmp_path):
    run = FakeRun(
        tmp_path,
        replace(ExperimentConfig(), data=DataConfig(training_files=20, validation_files=None)),
    )
    raw_fixture(run, [(f"{repo_for('train')}/file.py", "x = 1\n", "mit")])
    with pytest.raises(ValueError, match="found 1"):
        prepare_data(run)
    assert (
        read_json(run.path / "evaluation" / "data_preparation_failure.json")[
            "requested_training_files"
        ]
        == 20
    )


@pytest.mark.parametrize("value", [0, -1, True, False, 1.5, "1000"])
def test_validation_file_limit_requires_positive_integer_or_none(value):
    with pytest.raises(ValueError, match="validation_files"):
        DataConfig(validation_files=value)


def test_pilot_defaults_roundtrip_and_legacy_all_validation_semantics():
    fresh = ExperimentConfig()
    assert fresh.data.training_files == 5000
    assert fresh.data.validation_files == 1000
    assert fresh.training.max_epochs == 15
    assert fresh.training.min_epochs == 5
    assert fresh.training.patience == 3
    assert ExperimentConfig.from_dict(fresh.to_dict()) == fresh
    saved = fresh.to_dict()
    saved["data"].pop("validation_files")
    saved["data"]["training_files"] = 20_000
    saved["training"]["max_epochs"] = 20
    restored = ExperimentConfig.from_dict(saved)
    assert restored.data.training_files == 20_000
    assert restored.data.validation_files is None
    assert restored.training.max_epochs == 20
    assert "validation_files" not in saved["data"]  # Do not mutate the saved document.


def test_deterministic_nested_pilot_preserves_splits_caches_and_train_only_fit(
    tmp_path, monkeypatch
):
    full_config = replace(
        small_configuration(),
        data=DataConfig(training_files=4, validation_files=None, vocab_size=260),
    )
    full_run = FakeRun(tmp_path, full_config)
    raw_fixture(full_run, split_sources())
    full_data = prepare_data(full_run)
    full_tokenizer = train_tokenizer(full_run, full_data)
    full_encoded = build_encoded_windows(full_run, full_data, full_tokenizer)
    old_files = {
        path: path.read_bytes()
        for directory in (full_data.path, full_tokenizer.path, full_encoded.path)
        for path in directory.rglob("*")
        if path.is_file()
    }
    pilot_run = FakeRun(
        tmp_path,
        replace(full_config, data=replace(full_config.data, training_files=2, validation_files=2)),
        name="run_002",
    )
    forbid_cold_preparation(monkeypatch)
    pilot = prepare_data(pilot_run)
    assert pilot.path != full_data.path
    assert not any(path.suffix == ".py" or path.is_symlink() for path in pilot.path.rglob("*"))
    assert not (pilot.path / "texts").exists()
    assert not (pilot.path / "clone_edges.json").exists()
    metadata = read_json(pilot.path / "metadata.json")
    assert metadata["storage_mode"] == "subset_view"
    assert metadata["parent"] == full_data.identity
    assert pilot.identity["parent"] == full_data.identity
    assert pilot.text_root == full_data.path
    assert prepare_data(pilot_run).identity == pilot.identity
    reloaded = load_prepared_data(pilot_run)
    assert reloaded.path == pilot.path
    assert reloaded.identity == pilot.identity
    assert reloaded.records == pilot.records
    assert all(reloaded.text(row) == full_data.text(row) for row in reloaded.records)
    for split, cap in (("train", 2), ("validation", 2)):
        ranked = sorted(full_data.records_for(split), key=lambda row: stable_hash(42, row["id"]))
        assert {row["id"] for row in pilot.records_for(split)} == {
            row["id"] for row in ranked[:cap]
        }
    assert pilot.records_for("test") == full_data.records_for("test")
    groups = {
        split: {row["group"] for row in pilot.records_for(split)}
        for split in ("train", "validation", "test")
    }
    assert not (groups["train"] & groups["validation"])
    assert not (groups["train"] & groups["test"])
    assert not (groups["validation"] & groups["test"])
    audit = read_json(pilot.path / "audit.json")
    assert audit["eligible_split_counts"] == {"train": 6, "validation": 5, "test": 3}
    assert audit["selected_counts"] == {"train": 2, "validation": 2, "test": 3}
    assert audit["requested_training_files"] == audit["requested_validation_files"] == 2
    assert "SHA256(seed:file_id)" in audit["selection_rule"]
    assert len(full_data.records_for("validation")) == 5  # None means all eligible files.
    seen = []
    original_text = pilot.text

    def training_only_text(record):
        seen.append(record["id"])
        return original_text(record)

    pilot.text = training_only_text
    tokenizer = train_tokenizer(pilot_run, pilot)
    train_ids = [row["id"] for row in pilot.records_for("train")]
    assert seen == train_ids
    assert tokenizer.identity["fitted_file_ids_sha256"] == canonical_hash(train_ids)
    assert tokenizer.path != full_tokenizer.path
    encoded = build_encoded_windows(pilot_run, pilot, tokenizer)
    assert encoded.path != full_encoded.path
    assert encoded.identity["data"] == pilot.identity
    assert encoded.identity["tokenizer"] == tokenizer.identity
    assert all(path.read_bytes() == before for path, before in old_files.items())


@pytest.mark.parametrize("source_hash", sorted(data_module.REVIEWED_PREPARATION_SOURCES))
def test_subset_accepts_reviewed_full_parent_with_legacy_validation_config(
    full_parent, monkeypatch, source_hash
):
    run, parent = full_parent

    def legacy(metadata):
        metadata["implementation"]["source_sha256"] = source_hash
        metadata["config"].pop("validation_files")

    parent = rewrite_full_parent_metadata(run, parent, legacy)
    forbid_cold_preparation(monkeypatch)
    pilot = prepare_data(pilot_run_for(run))
    assert pilot.identity["parent"] == parent.identity
    assert pilot.text_root == parent.path
    assert len(pilot.records_for("validation")) == 2
    assert pilot.records_for("test") == parent.records_for("test")


def test_subset_views_always_reference_full_parent_and_never_form_chains(full_parent, monkeypatch):
    run, parent = full_parent
    forbid_cold_preparation(monkeypatch)
    first = prepare_data(pilot_run_for(run))
    second_run = pilot_run_for(run, name="run_003", training_files=1, validation_files=1)
    raw_inventory = verify_inventory(run.root / "data/raw" / MANIFEST_REVISION)
    assert (
        data_module._compatible_full_parent(
            first.path, asdict(second_run.config.data), raw_inventory
        )
        is None
    )
    second = prepare_data(second_run)
    assert second.identity["parent"] == parent.identity
    assert second.identity["parent"] != first.identity
    assert second.text_root == parent.path
    assert len(second.records_for("train")) == len(second.records_for("validation")) == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"training_files": 5},
        {"validation_files": 6},
        {"seed": 43},
        {"clone_threshold": 0.95},
        {"minhash_permutations": 64},
        {"shingle_size": 6},
    ],
)
def test_inadequate_or_differently_filtered_parent_is_not_reused(full_parent, changes):
    run, parent = full_parent
    requested = pilot_run_for(run, **changes)
    raw_inventory = verify_inventory(run.root / "data/raw" / MANIFEST_REVISION)
    assert (
        data_module._compatible_full_parent(
            parent.path, asdict(requested.config.data), raw_inventory
        )
        is None
    )


def test_limited_validation_parent_cannot_supply_all_validation(tmp_path):
    config = replace(
        small_configuration(),
        data=DataConfig(training_files=4, validation_files=2, vocab_size=260),
    )
    run = FakeRun(tmp_path, config)
    raw_fixture(run, split_sources())
    parent = prepare_data(run)
    requested = pilot_run_for(run, validation_files=None)
    raw_inventory = verify_inventory(run.root / "data/raw" / MANIFEST_REVISION)
    assert (
        data_module._compatible_full_parent(
            parent.path, asdict(requested.config.data), raw_inventory
        )
        is None
    )


@pytest.mark.parametrize(
    "field,value", [("source_sha256", "f" * 64), ("python", "3.12.0"), ("datasketch", "0.0.0")]
)
def test_unreviewed_source_or_incompatible_runtime_parent_is_not_reused(full_parent, field, value):
    run, parent = full_parent

    def incompatible(metadata):
        metadata["implementation"][field] = value

    parent = rewrite_full_parent_metadata(run, parent, incompatible)
    requested = pilot_run_for(run)
    raw_inventory = verify_inventory(run.root / "data/raw" / MANIFEST_REVISION)
    assert (
        data_module._compatible_full_parent(
            parent.path, asdict(requested.config.data), raw_inventory
        )
        is None
    )


def test_raw_inventory_change_rejects_prior_parent_provenance(full_parent, monkeypatch):
    run, parent = full_parent
    raw = run.root / "data/raw" / MANIFEST_REVISION
    source = raw / "source/data" / parent.records[0]["source_path"]
    source.write_text(source.read_text() + "# raw source changed\n")
    write_json(raw / "inventory.json", _inventory(raw))
    forbid_cold_preparation(monkeypatch)
    with pytest.raises(ValueError, match="provenance/cache ID mismatch"):
        prepare_data(pilot_run_for(run))


@pytest.mark.parametrize("corrupt", ["text", "metadata"])
def test_corrupt_parent_is_rejected_before_subset_creation(full_parent, monkeypatch, corrupt):
    run, parent = full_parent
    if corrupt == "text":
        (parent.path / parent.records[0]["text_path"]).write_text("tampered = 1\n")
    else:
        metadata = read_json(parent.path / "metadata.json")
        metadata["implementation"]["source_sha256"] = "f" * 64
        write_json(parent.path / "metadata.json", metadata)
    forbid_cold_preparation(monkeypatch)
    with pytest.raises(ValueError, match="verification"):
        prepare_data(pilot_run_for(run))
    assert list(parent.path.parent.iterdir()) == [parent.path]


def test_reloading_subset_detects_parent_corruption(full_parent):
    run, parent = full_parent
    pilot_run = pilot_run_for(run)
    pilot = prepare_data(pilot_run)
    row = pilot.records[0]
    (parent.path / row["text_path"]).write_text("tampered = 1\n")
    with pytest.raises(ValueError, match="verification"):
        load_prepared_data(pilot_run)
    with pytest.raises(ValueError, match="checksum"):
        pilot.text(row)


def test_subset_pins_parent_inventory_even_if_parent_is_rehashed(full_parent):
    run, parent = full_parent
    pilot_run = pilot_run_for(run)
    prepare_data(pilot_run)
    metadata = read_json(parent.path / "metadata.json")
    metadata["extra_field"] = "parent metadata changed after child creation"
    write_json(parent.path / "metadata.json", metadata)
    write_json(parent.path / "inventory.json", _inventory(parent.path))
    assert verify_inventory(parent.path)
    with pytest.raises(ValueError, match="parent identity/provenance mismatch"):
        load_prepared_data(pilot_run)


@pytest.mark.parametrize("change", ["cache_id", "config", "parent", "selection", "records"])
def test_subset_metadata_and_selection_remain_bound_to_identity(full_parent, change):
    run, _ = full_parent
    pilot_run = pilot_run_for(run)
    pilot = prepare_data(pilot_run)
    metadata = read_json(pilot.path / "metadata.json")
    if change == "cache_id":
        metadata["cache_id"] = "e" * 64
    elif change == "config":
        metadata["config"]["training_files"] = 1
    elif change == "parent":
        metadata["parent"]["inventory"] = "e" * 64
    elif change == "selection":
        metadata["selection_implementation"] = "unreviewed-selection"
    else:
        records = read_json(pilot.path / "records.json")
        records[0]["split"] = "test" if records[0]["split"] == "train" else "train"
        write_json(pilot.path / "records.json", records)
    write_json(pilot.path / "metadata.json", metadata)
    # Rehashing file inventory must not bypass semantic identity/provenance checks.
    write_json(pilot.path / "inventory.json", _inventory(pilot.path))
    with pytest.raises(ValueError, match="identity|cache ID|selection|records differ"):
        data_module._load_prepared(pilot.path)


def test_cold_preparation_materializes_only_selected_python_sources(tmp_path, monkeypatch):
    run = FakeRun(tmp_path, small_configuration())
    raw_fixture(run, split_sources())
    writes = []
    original = Path.write_text

    def tracked(path, data, *args, **kwargs):
        if path.suffix == ".py":
            writes.append(path)
        return original(path, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", tracked)
    prepared = prepare_data(run)
    selected_ids = {row["id"] for row in prepared.records}
    assert len(selected_ids) < len(split_sources())
    assert len(writes) == len(selected_ids)
    assert {path.stem for path in writes} == selected_ids
    assert all(path.parent.name == "texts" for path in writes)


def test_cold_preparation_rejects_raw_change_between_analysis_and_materialization(
    tmp_path, monkeypatch
):
    run = FakeRun(tmp_path, small_configuration())
    raw_fixture(run, split_sources())
    original = data_module.MinHash
    mutated = False

    def mutate_after_parse(*args, **kwargs):
        nonlocal mutated
        if not mutated:
            raw = run.root / "data/raw" / MANIFEST_REVISION / "source"
            for source in raw.rglob("*.py"):
                source.write_text(source.read_text() + "# changed after analysis\n")
            mutated = True
        return original(*args, **kwargs)

    monkeypatch.setattr(data_module, "MinHash", mutate_after_parse)
    with pytest.raises(ValueError, match="Raw source changed during preparation"):
        prepare_data(run)
    assert mutated
    assert not list((run.root / "data/prepared").iterdir())


def test_prepare_stops_with_audit_when_validation_limit_cannot_be_met(tmp_path):
    run = FakeRun(
        tmp_path,
        replace(ExperimentConfig(), data=DataConfig(training_files=1, validation_files=3)),
    )
    raw_fixture(
        run,
        [
            (f"{repo_for(split)}/{index}.py", f"value_{split}_{index} = {index}\n", "mit")
            for split, count in (("train", 1), ("validation", 2), ("test", 1))
            for index in range(count)
        ],
    )
    with pytest.raises(ValueError, match="Need 3 validation files after filtering; found 2"):
        prepare_data(run)
    audit = read_json(run.path / "evaluation/data_preparation_failure.json")
    assert audit["requested_validation_files"] == 3
    assert audit["eligible_split_counts"]["validation"] == 2
    assert audit["selected_counts"] == {"train": 1, "validation": 2, "test": 1}
    assert not any((tmp_path / "data/prepared").iterdir())


def test_tokenizer_fits_only_training_and_literal_specials_roundtrip(tmp_path):
    run = FakeRun(tmp_path, small_configuration())
    data = prepared_fixture(
        tmp_path / "prepared",
        {
            "train": "value = 'café 🐍'\n",
            "validation": "held_out_secret = 5\n",
            "test": "unknown_name = 6\n",
        },
    )
    seen = []
    original_text = data.text

    def tracked(record):
        seen.append(record["split"])
        return original_text(record)

    data.text = tracked
    tokenizer = train_tokenizer(run, data)
    assert seen == ["train"]
    assert tokenizer.vocab_size == 260
    for text in ("  x=1\n\treturn x", "λ café 🐍", "<|pad|><|bos|><|eos|><|unk|>"):
        assert tokenizer.decode(tokenizer.encode(text)) == text
        assert all(token_id >= 4 for token_id in tokenizer.encode(text))
    assert verify_roundtrips(run, data, tokenizer)["status"] == "passed"


def test_tokenizer_checksum_rejects_corrupt_file(tmp_path):
    run = FakeRun(tmp_path, small_configuration())
    data = prepared_fixture(tmp_path / "prepared", {"train": "x = 1\n"})
    tokenizer = train_tokenizer(run, data)
    (tokenizer.path / "tokenizer.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        load_tokenizer(tokenizer.path)


def test_windows_predict_every_target_once_and_never_cross_files(tmp_path):
    run = FakeRun(tmp_path, small_configuration())
    data = prepared_fixture(
        tmp_path / "prepared",
        {"train": "first_variable = 123\n", "validation": "other = 456\n", "test": "third = 789\n"},
    )
    tokenizer = train_tokenizer(run, data)
    encoded = build_encoded_windows(run, data, tokenizer)
    for split in ("train", "validation", "test"):
        source = data.text(data.records_for(split)[0])
        expected = tokenizer.encode(source) + [tokenizer.eos_id]
        actual = []
        dataset = encoded.dataset(split)
        for item in dataset:
            valid = item["labels"] != -100
            actual.extend(item["labels"][valid].tolist())
            assert item["input_ids"].shape == (8,)
            assert item["input_ids"][~valid].eq(0).all()
        assert actual == expected
        assert actual.count(tokenizer.eos_id) == 1
    assert (
        int(encoded.train_token_counts.sum())
        == len(tokenizer.encode(data.text(data.records_for("train")[0]))) + 1
    )


def test_window_partial_tail_has_no_duplicated_target(tmp_path):
    np.asarray([1, 5, 6, 7, 8, 9, 2], dtype=np.uint32).tofile(tmp_path / "test.bin")
    np.save(tmp_path / "test_windows.npy", np.asarray([[0, 4], [4, 2]], dtype=np.int64))
    dataset = WindowDataset(tmp_path, "test", 4)
    assert dataset[0]["labels"].tolist() == [5, 6, 7, 8]
    assert dataset[1]["input_ids"].tolist() == [8, 9, 0, 0]
    assert dataset[1]["labels"].tolist() == [9, 2, -100, -100]


def test_token_metric_weights_by_tokens_and_reports_empty():
    metric = TokenMetrics()
    assert metric.result()["cross_entropy"] is None
    metric.add(2.0, 2, 2)
    metric.add(12.0, 1, 3)
    assert metric.result()["cross_entropy"] == pytest.approx(14 / 5)
    assert metric.result()["accuracy"] == pytest.approx(3 / 5)
    with pytest.raises(ValueError):
        metric.add(float("nan"), 0, 1)


def test_completion_metric_keeps_indentation_and_newline():
    assert completion_scores("  return x\n", "  return x\n")["exact_match"] == 1
    assert completion_scores("return x\n", "  return x\n")["exact_match"] == 0
    assert completion_scores("x", "x\n")["edit_similarity"] < 1
    assert "indentation_mismatch" in failure_labels("return x\n", "  return x\n", "newline")


def test_line_panel_is_fixed_and_at_most_one_per_file(tmp_path):
    data = prepared_fixture(
        tmp_path / "prepared", {"test": "# " + "a" * 80 + "\n\nvalue = 1\n# skip\nother = 2\n"}
    )
    tokenizer = SimpleNamespace(encode=lambda text: list(text))
    cases = line_cases(data, tokenizer, 42)
    assert cases == line_cases(data, tokenizer, 42)
    assert len(cases) == 1
    assert cases[0]["target"] in {"value = 1\n", "other = 2\n"}
    assert len(cases[0]["prefix"]) >= 64


def test_stream_panel_uses_length_rule_and_caps_context(tmp_path):
    data = prepared_fixture(tmp_path / "prepared", {"test": "x = 1\n" * 1000})
    tokenizer = SimpleNamespace(encode=lambda text: [5] * len(text), bos_id=1, eos_id=2)
    cases = streaming_cases(data, tokenizer, 42)
    assert len(cases) == 1
    assert len(cases[0]["tokens"]) == 4096
    assert cases[0]["original_tokens"] == 6002


def test_evaluation_never_executes_generated_source():
    # AST inspection establishes the API boundary; functional correctness is not a metric.
    import ast

    import src.evaluate as evaluation

    source = Path(evaluation.__file__).read_text()
    calls = [node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Call)]
    assert not any(
        isinstance(node.func, ast.Name) and node.func.id in {"exec", "eval", "compile"}
        for node in calls
    )
    assert "subprocess" not in source


def test_data_records_do_not_alias_a_source_file(tmp_path):
    data = prepared_fixture(tmp_path / "prepared", {"train": "x=1\n"})
    record = data.records[0]
    (data.path / record["text_path"]).write_text("x=2\n")
    with pytest.raises(ValueError, match="checksum"):
        data.text(record)


def test_machine_rows_do_not_accept_nonfinite_json(tmp_path):
    from src.evaluate import _Rows

    with _Rows(tmp_path, "rows") as rows:
        rows.append({"loss": 1.0, "labels": ["example"]})
        with pytest.raises(ValueError):
            rows.append({"loss": float("nan"), "labels": []})
    assert json.loads((tmp_path / "rows.jsonl").read_text().splitlines()[0])["loss"] == 1.0


def test_prepare_verifies_and_removes_near_clones(tmp_path):
    run = FakeRun(
        tmp_path,
        replace(ExperimentConfig(), data=DataConfig(training_files=1, validation_files=None)),
    )
    sources = [
        (f"{repo_for(split)}/unique.py", f"distinct_{index} = {index}\n", "mit")
        for index, split in enumerate(("train", "validation", "test"))
    ]
    clone = "\n".join(f"variable_{index} = {index}" for index in range(100)) + "\n"
    repo = repo_for("validation")
    sources.extend(
        [
            (f"{repo}/clone_a.py", clone, "mit"),
            (f"{repo}/clone_b.py", clone.replace("variable_50 = 50", "variable_50 = 9999"), "mit"),
        ]
    )
    raw_fixture(run, sources)
    data = prepare_data(run)
    audit = read_json(data.path / "audit.json")
    assert audit["exclusions"]["near_duplicate"] == 1
    edges = read_json(data.path / "clone_edges.json")
    assert any(edge["kind"] == "near" and edge["jaccard"] >= 0.9 for edge in edges)
    groups = {}
    for row in data.records:
        groups.setdefault(row["group"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in groups.values())


def test_line_panel_only_splits_physical_lf(tmp_path):
    source = "# " + "a" * 80 + "\nvalue = 'left\u2028right'\n"
    data = prepared_fixture(tmp_path / "prepared", {"test": source})
    tokenizer = SimpleNamespace(encode=lambda text: list(text))
    assert line_cases(data, tokenizer, 42)[0]["target"] == "value = 'left\u2028right'\n"
