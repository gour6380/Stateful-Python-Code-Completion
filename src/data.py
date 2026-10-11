"""Explicit, provenance-preserving Py150 acquisition and leakage-aware preparation.

Nothing is downloaded or prepared at import time. The original official split is
not reused: repository/verified-clone connected components define our split.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import shutil
import sys
import tarfile
import tempfile
import tokenize as python_tokenize
import urllib.request
from collections import Counter
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any

from datasketch import MinHash, MinHashLSH
from tqdm.auto import tqdm

from .runs import canonical_hash, read_json, sha256_file, write_json

MANIFEST_REVISION = "864c140cafa7e104948d7c04cba6d1b1b4d483a6"
SOURCE_URL = "https://files.sri.inf.ethz.ch/data/py150_files.tar.gz"
MANIFEST_BASE = (
    "https://raw.githubusercontent.com/google-research-datasets/eth_py150_open/" + MANIFEST_REVISION
)
LICENSES = frozenset(
    {"mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "isc", "cc0-1.0", "unlicense"}
)
IGNORED_TOKENS = {
    python_tokenize.ENCODING,
    python_tokenize.ENDMARKER,
    python_tokenize.COMMENT,
    python_tokenize.INDENT,
    python_tokenize.DEDENT,
    python_tokenize.NEWLINE,
    python_tokenize.NL,
}
DATA_PROTOCOL = "py150-python313-repository-clones-v1"
SUBSET_STORAGE = "subset_view"
SUBSET_SELECTION = "stable-hash-prefix-v1"
# Reviewed implementations of the same filtering, grouping and hash-prefix selection.
REVIEWED_PREPARATION_SOURCES = frozenset(
    {
        "79f6ce72b802f6898ade6ee92044b98142caef0154ddc62b95189b23e51907ab",
        "a5b6b8a920495e0c740ddac80c712adb6a609186104a2e8548d0d0880bead173",
    }
)


def stable_hash(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def safe_extract(archive: Path, destination: Path) -> None:
    """Extract regular files/directories only, without overwriting or following links."""
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive, "r:*") as handle:
        for member in handle:
            relative = PurePosixPath(member.name)
            if relative.is_absolute() or ".." in relative.parts or "\\" in member.name:
                raise ValueError(f"Unsafe archive member: {member.name!r}")
            if not (member.isdir() or member.isfile()):
                raise ValueError(f"Links/special archive members are forbidden: {member.name!r}")
            target = destination.joinpath(*relative.parts)
            if not target.resolve().is_relative_to(root):
                raise ValueError(f"Archive path escapes destination: {member.name!r}")
            if member.isdir():
                if target.exists() and not target.is_dir():
                    raise ValueError(f"Archive directory conflict: {member.name!r}")
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source = handle.extractfile(member)
            if source is None:
                raise ValueError(f"Unreadable archive member: {member.name!r}")
            with source, target.open("xb") as output:
                shutil.copyfileobj(source, output)
            if target.stat().st_size != member.size:
                raise ValueError(f"Archive member size mismatch: {member.name!r}")


def _download(url: str, path: Path) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "StatefulCodeCompletion/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, path.open("xb") as out:
        total = int(response.headers.get("Content-Length", 0)) or None
        digest = hashlib.sha256()
        size = 0
        with tqdm(total=total, unit="B", unit_scale=True, desc=path.name) as progress:
            while block := response.read(1024 * 1024):
                out.write(block)
                digest.update(block)
                size += len(block)
                progress.update(len(block))
        if total is not None and size != total:
            raise IOError(f"Incomplete download: {url}")
        return {
            "url": url,
            "resolved_url": response.url,
            "size": size,
            "sha256": digest.hexdigest(),
        }


def _inventory(root: Path) -> dict[str, Any]:
    return {
        path.relative_to(root).as_posix(): {
            "sha256": sha256_file(path),
            "size": path.stat().st_size,
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != root / "inventory.json"
    }


def verify_inventory(root: Path) -> dict[str, Any]:
    saved = read_json(root / "inventory.json")
    for relative, info in saved.items():
        path = root / relative
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != info["size"]
            or sha256_file(path) != info["sha256"]
        ):
            raise ValueError(f"Immutable cache failed verification: {path}")
    actual = {
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and p != root / "inventory.json"
    }
    if actual != set(saved):
        raise ValueError(f"Unexpected or missing files in immutable cache: {root}")
    return saved


def acquire_data(run: Any) -> Path:
    """Explicit notebook action: download the raw archive and revision-pinned manifests."""
    root = run.root / "data" / "raw" / MANIFEST_REVISION
    if root.exists():
        inventory = verify_inventory(root)
        run.bind(
            "raw_data", {"revision": MANIFEST_REVISION, "inventory": canonical_hash(inventory)}
        )
        return root
    root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".acquire-", dir=root.parent))
    try:
        downloads = {"py150_files.tar.gz": _download(SOURCE_URL, stage / "py150_files.tar.gz")}
        for name in ("train__manifest.json", "dev__manifest.json", "eval__manifest.json"):
            downloads[name] = _download(f"{MANIFEST_BASE}/{name}", stage / name)
        safe_extract(stage / "py150_files.tar.gz", stage / "outer")
        nested = list((stage / "outer").rglob("data.tar.gz"))
        if len(nested) != 1:
            raise ValueError(f"Expected exactly one nested data.tar.gz, found {len(nested)}")
        safe_extract(nested[0], stage / "source")
        write_json(
            stage / "acquisition.json",
            {
                "manifest_revision": MANIFEST_REVISION,
                "downloads": downloads,
                "checksum_origin": "Locally computed on first acquisition; creator supplies no pinned archive checksum.",
            },
        )
        inventory = _inventory(stage)
        write_json(stage / "inventory.json", inventory)
        os.rename(stage, root)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    run.bind("raw_data", {"revision": MANIFEST_REVISION, "inventory": canonical_hash(inventory)})
    return root


def normalize_source(raw: bytes) -> str:
    encoding, _ = python_tokenize.detect_encoding(io.BytesIO(raw).readline)
    text = raw.decode(encoding).replace("\r\n", "\n").replace("\r", "\n")
    ast.parse(text, mode="exec")
    return text


def lexical_shingles(text: str, width: int = 5) -> frozenset[str]:
    lexical = [
        f"{token.type}:{token.string}"
        for token in python_tokenize.generate_tokens(io.StringIO(text).readline)
        if token.type not in IGNORED_TOKENS
    ]
    if not lexical:
        return frozenset()
    spans = (
        [lexical]
        if len(lexical) < width
        else (lexical[i : i + width] for i in range(len(lexical) - width + 1))
    )
    return frozenset(
        hashlib.sha256(json.dumps(span, ensure_ascii=False).encode()).hexdigest() for span in spans
    )


class DisjointSet:
    def __init__(self) -> None:
        self.parents: dict[str, str] = {}

    def find(self, value: str) -> str:
        self.parents.setdefault(value, value)
        root = value
        while self.parents[root] != root:
            root = self.parents[root]
        while self.parents[value] != value:
            parent = self.parents[value]
            self.parents[value] = root
            value = parent
        return root

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        self.parents[max(a, b)] = min(a, b)


def partition_group(group: str, seed: int) -> str:
    fraction = int(stable_hash(seed, group), 16) / (1 << 256)
    return "train" if fraction < 0.8 else "validation" if fraction < 0.9 else "test"


def _source_relative(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError(f"Unsafe manifest path: {value!r}")
    parts = list(path.parts)
    while parts and parts[0] in {".", "data"}:
        parts.pop(0)
    if len(parts) < 3:
        raise ValueError(f"Manifest source lacks owner/repository/file: {value!r}")
    return "/".join(parts)


@dataclass
class PreparedData:
    path: Path
    records: list[dict[str, Any]]
    identity: dict[str, Any]
    text_root: Path | None = None

    def records_for(self, split: str) -> list[dict[str, Any]]:
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Unknown split: {split}")
        return [row for row in self.records if row["split"] == split]

    def text(self, record: dict[str, Any]) -> str:
        relative = PurePosixPath(record["text_path"])
        if relative.is_absolute() or ".." in relative.parts or "\\" in record["text_path"]:
            raise ValueError("Prepared source path must stay inside its verified cache")
        root = self.text_root if self.text_root is not None else self.path
        path = root / relative
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Prepared source path escapes its verified cache")
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"Prepared source checksum mismatch: {record['id']}")
        return path.read_text(encoding="utf-8")


def _valid_cache_id(value: Any) -> bool:
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def _full_cache_id(
    raw_inventory: dict[str, Any], config: dict[str, Any], implementation: dict[str, Any]
) -> str:
    return canonical_hash(
        {
            "protocol": DATA_PROTOCOL,
            "raw": raw_inventory,
            "config": config,
            "implementation": implementation,
        }
    )


def _selected_records(
    records: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    selected = []
    for split, limit in (
        ("train", config["training_files"]),
        ("validation", config.get("validation_files")),
        ("test", None),
    ):
        ranked = sorted(
            (row for row in records if row["split"] == split),
            key=lambda row: stable_hash(config["seed"], row["id"]),
        )
        selected.extend(ranked[:limit])
    return sorted(selected, key=lambda row: row["id"])


def _view_cache_id(parent: dict[str, Any], config: dict[str, Any]) -> str:
    return canonical_hash(
        {
            "protocol": DATA_PROTOCOL,
            "storage_mode": SUBSET_STORAGE,
            "parent": parent,
            "config": config,
            "selection_implementation": SUBSET_SELECTION,
        }
    )


def _compatible_full_parent(
    path: Path, config: dict[str, Any], raw_inventory: dict[str, Any]
) -> PreparedData | None:
    """Reuse only provenance-verified full caches from reviewed selection code."""
    if path.is_symlink():
        raise ValueError("Prepared cache cannot be a symbolic link")
    inventory = verify_inventory(path)
    metadata = read_json(path / "metadata.json")
    if metadata.get("storage_mode", "full") != "full":
        return None  # Views never become parents, preventing chains and cycles.
    implementation = metadata.get("implementation", {})
    reviewed = REVIEWED_PREPARATION_SOURCES | {sha256_file(Path(__file__))}
    if (
        metadata.get("protocol") != DATA_PROTOCOL
        or metadata.get("manifest_revision") != MANIFEST_REVISION
        or implementation.get("source_sha256") not in reviewed
        or implementation.get("python") != platform.python_version()
        or implementation.get("datasketch") != importlib.metadata.version("datasketch")
    ):
        return None
    saved = metadata.get("config", {})
    if any(
        saved.get(key) != config.get(key)
        for key in ("seed", "clone_threshold", "minhash_permutations", "shingle_size")
    ):
        return None
    expected_id = _full_cache_id(raw_inventory, saved, implementation)
    if metadata.get("cache_id") != expected_id or path.name != expected_id:
        raise ValueError(f"Prepared parent provenance/cache ID mismatch: {path}")
    parent = PreparedData(
        path,
        read_json(path / "records.json"),
        {
            "protocol": DATA_PROTOCOL,
            "cache_id": expected_id,
            "inventory": canonical_hash(inventory),
        },
    )
    audit = read_json(path / "audit.json")
    counts = Counter(row["split"] for row in parent.records)
    if dict(counts) != audit.get("selected_counts") or len(
        {row["id"] for row in parent.records}
    ) != len(parent.records):
        raise ValueError("Prepared parent selection/count audit mismatch")
    if counts["train"] != saved.get("training_files"):
        raise ValueError("Prepared parent training selection is incomplete")
    expected_validation = saved.get("validation_files")
    if expected_validation is None:
        expected_validation = audit["eligible_split_counts"].get("validation", 0)
    if counts["validation"] != expected_validation or counts["test"] != audit[
        "eligible_split_counts"
    ].get("test", 0):
        raise ValueError("Prepared parent held-out selection is incomplete")
    needed_validation = config.get("validation_files")
    if counts["train"] < config["training_files"]:
        return None
    if needed_validation is None:
        if counts["validation"] != audit["eligible_split_counts"].get("validation", 0):
            return None
    elif counts["validation"] < needed_validation:
        return None
    return parent


def _load_prepared(path: Path, *, raw_inventory: dict[str, Any] | None = None) -> PreparedData:
    if path.is_symlink():
        raise ValueError("Prepared cache cannot be a symbolic link")
    inventory = verify_inventory(path)
    metadata = read_json(path / "metadata.json")
    records = read_json(path / "records.json")
    if metadata.get("protocol") != DATA_PROTOCOL or metadata.get("cache_id") != path.name:
        raise ValueError("Prepared cache metadata identity mismatch")
    identity = {
        "protocol": DATA_PROTOCOL,
        "cache_id": metadata["cache_id"],
        "inventory": canonical_hash(inventory),
    }
    if metadata.get("storage_mode", "full") == SUBSET_STORAGE:
        parent_identity = metadata.get("parent", {})
        if not _valid_cache_id(parent_identity.get("cache_id")):
            raise ValueError("Invalid prepared parent cache ID")
        if metadata.get("selection_implementation") != SUBSET_SELECTION:
            raise ValueError("Unknown prepared subset selection implementation")
        if metadata.get("manifest_revision") != MANIFEST_REVISION:
            raise ValueError("Prepared subset manifest revision mismatch")
        if _view_cache_id(parent_identity, metadata["config"]) != metadata["cache_id"]:
            raise ValueError("Prepared subset cache ID mismatch")
        if raw_inventory is None:
            raw_inventory = verify_inventory(path.parent.parent / "raw" / MANIFEST_REVISION)
        parent = _compatible_full_parent(
            path.parent / parent_identity["cache_id"], metadata["config"], raw_inventory
        )
        if parent is None or parent.identity != parent_identity:
            raise ValueError("Prepared subset parent identity/provenance mismatch")
        if records != _selected_records(parent.records, metadata["config"]):
            raise ValueError("Prepared subset records differ from deterministic parent selection")
        identity["parent"] = parent.identity
        return PreparedData(path, records, identity, text_root=parent.path)
    if metadata.get("storage_mode", "full") != "full":
        raise ValueError("Unknown prepared cache storage mode")
    return PreparedData(path, records, identity)


def _prepare_subset_view(run: Any, raw_inventory: dict[str, Any]) -> PreparedData | None:
    root = run.root / "data" / "prepared"
    if not root.exists():
        return None
    config = asdict(run.config.data)
    for path in sorted(root.iterdir()):
        if not path.is_dir() or not _valid_cache_id(path.name):
            continue
        parent = _compatible_full_parent(path, config, raw_inventory)
        if parent is None:
            continue
        cache_id = _view_cache_id(parent.identity, config)
        output = root / cache_id
        if output.exists():
            return _load_prepared(output, raw_inventory=raw_inventory)
        records = _selected_records(parent.records, config)
        audit = read_json(parent.path / "audit.json")
        audit.update(
            requested_training_files=config["training_files"],
            requested_validation_files=config.get("validation_files"),
            selected_counts=dict(Counter(row["split"] for row in records)),
            repository_distribution=dict(Counter(row["repository"] for row in records)),
            repositories_by_split={
                split: len({row["repository"] for row in records if row["split"] == split})
                for split in ("train", "validation", "test")
            },
            selection_rule="Within unchanged parent repository/clone partitions, retain the SHA256(seed:file_id) prefix for each requested train/validation count and every eligible test file.",
            preparation_mode=SUBSET_STORAGE,
            parent_identity=parent.identity,
            parent_available_counts=dict(Counter(row["split"] for row in parent.records)),
            inherited_global_audit="Manifest/eligible counts, exclusions, clone edges and split method are inherited from the verified full parent; no parsing or clone detection was repeated.",
            clone_edges_parent_cache_id=parent.identity["cache_id"],
            source_texts="Referenced in the verified parent; no source text files copied or created.",
        )
        stage = Path(tempfile.mkdtemp(prefix=".subset-", dir=root))
        try:
            write_json(stage / "records.json", records)
            write_json(stage / "audit.json", audit)
            write_json(
                stage / "metadata.json",
                {
                    "cache_id": cache_id,
                    "protocol": DATA_PROTOCOL,
                    "config": config,
                    "manifest_revision": MANIFEST_REVISION,
                    "storage_mode": SUBSET_STORAGE,
                    "parent": parent.identity,
                    "selection_implementation": SUBSET_SELECTION,
                },
            )
            write_json(stage / "inventory.json", _inventory(stage))
            os.rename(stage, output)
        except BaseException:
            shutil.rmtree(stage, ignore_errors=True)
            raise
        return _load_prepared(output, raw_inventory=raw_inventory)
    return None


def prepare_data(run: Any) -> PreparedData:
    """Parse, deduplicate, group, split and select; never triggers acquisition."""
    if sys.version_info[:2] != (3, 13):
        raise RuntimeError("Dataset compatibility filtering requires Python 3.13 exactly.")
    raw = run.root / "data" / "raw" / MANIFEST_REVISION
    raw_inventory = verify_inventory(raw)
    config = run.config.data
    implementation = {
        "python": platform.python_version(),
        "datasketch": importlib.metadata.version("datasketch"),
        "source_sha256": sha256_file(Path(__file__)),
    }
    cache_id = canonical_hash(
        {
            "protocol": DATA_PROTOCOL,
            "raw": raw_inventory,
            "config": asdict(config),
            "implementation": implementation,
        }
    )
    output = run.root / "data" / "prepared" / cache_id
    if output.exists():
        prepared = _load_prepared(output)
        run.bind("prepared_data", prepared.identity)
        return prepared
    prepared = _prepare_subset_view(run, raw_inventory)
    if prepared is not None:
        run.bind("prepared_data", prepared.identity)
        return prepared
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".prepare-", dir=output.parent))
    (stage / "texts").mkdir()
    (stage / "shingles").mkdir()
    exclusions: Counter[str] = Counter()
    repositories = DisjointSet()
    files = DisjointSet()
    try:
        manifest: dict[str, dict[str, Any]] = {}
        for name in ("train__manifest.json", "dev__manifest.json", "eval__manifest.json"):
            for entry in read_json(raw / name):
                relative = _source_relative(entry["filepath"])
                license_name = entry["license"].lower()
                if relative in manifest and manifest[relative]["license"] != license_name:
                    raise ValueError(f"Conflicting manifest licenses for {relative}")
                manifest[relative] = {
                    "license": license_name,
                    "original_split": name.split("__")[0],
                }
        source_lookup: dict[str, Path] = {}
        for path in sorted((raw / "source").rglob("*")):
            if path.is_file():
                try:
                    key = _source_relative(path.relative_to(raw / "source").as_posix())
                except ValueError:
                    continue
                if key in source_lookup:
                    raise ValueError(f"Ambiguous raw source path: {key}")
                source_lookup[key] = path
        records: dict[str, dict[str, Any]] = {}
        exact: dict[str, str] = {}
        clone_edges: list[dict[str, Any]] = []
        for relative, entry in tqdm(sorted(manifest.items()), desc="Parse Python 3.13"):
            if entry["license"] not in LICENSES:
                exclusions["license"] += 1
                continue
            if relative not in source_lookup:
                raise FileNotFoundError(f"Manifest entry missing from raw archive: {relative}")
            try:
                text = normalize_source(source_lookup[relative].read_bytes())
                shingles = lexical_shingles(text, config.shingle_size)
            except (SyntaxError, UnicodeError, LookupError, python_tokenize.TokenError, ValueError):
                exclusions["encoding_or_python313_parse"] += 1
                continue
            if not shingles:
                exclusions["no_lexical_tokens"] += 1
                continue
            repo = "/".join(relative.split("/")[:2])
            repositories.find(repo)
            file_id = hashlib.sha256(relative.encode()).hexdigest()
            digest = hashlib.sha256(text.encode()).hexdigest()
            files.find(file_id)
            if digest in exact:
                representative = exact[digest]
                repositories.union(repo, records[representative]["repository"])
                clone_edges.append(
                    {
                        "left": relative,
                        "right": records[representative]["source_path"],
                        "kind": "exact",
                        "jaccard": 1.0,
                    }
                )
                exclusions["exact_duplicate"] += 1
                continue
            exact[digest] = file_id
            write_json(stage / "shingles" / f"{file_id}.json", sorted(shingles))
            records[file_id] = {
                "id": file_id,
                "repository": repo,
                "source_path": relative,
                "text_path": f"texts/{file_id}.py",
                "sha256": digest,
                "license": entry["license"],
                "original_split": entry["original_split"],
                "characters": len(text),
                "bytes": len(text.encode()),
                "lines": text.count("\n") + int(bool(text) and not text.endswith("\n")),
            }
        # Candidate retrieval is probabilistic; exact similarity verifies every edge.
        # Disk-backed shingles avoid retaining the corpus's Python string sets in RAM.
        lsh = MinHashLSH(
            threshold=max(0.1, config.clone_threshold - 0.1), num_perm=config.minhash_permutations
        )

        @lru_cache(maxsize=64)
        def shingles_for(file_id: str) -> frozenset[str]:
            return frozenset(read_json(stage / "shingles" / f"{file_id}.json"))

        for file_id in tqdm(sorted(records), desc="Verify clone candidates"):
            shingles = shingles_for(file_id)
            signature = MinHash(num_perm=config.minhash_permutations, seed=config.seed)
            # Chunking avoids a huge num_tokens × permutations temporary allocation.
            batch: list[bytes] = []
            for shingle in sorted(shingles):
                batch.append(bytes.fromhex(shingle))
                if len(batch) == 256:
                    signature.update_batch(batch)
                    batch.clear()
            if batch:
                signature.update_batch(batch)
            for candidate in sorted(lsh.query(signature)):
                other = shingles_for(candidate)
                jaccard = len(shingles & other) / len(shingles | other)
                if jaccard >= config.clone_threshold:
                    files.union(file_id, candidate)
                    repositories.union(
                        records[file_id]["repository"], records[candidate]["repository"]
                    )
                    clone_edges.append(
                        {
                            "left": records[file_id]["source_path"],
                            "right": records[candidate]["source_path"],
                            "kind": "near",
                            "jaccard": jaccard,
                        }
                    )
            lsh.insert(file_id, signature)
        retained = []
        for file_id, record in sorted(records.items()):
            if files.find(file_id) != file_id:
                exclusions["near_duplicate"] += 1
                continue
            group = repositories.find(record["repository"])
            retained.append(
                {**record, "group": group, "split": partition_group(group, config.seed)}
            )
        eligible_counts = Counter(row["split"] for row in retained)
        selected_train = sorted(
            (r for r in retained if r["split"] == "train"),
            key=lambda row: stable_hash(config.seed, row["id"]),
        )[: config.training_files]
        selected_validation = sorted(
            (r for r in retained if r["split"] == "validation"),
            key=lambda row: stable_hash(config.seed, row["id"]),
        )[: config.validation_files]
        selected_test = [r for r in retained if r["split"] == "test"]
        audit = {
            "manifest_files": len(manifest),
            "eligible_split_counts": dict(eligible_counts),
            "exclusions": dict(exclusions),
            "clone_edges": len(clone_edges),
            "requested_training_files": config.training_files,
            "requested_validation_files": config.validation_files,
            "selected_counts": {
                "train": len(selected_train),
                "validation": len(selected_validation),
                "test": len(selected_test),
            },
            "split_method": "SHA256(seed:connected-component canonical repository), thresholds .8/.9; approximate file proportions",
            "selection_rule": "Within the unchanged repository/clone partitions, select the first requested training and validation counts ordered by SHA256(seed:file_id); validation_files=None retains all validation. Retain every eligible test file.",
            "clone_limit": "Deterministic MinHash candidate retrieval can miss near duplicates; candidates verified with exact Jaccard.",
        }
        if len(selected_train) != config.training_files:
            write_json(run.path / "evaluation" / "data_preparation_failure.json", audit)
            raise ValueError(
                f"Need {config.training_files:,} training files after filtering; found {eligible_counts['train']:,}. See data_preparation_failure.json."
            )
        if (
            config.validation_files is not None
            and len(selected_validation) != config.validation_files
        ):
            write_json(run.path / "evaluation" / "data_preparation_failure.json", audit)
            raise ValueError(
                f"Need {config.validation_files:,} validation files after filtering; found {eligible_counts['validation']:,}. See data_preparation_failure.json."
            )
        selected = sorted(
            selected_train + selected_validation + selected_test,
            key=lambda row: row["id"],
        )
        # Global shingles establish clone groups, but source files are materialized
        # only for the final selection. Detect raw-source changes between passes.
        for record in tqdm(selected, desc="Materialize selected source"):
            text = normalize_source(source_lookup[record["source_path"]].read_bytes())
            if hashlib.sha256(text.encode()).hexdigest() != record["sha256"]:
                raise ValueError(f"Raw source changed during preparation: {record['source_path']}")
            (stage / record["text_path"]).write_text(text, encoding="utf-8", newline="\n")
        shutil.rmtree(stage / "shingles")
        audit.update(
            {
                "selected_counts": dict(Counter(r["split"] for r in selected)),
                "repository_distribution": dict(Counter(r["repository"] for r in selected)),
                "repositories_by_split": {
                    split: len({r["repository"] for r in selected if r["split"] == split})
                    for split in ("train", "validation", "test")
                },
            }
        )
        if any(
            not any(row["split"] == split for row in selected) for split in ("validation", "test")
        ):
            raise ValueError(
                "Repository/clone grouping left an empty held-out partition; no silent resplit is permitted."
            )
        write_json(stage / "audit.json", audit)
        write_json(stage / "clone_edges.json", clone_edges)
        write_json(stage / "records.json", selected)
        write_json(
            stage / "metadata.json",
            {
                "cache_id": cache_id,
                "protocol": DATA_PROTOCOL,
                "config": asdict(config),
                "manifest_revision": MANIFEST_REVISION,
                "implementation": implementation,
            },
        )
        write_json(stage / "inventory.json", _inventory(stage))
        os.rename(stage, output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    prepared = _load_prepared(output)
    run.bind("prepared_data", prepared.identity)
    return prepared


def load_prepared_data(run: Any) -> PreparedData:
    bindings = read_json(run.path / "config" / "bindings.json")
    identity = bindings["prepared_data"]
    prepared = _load_prepared(run.root / "data" / "prepared" / identity["cache_id"])
    if prepared.identity != identity:
        raise ValueError("Run/prepared-data identity mismatch")
    return prepared


def data_audit(run: Any, data: PreparedData) -> dict[str, Any]:
    report = read_json(data.path / "audit.json")
    write_json(run.path / "evaluation" / "data_audit.json", report)
    return report


def training_eda(run: Any, data: PreparedData) -> dict[str, Any]:
    import numpy as np

    rows = data.records_for("train")
    report = {
        "split": "train",
        "files": len(rows),
        "licenses": dict(Counter(r["license"] for r in rows)),
        "lengths": {
            key: dict(
                zip(
                    ("min", "p25", "median", "p75", "p95", "max"),
                    np.percentile([r[key] for r in rows], [0, 25, 50, 75, 95, 100]).tolist(),
                    strict=True,
                )
            )
            for key in ("characters", "bytes", "lines")
        },
    }
    write_json(run.path / "evaluation" / "training_eda.json", report)
    return report
