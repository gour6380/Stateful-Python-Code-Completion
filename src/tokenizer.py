"""Train-only byte BPE, exact UTF-8 byte access, and disk-backed causal windows."""

from __future__ import annotations

import codecs
import importlib.metadata
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from .data import PreparedData, _inventory, load_prepared_data, verify_inventory
from .runs import canonical_hash, read_json, sha256_file, write_json

SPECIAL_TOKENS = ["<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>"]
TOKENIZER_PROTOCOL = "python-byte-bpe-v1"
ENCODING_PROTOCOL = "file-local-every-target-once-v1"


def byte_decoder() -> dict[str, int]:
    """Inverse of the standard GPT-2/ByteLevel byte-to-Unicode bijection."""
    byte_values = (
        list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    )
    unicode_values = byte_values.copy()
    extra = 0
    for value in range(256):
        if value not in byte_values:
            byte_values.append(value)
            unicode_values.append(256 + extra)
            extra += 1
    return {chr(point): value for value, point in zip(byte_values, unicode_values, strict=True)}


class CodeTokenizer:
    pad_id = 0
    bos_id = 1
    eos_id = 2
    unk_id = 3

    def __init__(self, backend: Tokenizer, path: Path, identity: dict[str, Any]):
        self.backend = backend
        # Literal spellings inside Python string literals must remain ordinary text.
        self.backend.encode_special_tokens = True
        self.path = Path(path)
        self.identity = identity
        inverse = byte_decoder()
        self._bytes = {
            token_id: b"" if token_id < 4 else bytes(inverse[ch] for ch in token)
            for token, token_id in backend.get_vocab().items()
        }
        if [backend.token_to_id(token) for token in SPECIAL_TOKENS] != list(range(4)):
            raise ValueError("Tokenizer special-token IDs differ from model contract")

    def encode(self, text: str) -> list[int]:
        ids = self.backend.encode(text, add_special_tokens=False).ids
        if any(token < 4 for token in ids):
            raise ValueError("Source text unexpectedly encoded as reserved special tokens")
        return ids

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.token_bytes(token) for token in ids).decode("utf-8", errors="replace")

    def token_bytes(self, token_id: int) -> bytes:
        return self._bytes[token_id]

    @property
    def vocab_size(self) -> int:
        return self.backend.get_vocab_size()


def train_tokenizer(run: Any, data: PreparedData) -> CodeTokenizer:
    implementation = {
        "tokenizers": importlib.metadata.version("tokenizers"),
        "source_sha256": sha256_file(Path(__file__)),
    }
    cache_id = canonical_hash(
        {
            "protocol": TOKENIZER_PROTOCOL,
            "data": data.identity,
            "vocab_size": run.config.data.vocab_size,
            "implementation": implementation,
        }
    )
    output = run.root / "data" / "tokenizers" / cache_id
    if output.exists():
        tokenizer = load_tokenizer(output)
        run.bind("tokenizer", tokenizer.identity)
        return tokenizer
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".tokenizer-", dir=output.parent))
    try:
        backend = Tokenizer(models.BPE(unk_token=SPECIAL_TOKENS[3]))
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
        backend.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=run.config.data.vocab_size,
            min_frequency=2,
            special_tokens=SPECIAL_TOKENS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=True,
        )
        records = data.records_for("train")
        backend.train_from_iterator(
            (data.text(row) for row in records), trainer=trainer, length=len(records)
        )
        if backend.get_vocab_size() != run.config.data.vocab_size:
            raise ValueError(
                f"Tokenizer produced {backend.get_vocab_size()} entries, expected {run.config.data.vocab_size}; no silent vocabulary shrink."
            )
        backend.encode_special_tokens = True
        backend.save(str(stage / "tokenizer.json"))
        identity = {
            "protocol": TOKENIZER_PROTOCOL,
            "cache_id": cache_id,
            "data": data.identity,
            "sha256": sha256_file(stage / "tokenizer.json"),
            "vocab_size": backend.get_vocab_size(),
            "special_ids": dict(zip(SPECIAL_TOKENS, range(4), strict=True)),
            "fitted_file_ids_sha256": canonical_hash([row["id"] for row in records]),
            "fit_split": "train",
            "implementation": implementation,
        }
        write_json(stage / "metadata.json", {"identity": identity})
        os.rename(stage, output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    tokenizer = load_tokenizer(output)
    run.bind("tokenizer", tokenizer.identity)
    return tokenizer


def load_tokenizer(path: Path | str) -> CodeTokenizer:
    directory = Path(path)
    if directory.is_file():
        directory = directory.parent
    metadata = read_json(directory / "metadata.json")
    identity = metadata["identity"]
    if sha256_file(directory / "tokenizer.json") != identity["sha256"]:
        raise ValueError("Tokenizer checksum mismatch")
    backend = Tokenizer.from_file(str(directory / "tokenizer.json"))
    if backend.get_vocab_size() != identity["vocab_size"]:
        raise ValueError("Tokenizer vocabulary identity mismatch")
    return CodeTokenizer(backend, directory, identity)


def verify_roundtrips(run: Any, data: PreparedData, tokenizer: CodeTokenizer) -> dict[str, Any]:
    # Only training examples enter this development-time diagnostic.
    probes = [
        "",
        "  x = 1\n\treturn x\n",
        "# λ 🐍\nname = 'café'\n",
        "<|pad|><|bos|><|eos|><|unk|>",
    ]
    probes.extend(data.text(record) for record in data.records_for("train")[:128])
    for text in probes:
        ids = tokenizer.encode(text)
        if tokenizer.decode(ids) != text:
            raise ValueError("Tokenizer text round-trip failed")
        incremental = codecs.getincrementaldecoder("utf-8")()
        reconstructed = "".join(
            incremental.decode(tokenizer.token_bytes(token), final=False) for token in ids
        )
        reconstructed += incremental.decode(b"", final=True)
        if reconstructed != text:
            raise ValueError("Tokenizer incremental-byte round-trip failed")
    receipt = {
        "status": "passed",
        "probes": len(probes),
        "tokenizer": tokenizer.identity,
        "data_split": "train",
    }
    write_json(run.path / "evaluation" / "tokenizer_roundtrips.json", receipt)
    return receipt


class WindowDataset(Dataset):
    """Copy only one small window from a read-only token memmap per sample."""

    def __init__(self, path: Path, split: str, context_length: int):
        self.tokens = np.memmap(path / f"{split}.bin", dtype=np.uint32, mode="r")
        self.windows = np.load(path / f"{split}_windows.npy", mmap_mode="r", allow_pickle=False)
        self.context_length = context_length

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        offset, length = map(int, self.windows[index])
        inputs = torch.zeros(self.context_length, dtype=torch.long)
        labels = torch.full((self.context_length,), -100, dtype=torch.long)
        inputs[:length] = torch.from_numpy(
            np.array(self.tokens[offset : offset + length], dtype=np.int64)
        )
        labels[:length] = torch.from_numpy(
            np.array(self.tokens[offset + 1 : offset + length + 1], dtype=np.int64)
        )
        return {"input_ids": inputs, "labels": labels}


@dataclass
class EncodedData:
    path: Path
    identity: dict[str, Any]
    tokenizer: CodeTokenizer
    data: PreparedData
    context_length: int

    def dataset(self, split: str) -> WindowDataset:
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Unknown split: {split}")
        return WindowDataset(self.path, split, self.context_length)

    @property
    def train_token_counts(self) -> np.ndarray:
        return np.load(self.path / "train_token_counts.npy", allow_pickle=False)


def _load_encoded(path: Path, data: PreparedData, tokenizer: CodeTokenizer) -> EncodedData:
    inventory = verify_inventory(path)
    metadata = read_json(path / "metadata.json")
    if metadata["data"] != data.identity or metadata["tokenizer"] != tokenizer.identity:
        raise ValueError("Encoded cache data/tokenizer mismatch")
    identity = {
        "protocol": ENCODING_PROTOCOL,
        "cache_id": metadata["cache_id"],
        "inventory": canonical_hash(inventory),
        "data": data.identity,
        "tokenizer": tokenizer.identity,
    }
    return EncodedData(path, identity, tokenizer, data, metadata["context_length"])


def build_encoded_windows(run: Any, data: PreparedData, tokenizer: CodeTokenizer) -> EncodedData:
    context = run.config.model.context_length
    implementation = {"source_sha256": sha256_file(Path(__file__)), "numpy": np.__version__}
    cache_id = canonical_hash(
        {
            "protocol": ENCODING_PROTOCOL,
            "data": data.identity,
            "tokenizer": tokenizer.identity,
            "context": context,
            "implementation": implementation,
        }
    )
    output = run.root / "data" / "encoded" / cache_id
    if output.exists():
        encoded = _load_encoded(output, data, tokenizer)
        run.bind("encoded", encoded.identity)
        return encoded
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".encode-", dir=output.parent))
    try:
        counts = np.zeros(tokenizer.vocab_size, dtype=np.int64)
        lengths: list[dict[str, Any]] = []
        for split in ("train", "validation", "test"):
            windows: list[tuple[int, int]] = []
            offset = 0
            with (stage / f"{split}.bin").open("xb") as stream:
                for record in tqdm(data.records_for(split), desc=f"Encode {split}"):
                    ids = np.asarray(
                        [tokenizer.bos_id, *tokenizer.encode(data.text(record)), tokenizer.eos_id],
                        dtype=np.uint32,
                    )
                    ids.tofile(stream)
                    if split == "train":
                        counts += np.bincount(ids[1:], minlength=tokenizer.vocab_size)
                    windows.extend(
                        (offset + start, min(context, len(ids) - 1 - start))
                        for start in range(0, len(ids) - 1, context)
                    )
                    lengths.append(
                        {
                            "id": record["id"],
                            "split": split,
                            "tokens_with_boundaries": len(ids),
                            "offset": offset,
                        }
                    )
                    offset += len(ids)
            if not windows:
                raise ValueError(f"No encoded windows in {split}")
            np.save(
                stage / f"{split}_windows.npy",
                np.asarray(windows, dtype=np.int64),
                allow_pickle=False,
            )
        np.save(stage / "train_token_counts.npy", counts, allow_pickle=False)
        write_json(stage / "file_lengths.json", lengths)
        write_json(
            stage / "metadata.json",
            {
                "cache_id": cache_id,
                "protocol": ENCODING_PROTOCOL,
                "context_length": context,
                "data": data.identity,
                "tokenizer": tokenizer.identity,
                "implementation": implementation,
            },
        )
        write_json(stage / "inventory.json", _inventory(stage))
        os.rename(stage, output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    encoded = _load_encoded(output, data, tokenizer)
    run.bind("encoded", encoded.identity)
    return encoded


def load_encoded_data(run: Any) -> EncodedData:
    bindings = read_json(run.path / "config" / "bindings.json")
    data = load_prepared_data(run)
    tokenizer = load_tokenizer(run.root / "data" / "tokenizers" / bindings["tokenizer"]["cache_id"])
    identity = bindings["encoded"]
    encoded = _load_encoded(run.root / "data" / "encoded" / identity["cache_id"], data, tokenizer)
    if encoded.identity != identity:
        raise ValueError("Run/encoded-data identity mismatch")
    return encoded
