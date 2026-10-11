"""Immutable experiment settings; importing this module performs no work."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class DataConfig:
    training_files: int = 5_000
    validation_files: int | None = 1_000
    seed: int = 42
    vocab_size: int = 8192
    minhash_permutations: int = 128
    shingle_size: int = 5
    clone_threshold: float = 0.9

    def __post_init__(self) -> None:
        if self.validation_files is not None and (
            type(self.validation_files) is not int or self.validation_files < 1
        ):
            raise ValueError("validation_files must be a positive integer or None")
        if min(self.training_files, self.minhash_permutations, self.shingle_size) < 1:
            raise ValueError("Data counts must be positive")
        if self.vocab_size < 260 or not 0 < self.clone_threshold <= 1:
            raise ValueError("Invalid vocabulary size or clone threshold")


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 8192
    layers: int = 8
    width: int = 512
    heads: int = 8
    ff_width: int = 1536
    context_length: int = 1024
    dropout: float = 0.1
    norm_eps: float = 1e-5
    rope_base: float = 10_000.0

    def __post_init__(self) -> None:
        if min(self.vocab_size, self.layers, self.width, self.heads, self.ff_width) < 1:
            raise ValueError("Model dimensions must be positive")
        if self.width % self.heads or (self.width // self.heads) % 2:
            raise ValueError("Width must divide into even RoPE head dimensions")
        if self.context_length < 2 or not 0 <= self.dropout < 1:
            raise ValueError("Invalid context length or dropout")
        if self.norm_eps <= 0 or self.rope_base <= 1:
            raise ValueError("Invalid normalization or RoPE constant")


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    max_epochs: int = 15
    min_epochs: int = 5
    patience: int = 3
    min_delta: float = 1e-4
    batch_size: int = 4
    accumulation_steps: int = 8
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_fraction: float = 0.05
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.1
    gradient_clip: float = 1.0
    checkpoint_interval: int = 100
    validation_batch_size: int | None = 16
    cpu_threads: int | None = 6
    num_workers: int = 0
    prefer_metal: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "betas", tuple(self.betas))
        for name in ("validation_batch_size", "cpu_threads"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be a positive integer or None")
        if type(self.num_workers) is not int or self.num_workers < 0:
            raise ValueError("num_workers must be a nonnegative integer")
        if type(self.prefer_metal) is not bool:
            raise ValueError("prefer_metal must be a boolean")
        if not 1 <= self.min_epochs <= self.max_epochs:
            raise ValueError("Require 1 <= min_epochs <= max_epochs")
        if (
            min(self.patience, self.batch_size, self.accumulation_steps, self.checkpoint_interval)
            < 1
        ):
            raise ValueError("Training counts must be positive")
        if not 0 < self.min_learning_rate <= self.learning_rate or self.eps <= 0:
            raise ValueError("Invalid learning rate or optimizer epsilon")
        if self.min_delta < 0 or self.weight_decay < 0 or self.gradient_clip <= 0:
            raise ValueError("Invalid training regularization or selection threshold")
        if not 0 <= self.warmup_fraction < 1 or len(self.betas) != 2:
            raise ValueError("Invalid warmup or Adam betas")
        if not all(0 <= value < 1 for value in self.betas):
            raise ValueError("Adam betas must be in [0, 1)")


@dataclass(frozen=True)
class ServiceConfig:
    budgets: tuple[int, ...] = (256, 512, 1024)
    sink_tokens: int = 4
    max_sessions: int = 4
    queue_size: int = 8
    deadline_seconds: float = 120.0
    max_source_bytes: int = 1_048_576
    max_new_tokens: int = 64
    host: str = "127.0.0.1"
    port: int = 8000

    def __post_init__(self) -> None:
        object.__setattr__(self, "budgets", tuple(self.budgets))
        if not self.budgets or len(set(self.budgets)) != len(self.budgets):
            raise ValueError("Budgets must be nonempty and unique")
        if self.sink_tokens < 0 or min(self.budgets) <= self.sink_tokens:
            raise ValueError("Every total cache budget must exceed the sink count")
        if min(self.max_sessions, self.queue_size, self.max_source_bytes, self.max_new_tokens) < 1:
            raise ValueError("Service limits must be positive")
        if self.deadline_seconds <= 0 or not 1 <= self.port <= 65535:
            raise ValueError("Invalid deadline or port")
        if self.host not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("This experiment serves localhost only")


@dataclass(frozen=True)
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    service: ServiceConfig = field(default_factory=ServiceConfig)

    def __post_init__(self) -> None:
        if self.data.vocab_size != self.model.vocab_size:
            raise ValueError("Tokenizer and model vocabulary sizes must match")
        if max(self.service.budgets) > self.model.context_length:
            raise ValueError("Service budgets cannot exceed trained context length")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ExperimentConfig:
        if set(value) != {"data", "model", "training", "service"}:
            raise ValueError("Configuration must have exactly data/model/training/service")
        return cls(
            # Older saved experiments used the entire validation partition.
            data=DataConfig(**{"validation_files": None, **value["data"]}),
            model=ModelConfig(**value["model"]),
            training=TrainingConfig(**value["training"]),
            service=ServiceConfig(**value["service"]),
        )
