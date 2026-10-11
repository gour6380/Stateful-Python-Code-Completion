"""Canonical, request-chunk-independent bounded-context inference policies."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F

from .model import CodeTransformer, ModelCache


def _check_cancel(cancel) -> None:
    if cancel is not None and bool(cancel() if callable(cancel) else cancel.is_set()):
        raise InterruptedError("Cache update cancelled")


class CachePolicy:
    def __init__(self, model: CodeTransformer, policy: str, budget: int, sink_tokens: int = 4):
        if policy not in {"stateless", "exact", "sink"}:
            raise ValueError("Unknown cache policy")
        if not 1 <= budget <= model.config.context_length:
            raise ValueError("Cache budget must fit the architecture context")
        if not 0 <= sink_tokens < budget:
            raise ValueError("Sink count must be smaller than cache budget")
        self.model, self.policy, self.budget = model, policy, budget
        self.sink_tokens = sink_tokens
        self._tokens: list[int] = []
        self.cache: ModelCache | None = None
        self.logits: Tensor | None = None
        self._has_evicted = False
        self._stats = {
            "full_recomputations": 0,
            "canonical_replays": 0,
            "reused_tokens": 0,
            "evictions": 0,
        }

    @property
    def tokens(self) -> list[int]:
        return list(self._tokens)

    @property
    def cache_size(self) -> int:
        return 0 if self.cache is None else self.cache.length

    def clone(self) -> CachePolicy:
        other = CachePolicy(self.model, self.policy, self.budget, self.sink_tokens)
        other._tokens = list(self._tokens)
        other.cache = None if self.cache is None else self.cache.clone()
        other.logits = None if self.logits is None else self.logits.clone()
        other._stats = dict(self._stats)
        other._has_evicted = self._has_evicted
        return other

    def diagnostics(self) -> dict:
        total = len(self._tokens)
        if self.policy == "sink" and total > self.budget:
            indices = list(range(self.sink_tokens)) + list(
                range(total - self.budget + self.sink_tokens, total)
            )
        else:
            indices = list(range(max(0, total - self.budget), total))
        return {
            "policy": self.policy,
            "budget": self.budget,
            "sink_tokens": self.sink_tokens,
            "canonical_tokens": total,
            "cache_tokens": self.cache_size,
            "current_history_has_evicted": self._has_evicted,
            "cache_bytes": 0 if self.cache is None else self.cache.bytes,
            "retained_token_indices": indices if self.policy != "stateless" else [],
            **self._stats,
        }

    @torch.inference_mode()
    def set_tokens(self, tokens: list[int], cancel=None) -> Tensor:
        _check_cancel(cancel)
        tokens = list(tokens)
        if not tokens or any(
            isinstance(t, bool)
            or not isinstance(t, int)
            or not 0 < t < self.model.config.vocab_size
            for t in tokens
        ):
            raise ValueError("Canonical history must contain non-PAD vocabulary token IDs")
        if self.model.training:
            raise RuntimeError("Call model.eval() before constructing inference state")
        if tokens == self._tokens and self.logits is not None:
            # Stateless deliberately pays for each requested prediction.
            if self.policy != "stateless":
                return self.logits
        previous = self.cache, self.logits, dict(self._stats), self._has_evicted
        try:
            if self.policy == "stateless":
                self.logits, _ = self.model.prefill(tokens[-self.budget :])
                self.cache = None
                self._stats["full_recomputations"] += 1
            elif self.policy == "exact":
                self._set_exact(tokens, cancel)
            else:
                self._set_sink(tokens, cancel)
            _check_cancel(cancel)
            if not torch.isfinite(self.logits).all():
                raise RuntimeError("Inference produced non-finite logits; cache update rolled back")
            _check_cancel(cancel)
        except BaseException:
            # No cache helper mutates tensors in place: retaining the prior references
            # makes cancellation, device failures and notebook interrupts transactional.
            self.cache, self.logits, self._stats, self._has_evicted = previous
            raise
        self._tokens = tokens
        return self.logits

    def _set_exact(self, tokens: list[int], cancel=None) -> None:
        old_start = max(0, len(self._tokens) - self.budget)
        new_start = max(0, len(tokens) - self.budget)
        effective = tokens[new_start:]
        common = 0
        if self.cache is not None and old_start == new_start:
            for old, new in zip(self._tokens[old_start:], effective):
                if old != new:
                    break
                common += 1
        # Reprocess the final token after deletion so its next-token logits exist.
        common = min(common, len(effective) - 1)
        if common == 0:
            self.logits, self.cache = self.model.prefill(effective)
            self._stats["full_recomputations"] += 1
            return
        if common != self.cache.length:
            self.cache = self.cache.truncate(common)
        self._stats["reused_tokens"] += common
        for token in effective[common:]:
            _check_cancel(cancel)
            self.logits, self.cache = self.model.step(token, self.cache)

    def _sink_step(self, token: int) -> None:
        if self.cache.length == self.budget:
            self.cache = self.cache.evict(self.sink_tokens, self.sink_tokens + 1)
            self._has_evicted = True
            self._stats["evictions"] += 1
        self.logits, self.cache = self.model.step(token, self.cache)

    def _set_sink(self, tokens: list[int], cancel=None) -> None:
        append_only = bool(self._tokens) and tokens[: len(self._tokens)] == self._tokens
        if self.cache is not None and append_only:
            self._stats["reused_tokens"] += self.cache.length
            for token in tokens[len(self._tokens) :]:
                _check_cancel(cancel)
                self._sink_step(token)
            return
        # Before the first eviction, an unchanged prefix has exactly the same context
        # and can be reused. Once entries have been discarded, edits need full replay.
        if self.cache is not None and not self._has_evicted:
            common = 0
            for old, new in zip(self._tokens, tokens):
                if old != new:
                    break
                common += 1
            common = min(common, len(tokens) - 1)
            if common:
                self.cache = self.cache.truncate(common)
                self._stats["reused_tokens"] += common
                for token in tokens[common:]:
                    _check_cancel(cancel)
                    self._sink_step(token)
                return
        # Edits after eviction rebuild canonical state. No cache rollback journal.
        initial = min(self.budget, len(tokens))
        self._has_evicted = False
        self.logits, self.cache = self.model.prefill(tokens[:initial])
        self._stats["canonical_replays"] += 1
        for token in tokens[initial:]:
            _check_cancel(cancel)
            self._sink_step(token)

    def append(self, token: int, cancel=None) -> Tensor:
        return self.set_tokens([*self._tokens, token], cancel=cancel)


@torch.inference_mode()
def sink_reference(
    model: CodeTransformer, tokens: list[int], budget: int, sink_tokens: int = 4
) -> Tensor:
    """Slow independent one-token sink oracle using explicit attention, no cache helpers.

    Intended for correctness checks only. It intentionally does not call the model's
    forward, RoPE, attention, cache eviction, or CachePolicy implementation.
    """
    if model.training or not tokens or not 0 <= sink_tokens < budget <= model.config.context_length:
        raise ValueError("Reference requires an eval model, tokens and valid cache budget")
    device = model.embedding.weight.device
    memories: list[tuple[Tensor, Tensor] | None] = [None] * len(model.blocks)
    cfg = model.config
    head_width = cfg.width // cfg.heads

    def norm(x, layer):
        values = x.float()
        return (values / (values.square().mean(-1, keepdim=True) + layer.eps).sqrt()).to(
            x.dtype
        ) * layer.weight

    def rotate(x, offset=0):
        length = x.shape[-2]
        frequency = torch.tensor(
            [cfg.rope_base ** (-2 * i / head_width) for i in range(head_width // 2)], device=device
        )
        angle = torch.arange(offset, offset + length, device=device)[:, None] * frequency
        result = torch.empty_like(x)
        result[..., 0::2] = x[..., 0::2] * angle.cos() - x[..., 1::2] * angle.sin()
        result[..., 1::2] = x[..., 1::2] * angle.cos() + x[..., 0::2] * angle.sin()
        return result

    for token in tokens:
        x = model.embedding.weight[token].view(1, 1, cfg.width)
        for index, block in enumerate(model.blocks):
            normalized = norm(x, block.attention_norm)
            attention = block.attention
            q = (
                F.linear(normalized, attention.q_proj.weight)
                .view(1, 1, cfg.heads, head_width)
                .transpose(1, 2)
            )
            key = (
                F.linear(normalized, attention.k_proj.weight)
                .view(1, 1, cfg.heads, head_width)
                .transpose(1, 2)
            )
            value = (
                F.linear(normalized, attention.v_proj.weight)
                .view(1, 1, cfg.heads, head_width)
                .transpose(1, 2)
            )
            previous = memories[index]
            if previous is not None:
                old_key, old_value = previous
                if old_key.shape[-2] == budget:
                    old_key = torch.cat(
                        (old_key[..., :sink_tokens, :], old_key[..., sink_tokens + 1 :, :]), -2
                    )
                    old_value = torch.cat(
                        (old_value[..., :sink_tokens, :], old_value[..., sink_tokens + 1 :, :]), -2
                    )
                key, value = torch.cat((old_key, key), -2), torch.cat((old_value, value), -2)
            memories[index] = key, value
            probabilities = (
                (rotate(q, key.shape[-2] - 1) @ rotate(key).transpose(-2, -1))
                / math.sqrt(head_width)
            ).softmax(-1)
            attended = (probabilities @ value).transpose(1, 2).reshape(1, 1, cfg.width)
            x = x + F.linear(attended, attention.out_proj.weight)
            normalized = norm(x, block.ff_norm)
            feedforward = F.silu(F.linear(normalized, block.gate_proj.weight)) * F.linear(
                normalized, block.up_proj.weight
            )
            x = x + F.linear(feedforward, block.down_proj.weight)
    return F.linear(norm(x, model.final_norm), model.embedding.weight)[0, -1]
