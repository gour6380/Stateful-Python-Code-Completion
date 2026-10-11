"""Explicit pre-norm causal Transformer; caches contain unrotated keys."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import ModelConfig


@dataclass
class ModelCache:
    """One immutable-by-convention pair of [batch, heads, time, dim] tensors/layer."""

    layers: list[tuple[Tensor, Tensor]]

    @property
    def length(self) -> int:
        return 0 if not self.layers else self.layers[0][0].shape[-2]

    def clone(self) -> ModelCache:
        return ModelCache([(k.clone(), v.clone()) for k, v in self.layers])

    def truncate(self, length: int) -> ModelCache:
        if not 0 <= length <= self.length:
            raise ValueError("Invalid cache truncation")
        return ModelCache(
            [(k[..., :length, :].clone(), v[..., :length, :].clone()) for k, v in self.layers]
        )

    def evict(self, start: int, end: int) -> ModelCache:
        if not 0 <= start <= end <= self.length:
            raise ValueError("Invalid cache eviction range")
        return ModelCache(
            [
                (
                    torch.cat((k[..., :start, :], k[..., end:, :]), dim=-2),
                    torch.cat((v[..., :start, :], v[..., end:, :]), dim=-2),
                )
                for k, v in self.layers
            ]
        )

    @property
    def bytes(self) -> int:
        return sum(t.numel() * t.element_size() for pair in self.layers for t in pair)


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        # Accumulate normalization in fp32 even if a future experiment changes dtype.
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return normalized.to(x.dtype) * self.weight


def apply_rope(
    x: Tensor, positions: Tensor, base: float, tables: tuple[Tensor, Tensor] | None = None
) -> Tensor:
    """Rotate consecutive feature pairs; x is [B,H,T,D], positions is [T]."""
    dim = x.shape[-1]
    if tables is None:
        inverse = base ** (-torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim)
        angles = positions.to(torch.float32)[:, None] * inverse[None, :]
        cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
    else:
        cos, sin = (table.index_select(0, positions).to(x.dtype) for table in tables)
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


class CausalAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.heads = config.heads
        self.head_dim = config.width // config.heads
        self.rope_base = config.rope_base
        self.dropout = config.dropout
        self.q_proj = nn.Linear(config.width, config.width, bias=False)
        self.k_proj = nn.Linear(config.width, config.width, bias=False)
        self.v_proj = nn.Linear(config.width, config.width, bias=False)
        self.out_proj = nn.Linear(config.width, config.width, bias=False)
        inverse = config.rope_base ** (
            -torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim
        )
        angles = (
            torch.arange(config.context_length, dtype=torch.float32)[:, None] * inverse[None, :]
        )
        # Reuse device-resident tables instead of launching trigonometric kernels
        # for every layer and generated token. Rebuilt from architecture on load.
        self.register_buffer("rope_cos", angles.cos(), persistent=False)
        self.register_buffer("rope_sin", angles.sin(), persistent=False)

    def forward(
        self,
        x: Tensor,
        *,
        previous: tuple[Tensor, Tensor] | None = None,
        key_padding: Tensor | None = None,
        reference: bool = False,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        batch, query_length, width = x.shape

        def heads(tensor: Tensor) -> Tensor:
            return tensor.view(batch, query_length, self.heads, self.head_dim).transpose(1, 2)

        q, new_k, new_v = heads(self.q_proj(x)), heads(self.k_proj(x)), heads(self.v_proj(x))
        past_length = 0 if previous is None else previous[0].shape[-2]
        k = new_k if previous is None else torch.cat((previous[0], new_k), dim=-2)
        v = new_v if previous is None else torch.cat((previous[1], new_v), dim=-2)
        key_length = k.shape[-2]
        query_positions = torch.arange(past_length, key_length, device=x.device)
        key_positions = torch.arange(key_length, device=x.device)
        tables = self.rope_cos, self.rope_sin
        q = apply_rope(q, query_positions, self.rope_base, tables)
        rotated_k = apply_rope(k, key_positions, self.rope_base, tables)
        allowed = (key_positions[None, :] <= query_positions[:, None])[None, None, :, :]
        if key_padding is not None:
            allowed = allowed & key_padding[:, None, None, :]
        dropout = self.dropout if self.training else 0.0
        if reference:
            scores = (q @ rotated_k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            weights = scores.masked_fill(~allowed, float("-inf")).softmax(dim=-1)
            weights = torch.nan_to_num(weights, nan=0.0)
            output = F.dropout(weights, p=dropout, training=self.training) @ v
        else:
            output = F.scaled_dot_product_attention(
                q, rotated_k, v, attn_mask=allowed, dropout_p=dropout, is_causal=False
            )
        output = output.transpose(1, 2).contiguous().view(batch, query_length, width)
        return self.out_proj(output), (k, v)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention_norm = RMSNorm(config.width, config.norm_eps)
        self.attention = CausalAttention(config)
        self.ff_norm = RMSNorm(config.width, config.norm_eps)
        self.gate_proj = nn.Linear(config.width, config.ff_width, bias=False)
        self.up_proj = nn.Linear(config.width, config.ff_width, bias=False)
        self.down_proj = nn.Linear(config.ff_width, config.width, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor, **kwargs) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        attended, cache = self.attention(self.attention_norm(x), **kwargs)
        x = x + self.dropout(attended)
        normalized = self.ff_norm(x)
        x = x + self.dropout(
            self.down_proj(F.silu(self.gate_proj(normalized)) * self.up_proj(normalized))
        )
        return x, cache


class CodeTransformer(nn.Module):
    """The default configuration has exactly 31,465,984 trainable parameters."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.width % config.heads or (config.width // config.heads) % 2:
            raise ValueError("Width must divide heads with an even RoPE head dimension")
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.width, padding_idx=0)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.layers)])
        self.final_norm = RMSNorm(config.width, config.norm_eps)
        self.apply(self._initialize)
        with torch.no_grad():
            self.embedding.weight[0].zero_()
            for block in self.blocks:
                block.attention.out_proj.weight.div_(math.sqrt(2 * config.layers))
                block.down_proj.weight.div_(math.sqrt(2 * config.layers))

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _forward(
        self, input_ids: Tensor, *, cache: ModelCache | None = None, reference: bool = False
    ) -> tuple[Tensor, ModelCache]:
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must be a nonempty [batch,time] tensor")
        past_length = 0 if cache is None else cache.length
        if input_ids.shape[1] + past_length > self.config.context_length:
            raise ValueError("Context exceeds the model architecture limit")
        if cache is not None and len(cache.layers) != len(self.blocks):
            raise ValueError("Cache layer count does not match model")
        if cache is None:
            key_padding = input_ids.ne(0)
        else:
            if torch.any(input_ids == 0):
                raise ValueError("Cached inference does not accept PAD tokens")
            key_padding = None
        x = self.embedding(input_ids)
        layers = []
        for index, block in enumerate(self.blocks):
            x, layer = block(
                x,
                previous=None if cache is None else cache.layers[index],
                key_padding=key_padding,
                reference=reference,
            )
            layers.append(layer)
        logits = F.linear(self.final_norm(x), self.embedding.weight)
        return logits, ModelCache(layers)

    def forward(self, input_ids: Tensor, *, reference: bool = False) -> Tensor:
        return self._forward(input_ids, reference=reference)[0]

    @torch.inference_mode()
    def prefill(self, tokens: list[int]) -> tuple[Tensor, ModelCache]:
        if self.training:
            raise RuntimeError("Call model.eval() before inference")
        if not tokens or 0 in tokens:
            raise ValueError("Prefill requires nonempty tokens without PAD")
        ids = torch.tensor([tokens], dtype=torch.long, device=self.embedding.weight.device)
        logits, cache = self._forward(ids)
        return logits[0, -1], cache

    @torch.inference_mode()
    def step(self, token: int, cache: ModelCache) -> tuple[Tensor, ModelCache]:
        if self.training:
            raise RuntimeError("Call model.eval() before inference")
        ids = torch.tensor([[token]], dtype=torch.long, device=self.embedding.weight.device)
        logits, updated = self._forward(ids, cache=cache)
        return logits[0, -1], updated


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def inspect_model(model: CodeTransformer) -> dict:
    return {
        "architecture": asdict(model.config),
        "parameters": parameter_count(model),
        "tied_embeddings": True,
        "attention": "causal MHA with SDPA/reference",
        "cache_keys": "unrotated",
        "parameter_dtype": str(model.embedding.weight.dtype),
        "device": str(model.embedding.weight.device),
        "model_repr": repr(model),
    }
