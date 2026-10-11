"""Greedy completion events with byte-safe UTF-8 decoding and cancellation."""

from __future__ import annotations

import codecs
from collections.abc import Callable, Iterator

import torch

from .cache import CachePolicy


def _cancelled(cancel) -> bool:
    if cancel is None:
        return False
    return bool(cancel() if isinstance(cancel, Callable) else cancel.is_set())


@torch.inference_mode()
def greedy_token(logits, tokenizer) -> int:
    """Shared generation/parity rule: suppress PAD/BOS/UNK, retain EOS as a stop."""
    if logits.ndim != 1 or not torch.isfinite(logits).all():
        raise RuntimeError("Inference produced invalid or non-finite completion logits")
    scores = logits.detach().clone()
    for forbidden in (tokenizer.pad_id, tokenizer.bos_id, tokenizer.unk_id):
        scores[forbidden] = float("-inf")
    if not torch.isfinite(scores).any():
        raise RuntimeError("No finite completion logits")
    return int(scores.argmax().item())


def generate_completion(
    policy: CachePolicy,
    tokenizer,
    max_new_tokens: int = 64,
    stop_at_newline: bool = True,
    cancel=None,
) -> Iterator[dict]:
    """Mutates the supplied branch only; the session manager owns branch isolation.

    ``token_ids`` are sampled model IDs, including any final token whose decoded
    text was cut at a newline. Text acceptance must always re-tokenize canonical
    text. Unfinished or invalid UTF-8 never leaks replacement characters to clients.
    """
    if (
        isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or max_new_tokens < 1
    ):
        raise ValueError("max_new_tokens must be a positive integer")
    if policy.logits is None:
        raise ValueError("Initialize the branch with canonical tokens first")
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    generated: list[int] = []
    text = ""
    reason = "length"
    for _ in range(max_new_tokens):
        if _cancelled(cancel):
            reason = "cancelled"
            break
        token = greedy_token(policy.logits, tokenizer)
        if token == tokenizer.eos_id:
            reason = "eos"
            break
        decoder_state = decoder.getstate()
        try:
            delta = decoder.decode(tokenizer.token_bytes(token), final=False)
        except UnicodeDecodeError:
            decoder.setstate(decoder_state)
            reason = "invalid_utf8"
            break
        generated.append(token)
        newline = stop_at_newline and "\n" in delta
        if newline:
            delta = delta[: delta.index("\n") + 1]
        text += delta
        # Emit an event even for a partial UTF-8 byte so token latency is measurable.
        yield {"type": "token", "text": delta, "token_id": token, "index": len(generated) - 1}
        if _cancelled(cancel):
            reason = "cancelled"
            break
        if newline:
            reason = "newline"
            break
        if len(generated) == max_new_tokens:
            break
        try:
            policy.append(token, cancel=cancel)
        except InterruptedError:
            reason = "cancelled"
            break
    pending = len(decoder.getstate()[0])
    if pending and reason in {"length", "eos"}:
        reason = "incomplete_utf8"
    yield {
        "type": "done",
        "text": text,
        "token_ids": generated,
        "stop_reason": reason,
        "incomplete_utf8_bytes": pending,
    }
