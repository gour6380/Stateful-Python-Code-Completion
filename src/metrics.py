"""Metrics with explicit denominators; no generated-source execution."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from rapidfuzz.distance import Levenshtein


@dataclass
class TokenMetrics:
    loss_sum: float = 0.0
    correct: int = 0
    tokens: int = 0

    def add(self, loss_sum: float, correct: int, tokens: int) -> None:
        if tokens < 0 or not math.isfinite(loss_sum):
            raise ValueError("Nonfinite loss or invalid token count")
        self.loss_sum += float(loss_sum)
        self.correct += int(correct)
        self.tokens += int(tokens)

    def result(self) -> dict[str, Any]:
        if not self.tokens:
            return {"tokens": 0, "cross_entropy": None, "perplexity": None, "accuracy": None}
        loss = self.loss_sum / self.tokens
        return {
            "tokens": self.tokens,
            "cross_entropy": loss,
            "perplexity": math.exp(loss) if loss < 709 else None,
            "perplexity_overflow": loss >= 709,
            "accuracy": self.correct / self.tokens,
        }


def completion_scores(prediction: str, target: str) -> dict[str, float]:
    """Do not strip indentation, whitespace, or normalize source text."""
    return {
        "exact_match": float(prediction == target),
        "edit_similarity": Levenshtein.normalized_similarity(prediction, target),
    }


def failure_labels(prediction: str, target: str, stop_reason: str) -> list[str]:
    """Observable text heuristics, not claims of semantic correctness."""
    import io
    import token
    import tokenize

    labels = []
    if not prediction:
        labels.append("empty_suggestion")
    if stop_reason in {"invalid_utf8", "incomplete_utf8"}:
        labels.append(stop_reason)
    if stop_reason == "eos" and prediction != target:
        labels.append("premature_eos")
    if stop_reason in {"max_tokens", "max_new_tokens", "length"}:
        labels.append("truncation")
    if (
        prediction[: len(prediction) - len(prediction.lstrip(" \t"))]
        != target[: len(target) - len(target.lstrip(" \t"))]
    ):
        labels.append("indentation_mismatch")
    words = prediction.split()
    if len(words) >= 6 and len(set(words[-6:])) <= 2:
        labels.append("repetitive_output")

    def identifiers(text: str) -> set[str]:
        try:
            return {
                item.string
                for item in tokenize.generate_tokens(io.StringIO(text).readline)
                if item.type == token.NAME
            }
        except (tokenize.TokenError, IndentationError):
            return set()

    if identifiers(prediction) - identifiers(target):
        labels.append("identifier_mismatch")
    if prediction != target and not labels:
        labels.append("other_text_mismatch")
    return labels
