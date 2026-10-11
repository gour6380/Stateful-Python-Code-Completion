"""Authored correctness gates; these tests are not executed during delivery."""

import threading

import pytest
import torch

from src.cache import CachePolicy, sink_reference
from src.config import ModelConfig
from src.model import CodeTransformer, parameter_count


@pytest.fixture
def model():
    torch.manual_seed(17)
    return CodeTransformer(
        ModelConfig(
            vocab_size=64, layers=2, width=32, heads=4, ff_width=48, context_length=16, dropout=0
        )
    ).eval()


def assert_logits(left, right):
    torch.testing.assert_close(left, right, rtol=1e-4, atol=1e-5)


def test_default_parameter_count_and_single_tied_parameter():
    model = CodeTransformer(ModelConfig())
    assert parameter_count(model) == 31_465_984
    assert "embedding.weight" in model.state_dict()
    assert not any("lm_head" in name for name in model.state_dict())


def test_explicit_reference_matches_sdpa_with_causal_padding(model):
    ids = torch.tensor([[1, 7, 4, 9, 0, 0], [1, 8, 3, 2, 5, 6]])
    assert_logits(model(ids), model(ids, reference=True))
    assert_logits(model(ids)[0, :4], model(ids[:1, :4])[0])
    assert torch.isfinite(model(torch.zeros((1, 4), dtype=torch.long), reference=True)).all()


def test_future_tokens_cannot_change_prefix_logits(model):
    left = torch.tensor([[1, 9, 5, 8, 10, 11]])
    right = torch.tensor([[1, 9, 5, 8, 12, 13]])
    assert_logits(model(left)[:, :4], model(right)[:, :4])


def test_incremental_cache_matches_full_forward_at_every_position(model):
    tokens = [1, 4, 8, 15, 16, 23, 42]
    actual, cache = model.prefill(tokens[:1])
    assert_logits(actual, model(torch.tensor([tokens[:1]]))[0, -1])
    for position, token in enumerate(tokens[1:], start=2):
        actual, cache = model.step(token, cache)
        assert cache.length == position
        assert_logits(actual, model(torch.tensor([tokens[:position]]))[0, -1])


@pytest.mark.parametrize("policy", ["stateless", "exact"])
def test_rolling_window_matches_full_recompute_through_boundary_and_edits(model, policy):
    engine = CachePolicy(model, policy, 8, sink_tokens=4)
    history = [1] + [4 + i % 40 for i in range(23)]
    for end in range(1, len(history) + 1):
        actual = engine.set_tokens(history[:end])
        expected = model(torch.tensor([history[:end][-8:]]))[0, -1]
        assert_logits(actual, expected)
        assert engine.cache_size <= 8
    for changed in [history[:-1], history[:5] + [52] + history[6:], history[:4], [1, 11]]:
        assert_logits(engine.set_tokens(changed), model(torch.tensor([changed[-8:]]))[0, -1])


def test_exact_reuses_before_capacity_and_recomputes_on_shift(model):
    engine = CachePolicy(model, "exact", 8)
    engine.set_tokens([1, 4, 5])
    engine.append(6)
    assert engine.diagnostics()["reused_tokens"] > 0
    engine.set_tokens([1, 4, 5, 6, 7, 8, 9, 10])
    count = engine.diagnostics()["full_recomputations"]
    engine.append(11)
    assert engine.diagnostics()["full_recomputations"] == count + 1


def test_sink_matches_independent_explicit_reference_after_many_evictions(model):
    tokens = [1] + [4 + (i * 7) % 53 for i in range(34)]
    engine = CachePolicy(model, "sink", 8, sink_tokens=4)
    actual = engine.set_tokens(tokens)
    assert_logits(actual, sink_reference(model, tokens, 8, sink_tokens=4))
    assert engine.diagnostics()["retained_token_indices"] == [0, 1, 2, 3, 31, 32, 33, 34]
    assert engine.cache_size == 8
    # This is an approximation, so no full-context equality assertion follows eviction.


@pytest.mark.parametrize("policy", ["stateless", "exact", "sink"])
def test_request_chunk_boundaries_do_not_change_canonical_outputs(model, policy):
    tokens = [1] + [4 + i % 50 for i in range(37)]
    once = CachePolicy(model, policy, 8)
    expected = once.set_tokens(tokens)
    irregular = CachePolicy(model, policy, 8)
    for end in [1, 3, 8, 9, 17, 29, len(tokens)]:
        actual = irregular.set_tokens(tokens[:end])
    assert_logits(actual, expected)
    tokenwise = CachePolicy(model, policy, 8)
    tokenwise.set_tokens(tokens[:1])
    for token in tokens[1:]:
        tokenwise.append(token)
    assert_logits(tokenwise.logits, expected)


def test_sink_edit_replays_even_when_changed_token_was_evicted(model):
    tokens = [1] + list(range(4, 29))
    engine = CachePolicy(model, "sink", 8)
    engine.set_tokens(tokens)
    changed = [*tokens[:7], 45, *tokens[8:]]
    actual = engine.set_tokens(changed)
    assert_logits(actual, sink_reference(model, changed, 8))
    assert engine.diagnostics()["canonical_replays"] == 2


def test_sink_before_eviction_reuses_prefix_for_edit_and_deletion(model):
    engine = CachePolicy(model, "sink", 8)
    engine.set_tokens([1, 4, 5, 6, 7])
    for changed in ([1, 4, 5, 17, 18, 19], [1, 4, 5]):
        actual = engine.set_tokens(list(changed))
        assert_logits(actual, sink_reference(model, list(changed), 8))
    assert engine.diagnostics()["canonical_replays"] == 1
    assert engine.diagnostics()["reused_tokens"] > 0


def test_cache_clone_does_not_share_mutable_tensors(model):
    engine = CachePolicy(model, "exact", 8)
    engine.set_tokens([1, 4, 5])
    branch = engine.clone()
    assert branch.cache.layers[0][0].data_ptr() != engine.cache.layers[0][0].data_ptr()
    branch.append(6)
    assert engine.tokens == [1, 4, 5]
    assert engine.cache_size == 3


def test_cancelled_replay_restores_previous_engine_state(model, monkeypatch):
    engine = CachePolicy(model, "sink", 8)
    engine.set_tokens([1, 4, 5])
    original_tokens = engine.tokens
    original_logits = engine.logits.clone()
    cancelled = threading.Event()
    original_step = model.step

    def step(token, cache):
        result = original_step(token, cache)
        cancelled.set()
        return result

    monkeypatch.setattr(model, "step", step)
    with pytest.raises(InterruptedError):
        engine.set_tokens([1, 4, 5, 6, 7, 8], cancel=cancelled)
    assert engine.tokens == original_tokens
    assert_logits(engine.logits, original_logits)
    assert engine.cache_size == 3


def test_context_and_training_mode_guards(model):
    with pytest.raises(ValueError):
        model.prefill([1] * 17)
    with pytest.raises(ValueError):
        CachePolicy(model, "sink", 4, sink_tokens=4)
    model.train()
    with pytest.raises(RuntimeError):
        model.prefill([1, 2])


def test_nonfinite_update_rolls_back_existing_cache(model, monkeypatch):
    engine = CachePolicy(model, "exact", 8)
    engine.set_tokens([1, 4, 5])
    before = engine.logits.clone()
    original_step = model.step

    def broken_step(token, cache):
        logits, updated = original_step(token, cache)
        return torch.full_like(logits, float("nan")), updated

    monkeypatch.setattr(model, "step", broken_step)
    with pytest.raises(RuntimeError, match="non-finite"):
        engine.append(6)
    assert engine.tokens == [1, 4, 5]
    assert engine.cache_size == 3
    assert_logits(engine.logits, before)
