"""Editor transactions, cancellation, restart identity, and UTF-8 event contracts."""

import json
import threading

import pytest
import torch

from src.cache import CachePolicy
from src.config import ModelConfig, ServiceConfig
from src.inference import generate_completion
from src.model import CodeTransformer
from src.sessions import SessionError, SessionManager


class ByteTokenizer:
    pad_id, bos_id, eos_id, unk_id = 0, 1, 2, 3
    identity = {"fixture": "byte-with-ab-merge-v1"}

    def encode(self, text):
        raw = text.encode("utf-8")
        result, position = [], 0
        while position < len(raw):
            if raw[position : position + 2] == b"ab":
                result.append(260)
                position += 2
            else:
                result.append(raw[position] + 4)
                position += 1
        return result

    def token_bytes(self, token):
        return b"ab" if token == 260 else (bytes([token - 4]) if token >= 4 else b"")

    def decode(self, tokens):
        return b"".join(self.token_bytes(token) for token in tokens).decode("utf-8")


@pytest.fixture
def manager(tmp_path):
    torch.manual_seed(37)
    model = CodeTransformer(
        ModelConfig(
            vocab_size=261, layers=2, width=16, heads=2, ff_width=24, context_length=16, dropout=0
        )
    ).eval()
    config = ServiceConfig(budgets=(8, 16), sink_tokens=4, max_source_bytes=1024)
    return SessionManager(model, ByteTokenizer(), tmp_path, config, {"weights": "fixture-v1"})


def fake_suggestion(monkeypatch, text="ab🙂\n"):
    def generate(policy, tokenizer, max_new_tokens, cancel=None):
        yield {"type": "token", "text": text, "token_id": 100, "index": 0}
        yield {"type": "done", "text": text, "token_ids": [100], "stop_reason": "newline"}

    monkeypatch.setattr("src.sessions.generate_completion", generate)


@pytest.mark.parametrize("policy", ["exact", "sink"])
def test_append_retokenization_matches_clean_canonical_replay(manager, policy):
    created = manager.create(policy, 8)
    sid = created["session_id"]
    text = "x" * 24 + "a"
    first = manager.update_source(sid, text, 0)
    second = manager.update_source(sid, text + "b", first["revision"])
    tokens = [manager.tokenizer.bos_id, *manager.tokenizer.encode(text + "b")]
    fresh = CachePolicy(manager.model, policy, 8)
    expected = fresh.set_tokens(tokens)
    assert manager._sessions[sid].policy.tokens == tokens
    torch.testing.assert_close(manager._sessions[sid].policy.logits, expected, rtol=1e-4, atol=1e-5)
    assert second["revision"] == 2


def test_suggestion_is_isolated_and_partial_accept_uses_unicode_characters(manager, monkeypatch):
    fake_suggestion(monkeypatch)
    created = manager.create("exact", 8)
    sid = created["session_id"]
    snapshot = manager._path(sid).read_bytes()
    done = list(manager.suggest(sid, 0))[-1]
    assert manager.inspect(sid)["text"] == ""
    assert manager.inspect(sid)["revision"] == 0
    assert manager._path(sid).read_bytes() == snapshot
    accepted = manager.accept(sid, done["proposal_id"], 0, characters=3)
    assert accepted["text"] == "ab🙂"
    assert accepted["revision"] == 1
    assert manager._sessions[sid].policy.tokens == [1, *manager.tokenizer.encode("ab🙂")]
    assert accepted["proposals"] == []


def test_stale_revision_and_invalidated_proposal_rejected(manager, monkeypatch):
    fake_suggestion(monkeypatch)
    sid = manager.create("exact", 8)["session_id"]
    done = list(manager.suggest(sid, 0))[-1]
    manager.update_source(sid, "x", 0)
    with pytest.raises(SessionError) as stale:
        manager.accept(sid, done["proposal_id"], 0)
    assert stale.value.status_code == 409
    with pytest.raises(SessionError) as invalid:
        manager.accept(sid, done["proposal_id"], 1)
    assert invalid.value.status_code == 404


def test_abandoned_generator_releases_busy_without_publishing_proposal(manager, monkeypatch):
    fake_suggestion(monkeypatch)
    sid = manager.create("exact", 8)["session_id"]
    stream = manager.suggest(sid, 0)
    next(stream)
    assert manager.inspect(sid)["busy"]
    with pytest.raises(SessionError) as busy:
        manager.update_source(sid, "x", 0)
    assert busy.value.status_code == 409
    stream.close()
    assert not manager.inspect(sid)["busy"]
    assert manager.inspect(sid)["proposals"] == []
    manager.update_source(sid, "x", 0)


def test_cancelled_suggestion_does_not_change_accepted_state(manager):
    sid = manager.create("exact", 8)["session_id"]
    cancelled = threading.Event()
    cancelled.set()
    events = list(manager.suggest(sid, 0, cancel=cancelled))
    assert events[-1]["stop_reason"] == "cancelled"
    assert not events[-1]["committable"]
    assert manager.inspect(sid)["revision"] == 0
    assert manager.inspect(sid)["proposals"] == []


def test_completion_failure_releases_busy_and_does_not_commit(manager, monkeypatch):
    def fail(*args, **kwargs):
        yield {"type": "token", "text": "x", "token_id": 124, "index": 0}
        raise RuntimeError("simulated device failure")

    monkeypatch.setattr("src.sessions.generate_completion", fail)
    sid = manager.create("exact", 8)["session_id"]
    with pytest.raises(RuntimeError, match="device failure"):
        list(manager.suggest(sid, 0))
    assert manager.inspect(sid)["text"] == ""
    assert manager.inspect(sid)["proposals"] == []
    assert not manager.inspect(sid)["busy"]


def test_snapshot_failure_rolls_back_text_cache_revision_and_proposals(manager, monkeypatch):
    sid = manager.create("sink", 8)["session_id"]
    before = manager.inspect(sid)
    logits = manager._sessions[sid].policy.logits.clone()

    def fail(*args, **kwargs):
        raise OSError("simulated full disk")

    monkeypatch.setattr(manager, "_persist", fail)
    with pytest.raises(OSError):
        manager.update_source(sid, "x" * 24, 0)
    assert manager.inspect(sid) == before
    torch.testing.assert_close(manager._sessions[sid].policy.logits, logits)


def test_cancellation_after_replay_prevents_durable_commit(manager, monkeypatch):
    sid = manager.create("sink", 8)["session_id"]
    snapshot = manager._path(sid).read_bytes()
    cancelled = threading.Event()
    original = CachePolicy.set_tokens

    def cancel_after_replay(self, tokens, cancel=None):
        result = original(self, tokens, cancel=cancel)
        cancelled.set()
        return result

    monkeypatch.setattr(CachePolicy, "set_tokens", cancel_after_replay)
    with pytest.raises(SessionError) as error:
        manager.update_source(sid, "x" * 24, 0, cancel=cancelled)
    assert error.value.status_code == 408
    assert manager.inspect(sid)["revision"] == 0
    assert manager._path(sid).read_bytes() == snapshot


@pytest.mark.parametrize("policy", ["stateless", "exact", "sink"])
def test_restart_replays_canonical_text_and_forgets_proposals(manager, policy, monkeypatch):
    fake_suggestion(monkeypatch)
    sid = manager.create(policy, 8)["session_id"]
    manager.update_source(sid, "value = 123\n" * 4, 0)
    before_logits = manager._sessions[sid].policy.logits.clone()
    list(manager.suggest(sid, 1))
    restarted = SessionManager(
        manager.model,
        manager.tokenizer,
        manager.directory,
        manager.config,
        {"weights": "fixture-v1"},
    )
    restored = restarted.restore(sid)
    assert restored["text"] == manager.inspect(sid)["text"]
    assert restored["revision"] == 1
    assert restored["proposals"] == []
    torch.testing.assert_close(
        restarted._sessions[sid].policy.logits, before_logits, rtol=1e-4, atol=1e-5
    )


def test_restore_checks_identity_and_snapshot_integrity(manager):
    sid = manager.create("exact", 8)["session_id"]
    incompatible = SessionManager(
        manager.model,
        manager.tokenizer,
        manager.directory,
        manager.config,
        {"weights": "different"},
    )
    with pytest.raises(SessionError, match="identity"):
        incompatible.restore(sid)
    path = manager._path(sid)
    record = json.loads(path.read_text())
    record["payload"]["text"] = "tampered"
    path.write_text(json.dumps(record))
    with pytest.raises(SessionError, match="integrity"):
        manager.restore(sid)


def test_cancelled_restore_does_not_replace_existing_session(manager):
    sid = manager.create("sink", 8)["session_id"]
    manager.update_source(sid, "x" * 30, 0)
    previous = manager._sessions[sid]
    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(SessionError) as error:
        manager.restore(sid, cancel=cancelled)
    assert error.value.status_code == 408
    assert manager._sessions[sid] is previous
    assert not previous.deleted


def test_null_revision_cannot_bypass_optimistic_concurrency(manager):
    sid = manager.create("exact", 8)["session_id"]
    with pytest.raises(SessionError):
        manager.update_source(sid, "unversioned", None)
    assert manager.inspect(sid)["revision"] == 0


def test_sessions_have_independent_cache_storage_and_revision(manager):
    left = manager.create("exact", 8)["session_id"]
    right = manager.create("exact", 8)["session_id"]
    right_logits = manager._sessions[right].policy.logits.clone()
    manager.update_source(left, "left", 0)
    assert manager.inspect(right)["revision"] == 0
    assert manager.inspect(right)["text"] == ""
    torch.testing.assert_close(manager._sessions[right].policy.logits, right_logits)


def test_paths_and_source_byte_limits(manager):
    with pytest.raises(SessionError):
        manager.restore("../../escape")
    sid = manager.create("exact", 8)["session_id"]
    with pytest.raises(SessionError) as error:
        manager.update_source(sid, "🙂" * 257, 0)
    assert error.value.status_code == 413


class FixedPolicy:
    def __init__(self, tokens):
        self.sequence = [*tokens, 2]
        self.position = 0
        self.logits = self._logits()

    def _logits(self):
        logits = torch.full((261,), -100.0)
        logits[self.sequence[self.position]] = 100.0
        return logits

    def append(self, token, cancel=None):
        self.position += 1
        self.logits = self._logits()


def test_utf8_bytes_are_buffered_until_a_complete_character():
    tokenizer = ByteTokenizer()
    events = list(
        generate_completion(FixedPolicy([byte + 4 for byte in "€\n".encode()]), tokenizer)
    )
    assert [event["text"] for event in events if event["type"] == "token"] == ["", "", "€", "\n"]
    assert events[-1]["text"] == "€\n"
    assert events[-1]["stop_reason"] == "newline"


def test_incomplete_or_invalid_utf8_is_never_replaced_in_output():
    tokenizer = ByteTokenizer()
    incomplete = list(generate_completion(FixedPolicy([0xE2 + 4]), tokenizer, max_new_tokens=1))[-1]
    assert incomplete["text"] == ""
    assert incomplete["stop_reason"] == "incomplete_utf8"
    invalid = list(generate_completion(FixedPolicy([0xFF + 4]), tokenizer))[-1]
    assert invalid["text"] == ""
    assert invalid["stop_reason"] == "invalid_utf8"
