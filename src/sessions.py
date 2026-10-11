"""Revisioned editor-prefix sessions with transactional canonical-text snapshots."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .cache import CachePolicy
from .config import ServiceConfig
from .inference import generate_completion

_ANY_REVISION = object()


class SessionError(Exception):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def _digest(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _check_cancel(cancel) -> None:
    if cancel is not None and bool(cancel() if callable(cancel) else cancel.is_set()):
        raise SessionError("Request cancelled before source commit", 408)


@dataclass
class _Session:
    session_id: str
    policy: CachePolicy
    text: str = ""
    revision: int = 0
    busy: bool = False
    deleted: bool = False
    proposals: dict = field(default_factory=dict)
    lock: threading.RLock = field(default_factory=threading.RLock)


class SessionManager:
    def __init__(
        self, model, tokenizer, directory: Path, config: ServiceConfig, model_identity: dict
    ):
        self.model, self.tokenizer = model, tokenizer
        self.directory, self.config = Path(directory), config
        self.identity = {
            "model": model_identity,
            "tokenizer": tokenizer.identity,
            "session_schema": 1,
            "cache_semantics": "canonical-replay-v1",
        }
        # Check serializability now; snapshots must never pickle Python objects.
        _digest(self.identity)
        self._sessions: dict[str, _Session] = {}
        self._registry_lock = threading.RLock()
        self.directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _validate_id(session_id: str) -> None:
        if not isinstance(session_id, str) or re.fullmatch(r"[a-f0-9]{32}", session_id) is None:
            raise SessionError("Invalid session identifier", 400)

    def _path(self, session_id: str) -> Path:
        self._validate_id(session_id)
        return self.directory / f"{session_id}.json"

    def _get(self, session_id: str) -> _Session:
        self._validate_id(session_id)
        with self._registry_lock:
            try:
                return self._sessions[session_id]
            except KeyError as error:
                raise SessionError(
                    "Session is not loaded; restore it explicitly if a snapshot exists", 404
                ) from error

    def _validate_text(self, text: str) -> None:
        if not isinstance(text, str):
            raise SessionError("Source must be a string")
        try:
            size = len(text.encode("utf-8", errors="strict"))
        except UnicodeEncodeError as error:
            raise SessionError("Source must be valid Unicode text") from error
        if size > self.config.max_source_bytes:
            raise SessionError("Source exceeds the configured byte limit", 413)

    def _tokens(self, text: str) -> list[int]:
        return [self.tokenizer.bos_id, *self.tokenizer.encode(text)]

    @staticmethod
    def _guard(session: _Session, expected_revision=_ANY_REVISION) -> None:
        if session.deleted:
            raise SessionError("Session was deleted or replaced by restore", 404)
        if session.busy:
            raise SessionError("Session has an active completion", 409)
        if expected_revision is not _ANY_REVISION:
            if (
                isinstance(expected_revision, bool)
                or not isinstance(expected_revision, int)
                or expected_revision < 0
            ):
                raise SessionError("Expected revision must be a nonnegative integer")
            if expected_revision != session.revision:
                raise SessionError("Source revision is stale", 409)

    def _view(self, session: _Session) -> dict:
        if session.deleted:
            raise SessionError("Session was deleted or replaced by restore", 404)
        return {
            "session_id": session.session_id,
            "revision": session.revision,
            "text": session.text,
            "policy": session.policy.policy,
            "budget": session.policy.budget,
            "busy": session.busy,
            "proposals": list(session.proposals),
            "cache": session.policy.diagnostics(),
            "identity": self.identity,
        }

    def _persist(
        self, session: _Session, text: str, revision: int, policy: CachePolicy, cancel=None
    ) -> None:
        payload = {
            "session_id": session.session_id,
            "revision": revision,
            "text": text,
            "policy": policy.policy,
            "budget": policy.budget,
            "sink_tokens": policy.sink_tokens,
            "identity": self.identity,
            "token_sha256": _digest(policy.tokens),
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        record = {"payload": payload, "sha256": _digest(payload)}
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.directory,
                prefix=".session-",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = handle.name
                json.dump(record, handle, ensure_ascii=True, sort_keys=True, allow_nan=False)
                handle.flush()
                os.fsync(handle.fileno())
            _check_cancel(cancel)
            os.replace(temporary, self._path(session.session_id))
            temporary = None
        finally:
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)

    def _replace(self, session: _Session, text: str, cancel=None) -> dict:
        _check_cancel(cancel)
        self._validate_text(text)
        candidate = session.policy.clone()
        try:
            candidate.set_tokens(self._tokens(text), cancel=cancel)
        except InterruptedError as error:
            raise SessionError("Request cancelled during canonical replay", 408) from error
        revision = session.revision + 1
        # Publish durable state before assigning in-memory accepted state. On failure
        # the old session and all of its proposals remain available.
        _check_cancel(cancel)
        self._persist(session, text, revision, candidate, cancel=cancel)
        session.text, session.revision, session.policy = text, revision, candidate
        session.proposals.clear()
        return self._view(session)

    def create(self, policy: str = "exact", budget: int = 1024) -> dict:
        if budget not in self.config.budgets:
            raise SessionError("Budget is not one of the configured study budgets")
        with self._registry_lock:
            if len(self._sessions) >= self.config.max_sessions:
                raise SessionError("Maximum active session count reached", 429)
            try:
                engine = CachePolicy(self.model, policy, budget, self.config.sink_tokens)
            except ValueError as error:
                raise SessionError(str(error)) from error
            engine.set_tokens(self._tokens(""))
            session = _Session(uuid.uuid4().hex, engine)
            self._persist(session, "", 0, engine)
            self._sessions[session.session_id] = session
            return self._view(session)

    def inspect(self, session_id: str) -> dict:
        session = self._get(session_id)
        with session.lock:
            return self._view(session)

    def update_source(
        self, session_id: str, text: str, expected_revision: int, cancel=None
    ) -> dict:
        session = self._get(session_id)
        with session.lock:
            self._guard(session, expected_revision)
            _check_cancel(cancel)
            self._validate_text(text)
            if text == session.text:
                return self._view(session)
            return self._replace(session, text, cancel=cancel)

    def suggest(
        self, session_id: str, expected_revision: int, max_new_tokens: int = 64, cancel=None
    ):
        session = self._get(session_id)
        with session.lock:
            self._guard(session, expected_revision)
            if (
                isinstance(max_new_tokens, bool)
                or not isinstance(max_new_tokens, int)
                or not 1 <= max_new_tokens <= self.config.max_new_tokens
            ):
                raise SessionError("Completion token count exceeds the configured limit")
            branch = session.policy.clone()
            session.busy = True
            proposal_id = uuid.uuid4().hex
            base_revision = session.revision
        try:
            for event in generate_completion(branch, self.tokenizer, max_new_tokens, cancel=cancel):
                event = {
                    **event,
                    "session_id": session_id,
                    "proposal_id": proposal_id,
                    "revision": base_revision,
                }
                if event["type"] == "done":
                    committable = bool(event["text"]) and event["stop_reason"] != "cancelled"
                    event["committable"] = committable
                    if committable:
                        with session.lock:
                            # At most one pending proposal per session; a new completed
                            # suggestion supersedes its predecessor without changing source.
                            session.proposals.clear()
                            session.proposals[proposal_id] = {
                                "text": event["text"],
                                "revision": base_revision,
                                "token_ids": event["token_ids"],
                                "stop_reason": event["stop_reason"],
                            }
                yield event
        finally:
            with session.lock:
                session.busy = False

    def accept(
        self,
        session_id: str,
        proposal_id: str,
        expected_revision: int,
        characters: int | None = None,
        cancel=None,
    ) -> dict:
        session = self._get(session_id)
        with session.lock:
            self._guard(session, expected_revision)
            _check_cancel(cancel)
            proposal = session.proposals.get(proposal_id)
            if proposal is None:
                raise SessionError("Proposal does not exist or has been invalidated", 404)
            if proposal["revision"] != session.revision:
                raise SessionError("Proposal belongs to a stale source revision", 409)
            text = proposal["text"]
            characters = len(text) if characters is None else characters
            if (
                isinstance(characters, bool)
                or not isinstance(characters, int)
                or not 1 <= characters <= len(text)
            ):
                raise SessionError(
                    "Accepted characters must be a nonempty Unicode-character prefix"
                )
            return self._replace(session, session.text + text[:characters], cancel=cancel)

    def reject(self, session_id: str, proposal_id: str) -> dict:
        session = self._get(session_id)
        with session.lock:
            self._guard(session)
            if proposal_id not in session.proposals:
                raise SessionError("Proposal does not exist", 404)
            # Proposals are ephemeral; rejecting one does not change accepted text/revision.
            session.proposals.pop(proposal_id)
            return self._view(session)

    def reset(self, session_id: str, expected_revision: int, cancel=None) -> dict:
        session = self._get(session_id)
        with session.lock:
            self._guard(session, expected_revision)
            return self._replace(session, "", cancel=cancel)

    def delete(self, session_id: str) -> None:
        # Consistent registry -> session lock ordering with create/restore.
        with self._registry_lock:
            session = self._get(session_id)
            with session.lock:
                self._guard(session)
                self._path(session_id).unlink(missing_ok=True)
                session.deleted = True
                self._sessions.pop(session_id)

    def restore(self, session_id: str, cancel=None) -> dict:
        _check_cancel(cancel)
        path = self._path(session_id)
        with self._registry_lock:
            previous = self._sessions.get(session_id)
            lock = previous.lock if previous is not None else threading.RLock()
            with lock:
                if previous is not None:
                    self._guard(previous)
                elif len(self._sessions) >= self.config.max_sessions:
                    raise SessionError("Maximum active session count reached", 429)
                try:
                    record = json.loads(path.read_text(encoding="utf-8"))
                except FileNotFoundError as error:
                    raise SessionError("Session snapshot does not exist", 404) from error
                except (json.JSONDecodeError, UnicodeDecodeError) as error:
                    raise SessionError("Session snapshot is malformed", 409) from error
                try:
                    payload = record["payload"]
                    if record["sha256"] != _digest(payload) or payload["session_id"] != session_id:
                        raise ValueError("Snapshot integrity mismatch")
                    if _digest(payload["identity"]) != _digest(self.identity):
                        raise ValueError("Model/tokenizer/session semantics identity mismatch")
                    if (
                        payload["budget"] not in self.config.budgets
                        or payload["sink_tokens"] != self.config.sink_tokens
                    ):
                        raise ValueError("Snapshot cache settings are incompatible")
                    self._validate_text(payload["text"])
                    revision = payload["revision"]
                    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
                        raise ValueError("Invalid snapshot revision")
                    tokens = self._tokens(payload["text"])
                    if _digest(tokens) != payload["token_sha256"]:
                        raise ValueError("Canonical token identity mismatch")
                    policy = CachePolicy(
                        self.model, payload["policy"], payload["budget"], payload["sink_tokens"]
                    )
                    policy.set_tokens(tokens, cancel=cancel)
                except InterruptedError as error:
                    raise SessionError(
                        "Request cancelled during session restoration", 408
                    ) from error
                except (KeyError, TypeError, ValueError) as error:
                    raise SessionError(f"Cannot restore session: {error}", 409) from error
                _check_cancel(cancel)
                restored = _Session(session_id, policy, payload["text"], revision)
                if previous is not None:
                    previous.deleted = True
                self._sessions[session_id] = restored
                return self._view(restored)
