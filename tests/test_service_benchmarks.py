"""Authored transport/queue checks. Handoff does not execute this file."""

import threading

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from src.benchmark import _summary
from src.config import ServiceConfig
from src.service import SerialWorker, create_app
from src.sessions import SessionError


def test_bounded_queue_and_serial_execution():
    worker = SerialWorker(ServiceConfig(queue_size=1, deadline_seconds=5))
    entered, release = threading.Event(), threading.Event()
    order = []

    def blocking(cancel):
        entered.set()
        assert release.wait(3)
        order.append(1)
        return "first"

    try:
        first = worker.submit(blocking)
        assert entered.wait(2)
        second = worker.submit(lambda cancel: order.append(2) or "second")
        with pytest.raises(HTTPException) as caught:
            worker.submit(lambda cancel: "must not run")
        assert caught.value.status_code == 429
        release.set()
        assert list(worker.receive(first)) == ["first"]
        assert list(worker.receive(second)) == ["second"]
        assert order == [1, 2]
    finally:
        release.set()
        worker.close()


def test_cancellation_reaches_active_operation_and_preserves_worker():
    worker = SerialWorker(ServiceConfig(deadline_seconds=5))
    entered, stopped = threading.Event(), threading.Event()

    def operation(cancel):
        entered.set()
        assert cancel.wait(3)
        stopped.set()
        return "discarded"

    try:
        job = worker.submit(operation)
        assert entered.wait(2)
        assert worker.cancel_job(job.id)
        with pytest.raises(HTTPException) as caught:
            list(worker.receive(job))
        assert caught.value.status_code == 408
        assert stopped.wait(2)
        assert worker.call(lambda cancel: "still available") == "still available"
    finally:
        worker.close()


def test_deadline_signals_cooperative_cancel():
    worker = SerialWorker(ServiceConfig(deadline_seconds=0.1))
    observed = threading.Event()

    def operation(cancel):
        if cancel.wait(2):
            observed.set()

    try:
        with pytest.raises(HTTPException) as caught:
            worker.call(operation)
        assert caught.value.status_code == 408
        assert observed.wait(2)
    finally:
        worker.close()


class FakeManager:
    identity = {"model": {"sha256": "fixture"}}

    def create(self, policy, budget):
        return {"session_id": "s", "revision": 0, "text": ""}

    def update_source(self, sid, text, revision, cancel=None):
        if revision != 0:
            raise SessionError("stale revision", 409)
        assert cancel is not None
        return {"session_id": sid, "revision": 1, "text": text}

    def suggest(self, sid, revision, maximum, cancel=None):
        if revision != 1:
            raise SessionError("stale revision", 409)
        yield {"type": "token", "text": "π", "token_id": 42}
        yield {"type": "done", "text": "π", "proposal_id": "p", "token_ids": [42]}


def test_http_validation_stale_revision_and_ndjson_stream():
    with TestClient(create_app(FakeManager())) as client:
        assert client.get("/health").json()["identity"]["model"]["sha256"] == "fixture"
        assert client.post("/sessions", json={"policy": "unknown"}).status_code == 422
        assert (
            client.put("/sessions/s/source", json={"text": "x", "expected_revision": 2}).status_code
            == 409
        )
        assert (
            client.put("/sessions/s/source", json={"text": "x", "expected_revision": 0}).json()[
                "revision"
            ]
            == 1
        )
        response = client.post("/sessions/s/complete", json={"expected_revision": 1})
        import json

        events = [json.loads(line) for line in response.text.splitlines()]
        assert response.headers["content-type"].startswith("application/x-ndjson")
        assert [event["type"] for event in events] == ["request", "token", "done"]
        assert events[-1]["text"] == "π"
        stale = client.post("/sessions/s/complete", json={"expected_revision": 0})
        assert json.loads(stale.text.splitlines()[-1])["status"] == 409


def test_summary_does_not_invent_timings_for_failed_gates():
    rows = [
        {"policy": "exact", "status": "blocked_parity"},
        {
            "policy": "stateless",
            "status": "measured",
            "budget": 256,
            "regime": "after_capacity",
            "workload": "fixed_32_tokens",
            "block": 0,
            "seconds": 2.0,
            "tokens": 32,
        },
        {
            "policy": "stateless",
            "status": "measured",
            "budget": 256,
            "regime": "after_capacity",
            "workload": "fixed_32_tokens",
            "block": 0,
            "seconds": 6.0,
            "tokens": 32,
        },
    ]
    summary = _summary(rows)
    assert len(summary) == 1
    assert summary[0]["p50_seconds"] == 4.0
    assert summary[0]["tokens_per_second"] == 8.0
