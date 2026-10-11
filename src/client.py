"""Notebook client; each method exposes one session operation."""

from __future__ import annotations

import json

import httpx


class CompletionClient:
    def __init__(self, url: str, timeout: float = 130.0):
        self.http = httpx.Client(base_url=url, timeout=timeout)

    def _request(self, method: str, path: str, **kwargs):
        response = self.http.request(method, path, **kwargs)
        response.raise_for_status()
        return response.json()

    def health(self):
        return self._request("GET", "/health")

    def create(self, policy="exact", budget=1024):
        return self._request("POST", "/sessions", json={"policy": policy, "budget": budget})

    def inspect(self, session_id):
        return self._request("GET", f"/sessions/{session_id}")

    def update_source(self, session_id, text, expected_revision):
        return self._request(
            "PUT",
            f"/sessions/{session_id}/source",
            json={"text": text, "expected_revision": expected_revision},
        )

    def stream(self, session_id, expected_revision, max_new_tokens=64):
        with self.http.stream(
            "POST",
            f"/sessions/{session_id}/complete",
            json={"expected_revision": expected_revision, "max_new_tokens": max_new_tokens},
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    event = json.loads(line)
                    if event.get("type") == "error":
                        raise RuntimeError(f"Completion failed: {event}")
                    yield event

    def suggest(self, session_id, expected_revision, max_new_tokens=64):
        events = list(self.stream(session_id, expected_revision, max_new_tokens))
        done = [event for event in events if event.get("type") == "done"]
        if not done:
            raise RuntimeError("Completion ended without a committed proposal receipt")
        return {**done[-1], "events": events}

    def accept(self, session_id, proposal_id, expected_revision, characters=None):
        return self._request(
            "POST",
            f"/sessions/{session_id}/accept",
            json={
                "proposal_id": proposal_id,
                "expected_revision": expected_revision,
                "characters": characters,
            },
        )

    def reject(self, session_id, proposal_id):
        return self._request("DELETE", f"/sessions/{session_id}/proposals/{proposal_id}")

    def reset(self, session_id, expected_revision):
        return self._request(
            "POST", f"/sessions/{session_id}/reset", json={"expected_revision": expected_revision}
        )

    def restore(self, session_id):
        return self._request("POST", f"/sessions/{session_id}/restore")

    def delete(self, session_id):
        return self._request("DELETE", f"/sessions/{session_id}")

    def cancel(self, request_id):
        return self._request("POST", f"/requests/{request_id}/cancel")

    def close(self):
        self.http.close()
