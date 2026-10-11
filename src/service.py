"""Local, notebook-managed HTTP service. Importing this module starts nothing."""

from __future__ import annotations

import asyncio
import json
import multiprocessing as mp
import queue
import socket
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .config import ServiceConfig
from .sessions import SessionError, SessionManager


@dataclass
class Job:
    operation: Callable[[threading.Event], Any]
    streamed: bool
    deadline: float
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    cancel: threading.Event = field(default_factory=threading.Event)
    output: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=32))


class SerialWorker:
    """One model worker; bounded input/output queues and cooperative cancellation."""

    def __init__(self, config: ServiceConfig, error_handler=None):
        self.config = config
        self.error_handler = error_handler
        self.pending: queue.Queue = queue.Queue(maxsize=config.queue_size)
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()
        self.closed = threading.Event()
        self.thread = threading.Thread(target=self._work, name="completion-worker", daemon=True)
        self.thread.start()

    def submit(self, operation: Callable, *, streamed: bool = False) -> Job:
        if self.closed.is_set():
            raise HTTPException(503, "Service is stopping")
        job = Job(operation, streamed, time.monotonic() + self.config.deadline_seconds)
        with self.lock:
            self.jobs[job.id] = job
            try:
                self.pending.put_nowait(job)
            except queue.Full as error:
                self.jobs.pop(job.id, None)
                raise HTTPException(429, "Request queue is full") from error
        return job

    def _put(self, job: Job, item: tuple[str, Any]) -> bool:
        while not job.cancel.is_set() and not self.closed.is_set():
            if time.monotonic() >= job.deadline:
                job.cancel.set()
                return False
            try:
                job.output.put(item, timeout=0.05)
                return True
            except queue.Full:
                continue
        return False

    def _work(self) -> None:
        while not self.closed.is_set():
            try:
                job = self.pending.get(timeout=0.1)
            except queue.Empty:
                continue
            iterator = None
            try:
                if job.cancel.is_set() or time.monotonic() >= job.deadline:
                    job.cancel.set()
                    continue
                result = job.operation(job.cancel)
                if job.streamed:
                    iterator = iter(result)
                    for event in iterator:
                        if not self._put(job, ("value", event)):
                            break
                else:
                    self._put(job, ("value", result))
            except Exception as error:
                status = error.status_code if isinstance(error, SessionError) else 500
                if status == 500 and self.error_handler is not None:
                    try:
                        self.error_handler(job, error)
                    except Exception:
                        # Diagnostic storage failure must not kill the sole worker.
                        pass
                self._put(job, ("error", {"status": status, "message": str(error)}))
            finally:
                if iterator is not None and hasattr(iterator, "close"):
                    try:
                        iterator.close()
                    except Exception as error:
                        self._put(job, ("error", {"status": 500, "message": str(error)}))
                self._put(job, ("end", None))
                self.pending.task_done()
                with self.lock:
                    self.jobs.pop(job.id, None)

    def receive(self, job: Job):
        try:
            while True:
                remaining = job.deadline - time.monotonic()
                if remaining <= 0 or job.cancel.is_set():
                    raise HTTPException(408, "Request cancelled or deadline exceeded")
                try:
                    kind, value = job.output.get(timeout=min(0.1, remaining))
                except queue.Empty:
                    continue
                if kind == "end":
                    return
                if kind == "error":
                    raise HTTPException(value["status"], value["message"])
                yield value
        finally:
            job.cancel.set()

    def call(self, operation: Callable):
        job = self.submit(operation)
        values = list(self.receive(job))
        return values[0] if values else None

    def cancel_job(self, job_id: str) -> bool:
        with self.lock:
            job = self.jobs.get(job_id)
            if job is not None:
                job.cancel.set()
            return job is not None

    def close(self) -> None:
        self.closed.set()
        with self.lock:
            for job in self.jobs.values():
                job.cancel.set()
        self.thread.join(timeout=5)


class CreateRequest(BaseModel):
    policy: Literal["stateless", "exact", "sink"] = "exact"
    budget: int = 1024


class SourceRequest(BaseModel):
    text: str
    expected_revision: int = Field(ge=0)


class RevisionRequest(BaseModel):
    expected_revision: int = Field(ge=0)


class SuggestRequest(RevisionRequest):
    max_new_tokens: int = Field(default=64, ge=1, le=64)


class AcceptRequest(RevisionRequest):
    proposal_id: str
    characters: int | None = Field(default=None, ge=0)


def create_app(manager: SessionManager, config: ServiceConfig | None = None) -> FastAPI:
    """Construct an app around an explicitly loaded manager; no model loading here."""
    config = config or ServiceConfig()

    def record_error(job, error):
        from .runs import write_json

        if hasattr(manager, "directory"):
            write_json(
                manager.directory.parent / "logs" / f"service_failure_{job.id}.json",
                {
                    "request_id": job.id,
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "recipe_fallback": False,
                },
            )

    worker = SerialWorker(config, error_handler=record_error)

    @asynccontextmanager
    async def lifespan(app):
        yield
        worker.close()

    app = FastAPI(title="Stateful Python Code Completion", lifespan=lifespan)
    app.state.worker = worker
    app.state.manager = manager

    @app.get("/health")
    def health():
        return {"status": "ready", "identity": manager.identity, "queued": worker.pending.qsize()}

    @app.post("/sessions")
    def create(request: CreateRequest):
        return worker.call(lambda _: manager.create(request.policy, request.budget))

    @app.get("/sessions/{session_id}")
    def inspect(session_id: str):
        return worker.call(lambda _: manager.inspect(session_id))

    @app.put("/sessions/{session_id}/source")
    def source(session_id: str, request: SourceRequest):
        return worker.call(
            lambda cancel: manager.update_source(
                session_id, request.text, request.expected_revision, cancel=cancel
            )
        )

    @app.post("/sessions/{session_id}/complete")
    def complete(session_id: str, request: SuggestRequest):
        job = worker.submit(
            lambda cancel: manager.suggest(
                session_id, request.expected_revision, request.max_new_tokens, cancel=cancel
            ),
            streamed=True,
        )

        async def events():
            receiver = worker.receive(job)
            sentinel = object()
            try:
                yield json.dumps({"type": "request", "request_id": job.id}) + "\n"
                while True:
                    event = await asyncio.to_thread(next, receiver, sentinel)
                    if event is sentinel:
                        break
                    yield json.dumps(event, ensure_ascii=True, allow_nan=False) + "\n"
            except HTTPException as error:
                yield (
                    json.dumps(
                        {"type": "error", "status": error.status_code, "message": error.detail}
                    )
                    + "\n"
                )
            finally:
                job.cancel.set()

        return StreamingResponse(events(), media_type="application/x-ndjson")

    @app.post("/requests/{request_id}/cancel")
    def cancel(request_id: str):
        return {"cancelled": worker.cancel_job(request_id)}

    @app.post("/sessions/{session_id}/accept")
    def accept(session_id: str, request: AcceptRequest):
        return worker.call(
            lambda cancel: manager.accept(
                session_id,
                request.proposal_id,
                request.expected_revision,
                request.characters,
                cancel=cancel,
            )
        )

    @app.delete("/sessions/{session_id}/proposals/{proposal_id}")
    def reject(session_id: str, proposal_id: str):
        return worker.call(lambda _: manager.reject(session_id, proposal_id))

    @app.post("/sessions/{session_id}/reset")
    def reset(session_id: str, request: RevisionRequest):
        return worker.call(
            lambda cancel: manager.reset(session_id, request.expected_revision, cancel=cancel)
        )

    @app.post("/sessions/{session_id}/restore")
    def restore(session_id: str):
        return worker.call(lambda cancel: manager.restore(session_id, cancel=cancel))

    @app.delete("/sessions/{session_id}")
    def delete(session_id: str):
        worker.call(lambda _: manager.delete(session_id))
        return {"deleted": session_id}

    return app


@dataclass
class ServiceHandle:
    process: Any
    stop_event: Any
    url: str
    startup: dict
    run_path: str


def _serve(root: str, name: str, ready, stop_event) -> None:
    """Spawn target, called only by start_service from an executed notebook cell."""
    from pathlib import Path

    import uvicorn

    from .checkpoints import load_inference
    from .runs import open_run, write_json
    from .runtime import accelerator_lease, configure_execution, synchronize

    started = time.perf_counter()
    try:
        run = open_run(Path(root), name)
        with accelerator_lease(run.root):
            # Spawn inherits the parent's environment before importing torch.
            # Validate it here and apply this run's CPU profile before model work.
            execution = configure_execution(run.config.training)
            load_started = time.perf_counter()
            model, tokenizer, identity = load_inference(run)
            synchronize(next(model.parameters()).device)
            loaded = time.perf_counter()
            manager = SessionManager(
                model, tokenizer, run.path / "sessions", run.config.service, identity
            )
            app = create_app(manager, run.config.service)
            server = uvicorn.Server(
                uvicorn.Config(
                    app,
                    host=run.config.service.host,
                    port=run.config.service.port,
                    log_level="warning",
                    access_log=False,
                )
            )
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind((run.config.service.host, run.config.service.port))
                port = sock.getsockname()[1]
                receipt = {
                    "url": f"http://{run.config.service.host}:{port}",
                    "load_seconds": loaded - load_started,
                    "load_boundary": "Verified inference-artifact load through device synchronization",
                    "identity": identity,
                    "execution": execution,
                    "pid": mp.current_process().pid,
                }
                write_json(run.path / "benchmarks" / "service_startup.json", receipt)

                def monitor():
                    while not server.started and not stop_event.wait(0.01):
                        pass
                    if server.started:
                        complete_receipt = {
                            **receipt,
                            "startup_seconds": time.perf_counter() - started,
                        }
                        write_json(
                            run.path / "benchmarks" / "service_startup.json", complete_receipt
                        )
                        ready.put(complete_receipt)
                    stop_event.wait()
                    server.should_exit = True

                threading.Thread(target=monitor, daemon=True).start()
                server.run(sockets=[sock])
    except Exception as error:
        write_json(
            Path(root) / "runs" / name / "logs" / "service_process_failure.json",
            {"error_type": type(error).__name__, "message": str(error), "recipe_fallback": False},
        )
        ready.put({"error": f"{type(error).__name__}: {error}"})


def start_service(run, *, startup_timeout: float = 120.0) -> ServiceHandle:
    """Start a localhost child process after training resources have been released."""
    from .bootstrap import validate_environment
    from .runs import write_json

    if run.config.service.host not in {"127.0.0.1", "localhost"}:
        raise ValueError("This project exposes a localhost service only")
    run.verify()
    # Do not change Metal flags after torch may have cached them. A mismatched
    # notebook must restart and bootstrap the persisted run's profile explicitly.
    validate_environment(run.config.training.prefer_metal)
    started = time.perf_counter()
    context = mp.get_context("spawn")
    ready, stop_event = context.Queue(), context.Event()
    process = context.Process(target=_serve, args=(str(run.root), run.path.name, ready, stop_event))
    process.start()
    try:
        receipt = ready.get(timeout=startup_timeout)
        if "error" in receipt:
            raise RuntimeError(receipt["error"])
    except BaseException as error:
        stop_event.set()
        process.join(timeout=5)
        if process.is_alive():
            process.terminate()
            process.join()
        write_json(
            run.path / "logs" / "service_start_failure.json",
            {"error_type": type(error).__name__, "message": str(error), "recipe_fallback": False},
        )
        raise
    receipt["worker_startup_seconds"] = receipt["startup_seconds"]
    receipt["startup_seconds"] = time.perf_counter() - started
    receipt["startup_boundary"] = "Notebook process spawn through server-ready handshake"
    write_json(run.path / "benchmarks" / "service_startup.json", receipt)
    return ServiceHandle(process, stop_event, receipt["url"], receipt, str(run.path))


def stop_service(handle: ServiceHandle) -> dict:
    handle.stop_event.set()
    handle.process.join(timeout=10)
    forced = handle.process.is_alive()
    if forced:
        handle.process.terminate()
        handle.process.join(timeout=5)
    return {
        "stopped": not handle.process.is_alive(),
        "forced": forced,
        "exitcode": handle.process.exitcode,
    }
