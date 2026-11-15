"""Background jobs with live progress for the slow, model-backed actions.

Creating a concept, writing a lesson or grading answers can take minutes. The
web layer runs those actions in a background thread and the page polls
``/api/jobs/{id}`` to show which step is running, how long it has been
running and what the Hermes agent is doing (model calls, retries, tools),
read from the agent's log.

Steps are driven by the generator calls themselves: ``ReportingGenerator``
wraps the real generator and moves the current job to the step matching the
method being called (``propose_graph`` → "graph", and so on).
"""

from __future__ import annotations

import functools
import os
import re
import threading
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

_local = threading.local()


def current_job() -> "Job | None":
    """Return the job running in this thread, if any."""
    return getattr(_local, "job", None)


class Job:
    """One background action and its progress, safe to read from any thread."""

    def __init__(self, kind: str, concept_id: str | None, steps: list[tuple[str, str]]) -> None:
        self.job_id = uuid4().hex
        self.kind = kind
        self.concept_id = concept_id
        self.steps = [
            {"key": key, "label": label, "status": "pending", "started": None, "ended": None}
            for key, label in steps
        ]
        self.status = "running"
        self.error: str | None = None
        self.result: dict[str, Any] = {}
        self.redirect: str | None = None
        self.activity: list[dict[str, Any]] = []
        self.created = time.time()
        self.finished: float | None = None
        self._lock = threading.Lock()
        self._log = HermesLogFeed()

    # -- progress -------------------------------------------------------
    def step(self, key: str) -> None:
        """Mark ``key`` as the running step; earlier steps become done."""
        now = time.time()
        with self._lock:
            keys = [step["key"] for step in self.steps]
            if key not in keys:
                return
            target = keys.index(key)
            for index, step in enumerate(self.steps):
                if index < target and step["status"] != "done":
                    was_running = step["status"] == "active"
                    step["status"] = "done"
                    step["started"] = step["started"] or now
                    step["ended"] = now
                    if was_running:
                        self.activity.append(
                            {"time": now, "text": f"{step['label']} : terminé en {_duration(now - step['started'])}"}
                        )
                elif index == target and step["status"] == "pending":
                    step["status"] = "active"
                    step["started"] = now
                    self.activity.append({"time": now, "text": f"{step['label']} : démarrage"})

    def note(self, text: str) -> None:
        with self._lock:
            self.activity.append({"time": time.time(), "text": text})
            del self.activity[:-40]

    def finish(self, redirect: str | None, result: dict[str, Any] | None = None) -> None:
        now = time.time()
        with self._lock:
            for step in self.steps:
                if step["status"] == "active":
                    self.activity.append(
                        {"time": now, "text": f"{step['label']} : terminé en {_duration(now - step['started'])}"}
                    )
                if step["status"] != "done":
                    step["status"] = "done"
                    step["started"] = step["started"] or now
                    step["ended"] = now
            self.status = "done"
            self.redirect = redirect
            self.result = result or {}
            self.finished = now

    def fail(self, message: str, redirect: str | None = None) -> None:
        now = time.time()
        with self._lock:
            for step in self.steps:
                if step["status"] == "active":
                    step["status"] = "error"
                    step["ended"] = now
            self.status = "error"
            self.error = message
            self.redirect = redirect
            self.finished = now

    def to_dict(self) -> dict[str, Any]:
        for text in self._log.read_new():
            self.note(text)
        now = time.time()
        with self._lock:
            return {
                "job_id": self.job_id,
                "kind": self.kind,
                "status": self.status,
                "error": self.error,
                "redirect": self.redirect,
                "result": self.result,
                "elapsed": round((self.finished or now) - self.created, 1),
                "steps": [
                    {
                        **step,
                        "elapsed": (
                            round((step["ended"] or now) - step["started"], 1)
                            if step["started"]
                            else None
                        ),
                    }
                    for step in self.steps
                ],
                "activity": [
                    {"seconds": round(item["time"] - self.created), "text": item["text"]}
                    for item in self.activity[-8:]
                ],
            }


def _duration(seconds: float) -> str:
    seconds = int(round(seconds))
    return f"{seconds // 60} min {seconds % 60:02d} s" if seconds >= 60 else f"{seconds} s"


class JobManager:
    """Start jobs in daemon threads; at most one running job per concept."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def running_for(self, concept_id: str | None) -> Job | None:
        if concept_id is None:
            return None
        with self._lock:
            for job in self._jobs.values():
                if job.concept_id == concept_id and job.status == "running":
                    return job
        return None

    def start(
        self,
        kind: str,
        steps: list[tuple[str, str]],
        work: Callable[[Job], None],
        *,
        concept_id: str | None = None,
    ) -> Job:
        """Run ``work(job)`` in a thread. ``work`` must call finish or fail."""
        if self.running_for(concept_id) is not None:
            raise RuntimeError("une action est déjà en cours pour ce concept")
        job = Job(kind, concept_id, steps)
        with self._lock:
            self._jobs[job.job_id] = job
            # forget finished jobs older than an hour
            limit = time.time() - 3600
            for key in [k for k, j in self._jobs.items() if j.finished and j.finished < limit]:
                del self._jobs[key]

        def run() -> None:
            _local.job = job
            try:
                if steps:
                    job.step(steps[0][0])
                work(job)
                if job.status == "running":
                    job.finish(None)
            except Exception as error:  # surfaced to the page, never swallowed
                traceback.print_exc()
                if job.status == "running":
                    job.fail(str(error) or error.__class__.__name__)
            finally:
                _local.job = None

        threading.Thread(target=run, name=f"job-{kind}", daemon=True).start()
        return job


# Generator method → progress step it belongs to.
STEP_OF_METHOD = {
    "propose_graph": "graph",
    "generate_initial_evaluation": "evaluation",
    "generate_course": "lesson",
    "generate_exercise": "exercise",
    "generate_final_exam": "exam",
    "grade_response": "grading",
    "grade_responses": "grading",
}


class ReportingGenerator:
    """Proxy that reports generator calls to the current job.

    Every attribute is delegated to the wrapped generator, so tests and the
    engine see the same object; methods are wrapped with functools.wraps so
    their signature (``feedback`` support) stays visible.
    """

    def __init__(self, generator: Any) -> None:
        object.__setattr__(self, "_generator", generator)

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._generator, name)
        step = STEP_OF_METHOD.get(name)
        if step is None or not callable(value):
            return value

        @functools.wraps(value)
        def reported(*args: Any, **kwargs: Any) -> Any:
            job = current_job()
            if job is not None:
                job.step(step)
                if kwargs.get("feedback"):
                    job.note("Questions invalides écartées : nouvelle demande au tuteur")
            return value(*args, **kwargs)

        return reported

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._generator, name, value)


# ----------------------------------------------------------------------
# Hermes activity feed
# ----------------------------------------------------------------------
_PATTERNS: list[tuple[re.Pattern[str], Callable[[re.Match[str]], str]]] = [
    (
        re.compile(r"API call #(\d+):.*?out=(\d+).*?latency=([\d.]+)s"),
        lambda m: f"Appel au modèle n°{m.group(1)} terminé · {int(m.group(2))} jetons écrits en {float(m.group(3)):.0f} s",
    ),
    (
        re.compile(r"tool (\w+) completed"),
        lambda m: f"Outil utilisé : {m.group(1)}",
    ),
    (
        re.compile(r"Retrying API call in ([\d.]+)s \(attempt (\d+)/(\d+)\)"),
        lambda m: f"Modèle saturé, nouvel essai dans {float(m.group(1)):.0f} s ({m.group(2)}/{m.group(3)})",
    ),
    (
        re.compile(r"inbound message: .*msg='([^']{0,60})"),
        lambda m: "Demande transmise au tuteur",
    ),
    (
        re.compile(r"Turn ended: .*api_calls=(\d+)"),
        lambda m: f"Réponse reçue après {m.group(1)} appel(s) au modèle",
    ),
]


class HermesLogFeed:
    """Read the lines Hermes appended to its log since the job started."""

    def __init__(self) -> None:
        home = os.environ.get("HERMES_HOME")
        self.path = Path(home) / "logs" / "agent.log" if home else None
        try:
            self.offset = self.path.stat().st_size if self.path else 0
        except OSError:
            self.offset = 0
        self._lock = threading.Lock()

    def read_new(self) -> list[str]:
        if self.path is None:
            return []
        with self._lock:
            try:
                size = self.path.stat().st_size
                if size < self.offset:  # log rotated
                    self.offset = 0
                with self.path.open("rb") as handle:
                    handle.seek(self.offset)
                    data = handle.read(min(size - self.offset, 1_000_000))
                    self.offset += len(data)
            except OSError:
                return []
        notes = []
        for line in data.decode("utf-8", "replace").splitlines():
            for pattern, render in _PATTERNS:
                match = pattern.search(line)
                if match:
                    notes.append(render(match))
                    break
        return notes
