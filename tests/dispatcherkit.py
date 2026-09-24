"""Helpers for driving the dispatcher without a GPU, a port or a subprocess.

Everything the dispatcher would shell out to (``serve.sh``, ``pgrep``, ``systemctl``,
``hf``, the runner) and everything it would read a clock from is replaced here, so a
dispatcher test is a pure state-machine test: feed it a job file, see what it wrote to
``status.json`` and to the queue.
"""
from __future__ import annotations

import json
import pathlib
import time as _real_time

import console_core as C


class FakeTime:
    """A clock the test can advance by hand. ``step`` seconds pass on every read."""

    def __init__(self, start: float = 1_700_000_000.0, step: float = 0.0):
        self.now = start
        self.step = step
        self.slept = 0.0

    def time(self) -> float:
        current = self.now
        self.now += self.step
        return current

    def sleep(self, seconds: float) -> None:
        self.slept += seconds

    def strftime(self, *a, **kw):
        return _real_time.strftime(*a, **kw)

    def gmtime(self, *a, **kw):
        return _real_time.gmtime(*a, **kw)


def fake_popen(rc: int = 0, lines=(), polls_before_exit: int = 1,
               never_exits: bool = False):
    """A ``subprocess.Popen`` stand-in.

    ``lines`` are written into the log file the caller handed us as ``stdout`` -- that is
    how the dispatcher's progress reader sees a running job. ``polls_before_exit`` controls
    how many times the supervisor loop spins before the process is considered finished;
    ``never_exits`` makes ``poll()`` return None forever, which is what the timeout paths
    need (the loop must be broken by the deadline, not by the process).
    """
    instances = []

    class P:
        def __init__(self, *args, **kwargs):
            self.args = list(args[0]) if args else kwargs.get("args")
            self.kwargs = kwargs
            self._polls = 0
            self.terminated = False
            self.killed = False
            stdout = kwargs.get("stdout")
            if stdout is not None and lines:
                stdout.write("\n".join(lines) + "\n")
                stdout.flush()
            instances.append(self)

        @property
        def returncode(self):
            return rc

        def poll(self):
            self._polls += 1
            if never_exits:
                return None
            return None if self._polls < polls_before_exit else rc

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.killed = True

    P.instances = instances
    return P


def queue_job(job: dict, name: str | None = None) -> pathlib.Path:
    """Drop a job into the queue the way the web process would."""
    C.QUEUE.mkdir(parents=True, exist_ok=True)
    job.setdefault("job_id", name or "job1")
    path = C.QUEUE / f"{job['job_id']}.json"
    path.write_text(json.dumps(job, indent=2))
    return path


def status_doc() -> dict:
    """The dispatcher's published receipt."""
    return json.loads(C.STATUS.read_text())


def history_jobs() -> list[dict]:
    return json.loads(C.HISTORY.read_text())["jobs"]


def make_hf_cli(monkeypatch, which: str = "first"):
    """Pretend an ``hf`` CLI exists at one of the hardcoded production paths.

    The paths are absolute production locations; the test must not create them, so
    ``exists`` is answered for that one string only. ``which="none"`` answers False for
    both, which is how the "no hf CLI on this box" path is reached deterministically.
    """
    import dispatcher

    real_exists = pathlib.Path.exists
    candidates = ["/root/dlvenv/bin/hf",
                  "/root/exl3-engine/r0b0tlab-exllamav3/.venv/bin/hf"]
    wanted = {"first": candidates[0], "second": candidates[1], "none": None}[which]

    def exists(self):
        if str(self) in candidates:
            return str(self) == wanted
        return real_exists(self)

    monkeypatch.setattr(pathlib.Path, "exists", exists)
    return dispatcher
