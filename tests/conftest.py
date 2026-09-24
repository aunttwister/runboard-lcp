"""Hermetic test sandbox.

Every ``RUNBOARD_*`` root is redirected at a throwaway directory **before** the modules are
imported, so no test can read or write the production tree. An autouse guard then fails any
test that reaches a production path anyway -- the encoding is deliberately belt-and-braces,
because the failure mode this prevents (a unit test switching the engine on a live box) is
expensive and silent.

Tests must not require: the network, a GPU, a listening port, systemd, or the real :18300.
Anything that would touch one of those belongs behind monkeypatch.
"""
from __future__ import annotations

import os
import pathlib
import shutil
import sys
import tempfile

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

SANDBOX = pathlib.Path(tempfile.mkdtemp(prefix="runboard-test-"))

# ---------------------------------------------------------------- process/network guard
#
# The sandbox redirection above stops a test from reaching a production PATH. It cannot stop
# a test from spawning a process (`serve.sh`, `systemctl`, `nvidia-smi`) or opening a socket,
# which is the other half of "hermetic". An audit hook closes that half: the real
# subprocess/os.exec/socket entry points record a violation, and the autouse guard fails the
# test that tripped them. Note that a monkeypatched Popen does NOT fire these events -- so an
# empty list is positive evidence that every process call in this suite was faked.
HOST_ACTIONS: list[str] = []
_WATCHED_EVENTS = ("subprocess.Popen", "os.system", "os.exec", "os.posix_spawn",
                   "os.spawn", "socket.connect")


def _audit_host_actions(event: str, args: tuple) -> None:
    if event in _WATCHED_EVENTS:
        HOST_ACTIONS.append(event)


sys.addaudithook(_audit_host_actions)


def _mk(*parts: str) -> pathlib.Path:
    p = SANDBOX.joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


# Forced, not setdefault: a test run must never inherit production roots from the ambient env.
os.environ["RUNBOARD_LOAD"] = str(_mk("load"))
os.environ["RUNBOARD_STATIC"] = str(_mk("static"))
os.environ["RUNBOARD_RUNS_DIR"] = str(_mk("runs"))
os.environ["RUNBOARD_BENCH"] = str(_mk("bench"))
os.environ["RUNBOARD_HF_CACHE"] = str(_mk("hf"))
os.environ["RUNBOARD_SERVE"] = str(_mk("bin") / "serve.sh")
os.environ["RUNBOARD_RUNNER_LITE"] = str(_mk("bin") / "q200_lite.py")
os.environ["RUNBOARD_RUNNER_FROZEN"] = str(_mk("bin") / "run_quality_set.py")
os.environ["RUNBOARD_DISK_PATH"] = str(SANDBOX)

# Paths that must never appear in a test's arguments, a fixture's return value, or a written
# file. /root/load is checked as a directory prefix only, so /root/loadtest is not caught by
# accident -- but /root/load/anything is.
PROD_PREFIXES = (
    "/root/load/",
    "/root/exl3-bench/",
    "/root/.cache/huggingface/",
)


def touches_prod(value: object) -> str | None:
    """Return the offending string if ``value`` names a production path."""
    text = str(value)
    for prefix in PROD_PREFIXES:
        if prefix in text:
            return text
    if text.rstrip("/") in ("/root/load", "/root/exl3-bench", "/root/.cache/huggingface"):
        return text
    return None


@pytest.fixture(autouse=True)
def _fresh_sandbox():
    """Wipe the throwaway tree before every test.

    One sandbox is shared by the whole session, so a test that leaves a queue entry or a
    status.json behind would change what the next test sees (queue_depth, oldest_job,
    load_metrics...). Resetting keeps every test independent without needing a new temp
    root per test.
    """
    for child in SANDBOX.iterdir():
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink()
    for rel in ("load/dispatch/queue", "load/dispatch/done", "load/run",
                "runs", "bench", "static", "hf", "bin"):
        _mk(*rel.split("/"))
    yield


@pytest.fixture(autouse=True)
def _guard_production_paths():
    """Fail loudly if a test leaves the sandbox -- on disk or on the host."""
    HOST_ACTIONS.clear()
    yield
    import console_core as C

    assert HOST_ACTIONS == [], (
        "a test reached the host machine: " + ", ".join(HOST_ACTIONS)
    )
    for attr in ("LOAD", "RUNS_DIR"):
        value = getattr(C, attr, None)
        assert value is not None, f"console_core.{attr} is unset"
        assert str(value).startswith(str(SANDBOX)), (
            f"console_core.{attr} escaped the sandbox: {value}"
        )


@pytest.fixture
def sandbox() -> pathlib.Path:
    """The throwaway root every RUNBOARD_* variable points at."""
    return SANDBOX


@pytest.fixture
def run_dir(sandbox) -> pathlib.Path:
    """``$RUNBOARD_LOAD/run`` -- where the runner writes state.json and requests.jsonl."""
    d = SANDBOX / "load" / "run"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def bench(sandbox) -> pathlib.Path:
    """``$RUNBOARD_BENCH`` -- where the banked run artifacts live (``runs/`` included)."""
    d = SANDBOX / "bench"
    (d / "runs").mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def load_tree() -> pathlib.Path:
    """A stand-in for /root/load, with the subdirectories the code expects to exist."""
    load = SANDBOX / "load"
    for sub in ("dispatch/queue", "dispatch/done", "run"):
        (load / sub).mkdir(parents=True, exist_ok=True)
    return load
