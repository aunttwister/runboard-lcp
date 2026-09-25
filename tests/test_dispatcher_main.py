"""dispatcher.main(): the job state machine, end to end, with every external call faked.

This is the module whose untested branches are untested ways for the box to end up on the
wrong engine, so each path is exercised as a whole tick: queue in, status.json and the
history index out, plus exactly which ``serve.sh`` invocations happened.

Nothing here touches the live box: ``run`` is faked, so ``serve.sh``, ``systemctl`` and
``registry.py`` are recorded rather than executed.
"""
from __future__ import annotations

import json
import pathlib

import pytest

import console_core as C
import dispatcher
from dispatcherkit import FakeTime, fake_popen, make_hf_cli, queue_job, status_doc


@pytest.fixture
def calls(monkeypatch):
    """Every command main() would have run, plus the fakes that let it run."""
    recorded: list[list] = []

    def fake_run(cmd, timeout=60, env=None):
        recorded.append(list(cmd))
        return 0, "serve.sh: engine up\n", ""

    monkeypatch.setattr(dispatcher, "run", fake_run)
    monkeypatch.setattr(dispatcher.subprocess, "Popen",
                        fake_popen(rc=0, lines=("row ok",), polls_before_exit=2))
    monkeypatch.setattr(dispatcher, "time", FakeTime())
    monkeypatch.setattr(dispatcher, "probe_generation", lambda timeout=180: (True, "pong 4"))
    # no other driver owns the box in a test (the pgrep path has its own tests)
    monkeypatch.setattr(dispatcher, "busy_with_something_else", lambda: None)
    monkeypatch.setattr(C, "disk_free_gb", lambda path=None: 5000.0)
    return recorded


def _serving(target: str = "vllm-prod", build: str = "vLLM", model_id: str = "qwen38"):
    return {"target": target, "engine_build": build, "model_id": model_id,
            "healthy": True, "pid": 42, "unit": None, "container": None}


def _serve_calls(calls, engine):
    return [c for c in calls if c and c[0] == dispatcher.SERVE and c[-1] == engine]


def _run(override=None, default=(0, "serve.sh: engine up\n", "")):
    """A fake ``dispatcher.run``: pgrep finds nothing, other commands return a canned rc."""
    override = override or {}

    def fake_run(cmd, timeout=60, env=None):
        if cmd[0] == "pgrep":
            return (1, "", "")
        return override.get(cmd[-1], default)

    return fake_run


# ---------------------------------------------------------------- idle tick

def test_main_stands_down_when_another_dispatcher_holds_the_lock(load_tree, monkeypatch):
    monkeypatch.setattr(dispatcher.fcntl, "flock",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("held")))
    assert dispatcher.main() == 0
    assert not C.STATUS.exists()          # a losing dispatcher must not publish


def test_main_publishes_an_idle_receipt_when_the_queue_is_empty(load_tree):
    assert dispatcher.main() == 0
    doc = status_doc()
    assert doc["state"] == "idle" and doc["job"] is None and doc["queue_depth"] == 0
    assert not C.HISTORY.exists()         # an idle tick writes no history


def test_the_idle_tick_keeps_the_receipt_of_the_job_that_just_finished(load_tree):
    """The reason ``Status.resume()`` exists: restored_to/off_baseline are the only record
    that :18300 came back to the baseline, and a blank Status() erased them in 15 s."""
    first = dispatcher.Status()
    first.set(state="done", restored_to={"target": "cruz", "rc": 0, "ok": True},
              off_baseline=False)
    first.log("restored to baseline cruz")

    assert dispatcher.main() == 0
    doc = status_doc()
    assert doc["restored_to"] == {"target": "cruz", "rc": 0, "ok": True}
    assert doc["off_baseline"] is False
    assert any("restored to baseline cruz" in line for line in doc["log_tail"])
    assert doc["state"] == "idle" and doc["job"] is None


# ---------------------------------------------------------------- bad jobs

def test_an_unreadable_job_file_fails_and_is_archived(load_tree):
    C.QUEUE.mkdir(parents=True, exist_ok=True)
    (C.QUEUE / "broken.json").write_text("{this is not json")
    assert dispatcher.main() == 1
    doc = status_doc()
    assert doc["state"] == "failed" and "unreadable job file" in doc["error"]
    assert (C.DONE / "broken.json").exists() and not (C.QUEUE / "broken.json").exists()


def test_an_invalid_job_is_rejected_before_anything_else_runs(load_tree, calls):
    queue_job({"job_id": "bad", "action": "rm -rf /"}, name="bad")
    assert dispatcher.main() == 2
    doc = status_doc()
    assert doc["state"] == "failed" and "validation: action must be" in doc["error"]
    assert calls == []                    # nothing was executed for a rejected job
    assert json.loads(C.HISTORY.read_text())["jobs"][-1]["state"] == "failed"
    assert (C.DONE / "bad.json").exists()
    assert any("validation failed" in line for line in doc["log_tail"])


# ---------------------------------------------------------------- the eval path

def test_an_eval_switches_runs_and_restores_the_baseline(load_tree, calls, monkeypatch):
    monkeypatch.setattr(dispatcher, "live_engine", lambda: _serving("vllm-prod"))
    queue_job({"job_id": "m1", "action": "eval", "preset": "smoke-20", "engine": "exl3"},
              name="m1")

    assert dispatcher.main() == 0
    doc = status_doc()
    assert doc["state"] == "done" and doc["phase"] == "done"
    assert doc["previous_engine"] == {"target": "vllm-prod", "engine_build": "vLLM"}
    assert doc["restored_to"] == {"target": "cruz", "rc": 0, "ok": True,
                                  "previous": "vllm-prod"}
    assert doc["off_baseline"] is False
    assert doc["result"]["run_id"] == "m1-exl3"
    # switched out to the job's engine, then back to the baseline
    assert _serve_calls(calls, "exl3") and _serve_calls(calls, "cruz")
    assert calls.index(["systemctl", "start", "load-history.service"]) >= 0
    assert not (C.QUEUE / "m1.json").exists() and (C.DONE / "m1.json").exists()
    joined = " ".join(doc["log_tail"])
    assert "picked job m1 (eval)" in joined and "validated: ok" in joined
    assert "before: serving=vllm-prod" in joined
    assert "restoring baseline: cruz" in joined and "restored to baseline cruz" in joined
    assert "job m1 done" in joined


def test_a_run_that_raises_still_restores_the_baseline(load_tree, calls, monkeypatch):
    """Restore is a finally-style guarantee, not a happy path."""
    monkeypatch.setattr(dispatcher, "live_engine", lambda: _serving("vllm-prod"))

    def boom(job, st, run_id):
        raise RuntimeError("runner exploded")

    monkeypatch.setattr(dispatcher, "do_eval", boom)
    queue_job({"job_id": "m2", "action": "eval", "preset": "smoke-20", "engine": "current"},
              name="m2")

    assert dispatcher.main() == 3
    doc = status_doc()
    assert doc["state"] == "failed" and doc["error"] == "RuntimeError: runner exploded"
    assert doc["restored_to"]["target"] == "cruz" and doc["off_baseline"] is False
    assert _serve_calls(calls, "cruz")    # the box is put back even after a crash
    assert any("ERROR: RuntimeError: runner exploded" in line for line in doc["log_tail"])


def test_a_failed_restore_flags_the_box_as_off_baseline(load_tree, calls, monkeypatch):
    """A restore that returns non-zero must not be reported as if the box came back."""
    monkeypatch.setattr(dispatcher, "run",
                        _run({"cruz": (5, "", "serve.sh: no such engine")}))
    monkeypatch.setattr(dispatcher, "live_engine", lambda: _serving("vllm-prod"))
    queue_job({"job_id": "m3", "action": "eval", "preset": "smoke-20", "engine": "current"},
              name="m3")

    assert dispatcher.main() == 0        # the job itself succeeded
    doc = status_doc()
    assert doc["off_baseline"] is True
    assert doc["restored_to"]["rc"] == 5 and doc["restored_to"]["ok"] is False
    assert any("restore rc=5" in line for line in doc["log_tail"])


def test_a_restore_that_raises_is_visible_in_the_fields_the_console_reads(load_tree, calls,
                                                                          monkeypatch):
    """A restore that raises must set off_baseline, not just write a log line.

    The failure used to be swallowed into the log tail while ``off_baseline`` stayed null and
    the job finished ``done`` -- so "the box may be off the baseline" was invisible to
    everything except a human reading the log. The field the console reads is the signal.
    """
    seq = [_serving("vllm-prod"), _serving("vllm-prod")]

    def flaky():
        return seq.pop(0) if seq else (_ for _ in ()).throw(RuntimeError("no /proc"))

    monkeypatch.setattr(dispatcher, "live_engine", flaky)
    queue_job({"job_id": "m4", "action": "eval", "preset": "smoke-20", "engine": "current"},
              name="m4")

    assert dispatcher.main() == 0
    doc = status_doc()
    assert any("!! restore raised: RuntimeError: no /proc" in line for line in doc["log_tail"])
    assert doc["off_baseline"] is True
    assert doc["restored_to"] == {"target": "cruz", "rc": None, "ok": False,
                                  "previous": "vllm-prod", "error": "RuntimeError"}
    # the job itself still finished: the eval ran, it is the restore that failed
    assert doc["state"] == "done"


def test_baseline_already_serving_needs_no_restore(load_tree, calls, monkeypatch):
    monkeypatch.setattr(dispatcher, "live_engine", lambda: _serving("cruz", "CRUZ FORK"))
    queue_job({"job_id": "m5", "action": "eval", "preset": "smoke-20", "engine": "current"},
              name="m5")

    assert dispatcher.main() == 0
    assert _serve_calls(calls, "cruz") == []
    doc = status_doc()
    assert doc["off_baseline"] is False
    assert any("baseline already serving (cruz)" in line for line in doc["log_tail"])


def test_a_failing_history_refresh_does_not_fail_a_good_job(load_tree, calls, monkeypatch):
    def fake_run(cmd, timeout=60, env=None):
        if cmd[0] == "systemctl":
            raise RuntimeError("systemd is not here")
        return (1, "", "") if cmd[0] == "pgrep" else (0, "up", "")

    monkeypatch.setattr(dispatcher, "run", fake_run)
    monkeypatch.setattr(dispatcher, "live_engine", lambda: _serving("cruz"))
    queue_job({"job_id": "m6", "action": "eval", "preset": "smoke-20", "engine": "current"},
              name="m6")
    assert dispatcher.main() == 0
    assert status_doc()["state"] == "done"


# ---------------------------------------------------------------- defer / serve / download

def test_a_job_is_deferred_while_another_driver_owns_the_model(load_tree, calls, monkeypatch):
    monkeypatch.setattr(dispatcher, "busy_with_something_else",
                        lambda: "load soak is running (pid 999)")
    queue_job({"job_id": "m7", "action": "eval", "preset": "smoke-20"}, name="m7")

    assert dispatcher.main() == 0
    doc = status_doc()
    assert doc["state"] == "queued" and doc["phase"] == "waiting"
    assert doc["error"] == "deferred — load soak is running (pid 999)"
    assert calls == [] and not (C.DONE / "m7.json").exists()
    requeued = json.loads((C.QUEUE / "m7.json").read_text())
    assert requeued["job_id"] == "m7" and "_path" not in requeued


def test_a_requeue_that_fails_does_not_lose_the_job(load_tree, monkeypatch):
    monkeypatch.setattr(dispatcher, "busy_with_something_else", lambda: "preset run is running")
    monkeypatch.setattr(pathlib.Path, "unlink",
                        lambda self: (_ for _ in ()).throw(PermissionError("no")))
    queue_job({"job_id": "m8", "action": "eval", "preset": "smoke-20"}, name="m8")

    assert dispatcher.main() == 0
    assert (C.QUEUE / "m8.json").exists()      # still queued, not silently dropped
    assert status_doc()["state"] == "queued"


def test_an_explicit_serve_is_left_in_place_and_the_drift_is_recorded(load_tree, calls,
                                                                      monkeypatch):
    monkeypatch.setattr(dispatcher, "live_engine", lambda: _serving("vllm-prod"))
    queue_job({"job_id": "m9", "action": "serve", "engine": "exl3"}, name="m9")

    assert dispatcher.main() == 0
    doc = status_doc()
    assert doc["off_baseline"] is True
    assert doc["result"]["switched_to"] == "exl3" and doc["result"]["target"] == "exl3"
    assert _serve_calls(calls, "exl3") and not _serve_calls(calls, "cruz")
    assert any("!! off-baseline" in line for line in doc["log_tail"])


def test_an_explicit_serve_that_lands_on_the_baseline_is_not_flagged(load_tree, calls,
                                                                    monkeypatch):
    monkeypatch.setattr(dispatcher, "live_engine", lambda: _serving("cruz", "CRUZ FORK"))
    queue_job({"job_id": "m10", "action": "serve", "engine": "cruz"}, name="m10")

    assert dispatcher.main() == 0
    doc = status_doc()
    assert doc["off_baseline"] is False
    assert any("it is the baseline (cruz)" in line for line in doc["log_tail"])


def test_a_failed_explicit_serve_is_a_failed_job(load_tree, monkeypatch):
    monkeypatch.setattr(dispatcher, "live_engine", lambda: _serving("vllm-prod"))
    monkeypatch.setattr(dispatcher, "run", lambda *a, **k: (4, "", "no such engine"))
    queue_job({"job_id": "m11", "action": "serve", "engine": "exl3"}, name="m11")

    assert dispatcher.main() == 3
    doc = status_doc()
    assert doc["state"] == "failed" and "serve.sh exl3 rc=4" in doc["error"]
    assert any("!! off-baseline" in line for line in doc["log_tail"])


def test_an_explicit_serve_that_does_not_answer_is_a_failed_job(load_tree, calls,
                                                               monkeypatch):
    monkeypatch.setattr(dispatcher, "live_engine", lambda: _serving("vllm-prod"))
    monkeypatch.setattr(dispatcher, "run", _run())
    monkeypatch.setattr(dispatcher, "probe_generation", lambda timeout=180: (False, "no pong"))
    queue_job({"job_id": "m12", "action": "serve", "engine": "vllm"}, name="m12")

    assert dispatcher.main() == 3
    assert "does not answer" in status_doc()["error"]


def test_a_runner_that_exits_nonzero_fails_the_job(load_tree, calls, monkeypatch):
    """A non-zero runner rc means there is no usable measurement, so the job is FAILED.

    It used to finish as ``state: done`` with the rc hidden inside ``result`` -- so a preset
    run that failed every row was indistinguishable from a good one on the console, whose
    history table shows state, not rc. The tick still exits 0: the dispatcher did its own
    work (it ran the job and restored the baseline); it is the JOB that failed.
    """
    monkeypatch.setattr(dispatcher.subprocess, "Popen",
                        fake_popen(rc=1, lines=("row 1 FAILED",), polls_before_exit=2))
    monkeypatch.setattr(dispatcher, "live_engine", lambda: _serving("cruz"))
    queue_job({"job_id": "m14", "action": "eval", "preset": "smoke-20", "engine": "current"},
              name="m14")

    assert dispatcher.main() == 0
    doc = status_doc()
    assert doc["state"] == "failed"
    assert doc["error"] == "runner exited rc=1"
    assert doc["result"]["rc"] == 1
    assert any("preset finished rc=1" in line for line in doc["log_tail"])
    assert any("FAILED (runner rc=1)" in line for line in doc["log_tail"])


def test_a_runner_that_exits_zero_is_still_done(load_tree, calls, monkeypatch):
    """The other half of the pair: rc=0 must not be reported as a failure."""
    monkeypatch.setattr(dispatcher.subprocess, "Popen",
                        fake_popen(rc=0, lines=("row 1 ok",), polls_before_exit=2))
    monkeypatch.setattr(dispatcher, "live_engine", lambda: _serving("cruz"))
    queue_job({"job_id": "m14b", "action": "eval", "preset": "smoke-20", "engine": "current"},
              name="m14b")

    assert dispatcher.main() == 0
    doc = status_doc()
    assert doc["state"] == "done" and doc["error"] is None
    assert doc["result"]["rc"] == 0


def test_a_download_job_runs_through_main(load_tree, calls, monkeypatch):
    make_hf_cli(monkeypatch)
    monkeypatch.setattr(dispatcher, "live_engine", lambda: _serving("cruz"))
    queue_job({"job_id": "m13", "action": "download", "repo": "owner/name",
               "expected_gb": 85}, name="m13")

    assert dispatcher.main() == 0
    doc = status_doc()
    assert doc["state"] == "done" and doc["result"]["repo"] == "owner/name"
    assert json.loads(C.HISTORY.read_text())["jobs"][-1]["action"] == "download"
