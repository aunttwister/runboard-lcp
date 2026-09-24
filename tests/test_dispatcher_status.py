"""dispatcher: the clock, the process runner, the status receipt and the guards.

``status.json`` is the only thing the console page reads about a job, and it is also the
only record that :18300 came back to the baseline -- so its write path and its resume
path are pinned here rather than left to an integration run.
"""
from __future__ import annotations

import json
import subprocess
import time

import pytest

import console_core as C
import dispatcher
import registry
from dispatcherkit import FakeTime, queue_job, status_doc


# ---------------------------------------------------------------- basics

def test_now_is_a_utc_stamp_the_page_can_parse():
    stamp = dispatcher.now()
    assert stamp.endswith("Z") and len(stamp) == 20
    time.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ")


def test_run_returns_rc_and_streams(monkeypatch):
    class P:
        returncode = 3
        stdout = "out\n"
        stderr = "err\n"

    monkeypatch.setattr(dispatcher.subprocess, "run", lambda *a, **k: P())
    assert dispatcher.run(["echo", "hi"]) == (3, "out\n", "err\n")


def test_run_reports_a_timeout_as_rc_124(monkeypatch):
    def boom(*a, **k):
        raise subprocess.TimeoutExpired("cmd", 60)

    monkeypatch.setattr(dispatcher.subprocess, "run", boom)
    assert dispatcher.run(["sleep"]) == (124, "", "timeout")


def test_run_reports_an_unlaunchable_command_instead_of_raising(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("no such binary")

    monkeypatch.setattr(dispatcher.subprocess, "run", boom)
    rc, out, err = dispatcher.run(["nope"])
    assert rc == 1 and out == "" and "FileNotFoundError" in err


def test_default_doc_is_idle_with_every_field_the_page_reads():
    doc = dispatcher.default_doc()
    assert doc["state"] == "idle" and doc["job"] is None
    for key in ("restored_to", "off_baseline", "previous_engine", "log_tail",
                "queue_depth", "schema"):
        assert key in doc


# ---------------------------------------------------------------- Status

def test_status_writes_a_receipt_on_construction(load_tree):
    queue_job({"action": "eval", "preset": "smoke-20"}, name="a")
    queue_job({"action": "eval", "preset": "smoke-20"}, name="b")
    st = dispatcher.Status()
    assert st.logs == []
    doc = status_doc()
    assert doc["queue_depth"] == 2 and doc["state"] == "idle"
    assert not C.STATUS.with_name(C.STATUS.name + ".tmp").exists()


def test_status_set_updates_the_stamp_and_trims_the_log(load_tree):
    st = dispatcher.Status()
    for i in range(dispatcher.LOG_KEEP + 5):
        st.log(f"line {i}")
    doc = status_doc()
    assert len(doc["log_tail"]) == dispatcher.LOG_KEEP
    assert doc["log_tail"][-1].endswith("line 64")
    assert doc["updated_utc"] == dispatcher.now()


def test_status_log_ignores_a_blank_line_but_still_flushes(load_tree):
    st = dispatcher.Status()
    st.log("  \n")
    assert st.logs == []
    assert status_doc()["log_tail"] == []


def test_status_resume_keeps_the_previous_receipt(load_tree):
    first = dispatcher.Status()
    first.set(state="done", restored_to={"rc": 0}, off_baseline=False)
    first.log("restored to baseline cruz")

    resumed = dispatcher.Status.resume()
    assert resumed.doc["state"] == "done"
    assert resumed.doc["restored_to"] == {"rc": 0}
    assert resumed.logs[-1].endswith("restored to baseline cruz")
    # resume must not blank the document the page is reading
    assert status_doc()["off_baseline"] is False


def test_status_resume_starts_from_defaults_when_the_file_is_absent(load_tree):
    assert not C.STATUS.exists()
    resumed = dispatcher.Status.resume()
    assert resumed.doc["state"] == "idle" and resumed.logs == []


def test_status_resume_falls_back_when_the_receipt_is_corrupt(load_tree):
    C.STATUS.parent.mkdir(parents=True, exist_ok=True)
    C.STATUS.write_text("{not json")
    assert dispatcher.Status.resume().doc["state"] == "idle"


# ---------------------------------------------------------------- queue guards

def test_queue_depth_counts_json_only(load_tree):
    queue_job({"action": "eval"}, name="one")
    (C.QUEUE / "notes.txt").write_text("ignore me")
    assert dispatcher.queue_depth() == 1


def test_queue_depth_is_zero_when_the_queue_cannot_be_read(monkeypatch):
    monkeypatch.setattr(dispatcher.C, "QUEUE", "/definitely/not/a/path/object")
    assert dispatcher.queue_depth() == 0


def test_oldest_job_is_the_first_by_name(load_tree):
    queue_job({"action": "eval"}, name="b")
    queue_job({"action": "eval"}, name="a")
    assert dispatcher.oldest_job().name == "a.json"


def test_oldest_job_skips_hidden_files(load_tree):
    (C.QUEUE / ".hidden.json").write_text("{}")
    assert dispatcher.oldest_job() is None


def test_oldest_job_returns_none_when_the_queue_is_unreadable(monkeypatch):
    monkeypatch.setattr(dispatcher.C, "QUEUE", "/definitely/not/a/path/object")
    assert dispatcher.oldest_job() is None


def test_live_engine_delegates_to_registry(monkeypatch, load_tree):
    monkeypatch.setattr(registry, "live_engine", lambda: {"target": "cruz"})
    assert dispatcher.live_engine() == {"target": "cruz"}


def test_live_engine_reports_unknown_instead_of_raising(monkeypatch):
    def boom():
        raise RuntimeError("no registry")

    monkeypatch.setattr(registry, "live_engine", boom)
    got = dispatcher.live_engine()
    assert got["target"] == "unknown" and "RuntimeError" in got["error"]


@pytest.mark.parametrize("target,expected", [
    ("cruz", "cruz"), ("exl3-2.5bpw", "exl3"), ("vllm-prod", "vllm"),
    ("current", None), ("none", None),
])
def test_switch_value_for(target, expected):
    assert dispatcher.switch_value_for(target) == expected


@pytest.mark.parametrize("pattern,label", [
    ("load_soak.py", "load soak"),
    ("run_quality_set.py", "frozen kit run"),
    ("q200_lite.py", "preset run"),
    ("bench_thr.py", "throughput sweep"),
])
def test_busy_with_something_else_names_the_other_driver(monkeypatch, pattern, label):
    def fake_run(cmd, timeout=60, env=None):
        # the pattern is used as a pgrep regex, e.g. "load_soak[.]py"
        return (0, "4321\n", "") if cmd[-1].replace("[.]", ".") == pattern else (1, "", "")

    monkeypatch.setattr(dispatcher, "run", fake_run)
    busy = dispatcher.busy_with_something_else()
    assert busy == f"{label} is running (pid 4321)"


def test_busy_with_something_else_is_none_when_nothing_matches(monkeypatch):
    monkeypatch.setattr(dispatcher, "run", lambda *a, **k: (1, "", ""))
    assert dispatcher.busy_with_something_else() is None


# ---------------------------------------------------------------- probes

class _Resp:
    status = 200

    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self.payload


def _patch_urlopen(monkeypatch, payload=None, error=None):
    import urllib.request

    def fake(req, timeout=None):
        if error:
            raise error
        return _Resp(json.dumps(payload).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake)


def test_probe_generation_asks_the_model_to_actually_generate(monkeypatch):
    _patch_urlopen(monkeypatch, {"choices": [{"message": {"content": "pong"}}]})
    ok, why = dispatcher.probe_generation()
    assert ok is True and "pong" in why and "generated 4 chars" in why


def test_probe_generation_treats_an_empty_answer_as_an_answer(monkeypatch):
    _patch_urlopen(monkeypatch, {"choices": [{"message": {"content": None}}]})
    ok, why = dispatcher.probe_generation()
    assert ok is True and "generated 0 chars" in why


def test_probe_generation_reports_a_wedged_server(monkeypatch):
    _patch_urlopen(monkeypatch, error=OSError("connection refused"))
    ok, why = dispatcher.probe_generation()
    assert ok is False and why.startswith("OSError")


def test_log_tail_reads_the_last_lines(load_tree, tmp_path):
    p = tmp_path / "run.log"
    p.write_text("\n".join(f"line{i}" for i in range(20)))
    assert dispatcher.log_tail(p, 3) == ["line17", "line18", "line19"]


def test_log_tail_is_empty_for_a_missing_file(tmp_path):
    assert dispatcher.log_tail(tmp_path / "absent.log") == []
