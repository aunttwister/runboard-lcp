"""dispatcher: do_download and finish -- the disk gate and the job receipt.

do_download is the only path that can add a multi-GB pack to the box, so its execution-time
disk re-check is pinned here; ``finish`` is what the console reads to answer "what did that
job do", and it is also what archives the job file.
"""
from __future__ import annotations

import json

import pytest

import console_core as C
import dispatcher
from dispatcherkit import FakeTime, fake_popen, make_hf_cli, queue_job, status_doc


@pytest.fixture
def st(load_tree):
    return dispatcher.Status()


def _dl_env(monkeypatch, rc=0, lines=("Fetching 12 files: 42%",), polls=2,
            never_exits=False, time_step=0.0):
    monkeypatch.setattr(dispatcher.subprocess, "Popen",
                        fake_popen(rc=rc, lines=lines, polls_before_exit=polls,
                                   never_exits=never_exits))
    monkeypatch.setattr(dispatcher, "time", FakeTime(step=time_step))
    monkeypatch.setattr(dispatcher, "run", lambda *a, **k: (0, "", ""))
    # the sandbox volume is tiny; the disk gate itself is covered in test_console_core
    monkeypatch.setattr(C, "disk_free_gb", lambda path=None: 5000.0)


# ---------------------------------------------------------------- do_download

def test_do_download_refuses_when_no_hf_cli_exists(st, monkeypatch):
    make_hf_cli(monkeypatch, which="none")
    monkeypatch.setattr(dispatcher, "run", lambda *a, **k: (0, "", ""))
    with pytest.raises(RuntimeError, match="no hf CLI found"):
        dispatcher.do_download({"job_id": "d0", "repo": "owner/name"}, st)


def test_do_download_re_checks_the_disk_at_execution_time(monkeypatch, st):
    make_hf_cli(monkeypatch)
    monkeypatch.setattr(C, "download_allowed", lambda gb: (False, "only 3 GB free"))
    with pytest.raises(RuntimeError, match="refused at execution time: only 3 GB free"):
        dispatcher.do_download({"job_id": "d1", "repo": "owner/name", "expected_gb": 900}, st)


def test_do_download_runs_hf_with_a_revision_and_reports_progress(monkeypatch, st):
    make_hf_cli(monkeypatch)
    _dl_env(monkeypatch, lines=("Downloading model-00001 37%", "fetching done"))
    job = {"job_id": "d2", "repo": "owner/name", "revision": "3.05bpw", "expected_gb": 85}
    result = dispatcher.do_download(job, st)

    cmd = dispatcher.subprocess.Popen.instances[0].args
    assert cmd == ["/root/dlvenv/bin/hf", "download", "owner/name", "--revision", "3.05bpw"]
    assert result["rc"] == 0 and result["repo"] == "owner/name"
    assert result["revision"] == "3.05bpw"
    assert result["free_gb_after"] > 0
    assert status_doc()["percent"] == "37"          # last percentage seen in the log
    assert any("(anonymous)" in line for line in st.logs)
    assert any("download rc=0" in line for line in st.logs)
    assert any("fetching done" in line for line in st.logs)


def test_do_download_without_a_revision_and_with_a_token(monkeypatch, st):
    make_hf_cli(monkeypatch)
    (C.LOAD / "hf.token").write_text("hf_secret\n")
    _dl_env(monkeypatch, lines=("no percentage here",))
    dispatcher.do_download({"job_id": "d3", "repo": "owner/name", "expected_gb": 10}, st)
    proc = dispatcher.subprocess.Popen.instances[0]
    assert proc.args == ["/root/dlvenv/bin/hf", "download", "owner/name"]
    assert proc.kwargs["env"]["HF_TOKEN"] == "hf_secret"
    assert proc.kwargs["env"]["HUGGING_FACE_HUB_TOKEN"] == "hf_secret"
    assert status_doc()["percent"] is None          # no percentage found, not "0"
    assert any("(authenticated)" in line for line in st.logs)


def test_do_download_falls_through_to_the_second_hf_candidate(monkeypatch, st):
    make_hf_cli(monkeypatch, which="second")
    _dl_env(monkeypatch)
    dispatcher.do_download({"job_id": "d4", "repo": "owner/name", "expected_gb": 10}, st)
    cmd = dispatcher.subprocess.Popen.instances[0].args
    assert cmd[0] == "/root/exl3-engine/r0b0tlab-exllamav3/.venv/bin/hf"


def test_do_download_raises_on_a_nonzero_hf_exit(monkeypatch, st):
    make_hf_cli(monkeypatch)
    _dl_env(monkeypatch, rc=2, lines=("401 Unauthorized",))
    with pytest.raises(RuntimeError, match="hf download rc=2"):
        dispatcher.do_download({"job_id": "d5", "repo": "owner/name", "expected_gb": 10}, st)
    assert any("401 Unauthorized" in line for line in st.logs)


def test_do_download_kills_a_download_that_runs_over_four_hours(monkeypatch, st):
    make_hf_cli(monkeypatch)
    _dl_env(monkeypatch, never_exits=True, time_step=20_000.0)
    with pytest.raises(RuntimeError, match="exceeded 4 h"):
        dispatcher.do_download({"job_id": "d6", "repo": "owner/name", "expected_gb": 10}, st)
    assert dispatcher.subprocess.Popen.instances[0].killed is True


# ---------------------------------------------------------------- finish

def test_finish_records_the_job_and_archives_the_file(monkeypatch, st):
    path = queue_job({"action": "eval", "preset": "smoke-20", "engine": "current"}, name="f1")
    job = json.loads(path.read_text())
    job["_path"] = str(path)
    st.set(state="running", started_utc="2026-09-25T00:00:00Z",
           previous_engine={"target": "vllm-prod", "engine_build": "vLLM"},
           restored_to={"target": "cruz", "rc": 0})
    dispatcher.finish(job, st, "done", result={"rc": 0})

    assert not path.exists()
    assert (C.DONE / "f1.json").exists()
    jobs = json.loads(C.HISTORY.read_text())["jobs"]
    assert jobs[-1]["job_id"] == "f1" and jobs[-1]["state"] == "done"
    assert jobs[-1]["previous_engine"] == {"target": "vllm-prod", "engine_build": "vLLM"}
    assert jobs[-1]["restored_to"]["target"] == "cruz"
    doc = status_doc()
    assert doc["state"] == "done" and doc["phase"] == "done"
    published = {k: v for k, v in doc["job"].items() if k != "_path"}
    assert published == {"job_id": "f1", "action": "eval", "preset": "smoke-20",
                         "engine": "current"}
    assert doc["finished_utc"]


def test_finish_keeps_a_failed_job_failed_and_hides_the_token(monkeypatch, st):
    path = queue_job({"action": "download", "repo": "owner/name", "token": "sekrit"}, name="f2")
    job = json.loads(path.read_text())
    job["_path"] = str(path)
    st.set(phase="download")
    dispatcher.finish(job, st, "failed", error="RuntimeError: hf download rc=2")

    doc = status_doc()
    assert doc["state"] == "failed" and doc["error"] == "RuntimeError: hf download rc=2"
    assert doc["phase"] == "download"            # a failure is not rewritten as done
    assert "token" not in doc["job"]
    assert json.loads(C.HISTORY.read_text())["jobs"][-1]["error"].startswith("RuntimeError")


def test_finish_logs_instead_of_raising_when_the_job_file_cannot_be_archived(monkeypatch, st):
    path = queue_job({"action": "eval"}, name="f3")
    job = json.loads(path.read_text())
    job["_path"] = str(path)
    (C.DONE / "f3.json").mkdir(parents=True)     # an existing directory of that name
    (C.DONE / "f3.json" / "occupant").write_text("x")
    dispatcher.finish(job, st, "done")
    assert any("could not archive job file" in line for line in st.logs)
    assert status_doc()["state"] == "done"       # the receipt is still published


def test_finish_appends_to_an_existing_history_and_caps_it(load_tree, monkeypatch, st):
    path = queue_job({"action": "eval"}, name="f4")
    job = json.loads(path.read_text())
    job["_path"] = str(path)
    C.HISTORY.write_text(json.dumps({"schema": "zgx.console.dispatch_history.v1",
                                     "jobs": [{"job_id": f"old{i}"} for i in range(200)]}))
    dispatcher.finish(job, st, "done")
    jobs = json.loads(C.HISTORY.read_text())["jobs"]
    assert len(jobs) == 200
    assert jobs[-1]["job_id"] == "f4" and jobs[0]["job_id"] == "old1"
