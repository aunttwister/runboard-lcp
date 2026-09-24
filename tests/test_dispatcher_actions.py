"""dispatcher: do_switch and do_eval -- the engine-switch and the preset run.

do_switch is the code that can leave :18300 on the wrong engine, and do_eval is the code
that owns the run's log, its timeout and its summary. Both are driven here with a fake
clock and a fake process, so a "timeout" costs no wall-clock time.
"""
from __future__ import annotations

import json

import pytest

import console_core as C
import dispatcher
from dispatcherkit import FakeTime, fake_popen, status_doc


@pytest.fixture
def st(load_tree):
    return dispatcher.Status()


# ---------------------------------------------------------------- do_switch

def test_do_switch_is_a_no_op_for_the_current_engine(monkeypatch, st):
    calls = []
    monkeypatch.setattr(dispatcher, "run", lambda *a, **k: calls.append(a) or (0, "", ""))
    dispatcher.do_switch({"engine": "current"}, st)
    assert calls == []
    assert st.logs[-1].endswith("no switch requested")


def test_do_switch_skips_when_the_engine_is_already_loaded(monkeypatch, st):
    monkeypatch.setattr(dispatcher, "live_engine", lambda: {"target": "cruz"})
    monkeypatch.setattr(dispatcher, "run", lambda *a, **k: (0, "should not run", ""))
    dispatcher.do_switch({"engine": "cruz"}, st)
    assert st.logs[-1].endswith("already on cruz, nothing to switch")


def test_do_switch_runs_serve_sh_and_probes_the_engine(monkeypatch, st):
    seen = []

    def fake_run(cmd, timeout=60, env=None):
        seen.append(cmd)
        return (0, "engine up\nlistening on 18300\n", "")

    monkeypatch.setattr(dispatcher, "live_engine", lambda: {"target": "vllm-prod"})
    monkeypatch.setattr(dispatcher, "run", fake_run)
    monkeypatch.setattr(dispatcher, "probe_generation", lambda timeout=180: (True, "pong 4"))
    dispatcher.do_switch({"engine": "exl3"}, st)
    assert seen == [[dispatcher.SERVE, "exl3"]]
    assert status_doc()["phase"] == "switch"
    assert "switching :18300 to exl3 (serve.sh exl3)" in " ".join(st.logs)
    assert any("listening on 18300" in line for line in st.logs)
    assert any("generation probe: OK" in line for line in st.logs)


def test_do_switch_raises_when_serve_sh_fails(monkeypatch, st):
    monkeypatch.setattr(dispatcher, "live_engine", lambda: {"target": "vllm-prod"})
    monkeypatch.setattr(dispatcher, "run",
                        lambda *a, **k: (2, "partial output", "bad flag"))
    with pytest.raises(RuntimeError, match="switch to exl3 failed rc=2"):
        dispatcher.do_switch({"engine": "exl3"}, st)
    assert any("switch failed rc=2" in line for line in st.logs)


def test_do_switch_raises_when_the_probe_does_not_answer(monkeypatch, st):
    monkeypatch.setattr(dispatcher, "live_engine", lambda: {"target": "vllm-prod"})
    monkeypatch.setattr(dispatcher, "run", lambda *a, **k: (0, "engine up", ""))
    monkeypatch.setattr(dispatcher, "probe_generation",
                        lambda timeout=180: (False, "OSError: refused"))
    with pytest.raises(RuntimeError, match="does not answer"):
        dispatcher.do_switch({"engine": "exl3"}, st)
    assert any("generation probe: FAIL" in line for line in st.logs)


# ---------------------------------------------------------------- do_eval

def _eval_env(monkeypatch, rc=0, lines=("row 1 ok",), polls=2, never_exits=False,
              time_step=0.0):
    monkeypatch.setattr(dispatcher.subprocess, "Popen",
                        fake_popen(rc=rc, lines=lines, polls_before_exit=polls,
                                   never_exits=never_exits))
    monkeypatch.setattr(dispatcher, "time", FakeTime(step=time_step))
    monkeypatch.setattr(dispatcher, "live_engine", lambda: {"model_id": "qwen38-flash-next-exl3"})


def _summary(run_id, **over):
    d = C.RUNS_DIR / run_id
    d.mkdir(parents=True, exist_ok=True)
    doc = {"families": {"gsm8k": {"correct": 6, "graded": 7, "accuracy_pct": 85.7}},
           "rows_attempted": 21, "auto_graded_correct": 18, "auto_graded_total": 20,
           "e2e_tok_s_mean": 41.4, "e2e_tok_s_p50": 40.1, "wall_seconds": 300,
           "completion_tokens_total": 12345, "grader": "q200v2", "model": "qwen38",
           "dataset_sha256": "a" * 64, "kit": None, "coverage_note": "n",
           "skipped_rows": 1}
    doc.update(over)
    (d / "summary.json").write_text(json.dumps(doc))
    return doc


def test_do_eval_runs_the_lite_runner_and_reads_the_summary(monkeypatch, st):
    _eval_env(monkeypatch, lines=("[1] 0.5s 4 tok/s", "row 1 ok"))
    _summary("j1-current")
    job = {"job_id": "j1", "action": "eval", "preset": "smoke-20", "engine": "current"}
    result = dispatcher.do_eval(job, st, "j1-current")

    assert result["rc"] == 0 and result["killed_by_timeout"] is False
    assert result["run_id"] == "j1-current" and result["preset"] == "smoke-20"
    assert result["log"] == str(C.LOGDIR / "j1-current.log")
    assert (C.LOGDIR / "j1-current.log").exists()
    cmd = dispatcher.subprocess.Popen.instances[0].args
    assert "--limit-per-family" in cmd and "7" in cmd
    assert "--max-tokens" in cmd and "8192" in cmd
    assert "--run-id" in cmd and "--out-dir" in cmd and str(C.RUNS_DIR) in cmd
    s = result["summary"]
    assert s["auto_graded"] == "18/20"
    assert s["families"] == {"gsm8k": "6/7 (85.7%)"}
    assert s["dataset_sha256"] == "a" * 16
    assert s["kit"] == "21 rows"          # falls back to rows_attempted
    assert s["e2e_tok_s_mean"] == 41.4
    assert status_doc()["rows_seen"] == 1  # the "[1] ..." line is a row marker
    assert any("summary:" in line for line in st.logs)


def test_do_eval_runs_the_frozen_runner_for_kit_180(monkeypatch, st):
    _eval_env(monkeypatch)
    job = {"job_id": "j2", "action": "eval", "preset": "kit-180", "engine": "exl3"}
    dispatcher.do_eval(job, st, "j2-exl3")
    cmd = dispatcher.subprocess.Popen.instances[0].args
    assert cmd[1] == C.RUNNER_FROZEN
    assert C.SANDBOX_IMAGE in cmd and "--workers" in cmd and "--admission-config" in cmd
    assert "quality-text-180-v2.jsonl" in " ".join(cmd)
    assert any("no summary.json" in line for line in st.logs)


def test_do_eval_without_a_summary_says_the_run_may_have_failed(monkeypatch, st):
    _eval_env(monkeypatch, rc=1)
    result = dispatcher.do_eval({"job_id": "j3", "action": "eval", "preset": "quick-60"},
                                st, "j3-current")
    assert result["rc"] == 1 and "summary" not in result
    assert any("may have failed before writing" in line for line in st.logs)


def test_do_eval_reports_an_unreadable_summary_instead_of_pretending(monkeypatch, st):
    _eval_env(monkeypatch)
    d = C.RUNS_DIR / "j4-current"
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text("{truncated")
    result = dispatcher.do_eval({"job_id": "j4", "action": "eval", "preset": "smoke-20"},
                               st, "j4-current")
    assert "summary_error" in result and "JSONDecodeError" in result["summary_error"]


def test_do_eval_summary_omits_the_grade_pair_when_the_artifact_has_none(monkeypatch, st):
    _eval_env(monkeypatch)
    _summary("j5-current", auto_graded_total=None, rows_attempted=None, kit="kit 180 frozen")
    result = dispatcher.do_eval({"job_id": "j5", "action": "eval", "preset": "smoke-20"},
                               st, "j5-current")
    s = result["summary"]
    assert s["auto_graded"] is None          # never "None/None"
    assert s["rows_attempted"] is None
    assert s["kit"] == "kit 180 frozen"      # the artifact's own label wins


def test_do_eval_kills_a_run_that_exceeds_its_preset_timeout(monkeypatch, st):
    _eval_env(monkeypatch, never_exits=True, time_step=60_000.0)
    result = dispatcher.do_eval({"job_id": "j6", "action": "eval", "preset": "smoke-20"},
                               st, "j6-current")
    proc = dispatcher.subprocess.Popen.instances[0]
    assert proc.terminated is True and proc.killed is True
    assert result["killed_by_timeout"] is True
    assert dispatcher.time.slept == 10.0


def test_do_eval_relogs_an_unchanged_tail_line_on_every_poll(monkeypatch, st):
    """Pins observed behaviour, NOT intended behaviour.

    The guard on dispatcher.py:263 compares the stored line (``"  " + self_last``, as
    ``Status.log`` writes it) against ``self_last`` itself, so the two can never be equal and
    the "already logged this row" suppression never fires: the same tail row is appended every
    10 s poll. Deliberately left as-is -- making it fire would change what the live console
    displays, and this mission's rule is that deployed behaviour stays identical. Reported in
    MISSION_REPORT.md as a latent cosmetic bug with a one-line fix.
    """
    _eval_env(monkeypatch, lines=("same row forever",), polls=4)
    job = {"job_id": "j7", "action": "eval", "preset": "smoke-20"}
    dispatcher.do_eval(job, st, "j7-current")
    assert sum(1 for line in st.logs if line.endswith("same row forever")) == 3


def test_do_eval_uses_the_fallback_model_when_nothing_is_serving(monkeypatch, st):
    _eval_env(monkeypatch)
    monkeypatch.setattr(dispatcher, "live_engine", lambda: None)
    dispatcher.do_eval({"job_id": "j8", "action": "eval", "preset": "quick-60"}, st, "j8-current")
    cmd = dispatcher.subprocess.Popen.instances[0].args
    assert cmd[cmd.index("--model") + 1] == "qwen38-flash-next-exl3"
