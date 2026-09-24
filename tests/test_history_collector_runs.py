"""history_collector: collect_run, the manual-review verdicts and the soak block.

Every field here is read from an artifact on disk. The tests are shaped to catch exactly
the failure the module was written to avoid: a key mismatch that renders as "no data", and
a quarantined run that quietly gets averaged into the others.
"""
from __future__ import annotations

import json

import pytest

import history_collector as HC


def j(payload):
    return json.dumps(payload)


FULL_SUMMARY = {
    "model": "qwen38-flash-next", "rows_attempted": 140, "dataset_sha256": "a" * 64,
    "timestamp": "2026-09-01T00:00:00Z", "wall_seconds": 900, "max_tokens": 16384,
    "temperature": 0, "top_p": 0.95, "auto_graded_correct": 118, "auto_graded_total": 120,
    "families": {"gsm8k": {"correct": 6, "grade_complete": 7, "n": 7, "transported": 7,
                           "ungraded": 0, "accuracy_pct": 85.7}},
    "e2e_tok_s_mean": 40.0, "e2e_tok_s_p50": 39.0, "completion_tokens_total": 1000,
    "coverage_note": "note", "skipped_rows": 2,
}
ROWS = [{"family": "gsm8k", "completion_tokens": 100, "elapsed_s": 10.0},
        {"family": "gsm8k", "completion_tokens": 300, "elapsed_s": 10.0}]
THR = {"decode_tok_s_slope": 51.1, "prefill_overhead_s_est": 0.831}


@pytest.fixture
def artifacts(tmp_path, bench):
    def write(run_id, summary=None, rows=None, thr=None, base=None):
        root = base or bench / "runs" / run_id
        root.mkdir(parents=True, exist_ok=True)
        if summary is not None:
            (root / "summary.json").write_text(j(summary))
        if rows is not None:
            (root / "rows.jsonl").write_text("".join(j(r) + "\n" for r in rows))
        thr_path = None
        if thr is not None:
            thr_path = (base or bench / "runs") / f"thr-{run_id}.json"
            thr_path.write_text(j(thr))
        return {"run_id": run_id, "summary_path": root / "summary.json",
                "rows_path": root / "rows.jsonl", "thr_path": thr_path}

    return write


def test_collect_run_builds_a_complete_entry(artifacts):
    a = artifacts("exl3-2.50bpw-mcs0-mtp", FULL_SUMMARY, ROWS, THR)
    e = HC.collect_run(a["run_id"], a["summary_path"], a["rows_path"], a["thr_path"])
    assert e["run_id"] == "exl3-2.50bpw-mcs0-mtp"
    assert e["model"] == "Qwen3.8-Flash-Next"
    assert e["kit"] == "140 rows"
    assert e["engine"] == "exllamav3 (r0b0tlab gb10)"       # from LABELS
    assert e["quant"] == "EXL3 2.50bpw" and e["note"] == "the pack chosen for deployment"
    assert e["timestamp"] == "2026-09-01T00:00:00Z" and e["wall_seconds"] == 900
    assert e["max_tokens"] == 16384 and e["temperature"] == 0 and e["top_p"] == 0.95
    assert e["quality"]["auto_graded_correct"] == 118
    assert e["quality"]["auto_graded_total"] == 120
    assert e["quality"]["families"]["gsm8k"] == {
        "kind": "math / word problems", "correct": 6, "graded": 7, "n": 7,
        "transported": 7, "ungraded": 0, "accuracy_pct": 85.7}
    assert e["speed"]["decode_tok_s_slope"] == 51.1
    assert e["speed"]["prefill_overhead_s_est"] == 0.831
    assert e["speed"]["e2e_tok_s_mean"] == 40.0
    assert e["coverage"]["decode_slope"] is True and e["coverage"]["reason"] is None
    assert e["coverage"]["per_eval_type_e2e"] is True
    assert e["coverage"]["manual_review"] == "report-only"
    assert e["coverage"]["load_soak"] is False
    assert e["coverage"]["families"] == ["gsm8k"]
    assert e["by_eval_type"]["gsm8k"]["e2e_tok_s_mean"] == 20.0
    assert e["rows_seen"] == 2 and e["quarantined"] is None


def test_collect_run_says_why_decode_only_was_never_measured(artifacts):
    a = artifacts("cruz-fork-305bpw", {"model": "qwen38"}, ROWS, None)
    e = HC.collect_run(a["run_id"], a["summary_path"], a["rows_path"], a["thr_path"])
    assert e["coverage"]["decode_slope"] is False
    assert "no max_tokens sweep" in e["coverage"]["reason"]
    assert e["coverage"]["manual_review"] == "report-only"


def test_collect_run_falls_back_to_the_older_summary_key_names(artifacts):
    summ = {"correct_count": 10, "dataset_count": 12, "started_utc": "2026-09-02T00:00:00Z",
            "families": {"ifeval": {"correct": 3, "graded": 4, "n": 4}}}
    a = artifacts("unlabelled-run", summ, ROWS, None)
    e = HC.collect_run(a["run_id"], a["summary_path"], a["rows_path"], a["thr_path"])
    assert e["quality"]["auto_graded_correct"] == 10
    assert e["quality"]["auto_graded_total"] == 12
    assert e["quality"]["families"]["ifeval"]["graded"] == 4      # the older key name
    assert e["timestamp"] == "2026-09-02T00:00:00Z"               # timestamp falls back
    assert e["engine"] == "unknown" and e["quant"] == a["run_id"]  # no model, no LABELS
    assert e["note"] == ""
    assert e["kit"] == "2 rows"                                   # len(rows) fallback
    assert e["coverage"]["manual_review"] == "not-reviewed"


def test_collect_run_accepts_a_label_written_as_a_list(artifacts, monkeypatch):
    monkeypatch.setitem(HC.LABELS, "listed", ["E", "Q", "N"])
    a = artifacts("listed", {"model": "x"}, ROWS, None)
    e = HC.collect_run(a["run_id"], a["summary_path"], a["rows_path"], a["thr_path"])
    assert (e["engine"], e["quant"], e["note"]) == ("E", "Q", "N")


def test_collect_run_records_why_a_run_is_quarantined(artifacts):
    a = artifacts("INVALID-x", {"model": "x"}, ROWS, None)
    e = HC.collect_run(a["run_id"], a["summary_path"], a["rows_path"], a["thr_path"],
                       quarantine_reason="contended box")
    assert e["quarantined"] == "contended box"


def test_collect_run_uses_an_unknown_engine_when_the_summary_has_no_model(artifacts):
    a = artifacts("mystery", {}, ROWS, None)
    e = HC.collect_run(a["run_id"], a["summary_path"], a["rows_path"], a["thr_path"])
    assert e["engine"] == "unknown" and e["model"] == "mystery"


@pytest.mark.parametrize("run_id,expected", [
    ("exl3-2.5bpw-deployed-180", "artifact"),
    ("cruz-fork-305bpw", "report-only"),
    ("something-else", "not-reviewed"),
])
def test_manual_status_names_where_the_verdicts_live(run_id, expected):
    kind, reason = HC.manual_status(run_id)
    assert kind == expected and reason


def test_closed_summary_families_is_empty_for_a_run_without_a_manual_artifact(bench):
    assert HC.closed_summary_families("exl3-2.5bpw-mcs0-mtp") == {}


def test_closed_summary_families_is_empty_when_the_artifact_is_missing(bench):
    assert HC.closed_summary_families("exl3-2.5bpw-deployed-180") == {}


def test_closed_summary_families_recovers_the_adjudicated_rows(bench):
    (bench / "exl3-2.5bpw-deployed-180.closed-summary.json").write_text(j({
        "families": {"hard_reasoning": {"correct": 20, "n": 20, "grade_complete": None,
                                        "transported": 20, "ungraded": 0,
                                        "accuracy_pct": 100.0},
                     "empty": None}}))
    out = HC.closed_summary_families("exl3-2.5bpw-deployed-180")
    assert set(out) == {"hard_reasoning"}
    assert out["hard_reasoning"]["graded"] == 20        # no grade_complete -> use n
    assert out["hard_reasoning"]["kind"] == "long-form reasoning"
    assert out["hard_reasoning"]["accuracy_pct"] == 100.0


def test_soak_block_is_none_until_a_soak_has_run(run_dir):
    assert HC.soak_block() is None


def test_soak_block_publishes_the_phases_that_have_results(run_dir):
    (run_dir / "state.json").write_text(j({
        "started_utc": "2026-09-03T00:00:00Z", "phase": "DONE",
        "totals": {"requests": 100, "errors": 1, "completion_tokens": 5000},
        "phases": [{"name": "c1", "c": 1, "minutes": 5,
                    "result": {"requests": 20, "errors": 0, "aggregate_tok_s": 55.0,
                               "per_stream_mean": 55.0, "per_stream_p50": 54.0,
                               "scaling_x": 1.0}},
                   {"name": "c2", "result": None}]}))
    (run_dir / "requests.jsonl").write_text(j(
        {"phase": "c1", "ok": True, "completion_tokens": 100, "elapsed": 10.0,
         "concurrency": 1, "ts": 1.0}) + "\n")
    soak = HC.soak_block()
    assert soak["model_id"] == "Qwen3.8-Flash-Next"
    assert soak["unit"] == "exl3-2.5bpw.service" and soak["phase"] == "DONE"
    assert [p["name"] for p in soak["phases"]] == ["c1"]     # the empty phase is dropped
    assert soak["phases"][0]["concurrency"] == 1 and soak["phases"][0]["minutes"] == 5
    assert soak["totals"]["requests"] == 100


def test_soak_block_falls_back_to_the_result_fields_for_concurrency(run_dir):
    (run_dir / "state.json").write_text(j({
        "phases": [{"name": "c4", "result": {"concurrency": 4, "minutes": 20,
                                             "aggregate_tok_s": 60.0}}]}))
    soak = HC.soak_block()
    assert soak["phases"][0]["concurrency"] == 4 and soak["phases"][0]["minutes"] == 20


def test_soak_block_survives_a_broken_correction_step(run_dir, monkeypatch):
    (run_dir / "state.json").write_text(j({"phases": [{"name": "c1", "result": {"n": 1}}]}))
    import corrected_metrics

    def boom(state):
        raise ValueError("bad rows")

    monkeypatch.setattr(corrected_metrics, "overlay", boom)
    assert HC.soak_block()["phases"][0]["name"] == "c1"
