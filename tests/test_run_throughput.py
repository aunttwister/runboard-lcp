"""run_throughput -- the throughput of the job that is ACTUALLY running.

The defect these pin (2026-09-26): a smoke-20 that was emitting ~42 tok/s showed
"aggregate tok/s 0.0", the SOAK's concurrency and a chart of the five soak phases. The panel
was not broken -- it was answering about a different run. So these tests are deliberately
about WHERE the number comes from and WHEN it is allowed to be shown, not about markup:

  * a job only counts as "in progress" in a state the dispatcher actually uses;
  * a run's rows are read from that job's own directory, matched so one job can never be
    attributed to another whose id merely starts with the same characters;
  * an unrated row (written for adjudication) counts as done but must NEVER be read as
    0 tok/s -- that is the same class of lie as the bug above;
  * with no rows, the figures are null with a reason, never 0.
"""
from __future__ import annotations

import json

import console_core as C
import history_collector
import live_metrics as LM
import run_throughput as RT


def row(ct, el, **kw):
    d = {"completion_tokens": ct, "elapsed_s": el}
    d.update(kw)
    return d


def write_rows(directory, rows):
    directory.mkdir(parents=True, exist_ok=True)
    p = directory / RT.ROWS_NAME
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return p


# --------------------------------------------------------------- shared constant, pinned
def test_the_runs_root_is_the_one_history_collector_uses():
    """Two modules, one directory: pinned so they cannot drift apart silently.

    Kept as separate constants rather than an import because nothing in this repo imports
    another module; the cost of that choice is this test.
    """
    assert RT.RUNS == history_collector.RUNS


# --------------------------------------------------------------------------- find_run_dir
def test_find_run_dir_matches_a_job_that_left_an_engine_suffixed_directory(bench):
    d = bench / "runs" / "20260101T000000Z-eval-smoke-20-current"
    d.mkdir(parents=True)
    assert RT.find_run_dir("20260101T000000Z-eval-smoke-20", bench / "runs") == d


def test_find_run_dir_matches_a_bare_job_id(bench):
    d = bench / "runs" / "J1"
    d.mkdir(parents=True)
    assert RT.find_run_dir("J1", bench / "runs") == d


def test_find_run_dir_never_attributes_a_longer_id_to_a_shorter_one(bench):
    (bench / "runs" / "20260101T000000Z-eval-smoke-20extra").mkdir(parents=True)
    assert RT.find_run_dir("20260101T000000Z-eval-smoke-20", bench / "runs") is None


def test_find_run_dir_prefers_the_directory_the_job_says_it_used(bench):
    for name in ("J1-current", "J1-cruz"):
        (bench / "runs" / name).mkdir(parents=True)
    assert RT.find_run_dir("J1", bench / "runs", engine="cruz").name == "J1-cruz"


def test_find_run_dir_falls_back_to_the_one_on_disk_when_the_named_engine_left_none(bench):
    (bench / "runs" / "J1-current").mkdir(parents=True)
    assert RT.find_run_dir("J1", bench / "runs", engine="exl3").name == "J1-current"


def test_find_run_dir_breaks_a_tie_by_name_without_claiming_to_know_time(bench):
    """`-cruz` sorts before `-current`, so name order is NOT time order here.

    The guarantee is that the choice is deterministic, not that the chosen directory is the
    newer one -- hence the engine preference above and the docstring's wording.
    """
    for name in ("J1-current", "J1-cruz"):
        (bench / "runs" / name).mkdir(parents=True)
    assert RT.find_run_dir("J1", bench / "runs").name == "J1-current"


def test_find_run_dir_ignores_files_that_share_the_prefix(bench):
    (bench / "runs" / "J1-note.txt").write_text("not a run", encoding="utf-8")
    assert RT.find_run_dir("J1", bench / "runs") is None


def test_find_run_dir_without_an_id_or_a_directory(bench):
    assert RT.find_run_dir(None, bench / "runs") is None
    assert RT.find_run_dir("", bench / "runs") is None
    assert RT.find_run_dir("J1", bench / "runs" / "absent") is None


def test_find_run_dir_falls_back_to_the_module_root(bench):
    d = bench / "runs" / "J1"
    d.mkdir(parents=True)
    assert RT.find_run_dir("J1") == d


# ------------------------------------------------------------------------------ read_rows
def test_read_rows_reads_a_partial_file(tmp_path):
    p = write_rows(tmp_path, [row(10, 1.0), row(20, 2.0)])
    assert len(RT.read_rows(p)) == 2


def test_read_rows_skips_a_half_written_trailing_line(tmp_path):
    p = tmp_path / RT.ROWS_NAME
    p.write_text('{"completion_tokens": 10, "elapsed_s": 1.0}\n{"completion_tokens": 3',
                 encoding="utf-8")
    rows = RT.read_rows(p)
    assert len(rows) == 1            # the run is mid-write; a torn line is not an error


def test_read_rows_skips_blank_lines(tmp_path):
    p = tmp_path / RT.ROWS_NAME
    p.write_text('\n{"completion_tokens": 10, "elapsed_s": 1.0}\n\n', encoding="utf-8")
    assert len(RT.read_rows(p)) == 1


def test_read_rows_on_a_missing_or_unopenable_path_is_empty(tmp_path):
    assert RT.read_rows(tmp_path / "absent.jsonl") == []
    d = tmp_path / "a-directory"
    d.mkdir()
    assert RT.read_rows(d) == []


# -------------------------------------------------------------------------------- compute
def test_compute_with_no_rows_reports_null_not_zero():
    out = RT.compute([])
    assert out["aggregate_tok_s"] is None and out["per_stream_tok_s"] is None
    assert out["tokens_total"] == 0 and out["busy_s"] == 0.0
    assert out["series"] == {"ts": [], "agg_tps": [], "stream_tps": []}


def test_compute_is_tokens_over_the_runs_own_request_time():
    out = RT.compute([row(100, 2.0), row(200, 4.0)])
    assert out["tokens_total"] == 300
    assert out["busy_s"] == 6.0
    assert out["aggregate_tok_s"] == 50.0        # 300 tokens / 6 s
    assert out["per_stream_tok_s"] == 50.0       # mean(100/2, 200/4)


def test_compute_separates_the_box_rate_from_the_row_rate():
    """The two series answer different questions and must not be the same number."""
    out = RT.compute([row(100, 2.0), row(100, 8.0)])
    s = out["series"]
    assert s["ts"] == [2.0, 10.0]
    assert s["agg_tps"] == [50.0, 20.0]          # running: 100/2 then 200/10
    assert s["stream_tps"] == [50.0, 12.5]       # this row: 100/2 then 100/8
    assert len(s["ts"]) == len(s["agg_tps"]) == len(s["stream_tps"])


def test_compute_counts_an_unrated_row_without_reading_it_as_zero():
    out = RT.compute([row(100, 2.0), {"id": "hard-1", "verdict": None}])
    assert out["rows_done"] == 2
    assert out["aggregate_tok_s"] == 50.0        # NOT 100/2 + 0
    assert len(out["series"]["ts"]) == 1


def test_compute_counts_rows_that_errored():
    out = RT.compute([row(100, 2.0),
                      {"completion_tokens": None, "elapsed_s": None, "error": "boom"}])
    assert out["errors"] == 1
    assert out["rows_done"] == 2
    assert out["aggregate_tok_s"] == 50.0


def test_compute_ignores_junk_rows_and_impossible_elapsed():
    out = RT.compute(["junk", None, 3, row(0, 0), row(100, 0), row(100, 2.0)])
    assert out["rows_done"] == 6                 # they were written, so they are done
    assert out["aggregate_tok_s"] == 50.0        # but only the usable one is rated


# ---------------------------------------------------------------------------------- block
def test_block_with_no_job_says_so_instead_of_showing_nothing():
    out = RT.block(None)
    assert out["active"] is False
    assert out["reason"] == "no run active"
    assert out["aggregate_tok_s"] is None and out["rows_total"] is None


def test_block_treats_a_non_dict_job_document_as_no_job():
    assert RT.block("nonsense")["active"] is False
    assert RT.block([])["active"] is False


def test_block_refuses_a_job_with_no_id_even_while_running():
    assert RT.block({"state": "running"})["active"] is False


def test_block_names_the_state_of_a_job_that_is_not_running(bench):
    out = RT.block({"job_id": "J1", "state": "done", "preset": "smoke-20"}, presets=C.PRESETS)
    assert out["active"] is False
    assert out["reason"] == "job J1 is done"
    assert out["rows_total"] == 21               # real preset table, not a copy of it


def test_block_names_an_unexpected_state_rather_than_hiding_behind_a_generic_word(bench):
    out = RT.block({"job_id": "J1", "state": "teleporting"})
    assert out["active"] is False
    assert out["reason"] == "job J1 is teleporting"


def test_block_with_a_job_that_carries_no_state_at_all(bench):
    out = RT.block({"job_id": "J1"})
    assert out["active"] is False
    assert out["reason"] == "job J1 is in an unknown state"


def test_block_uses_the_engine_directory_the_job_names(bench):
    write_rows(bench / "runs" / "J1-cruz", [row(100, 2.0)])
    out = RT.block({"job_id": "J1", "state": "running", "engine": "cruz"},
                   runs_dir=bench / "runs")
    assert out["active"] is True
    assert out["source"].endswith("J1-cruz/" + RT.ROWS_NAME)


def test_block_with_a_running_job_whose_directory_does_not_exist_yet(bench):
    out = RT.block({"job_id": "J1", "state": "running", "preset": "smoke-20"},
                   runs_dir=bench / "runs", presets=C.PRESETS)
    assert out["active"] is True                 # it IS running; we just cannot rate it yet
    assert out["reason"] == "no run directory for J1 yet"
    assert out["aggregate_tok_s"] is None


def test_block_with_a_run_directory_but_no_rows_written_yet(bench):
    (bench / "runs" / "J1-current").mkdir(parents=True)
    out = RT.block({"job_id": "J1", "state": "running"}, runs_dir=bench / "runs")
    assert out["active"] is True
    assert out["reason"] == "no rows written yet"
    assert out["rows_done"] == 0


def test_block_reports_the_run_in_progress(bench):
    write_rows(bench / "runs" / "J1-current", [row(100, 2.0), row(100, 8.0)])
    out = RT.block({"job_id": "J1", "state": "running", "preset": "smoke-20"},
                   runs_dir=bench / "runs", presets=C.PRESETS)
    assert out["active"] is True
    assert out["aggregate_tok_s"] == 20.0
    assert out["rows_done"] == 2
    assert out["rows_total"] == 21
    assert out["reason"] is None
    assert out["source"].endswith("J1-current/" + RT.ROWS_NAME)


# ------------------------------------------------------------------ wiring into /api/live
def test_build_live_without_the_preset_table_still_answers(bench):
    """The row total is a nicety; losing it must not lose the throughput.

    `job` is the dispatcher's status document -- the raw shape with the job nested -- because
    _job_block is the one place that knows how to read it. Passing an already-normalised block
    here silently drops the job id, which is the bug the raw-shape tests above pin.
    """
    write_rows(bench / "runs" / "20260101T000000Z-eval-smoke-20-current", [row(100, 2.0)])
    raw_status = {"state": "running",
                  "job": {"preset": "smoke-20", "engine": "current",
                          "job_id": "20260101T000000Z-eval-smoke-20"}}
    doc = LM.build_live(job=raw_status, sparks=[], fetch=lambda u: b"")
    assert doc["throughput"]["active"] is True
    assert doc["throughput"]["aggregate_tok_s"] == 50.0
    assert doc["throughput"]["rows_total"] is None       # no preset table was offered


def test_build_live_reads_the_status_document_the_dispatcher_actually_writes(bench):
    """The regression a live check caught: `state` is top level, the job id is nested.

    Everything above passes a NORMALISED job block, so the suite stayed green while /api/live
    answered `active: false` during a real run -- the block was reading `job_id` off the top
    of the status document, where it does not exist. This test uses the shape dispatcher.py
    actually writes, so the wiring cannot silently regress again.
    """
    write_rows(bench / "runs" / "20260101T000000Z-eval-smoke-20-current", [row(100, 2.0)])
    raw_status = {"schema": "zgx.console.dispatch.v1", "state": "running", "phase": "eval",
                  "queue_depth": 0, "rows_seen": 1,
                  "job": {"action": "eval", "preset": "smoke-20", "engine": "current",
                          "job_id": "20260101T000000Z-eval-smoke-20"}}
    doc = LM.build_live(job=raw_status, presets=C.PRESETS, sparks=[], fetch=lambda u: b"")
    t = doc["throughput"]
    assert t["active"] is True, "a running job in the dispatcher's own document is active"
    assert t["job_id"] == "20260101T000000Z-eval-smoke-20"
    assert t["preset"] == "smoke-20"
    assert t["rows_total"] == 21
    assert t["aggregate_tok_s"] == 50.0
    assert t["source"].endswith("-current/" + RT.ROWS_NAME)


def test_build_live_reads_a_finished_status_document_as_not_active(bench):
    """The same raw shape, after the job ends: the card must fall back to the archive."""
    raw_status = {"state": "done", "finished_utc": "2026-01-01T00:05:00Z",
                  "job": {"preset": "smoke-20", "engine": "current",
                          "job_id": "20260101T000000Z-eval-smoke-20"}}
    doc = LM.build_live(job=raw_status, presets=C.PRESETS, sparks=[], fetch=lambda u: b"")
    assert doc["throughput"]["active"] is False
    assert doc["throughput"]["reason"] == "job 20260101T000000Z-eval-smoke-20 is done"


def test_build_live_carries_the_block_even_when_nothing_is_running(bench):
    """Always present, so a consumer cannot mistake 'no run' for 'not wired up'."""
    doc = LM.build_live(job=None, presets=C.PRESETS, sparks=[], fetch=lambda u: b"")
    assert doc["throughput"]["active"] is False
    assert doc["throughput"]["reason"] == "no run active"
