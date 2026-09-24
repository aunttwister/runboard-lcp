"""corrected_metrics: the ONE definition of aggregate throughput.

The bug this module exists to fix: the runner divided completion tokens by the SUM of
per-request elapsed times, which is the per-stream mean under concurrency, so the published
figure fell as concurrency rose (57 -> 29 -> 16) while the box's real output stayed flat.
The corrected definition is tokens / wall-clock. Both consumers (the live page and the
Prometheus exporter) must therefore agree on it, and these tests pin the definition.
"""
from __future__ import annotations

import json

import pytest

import corrected_metrics as M


def _row(**over):
    row = {"phase": "c1", "ok": True, "completion_tokens": 100, "elapsed": 10.0,
           "ts": 1000.0, "concurrency": 2}
    row.update(over)
    return row


# ---------------------------------------------------------------- load_rows

def test_load_rows_reads_jsonl(tmp_path):
    p = tmp_path / "requests.jsonl"
    p.write_text(json.dumps(_row()) + "\n")
    assert M.load_rows(p) == [_row()]


def test_load_rows_skips_blank_and_corrupt_lines(tmp_path):
    p = tmp_path / "requests.jsonl"
    p.write_text(json.dumps(_row()) + "\n\n{truncated\n" + json.dumps(_row(ts=2000.0)) + "\n")
    rows = M.load_rows(p)
    assert [r["ts"] for r in rows] == [1000.0, 2000.0]


def test_load_rows_is_empty_when_there_is_nothing_to_read(tmp_path):
    assert M.load_rows(tmp_path / "absent.jsonl") == []
    assert M.load_rows(tmp_path) == []          # an unreadable path, not a crash


def test_load_rows_defaults_to_the_sandbox_run_directory(run_dir):
    (run_dir / "requests.jsonl").write_text(json.dumps(_row()) + "\n")
    assert len(M.load_rows()) == 1


# ---------------------------------------------------------------- summarize

def test_summarize_divides_by_wall_clock_not_by_the_sum_of_elapsed():
    rows = [_row(ts=1000.0, elapsed=10.0, completion_tokens=300),
            _row(ts=1010.0, elapsed=10.0, completion_tokens=300)]
    out = M.summarize(rows)["c1"]
    # wall clock is 20 s (1000-10 -> 1010), so 600 tokens over 20 s
    assert out["wall_seconds"] == 20.0
    assert out["aggregate_tok_s"] == 30.0
    assert out["per_stream_mean"] == 30.0
    assert out["scaling_x"] == 1.0
    assert out["requests"] == 2 and out["total_completion_tokens"] == 600
    assert out["concurrency"] == 2
    assert out["stream_utilization"] == 0.5       # 20 s of stream time over 20 s x 2
    assert out["source"] == "recomputed_from_raw"


def test_summarize_shows_batching_buying_something():
    rows = [_row(ts=1000.0, elapsed=10.0, completion_tokens=50),
            _row(ts=1005.0, elapsed=10.0, completion_tokens=50),
            _row(ts=1010.0, elapsed=10.0, completion_tokens=50)]
    out = M.summarize(rows)["c1"]
    assert out["per_stream_mean"] == 5.0
    assert out["aggregate_tok_s"] == 7.5          # 150 tokens over 20 s of wall clock
    assert out["scaling_x"] == 1.5                # > 1: concurrency helped
    assert out["stream_utilization"] == 0.75      # 30 s of stream time over 20 s x 2


def test_summarize_assumes_one_stream_when_concurrency_is_absent():
    out = M.summarize([_row(concurrency=None)])["c1"]
    assert out["concurrency"] == 1 and out["stream_utilization"] == 1.0


@pytest.mark.parametrize("over", [
    {"ok": False},                 # the request failed
    {"completion_tokens": None},   # nothing measured
    {"completion_tokens": 0},
    {"elapsed": None},
    {"elapsed": 0},
    {"elapsed": -1.0},             # a negative duration is not a measurement
    {"phase": None},               # cannot be attributed to a phase
])
def test_summarize_ignores_rows_it_cannot_measure(over):
    assert M.summarize([_row(**over)]) == {}


def test_summarize_groups_by_phase_and_drops_an_impossible_window():
    """The `wall <= 0` guard is defensive: for real (float) timestamps the row filter above
    guarantees a positive window, so it can only be reached with a clock that subtracts
    backwards. It is pinned anyway -- an unreachable guard that produced a ZeroDivisionError
    on the live page would be a bad way to find out it was wrong."""
    class _BackwardsClock(float):
        def __sub__(self, other):
            return 0.0

    assert M.summarize([_row(ts=_BackwardsClock(0.0))]) == {}
    both = [_row(), _row(phase="c2", ts=2000.0)]
    assert sorted(M.summarize(both)) == ["c1", "c2"]


# ---------------------------------------------------------------- rolling_aggregate

def test_rolling_aggregate_sums_only_the_trailing_window():
    rows = [_row(ts=950.0, completion_tokens=60),
            _row(ts=700.0, completion_tokens=600)]
    # only the row inside the last 60 s counts: 60 tokens / 60 s
    assert M.rolling_aggregate(rows, now=1000.0) == 1.0
    assert M.rolling_aggregate(rows, now=1000.0, window=300.0) == 2.2
    assert M.rolling_aggregate([], now=1000.0) is None


# ---------------------------------------------------------------- overlay

def test_overlay_says_plainly_that_there_are_no_rows_yet():
    state = {"phases": []}
    assert M.overlay(state, rows=[], now=1000.0) is state
    assert state["correction"] == {"applied": False, "reason": "no raw rows yet"}


def test_overlay_loads_the_rows_when_it_is_not_given_any(run_dir):
    state = {"phases": []}
    M.overlay(state, now=1000.0)
    assert state["correction"]["applied"] is False


def test_overlay_merges_the_corrected_phase_rows_in_place():
    state = {"phases": [{"name": "c1", "result": {"aggregate_tok_s": 999.0}},
                        {"name": "c2"},                       # no result to correct
                        {"name": "unknown", "result": {"n": 1}}]}
    out = M.overlay(state, rows=[_row(ts=990.0, completion_tokens=600, elapsed=10.0)],
                    now=1000.0)
    assert out is state
    assert state["phases"][0]["result"]["aggregate_tok_s"] == 60.0
    assert state["phases"][0]["result"]["source"] == "recomputed_from_raw"
    assert state["phases"][1] == {"name": "c2"}
    assert state["phases"][2]["result"] == {"n": 1}
    assert state["live_agg_tps"] == 10.0
    assert state["live_agg_source"].startswith("recomputed: sum(completion_tokens")
    assert state["correction"]["applied"] is True
    assert "divided by sum(per-request elapsed)" in state["correction"]["note"]
