#!/usr/bin/env python3
"""Live throughput for the job that is ACTUALLY running.

Why this exists
  The Throughput panel was fed by the SOAK harness. `corrected_metrics` recomputes from
  `<RUNBOARD_LOAD>/run/requests.jsonl` -- a fixed log written by the 2026-09-24 soak -- and
  the exporter's `zgx_load_*` gauges are written by the soak harness only. During an EVAL
  run none of that moves, so on 2026-09-26 a smoke-20 emitting ~42 tok/s showed
  `aggregate tok/s 0.0`, `concurrency 4` (the soak's concurrency) and a chart of the five
  soak phases. The panel was not broken. It was answering about a different run. This
  module answers about the run in progress.

Source of truth
  `<RUNS>/<job_id>-<engine>/rows.jsonl`, written ROW BY ROW while the run proceeds, so a
  partial file is the normal case rather than an error. Read-only; nothing is invented --
  when the file is missing or still empty the block says so and names the reason.

Definitions (an eval run is sequential, so concurrency is 1)
  per-stream mean   = mean(completion_tokens / elapsed_s over rows written)
                      -> what a typical request achieved
  aggregate         = sum(completion_tokens) / sum(elapsed_s)
                      -> what the box emitted per second of work done so far
  Both are END-TO-END (prefill included), matching the convention `history_collector`
  already uses for eval-type throughput, and the block says so.

  Note the deliberate contrast with `corrected_metrics`, which fixes a *different* bug: the
  soak's runner divided tokens by the SUM of per-request elapsed times, which under
  concurrency equals the per-stream mean. For a sequential eval run that denominator is
  already correct, so no correction is needed or applied here.
"""
import json
import os
import statistics
from pathlib import Path

BENCH = Path(os.environ.get("RUNBOARD_BENCH", "/root/exl3-bench"))
# Must equal history_collector.RUNS. Kept as its own constant rather than an import so the
# modules stay standalone (nothing in this repo imports another), and pinned by a test.
RUNS = BENCH / "runs"
ROWS_NAME = "rows.jsonl"

# A job is only "the run in progress" in these states. Anything else -- done, failed, idle,
# queued (deferred because a measurement in flight owns the model), or a state this module
# has never seen -- is reported as NOT active, with the state named, so a stale job document
# can never masquerade as a live run. Read from dispatcher.py's own vocabulary:
# {idle, running, queued, done, failed}.
ACTIVE_STATES = ("running",)

NOTE = ("end-to-end rates from this run's rows (prefill included); eval rows run "
        "sequentially, so concurrency is 1")


def find_run_dir(job_id, runs_dir=None, engine=None):
    """The directory a job writes into: `<runs>/<job_id>-<engine>`.

    Matched on the job id EXACTLY or followed by `-`, so one job can never be attributed to
    another whose id merely starts with the same characters (`...smoke-20` must not claim
    `...smoke-20extra`).

    When the job document names an engine, `<job_id>-<engine>` is preferred, because that is
    the directory this particular run wrote. Only then does it fall back to the matches on
    disk, and if a job somehow left several the last by NAME is taken -- a deterministic
    tiebreak, not a claim about time. (Name order is time order for the job id itself, since
    ids are ISO timestamps, but an engine suffix decides the tail: `-cruz` sorts before
    `-current`, so the name alone cannot say which run is newer.)
    """
    if not job_id:
        return None
    base = Path(runs_dir) if runs_dir else RUNS
    if engine:
        named = base / f"{job_id}-{engine}"
        if named.is_dir():
            return named
    try:
        cands = sorted(p for p in base.iterdir()
                       if p.is_dir() and (p.name == job_id or p.name.startswith(job_id + "-")))
    except OSError:
        return None
    return cands[-1] if cands else None


def read_rows(path):
    """Rows written so far. A half-written trailing line is skipped, not an error."""
    rows = []
    try:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return rows


def compute(rows):
    """Throughput of the rows written so far, plus one chart point per usable row."""
    out = {"rows_done": len(rows), "errors": 0, "tokens_total": 0, "busy_s": 0.0,
           "aggregate_tok_s": None, "per_stream_tok_s": None,
           "series": {"ts": [], "agg_tps": [], "stream_tps": []}}
    good = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        if r.get("error"):
            out["errors"] += 1
        ct, el = r.get("completion_tokens"), r.get("elapsed_s")
        # a row written for adjudication carries no tokens and no grade: it counts as done
        # but contributes no rate, and must never be read as 0 tok/s.
        if ct and el and el > 0:
            good.append((ct, float(el)))
    if not good:
        return out
    tokens = sum(ct for ct, _ in good)
    busy = sum(el for _, el in good)
    out["tokens_total"] = tokens
    out["busy_s"] = round(busy, 2)
    out["aggregate_tok_s"] = round(tokens / busy, 2)
    out["per_stream_tok_s"] = round(statistics.mean(ct / el for ct, el in good), 2)
    t = 0.0
    tok = 0
    for ct, el in good:
        t += el
        tok += ct
        out["series"]["ts"].append(round(t, 2))
        out["series"]["agg_tps"].append(round(tok / t, 2))
        out["series"]["stream_tps"].append(round(ct / el, 2))
    return out


def block(job, runs_dir=None, presets=None):
    """The `/api/live` `throughput` block for the dispatcher's job document."""
    job = job if isinstance(job, dict) else {}
    jid = job.get("job_id")
    preset = (presets or {}).get(job.get("preset")) or {}
    out = {"active": False, "job_id": jid, "state": job.get("state"),
           "preset": job.get("preset"), "rows_done": 0, "rows_total": preset.get("rows_hint"),
           "errors": 0, "tokens_total": 0, "busy_s": 0.0,
           "aggregate_tok_s": None, "per_stream_tok_s": None,
           "series": {"ts": [], "agg_tps": [], "stream_tps": []},
           "source": None, "reason": None, "note": NOTE}
    if job.get("state") not in ACTIVE_STATES or not jid:
        out["reason"] = ("no run active" if not jid
                         else f"job {jid} is {job.get('state') or 'in an unknown state'}")
        return out
    out["active"] = True
    run_dir = find_run_dir(jid, runs_dir, engine=job.get("engine"))
    if run_dir is None:
        out["reason"] = f"no run directory for {jid} yet"
        return out
    path = run_dir / ROWS_NAME
    out["source"] = str(path)
    rows = read_rows(path)
    if not rows:
        out["reason"] = "no rows written yet"
        return out
    out.update(compute(rows))
    return out
