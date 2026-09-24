#!/usr/bin/env python3
"""Corrected throughput view derived from the raw request log.

The runner's recorded `aggregate_tok_s` divided completion tokens by the SUM of
per-request elapsed times. For concurrent requests of similar length that is
algebraically the per-stream mean, so the figure FELL as concurrency rose —
57 -> 29 -> 16 tok/s across c1/c2/c4 on 2026-09-24 — while the box's real output
stayed flat (57 -> 58 -> 62). Nothing was slow; the denominator was wrong.

This module recomputes the honest numbers from requests.jsonl and overlays them, so
both consumers (the live page server and the Prometheus exporter) present ONE
consistent corrected view. The runner's own recorded values stay on disk untouched —
nothing here rewrites history, and `source` records where each figure came from.

Definitions:
  aggregate_tok_s   = sum(completion_tokens) / (last completion - first start)
                      -> tokens the box actually emits per second of wall clock
  per_stream_mean   = mean(completion_tokens / elapsed) per request (unchanged)
  scaling_x         = aggregate / per_stream_mean; ~= concurrency when the engine
                      scales perfectly, 1.0 means batching buys nothing
  stream_utilization= sum(elapsed) / (wall * concurrency)
"""
import os
import json
import time
from pathlib import Path

RUN = Path(os.environ.get("RUNBOARD_LOAD", "/root/load")) / "run"
WINDOW_S = 60.0


def load_rows(path=None):
    p = Path(path) if path else RUN / "requests.jsonl"
    rows = []
    try:
        with open(p) as fh:
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


def summarize(rows):
    """Corrected per-phase metrics keyed by phase name."""
    groups = {}
    for r in rows:
        if r.get("ok") is False:
            continue
        ct, el = r.get("completion_tokens"), r.get("elapsed")
        if not ct or not el or el <= 0 or not r.get("phase"):
            continue
        groups.setdefault(r["phase"], []).append(r)

    out = {}
    for phase, rs in groups.items():
        conc = rs[0].get("concurrency") or 1
        tok = sum(r["completion_tokens"] for r in rs)
        t_start = min(r["ts"] - r["elapsed"] for r in rs)
        t_end = max(r["ts"] for r in rs)
        wall = t_end - t_start
        if wall <= 0:
            continue
        per_stream = [r["completion_tokens"] / r["elapsed"] for r in rs]
        ps_mean = sum(per_stream) / len(per_stream)
        agg = tok / wall
        busy = sum(r["elapsed"] for r in rs)
        out[phase] = {
            "requests": len(rs),
            "total_completion_tokens": tok,
            "wall_seconds": round(wall, 2),
            "aggregate_tok_s": round(agg, 2),
            "per_stream_mean": round(ps_mean, 2),
            "scaling_x": round(agg / ps_mean, 2) if ps_mean else None,
            "stream_utilization": round(busy / (wall * conc), 3),
            "concurrency": conc,
            "source": "recomputed_from_raw",
        }
    return out


def rolling_aggregate(rows, now, window=WINDOW_S):
    if not rows:
        return None
    tok = sum(r.get("completion_tokens", 0) for r in rows if now - r["ts"] <= window)
    return round(tok / window, 2)


def overlay(state, rows=None, now=None):
    """Return `state` with corrected throughput figures merged in (mutates in place)."""
    if now is None:
        now = time.time()
    if rows is None:
        rows = load_rows()
    if not rows:
        state["correction"] = {"applied": False, "reason": "no raw rows yet"}
        return state

    phases = summarize(rows)
    state["live_agg_tps"] = rolling_aggregate(rows, now)
    state["live_agg_source"] = "recomputed: sum(completion_tokens in last 60s) / 60"
    for p in state.get("phases", []):
        fix = phases.get(p.get("name"))
        if fix and p.get("result"):
            p["result"].update(fix)
    state["correction"] = {
        "applied": True,
        "note": ("aggregate_tok_s = total completion tokens / wall-clock seconds. The "
                 "runner divided by sum(per-request elapsed), which equals the "
                 "per-stream mean under concurrency and falls as concurrency rises."),
    }
    return state
