#!/usr/bin/env python3
"""Engine load: what :18300 is serving RIGHT NOW -- every request, eval or not.

Why this exists
---------------
The board could already describe one run (the running job's own ``rows.jsonl``) and it could
describe the last SOAK (the archive in ``/api/state``), but it could not answer "what is the
engine doing at this moment" when neither of those was true. Measured on 2026-09-28: the engine
was mid-generation for a client that was NOT an eval -- ``vllm:num_requests_running`` 1,
``generation_tokens_total`` rising at ~20 tok/s -- while the only load card on the page showed
the 2026-09-24 soak archive. Anything else that uses the box (an agent's own model calls, a
manual probe, a second client) was invisible and unrecorded on every surface.

Two sources, for the same reason live_metrics has two:

  * the engine's own ``/metrics`` on this box (loopback) -> the instantaneous gauges: requests
    running / waiting, KV-cache occupancy, the model being served, and whether the endpoint
    answers at all;
  * Prometheus, which has been scraping that same endpoint as job ``zgx-vllm`` all along (30 s
    scrape, 1 y retention) -> every rate and quantile, plus the history the chart is drawn from.

Reading the rates from Prometheus instead of differencing them here is deliberate: the recorder
already exists, so this process has to persist nothing (a web process that accumulates state is
a web process whose bug becomes a data bug), and the panel and the Grafana-era dashboard answer
from the same scrape of the same endpoint.

Honesty rules, the same three the rest of the board follows:

  * a value that cannot be read is ABSENT -> the page draws a dash. Never 0, because a zero on a
    throughput column is indistinguishable from a real idle measurement, and "idle vs busy" is
    the exact question this card exists to answer;
  * ``null`` for a rate means "no such traffic in that interval" (e.g. no request FINISHED in
    the window, which is the normal state while one long generation is in flight) -- not 0 tok/s;
  * the engine behind the port is swappable and the EXL3 path publishes no /metrics at all, so
    availability is probed on every read and stated, never hardcoded.
"""
from __future__ import annotations

import json
import math
import os
import time
import urllib.parse
import urllib.request

ENGINE_URL = os.environ.get("RUNBOARD_ENGINE_URL", "http://127.0.0.1:18300/metrics")
PROM_URL = os.environ.get("RUNBOARD_PROM_URL", "http://192.168.1.202:9090")
PROM_JOB = os.environ.get("RUNBOARD_ENGINE_PROM_JOB", "zgx-vllm")
PROM_RETENTION = os.environ.get("RUNBOARD_PROM_RETENTION", "1y")
TIMEOUT = float(os.environ.get("RUNBOARD_ENGINE_TIMEOUT", "6"))
# The page polls every 3 s; a dozen instant queries per poll would be a dozen queries per 3 s
# for numbers that only move on a 30 s scrape. The cache is what makes reading Prometheus on the
# request path affordable, and it is short enough that "read now" stays true to within one badge.
CACHE_S = float(os.environ.get("RUNBOARD_ENGINE_CACHE_S", "15"))
SERIES_CACHE_S = float(os.environ.get("RUNBOARD_ENGINE_SERIES_CACHE_S", "30"))

# Instantaneous gauges, read straight off the engine (one loopback call, no cache needed).
# (key, label, unit, metric name, format)
GAUGE_ITEMS = [
    ("running", "Requests running", "", "vllm:num_requests_running", "int"),
    ("waiting", "Requests waiting", "", "vllm:num_requests_waiting", "int"),
    ("kv_pct", "KV cache used", "%", "vllm:kv_cache_usage_perc", "pct"),
]

# ONE expression per number, used for BOTH the headline (instant query) and the chart (range
# query). Two copies of a formula eventually disagree, and the operator believes whichever one
# they opened first -- so the definition travels with the number.
EXPR = {
    "output_tps": "sum(rate(vllm:generation_tokens_total[2m]))",
    # KV-COMPUTED tokens, not prompt tokens: prompt_tokens_sum counts tokens served from the
    # prefix cache too, and this box runs at a 97 % hit rate, so the prompt-token form reported a
    # 16,292 tok/s "prefill" where the GPU actually computed 1,075 tok/s (both measured
    # 2026-09-28). A cache-heavy workload must not read as a faster prefill than a cold one.
    "prefill_tps": ("sum(rate(vllm:request_prefill_kv_computed_tokens_sum[2m]))"
                    " / sum(rate(vllm:request_prefill_time_seconds_sum[2m]))"),
    "decode_tps": ("sum(rate(vllm:request_generation_tokens_sum[2m]))"
                   " / sum(rate(vllm:request_decode_time_seconds_sum[2m]))"),
    "ttft_p50": ("histogram_quantile(0.50,"
                 " sum(rate(vllm:time_to_first_token_seconds_bucket[5m])) by (le))"),
    "ttft_p95": ("histogram_quantile(0.95,"
                 " sum(rate(vllm:time_to_first_token_seconds_bucket[5m])) by (le))"),
    # 30 m for the two ratios: both counters are sparse (a long generation can freeze the
    # accepted-token counter for minutes), and a 5 m ratio then reads a hard 0 % that says
    # "spec decode is doing nothing" when it means "nothing was accepted in five minutes".
    "prefix_hit": ("100 * sum(rate(vllm:prefix_cache_hits_total[30m]))"
                   " / sum(rate(vllm:prefix_cache_queries_total[30m]))"),
    "mtp_accept": ("100 * sum(rate(vllm:spec_decode_num_accepted_tokens_total[30m]))"
                   " / sum(rate(vllm:spec_decode_num_draft_tokens_total[30m]))"),
    "req_per_min": "60 * sum(rate(vllm:request_success_total[5m]))",
    "e2e_ms": ("1000 * sum(rate(vllm:e2e_request_latency_seconds_sum[5m]))"
               " / sum(rate(vllm:e2e_request_latency_seconds_count[5m]))"),
}

# (key, label, unit, format, basis) -- what each headline number is, so a reader never has to
# guess whether it is decode-only, end-to-end, or engine-wide.
RATE_ITEMS = [
    ("output_tps", "Output tok/s", "tok/s", "num2",
     "generated tokens/s for the whole engine, 2 m rate (moves while a request is in flight)"),
    ("prefill_tps", "Prefill tok/s", "tok/s", "num2",
     "prompt tokens the GPU actually computed per second of prefill time, finished requests, 2 m"),
    ("decode_tps", "Decode tok/s", "tok/s", "num2",
     "generated tokens per second of decode time, finished requests, 2 m"),
    ("ttft_p50", "TTFT p50", "s", "num2", "5 m histogram quantile"),
    ("ttft_p95", "TTFT p95", "s", "num2", "5 m histogram quantile"),
    ("prefix_hit", "Prefix hit", "%", "num1", "hits/queries, 30 m"),
    ("mtp_accept", "MTP accepted", "%", "num1", "accepted/draft tokens, 30 m"),
    ("req_per_min", "Requests", "/min", "num1", "finished requests, 5 m rate"),
    ("e2e_ms", "End-to-end", "ms", "num1", "mean over finished requests, 5 m"),
]

# The chart. Four series would need two axes to stay honest, so the chart carries the three
# throughput numbers (same scale) and `running` remains a headline KPI.
# Every key the block publishes, in card order. The page's KPI wiring is checked against this
# tuple (tests/test_engine_metrics.py): a key published but never rendered, or rendered but never
# published, shows up only as an empty cell on the board -- which is indistinguishable from "no
# traffic" and is exactly the failure this card exists to avoid.
ITEM_KEYS = (tuple(k for k, *_ in GAUGE_ITEMS)
             + tuple(k for k, *_ in RATE_ITEMS)
             + ("stream_tps",))

SERIES = [
    ("output_tps", "output tok/s (engine-wide)"),
    ("prefill_tps", "prefill tok/s"),
    ("decode_tps", "decode tok/s"),
]

_cache: dict = {"docs": {}, "at": {}, "series": {}, "series_at": {}}


def reset_cache() -> None:
    """Drop the memoised documents and series (tests, and any caller that wants a cold read)."""
    _cache.update({"docs": {}, "at": {}, "series": {}, "series_at": {}})


# ------------------------------------------------------------------ transport

def http_get(url, timeout=None):
    """One GET -> bytes. Raises on failure; callers decide what that means."""
    req = urllib.request.Request(url, headers={"User-Agent": "zgx-engine-load/1"})
    with urllib.request.urlopen(req, timeout=timeout or TIMEOUT) as fh:  # noqa: S310 - fixed URLs
        return fh.read()


# ------------------------------------------------------------------ parsing

def parse_engine(text):
    """Prometheus exposition text -> ({metric: summed value}, model_name|None, series seen).

    vLLM publishes labelled series (``{model_name="...",engine="0"}``), so every series of a
    metric is summed: the question here is about the box, not about one label.
    """
    sums: dict = {}
    model = None
    seen = 0
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        head, _, val = line.rpartition(" ")
        name = head
        if "{" in head:
            name, _, rest = head.partition("{")
            for part in rest.rstrip("}").split(","):
                k, _, v = part.partition("=")
                if k.strip() == "model_name" and model is None:
                    model = v.strip().strip('"')
        try:
            number = float(val)
        except (TypeError, ValueError):
            continue                      # NaN/Inf/text are dropped, never smuggled through
        if not math.isfinite(number) or not name:
            continue
        sums[name] = sums.get(name, 0.0) + number
        seen += 1
    return sums, model, seen


def _vector_value(payload):
    """First finite value out of a Prometheus instant-query payload, else None."""
    for res in (payload.get("data") or {}).get("result") or []:
        pair = res.get("value") or [None, None]
        try:
            number = float(pair[1])
        except (TypeError, ValueError, IndexError):
            continue
        return number if math.isfinite(number) else None
    return None


def instant(query, fetch=None):
    """One instant PromQL query -> float, or None when there is no usable number.

    None covers all three of "no series", ``NaN`` (0/0 -- the counter pair that defines a rate
    saw nothing) and ``+Inf``. The caller renders a dash; a fabricated 0 would read as idle.
    """
    fetch = fetch or http_get
    url = f"{PROM_URL}/api/v1/query?query=" + urllib.parse.quote(query)
    return _vector_value(json.loads(fetch(url).decode("utf-8")))


def prom_range(query, window_s=1800, step_s=60, now=None, fetch=None):
    """Prometheus query_range -> [{t, v}] with gaps left out (never interpolated)."""
    fetch = fetch or http_get
    end = int(now if now is not None else time.time())
    start = end - int(window_s)
    url = (f"{PROM_URL}/api/v1/query_range?query={urllib.parse.quote(query)}"
           f"&start={start}&end={end}&step={int(step_s)}")
    payload = json.loads(fetch(url).decode("utf-8"))
    out = []
    for res in (payload.get("data") or {}).get("result") or []:
        for ts, val in res.get("values") or []:
            try:
                number = float(val)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                out.append({"t": int(ts), "v": number})
    return out


def on_grid(points, start, step, n):
    """Project [{t,v}] onto the fixed step grid the chart draws, with gaps as ``null``.

    query_range answers ON the step grid, but a series that has no data over part of the window
    comes back SHORTER than its neighbours. Plotting the raw arrays positionally (the first cut
    of this did) silently shifts one line against another, which reads as a traffic pattern that
    never happened. One grid, one index per timestamp.
    """
    out: list = [None] * n
    for p in points:
        i = int(round((p["t"] - start) / step))
        if 0 <= i < n:
            out[i] = p["v"]
    return out


# ------------------------------------------------------------------ assembly

def _fmt_val(value, fmt):
    """None (and anything non-finite) stays None: the page renders a dash, never a zero."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    if fmt == "int":
        return int(round(number))
    if fmt == "num1":
        return round(number, 1)
    if fmt == "num2":
        return round(number, 2)
    if fmt == "pct":
        return round(number * 100.0, 1)
    return number


def _item(key, label, unit, raw, fmt, basis):
    return {"key": key, "label": label, "unit": unit, "fmt": fmt,
            "value": _fmt_val(raw, fmt), "basis": basis}


def series(now=None, fetch=None, window_s=1800, step_s=60):
    """The engine's throughput over the requested window, memoised per window.

    A Prometheus that dies mid-loop must not produce a half-drawn chart that looks like traffic,
    so the loop breaks and the last real chart is served instead (the document's source block is
    what says the source is down).
    """
    now = time.time() if now is None else now
    key = f"{int(window_s)}:{int(step_s)}"
    cached = _cache["series"].get(key)
    if cached and (now - _cache["series_at"].get(key, 0.0)) < SERIES_CACHE_S:
        return cached

    start = int(now) - int(window_s)
    n = int(window_s // step_s) + 1
    out = []
    for skey, label in SERIES:
        try:
            pts = prom_range(EXPR[skey], window_s=window_s, step_s=step_s, now=now, fetch=fetch)
        except Exception:
            break
        values = on_grid(pts, start, int(step_s), n)
        if any(v is not None for v in values):
            out.append({"key": skey, "label": label, "start": start,
                        "step": int(step_s), "values": values})
    if out:
        _cache["series"][key] = out
        _cache["series_at"][key] = now
        return out
    return cached or []


def _note(reachable: bool, no_metrics: bool, prom_ok: bool, model) -> str:
    """One honest sentence pair: what this card is, and why a number may be missing."""
    if not reachable:
        head = ("the serving engine on :18300 did not answer /metrics, so nothing here is "
                "sampled right now")
    elif no_metrics:
        head = ("the serving engine on :18300 answered but published no metric series -- the "
                "EXL3 path exposes none, so there is nothing to read while it serves")
    else:
        head = ("every request this engine served, whoever sent it (model %s) -- an eval run is "
                "not required for this card" % (model or "unknown"))
    if prom_ok:
        tail = ("rates come from the engine's own counters over the trailing window, so a gap or "
                "a dash means that interval carried no such traffic -- it is not zero")
    else:
        tail = ("the rates could not be read from Prometheus, so only the gauges above are "
                "sampled -- those come from the engine directly")
    return f"{head}. {tail}."


def _ratio(item_a, item_b):
    """a / b from two already-built items: None when either side is absent or b is zero.

    Dividing by a zero `running` is not "0 tok/s per stream", it is "no stream to divide by" --
    the same rule as every other number on this card.
    """
    a, b = item_a.get("value"), item_b.get("value")
    if a is None or b is None or b == 0:
        return None
    return a / b


def block(now=None, fetch=None, use_cache=True, window_s=1800, step_s=60):
    """Assemble the engine block. Never raises: a dead source degrades this card only."""
    fetch = fetch or http_get
    now = time.time() if now is None else now
    # Per window, like the spark cache: a document assembled for 30 m must never answer a 6 h
    # request, because the chart it carries is a different shape.
    ckey = f"{int(window_s)}:{int(step_s)}"
    if use_cache and _cache["docs"].get(ckey) is not None \
            and (now - _cache["at"].get(ckey, 0.0)) < CACHE_S:
        return dict(_cache["docs"][ckey])

    items: list = []
    reachable = no_metrics = False
    model = None
    detail = ""

    # ---- source 1: the engine itself (instantaneous gauges)
    sums: dict = {}
    try:
        sums, model, _seen = parse_engine(fetch(ENGINE_URL).decode("utf-8", "replace"))
        reachable = True
        no_metrics = not sums
        detail = "no metric series in the response" if no_metrics else f"{len(sums)} series"
    except Exception as exc:
        detail = type(exc).__name__
    # Emitted even when the engine did not answer, so the key set does not change under a
    # consumer that iterates it -- the VALUES are what goes missing, never the shape.
    for key, label, unit, metric, fmt in GAUGE_ITEMS:
        items.append(_item(key, label, unit, sums.get(metric), fmt, "engine gauge, read now"))
    sources = [{"name": "engine", "url": ENGINE_URL,
                "ok": reachable and not no_metrics, "detail": detail}]

    # ---- source 2: Prometheus (rates and quantiles; also the recorder for the chart)
    prom_ok = True
    answered = 0
    for key, label, unit, fmt, basis in RATE_ITEMS:
        raw = None
        if prom_ok:
            try:
                raw = instant(EXPR[key], fetch=fetch)
                answered += 1
            except Exception as exc:
                prom_ok = False
                sources.append({"name": "prometheus", "url": PROM_URL, "ok": False,
                                "detail": type(exc).__name__})
        items.append(_item(key, label, unit, raw, fmt, basis))
    if prom_ok:
        sources.append({"name": "prometheus", "url": PROM_URL, "ok": True,
                        "detail": f"{answered} instant queries, job {PROM_JOB}"})

    # The per-stream decode rate: engine-wide output over requests running. It is the number the
    # retired Grafana dashboard called "THE decoding capability metric", and it is what "tok/s"
    # means to a reader asking how fast ONE client is being served -- the question behind "whatever
    # L1 is doing". Derived from the two items above (their DISPLAYED values, so it cannot
    # disagree with them by a rounding) rather than a third query of its own.
    by_key = {it["key"]: it for it in items}
    items.append(_item("stream_tps", "tok/s per stream", "tok/s",
                       _ratio(by_key["output_tps"], by_key["running"]), "num2",
                       "output rate / requests running, read now"))

    doc = {
        "ok": bool((reachable and not no_metrics) or prom_ok),
        "reachable": bool(reachable and not no_metrics),
        "no_metrics": bool(no_metrics),
        "model": model,
        "at": now,
        "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        "sources": sources,
        "items": items,
        # ...and the chart is not re-asked either: with the source already down, four more
        # range queries are four more timeouts. The last real chart is served instead.
        "series": (series(now=now, fetch=fetch, window_s=window_s, step_s=step_s)
                   if prom_ok else (_cache["series"].get(ckey) or [])),
        "series_keys": [k for k, _ in SERIES],
        "recorded": (f"Prometheus job {PROM_JOB} scrapes this same endpoint every 30 s and keeps "
                     f"{PROM_RETENTION}, so the history survives with no recorder here"),
        "note": _note(reachable, no_metrics, prom_ok, model),
        "cache_s": CACHE_S,
    }
    if use_cache:
        _cache["docs"][ckey] = doc
        _cache["at"][ckey] = now
    return dict(doc)


if __name__ == "__main__":                 # on-box probe: what will the card say?
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "json":
        print(json.dumps(block(use_cache=False), indent=1, default=str))
    else:
        b = block(use_cache=False)
        print(f"engine {ENGINE_URL} reachable={b['reachable']} model={b['model']}")
        for s in b["sources"]:
            print(f"  {s['name']:<11} ok={s['ok']} {s['detail']}")
        for it in b["items"]:
            print(f"  {it['label']:<18} {it['value']} {it['unit']}")
        print(f"  series: {[(s['key'], sum(1 for v in s['values'] if v is not None)) for s in b['series']]}")
        print(f"  {b['note']}")
