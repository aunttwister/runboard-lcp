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

The numbers, what each one measures, and the two readings that were WRONG here until
2026-10-06 -- both were caught live, by the operator, from the card itself.

  * ``output_tps``    -- generated tokens per second of WALL CLOCK for the whole engine. The
    aggregate. It is a TIME AVERAGE, so it sags toward zero whenever the box pauses between
    requests: a low aggregate is not slow decoding, it is mostly idle time.
  * ``decode_tps``    -- generated tokens per second of DECODE TIME, summed over requests. A lane
    that decodes for one second contributes one second, so dividing by it removes idle time and
    yields the PER-STREAM rate (tokens per lane-second). It can therefore read HIGHER than
    ``output_tps`` on a box that is mostly waiting, and that is correct, not a bug.
    Ground truth, measured 2026-10-06: 4.63 lanes of decode work over 120 s produced 78.4 tok/s
    = 16.9 tok/s per lane, with ``decode_tps`` reading 13--16.5.
  * ``prefill_tps``   -- prompt tokens the GPU actually COMPUTED per second of prefill time.
  * ``ss_decode_tps`` -- one request's decode speed, measured only from intervals in which exactly
    one request finished. A dash most of the time on a busy box, by design.

DELETED ON 2026-10-06: ``stream_tps`` (``output_tps / running``). It was meaningless -- a two
minute time average divided by an instantaneous snapshot gauge, so the quotient describes no
single moment. Measured 1.21 tok/s against a true per-lane 13.5, and 48.4 against 16.1 an hour
earlier: arbitrary, not merely biased. ``decode_tps`` already IS the per-stream rate.

CONCURRENCY: read ``decoding`` (``streams{state="decoding"}``), never ``running``. The engine
documents ``tensorfold:requests_running`` as "Requests in prefill or decode", and sampling
confirms ``requests_running == streams{decoding} + streams{filling}`` exactly -- it counts
requests parked in PREFILL, which generate no decode tokens at all. That is how "8 concurrent
streams" can sit beside an aggregate BELOW the single-stream rate: 8 requests held, well under
one lane's worth of actual decode work.
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
    # The TRUE concurrency: lanes actually generating tokens, from the engine's per-state series.
    # Kept separate from `running` because the two disagree and the difference is the whole reason
    # an aggregate below the single-stream rate is possible.
    ("decoding", "Streams decoding", "", "vllm:streams_decoding", "int"),
    # The lane count -- the `--parallel` the engine was actually started with, read from the
    # engine rather than assumed. It is half of the reading: 7 of 8 lanes busy and 7 of 8 lanes
    # free are the same number and opposite situations, so a throughput figure without the cap
    # beside it cannot be interpreted. An engine that publishes no cap leaves this a DASH, never
    # an invented number (see FAMILY_ALIASES -- the canonical name has no vLLM counterpart).
    ("streams_max", "Concurrent streams max", "", "vllm:streams_max", "int"),
    # NOT the number of decoding streams. The engine HELP reads "Requests in prefill or decode",
    # and sampling confirms requests_running == streams{decoding} + streams{filling}. It was
    # labelled "concurrent streams in use" until 2026-10-06, and that mislabel is what made a
    # card with correct arithmetic look broken.
    ("running", "Requests in flight (prefill or decode)", "", "vllm:num_requests_running", "int"),
    ("filling", "Streams pre-filling", "", "vllm:streams_filling", "int"),
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

# ------------------------------------------------------------------ metric-name families
#
# The engine behind :18300 is swappable, and different engines publish different NAME
# FAMILIES for the same concepts: vLLM publishes ``vllm:*``; the TensorFold engine
# publishes ``tensorfold:*`` / ``tensorfold_health:*``. The tables above keep ONE canonical
# spelling (the vLLM one, which this card was built on) and the alias table below translates
# per family, detected from what the exposition actually contains. A family with no alias
# resolves to the canonical names -- so an engine we have no map for renders honest dashes,
# never a wrong-name null that looks like idle.
FAMILY_ALIASES = {
    "tensorfold": {
        "vllm:num_requests_running": "tensorfold:requests_running",
        # The lane cap (the engine's `--parallel`, published as
        # ``tensorfold_health:streams_max``). vLLM publishes no equivalent, so for that family the
        # canonical name resolves unchanged to a metric that does not exist and the card draws a
        # dash -- the honest answer to "what is the cap here", as against a plausible 0.
        "vllm:streams_max": "tensorfold_health:streams_max",
        # Per-STATE stream counts. The label selector is part of the name here, which is why
        # parse_engine keeps a label-qualified key as well as the bare one: summing the three
        # states would answer a different question than "how many lanes are decoding".
        "vllm:streams_decoding": 'tensorfold_health:streams{state="decoding"}',
        "vllm:streams_filling": 'tensorfold_health:streams{state="filling"}',
        "vllm:num_requests_waiting": "tensorfold:requests_waiting",
        "vllm:kv_cache_usage_perc": "tensorfold:kv_cache_usage_ratio",
        "vllm:generation_tokens_total": "tensorfold:generation_tokens_total",
        # The per-request generation-token sum, i.e. the numerator of `decode_tps`. Same quantity,
        # different spelling ("generated tokens of finished requests"). A MISSED mapping here is
        # silent -- the expression matches nothing, so the KPI becomes a permanent dash that reads
        # as "no decode happened". This one was missing until 2026-10-06; the guard
        # `test_every_name_the_card_queries_is_a_name_this_engine_publishes` is what stops the next
        # one, by requiring every name in every expression to appear in the engine's own inventory.
        "vllm:request_generation_tokens_sum": "tensorfold:generation_tokens_total",
        # TensorFold publishes no decode ``_count``; its cumulative finished-request
        # counter plays the same role for the single-stream rule (exactly one request
        # finished in the interval).
        "vllm:request_decode_time_seconds_count": "tensorfold_health:requests_total",
        "vllm:request_decode_time_seconds_sum": "tensorfold_health:decode_seconds_total",
        "vllm:time_to_first_token_seconds_bucket":
            "tensorfold:time_to_first_token_seconds_bucket",
        "vllm:spec_decode_num_accepted_tokens_total": "tensorfold:mtp_accepted_total",
        "vllm:spec_decode_num_draft_tokens_total": "tensorfold:mtp_drafted_total",
        "vllm:request_success_total": "tensorfold_health:requests_total",
        "vllm:e2e_request_latency_seconds_sum": "tensorfold:request_latency_seconds_sum",
        "vllm:e2e_request_latency_seconds_count": "tensorfold:request_latency_seconds_count",
    },
}

# Two KPIs cannot be expressed as a pure rename: TensorFold's prompt-token counter includes
# prefix-cached tokens, so KV-computed prefill must be computed by SUBTRACTION, and its
# prefix-cache signal is token-level rather than hits/queries. A family may therefore
# override the whole expression for a key.
FAMILY_EXPR = {
    "tensorfold": {
        "prefill_tps": ("(sum(rate(tensorfold:prompt_tokens_total[2m]))"
                        " - sum(rate(tensorfold_health:cached_tokens_total[2m])))"
                        " / sum(rate(tensorfold_health:prefill_seconds_total[2m]))"),
        "prefix_hit": ("100 * sum(rate(tensorfold_health:cached_tokens_total[30m]))"
                       " / sum(rate(tensorfold:prompt_tokens_total[30m]))"),
    },
}

# ...and the basis line travels with the expression, because the definition changed with it.
FAMILY_BASIS = {
    "tensorfold": {
        "prefill_tps": "prompt tokens minus prefix-cached tokens per second of prefill time,"
                       " finished requests, 2m",
        "prefix_hit": "cached/prompt tokens, 30m",
        "mtp_accept": "accepted/drafted tokens, 30m",
        "req_per_min": "all requests, 5m rate",
    },
}


def detect_family(sums) -> str | None:
    """Which metric-name family the exposition speaks, read off its own keys."""
    names = set(sums or {})
    if any(n.startswith("tensorfold") for n in names):
        return "tensorfold"
    if any(n.startswith("vllm:") for n in names):
        return "vllm"
    return None


def _m(metric: str, family) -> str:
    """Canonical metric name -> the family's spelling (unchanged for an unknown family)."""
    return FAMILY_ALIASES.get(family, {}).get(metric, metric)


def expr_for(key: str, family) -> str:
    """The PromQL for a RATE_ITEMS key, in the family's metric names."""
    override = FAMILY_EXPR.get(family, {}).get(key)
    if override:
        return override
    out = EXPR[key]
    for src, dst in FAMILY_ALIASES.get(family, {}).items():
        out = out.replace(src, dst)
    return out


# The chart. Four series would need two axes to stay honest, so the chart carries the three
# throughput numbers (same scale) and `running` remains a headline KPI.
# Every key the block publishes, in card order. The page's KPI wiring is checked against this
# tuple (tests/test_engine_metrics.py): a key published but never rendered, or rendered but never
# published, shows up only as an empty cell on the board -- which is indistinguishable from "no
# traffic" and is exactly the failure this card exists to avoid.
ITEM_KEYS = (tuple(k for k, *_ in GAUGE_ITEMS)
             + tuple(k for k, *_ in RATE_ITEMS)
             + ("ss_decode_tps", "gen_win", "last_gen"))

SERIES = [
    ("output_tps", "output tok/s (engine-wide)"),
    ("prefill_tps", "prefill tok/s"),
    ("decode_tps", "decode tok/s"),
]

SS_MAXLEN = int(os.environ.get("RUNBOARD_ENGINE_SS_SAMPLES", "20"))

_cache: dict = {"docs": {}, "at": {}, "series": {}, "series_at": {}, "series_family": {}}

# Single-stream decode: the rate ONE stream actually sees. vLLM's counters are engine-wide and
# carry no per-request label, so this can only be measured from an interval in which exactly one
# request finished -- and such intervals are exactly what "single stream" means. Kept in process:
# the live server polls this module every few seconds, so the deltas come for free with the gauge
# read that already happens.
_ss: dict = {"prev": None, "samples": []}


def reset_cache() -> None:
    """Drop the memoised documents and series (tests, and any caller that wants a cold read)."""
    _cache.update({"docs": {}, "at": {}, "series": {}, "series_at": {}, "series_family": {}})
    _ss.update({"prev": None, "samples": []})


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
        if "{" in head:
            # ...and the LABEL-QUALIFIED key too, so a gauge that means ONE state of a
            # multi-series metric (streams{state="decoding"}) can be read alone instead of
            # being summed with its siblings. Spacing inside the braces is stripped on both
            # sides so a lookup does not depend on how the exposition formats its labels.
            lkey = _label_key(head)
            sums[lkey] = sums.get(lkey, 0.0) + number
        seen += 1
    return sums, model, seen


def _label_key(head: str) -> str:
    """``name{a="b"}`` with every space removed, so a lookup is spacing-independent."""
    return head.replace(" ", "")


def note_single_stream(counters, now=None, family=None) -> None:
    """Record a single-stream decode sample, but only from an interval that finished ONE request.

    Intervals are skipped rather than averaged in when they finished none (nothing to measure) or
    two or more (that is aggregate throughput on a shared engine, which is NOT a per-stream rate --
    conflating the two is what makes a busy engine look slow). A non-positive delta is a counter
    reset, i.e. an engine restart, and is skipped too. Counter names resolve through the
    family's alias table, so a TensorFold engine feeds the same sampler.
    """
    gen = counters.get(_m("vllm:generation_tokens_total", family))
    dec = counters.get(_m("vllm:request_decode_time_seconds_sum", family))
    cnt = counters.get(_m("vllm:request_decode_time_seconds_count", family))
    prev = _ss["prev"]
    _ss["prev"] = (gen, dec, cnt, now)
    if prev is None or None in (gen, dec, cnt) or None in prev[:3]:
        return
    if cnt - prev[2] != 1:
        return
    dgen, ddec = gen - prev[0], dec - prev[1]
    if dgen <= 0 or ddec <= 0:
        return
    _ss["samples"].append(dgen / ddec)
    del _ss["samples"][:-SS_MAXLEN]


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


# Recency, from the COUNTER's own steps -- never from a rate. `rate(counter[2m])` is 0, not
# absent, on an idle engine: it can say "nothing in the last two minutes" but it can never say
# WHEN the engine last did anything. On 2026-09-28 that gap read as "the board has no data" while
# a 40-minute generation was in flight, which is the exact confusion this card exists to prevent.
RECENT_S = 3600               # an hour of steps answers "is it working now" without a second page
RECENT_STEP_S = 30            # the scrape interval, so the reported age is one step accurate
GEN_COUNTER_EXPR = "sum(vllm:generation_tokens_total)"


def recent(now=None, fetch=None, window_s=RECENT_S, step_s=RECENT_STEP_S, family=None):
    """(tokens produced in the window, seconds since the interval that produced them).

    Counter steps are summed directly, negative steps dropped -- a restart inside the window
    resets the counter, and the tokens served before it must not be lost or subtracted.
    ``(0, None)`` is a real measurement ("the counter was scraped and never advanced"); ``None``
    in either slot means the counter could not answer, which the page draws as a dash.
    """
    now = time.time() if now is None else now
    counter_expr = GEN_COUNTER_EXPR
    for src, dst in FAMILY_ALIASES.get(family, {}).items():
        counter_expr = counter_expr.replace(src, dst)
    pts = prom_range(counter_expr, window_s=window_s, step_s=step_s, now=now, fetch=fetch)
    if len(pts) < 2:                      # one sample cannot show a change, let alone when
        return None, None
    tokens = 0.0
    last = None
    prev = pts[0]["v"]
    for p in pts[1:]:
        step = p["v"] - prev
        prev = p["v"]
        if step <= 0:
            continue
        tokens += step
        last = p["t"]
    if last is None:
        return 0, None
    return int(round(tokens)), max(0, int(round(now - last)))


# ------------------------------------------------------------------ assembly

def _ago(seconds):
    """A human age, at the resolution the counter behind it can support (30 s scrape steps).

    "12 s ago" is a claim the 30 s grid can still make; "0.2 m ago" is not.
    """
    s = float(seconds)
    if s < 90:
        return f"{int(round(s))} s ago"
    if s < 5400:
        return f"{int(round(s / 60.0))} m ago"
    return f"{s / 3600.0:.1f} h ago"


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
    if fmt == "count":
        return f"{int(round(number)):,}"
    if fmt == "ago":
        return _ago(number)
    return number


def _item(key, label, unit, raw, fmt, basis):
    return {"key": key, "label": label, "unit": unit, "fmt": fmt,
            "value": _fmt_val(raw, fmt), "basis": basis}


def _openai_model(fetch=None):
    """The served model id from the OpenAI API, through the same fetch as everything else.

    Needed because not every engine labels its /metrics series with ``model_name``: the
    TensorFold engine publishes bare series, so the label probe in parse_engine comes back
    empty while the card still has to say WHAT is being served. The exllamav3 path is the
    mirror case (a /v1/models answer but no metrics at all), so the probe runs whenever the
    exposition left the model unnamed -- it degrades to None and the dash stays honest.
    """
    base = ENGINE_URL.rsplit("/metrics", 1)[0]
    try:
        payload = json.loads((fetch or http_get)(base + "/v1/models").decode("utf-8", "replace"))
        return ((payload.get("data") or [{}])[0] or {}).get("id") or None
    except Exception:
        return None


def series(now=None, fetch=None, window_s=1800, step_s=60, family=None):
    """The engine's throughput over the requested window, memoised per window.

    A Prometheus that dies mid-loop must not produce a half-drawn chart that looks like traffic,
    so the loop breaks and the last real chart is served instead (the document's source block is
    what says the source is down). The memoised document carries the metric-name family it was
    built for: an engine swap must not be answered from a chart of the previous engine.
    """
    now = time.time() if now is None else now
    key = f"{int(window_s)}:{int(step_s)}"
    cached = _cache["series"].get(key)
    if cached and _cache["series_family"].get(key) == family \
            and (now - _cache["series_at"].get(key, 0.0)) < SERIES_CACHE_S:
        return cached

    start = int(now) - int(window_s)
    n = int(window_s // step_s) + 1
    out = []
    for skey, label in SERIES:
        try:
            pts = prom_range(expr_for(skey, family), window_s=window_s, step_s=step_s,
                             now=now, fetch=fetch)
        except Exception:
            break
        values = on_grid(pts, start, int(step_s), n)
        if any(v is not None for v in values):
            out.append({"key": skey, "label": label, "start": start,
                        "step": int(step_s), "values": values})
    if out:
        _cache["series"][key] = out
        _cache["series_at"][key] = now
        _cache["series_family"][key] = family
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
    # ---- source 1: the engine itself (instantaneous gauges)
    sums: dict = {}
    family = None
    reachable = no_metrics = False
    model = None
    detail = ""

    try:
        sums, model, _seen = parse_engine(fetch(ENGINE_URL).decode("utf-8", "replace"))
        reachable = True
        no_metrics = not sums
        family = detect_family(sums)
        if model is None:
            # Bare exposition (TensorFold) or a metrics-less engine (exllamav3): ask the
            # OpenAI API what is being served before giving up on the name.
            model = _openai_model(fetch)
        detail = ("no metric series in the response" if no_metrics
                  else f"{len(sums)} series ({family or 'unrecognised'} names)")
    except Exception as exc:
        detail = type(exc).__name__
    # The sampler reads the counters this call already parsed, so it costs no extra request.
    note_single_stream(sums, now, family)
    # Emitted even when the engine did not answer, so the key set does not change under a
    # consumer that iterates it -- the VALUES are what go missing, never the shape.
    for key, label, unit, metric, fmt in GAUGE_ITEMS:
        items.append(_item(key, label, unit, sums.get(_m(metric, family)), fmt,
                           "engine gauge, read now"))
    sources = [{"name": "engine", "url": ENGINE_URL,
                "ok": reachable and not no_metrics, "detail": detail}]

    # ---- source 2: Prometheus (rates and quantiles; also the recorder for the chart)
    prom_ok = True
    answered = 0
    for key, label, unit, fmt, basis in RATE_ITEMS:
        raw = None
        if prom_ok:
            try:
                raw = instant(expr_for(key, family), fetch=fetch)
                answered += 1
            except Exception as exc:
                prom_ok = False
                sources.append({"name": "prometheus", "url": PROM_URL, "ok": False,
                                "detail": type(exc).__name__})
        items.append(_item(key, label, unit, raw, fmt,
                           FAMILY_BASIS.get(family, {}).get(key, basis)))
    # ---- source 2b: the counter's own steps -- when it last generated, and how much
    gen_win = last_gen = None
    if prom_ok:
        try:
            gen_win, last_gen = recent(now=now, fetch=fetch, family=family)
        except Exception as exc:
            prom_ok = False
            sources.append({"name": "prometheus", "url": PROM_URL, "ok": False,
                            "detail": type(exc).__name__})
    items.append(_item("gen_win", "Tokens (1 h)", "tok", gen_win, "count",
                       "counter increase over the last hour, measured -- 0 means it really "
                       "served none, a dash means the counter could not be read"))
    items.append(_item("last_gen", "Last generation", "", last_gen, "ago",
                       "time since the last 30 s interval in which output tokens appeared"))
    if prom_ok:
        sources.append({"name": "prometheus", "url": PROM_URL, "ok": True,
                        "detail": f"{answered} instant queries, job {PROM_JOB}"})

    # ...and what ONE stream gets, measured rather than divided. This is the number to read when
    # the question is "how fast is a single request being served", which is what the operator asks
    # for as "tok/s single stream". A dash means no interval in this process has yet finished
    # exactly one request -- never a zero.
    ss = sorted(_ss["samples"])
    items.append(_item("ss_decode_tps", "Single-stream decode", "tok/s",
                       (ss[len(ss) // 2] if ss else None), "num2",
                       f"measured over the last {len(ss)} interval(s) in which exactly one request "
                       "finished: tokens generated / that request's decode time"))

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
        "series": (series(now=now, fetch=fetch, window_s=window_s, step_s=step_s,
                          family=family)
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
