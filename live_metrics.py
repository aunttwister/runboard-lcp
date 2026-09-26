#!/usr/bin/env python3
"""Live telemetry for the console page -- the "fit grafana in here" layer.

Two sources, on purpose:

  * the exporter on THIS box (``:9400/metrics``) for CURRENT values -- one local HTTP
    call, always fresh, and it is the same exporter Prometheus scrapes so the page and
    the dashboard can never disagree about what the machine is doing;
  * Prometheus on the monitoring host for the HISTORY needed to draw sparklines --
    we do not keep a ring buffer here, because a web process that accumulates state
    is a web process whose bug becomes a data bug.

What this deliberately does NOT do is pretend the vLLM metrics still exist. The EXL3
engine exposes no ``/metrics`` endpoint at all, so ``vllm:time_to_first_token_*``,
KV-cache usage, prefix-cache hit rate and spec-decode acceptance have no source while
EXL3 serves :18300. They are reported as unavailable rather than approximated --
a panel that invents a plausible TTFT is worse than a panel that says it has none.

Honesty rules carried over from the exporter and the load harness:

  * a value that could not be sampled is ABSENT (the page renders a dash), never 0 --
    a fake 0 on a power graph is indistinguishable from real idle;
  * the load metrics are NOT live when no run is active. ``zgx_load_running`` decides
    that, and when it is 0 the numbers are labelled with their age instead of being
    presented as "now";
  * derived rates keep ``null`` as "no traffic in that interval" rather than 0.
"""
import json
import math
import os
import re
import time
import urllib.parse
import urllib.request

import run_throughput as RT

# One env var per source so a checkout/test never talks to production by default.
EXPORTER_URL = os.environ.get("RUNBOARD_EXPORTER_URL", "http://127.0.0.1:9400/metrics")
PROM_URL = os.environ.get("RUNBOARD_PROM_URL", "http://192.168.1.202:9090")
TIMEOUT = 6

# Sparkline windows. The operator's grafana link carried from=now-6h, so the console has
# to be able to look back at least that far or retiring grafana would lose the view. Step
# scales with the window so the point count stays ~SPARK_POINTS whatever is selected --
# a fixed 60 s step over 24 h would be 1440 points per series for no extra information.
SPARK_POINTS = 180
WINDOWS = {"30m": 1800, "6h": 21600, "24h": 86400}
DEFAULT_WINDOW = "30m"
SPARK_WINDOW_S = WINDOWS[DEFAULT_WINDOW]
SPARK_STEP_S = 60
SPARK_CACHE_S = 30


def window_spec(name):
    """(window_name, seconds, step_s) for a requested window; unknown falls back."""
    key = name if isinstance(name, str) and name in WINDOWS else DEFAULT_WINDOW
    secs = WINDOWS[key]
    return key, secs, max(60, secs // SPARK_POINTS)


SPARKS = [
    ("gpu_util", "GPU util %", "%", "zgx_gpu_utilization_percent"),
    ("gpu_power", "GPU power W", "W", "zgx_gpu_power_watts"),
    ("gpu_temp", "GPU temp C", "C", "zgx_gpu_temperature_celsius"),
    ("per_stream", "Per-stream tok/s", "tok/s", "zgx_load_live_per_stream_tok_s"),
]

# (key, label, unit, exporter metric, format)
MACHINE = [
    ("serving_up", "Serving :18300", "", "zgx_serving_up", "updown"),
    ("gpu_util", "GPU util", "%", "zgx_gpu_utilization_percent", "int"),
    ("gpu_power", "GPU power", "W", "zgx_gpu_power_watts", "num1"),
    ("gpu_temp", "GPU temp", "C", "zgx_gpu_temperature_celsius", "int"),
    ("gpu_clock", "GPU clock", "MHz", "zgx_gpu_clock_mhz", "int"),
    ("mem_avail", "Unified mem free", "GB", "zgx_memory_available_bytes", "gb"),
]

# (key, label, unit, exporter metric, format) -- only meaningful during a load run.
RUN_ITEMS = [
    ("concurrency", "Concurrency", "", "zgx_load_phase_concurrency", "int"),
    ("aggregate_tok_s", "Aggregate decode", "tok/s", "zgx_load_live_aggregate_tok_s", "num2"),
    ("per_stream_tok_s", "Per-stream mean", "tok/s", "zgx_load_live_per_stream_tok_s", "num2"),
    ("requests", "Requests measured", "", "zgx_load_requests_total", "int"),
    ("errors", "Request errors", "", "zgx_load_errors_total", "int"),
]

# Metrics the Grafana vLLM dashboard showed that have NO source while EXL3 serves the
# port. Named explicitly so the page can state the gap instead of leaving it implied.
UNAVAILABLE = [
    "TTFT (p50/p90)",
    "KV cache usage",
    "prefix-cache hit rate",
    "spec-decode / MTP acceptance",
    "requests running / waiting",
]

_LINE = re.compile(r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?P<labels>\{.*\})?\s+(?P<value>\S+)")


def parse_prom_text(text):
    """Parse Prometheus exposition text -> {metric_name: value} for unlabelled series.

    Labelled series are kept separately (they are per-phase results, not a machine
    gauge) and non-numeric values (NaN/Inf) are dropped rather than smuggled through as
    a float that would render as 'nan' on the page.
    """
    plain, labelled = {}, {}
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _LINE.match(line)
        if not m:
            continue
        try:
            val = float(m.group("value"))
        except (TypeError, ValueError):
            continue
        if math.isnan(val) or math.isinf(val):
            continue
        name = m.group("name")
        if m.group("labels"):
            labelled[(name, m.group("labels"))] = val
        else:
            plain[name] = val
    return plain, labelled


def http_get(url, timeout=TIMEOUT):
    """One GET -> bytes. Raises on any failure; callers decide what that means."""
    req = urllib.request.Request(url, headers={"User-Agent": "zgx-console-live/1"})
    with urllib.request.urlopen(req, timeout=timeout) as fh:  # noqa: S310 - fixed URLs
        return fh.read()


def prom_range(query, window_s=SPARK_WINDOW_S, step_s=SPARK_STEP_S, now=None,
               fetch=None):
    """Prometheus query_range -> [{t, v}] with nulls preserved as gaps.

    Prometheus answers a range query for a bursty series with fewer points than the
    window implies; those missing intervals are ""no traffic", so they stay out of the
    series (the chart draws a gap) rather than being interpolated into a fake line.
    """
    fetch = fetch or (lambda u: http_get(u))
    end = int(now if now is not None else time.time())
    start = end - int(window_s)
    url = (f"{PROM_URL}/api/v1/query_range?query={urllib.parse.quote(query)}"
           f"&start={start}&end={end}&step={int(step_s)}")
    payload = json.loads(fetch(url).decode("utf-8"))
    out = []
    for res in (payload.get("data") or {}).get("result") or []:
        for ts, val in res.get("values") or []:
            try:
                v = float(val)
            except (TypeError, ValueError):
                continue
            if math.isnan(v):
                continue
            out.append({"t": int(ts), "v": v})
    return out


def _fmt_val(value, fmt):
    """Format a value for display. None stays None so the page can render a dash."""
    if value is None:
        return None
    try:
        if fmt == "int":
            return int(round(float(value)))
        if fmt == "num1":
            return round(float(value), 1)
        if fmt == "num2":
            return round(float(value), 2)
        if fmt == "gb":
            return round(float(value) / (1024 ** 3), 1)
        return value
    except (TypeError, ValueError):
        return None


def _items(spec, plain):
    out = []
    for key, label, unit, metric, fmt in spec:
        raw = plain.get(metric)
        out.append({"key": key, "label": label, "unit": unit,
                    "value": _fmt_val(raw, fmt), "fmt": fmt})
    return out


def _job_block(status):
    """The dispatcher's own view of whether a run is active.

    This is load-bearing. The exporter's `zgx_load_*` gauges are written by the SOAK
    harness only: during an eval preset run they still hold the last soak's final numbers
    while the GPU sits at 90 %+ and the dispatcher is mid-job. Gating the "is anything
    running" question on those gauges made the page announce "no load run active" on a box
    that was demonstrably busy -- caught only by watching a real run, because the tests
    fake the exporter.
    """
    if not isinstance(status, dict):
        return {"active": False}
    job = status.get("job") if isinstance(status.get("job"), dict) else {}
    state = status.get("state")
    active = state in ("queued", "running")
    out = {"state": state, "active": active, "job_id": job.get("job_id"),
           "preset": job.get("preset")}
    if active:
        out["phase"] = status.get("phase")
        out["elapsed_s"] = status.get("elapsed_s")
    return out


def build_live(now=None, fetch=None, exporter_url=None, sparks=None, window=None, job=None,
               presets=None):
    """Assemble the /api/live document. Never raises: a dead source degrades the page.

    `sparks` lets the caller pass cached series in; when it is None the range queries
    run. Either way a Prometheus failure empties the series and marks the source down,
    while the current values (which come from the local exporter) still render.
    `job` is the dispatcher's status document, which is what says whether a run is
    actually in progress (see _job_block).
    `presets` is console_core.PRESETS, passed in rather than imported so the modules stay
    standalone; it supplies each preset's row count for the throughput block.
    """
    fetch = fetch or (lambda u: http_get(u))
    now = time.time() if now is None else now
    wname, wsecs, wstep = window_spec(window)
    doc = {"schema": "zgx.console.live.v1", "fetched_utc":
           time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
           "sources": [], "machine": [], "run": {}, "sparks": [],
           "unavailable": UNAVAILABLE, "age_s": None,
           "window": wname, "window_s": wsecs, "step_s": wstep,
           "job": _job_block(job)}

    # ---- throughput of the run in progress (see run_throughput): the ONLY place a live
    # eval run's tok/s exists. Always present, so a consumer never has to guess whether the
    # absence of the block means "no run" or "not wired up". It consumes doc["job"] -- the
    # NORMALISED block -- not the raw status document, because the status document keeps
    # `state` at the top level while the job's own id/preset/engine sit under `job`;
    # _job_block is the one place that knows that shape.
    doc["throughput"] = RT.block(doc["job"], presets=presets)

    # ---- source 1: the local exporter (current values)
    plain = {}
    try:
        text = fetch(exporter_url or EXPORTER_URL).decode("utf-8")
        plain, _labelled = parse_prom_text(text)
        doc["sources"].append({"name": "exporter", "url": exporter_url or EXPORTER_URL,
                               "ok": True, "detail": f"{len(plain)} series"})
    except Exception as exc:
        doc["sources"].append({"name": "exporter", "url": exporter_url or EXPORTER_URL,
                               "ok": False, "detail": f"{type(exc).__name__}"})

    doc["machine"] = _items(MACHINE, plain)

    # ---- the load block: only "live" while a run is actually running
    running = plain.get("zgx_load_running")
    age = plain.get("zgx_load_state_age_seconds")
    is_running = bool(running) and running >= 1
    doc["run"] = {
        "running": 1 if is_running else 0,
        "state_age_s": None if age is None else round(age, 1),
        "items": _items(RUN_ITEMS, plain),
    }
    if age is not None:
        doc["run"]["age_label"] = _age_label(age)
    if not is_running:
        # These four gauges belong to the SOAK harness. Saying "no run active" here was
        # wrong while an eval preset was mid-flight (see _job_block) -- so the wording now
        # names whose gauges they are, and the job block above carries the real answer.
        doc["run"]["note"] = ("load-soak gauges: the soak harness writes these and an eval "
                              "run does not, so they still hold the last soak's final numbers"
                              if age is not None and age > 120 else
                              "no load-soak running")

    # ---- source 2: Prometheus (history only)
    if sparks is None:
        sparks, prom_ok, prom_detail = [], False, "not queried"
        try:
            for key, _label, _unit, metric in SPARKS:
                series = prom_range(metric, window_s=wsecs, step_s=wstep, now=now, fetch=fetch)
                if series:
                    sparks.append({"key": key, "series": series})
            prom_ok, prom_detail = True, f"{len(sparks)} series"
        except Exception as exc:
            sparks, prom_ok, prom_detail = [], False, f"{type(exc).__name__}"
        doc["sparks"] = sparks
        doc["sources"].append({"name": "prometheus", "url": PROM_URL,
                               "ok": prom_ok, "detail": prom_detail})
    else:
        doc["sparks"] = sparks
        doc["sources"].append({"name": "prometheus", "url": PROM_URL,
                               "ok": bool(sparks),
                               "detail": f"cached, {len(sparks)} series"})
    if not doc["sparks"]:
        doc["sparks_note"] = ("no history source: sparklines need Prometheus on the "
                              "monitoring host -- the current values above are still live")
    return doc


def _age_label(age_s):
    """Human age of the last load run's state file."""
    try:
        age = float(age_s)
    except (TypeError, ValueError):
        return None
    if age < 90:
        return f"{int(age)}s ago"
    if age < 5400:
        return f"{age / 60:.0f} min ago"
    if age < 172800:
        return f"{age / 3600:.1f} h ago"
    return f"{age / 86400:.1f} d ago"


_cache = {"at": {}, "sparks": {}}


def cached_sparks(now=None, fetch=None, window=None):
    """Sparklines, cached briefly per window: a 5 s page refresh must not mean 5 s x 4
    queries, and switching windows must not serve the other window's series."""
    now = time.time() if now is None else now
    wname, wsecs, wstep = window_spec(window)
    if _cache["sparks"].get(wname) and (now - _cache["at"].get(wname, 0.0)) < SPARK_CACHE_S:
        return _cache["sparks"][wname]
    sparks = []
    for key, _label, _unit, metric in SPARKS:
        try:
            series = prom_range(metric, window_s=wsecs, step_s=wstep, now=now, fetch=fetch)
        except Exception:
            return _cache["sparks"].get(wname) or []
        if series:
            sparks.append({"key": key, "series": series})
    if sparks:
        _cache["at"][wname], _cache["sparks"][wname] = now, sparks
    return sparks or _cache["sparks"].get(wname) or []


def live_doc(now=None, fetch=None, window=None, job=None, presets=None):
    """What the route calls: local values always, history from the cache."""
    return build_live(now=now, fetch=fetch, window=window, job=job, presets=presets,
                      sparks=cached_sparks(now=now, fetch=fetch, window=window))
