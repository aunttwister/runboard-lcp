"""live_metrics: the grafana-replacement layer, including both failure paths.

The rule this file exists to protect is that neither source may be able to lie. A dead
exporter must produce dashes (not zeros), a dead Prometheus must empty only the history,
and a metric the engine cannot expose must be reported as unavailable rather than
approximated. Every test here fakes the HTTP boundary, so no test resolves a hostname,
opens a socket, or reaches production.
"""
from __future__ import annotations

import json

import pytest

import live_metrics as LM


@pytest.fixture(autouse=True)
def clean_cache():
    """Module-level spark cache would otherwise leak between tests."""
    LM._cache.update({"at": {}, "sparks": {}})
    yield
    LM._cache.update({"at": {}, "sparks": {}})


class _Resp:
    """Minimal urlopen context manager."""

    def __init__(self, data):
        self._data = data

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


# ------------------------------------------------------------------ the text parser

def test_parse_reads_plain_and_labelled_series_and_skips_noise():
    text = (
        "# HELP zgx_serving_up 1 when the port answers\n"
        "# TYPE zgx_serving_up gauge\n"
        "zgx_serving_up 1\n"
        "zgx_gpu_power_watts 10.91\n"
        'zgx_load_phase_aggregate_tok_s{phase="c4",concurrency="4"} 57.74\n'
        "zgx_gpu_temp_celsius nan\n"
        "zgx_gpu_other +Inf\n"
        "zgx_odd notanumber\n"
        "1234 a line that cannot even start a metric name\n"
        "\n"
    )
    plain, labelled = LM.parse_prom_text(text)
    assert plain == {"zgx_serving_up": 1.0, "zgx_gpu_power_watts": 10.91}
    assert labelled == {("zgx_load_phase_aggregate_tok_s",
                         '{phase="c4",concurrency="4"}'): 57.74}
    # NaN/Inf are dropped, never smuggled through as a float that renders as "nan"
    assert "zgx_gpu_temp_celsius" not in plain and "zgx_gpu_other" not in plain


@pytest.mark.parametrize("text", ["", None, "# only a comment\n", "   \n"])
def test_parse_of_nothing_is_empty(text):
    assert LM.parse_prom_text(text) == ({}, {})


# ------------------------------------------------------------------ transport

def test_http_get_reads_bytes_without_a_socket(monkeypatch):
    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["url"] = getattr(req, "full_url", req)
        seen["timeout"] = timeout
        return _Resp(b"payload")

    monkeypatch.setattr(LM.urllib.request, "urlopen", fake_urlopen)
    assert LM.http_get("http://example/metrics") == b"payload"
    assert seen["url"] == "http://example/metrics"
    assert seen["timeout"] == LM.TIMEOUT


# ------------------------------------------------------------------ prometheus range

def _range_body(points, name="zgx_gpu_power_watts"):
    return json.dumps({"data": {"result": [{"metric": {"__name__": name},
                                            "values": points}]}}).encode()


def test_prom_range_maps_points_and_drops_unusable_values():
    body = json.dumps({"data": {"result": [
        {"values": [[100, "1.5"], [160, "nan"], [220, "2.5"], [280, "oops"]]},
    ]}}).encode()
    out = LM.prom_range("zgx_x", now=1000, fetch=lambda u: body)
    assert out == [{"t": 100, "v": 1.5}, {"t": 220, "v": 2.5}]


def test_prom_range_of_an_empty_result_is_empty():
    body = json.dumps({"data": {"result": []}}).encode()
    assert LM.prom_range("zgx_x", now=1000, fetch=lambda u: body) == []


def test_prom_range_tolerates_a_payload_with_no_data_key():
    assert LM.prom_range("zgx_x", now=1000, fetch=lambda u: b"{}") == []


def test_prom_range_builds_a_windowed_query(monkeypatch):
    seen = {}

    def fake_http_get(url, timeout=LM.TIMEOUT):
        seen["url"] = url
        return b'{"data":{"result":[]}}'

    monkeypatch.setattr(LM, "http_get", fake_http_get)
    assert LM.prom_range("zgx_gpu_power_watts", now=10_000) == []
    assert "/api/v1/query_range?" in seen["url"]
    assert "start=8200" in seen["url"] and "end=10000" in seen["url"]
    assert "step=60" in seen["url"]


# ------------------------------------------------------------------ formatting

@pytest.mark.parametrize("value,fmt,expected", [
    (1.0, "int", 1),
    (10.94, "num1", 10.9),
    (24.146, "num2", 24.15),
    (24116695024.0, "gb", 22.5),
    ("text", None, "text"),
])
def test_fmt_val_renders_each_declared_format(value, fmt, expected):
    assert LM._fmt_val(value, fmt) == expected


def test_fmt_val_of_a_missing_or_unparseable_value_is_none():
    assert LM._fmt_val(None, "int") is None
    assert LM._fmt_val("not-a-number", "int") is None


def test_items_leave_unsampled_metrics_as_none_not_zero():
    items = LM._items(LM.MACHINE, {"zgx_serving_up": 1})
    by_key = {i["key"]: i["value"] for i in items}
    assert by_key["serving_up"] == 1
    assert all(v is None for k, v in by_key.items() if k != "serving_up")


# ------------------------------------------------------------------ age labels

@pytest.mark.parametrize("age,expected", [
    (10, "10s ago"), (600, "10 min ago"), (18_000, "5.0 h ago"), (259_200, "3.0 d ago"),
])
def test_age_label_buckets(age, expected):
    assert LM._age_label(age) == expected


def test_age_label_of_a_non_number_is_none():
    assert LM._age_label("soon") is None


# ------------------------------------------------------------------ build_live

EXPORTER_TEXT = (
    "zgx_serving_up 1\n"
    "zgx_gpu_utilization_percent 0\n"
    "zgx_gpu_power_watts 10.91\n"
    "zgx_memory_available_bytes 24116695024\n"
    "zgx_load_running 0\n"
    "zgx_load_state_age_seconds 34918.9\n"
    "zgx_load_requests_total 557\n"
).encode()


def _fetch(exporter=EXPORTER_TEXT, prom_body=None, prom_error=False):
    """Route fake fetches by URL so one callable serves both sources."""
    def f(url, timeout=None):
        if "9400" in url or "exporter" in url:
            if isinstance(exporter, Exception):
                raise exporter
            return exporter
        if prom_error:
            raise OSError("prometheus down")
        return prom_body if prom_body is not None else _range_body([[100, "1.0"]])
    return f


def test_build_live_reports_current_values_and_labels_a_finished_run():
    doc = LM.build_live(now=10_000, fetch=_fetch())
    src = {s["name"]: s for s in doc["sources"]}
    assert src["exporter"]["ok"] is True and src["prometheus"]["ok"] is True
    machine = {m["key"]: m["value"] for m in doc["machine"]}
    assert machine["serving_up"] == 1
    assert machine["gpu_power"] == 10.9
    assert machine["mem_avail"] == 22.5
    assert doc["run"]["running"] == 0
    assert doc["run"]["age_label"] == "9.7 h ago"
    assert "last run's final values" in doc["run"]["note"]
    assert [s["key"] for s in doc["sparks"]] == [k for k, *_ in LM.SPARKS]
    assert "sparks_note" not in doc


def test_build_live_does_not_call_a_running_run_stale():
    text = EXPORTER_TEXT.replace(b"zgx_load_running 0", b"zgx_load_running 1")
    doc = LM.build_live(now=10_000, fetch=_fetch(exporter=text))
    assert doc["run"]["running"] == 1
    assert "note" not in doc["run"]


def test_build_live_notes_a_recently_finished_run_without_claiming_an_age():
    text = EXPORTER_TEXT.replace(b"zgx_load_state_age_seconds 34918.9",
                                 b"zgx_load_state_age_seconds 30")
    doc = LM.build_live(now=10_000, fetch=_fetch(exporter=text))
    assert doc["run"]["age_label"] == "30s ago"
    assert doc["run"]["note"] == "no load run active"


def test_build_live_without_a_state_file_has_no_age_at_all():
    text = b"zgx_serving_up 1\n"
    doc = LM.build_live(now=10_000, fetch=_fetch(exporter=text))
    assert doc["run"]["state_age_s"] is None
    assert doc["run"]["age_label" if "age_label" in doc["run"] else "note"] == "no load run active"


def test_a_dead_exporter_dashes_the_values_but_keeps_history():
    doc = LM.build_live(now=10_000, fetch=_fetch(exporter=OSError("refused")))
    src = {s["name"]: s for s in doc["sources"]}
    assert src["exporter"]["ok"] is False and src["exporter"]["detail"] == "OSError"
    assert all(m["value"] is None for m in doc["machine"])
    assert src["prometheus"]["ok"] is True
    assert doc["sparks"], "history must survive a dead exporter"


def test_a_dead_prometheus_empties_the_history_but_keeps_the_values():
    doc = LM.build_live(now=10_000, fetch=_fetch(prom_error=True))
    src = {s["name"]: s for s in doc["sources"]}
    assert src["prometheus"]["ok"] is False and src["prometheus"]["detail"] == "OSError"
    assert doc["sparks"] == []
    assert "no history source" in doc["sparks_note"]
    assert {m["key"]: m["value"] for m in doc["machine"]}["gpu_power"] == 10.9


def test_build_live_with_cached_sparks_does_not_query_prometheus():
    def explode(url, timeout=None):
        if "exporter" in url or "9400" in url:
            return EXPORTER_TEXT
        raise AssertionError("prometheus must not be queried when sparks are passed in")

    cached = [{"key": "gpu_power", "series": [{"t": 1, "v": 9.0}]}]
    doc = LM.build_live(now=10_000, fetch=explode, sparks=cached)
    src = {s["name"]: s for s in doc["sources"]}
    assert doc["sparks"] == cached
    assert src["prometheus"]["ok"] is True and "cached" in src["prometheus"]["detail"]


def test_build_live_says_so_when_even_the_cache_is_empty():
    doc = LM.build_live(now=10_000, fetch=_fetch(), sparks=[])
    src = {s["name"]: s for s in doc["sources"]}
    assert src["prometheus"]["ok"] is False
    assert "no history source" in doc["sparks_note"]


def test_the_unavailable_list_names_what_the_engine_cannot_expose():
    doc = LM.build_live(now=10_000, fetch=_fetch())
    assert "TTFT (p50/p90)" in doc["unavailable"]
    assert any("KV cache" in u for u in doc["unavailable"])


# ------------------------------------------------------------------ caching

def test_cached_sparks_fetches_then_serves_from_cache(monkeypatch):
    calls = []

    def fake_range(query, window_s=None, step_s=None, now=None, fetch=None):
        calls.append(query)
        return [{"t": 1, "v": 1.0}]

    monkeypatch.setattr(LM, "prom_range", fake_range)
    first = LM.cached_sparks(now=1000)
    assert len(first) == len(LM.SPARKS) and len(calls) == len(LM.SPARKS)
    second = LM.cached_sparks(now=1001)          # inside SPARK_CACHE_S
    assert second == first
    assert len(calls) == len(LM.SPARKS), "second call must be a cache hit"


def test_cached_sparks_refreshes_once_the_cache_expires(monkeypatch):
    calls = []
    monkeypatch.setattr(LM, "prom_range",
                        lambda q, **kw: (calls.append(q), [{"t": 1, "v": 2.0}])[1])
    LM.cached_sparks(now=1000)
    LM.cached_sparks(now=1000 + LM.SPARK_CACHE_S + 1)
    assert len(calls) == 2 * len(LM.SPARKS)


def test_cached_sparks_returns_the_stale_cache_when_prometheus_dies(monkeypatch):
    monkeypatch.setattr(LM, "prom_range", lambda q, **kw: [{"t": 1, "v": 3.0}])
    good = LM.cached_sparks(now=1000)
    assert good

    def boom(q, **kw):
        raise OSError("down")

    monkeypatch.setattr(LM, "prom_range", boom)
    assert LM.cached_sparks(now=9999) == good, "a dead source must not clear real history"


def test_cached_sparks_with_nothing_cached_stays_empty(monkeypatch):
    def boom(q, **kw):
        raise OSError("down")

    monkeypatch.setattr(LM, "prom_range", boom)
    assert LM.cached_sparks(now=1000) == []


def test_cached_sparks_drops_a_metric_with_no_points(monkeypatch):
    monkeypatch.setattr(LM, "prom_range",
                        lambda q, **kw: [] if q.endswith("power_watts") else [{"t": 1, "v": 1}])
    out = LM.cached_sparks(now=1000)
    assert all(s["key"] != "gpu_power" for s in out)
    assert out


# ------------------------------------------------------------------ live_doc

def test_live_doc_combines_live_values_with_cached_history(monkeypatch):
    monkeypatch.setattr(LM, "cached_sparks", lambda now=None, fetch=None, window=None: [])
    doc = LM.live_doc(now=10_000, fetch=_fetch())
    assert doc["schema"] == "zgx.console.live.v1"
    assert doc["fetched_utc"].endswith("Z")
    assert {m["key"]: m["value"] for m in doc["machine"]}["serving_up"] == 1


# ------------------------------------------------------------------ the window control

@pytest.mark.parametrize("name,expect", [
    ("30m", ("30m", 1800, 60)),
    ("6h", ("6h", 21600, 120)),
    ("24h", ("24h", 86400, 480)),
])
def test_window_spec_scales_the_step_to_keep_the_point_count_roughly_constant(name, expect):
    assert LM.window_spec(name) == expect


@pytest.mark.parametrize("name", ["nonsense", "", None, 7, ["6h"], "1800"])
def test_window_spec_falls_back_instead_of_raising(name):
    """An unknown window must not be able to 502 the page: it falls back to the default."""
    assert LM.window_spec(name) == ("30m", 1800, 60)


def test_build_live_reports_the_window_it_used_and_queries_that_window(monkeypatch):
    seen = []

    def fake_range(query, window_s=None, step_s=None, now=None, fetch=None):
        seen.append((window_s, step_s))
        return [{"t": 1, "v": 2.0}]

    monkeypatch.setattr(LM, "prom_range", fake_range)
    doc = LM.build_live(fetch=lambda u: b"zgx_serving_up 1\n", window="6h")

    assert (doc["window"], doc["window_s"], doc["step_s"]) == ("6h", 21600, 120)
    assert seen and all(pair == (21600, 120) for pair in seen)
    assert len(doc["sparks"]) == len(LM.SPARKS)


def test_cached_sparks_does_not_serve_one_window_for_another(monkeypatch):
    """A cached 30 min series must never answer a 6 h request -- the shapes differ."""
    calls = []

    def fake_range(query, window_s=None, step_s=None, now=None, fetch=None):
        calls.append(window_s)
        return [{"t": 1, "v": 1.0}]

    monkeypatch.setattr(LM, "prom_range", fake_range)
    LM.cached_sparks(now=1000.0, window="30m")
    LM.cached_sparks(now=1001.0, window="6h")          # well inside the 30 s cache TTL

    assert calls.count(1800) == len(LM.SPARKS)
    assert calls.count(21600) == len(LM.SPARKS)

    before = len(calls)
    LM.cached_sparks(now=1002.0, window="30m")         # the 30m entry is still cached
    assert len(calls) == before


def test_live_doc_passes_the_window_to_both_halves(monkeypatch):
    got = {}
    monkeypatch.setattr(LM, "cached_sparks",
                        lambda **kw: got.setdefault("sparks_kw", kw) and [] or [])
    monkeypatch.setattr(LM, "build_live", lambda **kw: got.update(kw) or {"ok": True})

    assert LM.live_doc(window="24h") == {"ok": True}
    assert got["window"] == "24h"
    assert got["sparks_kw"]["window"] == "24h"
