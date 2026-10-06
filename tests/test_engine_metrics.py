"""engine_metrics: the always-on engine-load block.

The rule this file protects is that the card must say something TRUE about the engine at every
moment -- during an eval, during agent traffic, while nothing at all is running, and when either
source is down. Three failure modes are pinned here because each one lies quietly:

  * a dead source must produce dashes, not zeros (a 0 tok/s card says "the box is idle" when the
    truth is "this process could not ask");
  * a rate with no traffic in its window must stay ``null`` (0.0 is a measurement, and a long
    generation in flight legitimately produces no FINISHED request to average over);
  * a series with gaps must not be shifted against its neighbours when the chart is drawn.

Every test here fakes the HTTP boundary, so nothing resolves a hostname or opens a socket.
"""
from __future__ import annotations

import json
import pathlib
import re
import urllib.parse

import pytest

import engine_metrics as EM

ENGINE_TEXT = (
    b"# HELP vllm:num_requests_running Requests running\n"
    b"# TYPE vllm:num_requests_running gauge\n"
    b'vllm:num_requests_running{model_name="qwen3.8-flash-next",engine="0"} 1.0\n'
    b'vllm:num_requests_waiting{model_name="qwen3.8-flash-next",engine="0"} 2.0\n'
    b'vllm:kv_cache_usage_perc{model_name="qwen3.8-flash-next",engine="0"} 0.103\n'
    b'vllm:generation_tokens_total{model_name="qwen3.8-flash-next",engine="0"} 4321.0\n'
    b"vllm:broken_series nan\n"
    b"a line that cannot start a metric\n"
)


@pytest.fixture(autouse=True)
def cold_cache():
    """The module memoises whole documents; a warm cache would leak between tests."""
    EM.reset_cache()
    yield
    EM.reset_cache()


def vector(value, t=1_700_000_000):
    if value is None:
        return json.dumps({"status": "success", "data": {"result": []}}).encode()
    return json.dumps({"status": "success", "data": {
        "result": [{"metric": {}, "value": [t, value]}]}}).encode()


def matrix(points):
    return json.dumps({"status": "success", "data": {
        "result": [{"metric": {}, "values": points}]}}).encode()


def route(engine=ENGINE_TEXT, engine_err=None, prom_err=None, values=None, calls=None,
          range_points=None, range_err=None):
    """A fake fetch routed by URL: the engine port, then Prometheus.

    ``values`` maps a metric-name substring to the string Prometheus would answer for the
    expressions that mention it; anything unmatched answers an empty result, which is how a
    window with no traffic looks. Every URL is recorded so a test can assert what was asked.
    """
    values = values or {}

    def f(url, timeout=None):
        if calls is not None:
            calls.append(url)
        if ":18300" in url:
            if engine_err is not None:
                raise engine_err
            return engine
        if prom_err is not None:
            raise prom_err
        query = urllib.parse.unquote(urllib.parse.parse_qs(
            urllib.parse.urlparse(url).query).get("query", [""])[0])
        if "query_range" in url:
            if range_err is not None:
                raise range_err
            return matrix(range_points if range_points is not None else [[1000, "1.5"]])
        for needle, value in values.items():
            if needle in query:
                return vector(value)
        return vector(None)

    return f


# ------------------------------------------------------------------ the engine's text format

def test_parse_engine_sums_labelled_series_and_reads_the_model_name():
    sums, model, seen = EM.parse_engine(ENGINE_TEXT.decode())
    assert model == "qwen3.8-flash-next"
    assert sums["vllm:num_requests_running"] == 1.0
    assert sums["vllm:kv_cache_usage_perc"] == 0.103
    assert sums["vllm:generation_tokens_total"] == 4321.0
    # NaN and a line that cannot start a metric are dropped, never smuggled through as floats
    assert "vllm:broken_series" not in sums
    assert seen == 4


def test_parse_engine_sums_every_series_of_a_metric():
    """Several engines/replicas behind one port is a box-level question, not a per-label one."""
    text = ('vllm:num_requests_running{model_name="a",engine="0"} 1\n'
            'vllm:num_requests_running{model_name="b",engine="1"} 2\n')
    sums, model, _seen = EM.parse_engine(text)
    assert sums["vllm:num_requests_running"] == 3.0
    assert model == "a"                     # first label seen wins; the card names one model


@pytest.mark.parametrize("text", ["", None, "# only a comment\n", "   \n"])
def test_parse_engine_of_nothing_is_empty(text):
    assert EM.parse_engine(text) == ({}, None, 0)


# ------------------------------------------------------------------ one instant value

def test_instant_returns_the_value_of_the_first_series():
    assert EM.instant("up", fetch=lambda u, timeout=None: vector("20.5")) == 20.5


@pytest.mark.parametrize("raw", ["NaN", "+Inf", "-Inf", "notanumber", None])
def test_instant_of_a_non_number_is_none(raw):
    """0/0 (a rate whose counters saw nothing) must render as a dash, not as 0 tok/s."""
    assert EM.instant("q", fetch=lambda u, timeout=None: vector(raw)) is None


def test_instant_of_an_empty_result_is_none():
    assert EM.instant("q", fetch=lambda u, timeout=None: vector(None)) is None


def test_instant_of_a_matrix_is_none():
    """An instant query answered with range data (the wrong shape) must not invent a value."""
    body = json.dumps({"data": {"result": [{"values": [[1, "5"]]}]}}).encode()
    assert EM.instant("q", fetch=lambda u, timeout=None: body) is None


def test_instant_asks_the_prometheus_api_for_the_query():
    seen = []
    EM.instant("sum(up)", fetch=lambda u, timeout=None: seen.append(u) or vector("1"))
    assert seen == [f"{EM.PROM_URL}/api/v1/query?query=sum%28up%29"]


# ------------------------------------------------------------------ the gapless grid

def test_on_grid_places_points_at_their_own_index_and_leaves_gaps_as_null():
    points = [{"t": 1000, "v": 1.0}, {"t": 1120, "v": 3.0}]
    assert EM.on_grid(points, start=1000, step=60, n=3) == [1.0, None, 3.0]


def test_on_grid_drops_points_outside_the_window():
    """A point from before the window must not be wrapped onto the end of the line."""
    points = [{"t": 940, "v": 9.0}, {"t": 9999, "v": 8.0}, {"t": 1060, "v": 2.0}]
    assert EM.on_grid(points, start=1000, step=60, n=2) == [None, 2.0]


def test_on_grid_of_nothing_is_all_gaps():
    assert EM.on_grid([], start=0, step=60, n=2) == [None, None]


# ------------------------------------------------------------------ formatting

@pytest.mark.parametrize("value,fmt,expected", [
    (1.4, "int", 1), (1.6, "int", 2), (1.25, "num1", 1.2), (1.234, "num2", 1.23),
    (0.103, "pct", 10.3), (7.0, "other", 7.0),
])
def test_fmt_val(value, fmt, expected):
    assert EM._fmt_val(value, fmt) == expected


@pytest.mark.parametrize("value,fmt", [(None, "num1"), ("soon", "num1"), (float("nan"), "num2")])
def test_fmt_val_never_invents_a_number(value, fmt):
    assert EM._fmt_val(value, fmt) is None


# ------------------------------------------------------------------ the block

def test_block_reads_the_gauges_from_the_engine_and_scales_the_kv_ratio():
    doc = EM.block(now=10_000, fetch=route())
    items = {i["key"]: i["value"] for i in doc["items"]}
    assert items["running"] == 1
    assert items["waiting"] == 2
    assert items["kv_pct"] == 10.3          # the engine publishes a 0..1 ratio
    assert doc["reachable"] is True and doc["model"] == "qwen3.8-flash-next"
    assert doc["ok"] is True
    src = {s["name"]: s for s in doc["sources"]}
    assert src["engine"]["ok"] is True and src["engine"]["url"] == EM.ENGINE_URL
    assert src["prometheus"]["ok"] is True


def test_block_reports_the_rates_prometheus_has_and_dashes_the_ones_it_does_not():
    """The live case: one long generation in flight, nothing finished, output tok/s moving."""
    fetch = route(values={
        "vllm:generation_tokens_total": "20.5",
        "time_to_first_token_seconds_bucket": "0.88",
    })
    items = {i["key"]: i["value"] for i in EM.block(now=10_000, fetch=fetch)["items"]}
    assert items["output_tps"] == 20.5
    assert items["ttft_p50"] == 0.88
    assert items["prefill_tps"] is None and items["decode_tps"] is None
    assert items["req_per_min"] is None


def test_block_states_that_only_the_gauges_are_sampled_when_prometheus_is_down():
    doc = EM.block(now=10_000, fetch=route(prom_err=OSError("refused")))
    items = {i["key"]: i["value"] for i in doc["items"]}
    assert items["running"] == 1 and items["output_tps"] is None
    src = {s["name"]: s for s in doc["sources"]}
    assert src["prometheus"]["ok"] is False and src["prometheus"]["detail"] == "OSError"
    assert "only the gauges above are sampled" in doc["note"]
    assert doc["ok"] is True, "the card is still useful: the engine answered"


def test_block_stops_asking_prometheus_after_the_first_failure():
    calls = []
    EM.block(now=10_000, fetch=route(prom_err=OSError("down"), calls=calls))
    prom_calls = [u for u in calls if ":9090" in u]
    assert len(prom_calls) == 1, "a dead Prometheus must not be retried once per metric"


def test_block_still_reports_the_rates_when_the_engine_port_is_unreachable():
    doc = EM.block(now=10_000, fetch=route(engine_err=OSError("refused"),
                                           values={"vllm:generation_tokens_total": "20.5"}))
    items = {i["key"]: i["value"] for i in doc["items"]}
    assert doc["reachable"] is False and doc["ok"] is True
    assert items["running"] is None and items["output_tps"] == 20.5
    assert "did not answer /metrics" in doc["note"]


def test_block_says_the_engine_publishes_no_metrics_when_exl3_serves_the_port():
    """The swappable-engine case: an empty body is 'nothing to read', not 'the box is idle'."""
    doc = EM.block(now=10_000, fetch=route(engine=b""))
    assert doc["reachable"] is False and doc["no_metrics"] is True
    assert {i["key"]: i["value"] for i in doc["items"]}["running"] is None
    assert "published no metric series" in doc["note"]
    assert doc["ok"] is True


def test_block_with_both_sources_down_is_not_ok_but_does_not_raise():
    doc = EM.block(now=10_000, fetch=route(engine_err=OSError("x"), prom_err=OSError("y")))
    assert doc["ok"] is False
    assert all(i["value"] is None for i in doc["items"])
    assert doc["series"] == []


def test_block_records_who_keeps_the_history():
    """The card claims the numbers are RECORDED, so it has to name the recorder."""
    doc = EM.block(now=10_000, fetch=route())
    assert EM.PROM_JOB in doc["recorded"] and EM.PROM_RETENTION in doc["recorded"]
    assert doc["cache_s"] == EM.CACHE_S
    assert doc["fetched_utc"].endswith("Z")


def test_block_is_memoised_and_per_window():
    calls = []
    fetch = route(calls=calls)
    first = EM.block(now=10_000, fetch=fetch, window_s=1800, step_s=60)
    n_before = len(calls)
    assert EM.block(now=10_001, fetch=fetch, window_s=1800, step_s=60)["at"] == first["at"]
    assert len(calls) == n_before, "a 3 s page poll must not re-query"

    EM.block(now=10_002, fetch=fetch, window_s=21600, step_s=120)
    assert len(calls) > n_before, "a 6 h request must not be answered from the 30 m document"


def test_block_cache_can_be_bypassed():
    calls = []
    fetch = route(calls=calls)
    EM.block(now=10_000, fetch=fetch)
    n = len(calls)
    EM.block(now=10_001, fetch=fetch, use_cache=False)
    assert len(calls) > n


# ------------------------------------------------------------------ the chart series

def test_series_projects_prometheus_points_onto_the_drawn_grid():
    pts = [[1000, "1.5"], [1060, "2.5"], [1180, "4.5"]]
    out = EM.series(now=1180, fetch=route(range_points=pts), window_s=180, step_s=60)
    keys = [s["key"] for s in out]
    assert keys == [k for k, _ in EM.SERIES]
    first = out[0]
    assert first["start"] == 1000 and first["step"] == 60
    assert first["values"] == [1.5, 2.5, None, 4.5]


def test_prefill_throughput_counts_only_the_tokens_the_gpu_actually_computed():
    """Pinned because the other form looks plausible and is wrong by 15x on this box.

    ``request_prompt_tokens_sum`` includes tokens served from the prefix cache; at the measured
    97.5 % hit rate that reported a 16,292 tok/s prefill where the GPU computed 1,075 tok/s. A
    cache-heavy agent workload would read as a faster prefill than a cold one.
    """
    assert "request_prefill_kv_computed_tokens_sum" in EM.EXPR["prefill_tps"]
    assert "request_prompt_tokens_sum" not in EM.EXPR["prefill_tps"]


def test_tok_per_stream_is_the_output_rate_over_requests_running():
    """The number a reader asking "how fast does it generate" means: one client's share of the
    engine. With a single stream it must equal the engine-wide rate, and with nothing running it
    must be absent rather than 0 -- 0 tok/s per stream claims a measured idle stream."""
    one = EM.block(now=1_000, fetch=route(values={"vllm:num_requests_running": "1"}), use_cache=False)
    items = {i["key"]: i for i in one["items"]}
    assert set(items) == set(EM.ITEM_KEYS), "the published key set is the contract"
    assert items["stream_tps"]["value"] == items["output_tps"]["value"]

    idle = EM.block(now=1_000, fetch=route(values={"vllm:num_requests_running": "0"}), use_cache=False)
    assert {i["key"]: i for i in idle["items"]}["stream_tps"]["value"] is None

    gone = EM.block(now=1_000, fetch=route(values={"vllm:num_requests_running": "0",
                                                   "rate(vllm:generation_tokens_total": "None"}),
                   use_cache=False)
    assert {i["key"]: i for i in gone["items"]}["stream_tps"]["value"] is None


def test_prom_range_drops_values_that_are_not_numbers():
    """A NaN or a text value in the middle of a series must leave a gap, not a break in the line."""
    pts = [[1000, "1.5"], [1060, "nan"], [1120, "oops"], [1180, "2.5"]]
    out = EM.prom_range("q", window_s=180, step_s=60, now=1180, fetch=route(range_points=pts))
    assert out == [{"t": 1000, "v": 1.5}, {"t": 1180, "v": 2.5}]


def test_series_drops_a_window_with_no_data_at_all():
    """No traffic in the window is an empty chart, not a flat line at zero."""
    assert EM.series(now=1180, fetch=route(range_points=[]), window_s=180, step_s=60) == []


def test_series_keeps_the_last_real_chart_when_prometheus_dies():
    good = EM.series(now=1000, fetch=route(range_points=[[940, "3.0"], [1000, "4.0"]]),
                     window_s=60, step_s=60)
    assert good
    EM.reset_cache()          # keep the cached series, drop the documents
    EM._cache["series"]["60:60"] = good
    EM._cache["series_at"]["60:60"] = 990
    stale = EM.series(now=1200, fetch=route(range_err=OSError("down")), window_s=60, step_s=60)
    assert stale == good, "a dead source must not blank a chart that is still true of its window"


def test_series_is_memoised_per_window(monkeypatch):
    calls = []

    def fake_range(query, **kw):
        calls.append(kw.get("window_s"))
        return [{"t": 1000, "v": 2.0}]

    monkeypatch.setattr(EM, "prom_range", fake_range)
    EM.series(now=1000, window_s=1800, step_s=60)
    assert calls.count(1800) == len(EM.SERIES)
    EM.series(now=1001, window_s=1800, step_s=60)
    assert calls.count(1800) == len(EM.SERIES), "the second read must be a cache hit"
    EM.series(now=1002, window_s=21600, step_s=120)
    assert calls.count(21600) == len(EM.SERIES)


# ------------------------------------------------------------------ the page's side of it
#
# The suite reads the pages as text, which cannot catch a runtime error -- so the two things
# that CAN be checked statically are checked here: every id the card's script touches has to
# exist in the markup, and every number the module publishes has to be one the card renders.
# A renamed key on either side of that boundary renders a silent dash, and nothing else in this
# file would notice.

PAGE = (pathlib.Path(__file__).resolve().parents[1] / "static" / "index.html").read_text(
    encoding="utf-8")
PUT_RE = re.compile(r'put\("([\w-]+)",\s*"([\w-]+)"\)')


def test_the_card_renders_every_number_the_module_publishes():
    pairs = PUT_RE.findall(PAGE)
    assert pairs, "the engine card has no KPI wiring at all"
    assert {key for _id, key in pairs} == set(EM.ITEM_KEYS)


def test_every_engine_card_id_exists_in_the_markup():
    ids = set(re.findall(r'id="([^"]+)"', PAGE))
    referenced = ({pid for pid, _key in PUT_RE.findall(PAGE)}
                  | {"c-eng", "c-eng-pre", "eng-src", "eng-note"})
    missing = referenced - ids
    assert missing == set(), f"the card's script reads ids that do not exist: {sorted(missing)}"


def test_the_card_charts_every_published_series():
    for key, _label in EM.SERIES:
        assert f"byKey.{key}" in PAGE, f"series {key} is published but never drawn"


def test_an_isolated_sample_is_drawn_as_a_dot_not_an_invisible_segment():
    """Prefill arrives in bursts: one interval out of 31 can carry traffic and the rest be gaps.

    A polyline through one point draws nothing, so the pane rendered axes with no trace -- the
    chart looked broken rather than sparse (seen on the deployed card 2026-09-28, then fixed by
    dotting every sample that has no neighbour). The shared helper draws all the card's charts, so
    this is asserted once, on the helper.
    """
    # up to the NEXT top-level function, not to the next closing brace (the helper is nested)
    helper = PAGE.split("function line(")[1].split("\nfunction ")[0]
    assert "fit(cv)" in helper and "ctx.stroke()" in helper  # sanity: we are looking at the helper
    assert "neighbour" in helper
    assert helper.count("ctx.arc(") >= 2, "isolated samples and the last sample must both dot"


def test_prefill_is_charted_on_its_own_axis():
    """Prefill runs 100-1000x the generation rate, so a shared axis hides the output trace.

    Measured on the deployed card 2026-09-28: the axis scaled to 1,382 tok/s (a prefill burst)
    and the 20 tok/s output line sat pinned along the floor, unreadable. Two canvases, or the
    one number the operator added this card FOR is invisible.
    """
    assert 'id="c-eng-pre"' in PAGE
    # Every call site on each canvas: the card has a no-data fallback that draws an empty
    # generation chart, so "the first line() mentioning c-eng" would be the wrong one.
    gen_calls = [c.split(");")[0] for c in PAGE.split('line($("c-eng"),')[1:]]
    pre_calls = [c.split(");")[0] for c in PAGE.split('line($("c-eng-pre"),')[1:]]
    drawn = [c for c in gen_calls if "output_tps" in c]
    assert drawn and "decode_tps" in drawn[0]
    assert all("prefill_tps" not in c for c in gen_calls), \
        "prefill is back on the generation axis, where it flattens the output line"
    assert pre_calls and "prefill_tps" in pre_calls[0]


def test_the_engine_card_is_rendered_from_the_live_document_alone():
    """It is drawn BEFORE /api/state is required to parse.

    Pinned as the exact call site rather than as "the string appears somewhere before ...":
    the weaker form stays green with the call wrapped in `if (false)`, which is precisely the
    defect -- a card that exists, is drawn nowhere, and renders as permanent dashes. A missing
    soak archive must not be able to hide the only always-on panel on the page.
    """
    body = " ".join(PAGE.split("async function poll()")[1].split())
    wiring = ("if (rl.status === \"fulfilled\" && rl.value.ok) { "
              "try { lastLive = await rl.value.json(); } "
              "catch (e) { /* keep the previous live doc */ } "
              "renderEngine(lastLive); renderBoxes(lastLive); }")
    assert wiring in body, "the engine card is not drawn unconditionally from /api/live"
    assert body.index(wiring) < body.index('throw new Error("state HTTP')


# ------------------------------------------------------------------ transport

def test_http_get_sets_a_user_agent_and_reads_bytes(monkeypatch):
    seen = {}

    class Resp:
        def read(self):
            return b"body"

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None):
        seen["url"] = getattr(req, "full_url", req)
        seen["ua"] = req.get_header("User-agent")
        seen["timeout"] = timeout
        return Resp()

    monkeypatch.setattr(EM.urllib.request, "urlopen", fake_urlopen)
    assert EM.http_get("http://x/metrics") == b"body"
    assert seen["url"] == "http://x/metrics"
    assert seen["ua"] == "zgx-engine-load/1"
    assert seen["timeout"] == EM.TIMEOUT


# ------------------------------------------------------------------ recency: counter, not rate
#
# The card must be able to answer "when did this box last DO anything" -- the question the board
# failed on 2026-09-28 while a 40-minute generation was in flight: every 2 m rate read 0/dash and
# the operator read that as "no data". Rates cannot answer it (a rate is 0, not absent, when the
# engine is idle); the counter's own steps can.

def _counter(values, t0=1_700_000_000, step=30):
    return [[t0 + i * step, str(v)] for i, v in enumerate(values)]


def test_recent_reports_the_total_served_and_the_age_of_the_last_step_that_served():
    now = 1_700_000_000.0
    pts = _counter([1000, 1400, 1400, 1400], t0=int(now) - 90)
    tokens, age = EM.recent(now=now, fetch=route(range_points=pts))
    assert tokens == 400, "the served total must be the sum of the counter's positive steps"
    assert age == 60, "the age must point at the step that produced the tokens, not at now"


def test_recent_of_a_counter_that_never_moved_is_a_measured_zero_and_no_age():
    now = 1_700_000_000.0
    pts = _counter([500, 500, 500], t0=int(now) - 60)
    assert EM.recent(now=now, fetch=route(range_points=pts)) == (0, None)


def test_recent_drops_a_counter_reset_instead_of_subtracting_the_restart():
    """A restart mid-window resets the counter; the tokens served before it still happened."""
    now = 1_700_000_000.0
    pts = _counter([5000, 600, 1000, 1500], t0=int(now) - 90)
    tokens, age = EM.recent(now=now, fetch=route(range_points=pts))
    assert tokens == 900, "the reset step must be dropped, not counted as negative"
    assert age == 0, "the token served in the newest step is the newest token there is"


def test_recent_of_a_single_sample_claims_nothing():
    now = 1_700_000_000.0
    assert EM.recent(now=now, fetch=route(range_points=_counter([900], t0=int(now)))) == (None, None)


def test_the_recency_items_are_published_and_dashed_when_prometheus_is_down():
    b = EM.block(fetch=route(prom_err=OSError("prometheus down")), use_cache=False)
    items = {i["key"]: i for i in b["items"]}
    assert set(items) == set(EM.ITEM_KEYS)
    assert items["gen_win"]["value"] is None and items["last_gen"]["value"] is None
    down = [s for s in b["sources"] if s["name"] == "prometheus"]
    assert down and down[0]["ok"] is False


def test_a_measured_zero_is_printed_as_zero_and_never_as_a_dash():
    """0 tokens with the counter readable is a measurement; a dash would read as "unreadable"."""
    assert EM._fmt_val(0, "count") == "0"
    assert EM._fmt_val(None, "count") is None
    assert EM._fmt_val(300, "ago") == "5 m ago"
    assert EM._fmt_val(12, "ago") == "12 s ago"
    assert EM._fmt_val(7200, "ago") == "2.0 h ago"


def test_a_recency_query_that_fails_degrades_only_the_two_recency_items():
    """The range query is one more request to the same source; its failure is not an exception
    the card may swallow -- the two items go to dashes and the source block says why."""
    b = EM.block(fetch=route(range_err=OSError("query_range failed")), use_cache=False)
    items = {i["key"]: i for i in b["items"]}
    assert items["gen_win"]["value"] is None and items["last_gen"]["value"] is None
    down = [s for s in b["sources"] if s["name"] == "prometheus"]
    assert len(down) == 1 and down[0]["ok"] is False
    assert down[0]["detail"] == "OSError"
    assert set(items) == set(EM.ITEM_KEYS)


# ------------------------------------------------------- what ONE stream actually gets

def _engine_text(gen, dec, cnt, running=1):
    """The engine's exposition text with the three counters the single-stream sampler needs."""
    return (f'vllm:generation_tokens_total{{model_name="m"}} {gen}\n'
            f'vllm:request_decode_time_seconds_sum{{model_name="m"}} {dec}\n'
            f'vllm:request_decode_time_seconds_count{{model_name="m"}} {cnt}\n'
            f'vllm:num_requests_running{{model_name="m"}} {running}\n').encode()


def test_note_single_stream_only_records_intervals_that_finished_exactly_one_request():
    EM.reset_cache()
    EM.note_single_stream({"vllm:generation_tokens_total": 0.0,
                           "vllm:request_decode_time_seconds_sum": 0.0,
                           "vllm:request_decode_time_seconds_count": 0.0})
    assert EM._ss["samples"] == []          # the first look primes the delta, it measures nothing
    EM.note_single_stream({"vllm:generation_tokens_total": 100.0,
                           "vllm:request_decode_time_seconds_sum": 2.0,
                           "vllm:request_decode_time_seconds_count": 1.0})
    assert EM._ss["samples"] == [50.0]      # exactly one finish: 100 tokens / 2 s of decode
    EM.note_single_stream({"vllm:generation_tokens_total": 200.0,
                           "vllm:request_decode_time_seconds_sum": 4.0,
                           "vllm:request_decode_time_seconds_count": 3.0})
    assert EM._ss["samples"] == [50.0]      # two finishes in one interval = shared, not one stream
    EM.note_single_stream({"vllm:generation_tokens_total": 300.0,
                           "vllm:request_decode_time_seconds_sum": 20.0,
                           "vllm:request_decode_time_seconds_count": 4.0})
    assert EM._ss["samples"] == [50.0, pytest.approx(100 / 16)]
    EM.note_single_stream({"vllm:generation_tokens_total": 5.0,
                           "vllm:request_decode_time_seconds_sum": 0.1,
                           "vllm:request_decode_time_seconds_count": 1.0})
    assert len(EM._ss["samples"]) == 2      # counters that went backwards = a restart, not a rate


def test_note_single_stream_drops_the_oldest_sample_past_the_ring():
    EM.reset_cache()
    EM.note_single_stream({"vllm:generation_tokens_total": 0.0,
                           "vllm:request_decode_time_seconds_sum": 0.0,
                           "vllm:request_decode_time_seconds_count": 0.0})
    for i in range(1, EM.SS_MAXLEN + 8):
        EM.note_single_stream({"vllm:generation_tokens_total": float(i * 100),
                               "vllm:request_decode_time_seconds_sum": float(i * 2),
                               "vllm:request_decode_time_seconds_count": float(i)})
    assert len(EM._ss["samples"]) == EM.SS_MAXLEN


def test_note_single_stream_skips_a_restart_that_lands_mid_interval():
    EM.reset_cache()
    EM.note_single_stream({"vllm:generation_tokens_total": 500.0,
                           "vllm:request_decode_time_seconds_sum": 10.0,
                           "vllm:request_decode_time_seconds_count": 0.0})
    EM.note_single_stream({"vllm:generation_tokens_total": 100.0,
                           "vllm:request_decode_time_seconds_sum": 2.0,
                           "vllm:request_decode_time_seconds_count": 1.0})
    assert EM._ss["samples"] == []          # one finish, but the counters went backwards


def test_single_stream_item_is_a_dash_until_one_interval_qualified():
    doc = EM.block(now=1_700_000_000, fetch=route(engine=_engine_text(0, 0, 0)), use_cache=False)
    item = {i["key"]: i for i in doc["items"]}["ss_decode_tps"]
    assert item["value"] is None            # nothing measured yet: a dash, never a 0
    doc = EM.block(now=1_700_000_030, fetch=route(engine=_engine_text(200, 4, 1)), use_cache=False)
    item = {i["key"]: i for i in doc["items"]}["ss_decode_tps"]
    assert item["value"] == pytest.approx(50.0)
    assert "1 interval" in item["basis"]    # the sample count is stated, so 50 is not over-read


# ------------------------------------------------------------------ metric-name families
#
# The engine changed families under the card once already (vllm:* -> tensorfold:*) and every
# KPI went quietly null while ~70 tok/s left the port. The family layer is pinned end to end:
# detection from the exposition's own keys, per-family expression translation, and a full
# block() against a TensorFold-shaped document.

TF_TEXT = (
    b"tensorfold:requests_running 2\n"
    b"tensorfold:requests_waiting 1\n"
    b"tensorfold:kv_cache_usage_ratio 0.31\n"
    b"tensorfold:generation_tokens_total 1938130\n"
    b"tensorfold_health:requests_total 1419\n"
    b"tensorfold_health:decode_seconds_total 53713.4\n"
)


def tf_route(values=None, calls=None, models=None, metrics=TF_TEXT):
    """A fetch routed like route(), plus the OpenAI model list on the same port."""

    def f(url, timeout=None):
        if calls is not None:
            calls.append(url)
        if url.endswith("/v1/models"):
            payload = models if models is not None else \
                {"object": "list", "data": [{"id": "GLM-5.3-Flash-EXL3"}]}
            return json.dumps(payload).encode()
        if "query" in url:
            query = urllib.parse.unquote(urllib.parse.parse_qs(
                urllib.parse.urlparse(url).query).get("query", [""])[0])
            for needle, value in (values or {}).items():
                if needle in query:
                    return vector(value)
            return vector(None)
        return metrics

    return f


def test_detect_family_reads_the_expositions_own_keys():
    assert EM.detect_family({"tensorfold:requests_running": 1}) == "tensorfold"
    assert EM.detect_family({"vllm:num_requests_running": 1}) == "vllm"
    assert EM.detect_family({}) is None
    assert EM.detect_family(None) is None


def test_expr_for_translates_the_family_and_honours_expression_overrides():
    tf = "tensorfold"
    assert EM.expr_for("output_tps", tf) == "sum(rate(tensorfold:generation_tokens_total[2m]))"
    assert "tensorfold_health:requests_total" in EM.expr_for("req_per_min", tf)
    # prefill cannot be a rename: TF's prompt counter includes cached tokens, so the
    # KV-computed form must be computed by subtraction
    assert "tensorfold:prompt_tokens_total" in EM.expr_for("prefill_tps", tf)
    assert "tensorfold_health:cached_tokens_total" in EM.expr_for("prefill_tps", tf)
    # an unknown family keeps the canonical names -> honest dashes, never a wrong series
    assert EM.expr_for("output_tps", None) == EM.EXPR["output_tps"]


def test_block_reads_a_tensorfold_engine_and_names_the_model_from_the_api():
    """The live failure, reproduced: a bare exposition and a live model behind it."""
    fetch = tf_route(values={"tensorfold:generation_tokens_total[2m]": "71.2"})
    doc = EM.block(now=10_000, fetch=fetch, use_cache=False)
    items = {i["key"]: i["value"] for i in doc["items"]}
    assert items["running"] == 2 and items["waiting"] == 1
    assert items["kv_pct"] == 31.0
    assert items["output_tps"] == 71.2
    assert doc["model"] == "GLM-5.3-Flash-EXL3", \
        "a bare exposition must still name the served model, via the OpenAI API"
    assert "tensorfold names" in doc["sources"][0]["detail"]


def test_tensorfold_basis_travels_with_the_new_definitions():
    doc = EM.block(now=10_000, fetch=tf_route(), use_cache=False)
    basis = {i["key"]: i["basis"] for i in doc["items"]}
    assert "prefix-cached" in basis["prefill_tps"]
    assert basis["prefix_hit"] == "cached/prompt tokens, 30m"


def test_the_generation_counter_is_queried_in_the_familys_names():
    seen = []
    EM.recent(now=10_000, fetch=tf_route(calls=seen), family="tensorfold")
    queries = [urllib.parse.unquote(urllib.parse.parse_qs(
        urllib.parse.urlparse(u).query).get("query", [""])[0]) for u in seen]
    assert any("sum(tensorfold:generation_tokens_total)" in q for q in queries), \
        "recency must read the family's own counter, not a vllm: name nothing publishes"


def test_openai_model_probe_degrades_to_none_when_the_api_does_not_answer():
    def boom(url, timeout=None):
        raise OSError("refused")

    assert EM._openai_model(boom) is None
