"""zgx_exporter: the Prometheus surface, and the honesty rules it publishes.

Two rules, both the fix for something that actually went wrong: a metric that cannot be
sampled is OMITTED rather than reported as 0 (a fake zero on a power or throughput graph is
indistinguishable from a real idle reading), and the throughput series comes from the shared
corrected definition so the dashboard and the live page cannot disagree.
"""
from __future__ import annotations

import builtins
import io
import json
import urllib.request

import pytest

import corrected_metrics
import zgx_exporter as X
from httpkit import request


def _run_stdout(stdout, rc=0):
    class P:
        returncode = rc
    P.stdout = stdout
    return P


# ---------------------------------------------------------------- gauge

def test_gauge_omits_a_metric_it_cannot_sample():
    assert X.gauge("zgx_thing", None) == []


def test_gauge_renders_a_bare_value():
    assert X.gauge("zgx_thing", 3) == ["zgx_thing 3"]


def test_gauge_renders_labels_and_help():
    lines = X.gauge("zgx_thing", 1.5, {"phase": "c1", "concurrency": "2"}, "what it is")
    assert lines == ["# HELP zgx_thing what it is", "# TYPE zgx_thing gauge",
                     'zgx_thing{phase="c1",concurrency="2"} 1.5']


# ---------------------------------------------------------------- sampling

def test_sample_gpu_reads_one_nvidia_smi_call(monkeypatch):
    seen = {}

    def fake_run(cmd, capture_output, text, timeout):
        seen["cmd"] = cmd
        return _run_stdout(" 21.31, 42, 78, 1530, 0x0000000000000001\n")

    monkeypatch.setattr(X.subprocess, "run", fake_run)
    assert X.sample_gpu() == {"power": 21.31, "temp": 42.0, "util": 78.0,
                              "clock": "1530", "throttle": "0x0000000000000001"}
    assert seen["cmd"][0] == "nvidia-smi" and X.GPU_QUERY in seen["cmd"][1]


def test_sample_gpu_returns_nothing_when_smi_prints_nothing(monkeypatch):
    monkeypatch.setattr(X.subprocess, "run", lambda *a, **k: _run_stdout("\n"))
    assert X.sample_gpu() == {}


def test_sample_gpu_returns_nothing_for_a_non_numeric_row(monkeypatch):
    monkeypatch.setattr(X.subprocess, "run", lambda *a, **k: _run_stdout("N/A, N/A, N/A\n"))
    assert X.sample_gpu() == {}


def test_sample_gpu_returns_nothing_when_smi_is_not_installed(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(X.subprocess, "run", boom)
    assert X.sample_gpu() == {}


MEMINFO = ("MemTotal:       131596288 kB\n"
           "MemAvailable:   104857600 kB\n"
           "SwapTotal:        8388608 kB\n"
           "SwapFree:         7340032 kB\n")


@pytest.fixture
def meminfo(monkeypatch):
    def fake_open(path, *a, **kw):
        if str(path) == "/proc/meminfo":
            return io.StringIO(MEMINFO)
        return real_open(path, *a, **kw)

    real_open = builtins.open
    monkeypatch.setattr(builtins, "open", fake_open)


def test_sample_mem_converts_kb_to_bytes(meminfo):
    got = X.sample_mem()
    assert got["mem_available"] == 104857600 * 1024
    assert got["swap_free"] == 7340032 * 1024
    assert got["swap_total"] == 8388608 * 1024
    assert "mem_total" not in got            # only the three the dashboard plots


def test_sample_mem_returns_what_it_could_read_when_proc_is_unavailable(monkeypatch):
    def boom(path, *a, **kw):
        raise FileNotFoundError("no /proc here")

    monkeypatch.setattr(builtins, "open", boom)
    assert X.sample_mem() == {}


def _patch_urlopen(monkeypatch, status=200, error=None):
    class R:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake(url, timeout=None):
        if error:
            raise error
        R.status = status
        return R()

    monkeypatch.setattr(urllib.request, "urlopen", fake)


def test_serving_up_is_one_for_an_answering_endpoint(monkeypatch):
    _patch_urlopen(monkeypatch, status=200)
    assert X.serving_up() == 1


def test_serving_up_is_zero_for_a_non_200(monkeypatch):
    _patch_urlopen(monkeypatch, status=503)
    assert X.serving_up() == 0


def test_serving_up_is_zero_when_the_endpoint_is_down(monkeypatch):
    _patch_urlopen(monkeypatch, error=ConnectionRefusedError("refused"))
    assert X.serving_up() == 0


# ---------------------------------------------------------------- load_metrics

def test_load_metrics_is_empty_until_a_soak_has_written_state(run_dir):
    assert X.load_metrics() == []


def _state(run_dir, **over):
    doc = {"live_agg_tps": 60.0, "live_stream_tps": 30.0,
           "totals": {"requests": 10, "errors": 1, "completion_tokens": 500,
                      "elapsed": 100.0},
           "phase_index": 0, "phase": "LOAD", "phases": []}
    doc.update(over)
    (run_dir / "state.json").write_text(json.dumps(doc))
    return doc


def test_load_metrics_publishes_the_corrected_throughput_and_the_phase(run_dir):
    _state(run_dir, phases=[
        {"name": "c1", "c": 1, "minutes": 5,
         "result": {"aggregate_tok_s": 60.0, "per_stream_mean": 55.0, "per_stream_p50": 50.0,
                    "requests": 20, "errors": 0}},
        {"name": "c2", "c": 2, "result": {}},
        {"name": "c3", "result": None},
        {"name": "c4", "c": 4, "result": {"aggregate_tok_s": 40.0, "per_stream_p50": 20.0}}])
    text = "\n".join(X.load_metrics())
    assert "zgx_load_live_aggregate_tok_s 60.0" in text
    assert "zgx_load_live_per_stream_tok_s 30.0" in text
    assert "zgx_load_requests_total 10" in text
    assert "zgx_load_errors_total 1" in text
    assert "zgx_load_completion_tokens_total 500" in text
    assert "zgx_load_elapsed_seconds_total 100.0" in text
    assert "zgx_load_phase_concurrency 1" in text
    assert "zgx_load_phase_minutes 5" in text
    assert "zgx_load_running 1" in text
    assert "zgx_load_state_age_seconds" in text
    assert "zgx_load_phase_aggregate_tok_s" in text
    # scaling derived from the row it has: aggregate wall-clock / per-stream p50
    assert 'zgx_load_phase_scaling_x{phase="c1",concurrency="1"} 1.2' in text
    assert 'zgx_load_phase_scaling_x{phase="c4",concurrency="4"} 2.0' in text
    # a row with no result and a row with an empty result publish nothing
    assert 'phase="c2"' not in text and 'phase="c3"' not in text


def test_load_metrics_says_the_run_is_finished_when_it_is(run_dir):
    _state(run_dir, phase="DONE", phase_index=None,
           phases=[{"name": "c1", "result": {"aggregate_tok_s": 1.0}}])
    text = "\n".join(X.load_metrics())
    assert "zgx_load_running 0" in text
    assert "zgx_load_phase_concurrency" not in text      # no current phase to report


def test_load_metrics_ignores_a_phase_index_out_of_range(run_dir):
    _state(run_dir, phase_index=7, phases=[{"name": "c1", "result": None}])
    assert "zgx_load_phase_minutes" not in "\n".join(X.load_metrics())


def test_load_metrics_never_publishes_a_scaling_it_could_not_derive(run_dir):
    _state(run_dir, phases=[{"name": "c1", "result": {"aggregate_tok_s": 40.0}}])
    assert "zgx_load_phase_scaling_x" not in "\n".join(X.load_metrics())


def test_load_metrics_omits_a_total_it_cannot_read(run_dir):
    _state(run_dir, totals={"requests": 0, "elapsed": None})
    text = "\n".join(X.load_metrics())
    assert "zgx_load_requests_total 0" in text            # a real zero is a real zero
    assert "zgx_load_elapsed_seconds_total" not in text   # an unreadable one is omitted


def test_load_metrics_survives_a_broken_correction_step(run_dir, monkeypatch):
    _state(run_dir)

    def boom(st):
        raise ValueError("bad rows")

    monkeypatch.setattr(X, "_correct", boom)
    assert any("zgx_load_running 1" in line for line in X.load_metrics())


# ---------------------------------------------------------------- collect + handler

def test_collect_renders_every_section(monkeypatch, run_dir):
    monkeypatch.setattr(X, "sample_gpu", lambda: {"power": 21.3, "temp": 42.0, "util": 78.0,
                                                  "clock": "1530", "throttle": "0x1"})
    monkeypatch.setattr(X, "sample_mem", lambda: {"mem_available": 1024, "swap_free": 512,
                                                  "swap_total": 2048})
    monkeypatch.setattr(X, "serving_up", lambda: 1)
    text = X.collect()
    assert text.endswith("\n")
    assert "zgx_gpu_power_watts 21.3" in text
    assert "zgx_gpu_temperature_celsius 42.0" in text
    assert "zgx_gpu_utilization_percent 78.0" in text
    assert "zgx_gpu_clock_mhz 1530.0" in text
    assert "zgx_memory_available_bytes 1024" in text
    assert "zgx_swap_free_bytes 512" in text and "zgx_swap_total_bytes 2048" in text
    assert "zgx_serving_up 1" in text


def test_collect_omits_everything_it_could_not_sample(monkeypatch, run_dir):
    monkeypatch.setattr(X, "sample_gpu", lambda: {})            # no GPU, no fake zeros
    monkeypatch.setattr(X, "sample_mem", lambda: {})
    monkeypatch.setattr(X, "serving_up", lambda: 0)
    text = X.collect()
    for metric in ("zgx_gpu_power_watts", "zgx_gpu_temperature_celsius",
                   "zgx_gpu_utilization_percent", "zgx_gpu_clock_mhz",
                   "zgx_memory_available_bytes", "zgx_swap_free_bytes",
                   "zgx_swap_total_bytes"):
        assert metric not in text
    assert "zgx_serving_up 0" in text


@pytest.fixture
def isolated(monkeypatch, run_dir):
    monkeypatch.setattr(X, "sample_gpu", lambda: {"power": 10.0, "temp": 30.0, "util": 5.0})
    monkeypatch.setattr(X, "sample_mem", lambda: {"mem_available": 4096})
    monkeypatch.setattr(X, "serving_up", lambda: 1)
    return run_dir


def test_metrics_route_serves_the_prometheus_text_format(isolated):
    status, headers, body = request(X.Handler, "GET", "/metrics")
    assert status == 200
    assert headers["content-type"] == "text/plain; version=0.0.4"
    assert b"zgx_gpu_power_watts 10.0" in body
    assert int(headers["content-length"]) == len(body)


def test_health_route(isolated):
    status, headers, body = request(X.Handler, "GET", "/health")
    assert status == 200 and json.loads(body) == {"ok": True}


def test_an_unknown_route_is_404(isolated):
    status, _, body = request(X.Handler, "GET", "/nope")
    assert status == 404 and body == b""


def test_a_scrape_that_loses_its_client_does_not_raise(isolated):
    status, _, body = request(X.Handler, "GET", "/metrics", fail_write_at=2)
    assert status == 200 and body == b""
