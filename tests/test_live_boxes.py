"""live_metrics: the per-box (per-Spark) block.

The board reads EVERY box's own exporter, so a TP=2 pair can never be reported as though it
were one machine. The rule this file protects is the one the rest of live_metrics follows: a
box that did not answer shows a DASH, never a zero -- an unreachable Spark must not look like
a cold one. Every test here fakes the HTTP boundary, so none resolves a hostname or opens a
socket.
"""
from __future__ import annotations

import pytest

import engine_metrics as EM
import live_metrics as LM


@pytest.fixture(autouse=True)
def clean_cache():
    """The engine cache is keyed by window; a leaked document would answer the next test."""
    EM.reset_cache()
    yield
    EM.reset_cache()


HEAD_TEXT = (b"zgx_serving_up 1\n"
             b"zgx_gpu_utilization_percent 95\n"
             b"zgx_gpu_temperature_celsius 71\n"
             b"zgx_gpu_power_watts 58.2\n")

WORKER_TEXT = (b"zgx_serving_up 0\n"
               b"zgx_gpu_utilization_percent 96\n"
               b"zgx_gpu_temperature_celsius 75\n"
               b"zgx_gpu_power_watts 59.1\n")


def _fetch(boxes=(HEAD_TEXT, WORKER_TEXT)):
    """Route by the box's address so one callable serves both boxes and the engine probe."""
    def f(url, timeout=None):
        val = boxes[1] if "108" in url else boxes[0]
        if isinstance(val, Exception):
            raise val
        return val
    return f


# ------------------------------------------------------------------ the spec

def test_boxes_spec_defaults_to_the_pair():
    spec = LM.boxes_spec()
    assert [b[0] for b in spec] == ["head", "worker"]
    assert spec[0][2] == "192.168.1.107" and spec[1][2] == "192.168.1.108"
    assert spec[1][3].endswith(":9400/metrics")


def test_boxes_spec_reads_the_env_override(monkeypatch):
    monkeypatch.setenv("RUNBOARD_BOXES",
                       "a|Alpha|10.0.0.1|http://10.0.0.1:9400/metrics|solo")
    assert LM.boxes_spec() == [("a", "Alpha", "10.0.0.1",
                                "http://10.0.0.1:9400/metrics", "solo")]


def test_boxes_spec_skips_a_malformed_entry_but_keeps_the_good_one(monkeypatch):
    monkeypatch.setenv("RUNBOARD_BOXES",
                       "broken|only-two;x|X|1.1.1.1|http://x:9400/metrics|r")
    assert [b[0] for b in LM.boxes_spec()] == ["x"]


@pytest.mark.parametrize("raw", ["   ", "nope|only|three", "| | || "])
def test_boxes_spec_falls_back_to_the_default_when_nothing_parses(monkeypatch, raw):
    monkeypatch.setenv("RUNBOARD_BOXES", raw)
    assert [b[0] for b in LM.boxes_spec()] == ["head", "worker"]


# ------------------------------------------------------------------ the document

def test_build_live_reports_every_box_and_keeps_machine_as_the_head():
    doc = LM.build_live(now=10_000, fetch=_fetch(), sparks=[])
    assert [b["key"] for b in doc["boxes"]] == ["head", "worker"]
    head, worker = doc["boxes"]
    assert head["ok"] and worker["ok"]
    assert head["label"] == "DGX Spark" and worker["addr"] == "192.168.1.108"
    assert head["role"].endswith("head") and worker["role"].endswith("worker")
    assert {i["key"]: i["value"] for i in head["items"]}["gpu_temp"] == 71
    assert {i["key"]: i["value"] for i in worker["items"]}["gpu_temp"] == 75
    # `machine` stays the HEAD box's items, so a reader that only knows the one-box document
    # is unaffected -- and one that wants the pair reads `boxes`.
    assert doc["machine"] == head["items"]
    names = [s["name"] for s in doc["sources"]]
    assert "exporter" in names and "exporter:worker" in names


def test_a_box_that_does_not_answer_shows_dashes_not_zeros():
    doc = LM.build_live(now=10_000, fetch=_fetch(boxes=(HEAD_TEXT, OSError("refused"))),
                        sparks=[])
    worker = doc["boxes"][1]
    assert worker["ok"] is False and worker["detail"] == "OSError"
    assert all(i["value"] is None for i in worker["items"])
    # the head is unaffected: one dead box must not blank the other
    assert doc["boxes"][0]["ok"] is True
    assert {s["name"]: s for s in doc["sources"]}["exporter:worker"]["ok"] is False


def test_no_boxes_configured_is_a_stated_absence_not_an_empty_list():
    doc = LM.build_live(now=10_000, fetch=_fetch(), sparks=[], box_specs=[])
    assert len(doc["boxes"]) == 1
    assert doc["boxes"][0]["detail"] == "no boxes configured"
    assert doc["boxes"][0]["ok"] is False
    assert all(i["value"] is None for i in doc["boxes"][0]["items"])
    assert doc["machine"] == doc["boxes"][0]["items"]


def test_exporter_url_overrides_the_head_box_url():
    seen = []

    def f(url, timeout=None):
        seen.append(url)
        return HEAD_TEXT

    doc = LM.build_live(now=10_000, fetch=f, sparks=[],
                        exporter_url="http://127.0.0.1:9999/metrics")
    assert doc["boxes"][0]["url"] == "http://127.0.0.1:9999/metrics"
    assert "http://127.0.0.1:9999/metrics" in seen
