"""live_server: the /api/live route.

Same contract as every other read route -- open, read-only, and a failure is an HTTP
status rather than a stack trace. The route is also pinned as *not* needing the dispatch
token, because the whole point of the live card is that it opens on a phone.
"""
from __future__ import annotations

import json

import live_server
from httpkit import request

H = live_server.Handler


def test_live_route_serves_the_document_without_a_token(monkeypatch):
    doc = {"schema": "zgx.console.live.v1", "machine": [{"key": "gpu_temp", "value": 37}]}
    monkeypatch.setattr(live_server.LM, "live_doc", lambda **kw: doc)
    status, headers, body = request(H, "GET", "/api/live")
    assert status == 200
    assert headers["content-type"] == "application/json"
    assert json.loads(body) == doc


def test_live_route_reports_502_when_the_document_cannot_be_built(monkeypatch):
    def boom(**kw):
        raise RuntimeError("both sources exploded")

    monkeypatch.setattr(live_server.LM, "live_doc", boom)
    status, _, body = request(H, "GET", "/api/live")
    assert status == 502
    assert "RuntimeError" in json.loads(body)["error"]


def test_live_route_is_read_only(monkeypatch):
    """POSTing at it must not be a way in -- writes live on their own two routes."""
    monkeypatch.setattr(live_server.LM, "live_doc", lambda **kw: {"schema": "x"})
    status, _, body = request(H, "POST", "/api/live", body=b"{}",
                              headers={"Content-Type": "application/json"})
    assert status == 404 and b"no such route" in body


def test_live_route_passes_the_window_query_through(monkeypatch):
    """The sparkline window is a query parameter; the route must hand it over verbatim."""
    seen = {}

    def spy(**kw):
        seen.update(kw)
        return {"schema": "zgx.console.live.v1", "window": kw.get("window")}

    monkeypatch.setattr(live_server.LM, "live_doc", spy)
    status, _, body = request(H, "GET", "/api/live?window=6h")
    assert status == 200
    assert seen["window"] == "6h"
    assert json.loads(body)["window"] == "6h"


def test_live_route_sends_no_window_when_the_query_is_absent(monkeypatch):
    seen = {}
    monkeypatch.setattr(live_server.LM, "live_doc",
                        lambda **kw: seen.update(kw) or {"ok": True})
    status, _, _ = request(H, "GET", "/api/live")
    assert status == 200
    assert seen["window"] is None


def test_live_route_ignores_an_unknown_window(monkeypatch):
    """A junk window is the route's business, not a 502: live_metrics falls back."""
    seen = {}
    monkeypatch.setattr(live_server.LM, "live_doc",
                        lambda **kw: seen.update(kw) or {"ok": True})
    status, _, _ = request(H, "GET", "/api/live?window=all-of-it")
    assert status == 200
    assert seen["window"] == "all-of-it"      # passed through; the module decides the fallback


def test_live_route_passes_the_dispatcher_status_so_a_run_is_not_called_idle(monkeypatch):
    """The exporter cannot say "a preset run is in progress" -- the dispatcher can."""
    seen = {}
    monkeypatch.setattr(live_server.LM, "live_doc",
                        lambda **kw: seen.update(kw) or {"ok": True})
    monkeypatch.setattr(live_server.C, "read_json",
                        lambda p, d=None: {"state": "running", "job": {"job_id": "jx"}})
    status, _, _ = request(H, "GET", "/api/live")
    assert status == 200
    assert seen["job"]["state"] == "running"
    assert seen["job"]["job"]["job_id"] == "jx"
