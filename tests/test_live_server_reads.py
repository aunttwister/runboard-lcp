"""live_server: every read route, and the property that this process only ever reads.

The web process must never execute a job. What it may do is serve pages and JSON out of
files that something else wrote -- so each route is pinned here, including what it does
when the file is missing (503, not a stack trace) and when a static page is absent (500).
"""
from __future__ import annotations

import json

import pytest

import console_core as C
import live_server
from httpkit import request

H = live_server.Handler


# ---------------------------------------------------------------- health / pages

def test_health_is_open_and_tiny():
    status, headers, body = request(H, "GET", "/health")
    assert status == 200 and json.loads(body) == {"ok": True}
    assert headers["content-type"] == "application/json"


def test_the_three_pages_are_served_from_the_static_directory(sandbox):
    for path, name in (("/", "index.html"), ("/index.html", "index.html"),
                       ("/history", "history.html"), ("/console", "console.html"),
                       ("/console.html", "console.html")):
        (sandbox / "static" / name).write_text(f"<h1>{name}</h1>")
        status, headers, body = request(H, "GET", path)
        assert status == 200, path
        assert body == f"<h1>{name}</h1>".encode()
        assert headers["content-type"].startswith("text/html")


def test_a_missing_page_reports_500_with_the_filename():
    status, _, body = request(H, "GET", "/")
    assert status == 500 and body == b"index.html missing"


def test_an_unknown_route_is_a_plain_404():
    status, headers, body = request(H, "GET", "/nope")
    assert status == 404 and body == b"not found"
    assert headers["content-type"].startswith("text/plain")


# ---------------------------------------------------------------- /api/state

def test_state_is_served_corrected_with_an_age(run_dir):
    (run_dir / "state.json").write_text(json.dumps({"phase": "LOAD", "totals": {}}))
    status, _, body = request(H, "GET", "/api/state")
    doc = json.loads(body)
    assert status == 200
    assert doc["correction"]["applied"] is False      # no raw rows yet, said plainly
    assert doc["correction"]["reason"] == "no raw rows yet"
    assert isinstance(doc["now_age_s"], float)


def test_state_is_overlaid_from_the_raw_request_log(run_dir):
    (run_dir / "state.json").write_text(json.dumps(
        {"phases": [{"name": "c1", "result": {"n": 1}}]}))
    (run_dir / "requests.jsonl").write_text(json.dumps({
        "phase": "c1", "ok": True, "completion_tokens": 600, "elapsed": 10.0,
        "concurrency": 1, "ts": 1000.0}) + "\n")
    doc = json.loads(request(H, "GET", "/api/state")[2])
    assert doc["correction"]["applied"] is True
    # the phase row is corrected in place: 600 tokens over 10 s of wall clock
    assert doc["phases"][0]["result"]["aggregate_tok_s"] == 60.0
    assert doc["live_agg_source"].startswith("recomputed:")


def test_a_broken_correction_step_is_reported_not_raised(run_dir, monkeypatch):
    (run_dir / "state.json").write_text("{}")

    def boom(state):
        raise ValueError("bad rows")

    monkeypatch.setattr(live_server, "_correct", boom)
    doc = json.loads(request(H, "GET", "/api/state")[2])
    assert doc["correction"] == {"applied": False, "error": "ValueError"}


def test_state_without_a_file_is_503_not_a_traceback(run_dir):
    status, _, body = request(H, "GET", "/api/state")
    assert status == 503
    assert json.loads(body) == {"error": "FileNotFoundError", "detail": "no state yet"}


# ---------------------------------------------------------------- history / models

def test_history_is_served_straight_from_the_collectors_file(run_dir):
    (run_dir / "history.json").write_text('{"runs": [1]}')
    status, _, body = request(H, "GET", "/api/history")
    assert status == 200 and json.loads(body) == {"runs": [1]}


def test_history_not_built_yet_is_503(run_dir):
    status, _, body = request(H, "GET", "/api/history")
    assert status == 503 and json.loads(body)["detail"] == "history not built yet"


def test_models_is_served_from_the_registry_document(sandbox):
    C.MODELS.write_text('{"schema": "zgx.console.models.v1"}')
    status, _, body = request(H, "GET", "/api/models")
    assert status == 200 and json.loads(body)["schema"] == "zgx.console.models.v1"


def test_models_not_built_yet_says_how_to_build_it(sandbox):
    status, _, body = request(H, "GET", "/api/models")
    assert status == 503 and b"run registry.py" in body


# ---------------------------------------------------------------- /api/dispatch

def test_dispatch_view_reports_queue_status_and_history(run_dir):
    C.QUEUE.mkdir(parents=True, exist_ok=True)
    (C.QUEUE / "b.json").write_text("{}")
    (C.QUEUE / "a.json").write_text("{}")
    C.STATUS.write_text(json.dumps({"state": "running"}))
    C.HISTORY.write_text(json.dumps({"jobs": [{"job_id": "x"}]}))
    doc = json.loads(request(H, "GET", "/api/dispatch")[2])
    assert doc["queue"] == ["a.json", "b.json"]        # sorted, so the page is stable
    assert doc["status"]["state"] == "running"
    assert doc["history"]["jobs"] == [{"job_id": "x"}]


def test_dispatch_view_defaults_when_nothing_has_run_yet(run_dir):
    doc = json.loads(request(H, "GET", "/api/dispatch")[2])
    assert doc["status"] == {"state": "unknown"}
    assert doc["history"] == {"jobs": []} and doc["queue"] == []


# ---------------------------------------------------------------- HF passthrough

def test_hf_search_needs_a_query():
    status, _, body = request(H, "GET", "/api/hf/search")
    assert status == 400 and json.loads(body) == {"error": "q is required"}
    assert request(H, "GET", "/api/hf/search?q=%20")[0] == 400


def test_hf_search_passes_the_query_and_its_limit(monkeypatch):
    seen = {}

    def fake(query, limit=12, sort="downloads"):
        seen.update(query=query, limit=limit)
        return [{"repo": "owner/name"}]

    monkeypatch.setattr(C, "hf_search", fake)
    doc = json.loads(request(H, "GET", "/api/hf/search?q=qwen3.8&limit=5")[2])
    assert seen == {"query": "qwen3.8", "limit": 5}
    assert doc["query"] == "qwen3.8" and doc["results"] == [{"repo": "owner/name"}]


@pytest.mark.parametrize("given,expected", [("not-a-number", 12), ("0", 1), ("99", 40)])
def test_hf_search_clamps_the_limit(monkeypatch, given, expected):
    seen = {}
    monkeypatch.setattr(C, "hf_search", lambda q, limit=12: seen.update(limit=limit) or [])
    request(H, "GET", f"/api/hf/search?q=x&limit={given}")
    assert seen["limit"] == expected


def test_hf_search_reports_an_upstream_failure_as_502(monkeypatch):
    def boom(*a, **k):
        raise TimeoutError("huggingface did not answer")

    monkeypatch.setattr(C, "hf_search", boom)
    status, _, body = request(H, "GET", "/api/hf/search?q=x")
    assert status == 502 and json.loads(body)["error"].startswith("TimeoutError")


def test_hf_plan_needs_an_owner_and_a_name():
    status, _, body = request(H, "GET", "/api/hf/plan?repo=justaname")
    assert status == 400 and "owner/name" in json.loads(body)["error"]


def test_hf_plan_returns_the_plan(monkeypatch):
    monkeypatch.setattr(C, "hf_plan", lambda repo: {"repo": repo, "sizes": []})
    doc = json.loads(request(H, "GET", "/api/hf/plan?repo=owner/name")[2])
    assert doc == {"repo": "owner/name", "sizes": []}


def test_hf_plan_reports_an_upstream_failure_as_502(monkeypatch):
    def boom(repo):
        raise ValueError("no such repo")

    monkeypatch.setattr(C, "hf_plan", boom)
    status, _, body = request(H, "GET", "/api/hf/plan?repo=owner/name")
    assert status == 502 and "ValueError: no such repo" in json.loads(body)["error"]


def test_a_write_that_fails_mid_response_is_swallowed():
    # the client going away must not take the server process with it
    status, _, body = request(H, "GET", "/health", fail_write_at=2)
    assert status == 200 and len(body) == 0
