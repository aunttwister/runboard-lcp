"""live_server: every read route, and the property that this process only ever reads.

The web process must never execute a job. What it may do is serve pages and JSON out of
files that something else wrote -- so each route is pinned here, including what it does
when the file is missing (503, not a stack trace) and when a static page is absent (500).
"""
from __future__ import annotations

import json
import pathlib
import re

import pytest

import console_core as C
import live_server
import ui_chrome as UC
from httpkit import request

H = live_server.Handler

# The real page on disk -- for assertions about what the CONSOLE claims, as opposed to what the
# route serves. Drift between the two is exactly what this file is here to catch.
CONSOLE_PAGE = pathlib.Path(__file__).resolve().parent.parent / "static" / "console.html"

# A minimal stand-in for a real page: its own content, plus the three markers the server
# expands into the shared frame (stylesheet, nav + strip, strip script).
PAGE_WITH_CHROME = ("<!DOCTYPE html><html><head>" + UC.MARKER_HEAD + "</head><body>"
                    + UC.MARKER + "<h1>page body</h1>" + UC.MARKER_JS
                    + "</body></html>")


# ---------------------------------------------------------------- health / pages

def test_health_is_open_and_tiny():
    status, headers, body = request(H, "GET", "/health")
    assert status == 200 and json.loads(body) == {"ok": True}
    assert headers["content-type"] == "application/json"


def test_the_three_pages_are_served_from_the_static_directory(sandbox):
    for path, name in (("/", "index.html"), ("/index.html", "index.html"),
                       ("/history", "history.html"), ("/console", "console.html"),
                       ("/console.html", "console.html")):
        (sandbox / "static" / name).write_text(PAGE_WITH_CHROME)
        status, headers, body = request(H, "GET", path)
        assert status == 200, path
        assert headers["content-type"].startswith("text/html")
        text = body.decode("utf-8")
        assert "<h1>page body</h1>" in text          # the page's own content survives
        assert 'id="zgx-strip"' in text              # the shared status strip
        assert UC.MARKER not in text                 # markers were expanded, not leaked


def test_each_page_is_served_with_its_own_view_marked(sandbox):
    """The nav is one definition; which tab is lit depends on the URL asked for."""
    for path, name, active in (("/", "index.html", "/"),
                               ("/console", "console.html", "/console"),
                               ("/history", "history.html", "/history")):
        (sandbox / "static" / name).write_text(PAGE_WITH_CHROME)
        status, _, body = request(H, "GET", path)
        assert status == 200
        nav = re.search(r'<nav class="views".*?</nav>', body.decode("utf-8"), re.S).group(0)
        assert f'<a class="badge on" href="{active}" aria-current="page">' in nav
        assert nav.count('aria-current="page"') == 1


def test_a_page_without_the_chrome_markers_fails_loudly(sandbox):
    """No silent degradation: a page that cannot get the frame is a 500, named."""
    (sandbox / "static" / "index.html").write_text("<h1>no chrome here</h1>")
    status, headers, body = request(H, "GET", "/")
    assert status == 500
    assert headers["content-type"].startswith("text/plain")
    assert b"index.html" in body and b"chrome" in body


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


# ------------------------------------------------- "am I set up to write?" (/api/token/check)
#
# The operator met a 401 AFTER filling in the dispatch form, because the only token row on the
# page came from the SERVER's own file and said yes while his browser held nothing. This route
# is what makes the browser's own token a checkable fact. It is read-only and always 200.

def test_token_check_on_a_server_with_no_token_says_so():
    status, _, body = request(H, "GET", "/api/token/check")
    assert status == 200
    assert json.loads(body) == {"configured": False, "presented": False, "ok": False}


def test_token_check_without_a_header_never_claims_ok():
    (C.LOAD / "dispatch.token").write_text("s3cret")
    status, _, body = request(H, "GET", "/api/token/check")
    assert status == 200
    assert json.loads(body) == {"configured": True, "presented": False, "ok": False}


def test_token_check_rejects_a_wrong_token():
    (C.LOAD / "dispatch.token").write_text("s3cret")
    _, _, body = request(H, "GET", "/api/token/check",
                         headers={"Authorization": "Bearer wrong"})
    assert json.loads(body) == {"configured": True, "presented": True, "ok": False}


@pytest.mark.parametrize("header", ["Bearer s3cret", "s3cret"])
def test_token_check_accepts_the_configured_token(header):
    """Both the prefixed and the bare form work, exactly as the write path already allowed."""
    (C.LOAD / "dispatch.token").write_text("s3cret")
    _, _, body = request(H, "GET", "/api/token/check", headers={"Authorization": header})
    assert json.loads(body) == {"configured": True, "presented": True, "ok": True}


def test_token_check_is_read_only():
    """A check must never queue anything -- that is the write path's job, behind the token."""
    (C.LOAD / "dispatch.token").write_text("s3cret")
    before = sorted(p.name for p in C.QUEUE.glob("*.json"))
    request(H, "GET", "/api/token/check", headers={"Authorization": "Bearer s3cret"})
    assert sorted(p.name for p in C.QUEUE.glob("*.json")) == before


def test_the_console_states_the_server_token_and_the_browser_token_separately():
    """The regression, in the page itself: one row answering two questions.

    `token configured` came from the server's file and read as "you are set up" while the
    browser held no token. The page must now carry BOTH facts, and ask the server about the
    one it cannot know on its own.
    """
    html = (CONSOLE_PAGE).read_text(encoding="utf-8")
    assert '["token in this browser"' in html
    assert '["token on the server"' in html
    assert '["token configured"' not in html          # the misleading single row is gone
    assert "/api/token/check" in html                  # ...and the browser's token is checked
    # ...at LOAD, not only from inside the save/clear handlers. Pinned by indentation, because
    # a bare "checkToken()" substring is also satisfied by the indented calls in those handlers
    # -- which is exactly how this assertion was useless the first time it was written.
    assert any(line == "checkToken();" for line in html.splitlines())
