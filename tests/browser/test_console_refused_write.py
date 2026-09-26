"""Browser-level check of the console's refused-write message.

Why this exists: the pytest suite is string assertions over console.html, and twice in one day
it was green while the page was broken at runtime -- first a TypeError inside the click handler
(the hint sat on "queueing…" forever), and before that a message written and immediately
overwritten in the same turn. Neither is visible to a test that reads the file as text.

This drives a real browser at a real origin. It SKIPS (never fails) when playwright or the
deployment is unavailable, so it cannot break a laptop run of the suite. It only exercises the
*refused* path, so it needs no token and queues nothing.

Run against a live console:   pytest tests/browser -q
"""
from __future__ import annotations

import os
import socket

import pytest

URL = os.environ.get("RUNBOARD_URL", "https://runboard.local.curci.cc/console")
HOST = URL.split("//", 1)[-1].split("/", 1)[0]
PORT = 443 if URL.startswith("https") else 80

pytestmark = pytest.mark.browser

CANDIDATE_ENGINES = [
    "/root/.cache/ms-playwright/chromium-1228/chrome-linux64/chrome",
]


def _reachable() -> bool:
    try:
        socket.create_connection((HOST.split(":")[0], PORT), timeout=3).close()
        return True
    except OSError:
        return False


def _launch(pw):
    """The playwright client here may want a browser revision that is not on disk."""
    for path in CANDIDATE_ENGINES:
        if os.path.exists(path):
            return pw.chromium.launch(headless=True, args=["--no-sandbox"],
                                      executable_path=path)
    return pw.chromium.launch(headless=True, args=["--no-sandbox"])


@pytest.mark.skipif(not _reachable(), reason=f"{HOST} not reachable from here")
def test_a_refused_queue_explains_itself_without_dying():
    pw = pytest.importorskip("playwright.sync_api", reason="playwright not installed")
    with pw.sync_playwright() as p:
        browser = _launch(p)
        ctx = browser.new_context(ignore_https_errors=True)     # a profile that never
        page = ctx.new_page()                                   # saw this origin
        page.goto(URL, wait_until="networkidle")

        rejected = []
        page.on("pageerror", lambda e: rejected.append(str(e)))

        # no token in this fresh profile: the page must say so, on its own, before any click
        row = page.evaluate("""() => {
          for (const dt of document.querySelectorAll('dt')) {
            if (dt.textContent.trim() === 'token in this browser')
              return dt.nextElementSibling ? dt.nextElementSibling.textContent.trim() : '';
          }
          return null;
        }""")
        assert row and "not saved here" in row, row

        # click Queue for real. The server refuses (401); the page must end up saying WHY.
        with page.expect_response(lambda r: "/api/dispatch" in r.url) as ri:
            page.click("#dispatch-btn")
        assert ri.value.status == 401          # and nothing was queued

        page.wait_for_timeout(2500)
        hint = page.evaluate("() => document.getElementById('dispatch-hint').innerText")
        ctx.close()
        browser.close()

    assert not rejected, f"the click handler threw: {rejected}"
    assert "queueing" not in hint.lower(), f"the page never finished: {hint!r}"
    lines = [l for l in hint.split("\n") if l.strip()]
    assert len(lines) >= 2, f"expected the status line AND what to do about it: {lines}"
    assert "401" in lines[0], lines
    assert "token" in lines[1].lower() and "browser" in lines[1].lower(), lines
