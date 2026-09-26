"""ui_chrome: one nav, one strip, one look -- and the drift that motivated it.

Measured before this module existed, the three pages disagreed about all of it: three
titles for one product ("ZGX :18300 — load telemetry", "ZGX console", "Quant results —
history"), three different header link sets in three different orders, no page marking
where you were, and the design tokens re-declared in each file with a value already
drifted (``header { gap:12px }`` on two pages, ``gap:10px`` on the third).

The tests below pin the property that matters -- every page renders the SAME nav from the
SAME definition -- rather than pinning the markup, so a fourth page has to do the right
thing or the suite goes red.
"""
from __future__ import annotations

import pathlib
import re

import pytest

import ui_chrome as UC

REPO = pathlib.Path(__file__).resolve().parent.parent
PAGES = sorted((REPO / "static").glob("*.html"))
# The active path each page is served under, by filename.
ACTIVE = {"index.html": "/", "console.html": "/console", "history.html": "/history"}


def page(name: str) -> str:
    return (REPO / "static" / name).read_text(encoding="utf-8")


def nav_of(html: str) -> str:
    m = re.search(r'<nav class="views".*?</nav>', html, re.S)
    assert m, "injected page has no view nav"
    return m.group(0)


def hrefs(nav: str) -> list[str]:
    return re.findall(r'href="([^"]+)"', nav)


# ---------------------------------------------------------------- the view set

def test_the_view_set_is_one_ordered_definition():
    assert [h for h, _ in UC.VIEWS] == ["/", "/console", "/history"]
    assert len({label for _, label in UC.VIEWS}) == len(UC.VIEWS)


def test_a_page_under_an_alias_is_marked_as_the_same_view():
    for alias, canonical in (("/index.html", "/"), ("/console.html", "/console"),
                             ("/history.html", "/history"), ("/", "/")):
        assert UC.canonical(alias) == canonical
        assert UC.nav_html(alias) == UC.nav_html(canonical)


def test_an_unknown_path_passes_through_rather_than_being_guessed():
    assert UC.canonical("/nope") == "/nope"
    assert UC.view_label("/nope") == "/nope"


def test_every_known_view_has_a_label():
    for href, label in UC.VIEWS:
        assert UC.view_label(href) == label


# ---------------------------------------------------------------- all pages agree

def test_every_page_in_the_checkout_carries_all_three_markers():
    assert PAGES, "no pages found -- the glob is looking in the wrong place"
    for p in PAGES:
        text = p.read_text(encoding="utf-8")
        for marker in (UC.MARKER_HEAD, UC.MARKER, UC.MARKER_JS):
            assert marker in text, f"{p.name} is missing {marker}"


def test_every_page_renders_the_same_nav_in_the_same_order():
    """The whole point: one nav definition, identical wherever you are."""
    rendered = {p.name: nav_of(UC.inject(p.read_text(encoding="utf-8"),
                                        ACTIVE[p.name])) for p in PAGES}
    reference = hrefs(rendered["index.html"])
    assert reference == ["/", "/console", "/history"]
    for name, nav in rendered.items():
        assert hrefs(nav) == reference, f"{name} renders a different nav"


def test_each_page_marks_exactly_its_own_view():
    for p in PAGES:
        active = ACTIVE[p.name]
        html = UC.inject(p.read_text(encoding="utf-8"), active)
        nav = nav_of(html)
        marked = re.findall(r'<a class="badge on" href="([^"]+)" aria-current="page">',
                            nav)
        assert marked == [active], f"{p.name} marks {marked}, expected [{active}]"
        # and no other link claims to be current
        assert nav.count('aria-current="page"') == 1


def test_the_product_name_is_stated_once_for_every_page():
    for p in PAGES:
        html = UC.inject(p.read_text(encoding="utf-8"), ACTIVE[p.name])
        title = re.search(r"<title>(.*?)</title>", html, re.S).group(1)
        assert title.startswith(UC.PRODUCT + " · "), title
        assert title == f"{UC.PRODUCT} · {UC.view_label(ACTIVE[p.name])}"


# ---------------------------------------------------------------- no re-declared frame

def test_a_page_does_not_re_declare_the_shared_frame():
    """The drift guard: the frame is stated once (in the injected chrome), not per page.

    This is the test that would have caught the original divergence -- three copies of the
    token block, one of which had already drifted to a different header gap. Selectors are
    matched at the start of a line, so a page may still write a *qualified* rule of its own
    (the console's `body.kiosk header { display:none }`).
    """
    at_line_start = lambda sel: re.compile(r"(?m)^\s*" + re.escape(sel) + r"\s*\{")
    for p in PAGES:
        html = UC.inject(p.read_text(encoding="utf-8"), ACTIVE[p.name])
        assert len(at_line_start(":root").findall(html)) == 1, \
            f"{p.name} re-declares the design tokens"
        assert len(at_line_start("header").findall(html)) == 1, \
            f"{p.name} re-declares the header rule"
        assert html.count("--bg:#0d1117") == 1, f"{p.name} re-declares the palette"


def test_the_injected_frame_is_present_exactly_once_per_page():
    for p in PAGES:
        html = UC.inject(p.read_text(encoding="utf-8"), ACTIVE[p.name])
        assert html.count('id="zgx-strip"') == 1
        assert html.count('<nav class="views"') == 1
        for marker in (UC.MARKER_HEAD, UC.MARKER, UC.MARKER_JS):
            assert marker not in html, f"{p.name} still carries {marker}"


def test_the_shared_stylesheet_is_injected_before_the_page_own_style():
    """Order matters for the cascade: page rules must be able to override the frame."""
    html = UC.inject(page("index.html"), "/")
    first = html.index("<style>")
    second = html.index("<style>", first + 1)
    assert first < second, "the page's own <style> must come after the shared one"
    assert ":root {" in html[first:second]     # the shared sheet is the first block
    assert ".card" in html[second:]            # page-specific rules come after it


def test_the_palette_comes_from_the_shared_stylesheet():
    assert "--bg:#0d1117" in UC.CHROME_CSS
    for token in ("--a:#58a6ff", "--b:#3fb950", "--c:#d29922", "--d:#f85149"):
        assert token in UC.CHROME_CSS
    # the aliases the console used to declare locally now live in the shared sheet
    for alias in ("--accent:var(--a)", "--ok:var(--b)", "--warn:var(--c)", "--bad:var(--d)"):
        assert alias in UC.CHROME_CSS


def test_page_specific_rules_stay_in_the_page():
    """The frame is shared; content styling is not. `.card` belongs to the pages."""
    assert ".card" not in UC.CHROME_CSS
    assert ".card" in page("index.html")


# ---------------------------------------------------------------- the status strip

def test_the_strip_container_is_rendered_on_every_page():
    for p in PAGES:
        html = UC.inject(p.read_text(encoding="utf-8"), ACTIVE[p.name])
        assert '<div class="strip" id="zgx-strip"' in html


def test_the_strip_reads_the_one_authoritative_job_source():
    """All three pages must take run state from /api/live, and decide nothing themselves.

    /api/live's job block is the dispatcher's own status document -- the only component that
    knows about every kind of job. The exporter's soak gauges do not (live_metrics._job_block),
    which is the bug that made a busy box read as idle.
    """
    assert "/api/live" in UC.CHROME_JS
    assert "j.active" in UC.CHROME_JS           # renders the dispatcher's verdict
    assert "run active: " in UC.CHROME_JS
    assert "no run active" in UC.CHROME_JS
    # it must not consult the soak gauges to decide anything
    assert "zgx_load_running" not in UC.CHROME_JS
    assert "zgx_load_" not in UC.CHROME_JS


def test_the_strip_reports_a_dead_source_instead_of_filling_the_gap():
    assert "exporter unreachable" in UC.CHROME_JS
    assert "no history source" in UC.CHROME_JS
    assert "status unavailable" in UC.CHROME_JS


def test_the_strip_escapes_what_it_renders():
    assert "&amp;" in UC.CHROME_JS and "&lt;" in UC.CHROME_JS


def test_the_strip_is_labelled_with_its_as_of_stamp():
    assert "as of " in UC.CHROME_JS
    assert "refresh 10s" in UC.CHROME_JS


def test_the_strip_does_not_run_without_its_container():
    """A page that lacks the container must not throw on every poll."""
    assert 'if (!el) return;' in UC.CHROME_JS


def test_the_script_is_injected_at_the_end_of_the_body():
    html = UC.inject(page("index.html"), "/")
    assert html.index("<script>\n/* Shared status strip.") < html.rindex("</body>")


# ---------------------------------------------------------------- fail loud

@pytest.mark.parametrize("marker", [UC.MARKER_HEAD, UC.MARKER, UC.MARKER_JS])
def test_a_page_missing_a_marker_is_an_error_not_a_silent_degradation(marker):
    """A page served without the frame would have no navigation and no run-state strip."""
    text = page("index.html").replace(marker, "")
    with pytest.raises(ValueError) as err:
        UC.inject(text, "/")
    assert marker in str(err.value)


def test_head_and_script_helpers_are_the_injected_ones():
    assert UC.head_html().startswith("<style>")
    assert UC.CHROME_CSS in UC.head_html()
    assert UC.script_html() == UC.CHROME_JS
    assert "<script>" in UC.script_html()
