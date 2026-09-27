"""The card layout: ONE grid definition, and it must put a pair of cards in two columns.

Measured on the deployed board 2026-09-27 with an iframe probe (real geometry, not a
screenshot):

  * ``/`` at 1440px: ``main`` resolved to three 458.7px tracks and the Throughput + Machine
    cards sat in columns 1 and 2 -- 2 of 3, third track empty. At 1920px it was 2 of 4
    (460.5px tracks), at 3440px 2 of 7. ``/history`` (453.7px x 3) and ``/console``
    (453.7px x 3) did the same with their own pairs.
  * ``/`` at 390px: the document was 712px wide in a 390px viewport. Two causes, both pinned
    here: ``minmax(430px,1fr)`` cannot produce a track narrower than 430px, and the card
    holding the 9-column nowrap ``Phases`` table did not clip it (scrollWidth 693 in a 428px
    card, ``overflow-x: visible``), so the table widened the page instead of scrolling
    inside its own card.

``repeat(auto-fit,minmax(430px,1fr))`` is the whole defect: it derives the column COUNT from
the viewport, while the number of cards in a pair is fixed at two. The three pages carried
three copies of the rule and had already drifted apart -- ``console.html`` collapsed to one
column below 900px and the other two never did -- so the definition is pinned as one shared
value, not just corrected in place.
"""
from __future__ import annotations

import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parent.parent
PAGES = sorted((REPO / "static").glob("*.html"))

# `main` at the START of a line, so the console's `body.kiosk main { display:block }` override
# and the media query's inner `main {` are both left alone.
MAIN_RE = re.compile(r"(?m)^[ \t]*main\s*\{([^}]*)\}")
CARD_RE = re.compile(r"(?m)^[ \t]*\.card\s*\{([^}]*)\}")
COLLAPSE_RE = re.compile(r"@media\s*\(max-width:\s*900px\)\s*\{\s*main\s*\{([^}]*)\}")

TWO_TRACKS = "repeat(2,minmax(0,1fr))"
VIEWPORT_TRACKS = "repeat(auto-fit,minmax(430px,1fr))"
ONE_TRACK = "grid-template-columns:minmax(0,1fr);"


def page(name: str) -> str:
    return (REPO / "static" / name).read_text(encoding="utf-8")


def bodies(regex: re.Pattern[str], html: str) -> list[str]:
    return [" ".join(b.split()) for b in regex.findall(html)]


def one(regex: re.Pattern[str], html: str, what: str) -> str:
    """The single whitespace-normalised body of ``what``, or a loud failure.

    A second rule with the same selector would silently win the cascade, so "exactly one" is
    part of the property, not a convenience.
    """
    found = bodies(regex, html)
    assert len(found) == 1, f"expected exactly one {what}, found {len(found)}"
    return found[0]


def grid_of(html: str) -> str:
    return one(MAIN_RE, html, "`main` rule")


# ---------------------------------------------------------------- the guard predicates
# Kept as small pure functions so the mutation tests below can run the SAME check against the
# rule that was actually deployed when the defect was reported.

def has_viewport_sized_grid(html: str) -> bool:
    """True when the grid's column count is derived from the viewport -- the defect."""
    rule = " ".join(bodies(MAIN_RE, html))
    return "auto-fit" in rule or "auto-fill" in rule


def has_two_column_grid(html: str) -> bool:
    return TWO_TRACKS in " ".join(bodies(MAIN_RE, html))


def has_shrinkable_tracks(html: str) -> bool:
    return "minmax(0,1fr)" in " ".join(bodies(MAIN_RE, html))


def has_narrow_collapse(html: str) -> bool:
    return bodies(COLLAPSE_RE, html) == [ONE_TRACK]


def clips_wide_tables(html: str) -> bool:
    return "overflow-x:auto" in " ".join(bodies(CARD_RE, html))


# ---------------------------------------------------------------- the defect, pinned shut

def test_no_page_lets_the_viewport_decide_how_many_columns_the_grid_has():
    for p in PAGES:
        assert not has_viewport_sized_grid(p.read_text(encoding="utf-8")), \
            f"{p.name} sizes its tracks from the viewport, so a pair of cards cannot fill a row"


def test_the_half_width_pair_is_pinned_to_two_columns():
    """The row with the two half-width cards must be 2/2, at every viewport width."""
    for p in PAGES:
        assert has_two_column_grid(p.read_text(encoding="utf-8")), \
            f"{p.name} does not declare two grid columns"


def test_a_track_may_shrink_below_its_content_on_a_phone():
    """The 430px floor is what put 322px of overflow on a 390px viewport."""
    for p in PAGES:
        assert has_shrinkable_tracks(p.read_text(encoding="utf-8")), \
            f"{p.name} keeps a hard minimum track width: {grid_of(p.read_text(encoding='utf-8'))}"


def test_every_page_clips_the_wide_tables_inside_their_card():
    """A 9-column nowrap table must scroll in its card, not widen the document."""
    for p in PAGES:
        assert clips_wide_tables(p.read_text(encoding="utf-8")), \
            f"{p.name} does not clip overflow inside .card"


# ---------------------------------------------------------------- one definition, every page

def test_the_grid_is_one_definition_across_every_page():
    """The drift guard: console.html carried the narrow-screen collapse alone and nothing
    compared the three copies, so the other two kept the viewport-sized rule unnoticed."""
    assert len(PAGES) >= 3, "the page glob found too few pages to compare"
    rules = {p.name: grid_of(p.read_text(encoding="utf-8")) for p in PAGES}
    assert len(set(rules.values())) == 1, f"the pages disagree about the grid: {rules}"


def test_every_page_collapses_to_one_column_on_a_narrow_screen():
    for p in PAGES:
        assert has_narrow_collapse(p.read_text(encoding="utf-8")), \
            f"{p.name} never collapses to a single column"


# ---------------------------------------------------------------- the guards can fail
#
# A guard that cannot fail for the reason it exists is decoration. Each test below runs a
# guard above against the rule that was deployed when the defect was reported (or against a
# single page reverted to it) and requires the guard to fire.

def _reverted(name: str, *, drop_collapse: bool = False) -> str:
    html = page(name)
    assert TWO_TRACKS in html, "nothing to revert -- the fix is not in this page"
    html = html.replace(TWO_TRACKS, VIEWPORT_TRACKS)
    if drop_collapse:
        html = COLLAPSE_RE.sub("", html)
    return html


def test_the_viewport_guard_fires_on_the_deployed_rule():
    reverted = _reverted("index.html")
    assert has_viewport_sized_grid(reverted) is True
    assert has_two_column_grid(reverted) is False
    assert has_shrinkable_tracks(reverted) is False      # the 430px floor is back


def test_the_collapse_guard_fires_when_a_page_loses_the_media_query():
    assert has_narrow_collapse(_reverted("index.html", drop_collapse=True)) is False


def test_the_guard_on_the_clipped_card_fires_on_the_unclipped_copy():
    """The index page shipped without `overflow-x` on `.card`; stripping it must fail."""
    html = page("index.html").replace("min-width:0; overflow-x:auto; }", "min-width:0; }", 1)
    assert clips_wide_tables(html) is False


def test_the_drift_guard_fires_when_one_page_keeps_a_different_rule():
    rules = {p.name: grid_of(p.read_text(encoding="utf-8")) for p in PAGES}
    assert len(set(rules.values())) == 1
    rules["index.html"] = grid_of(_reverted("index.html"))     # this page alone kept the old rule
    assert len(set(rules.values())) > 1


def test_the_drift_guard_fires_when_one_page_keeps_a_different_collapse():
    collapses = {p.name: bodies(COLLAPSE_RE, p.read_text(encoding="utf-8")) for p in PAGES}
    assert set(map(tuple, collapses.values())) == {(ONE_TRACK,)}
    collapses["history.html"] = bodies(COLLAPSE_RE,
                                       _reverted("history.html", drop_collapse=True))
    assert set(map(tuple, collapses.values())) != {(ONE_TRACK,)}
