"""Where a number comes from must be visible on the page that shows it.

Two failure modes this file pins shut. Both were measured on the deployed board, and the second
is the reason the first was fixed by REMOVAL rather than by relabelling.

1. The ``age`` badge counted seconds since the LOAD-SOAK harness last wrote its state file
   (``/root/load/run/state.json``). With no soak in flight that file is *supposed* to be old --
   measured 2026-09-28 it read ``347325 s`` (4.0 d) in yellow while the engine card beside it was
   serving live traffic. A permanently yellow badge on a monitoring board cannot report anything:
   the one signal an operator is trained to read as "something is wrong" was lit permanently.

2. Worse: ``/`` was fetching ``/api/state`` -- the soak archive, 67 KB -- every 3 s. It rendered a
   Throughput card reading ``0.00 tok/s``, a Machine card holding that run's samples, and a Phases
   table, all beside engine readings four seconds old. The soak finished **2026-09-24** against
   **EXL3 2.50bpw**; ``:18300`` serves **GLM-5.3-Flash-EXL3**. A ``0.00`` on a live page does not
   read as "an archived run that finished" -- it reads as *the box is idle*, while the box is in
   fact serving. That is the same class of lie the Boxes card exists to prevent (0 °C on an
   unreachable Spark), and it cannot be fixed by relabelling a number whose neighbours are live.

So the fix is structural: ``/`` fetches ``/api/live`` and NOTHING else, and every archive number
lives on ``/history`` with its date and the model that produced it attached at the point of
display. The badge is gone because the thing it counted is no longer on that page at all.
"""
from __future__ import annotations

import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parent.parent


def page(name="index.html") -> str:
    return (REPO / "static" / name).read_text(encoding="utf-8")


# Everything that belonged to the soak archive and must not appear on the live page. These are
# the element ids the old page used, so the guard fails on a page that has them back -- not on a
# page that merely mentions the word "soak" in a note explaining where the archive went.
ARCHIVE_IDS = ("k-agg", "k-stream", "k-conc", "k-req", "k-err", "c-tps", "c-pow",
               "k-pow", "k-temp", "k-util", "k-mem", "phases", "feed")


def test_the_live_page_fetches_the_live_document_and_nothing_else():
    """One document on the live page. The archive is a different page's problem.

    Pinned as the fetch call, not as the substring ``/api/state``: the page legitimately NAMES
    /api/state in a comment explaining where the archive went, and a guard that fails on an
    explanatory comment is a guard somebody deletes.
    """
    src = page()
    assert 'fetch("/api/state"' not in src, "the live page is fetching the soak archive again"
    assert 'fetch("/api/live?window=30m"' in src, "the live page stopped reading the live document"


def test_the_live_page_has_one_document_so_there_is_one_failure_path():
    body = " ".join(page().split("async function poll()")[1].split())
    assert body.count("fetch(") == 1, "the live page fetches more than the live document"


def test_no_archive_element_survives_on_the_live_page():
    """A leftover id is a leftover writer: something would still populate it, from stale data."""
    src = page()
    for ident in ARCHIVE_IDS:
        assert f'id="{ident}"' not in src, f"the archive element #{ident} is back on the live page"


def test_the_live_page_says_where_the_archive_went():
    """"Nothing was deleted; it moved" is only true if the page says so."""
    src = page("index.html")
    assert "/history" in src, "the live page does not link to the archive page"


def test_the_archive_page_is_where_the_archive_is_fetched():
    src = page("history.html")
    assert 'fetch("/api/state"' in src, "the archive page does not read the archive document"


def test_every_archive_number_carries_the_date_and_the_model_that_made_it():
    """The stamp is the whole point of the archive card.

    The date and the model are read from the history index (server-side, ``d.soak``) so the page
    cannot invent a model name, and the state file's own start time is the fallback for the window
    before the index has been rebuilt.
    """
    src = page("history.html")
    body = " ".join(src.split("function renderArchive()")[1].split())
    assert "sk.model" in body, "the archive card does not name the model it measured"
    assert "sk.started_utc" in body and "st.started_utc" in body, \
        "the archive card does not name the date it was measured (nor fall back to the state file's)"
    assert "ARCHIVE" in body and "no longer serves" in body, \
        "the archive card does not say that the model is retired"


def test_the_archive_heading_names_the_date_and_not_just_the_model():
    """A heading that names only the model invites comparison with today's live page."""
    src = page("history.html")
    assert '"Load soak — the " + started + " archive, "' in src, \
        "the archive heading dropped the date, so it reads as current"


def test_the_archive_labels_its_kpis_with_the_measurement_date():
    """The KPI row is the thing a reader scans; its labels must carry the date too."""
    src = page("history.html")
    body = " ".join(src.split("function renderArchive()")[1].split())
    assert body.count("60 s rolling, ") >= 2, \
        "the archive KPIs do not carry the measurement date in their labels"


def test_the_machine_envelope_is_a_summary_and_states_its_sample_span():
    """An archive wants min/mean/max, not a single sample mislabelled as 'power W'."""
    src = page("history.html")
    body = " ".join(src.split("function stats(")[1].split())
    assert "min:" in body and "max:" in body and "mean:" in body, \
        "the machine summary is not an envelope"
    assert "samples, " in src, "the machine summary does not state how many samples / what span"


# ---------------------------------------------------------------- the guards can fail
#
# A guard that cannot fail for the reason it exists is decoration. Each test below runs one of the
# guards above against the page as it actually was before 2026-10-06 and requires it to fire.

OLD_LIVE_PAGE_SNIPPET = """
  <div class="card">
    <h2>Machine — power and temperature</h2>
    <div class="kpi"><div class="v" id="k-temp">—</div></div>
    <canvas id="c-pow"></canvas>
  </div>
  <div class="card wide"><table id="phases"><tbody></tbody></table></div>
    const [rs, rl] = await Promise.allSettled([
      fetch("/api/state", { cache: "no-store" }),
      fetch("/api/live?window=30m", { cache: "no-store" }),
    ]);
"""


def test_the_archive_guard_fires_on_the_page_that_shipped_the_archive():
    old = OLD_LIVE_PAGE_SNIPPET
    assert 'fetch("/api/state"' in old, "the fixture no longer resembles the old page"
    body = " ".join(('async function poll()' + old).split("async function poll()")[1].split())
    assert body.count("fetch(") == 2, "the one-document guard would not have caught the second fetch"
    assert any(f'id="{i}"' in old for i in ARCHIVE_IDS), \
        "the leftover-id guard would not have caught the archive elements"


def test_the_retired_model_guard_fires_on_a_stamp_that_omits_the_model():
    old = 'stamp.innerHTML = "ARCHIVE";'
    assert "no longer serves" not in old, \
        "the stamp guard would not catch a stamp that does not say the model is retired"


def test_the_header_badge_states_what_serves_now_and_is_never_hardcoded():
    """The header once claimed ':18300 now serves the Cruz fork' in static HTML -- and the
    claim survived two engine swaps. What the port serves must be derived from the engine's
    own API on every poll, never written into the page.

    It is now the ONLY badge on the live page: it is the one claim the header makes, and the
    other three (phase / soak state age / concurrency) were soak-archive state on a live page.
    """
    src = page()
    assert "serves the Cruz fork" not in src, "a hardcoded deployment claim is back on the page"
    assert 'id="hdr-port"' in src, "the derived header badge is missing"
    body = " ".join(src.split("function renderEngine")[1].split())
    assert 'hdr.textContent = up ? (":18300 serves "' in body, \
        "the header badge is not populated from the live engine document"
    # Scoped to <header>: the two card-source badges (eng-src / boxes-src) legitimately carry the
    # same class down in the cards. It is the HEADER that must state exactly one thing.
    header = src.split("<header>")[1].split("</header>")[0]
    assert header.count('class="badge" id=') == 1, \
        "the live page's header grew a second badge; it states exactly one thing"
