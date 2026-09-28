"""The strip badges: what each one actually measures, and when it is allowed to say "stale".

The ``age`` badge counts seconds since the LOAD-SOAK harness last wrote its state file
(``/root/load/run/state.json``) -- not the engine's age, not the page's data age, not L1's. It is
not a healthy heartbeat, and with no soak in flight that file is *supposed* to be old.

Measured on the deployed board 2026-09-28: the badge read ``state age 347325 s`` in yellow
(4.0 d since the last soak finished on 2026-09-24) while the engine card beside it was serving
live traffic. A permanently yellow badge on a monitoring board is worse than no badge: the one
signal an operator is trained to read as "something is wrong" was lit permanently, so it could
not report anything. It now names its source and goes yellow only while a soak IS running and
its state has stopped advancing -- the single failure it can actually detect.
"""
from __future__ import annotations

import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parent.parent


def page(name="index.html") -> str:
    return (REPO / "static" / name).read_text(encoding="utf-8")


def _stale_gate(src: str) -> str:
    """The expression that decides the badge is stale."""
    # `let ... staleMark = false;` is the declaration, not the gate: match the assignment.
    m = re.search(r"(?m)^\s*staleMark\s*=\s*([^;]+);", src)
    assert m, "the badge's stale gate is missing entirely"
    return m.group(1)


def test_the_age_badge_names_the_soak_harness_and_not_just_an_age():
    src = page()
    assert "state age" not in src, "the badge still describes itself as a bare age"
    assert "soak state " in src, "the badge does not say whose state it is counting"
    assert "load-soak harness last wrote its state file" in src, \
        "the source of the number is not stated in the badge's title"


def test_the_badge_is_stale_only_while_a_soak_is_actually_running():
    gate = _stale_gate(page())
    assert "soakRunning" in gate, "the stale gate does not consult whether a soak is running"
    assert "age > 30" in gate, "the stale gate lost its threshold"


def test_the_soak_in_flight_signal_comes_from_the_soak_state_itself():
    src = page()
    assert 'st.phase !== "DONE"' in src, \
        "a soak in flight is decided somewhere other than the soak state's own phase"


def test_the_guard_fires_on_the_gate_it_replaced():
    """A guard that cannot fail is decoration: feed it the line it replaced."""
    old = page().replace(
        "staleMark = soakRunning && age !== null && age !== undefined && age > 30;",
        "staleMark = (age !== null && age !== undefined && age > 30);")
    assert "soakRunning" not in _stale_gate(old), \
        "the guard would not have caught the permanently-yellow badge"


def test_the_guard_fires_when_the_threshold_is_dropped():
    old = page().replace("soakRunning && age !== null && age !== undefined && age > 30",
                         "soakRunning && age !== null")
    assert "age > 30" not in _stale_gate(old), "the guard would not catch a lost threshold"


def test_every_age_on_the_board_is_rendered_in_units_a_human_reads():
    """347325 s is not an answer; the badge must go through the shared short-age helper."""
    src = page()
    assert "function agoShort(" in src, "the short-age helper is missing"
    assert "agoShort(age)" in src, "the badge does not use the short-age helper"
