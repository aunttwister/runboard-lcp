"""serve.sh: only one engine can hold :18300, so every arm must clear the port first.

conftest.py's hermetic guard deliberately forbids a test from spawning a process -- running
serve.sh for real would switch the engine on this box -- so this reads the script rather
than executing it, and encodes the physical invariant instead of the current text: every
``case`` arm must stop each engine it is not starting.

This is the check that was missing on 2026-09-30. ``stop_vllm`` only stops the PROD
*container* and ``stop_exl3`` only stops the fork/stock exllamav3 units, so ``systemctl stop
vllm-exl3-cruz.service`` -- the default occupant of the port -- had no caller at all. Every
switch away from the baseline therefore left the incumbent listening, TensorFold aborted
with "port 18300 is already in use", and serve.sh still exited 0.
"""
from __future__ import annotations

import pathlib
import re

SERVE_SH = pathlib.Path(__file__).resolve().parent.parent / "host" / "serve.sh"

# Every engine that can hold :18300, by the variable serve.sh names it with.
ENGINES = {"PROD", "CRUZ", "STOCK", "TF", "VLLM_EXL3", "GLM"}

# stop helper -> the engines it stops
STOPPERS = {
    "stop_vllm": {"PROD"},
    "stop_exl3": {"CRUZ", "STOCK"},
    "stop_tensorfold": {"TF"},
    "stop_vllm_cruz": {"VLLM_EXL3"},
    "stop_glm": {"GLM"},
}

# systemctl start "$CRUZ" / docker start "$PROD" / systemctl start "$TF"
# The handle may carry a digit (VLLM_EXL3), so the class is not [A-Za-z_].
STARTERS = re.compile(r'(?:systemctl start|docker start)\s+"?\$\{?([A-Za-z_][A-Za-z0-9_]*)')
ARM = re.compile(r'^  ([a-z0-9|*"-]+)\)\s*$')


def _arms() -> dict[str, str]:
    """{arm label: body} for the dispatch case at the bottom of serve.sh."""
    body = SERVE_SH.read_text().split('case "${1:-status}" in', 1)[1]
    arms: dict[str, str] = {}
    name: str | None = None
    lines: list[str] = []
    for line in body.splitlines():
        m = ARM.match(line)
        if m:
            if name is not None:
                arms[name] = "\n".join(lines)
            name, lines = m.group(1), []
        elif line.startswith("esac"):
            break
        elif name is not None:
            lines.append(line)
    if name is not None:
        arms[name] = "\n".join(lines)
    return arms


def test_the_dispatch_case_is_parsed_at_all():
    """Guard the guard: a regex that stops matching must not turn the next test vacuous."""
    arms = _arms()
    assert {"vllm-cruz", "cruz", "exl3", "vllm", "tensorfold", "glm53"} <= set(arms)
    assert STARTERS.findall(arms["tensorfold"]) == ["TF"]


def test_every_arm_stops_each_engine_it_is_not_starting():
    for name, body in _arms().items():
        started = set(STARTERS.findall(body))
        if not started:            # status / usage: switches nothing
            continue
        stopped = set()
        for helper, handles in STOPPERS.items():
            if re.search(rf"^\s*.*\b{helper}\b", body, re.M):
                stopped |= handles
        still_holding = ENGINES - stopped - started
        assert not still_holding, (
            f"arm {name!r} starts {sorted(started)} but never stops {sorted(still_holding)}, "
            f"so the outgoing engine is still listening on :18300 -- the incoming one gets "
            f"'port already in use' or an empty health gate while serve.sh exits 0"
        )


def test_the_baseline_stopper_waits_for_the_port_rather_than_sleeping():
    """Releasing ~85 GB of weights takes as long as it takes; a fixed sleep is a guess.

    The port being free is the thing the next engine actually needs, and it is observable,
    so the helper must poll for it instead of trusting a sleep to have been enough.
    """
    text = SERVE_SH.read_text()
    helper = text.split("stop_vllm_cruz(){", 1)
    assert len(helper) == 2, "stop_vllm_cruz is not defined"
    body = helper[1].split("\n}", 1)[0]
    assert 'systemctl stop "$VLLM_EXL3"' in body
    assert "ss -ltn" in body and ":$PORT " in body, (
        "stop_vllm_cruz must wait for the port to be released, not just sleep"
    )
