"""Import and contract smoke tests.

These exist to prove the harness itself works -- modules import from the checkout (not from
/root/load), the sandbox is in effect, and the published constants the rest of the system
agrees on are still what the console and the dispatcher expect. Everything else is added by
the coverage work; this file is the floor, not the ceiling.
"""
from __future__ import annotations

import pathlib

import pytest


def test_modules_import_from_the_checkout():
    import console_core
    import corrected_metrics
    import dispatcher
    import history_collector
    import live_server
    import registry
    import zgx_exporter

    for mod in (console_core, corrected_metrics, dispatcher, history_collector,
                live_server, registry, zgx_exporter):
        origin = pathlib.Path(mod.__file__).resolve()
        assert origin.parent != pathlib.Path("/root/load"), f"{mod.__name__} came from prod"
        assert origin.name.endswith(".py")


def test_sandbox_is_in_effect(sandbox):
    import console_core as C

    assert str(C.LOAD).startswith(str(sandbox))
    assert str(C.RUNS_DIR).startswith(str(sandbox))
    assert "runboard-test-" in str(C.LOAD)


def test_baseline_is_the_contract_the_console_shows():
    import console_core as C

    assert C.BASELINE == "vllm-cruz"


def test_presets_are_the_four_the_console_offers():
    import console_core as C

    assert sorted(C.PRESETS) == ["kit-140", "kit-180", "quick-60", "smoke-20"]


def test_registry_catalogue_ids():
    import registry

    assert [e["id"] for e in registry.CATALOGUE] == [
        "cruz", "vllm-cruz", "exl3-2.5bpw", "vllm-prod"]
    default = [e for e in registry.CATALOGUE if e.get("default")]
    assert len(default) == 1, "exactly one catalogue entry must be the default"
    assert default[0]["id"] == "vllm-cruz"


def test_no_engine_list_has_drifted_from_the_catalogue():
    """The engine lists must agree. Drift is exactly what hid the serving engine.

    registry.CATALOGUE and console_core.ENGINES each name the engines, and serve.sh
    keeps a third list. On 2026-09-26 the vLLM engine actually serving :18300 was in
    none of them: the console read "serving: none", reported a false "OFF BASELINE",
    and a switch that failed had no way to restore the engine it had displaced. This
    asserts the two python lists cannot diverge again, and that the baseline is a real
    engine rather than a name nothing can start.
    """
    import console_core as C
    import registry

    ids = {e["id"] for e in registry.CATALOGUE}
    catalogue_switches = {e["switch"] for e in registry.CATALOGUE if e.get("switch")}
    console_switches = {v["switch"] for v in C.ENGINES.values() if v.get("switch")}
    assert catalogue_switches == console_switches
    assert C.BASELINE in ids and C.BASELINE in catalogue_switches


def test_serve_sh_knows_every_engine_the_catalogue_can_switch_to():
    """The third engine list is a shell script, so pin that one too.

    serve.sh decides what is serving and what to restore when a switch fails. While it did
    not know the vLLM engine it reported "none" as well, which meant a failed switch had
    nothing to restore and would have left :18300 empty. Two python lists agreeing is not
    enough while a shell script holds the third vote. host/serve.sh is the canonical copy;
    it is installed to /root/serve.sh on the box.
    """
    import re

    import registry

    src = (pathlib.Path(__file__).resolve().parents[1] / "host" / "serve.sh").read_text()
    arms = set(re.findall(r"^  ([a-z0-9-]+)\)$", src, re.M))
    expected = {e["switch"] for e in registry.CATALOGUE if e.get("switch")}
    assert expected <= arms, f"serve.sh cannot switch to: {sorted(expected - arms)}"
    assert "usage: serve.sh" in src


def test_console_routes_are_served_read_only():
    import live_server

    src = pathlib.Path(live_server.__file__).read_text(encoding="utf-8")
    for path in ('"/"', '"/history"', '"/console"'):
        assert path in src, f"{path} is not routed by live_server"
    # the web process must not be able to execute a job: no command spawning, no executor
    assert "subprocess" not in src, "live_server must not spawn processes"
    assert "Popen" not in src, "live_server must not spawn processes"
    assert "systemctl" not in src, "live_server must not call systemctl"
    assert "dispatcher" not in src, "live_server must not import the executor"


@pytest.mark.parametrize("prefix", ["/root/load/", "/root/exl3-bench/"])
def test_prod_guard_detects_real_production_paths(prefix):
    from conftest import touches_prod

    assert touches_prod(prefix + "thing") is not None
    assert touches_prod("/tmp/runboard-test-xyz/load/thing") is None
