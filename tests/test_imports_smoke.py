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

    assert C.BASELINE == "cruz"


def test_presets_are_the_four_the_console_offers():
    import console_core as C

    assert sorted(C.PRESETS) == ["kit-140", "kit-180", "quick-60", "smoke-20"]


def test_registry_catalogue_ids():
    import registry

    assert [e["id"] for e in registry.CATALOGUE] == ["cruz", "exl3-2.5bpw", "vllm-prod"]
    default = [e for e in registry.CATALOGUE if e.get("default")]
    assert len(default) == 1, "exactly one catalogue entry must be the default"
    assert default[0]["id"] == "cruz"


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
